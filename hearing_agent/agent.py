import json
import asyncio
import re
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from .retrieval import search_autoeq_files, fetch_frequency_response, is_bone_conduction_device
from .calibration import vet_and_combine_responses, calculate_calibration_correction
from .config import AUDIOMETRY_FREQUENCIES

def get_calibration_factors(user_headphone: str, baseline_headphone: str) -> dict:
    """Retrieves frequency responses and calculates calibration correction factors.
    
    Args:
        user_headphone (str): The model name of the user's headphone (e.g., 'Sony WH-1000XM4').
        baseline_headphone (str): The reference headphone used for calibration ('Apple AirPods Pro' or 'Google Pixel Buds Pro').
        
    Returns:
        dict: A dictionary containing correction factors, vetting details, and flags.
    """
    results = {
        "user_headphone": user_headphone,
        "baseline_headphone": baseline_headphone,
        "frequencies": AUDIOMETRY_FREQUENCIES,
        "correction_factors_db": [],
        "bone_conduction_warning": False,
        "vetting_warning": None,
        "clipping_warning": None,
        "status": "error",
        "error_message": None
    }
    
    # 1. Bone conduction check
    if is_bone_conduction_device(user_headphone):
        results["bone_conduction_warning"] = True
        
    try:
        # 2. Retrieve user headphone data
        user_files = search_autoeq_files(user_headphone)
        if not user_files:
            print(f"No local database or API match for '{user_headphone}'. Invoking AI Agent fallback...")
            try:
                agent_query = (
                    f"Search and retrieve the frequency response for '{user_headphone}' "
                    f"and calculate calibration offsets relative to '{baseline_headphone}'."
                )
                agent_res_text = asyncio.run(run_calibration_agent(agent_query))
                
                json_match = re.search(r'\{.*\}', agent_res_text, re.DOTALL)
                if json_match:
                    agent_data = json.loads(json_match.group(0))
                    if agent_data.get("gain_offsets_db"):
                        results["correction_factors_db"] = agent_data["gain_offsets_db"]
                        results["raw_correction_factors_db"] = agent_data.get("gain_offsets_db")
                        results["user_sources_used"] = ["ai_agent_research"]
                        results["user_selected_source"] = "ai_agent_research"
                        results["bone_conduction_warning"] = agent_data.get("bone_conduction", False)
                        results["status"] = "success"
                        return results
            except Exception as agent_err:
                print(f"AI Agent fallback failed: {agent_err}")
                
            results["error_message"] = f"Could not find frequency response data for user headphone: '{user_headphone}'."
            return results

            
        user_responses = []
        for file_info in user_files:
            try:
                resp = fetch_frequency_response(file_info["raw_url"], headphone_name=user_headphone)
                user_responses.append({
                    "database": file_info["database"],
                    "frequency": resp["frequency"],
                    "smoothed": resp["smoothed"]
                })
            except Exception as e:
                print(f"Warning: Failed to fetch CSV from {file_info['raw_url']}: {e}")
                continue
                
        if not user_responses:
            results["error_message"] = f"Failed to retrieve data from any database for: '{user_headphone}'."
            return results
            
        # 3. Vet and combine user responses
        user_vetted_curve, user_vetting_metadata = vet_and_combine_responses(user_responses)
        results["user_sources_used"] = user_vetting_metadata["all_sources_found"]
        results["user_selected_source"] = user_vetting_metadata["selected_source"]
        if user_vetting_metadata.get("warning"):
            results["vetting_warning"] = user_vetting_metadata["warning"]

        # 4. Retrieve baseline headphone data
        baseline_files = search_autoeq_files(baseline_headphone)
        if not baseline_files:
            # Fallback to default baseline curve (flat/0 dB) if baseline model not found
            print(f"Warning: Baseline headphone '{baseline_headphone}' not found in database. Using flat reference.")
            baseline_vetted_curve = [0.0] * len(AUDIOMETRY_FREQUENCIES)
            results["baseline_sources_used"] = ["default_flat"]
            results["baseline_selected_source"] = "default_flat"
        else:
            baseline_responses = []
            for file_info in baseline_files:
                try:
                    resp = fetch_frequency_response(file_info["raw_url"], headphone_name=baseline_headphone)
                    baseline_responses.append({
                        "database": file_info["database"],
                        "frequency": resp["frequency"],
                        "smoothed": resp["smoothed"]
                    })
                except Exception as e:
                    print(f"Warning: Failed to fetch baseline CSV: {e}")
                    continue
            
            if not baseline_responses:
                baseline_vetted_curve = [0.0] * len(AUDIOMETRY_FREQUENCIES)
                results["baseline_sources_used"] = ["default_flat_fallback"]
                results["baseline_selected_source"] = "default_flat_fallback"
            else:
                baseline_vetted_curve, baseline_vetting_metadata = vet_and_combine_responses(baseline_responses)
                results["baseline_sources_used"] = baseline_vetting_metadata["all_sources_found"]
                results["baseline_selected_source"] = baseline_vetting_metadata["selected_source"]

        # 5. Calculate correction factors
        raw_corr, clipped_corr, math_metadata = calculate_calibration_correction(
            user_vetted_curve, baseline_vetted_curve
        )
        
        results["correction_factors_db"] = [round(float(val), 2) for val in clipped_corr]
        results["raw_correction_factors_db"] = [round(float(val), 2) for val in raw_corr]
        
        if math_metadata["clipping_occurred"]:
            results["clipping_warning"] = (
                f"Correction factors exceeded safe calibration limits and were capped to prevent audio distortion. "
                f"Maximum clip delta was {math_metadata['max_clip_delta_db']:.2f} dB."
            )
            
        results["status"] = "success"
        
    except Exception as e:
        results["error_message"] = f"Calibration calculation pipeline failed: {str(e)}"
        
    return results


# Define ADK Agent
root_agent = Agent(
    name="HeadphoneCalibrationAgent",
    model="gemini-flash-latest",
    description="Calculates correction factors for audiometry headphones by querying AutoEq databases.",
    instruction=(
        "You are an expert Audio Data Scientist and Clinical Audiology Agent. "
        "Your task is to analyze user requests to calibrate headphones for a clinical hearing test. "
        "Use the 'get_calibration_factors' tool to fetch frequency responses and calculate calibration corrections. "
        "You must explain the results in a friendly, scientific manner, detailing: \n"
        "1. The databases searched and how you verified data consistency (Vetting).\n"
        "2. Any warnings, such as bone conduction devices (which require separate calibration) or output clipping.\n"
        "3. The calculated correction factors across frequencies (250Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz, 8000Hz).\n"
        "Always output the calibration results in a clear code block containing the raw JSON configuration."
    ),
    tools=[get_calibration_factors]
)

# Async runner helper function for integration
async def run_calibration_agent(user_query: str) -> str:
    """Helper to run the ADK Agent asynchronously and return its final text response."""
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="hearing_app", user_id="streamlit_user", session_id="session_001")
    
    runner = Runner(
        agent=root_agent,
        app_name="hearing_app",
        session_service=session_service
    )
    
    content = types.Content(role='user', parts=[types.Part(text=user_query)])
    final_response = "No response from agent."
    
    async for event in runner.run_async(user_id="streamlit_user", session_id="session_001", new_message=content):
        # In ADK python, we can check if it's the final output event
        # In some versions, events are dicts or have type attributes
        # Let's print event details or check for final response
        if hasattr(event, 'is_final_response') and event.is_final_response():
            final_response = event.message.content.parts[0].text
        elif isinstance(event, dict) and event.get("type") == "final_response":
            final_response = event.get("content", "")
        elif hasattr(event, 'message') and event.message and hasattr(event.message, 'content'):
            if event.message.content and event.message.content.parts:
                final_response = event.message.content.parts[0].text
                
    return final_response
