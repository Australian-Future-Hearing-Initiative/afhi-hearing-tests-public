from .retrieval import search_autoeq_files, fetch_frequency_response, is_bone_conduction_device
from .calibration_pipeline import vet_and_combine_responses, calculate_calibration_correction
from .config import AUDIOMETRY_FREQUENCIES

def get_calibration_factors(user_headphone: str, baseline_headphone: str) -> dict:
    """Retrieves frequency responses and calculates calibration correction factors.
    
    Args:
        user_headphone (str): The model name of the user's headphone (e.g., 'Sony WH-1000XM4') or a direct raw URL.
        baseline_headphone (str): The reference headphone used for calibration ('Apple AirPods Pro' or 'Google Pixel Buds Pro 2') or a direct raw URL.
        
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
    
    # 1. Bone conduction check (only if not a direct URL)
    if not (user_headphone.startswith("http://") or user_headphone.startswith("https://")):
        if is_bone_conduction_device(user_headphone):
            results["bone_conduction_warning"] = True
        
    try:
        # 2. Retrieve user headphone data
        if user_headphone.startswith("http://") or user_headphone.startswith("https://"):
            user_files = [{
                "name": user_headphone.split("/")[-1].replace(".csv", ""),
                "path": user_headphone,
                "database": "direct_url",
                "html_url": user_headphone,
                "raw_url": user_headphone
            }]
        else:
            user_files = search_autoeq_files(user_headphone)

        if not user_files:
            results["error_message"] = f"Could not find frequency response data for user headphone: '{user_headphone}'."
            return results
            
        user_responses = []
        for file_info in user_files:
            try:
                resp = fetch_frequency_response(file_info["raw_url"], headphone_name=user_headphone)
                # Attach database as attribute for vetting selection
                resp.database = file_info["database"]
                user_responses.append(resp)
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
        if baseline_headphone.startswith("http://") or baseline_headphone.startswith("https://"):
            baseline_files = [{
                "name": baseline_headphone.split("/")[-1].replace(".csv", ""),
                "path": baseline_headphone,
                "database": "direct_url",
                "html_url": baseline_headphone,
                "raw_url": baseline_headphone
            }]
        else:
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
                    resp.database = file_info["database"]
                    baseline_responses.append(resp)
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
