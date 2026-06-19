import numpy as np
from .config import AUDIOMETRY_FREQUENCIES, DATABASE_PRIORITY, MAX_CORRECTION_DB, MIN_CORRECTION_DB, VETTING_DISCREPANCY_THRESHOLD_DB

def interpolate_response(freqs: list, responses: list, target_freqs: list = AUDIOMETRY_FREQUENCIES) -> np.ndarray:
    """Interpolates frequency response values at target frequencies in log space."""
    if not freqs or not responses:
        raise ValueError("Frequency response data cannot be empty.")
    
    # Audio frequencies are interpolated logarithmically
    log_freqs = np.log10(freqs)
    log_targets = np.log10(target_freqs)
    
    return np.interp(log_targets, log_freqs, responses)

def vet_and_combine_responses(retrieved_data: list[dict]) -> tuple[np.ndarray, dict]:
    """Vets response data from multiple sources.
    
    Checks for similarity:
    - If similar (std_dev <= threshold), returns the average consensus.
    - If dissimilar, selects the highest priority database and returns details + warning.
    
    Input format: list of dicts with keys: 'database', 'frequency', 'smoothed'
    Returns:
    - vetted_responses: np.ndarray of response values at target frequencies.
    - metadata: dict containing information about selection, sources, and warning flags.
    """
    if not retrieved_data:
        raise ValueError("No response data available for vetting.")
        
    # Interpolate all source responses at target frequencies
    interpolated_curves = {}
    for entry in retrieved_data:
        db_name = entry["database"]
        try:
            curve = interpolate_response(entry["frequency"], entry["smoothed"])
            interpolated_curves[db_name] = curve
        except Exception as e:
            print(f"Warning: Failed to interpolate response for database {db_name}: {e}")
            continue

    if not interpolated_curves:
        raise ValueError("Could not interpolate frequency response data for any source.")

    # Convert curves to array for statistical analysis
    source_names = list(interpolated_curves.keys())
    curves_matrix = np.array([interpolated_curves[name] for name in source_names])
    
    # Calculate standard deviation at each target frequency across sources
    if len(source_names) > 1:
        std_devs = np.std(curves_matrix, axis=0)
        max_std = float(np.max(std_devs))
    else:
        std_devs = np.zeros(len(AUDIOMETRY_FREQUENCIES))
        max_std = 0.0

    metadata = {
        "all_sources_found": source_names,
        "max_discrepancy_db": max_std,
        "discrepancy_flag": max_std > VETTING_DISCREPANCY_THRESHOLD_DB
    }

    # Decide on final response curve
    if len(source_names) == 1:
        # Only one source available
        selected_source = source_names[0]
        final_curve = interpolated_curves[selected_source]
        metadata["selection_method"] = "single_source"
        metadata["selected_source"] = selected_source
    elif max_std <= VETTING_DISCREPANCY_THRESHOLD_DB:
        # Consensus: average all sources
        final_curve = np.mean(curves_matrix, axis=0)
        metadata["selection_method"] = "consensus_average"
        metadata["selected_source"] = "average_of_" + "_".join(source_names)
    else:
        # Discrepancy: select highest priority source
        selected_source = None
        for db in DATABASE_PRIORITY:
            if db in interpolated_curves:
                selected_source = db
                break
        
        if not selected_source:
            # Fallback to the first available source
            selected_source = source_names[0]
            
        final_curve = interpolated_curves[selected_source]
        metadata["selection_method"] = f"priority_select_{selected_source}"
        metadata["selected_source"] = selected_source
        metadata["warning"] = (
            f"Significant discrepancy detected between measurement databases (max SD = {max_std:.2f} dB). "
            f"Selected the most authoritative database: '{selected_source}'."
        )

    return final_curve, metadata

def calculate_calibration_correction(
    user_response: np.ndarray, 
    baseline_response: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calculates correction factors needed for user headphone to match baseline response.
    
    Correction = Baseline - User
    
    Returns:
    - raw_correction: raw difference in dB.
    - clipped_correction: correction clamped within safe limits.
    - math_metadata: dict containing clipping status and stats.
    """
    raw_correction = baseline_response - user_response
    
    # Clamping correction within safe limits
    clipped_correction = np.clip(raw_correction, MIN_CORRECTION_DB, MAX_CORRECTION_DB)
    
    clipping_occurred = not np.array_equal(raw_correction, clipped_correction)
    max_clip_delta = float(np.max(np.abs(raw_correction - clipped_correction)))
    
    math_metadata = {
        "clipping_occurred": clipping_occurred,
        "max_clip_delta_db": max_clip_delta if clipping_occurred else 0.0,
        "mean_absolute_correction_db": float(np.mean(np.abs(clipped_correction)))
    }
    
    return raw_correction, clipped_correction, math_metadata
