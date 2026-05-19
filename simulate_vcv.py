"""
Simulates the Adaptive VCV test using the ZEST procedure and uncertainty-based
scheduler. This script is essential for validating the implementation and
tuning parameters (e.g., assumed slope, hot start logic).
"""

import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import bayesian_vcv_estimator

# --- Simulation Configuration ---
N_SIMULATIONS = 50       # Number of full adaptive tests to run.
N_TRIALS_PER_TEST = 150  # Total trials simulated for each adaptive test.
SIM_EARS = ['left', 'right']
# Use consonant list directly from the estimator module for consistency.
SIM_CONSONANTS = list(bayesian_vcv_estimator.CONSONANT_LABELS.keys())

# The values below represent the "true" underlying parameters of the
# simulated listener.
# The actual slope of the simulated listener's psychometric function.
TRUE_SLOPE = 3.5
# The actual lapse rate of the simulated listener.
TRUE_LAPSE_RATE = 0.02

def generate_true_srts() -> Dict[Tuple[str, str], float]:
  """
  Generates a realistic set of "true" Speech Reception Thresholds (SRTs)
  for a virtual listener across all test conditions.
  """
  true_srts = {}
  population_mean = -5.0
  inter_subject_sd = 4.0
  intra_subject_sd = 2.0

  listener_avg_srt = np.random.normal(population_mean, inter_subject_sd)

  for ear in SIM_EARS:
    for consonant in SIM_CONSONANTS:
      condition_srt = np.random.normal(listener_avg_srt, intra_subject_sd)
      condition_srt = np.clip(
          condition_srt,
          bayesian_vcv_estimator.MIN_SNR_DB,
          bayesian_vcv_estimator.MAX_SNR_DB
      )
      true_srts[(ear, consonant)] = condition_srt
  return true_srts

def simulate_response(snr: float, true_srt: float) -> bool:
  """
  Simulates a virtual listener's binary (correct/incorrect) response
  for a given stimulus SNR and true SRT.
  """
  p_correct = bayesian_vcv_estimator.ZestEstimator.psychometric_function(
      snr, true_srt, TRUE_SLOPE,
      bayesian_vcv_estimator.CHANCE_RATE, TRUE_LAPSE_RATE
  )
  return random.random() < p_correct

def schedule_next_trial_sim(
    estimators: Dict[Tuple[str, str], bayesian_vcv_estimator.ZestEstimator]
):
  """
  Replicates the scheduler from `demo_vcv.py` for simulation purposes.
  Uses Weighted Random Sampling based on uncertainty^2.
  """
  candidates = []
  weights = []

  for key, estimator in estimators.items():
    _, uncertainty = estimator.get_estimate()

    weight = uncertainty ** 2
    candidates.append(key)
    weights.append(weight)

  if candidates:
    selected_key = random.choices(candidates, weights=weights, k=1)[0]
  else:
    # Fallback if estimates dict is empty (shouldn't happen)
    return random.choice(list(estimators.keys())), 0.0

  selected_estimator = estimators[selected_key]
  next_snr = selected_estimator.get_next_snr()

  return selected_key, np.clip(
      next_snr,
      bayesian_vcv_estimator.MIN_SNR_DB,
      bayesian_vcv_estimator.MAX_SNR_DB
  )


def run_single_simulation() -> pd.DataFrame:
  """Runs one full adaptive VCV test simulation for a virtual listener."""
  true_srts = generate_true_srts()

  estimators = {}
  for ear in SIM_EARS:
    for consonant in SIM_CONSONANTS:
      prior_mean = (
          bayesian_vcv_estimator
          .CONSONANT_INITIAL_SNR_DB[consonant]
      )
      estimators[(ear, consonant)] = (
          bayesian_vcv_estimator.ZestEstimator(
              prior_mean=prior_mean,
              prior_sd=bayesian_vcv_estimator.PRIOR_SD,
          )
      )

  for _ in range(N_TRIALS_PER_TEST):
    condition_key, next_snr = schedule_next_trial_sim(estimators)
    true_srt = true_srts[condition_key]
    response = simulate_response(next_snr, true_srt)
    estimators[condition_key].update(next_snr, response)

  results = []
  for key, estimator in estimators.items():
    ear, consonant = key
    estimated_srt, uncertainty = estimator.get_estimate()
    true_srt = true_srts[key]
    error = estimated_srt - true_srt
    results.append({
        'Ear': ear, 'Consonant': consonant, 'True_SRT': true_srt,
        'Estimated_SRT': estimated_srt, 'Error': error,
        'Uncertainty': uncertainty, 'Trials': len(estimator.history)
    })
  return pd.DataFrame(results)

def analyze_simulations(all_results_df: pd.DataFrame):
  """
  Analyzes and plots the aggregated results from multiple simulations.
  """
  print('\n--- Simulation Analysis ---')
  mae = all_results_df['Error'].abs().mean()
  bias = all_results_df['Error'].mean()
  rmse = np.sqrt((all_results_df['Error']**2).mean())
  avg_uncertainty = all_results_df['Uncertainty'].mean()
  avg_trials_per_condition = all_results_df['Trials'].mean()

  print(f'Total Simulations: {N_SIMULATIONS} | '
        f'Trials per Test: {N_TRIALS_PER_TEST}')
  print(f'Avg Trials per Condition: {avg_trials_per_condition:.2f}')
  print(f'Mean Absolute Error (MAE): {mae:.3f} dB')
  print(f'Bias (Mean Error): {bias:.3f} dB')
  print(f'Root Mean Squared Error (RMSE): {rmse:.3f} dB')
  print(f'Average Final Uncertainty (SD): {avg_uncertainty:.3f} dB')

if __name__ == '__main__':
  print(f'Starting VCV ZEST simulations (N={N_SIMULATIONS})...')
  print('Configuration:')
  print(f'  Assumed Slope (Estimator): {bayesian_vcv_estimator.ASSUMED_SLOPE}')
  print(f'  True Slope (Listener): {TRUE_SLOPE}')

  all_sim_results = []
  for i in range(N_SIMULATIONS):
    if (i + 1) % 10 == 0 or i == 0:
      print(f'  Running simulation {i+1}/{N_SIMULATIONS}...')
    sim_results_df = run_single_simulation()
    sim_results_df['Simulation_ID'] = i
    all_sim_results.append(sim_results_df)

  final_results_df = pd.concat(all_sim_results, ignore_index=True)
  analyze_simulations(final_results_df)
