""" Simulates pure tone audiometry to evaluate stimulus selection and
reconstruction algorithms.

Usage:
  python simulate_audiometry.py [options]

Examples:
  python simulate_audiometry.py
  python simulate_audiometry.py --test_set edge_cases --verbosity 2
  python simulate_audiometry.py --selector kernel --num_trials 50

Key Options:
  --test_set:       'realistic' (default), 'edge_cases', 'single_example'
  --selector:       'hybrid' (default) or 'kernel'
  --reconstructors: Comma-separated list (e.g. 'kernel,hybrid,global,local')
  --verbosity:      0=silent, 1=summary (default), 2=detailed plots, 3=debug
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

from typing import Tuple, Optional, List, Dict
from common import STANDARD_FREQS_HZ
from logistic_fit import GLOBAL_FIT_VARIANCE, fit_audiogram_hybrid
import audiogram_data
from pta_algorithms import (StimulusSelector, AudiogramReconstructor,
                            KernelSelector, KernelReconstructor,
                            HybridLogisticReconstructor, HybridSelector)
import adaptive_audiometry
from simulation_plotting import (plot_hearing_loss_basis_functions,
                               plot_probability_surface,
                               plot_hybrid_uncertainty,
                               plot_local_logistic_fits,
                               plot_results,
                               plot_weight_evolution,
                               display_summary_table)

NOISE_STD_DB = 4  # This needs to be better modeled/justified using real data.


def get_interpolated_threshold(freq_hz: float,
                               audiogram_data_local: dict) -> float:
  """Interpolates threshold from audiogram data using log-frequency scale.

  Args:
    freq_hz: The frequency (Hz) at which to find the threshold.
    audiogram_data_local: Dictionary {frequency: threshold}.

  Returns:
    The interpolated threshold (dB HL) at freq_hz.
  """
  # Extract frequencies and thresholds, sort by frequency.
  freqs = sorted(audiogram_data_local.keys())
  thresholds = [audiogram_data_local[f] for f in freqs]

  # Convert frequencies to log2 scale for interpolation.
  log2_freqs = np.log2(freqs)
  target_log2_freq = np.log2(freq_hz)

  # Use numpy interpolation.
  interpolated_threshold = np.interp(target_log2_freq, log2_freqs, thresholds)
  return interpolated_threshold


def _calculate_mae(
    estimated_audiogram: Dict[float, Optional[float]],
    true_audiogram: Dict[float, float]
) -> Optional[float]:
  """Calculates mean absolute error between an estimated and true audiogram."""
  errors = []
  for freq, estimated_thresh in estimated_audiogram.items():
    if estimated_thresh is not None:
      true_thresh = get_interpolated_threshold(freq, true_audiogram)
      errors.append(abs(estimated_thresh - true_thresh))

  if not errors:
    return None
  return np.mean(errors)



def determine_hybrid_phase(history: List[Tuple[float, float, bool]]) -> str:
  """
  Determines the current phase of the Hybrid test based on history.
  Duplicates logic from HybridSelector to avoid polluting the core API.
  """
  config = adaptive_audiometry.INITIAL_PHASE_CONFIG
  target_freq = config['target_freq_hz']
  min_level = config['min_level_dbhl']

  target_freq_results = [r for r in history if r[0] == target_freq]
  target_freq_not_heard = [r for r in target_freq_results if not r[2]]

  # Phase 1 check
  if not target_freq_not_heard:
    target_freq_heard = [r for r in target_freq_results if r[2]]
    # Check the edge case where we are at min_level and heard it.
    already_heard_min_level = any(
      r[1] == min_level for r in target_freq_heard)

    # If we haven't found a 'not heard' yet, we are in Descent,
    # UNLESS we hit the floor and heard it (in which case we move on).
    if not already_heard_min_level:
      return 'Descent'

  # Phase 2 check
  tested_freqs = {res[0] for res in history if res[0] != target_freq}
  remaining_sweep_freqs = [
    f for f in STANDARD_FREQS_HZ
    if f != target_freq and f not in tested_freqs
  ]
  if remaining_sweep_freqs:
    return 'Sweep'

  # Phase 3
  return 'Adaptive'


def run_simulation(selector_obj: StimulusSelector,
                   reconstructors: List[AudiogramReconstructor],
                   final_output_names_set: set[str],
                   num_trials: int,
                   test_audiogram: dict,
                   verbosity: int) -> Dict[str, Optional[float]]:
  """
  Runs a generic PTA simulation.

  Args:
      selector_obj: The stimulus selection algorithm to use.
      reconstructors: A list of audiogram reconstruction algorithms to run.
      final_output_names_set: A set of names of the final outputs to keep.
      num_trials: The total number of stimuli to present.
      test_audiogram: The ground truth audiogram for the simulation.
      verbosity: Controls the level of console and graphical output.
  """
  if verbosity >= 1:
    print(f'Running simulation for {num_trials} trials...')
  results_history: list[Tuple[float, float, bool]] = []
  noise_std_db = NOISE_STD_DB
  weight_history: List[Tuple[float, float]] = []
  phase_starts = {'Descent': 1}

  for trial in range(1, num_trials + 1):
    if verbosity == 1:
      # Single line progress update
      print(f'\r  Trial: {trial}/{num_trials}', end='')
    elif verbosity >= 2:
      print(f'\n--- Trial {trial}/{num_trials} ---')

    next_freq_hz, next_level_dbhl = selector_obj.next_stimulus(
      results_history, verbosity)

    # Check for phase transition if using HybridSelector
    if isinstance(selector_obj, HybridSelector):
      phase = determine_hybrid_phase(results_history)
      if phase not in phase_starts:
        phase_starts[phase] = trial
        if verbosity >= 2:
          print(f'>>> {phase} Phase Started')

    threshold = get_interpolated_threshold(next_freq_hz, test_audiogram)
    noisy_threshold = threshold + np.random.normal(0, noise_std_db)
    response = bool(next_level_dbhl > noisy_threshold)
    if verbosity >= 2:
      print(f"Simulated response: {'Heard' if response else 'Not Heard'}")

    results_history.append((next_freq_hz, next_level_dbhl, response))

    # Capture weight evolution if using Hybrid model
    if verbosity >= 2 and any(isinstance(r, HybridLogisticReconstructor)
                              for r in reconstructors):
      # We fit the hybrid model at each step to track variance.
      # This duplicates some work but ensures we have the data for plotting.
      _, local_data, _ = fit_audiogram_hybrid(results_history,
                                              STANDARD_FREQS_HZ)

      avg_local_weight = 0.0
      current_freq_weight = 0.0
      valid_freqs = 0
      for freq in STANDARD_FREQS_HZ:
        var_local = local_data['variances'].get(freq)
        if var_local is not None:
          w_local = 1.0 / var_local
          w_global = 1.0 / GLOBAL_FIT_VARIANCE
          norm_w_local = w_local / (w_local + w_global)
          avg_local_weight += norm_w_local
          valid_freqs += 1
          if abs(freq - next_freq_hz) < 1e-3:
            current_freq_weight = norm_w_local

      if valid_freqs > 0:
        weight_history.append((avg_local_weight / valid_freqs,
                               current_freq_weight))
      else:
        weight_history.append((0.0, 0.0))

  if verbosity == 1:
    print()  # Newline after the progress bar finishes

  all_reconstructed_audiograms = {}
  for reconstructor in reconstructors:
    reconstructed = reconstructor.reconstruct(results_history, verbosity)
    all_reconstructed_audiograms.update(reconstructed)

  # Filter the audiograms to keep only the ones requested on the command line.
  audiograms_to_plot = {
    name: audiogm for name, audiogm in all_reconstructed_audiograms.items()
    if name in final_output_names_set
  }

  # Calculate MAE for each audiogram that is being plotted.
  mae_results_dict = {}
  for recon_name, audiogram in audiograms_to_plot.items():
    audiogram_for_mae = audiogram
    # If the audiogram is a dictionary containing more than just thresholds
    # (e.g., Local or Hybrid fits), extract the 'thresholds' dict.
    if isinstance(audiogram, dict) and 'thresholds' in audiogram:
      audiogram_for_mae = audiogram['thresholds']
    mae_results_dict[recon_name] = _calculate_mae(
      audiogram_for_mae, test_audiogram)

  if verbosity >= 1:
    plot_results(results_history, test_audiogram,
                 audiograms_to_plot, mae_results_dict, num_trials)

  # Generate specialized, detailed plots only for the highest verbosity level.
  if verbosity >= 2:
    # Plot weight evolution if we have data
    if weight_history:
      plot_weight_evolution(weight_history, phase_starts)

    # Generate specialized plots for specific reconstructors.
    for r in reconstructors:
      if isinstance(r,
                    KernelReconstructor) and 'Kernel' in final_output_names_set:
        plot_probability_surface(
          r, results_history,
          lambda f: get_interpolated_threshold(f, test_audiogram),
          current_trial=num_trials,
          audiogram_estimates=list(audiograms_to_plot.items())
        )
      if isinstance(r, HybridLogisticReconstructor):
        # Only plot these if the corresponding outputs were requested
        if any(n in final_output_names_set for n in
               ['Hybrid', 'Global Parametric', 'Local']):
          # Plot the uncertainty landscape for the Hybrid model.
          if 'Hybrid' in final_output_names_set:
            hybrid_data = audiograms_to_plot.get('Hybrid', {})
            plot_hybrid_uncertainty(
              hybrid_data, results_history,
              lambda f: get_interpolated_threshold(f, test_audiogram),
              current_trial=num_trials
            )

        if 'Local' in all_reconstructed_audiograms:
          plot_local_logistic_fits(
            results_history, all_reconstructed_audiograms['Local'],
            STANDARD_FREQS_HZ
          )
  return mae_results_dict




if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Run pure tone audiometry simulations.')
  parser.add_argument('--test_set', type=str, default='realistic',
                      choices=['realistic', 'edge_cases', 'single_example'],
                      help='Select the set of audiograms to test.')
  parser.add_argument('--selector', type=str, default='hybrid',
                      choices=['kernel', 'hybrid'],
                      help='Select the stimulus selection algorithm.')
  parser.add_argument('--reconstructors', type=str, default='kernel,hybrid',
                      help='Comma-separated list of reconstruction algorithms '
                           '(e.g., kernel,hybrid,global,local).')
  parser.add_argument('--num_trials', type=int, default=40,
                      help='Number of trials to run in the simulation.')
  parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2, 3],
                      help='Set console and plot verbosity: 0=silent, '
                           '1=summary, 2=detailed, 3=debug.')

  args = parser.parse_args()

  # First, visualize the basis functions that the parameterized model uses.
  if args.verbosity >= 2:
    plot_hearing_loss_basis_functions()

  audiograms_to_test = {}
  if args.test_set == 'realistic':
    audiograms_to_test = audiogram_data.REALISTIC_AUDIOGRAMS
  elif args.test_set == 'edge_cases':
    audiograms_to_test = audiogram_data.EDGE_CASE_AUDIOGRAMS
  elif args.test_set == 'single_example':
    audiograms_to_test.update(audiogram_data.SINGLE_EXAMPLE)

  # --- Instantiate selected components based on requested outputs ---
  selector_map = {'kernel': KernelSelector, 'hybrid': HybridSelector}
  selector = selector_map[args.selector]()

  # Determine which reconstructors to run and which final outputs to keep.
  requested_outputs = set(r.strip() for r in args.reconstructors.split(','))
  reconstructors_to_run = []
  if 'kernel' in requested_outputs:
    reconstructors_to_run.append(KernelReconstructor())

  # If any of the hybrid-related outputs are requested, run the reconstructor.
  hybrid_outputs = {'hybrid', 'global', 'local'}
  if any(h in requested_outputs for h in hybrid_outputs):
    reconstructors_to_run.append(HybridLogisticReconstructor())

  # Map command-line names to the names returned by the reconstructors.
  output_name_map = {
    'kernel': 'Kernel',
    'hybrid': 'Hybrid',
    'global': 'Global Parametric',
    'local': 'Local'
  }
  final_output_names = {output_name_map[r] for r in requested_outputs
                        if r in output_name_map}

  all_results = []
  for name, current_ground_truth in audiograms_to_test.items():
    if args.verbosity >= 1:
      print(f'\nRunning simulation for "{name}" audiogram...')
    mae_results = run_simulation(selector_obj=selector,
                                 reconstructors=reconstructors_to_run,
                                 final_output_names_set=final_output_names,
                                 num_trials=args.num_trials,
                                 test_audiogram=current_ground_truth,
                                 verbosity=args.verbosity)

    # Store results for the final summary table.
    row = {'Audiogram': name}
    row.update(mae_results)
    all_results.append(row)

  # After all simulations are complete, display the summary table.
  display_summary_table(all_results, verbosity=args.verbosity)

  # Show all generated plots at the end of the script.
  if args.verbosity >= 1 and any(
      plt.get_fignums()):  # Check if any figures were created
    plt.show()
