"""Generates deterministic "golden data" for integration testing.

This script runs the adaptive PTA algorithm against a simulated, deterministic
human subject (no noise) for several defined scenarios. It outputs JSON files
containing the exact sequence of stimuli presented and the final audiogram.

Usage:
  python3 generate_golden_data.py                 # 70 dB (default).
  python3 generate_golden_data.py --max-level 85  # 85 dB.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

from common import STANDARD_FREQS_HZ, PTA_MAX_TRIALS_PER_EAR, PTA_MAX_LEVEL_DB_HL
from pta_algorithms import HybridSelector, HybridLogisticReconstructor
from simulate_audiometry import get_interpolated_threshold

# Define the scenarios to generate.
SCENARIOS = {
    'perfect_hearing': {
        'description': 'Flat -5 dB HL audiogram (perfect result)',
        'audiogram': {f: -5.0 for f in STANDARD_FREQS_HZ}
    },
    'sloping_loss': {
        'description': 'Sloping loss from 10 dB at 250 Hz to 80 dB at 8000 Hz',
        'audiogram': {
            250.0: 10.0,
            500.0: 20.0,
            1000.0: 30.0,
            2000.0: 50.0,
            3000.0: 60.0,
            4000.0: 70.0,
            6000.0: 75.0,
            8000.0: 80.0
        }
    },
    'profound_loss': {
        'description': 'Flat 90 dB HL audiogram (Profound Hearing Loss)',
        'audiogram': {f: 90.0 for f in STANDARD_FREQS_HZ}
    },
}

DEFAULT_OUTPUT_DIR = 'tests/golden_data'


def run_deterministic_simulation(
    audiogram: Dict[float, float],
    num_trials: int,
    max_level_dbhl: int
) -> Dict[str, Any]:
  """Runs a single simulation and captures the interaction trace."""
  selector = HybridSelector(max_level_dbhl=max_level_dbhl)
  reconstructor = HybridLogisticReconstructor()

  history: List[Tuple[float, float, bool]] = []
  trace = []

  print(f'  Simulating {num_trials} trials...')

  for i in range(num_trials):
    # 1. Ask Algorithm for Next Stimulus
    next_freq, next_level = selector.next_stimulus(history, verbosity=0)

    # 2. Get Deterministic Response (No Noise)
    true_threshold = get_interpolated_threshold(next_freq, audiogram)
    # If level > threshold, they hear it. strictly greater.
    response = bool(next_level > true_threshold)

    # 3. Record Step
    step_record = {
        'step_index': i,
        'stimulus_freq_hz': float(next_freq),
        'stimulus_level_dbhl': float(next_level),
        'response': response,
        'true_threshold_at_freq': float(true_threshold)
    }
    trace.append(step_record)

    # 4. Update History
    history.append((next_freq, next_level, response))

  # 5. Final Reconstruction
  final_results = reconstructor.reconstruct(history, verbosity=0)

  # We focus on the 'Hybrid' result as the primary output to verify.
  hybrid_result = final_results.get('Hybrid', {})

  return {
      'trace': trace,
      'final_result': {
          'thresholds': hybrid_result.get('thresholds'),
          'variances': hybrid_result.get('variances')
      }
  }


def plot_summary(results: List[Dict[str, Any]]):
  """Plots a summary of the ground truth vs. estimated audiograms."""
  num_scenarios = len(results)
  _, axes = plt.subplots(1, num_scenarios, figsize=(5 * num_scenarios, 5),
                           sharey=True)

  # If only one scenario, axes is not a list.
  if num_scenarios == 1:
    axes = [axes]

  for ax, res in zip(axes, results):
    name = res['scenario_name']
    ground_truth = res['ground_truth_audiogram']
    estimated = res['final_estimation']['thresholds']

    # Sort for plotting
    gt_freqs = sorted(ground_truth.keys())
    gt_vals = [ground_truth[f] for f in gt_freqs]

    est_freqs = sorted([f for f in estimated.keys() if
                        estimated[f] is not None])
    est_vals = [estimated[f] for f in est_freqs]

    ax.semilogx(gt_freqs, gt_vals, 'k-', label='Ground Truth', linewidth=2)
    ax.semilogx(est_freqs, est_vals, 'r--o', label='Estimated', markersize=6)

    ax.set_title(name)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xticks(STANDARD_FREQS_HZ)
    ax.set_xticklabels([int(f) for f in STANDARD_FREQS_HZ], rotation=45)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(100, -10)

  axes[0].set_ylabel('Threshold (dB HL)')
  axes[0].legend()
  plt.tight_layout()

  print('\nDisplaying summary plot. Close window to continue.')
  plt.show()


def main():
  parser = argparse.ArgumentParser(
      description='Generate golden data for adaptive PTA integration tests.')
  parser.add_argument(
      '--max-level', type=int, default=PTA_MAX_LEVEL_DB_HL,
      help=f'Maximum stimulus level in dB HL '
           f'(default: {PTA_MAX_LEVEL_DB_HL})')
  args = parser.parse_args()

  max_level = args.max_level
  if max_level == PTA_MAX_LEVEL_DB_HL:
    output_dir = DEFAULT_OUTPUT_DIR
  else:
    output_dir = f'tests/golden_data_{max_level}dB'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  print(f'Generating Golden Data in {output_dir} '
        f'(max_level={max_level} dB HL)...')
  all_results_for_plot = []

  for name, scenario in SCENARIOS.items():
    print(f'\nProcessing scenario: {name}')
    print(f"  Description: {scenario['description']}")

    simulation_data = run_deterministic_simulation(
        scenario['audiogram'],
        PTA_MAX_TRIALS_PER_EAR,
        max_level_dbhl=max_level
    )

    output_data = {
        'scenario_name': name,
        'description': scenario['description'],
        'ground_truth_audiogram': scenario['audiogram'],
        'simulation_parameters': {
            'num_trials': PTA_MAX_TRIALS_PER_EAR,
            'max_level_dbhl': max_level,
            'frequencies_hz': STANDARD_FREQS_HZ
        },
        'steps': simulation_data['trace'],
        'final_estimation': simulation_data['final_result']
    }
    # Collect data for plotting
    all_results_for_plot.append(output_data)

    filepath = os.path.join(output_dir, f'{name}.json')
    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(output_data, f, indent=2)

    print(f'  Saved to {filepath}')

  # Generate the plot
  plot_summary(all_results_for_plot)

  print('\nDone. You can now run adaptive_pta_integration_test.py to verify.')


if __name__ == '__main__':
  main()
