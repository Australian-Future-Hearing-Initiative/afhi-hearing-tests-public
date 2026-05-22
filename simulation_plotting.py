"""
Plotting functions for audiometry simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable

from common import STANDARD_FREQS_HZ
from logistic_fit import (PARAMETERIZED_FIT_BOUNDS,
                          PRIOR_LOW_DBHL, PRIOR_HIGH_DBHL,
                          LAPSE_RATE,
                          psychometric_logistic_curve)
from hearing_models import POLY_COEFFS
import audio_tools
from pta_algorithms import KernelReconstructor


def plot_hearing_loss_basis_functions():
  """
  Visualizes the basis functions of the parameterized hearing loss model.

  This function creates a plot with n-by-1 subplots, where n is the number
  of components in the hearing loss model. Each subplot shows how a single
  basis function contributes to the overall audiogram shape when multiplied
  by positive, negative, or zero coefficients. The y-axis is inverted to
  match audiogram conventions, where hearing loss is plotted downwards.
  """
  # The polynomial coefficients that define the basis functions are now
  # imported directly from the hearing_models module.
  poly_coeffs = POLY_COEFFS
  num_components = poly_coeffs.shape[1]
  poly_order = poly_coeffs.shape[0] - 1

  # Use the shared constant for plotting the bounds.
  bounds = PARAMETERIZED_FIT_BOUNDS

  # Generate a range of frequencies for a smooth plot.
  plot_freqs_hz = np.geomspace(250, 8000, 100)
  _, audf_powers = audio_tools.cf_to_audf(plot_freqs_hz, poly_order)

  # The basis shapes are the dot product of the audf powers and the coeffs.
  basis_shapes = audf_powers @ POLY_COEFFS

  fig, axes = plt.subplots(
    num_components, 1, figsize=(8, 2 * num_components), sharex=True)
  fig.suptitle('Global Model Basis Functions', fontsize=16)

  for i in range(num_components):
    ax = axes[i]
    shape = basis_shapes[:, i]
    min_bound, max_bound = bounds[f'c{i + 1}']

    # Plot the effect of the max and min allowed coefficients.
    ax.plot(plot_freqs_hz, shape * max_bound,
            label=f'Coefficient c{i + 1} = {max_bound} (Max)', color='blue')
    ax.plot(plot_freqs_hz, shape * min_bound,
            label=f'Coefficient c{i + 1} = {min_bound} (Min)', color='red')

    ax.set_ylabel('dBHL')
    ax.set_title(f'Basis Function {i + 1}')
    ax.set_xscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.invert_yaxis()  # Match audiogram convention (loss is downwards).
    ax.legend()

  # Common X-axis label.
  axes[-1].set_xlabel('Frequency (Hz)')
  # Common Y-axis label.
  fig.text(0, 0.5, 'Contribution to threshold',
           va='center', rotation='vertical', fontsize=12)
  # Use standard frequencies for x-ticks.
  plt.xticks(STANDARD_FREQS_HZ, [str(f) for f in STANDARD_FREQS_HZ])

  plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle


def setup_audiogram_plot(
    ax: plt.Axes,
    simulated_results: list[tuple[float, float, bool]],
    true_threshold_func: Callable[[float], float],
    log2_freqs: np.ndarray,
    levels: np.ndarray,
    audiogram_estimates: Optional[
      list[Tuple[str, dict[float, Optional[float]]]]
    ] = None
):
  """Helper function to draw the common elements of an audiogram plot."""
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Level (dBHL)')

  # Plot simulated data points
  sim_freqs = [r[0] for r in simulated_results]
  sim_levels = [r[1] for r in simulated_results]
  sim_responses = [r[2] for r in simulated_results]
  colors = ['lime' if r else 'red' for r in sim_responses]
  markers = ['o' if r else 'x' for r in sim_responses]
  for freq, level, color, marker in zip(sim_freqs, sim_levels, colors, markers):
    ax.scatter(np.log2(freq), level, color=color, marker=marker, s=100)

  # Plot true audiogram
  plot_freqs_hz = np.geomspace(
    np.power(2.0, log2_freqs[0]), np.power(2.0, log2_freqs[-1]), 100
  )
  true_thresholds = [true_threshold_func(f) for f in plot_freqs_hz]
  ax.plot(np.log2(plot_freqs_hz), true_thresholds, color='black',
          linestyle='--', linewidth=2, label='True Audiogram')

  # Plot audiogram estimates
  if audiogram_estimates:
    plot_colors = ['magenta', 'green', 'orange', 'purple', 'brown']
    plot_markers = ['*', 's', '^', 'D', 'v']
    for idx, (label, est_data) in enumerate(audiogram_estimates):
      thresholds_to_plot = est_data
      if isinstance(est_data, dict) and 'thresholds' in est_data:
        thresholds_to_plot = est_data['thresholds']
      if thresholds_to_plot:
        est_freqs = [
          f for f, t in thresholds_to_plot.items() if t is not None]
        est_thresholds = [
          t for t in thresholds_to_plot.values() if t is not None]
        if est_freqs:
          sort_indices = np.argsort(est_freqs)
          est_freqs_sorted = np.array(est_freqs)[sort_indices]
          est_thresholds_sorted = np.array(est_thresholds)[sort_indices]
          ax.plot(np.log2(est_freqs_sorted), est_thresholds_sorted,
                  marker=plot_markers[idx % len(plot_markers)],
                  color=plot_colors[idx % len(plot_colors)],
                  linestyle='-', markersize=8, label=label)

  # Formatting
  std_freqs_log2 = np.log2(STANDARD_FREQS_HZ)
  valid_ticks = [
    t for t in std_freqs_log2 if log2_freqs[0] <= t <= log2_freqs[-1]]
  ax.set_xticks(valid_ticks)
  ax.set_xticklabels(
    [str(int(np.round(np.power(2.0, t)))) for t in valid_ticks])
  ax.minorticks_off()
  ax.grid(True, which='major', axis='both', linestyle=':', linewidth=0.5)
  ax.set_xlim(log2_freqs[0] - 0.1, log2_freqs[-1] + 0.1)
  ax.set_ylim(levels[-1] + 10, levels[0] - 10)
  ax.legend(loc='lower left')


def plot_probability_surface(
    reconstructor: KernelReconstructor,
    simulated_results: list[tuple[float, float, bool]],
    true_threshold_func: Callable[[float], float],
    current_trial: Optional[int] = None,
    audiogram_estimates: Optional[
      list[Tuple[str, dict[float, Optional[float]]]]
    ] = None,
):
  """Plots the predicted probability surface and audiogram."""
  # This function does not need verbosity checks as it is only called
  # when verbosity >= 2

  predicted_probs = reconstructor.final_kernel_probs
  if predicted_probs is None:
    print('No kernel probabilities to plot.')
    return

  grid_points_x = reconstructor.grid_points_x
  prob_heard = predicted_probs.flatten()

  # DEBUG: Print the range of predicted probabilities.
  print(
    f"DEBUG: Min/Max predicted probability: "
    f"{np.nanmin(prob_heard):.4f} / {np.nanmax(prob_heard):.4f}"
  )

  # Reshape for plotting.
  log2_freqs = np.unique(grid_points_x[:, 0])
  levels = np.unique(grid_points_x[:, 1])

  # Check if reshaping is possible.
  expected_len = len(log2_freqs) * len(levels)
  if len(prob_heard) != expected_len:
    raise ValueError(
      f"Cannot reshape predicted_probs (len {len(prob_heard)}) into grid "
      f"({len(log2_freqs)} x {len(levels)} = {expected_len}). "
      f"Mismatch between create_test_grid and smoother output?"
    )
  # imshow expects shape (n_levels, n_freqs) for the chosen extent/origin.
  prob_grid = prob_heard.reshape(len(levels), len(log2_freqs))
  fig, ax = plt.subplots(1, 1, figsize=(12, 7))
  # Extent for imshow [left, right, bottom, top].
  extent = [
    log2_freqs[0], log2_freqs[-1],  # Use log2 freqs for extent.
    levels[-1], levels[0]  # Inverted y-axis for level.
  ]
  # Plot Probability.
  cmap_prob = 'viridis'
  im_prob = ax.imshow(
    prob_grid,
    aspect='auto',
    origin='upper',
    extent=extent,
    cmap=cmap_prob,
    interpolation='nearest',
    vmin=0,
    vmax=1
  )
  fig.colorbar(im_prob, ax=ax, label='Predicted Probability of Hearing')
  title = 'Kernel Smoother Fit (Probability)'
  if current_trial is not None:
    title += f' - Trial {current_trial}'
  ax.set_title(title)

  # Add all the common plot elements
  setup_audiogram_plot(ax, simulated_results, true_threshold_func,
                        log2_freqs, levels, audiogram_estimates)

  plt.tight_layout()


def plot_hybrid_uncertainty(
    hybrid_audiogram_data: dict[str, dict[float, Optional[float]]],
    simulated_results: list[tuple[float, float, bool]],
    true_threshold_func: Callable[[float], float],
    current_trial: Optional[int] = None,
):
  """Plots the hybrid audiogram with uncertainty bands and next stimulus."""
  print('Plotting Hybrid Uncertainty Landscape...')

  thresholds = hybrid_audiogram_data.get('thresholds', {})
  variances = hybrid_audiogram_data.get('variances', {})

  if not thresholds or not variances:
    print('No hybrid thresholds or variances to plot.')
    return

  # Extract frequencies, thresholds, and variances.
  freqs = sorted([f for f in thresholds.keys() if thresholds[f] is not None])
  if not freqs:
    return

  t_values = np.array([thresholds[f] for f in freqs])
  v_values = np.array([variances[f] for f in freqs])

  # Calculate standard deviation for plotting.
  std_devs = np.sqrt(v_values)

  # Identify the next stimulus (max variance).
  # Note: The actual next stimulus logic is in the Selector, but we replicate
  # the selection logic here for visualization:
  # Pick freq with max variance.
  valid_variances = {f: v for f, v in variances.items() if v is not None}
  if valid_variances:
    next_freq = max(valid_variances, key=valid_variances.get)
    next_level = thresholds.get(next_freq)
  else:
    next_freq = None
    next_level = None

  plt.figure(figsize=(12, 8))
  ax = plt.gca()
  title = 'Hybrid Model Uncertainty Landscape'
  if current_trial is not None:
    title += f' - Trial {current_trial}'
  ax.set_title(title)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Level (dBHL)')

  # 1. Plot the true audiogram (reference)
  log2_freqs_range = np.log2([min(STANDARD_FREQS_HZ), max(STANDARD_FREQS_HZ)])
  plot_freqs_hz = np.geomspace(
    np.power(2.0, log2_freqs_range[0]), np.power(2.0, log2_freqs_range[1]), 100
  )
  true_thresholds = [true_threshold_func(f) for f in plot_freqs_hz]
  ax.plot(np.log2(plot_freqs_hz), true_thresholds, color='black',
          linestyle='--', linewidth=2, label='True Audiogram', zorder=1)

  # 2. Plot the Hybrid estimate
  log2_est_freqs = np.log2(freqs)
  ax.plot(log2_est_freqs, t_values, color='blue', marker='o', linewidth=2,
          label='Hybrid Estimate', zorder=2)

  # 3. Plot the Uncertainty Band (Threshold +/- 2 SD)
  # Use fill_between.
  # We use 2 standard deviations to show a ~95% confidence interval, which is
  # more visually illustrative of the uncertainty range than 1 SD.
  ax.fill_between(log2_est_freqs, t_values - 2 * std_devs,
                  t_values + 2 * std_devs,
                  color='blue', alpha=0.2, label='Uncertainty (±2 SD)')

  # 4. Plot the Next Stimulus Indicator
  if next_freq is not None and next_level is not None:
    ax.scatter(np.log2(next_freq), next_level, color='red', s=200, marker='*',
               zorder=5, label='Next Stimulus Target')

  # 5. Plot past results for context
  sim_freqs = [r[0] for r in simulated_results]
  sim_levels = [r[1] for r in simulated_results]
  sim_responses = [r[2] for r in simulated_results]
  colors = ['lime' if r else 'red' for r in sim_responses]
  markers = ['o' if r else 'x' for r in sim_responses]
  for freq, level, color, marker in zip(sim_freqs, sim_levels, colors, markers):
    ax.scatter(np.log2(freq), level, color=color, marker=marker,
               s=50, alpha=0.6)

  std_freqs_log2 = np.log2(STANDARD_FREQS_HZ)
  ax.set_xticks(std_freqs_log2)
  ax.set_xticklabels([str(f) for f in STANDARD_FREQS_HZ])
  ax.minorticks_off()
  ax.grid(True, which='both', linestyle=':', linewidth=0.5)
  ax.invert_yaxis()
  ax.legend(loc='lower left')
  plt.tight_layout()


def plot_local_logistic_fits(
    results_history: list[tuple[float, float, bool]],
    local_audiogram: dict[str, dict[float, Optional[float]]],
    standard_freqs_hz: list[float]
):
  """
  Plots the per-frequency logistic curve fits on a grid of subplots.
  """
  num_freqs = len(standard_freqs_hz)
  # Create a 2x8 grid of subplots (2 rows, num_freqs columns).
  # Share x-axis across columns to align the two plots for each frequency.
  # Share y-axis across rows so distributions are comparable.
  fig, axes = plt.subplots(2, num_freqs, figsize=(24, 10),
                           sharex=True, sharey='row')
  fig.suptitle('Per-frequency logistic fit & Uncertainty', fontsize=16)

  # Sort frequencies for a consistent plotting order.
  sorted_freqs = sorted(standard_freqs_hz)

  for i, freq in enumerate(sorted_freqs):
    # Top row: Logistic Fit
    ax_top = axes[0, i]
    # Bottom row: Uncertainty Distribution
    ax_bot = axes[1, i]

    # Filter results for the current frequency.
    freq_results = [res for res in results_history if abs(res[0] - freq) < 1e-3]

    # --- TOP PLOT: Logistic Curve ---
    if not freq_results:
      ax_top.set_title(f'{freq} Hz - No Data')
      ax_top.grid(True, linestyle='--')
      ax_bot.grid(True, linestyle='--')
      continue

    levels = np.array([res[1] for res in freq_results])
    # Apply lapse rate to plotting positions, matching the fit logic.
    responses = np.array([1.0 - LAPSE_RATE if res[2] else
                          LAPSE_RATE for res in freq_results])

    # Plot the raw data points (heard vs. not heard).
    heard = responses >= 0.5
    not_heard = responses < 0.5
    ax_top.plot(levels[heard], responses[heard], 'o', color='lime',
                markersize=11, label='Heard')
    ax_top.plot(levels[not_heard], responses[not_heard], 'x', color='red',
            markersize=11, label='Not Heard')

    # Plot the anchor points (priors) used for the fit.
    ax_top.plot(PRIOR_LOW_DBHL, LAPSE_RATE, 'x', color='black', markersize=8,
                label='Anchor (NH)')
    ax_top.plot(PRIOR_HIGH_DBHL, 1.0 - LAPSE_RATE, 'o', color='black',
                markersize=8, label='Anchor (H)')

    # Get the fitted threshold and spread for this frequency.
    thresholds = local_audiogram.get('thresholds', {})
    spreads = local_audiogram.get('spreads', {})
    variances = local_audiogram.get('variances', {})

    fitted_threshold = thresholds.get(freq)
    fitted_spread = spreads.get(freq)
    fitted_variance = variances.get(freq)

    if fitted_threshold is not None and fitted_spread is not None:
      # Generate a smooth curve for plotting across the full range.
      plot_levels = np.linspace(PRIOR_LOW_DBHL, PRIOR_HIGH_DBHL, 200)

      # Calculate the fitted psychometric curve.
      prob_heard = psychometric_logistic_curve(
        plot_levels, fitted_threshold, fitted_spread
      )
      ax_top.plot(plot_levels, prob_heard, color='magenta', linewidth=2,
              label='Fitted Curve')

      # Plot a vertical line at the threshold (50% probability).
      ax_top.axvline(fitted_threshold, color='magenta', linestyle='--',
                 label='Threshold')

      ax_top.set_title(f'{freq} Hz\nSpread = {fitted_spread:.2f}')
      ax_top.grid(True, linestyle='--')
      # Only add legend to the first subplot to avoid clutter
      if i == 0:
        ax_top.legend(loc='upper left', fontsize='small')
    else:
      ax_top.set_title(f'{freq} Hz - Fit Failed')
      ax_top.grid(True, linestyle='--')

    # BOTTOM PLOT: Uncertainty (Gaussian).
    if fitted_threshold is not None and fitted_variance is not None:
      sigma = np.sqrt(fitted_variance)
      # Normal distribution probability density function values.
      pdf_values = ((1.0 / (sigma * np.sqrt(2 * np.pi))) *
                   np.exp(-0.5 * ((plot_levels - fitted_threshold) / sigma)
                          ** 2))

      ax_bot.plot(plot_levels, pdf_values, color='blue', linewidth=2,
                  fillstyle='full')
      ax_bot.fill_between(plot_levels, pdf_values, color='blue', alpha=0.3)
      ax_bot.axvline(fitted_threshold, color='magenta', linestyle='--')

      ax_bot.set_title(f'SD = {sigma:.1f} dB')
    else:
      ax_bot.text(0.5, 0.5, 'N/A', ha='center', va='center')

    ax_bot.grid(True, linestyle='--')

    # Y-ticks formatting
    # Top row always 0-1
    ax_top.set_yticks([0, 0.5, 1])
    ax_top.set_yticklabels(['0', '0.5', '1'])
    if i == 0:
      ax_top.set_ylabel('Prob. Hearing', fontsize=12)
      ax_bot.set_ylabel('Prob. Threshold', fontsize=12)


  # Add common axis labels.
  fig.text(0.5, 0.02, 'Level (dBHL)', ha='center', va='center', fontsize=14)

  plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])


def plot_results(
    results_history: List[Tuple[float, float, bool]],
    true_audiogram: Dict[float, float],
    reconstructed_audiograms: Dict[str, Dict[float, Optional[float]]],
    mae_results_local: Dict[str, Optional[float]],
    num_trials: int
):
  """
  Generates a single, comprehensive plot comparing all reconstructed audiograms
  against the true audiogram.
  """
  plt.figure(figsize=(12, 8))
  ax = plt.gca()
  ax.set_title(f'Simulation results after {num_trials} trials')
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Level (dBHL)')

  # 1. Plot the true audiogram
  true_freqs = sorted(true_audiogram.keys())
  true_thresholds = [true_audiogram[f] for f in true_freqs]
  ax.plot(true_freqs, true_thresholds, color='black', linestyle='--',
          linewidth=2, label='True Audiogram', zorder=10)

  # 2. Plot the raw simulation data points
  sim_freqs = [r[0] for r in results_history]
  sim_levels = [r[1] for r in results_history]
  sim_responses = [r[2] for r in results_history]
  heard_mask = [r for r in sim_responses]
  not_heard_mask = [not r for r in sim_responses]

  ax.scatter(np.array(sim_freqs)[heard_mask],
             np.array(sim_levels)[heard_mask],
             color='lime', marker='o', s=100, label='Heard')
  ax.scatter(np.array(sim_freqs)[not_heard_mask],
             np.array(sim_levels)[not_heard_mask],
             color='red', marker='x', s=100, label='Not Heard')

  # 3. Plot each reconstructed audiogram
  plot_colors = ['blue', 'magenta', 'green', 'orange', 'purple']
  plot_markers = ['*', 's', '^', 'D', 'v']
  for i, (recon_name, r_data) in enumerate(reconstructed_audiograms.items()):
    audiogram_to_plot = r_data
    # If the audiogram data is a dictionary (like for Local/Hybrid fits),
    # extract the 'thresholds' dictionary for plotting.
    if isinstance(r_data, dict) and 'thresholds' in r_data:
      audiogram_to_plot = r_data['thresholds']

    if audiogram_to_plot:
      # Use pre-calculated MAE for the legend
      mae = mae_results_local.get(recon_name)
      label = recon_name
      if mae is not None:
        label = f"{recon_name} (MAE = {mae:.1f} dB)"

      est_freqs = []
      est_thresholds = []
      for freq, thresh in audiogram_to_plot.items():
        if thresh is not None:
          est_freqs.append(freq)
          est_thresholds.append(thresh)

      if est_freqs:
        sort_indices = np.argsort(est_freqs)
        est_freqs_sorted = np.array(est_freqs)[sort_indices]
        est_thresholds_sorted = np.array(est_thresholds)[sort_indices]
        ax.plot(est_freqs_sorted, est_thresholds_sorted,
                marker=plot_markers[i % len(plot_markers)],
                color=plot_colors[i % len(plot_colors)],
                linestyle='-', markersize=8, label=label)

  # Formatting
  ax.set_xscale('log')
  ax.set_xticks(STANDARD_FREQS_HZ)
  ax.set_xticklabels([str(f) for f in STANDARD_FREQS_HZ])
  ax.minorticks_off()
  ax.grid(True, which='both', linestyle=':', linewidth=0.5)
  ax.invert_yaxis()
  ax.legend(loc='lower left')
  plt.tight_layout()


def plot_weight_evolution(weight_history: List[Tuple[float, float]],
                          phase_starts: Dict[str, int]):
  """Plots the evolution of the local vs global model weights over trials."""
  plt.figure(figsize=(12, 6))
  # Unpack history: (average_local_weight, specific_local_weight)
  avg_weights = [x[0] for x in weight_history]

  trials = np.arange(1, len(weight_history) + 1)
  max_trial = len(weight_history)

  # Define phase boundaries
  sweep_start = phase_starts.get('Sweep', max_trial)
  adaptive_start = phase_starts.get('Adaptive', max_trial)

  # Y-position for labels (slightly above plot area)
  label_y = 1.0

  # 1. Descent Phase (Start to Sweep Start)
  if sweep_start > 1:
    plt.axvspan(1, sweep_start, color='red', alpha=0.1)
    plt.text(1 + (sweep_start - 1) / 2, label_y, 'Phase 1: Descent',
             ha='center', va='bottom', color='gray', fontweight='bold')

  # 2. Sweep Phase (Sweep Start to Adaptive Start)
  if adaptive_start > sweep_start:
    plt.axvspan(sweep_start, adaptive_start, color='yellow', alpha=0.1)
    plt.text(sweep_start + (adaptive_start - sweep_start) / 2, label_y,
             'Phase 2: Sweep', ha='center', va='bottom', color='gray',
             fontweight='bold')

  # 3. Adaptive Phase (Adaptive Start to End)
  if max_trial > adaptive_start:
    plt.axvspan(adaptive_start, max_trial + 1, color='green', alpha=0.1)
    plt.text(adaptive_start + (max_trial - adaptive_start) / 2, label_y,
             'Phase 3: Adaptive', ha='center', va='bottom', color='gray',
             fontweight='bold')

  # Mask weights before adaptive phase
  # Trials are 1-based. adaptive_start is the first trial of adaptive phase.
  # We want to plot starting from that trial.
  # Index corresponding to trial T is T-1.
  start_index = adaptive_start - 1
  avg_weights_masked = [w if i >= start_index else np.nan
                        for i, w in enumerate(avg_weights)]

  # Plot Average Lines
  plt.plot(trials, avg_weights_masked, label='Avg Local Weight',
           color='blue', linewidth=3, alpha=0.8)
  plt.plot(trials, [1.0 - w for w in avg_weights_masked],
           label='Avg Global Weight', color='orange', linestyle='--',
           linewidth=3, alpha=0.8)

  plt.xlabel('Trial Number', fontsize=14)
  plt.ylabel('Normalized Weight', fontsize=14)
  plt.title('Evolution of Model Influence (Global vs Local)', fontsize=16,
            y=1.08)
  plt.legend(fontsize=12, loc='center right')
  plt.grid(True, linestyle=':', linewidth=0.5)
  plt.xlim(1, max_trial)
  plt.ylim(0, 1.05)
  plt.tight_layout()


def display_summary_table(results_summary: List[Dict[str, any]],
                          verbosity: int):
  """
  Displays a summary table of MAE results as a figure and in the console.

  Args:
      results_summary: A list of dictionaries, where each dictionary
                       contains the results for one audiogram simulation.
      verbosity: The verbosity level for console and graphical output.
  """
  if not results_summary:
    print('No results to summarize.')
    return

  df = pd.DataFrame(results_summary)
  df = df.set_index('Audiogram')

  # Transpose so methods are rows, and add a 'Mean MAE' column
  summary_df = df.transpose()
  summary_df['Mean MAE'] = summary_df.mean(axis=1)

  # Sort by the best (lowest) mean MAE
  summary_df = summary_df.sort_values(by='Mean MAE')

  if verbosity >= 1:
    print('\n--- MAE Summary (dB) ---')
    print(summary_df.to_string(float_format='%.2f'))

  # --- Plot table as a figure ---
  if verbosity >= 1:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = summary_df.round(2).reset_index()
    table = ax.table(
      cellText=table_data.values,
      colLabels=table_data.columns,
      loc='center',
      cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    fig.suptitle('Mean Absolute Error (MAE) Summary (dB)', fontsize=16)
