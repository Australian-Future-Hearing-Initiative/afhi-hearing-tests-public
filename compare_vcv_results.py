"""Compare adaptive VCV test results across conditions.

Reads a folder of subfolders (e.g. local_data/VCV_comparison/), each
containing vcv_srt_estimates.csv, and plots grouped class-mean SRTs with
95% confidence intervals.

Example:
  python compare_vcv_results.py local_data/VCV_comparison
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import bayesian_vcv_estimator


ESTIMATES_FILENAME = 'vcv_srt_estimates.csv'
OUTPUT_DIR = 'local_results'
CI_Z = 1.96
PLOT_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE']

STANDARD_10 = frozenset({
    'B', 'D', 'G', 'K', 'N', 'S', 'SH', 'T', 'V', 'Z',
})

CONSONANT_CLASSES = bayesian_vcv_estimator.CONSONANT_CLASSES
CLASS_DISPLAY_ORDER = bayesian_vcv_estimator.CLASS_DISPLAY_ORDER

REQUIRED_COLUMNS = [
    'Ear', 'Consonant', 'SRT (dB)', 'Uncertainty (SD)',
]


def discover_condition_dirs(comparison_dir: str) -> list[tuple[str, str]]:
  """Returns (folder_name, csv_path) for each immediate subfolder."""
  if not os.path.isdir(comparison_dir):
    print(f'Error: not a directory: {comparison_dir}')
    sys.exit(1)

  subdirs = sorted(
      name for name in os.listdir(comparison_dir)
      if os.path.isdir(os.path.join(comparison_dir, name))
  )
  if not subdirs:
    print(f'Error: no subfolders found in {comparison_dir}')
    sys.exit(1)

  conditions = []
  for name in subdirs:
    csv_path = os.path.join(comparison_dir, name, ESTIMATES_FILENAME)
    if not os.path.isfile(csv_path):
      print(
          f'Error: missing {ESTIMATES_FILENAME} in '
          f'{os.path.join(comparison_dir, name)}'
      )
      sys.exit(1)
    conditions.append((name, csv_path))

  return conditions


def load_estimates(csv_path: str) -> pd.DataFrame:
  """Loads and validates column names for an estimates CSV."""
  df = pd.read_csv(csv_path)
  missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
  if missing:
    print(f'Error: {csv_path} missing columns: {missing}')
    sys.exit(1)
  return df


def get_ear_mode(df: pd.DataFrame) -> str:
  """Returns 'both' or 'separate' based on Ear values present."""
  ears = set(df['Ear'].unique())
  if ears == {'both'}:
    return 'both'
  if ears == {'left', 'right'}:
    return 'separate'
  print(
      f'Error: unsupported Ear values {sorted(ears)}. '
      "Expected 'both' only, or 'left' and 'right'."
  )
  sys.exit(1)


def get_required_members(
    cls_name: str,
    file_consonants: set[str],
) -> set[str]:
  """Members of cls_name that must be present for this file."""
  class_members = set(CONSONANT_CLASSES[cls_name]['members'])
  if STANDARD_10 <= file_consonants:
    return class_members & STANDARD_10
  return class_members & file_consonants


def compute_class_statistics(
    df: pd.DataFrame,
    ear: str | None = None,
    condition_name: str = '',
) -> dict[str, dict[str, float]]:
  """Computes class-mean SRT and 95% CI half-width for one condition."""
  if ear is not None:
    df = df[df['Ear'] == ear]
  if df.empty:
    label = condition_name or 'unknown'
    print(f'Error: no data for ear={ear!r} in {label}')
    sys.exit(1)

  file_consonants = set(df['Consonant'].unique())
  stats = {}
  for cls_name in CLASS_DISPLAY_ORDER:
    required = get_required_members(cls_name, file_consonants)
    if not required:
      print(
          f'Error: no consonants for class {cls_name} in '
          f'{condition_name}'
      )
      sys.exit(1)

    rows = df[df['Consonant'].isin(required)]
    present = set(rows['Consonant'].unique())
    missing = required - present
    if missing:
      print(
          f'Error: {condition_name} missing class {cls_name} '
          f'consonants: {sorted(missing)}'
      )
      sys.exit(1)

    srts = rows.set_index('Consonant').loc[sorted(required), 'SRT (dB)']
    sds = rows.set_index('Consonant').loc[
        sorted(required), 'Uncertainty (SD)'
    ]
    if srts.isna().any() or sds.isna().any():
      print(f'Error: NaN SRT/SD values in {condition_name}, class {cls_name}')
      sys.exit(1)

    n = len(required)
    mean_srt = float(srts.mean())
    se = float(np.sqrt(np.sum(sds.values ** 2)) / n)
    stats[cls_name] = {
        'mean': mean_srt,
        'ci_half': CI_Z * se,
    }
  return stats


def create_comparison_plot(
    stats_by_condition: dict[str, dict[str, dict[str, float]]],
    ear_label: str | None = None,
) -> plt.Figure:
  """Creates a grouped plot of class-mean SRTs with 95% CI intervals.

  Each condition is shown as a coloured block spanning its 95% CI, with
  a tick mark at the class-mean SRT. This avoids anchoring bars to 0 dB.
  """
  conditions = list(stats_by_condition.keys())
  n_classes = len(CLASS_DISPLAY_ORDER)
  n_cond = len(conditions)
  x = np.arange(n_classes)
  group_width = 0.8
  bar_width = group_width / n_cond

  fig, ax = plt.subplots(figsize=(10, 6))

  for j, cond_name in enumerate(conditions):
    offset = (j - (n_cond - 1) / 2.0) * bar_width
    color = PLOT_COLORS[j % len(PLOT_COLORS)]
    stats = stats_by_condition[cond_name]
    for i, cls_name in enumerate(CLASS_DISPLAY_ORDER):
      mean_srt = stats[cls_name]['mean']
      ci_half = stats[cls_name]['ci_half']
      x_pos = x[i] + offset
      ax.bar(
          x_pos,
          2.0 * ci_half,
          width=bar_width,
          bottom=mean_srt - ci_half,
          color=color,
          edgecolor='white',
          linewidth=0.5,
          alpha=0.85,
      )
      half_w = bar_width * 0.45
      ax.plot(
          [x_pos - half_w, x_pos + half_w],
          [mean_srt, mean_srt],
          color='black',
          linewidth=1.5,
      )

  handles = [
      Patch(
          facecolor=PLOT_COLORS[j % len(PLOT_COLORS)],
          alpha=0.85,
          label=cond_name,
      )
      for j, cond_name in enumerate(conditions)
  ]
  handles.append(
      Line2D(
          [0], [0], color='black', linewidth=1.5,
          label='Class-mean SRT',
      )
  )

  title = 'VCV class-mean SRT comparison (95% CI)'
  if ear_label:
    title += f' — {ear_label}'

  ax.set_xticks(x)
  ax.set_xticklabels(CLASS_DISPLAY_ORDER)
  ax.set_ylabel('Mean SRT (dB SNR)', fontsize=12)
  ax.set_xlabel('Consonant class', fontsize=12)
  ax.set_title(title, fontsize=14)
  ax.legend(handles=handles, fontsize=10)
  ax.grid(axis='y', linestyle='--', alpha=0.7)

  y_lo, y_hi = ax.get_ylim()
  ax.set_ylim(y_lo - 5, y_hi + 5)

  fig.tight_layout()
  return fig


def save_plot(fig: plt.Figure, ear_suffix: str | None = None) -> str:
  """Saves figure to local_results/ and returns the file path."""
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  if ear_suffix:
    filename = f'vcv_class_comparison_{ear_suffix}_{timestamp}.png'
  else:
    filename = f'vcv_class_comparison_{timestamp}.png'
  output_path = os.path.join(OUTPUT_DIR, filename)
  fig.savefig(output_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  return output_path


def run_comparison(comparison_dir: str) -> list[str]:
  """Loads data, validates, plots, and saves PNG(s).

  Returns:
    List of output file paths written.
  """
  conditions = discover_condition_dirs(comparison_dir)
  print(f'Found {len(conditions)} condition(s) in {comparison_dir}')

  loaded: list[tuple[str, pd.DataFrame]] = []
  reference_consonants: set[str] | None = None
  ear_mode: str | None = None

  for name, csv_path in conditions:
    df = load_estimates(csv_path)
    mode = get_ear_mode(df)
    if ear_mode is None:
      ear_mode = mode
    elif mode != ear_mode:
      print(
          f'Error: inconsistent ear mode in {name} '
          f"(expected {ear_mode}, found {mode})"
      )
      sys.exit(1)

    consonants = set(df['Consonant'].unique())
    if reference_consonants is None:
      reference_consonants = consonants
    elif consonants != reference_consonants:
      print(
          f'Error: consonant set mismatch in {name}. '
          f'Expected {sorted(reference_consonants)}, '
          f'found {sorted(consonants)}.'
      )
      sys.exit(1)

    loaded.append((name, df))

  assert ear_mode is not None
  output_paths = []

  if ear_mode == 'both':
    stats_by_condition = {}
    for name, df in loaded:
      stats_by_condition[name] = compute_class_statistics(
          df, ear='both', condition_name=name,
      )
    fig = create_comparison_plot(stats_by_condition)
    path = save_plot(fig)
    output_paths.append(path)
  else:
    for ear in ('left', 'right'):
      stats_by_condition = {}
      for name, df in loaded:
        stats_by_condition[name] = compute_class_statistics(
            df, ear=ear, condition_name=f'{name} ({ear})',
        )
      fig = create_comparison_plot(
          stats_by_condition,
          ear_label=f'{ear.capitalize()}',
      )
      path = save_plot(fig, ear_suffix=ear)
      output_paths.append(path)

  return output_paths


def main():
  parser = argparse.ArgumentParser(
      description=(
          'Compare adaptive VCV class-mean SRTs across test conditions.'
      ),
  )
  parser.add_argument(
      'comparison_dir',
      type=str,
      help=(
          'Folder containing one subfolder per condition, each with '
          f'{ESTIMATES_FILENAME}.'
      ),
  )
  args = parser.parse_args()

  print('Starting VCV results comparison...')
  output_paths = run_comparison(args.comparison_dir)
  for path in output_paths:
    print(f'Plot saved to: {path}')


if __name__ == '__main__':
  main()
