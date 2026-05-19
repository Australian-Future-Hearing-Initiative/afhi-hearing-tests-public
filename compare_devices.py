"""Compare audiograms from different devices to calculate calibration offsets.

This script loads Pip PTA audiogram data from two sets of device directories
(e.g. Airpods* and PixelBuds*), plots them for visual comparison, and
calculates the per-frequency offset between the two devices.

Usage:
  python compare_devices.py Airpods_vs_PixelBuds
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Standard audiometric frequencies used in the Pip PTA test.
STANDARD_FREQS_HZ = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

# Colours for plotting.
COLOR_AIRPODS = '#FF6B35'     # Orange for Airpods.
COLOR_PIXELBUDS = '#4285F4'   # Google blue for Pixel Buds.


def load_audiogram_csv(file_path: str) -> dict[int, float]:
  """Loads an audiogram CSV file, returning a freq -> threshold dict.

  Expects a headerless CSV with comment lines prefixed by '#'.
  Each data row is: frequency_hz,threshold_db_hl

  Args:
    file_path: Path to the audiogram CSV file.

  Returns:
    A dictionary mapping frequency (int Hz) to threshold (float dB HL).
  """
  data = pd.read_csv(file_path, comment='#', header=None,
                      names=['frequency_hz', 'threshold_db_hl'])
  return {int(row['frequency_hz']): float(row['threshold_db_hl'])
          for _, row in data.iterrows()}


def discover_device_dirs(base_dir: str
                         ) -> tuple[list[str], list[str]]:
  """Finds all Airpods* and PixelBuds* subdirectories.

  Args:
    base_dir: The base directory to search.

  Returns:
    A tuple of (airpods_dirs, pixelbuds_dirs), each a sorted list of
    absolute paths.
  """
  airpods_dirs = []
  pixelbuds_dirs = []
  for item in sorted(os.listdir(base_dir)):
    full_path = os.path.join(base_dir, item)
    if not os.path.isdir(full_path):
      continue
    # Case-insensitive check to handle "AirPods" vs "Airpods".
    item_lower = item.lower()
    if item_lower.startswith('airpods'):
      airpods_dirs.append(full_path)
    elif item_lower.startswith('pixelbuds'):
      pixelbuds_dirs.append(full_path)
  return airpods_dirs, pixelbuds_dirs


def load_all_audiograms(dirs: list[str]
                        ) -> tuple[list[tuple[str, dict[int, float]]],
                                   list[tuple[str, dict[int, float]]]]:
  """Loads left and right audiograms from each directory.

  Args:
    dirs: List of directory paths, each containing pip_left_audiogram.csv
      and pip_right_audiogram.csv.

  Returns:
    A tuple of (left_audiograms, right_audiograms), each a list of
    (label, audiogram_dict) tuples.
  """
  left_audiograms = []
  right_audiograms = []
  for d in dirs:
    dir_name = os.path.basename(d)
    loaded_any = False
    # Check for both pip and pta test file prefixes.
    for prefix in ['pip', 'pta']:
      left_path = os.path.join(d, f'{prefix}_left_audiogram.csv')
      right_path = os.path.join(d, f'{prefix}_right_audiogram.csv')
      found_part = False
      if os.path.isfile(left_path):
        left_audiograms.append(
            (f'{dir_name} L', load_audiogram_csv(left_path)))
        found_part = True
      if os.path.isfile(right_path):
        right_audiograms.append(
            (f'{dir_name} R', load_audiogram_csv(right_path)))
        found_part = True
      if found_part:
        loaded_any = True
        # Once we found the files for this directory, we can move to the next.
        break
    if not loaded_any:
      print(f'Warning: Neither pip nor pta audiogram files found in {d}.')

  return left_audiograms, right_audiograms


def compute_mean_thresholds(
    audiograms: list[tuple[str, dict[int, float]]]
) -> dict[int, float]:
  """Computes the mean threshold at each frequency across all audiograms.

  Args:
    audiograms: A list of (label, audiogram_dict) tuples.

  Returns:
    A dictionary mapping frequency (int Hz) to mean threshold (float dB HL).
  """
  # Collect all thresholds for each frequency.
  freq_values = {f: [] for f in STANDARD_FREQS_HZ}
  for _, audiogram in audiograms:
    for freq in STANDARD_FREQS_HZ:
      if freq in audiogram:
        freq_values[freq].append(audiogram[freq])
  return {freq: np.mean(values) for freq, values in freq_values.items()
          if values}


def compute_offsets(mean_airpods: dict[int, float],
                    mean_pixelbuds: dict[int, float]) -> dict[int, float]:
  """Computes per-frequency offsets: Airpods mean - PixelBuds mean.

  Args:
    mean_airpods: Mean thresholds for Airpods at each frequency.
    mean_pixelbuds: Mean thresholds for PixelBuds at each frequency.

  Returns:
    A dictionary mapping frequency (int Hz) to offset (float dB).
  """
  offsets = {}
  for freq in STANDARD_FREQS_HZ:
    if freq in mean_airpods and freq in mean_pixelbuds:
      offsets[freq] = mean_airpods[freq] - mean_pixelbuds[freq]
  return offsets


def _format_audiogram_axis(ax, title: str):
  """Applies standard audiogram formatting to a matplotlib axis."""
  freqs = sorted(STANDARD_FREQS_HZ)
  ax.set_title(title, fontsize=13)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Threshold (dB HL)')
  ax.set_xlim(200, 10000)
  ax.set_ylim(40, -20)  # Audiogram convention: inverted y-axis.
  ax.set_xticks(freqs)
  ax.set_xticklabels([str(f) for f in freqs])
  ax.legend()
  ax.grid(True, linestyle='--', alpha=0.6)


def _plot_audiograms_on_axis(ax, airpods_list, pixelbuds_list):
  """Plots individual audiograms for both devices on a single axis."""
  for i, (_, audiogram) in enumerate(airpods_list):
    f = sorted(audiogram.keys())
    vals = [audiogram[freq] for freq in f]
    ax.semilogx(f, vals, marker='o', markersize=4, color=COLOR_AIRPODS,
                alpha=0.4, label='Airpods' if i == 0 else None)
  for i, (_, audiogram) in enumerate(pixelbuds_list):
    f = sorted(audiogram.keys())
    vals = [audiogram[freq] for freq in f]
    ax.semilogx(f, vals, marker='x', markersize=5, color=COLOR_PIXELBUDS,
                alpha=0.4, label='Pixel Buds' if i == 0 else None)


def plot_individual_audiograms(
    airpods_left, airpods_right, pixelbuds_left, pixelbuds_right):
  """Figure 1: Individual audiograms split into left and right subplots.

  Args:
    airpods_left: List of (label, audiogram_dict) for Airpods left ears.
    airpods_right: List of (label, audiogram_dict) for Airpods right ears.
    pixelbuds_left: List of (label, audiogram_dict) for PixelBuds left ears.
    pixelbuds_right: List of (label, audiogram_dict) for PixelBuds right ears.
  """
  _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
  _plot_audiograms_on_axis(ax_left, airpods_left, pixelbuds_left)
  _format_audiogram_axis(ax_left, 'Left Ear — Individual Audiograms')
  _plot_audiograms_on_axis(ax_right, airpods_right, pixelbuds_right)
  _format_audiogram_axis(ax_right, 'Right Ear — Individual Audiograms')
  plt.tight_layout()
  plt.show()


def plot_mean_audiograms(mean_airpods: dict[int, float],
                         mean_pixelbuds: dict[int, float]):
  """Figure 2: Mean audiograms for each device.

  Args:
    mean_airpods: Mean thresholds for Airpods at each frequency.
    mean_pixelbuds: Mean thresholds for PixelBuds at each frequency.
  """
  freqs = sorted(STANDARD_FREQS_HZ)
  mean_a_vals = [mean_airpods[f] for f in freqs]
  mean_p_vals = [mean_pixelbuds[f] for f in freqs]

  _, ax = plt.subplots(figsize=(10, 5))
  ax.semilogx(freqs, mean_a_vals, marker='o', linewidth=2,
              color=COLOR_AIRPODS, label='Airpods mean')
  ax.semilogx(freqs, mean_p_vals, marker='x', linewidth=2,
              color=COLOR_PIXELBUDS, label='Pixel Buds mean')
  ax.set_title('Mean Audiograms by Device', fontsize=14)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Threshold (dB HL)')
  ax.set_xlim(200, 10000)
  ax.set_ylim(30, -15)
  ax.set_xticks(freqs)
  ax.set_xticklabels([str(f) for f in freqs])
  ax.legend()
  ax.grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.show()


def main():
  parser = argparse.ArgumentParser(
      description='Compare Pip PTA audiograms from different devices.')
  parser.add_argument('input_dir',
                      help='Directory containing device subdirectories '
                           '(e.g. AirPods1, PixelBuds1, etc.)')
  args = parser.parse_args()

  if not os.path.isdir(args.input_dir):
    print(f'Error: {args.input_dir} is not a valid directory.')
    return

  # Discover device directories.
  airpods_dirs, pixelbuds_dirs = discover_device_dirs(args.input_dir)
  print(f'Found {len(airpods_dirs)} Airpods director(ies): '
        f'{[os.path.basename(d) for d in airpods_dirs]}')
  print(f'Found {len(pixelbuds_dirs)} PixelBuds director(ies): '
        f'{[os.path.basename(d) for d in pixelbuds_dirs]}')

  if not airpods_dirs or not pixelbuds_dirs:
    print('Error: Need at least one Airpods and one PixelBuds directory.')
    return

  # Load all audiograms, separated by ear.
  airpods_left, airpods_right = load_all_audiograms(airpods_dirs)
  pixelbuds_left, pixelbuds_right = load_all_audiograms(pixelbuds_dirs)
  all_airpods = airpods_left + airpods_right
  all_pixelbuds = pixelbuds_left + pixelbuds_right
  print(f'\nLoaded {len(all_airpods)} Airpods audiograms '
        f'(L+R from {len(airpods_dirs)} directories).')
  print(f'Loaded {len(all_pixelbuds)} PixelBuds audiograms '
        f'(L+R from {len(pixelbuds_dirs)} directories).')

  # Compute mean thresholds for each device (across both ears).
  mean_airpods = compute_mean_thresholds(all_airpods)
  mean_pixelbuds = compute_mean_thresholds(all_pixelbuds)

  # Compute offsets.
  offsets = compute_offsets(mean_airpods, mean_pixelbuds)

  # Print results.
  print('\n--- Mean Thresholds (dB HL) ---')
  header = ('Freq (Hz)'.ljust(12) + 'Airpods'.ljust(12) +
            'PixelBuds'.ljust(12) + 'Offset'.ljust(12))
  print(header)
  for freq in STANDARD_FREQS_HZ:
    print(f'{freq:<12} {mean_airpods[freq]:<12.2f} '
          f'{mean_pixelbuds[freq]:<12.2f} {offsets[freq]:<12.2f}')

  print('\n--- Offset Dict (Airpods - PixelBuds) ---')
  print('{')
  for freq in STANDARD_FREQS_HZ:
    print(f'  {freq}: {offsets[freq]:.1f},')
  print('}')

  # Generate figures.
  plot_individual_audiograms(airpods_left, airpods_right,
                             pixelbuds_left, pixelbuds_right)
  plot_mean_audiograms(mean_airpods, mean_pixelbuds)


if __name__ == '__main__':
  main()
