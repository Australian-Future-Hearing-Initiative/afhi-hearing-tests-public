"""Useful diagnostic scripts for the codebase, including plotting etc."""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import glob
import argparse
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages

import audio_synthesis
import common
import calibration

# matplotlib.use('TkAgg') # No longer needed, let matplotlib auto-select


def _create_noisy_stimulus(
    file_path: str, stimulus_type: str, snr_db: float | None = None
) -> tuple[np.ndarray, int]:
  """Reproduces the audio processing pipeline from the main app to create a
  scaled, and optionally noisy, VCV stimulus.
  This version uses a manual gain adjustment for synthetic VCVs, mirroring
  the logic used in the waveform/PSD plots in this script, which has been
  verified as visually correct.

  Args:
    file_path: The path to the clean (unscaled) VCV file.
    stimulus_type: Either 'Human' or 'Synthetic'.
    snr_db: The desired signal-to-noise ratio in dB. If None, no noise
            is added.

  Returns:
    A tuple containing the processed audio data (as a numpy array) and the
    sample rate.
  """
  # 1. Load the raw audio data.
  processed_data, sample_rate, _ = common.read_wav_as_float(file_path)

  # 2. If synthetic, apply the manual calibration offset.
  if stimulus_type == 'Synthetic':
    gain = 10**(-calibration.SYNTHETIC_VCV_DB_SPL_OFFSET / 20.0)
    processed_data = processed_data * gain

  # 3. If the stimulus is synthetic, add the leading silence.
  if stimulus_type == 'Synthetic':
    processed_data = common.prepend_silence(
        processed_data,
        sample_rate,
        calibration.SYNTHETIC_VCV_LEADING_SILENCE_S
    )

  # 4. If an SNR is specified, mix the scaled audio with noise.
  if snr_db is not None:
    # The ear doesn't matter here as mix_vcv_with_noise produces a stereo
    # signal regardless, which is needed for analysis.
    mixed_audio, _ = audio_synthesis.mix_vcv_with_noise(
        snr_db=snr_db,
        ear='both',
        noise_type='Advanced Speech-Shaped Noise',
        audio_data=processed_data,
        sample_rate=sample_rate
    )
    return mixed_audio, sample_rate

  return processed_data, sample_rate


def save_wav(file_path: str, noise: np.ndarray, sample_rate: int):
  """Normalizes noise and saves it as a 16-bit WAV file.

  Args:
    file_path: The path to save the WAV file to.
    noise: The audio data to save.
    sample_rate: The sample rate of the audio.
  """
  # Peak-normalize to the range [-1.0, 1.0].
  normalized_noise = noise / np.max(np.abs(noise))
  # Convert to 16-bit integer format for standard WAV files.
  noise_16bit = np.int16(normalized_noise * 32767)
  write(file_path, sample_rate, noise_16bit)
  print(f"Saved '{file_path}'")


def analyze_and_plot_noise_spectrums():
  """
  Generates, saves, and plots the power spectrums of basic and advanced
  speech-shaped noise.
  """
  # Parameters
  sample_rate = 44100  # Hz
  duration = 5  # seconds
  num_samples = int(duration * sample_rate)
  output_dir = 'diagnostics_output'
  os.makedirs(output_dir, exist_ok=True)

  # Generate a single source of white noise.
  white_noise = np.random.randn(num_samples)

  # Create the shaped noise versions from the same source.
  basic_noise = audio_synthesis.basic_speech_shaped_noise(
      white_noise, sample_rate
  )
  advanced_noise = audio_synthesis.advanced_speech_shaped_noise(
      white_noise, sample_rate
  )

  # Save WAV files for listening.
  save_wav(
      os.path.join(output_dir, 'white_noise.wav'), white_noise, sample_rate
  )
  save_wav(
      os.path.join(output_dir, 'basic_speech_shaped_noise.wav'),
      basic_noise,
      sample_rate,
  )
  save_wav(
      os.path.join(output_dir, 'advanced_speech_shaped_noise.wav'),
      advanced_noise,
      sample_rate,
  )

  # Visualize the spectrum.
  white_freqs, white_psd = signal.welch(
      white_noise, fs=sample_rate, nperseg=2048
  )
  advanced_freqs, advanced_psd = signal.welch(
      advanced_noise, fs=sample_rate, nperseg=2048
  )
  basic_freqs, basic_psd = signal.welch(
      basic_noise, fs=sample_rate, nperseg=2048
  )
  plt.figure(figsize=(10, 6))
  plt.semilogx(
      white_freqs,
      10 * np.log10(white_psd),
      label='Original White Noise',
      color='gray',
      linestyle=':',
      alpha=0.7,
  )
  plt.semilogx(
      advanced_freqs, 10 * np.log10(advanced_psd), label='Advanced Model'
  )
  plt.semilogx(
      basic_freqs, 10 * np.log10(basic_psd), alpha=0.8, label='Basic Model'
  )
  plt.title('Power Spectrum of Speech-Shaped Noise Generation Methods')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Power/Frequency (dB/Hz)')
  plt.xlim(20, sample_rate / 2)
  plt.grid(True, which='both', linestyle='--', linewidth=0.5)
  plt.legend()
  plt.show()


def _get_activity_boundaries(audio_data: np.ndarray, threshold_db: float):
  """Helper to find the start, end, and threshold of active audio."""
  peak_amplitude = np.max(np.abs(audio_data))
  if peak_amplitude == 0:
    return 0, len(audio_data), 0  # Treat silent signal as fully 'active'.

  threshold_linear = peak_amplitude * (10 ** (threshold_db / 20.0))
  active_indices = np.where(np.abs(audio_data) >= threshold_linear)[0]

  if len(active_indices) == 0:
    return 0, len(audio_data), threshold_linear

  start_index = active_indices[0]
  end_index = active_indices[-1]
  return start_index, end_index, threshold_linear


def visualize_vcv_pairs(threshold_db: float = common.DEFAULT_TRIM_DB_THRESHOLD):
  """
  Generates plots comparing pairs of human and synthetic VCVs, showing
  the effect of the silence detection threshold.
  """
  stimuli_dir = 'stimuli'
  clean_dir = os.path.join(stimuli_dir, 'clean')
  synthetic_dir = os.path.join(stimuli_dir, 'synthetic')

  # Basic labels to find corresponding files.
  consonant_labels = ['aba', 'ada', 'aga', 'aka', 'ana', 'asa', 'asha',
                      'ata', 'ava', 'aza']
  # Get a list of all human VCV files to iterate through.
  all_human_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))

  for clean_path in all_human_files:
    # Extract the base consonant label from the human filename.
    base_name = os.path.basename(clean_path)
    found_label = None
    for label in consonant_labels:
      if label in base_name:
        found_label = label
        break

    if not found_label:
      print(f'Warning: Could not determine label for "{base_name}", skipping.')
      continue

    # Find the corresponding synthetic file for the extracted label.
    synth_files = glob.glob(os.path.join(synthetic_dir, f'*{found_label}*.wav'))

    if not synth_files:
      print(
          'Warning: Could not find a synthetic match for label '
          f'"{found_label}", skipping.'
      )
      continue

    # Take the first match for simplicity (there should only be one).
    synth_path = synth_files[0]

    # --- Plotting (Clean Signals) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   sharey=True, sharex=True)
    fig.suptitle(
        f'Comparison for VCV: "{found_label.upper()}" (Clean Calibrated)',
        fontsize=16
    )

    # Process and plot Human VCV.
    clean_signal, fs, _ = common.read_wav_as_float(clean_path)
    time_ax = np.linspace(0., len(clean_signal) / fs, len(clean_signal))
    start, end, thresh = _get_activity_boundaries(clean_signal, threshold_db)
    rms_dbfs = 20 * np.log10(common.get_active_signal_rms(clean_signal))

    ax1.plot(time_ax, clean_signal, label='Waveform', color='blue')
    ax1.axvspan(0, start / fs, color='red', alpha=0.2, label='Silence')
    ax1.axvspan(end / fs, time_ax[-1], color='red', alpha=0.2)
    ax1.axhline(thresh, ls=':', color='gray', label=f'{threshold_db} dB Thresh')
    ax1.axhline(-thresh, ls=':', color='gray')
    ax1.set_title(
        f'Human: {os.path.basename(clean_path)}\n'
        f'Active RMS: {rms_dbfs:.2f} dBFS'
    )
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Process and plot Synthetic VCV.
    synth_signal, fs, _ = common.read_wav_as_float(synth_path)
    # Apply the calibration offset to the synthetic signal for comparison.
    gain = 10**(-calibration.SYNTHETIC_VCV_DB_SPL_OFFSET / 20.0)
    synth_signal = synth_signal * gain

    # Add the leading silence to the synthetic signal.
    synth_signal = common.prepend_silence(
        synth_signal, fs, calibration.SYNTHETIC_VCV_LEADING_SILENCE_S
    )

    time_ax = np.linspace(0., len(synth_signal) / fs, len(synth_signal))
    start, end, thresh = _get_activity_boundaries(synth_signal, threshold_db)
    # Use a small epsilon to avoid log10(0) for silent parts.
    rms_val = common.get_active_signal_rms(synth_signal)
    rms_dbfs = 20 * np.log10(rms_val if rms_val > 0 else 1e-9)


    ax2.plot(time_ax, synth_signal, label='Waveform', color='green')
    ax2.axvspan(0, start / fs, color='red', alpha=0.2, label='Silence')
    ax2.axvspan(end / fs, time_ax[-1], color='red', alpha=0.2)
    ax2.axhline(thresh, ls=':', color='gray', label=f'{threshold_db} dB Thresh')
    ax2.axhline(-thresh, ls=':', color='gray')
    ax2.set_title(
        f'Synthetic (Calibrated): {os.path.basename(synth_path)}\n'
        f'Active RMS: {rms_dbfs:.2f} dBFS'
    )
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()

    # --- Plotting (Power Spectral Density) ---
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    fig2.suptitle(
        f'PSD Comparison for VCV: "{found_label.upper()}"',
        fontsize=16
    )

    # Human VCV PSD.
    freqs_human, psd_human = signal.welch(clean_signal, fs, nperseg=1024)
    ax3.semilogx(freqs_human, 10 * np.log10(psd_human), label='Human')

    # Synthetic VCV PSD (using the calibrated signal from the plot above).
    freqs_synth, psd_synth = signal.welch(synth_signal, fs, nperseg=1024)
    ax3.semilogx(
        freqs_synth, 10 * np.log10(psd_synth), label='Synthetic (Calibrated)'
    )

    ax3.set_title('Power Spectral Density Comparison')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power/Frequency (dB/Hz)')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Plotting (Mel Spectrogram Comparison) ---
    noisy_snr_db = -3.0
    clean_human, fs_h = _create_noisy_stimulus(clean_path, 'Human', None)
    noisy_human, _ = _create_noisy_stimulus(clean_path, 'Human', noisy_snr_db)
    clean_synth, fs_s = _create_noisy_stimulus(synth_path, 'Synthetic', None)
    noisy_synth, _ = _create_noisy_stimulus(synth_path, 'Synthetic',
                                            noisy_snr_db)

    # If signals are stereo, convert to mono for librosa
    if clean_human.ndim > 1:
      clean_human = np.mean(clean_human, axis=1)
    if noisy_human.ndim > 1:
      noisy_human = np.mean(noisy_human, axis=1)
    if clean_synth.ndim > 1:
      clean_synth = np.mean(clean_synth, axis=1)
    if noisy_synth.ndim > 1:
      noisy_synth = np.mean(noisy_synth, axis=1)

    # Calculate Mel spectrograms and convert to dB
    mel_h = librosa.feature.melspectrogram(y=clean_human, sr=fs_h)
    mel_h_db = librosa.power_to_db(mel_h, ref=np.max)

    mel_h_noisy = librosa.feature.melspectrogram(y=noisy_human, sr=fs_h)
    mel_h_noisy_db = librosa.power_to_db(mel_h_noisy, ref=np.max)

    mel_s = librosa.feature.melspectrogram(y=clean_synth, sr=fs_s)
    mel_s_db = librosa.power_to_db(mel_s, ref=np.max)

    mel_s_noisy = librosa.feature.melspectrogram(y=noisy_synth, sr=fs_s)
    mel_s_noisy_db = librosa.power_to_db(mel_s_noisy, ref=np.max)

    # Create the 2x2 plot
    fig3, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    fig3.suptitle(f'Mel Spectrogram Comparison for: "{found_label.upper()}"',
                  fontsize=16)

    # Determine a common color scale from all four spectrograms
    all_sxx_values = np.concatenate([
        mel_h_db.flatten(), mel_h_noisy_db.flatten(),
        mel_s_db.flatten(), mel_s_noisy_db.flatten()
    ])
    vmin = np.percentile(all_sxx_values, 5)
    vmax = np.percentile(all_sxx_values, 99)

    # Plot Clean Human (Top Left)
    img1 = librosa.display.specshow(mel_h_db, sr=fs_h, x_axis='time',
                                    y_axis='mel', ax=axs[0, 0], vmin=vmin,
                                    vmax=vmax)
    axs[0, 0].set_title('Human (Clean)')
    axs[0, 0].set_ylabel('Frequency [Hz]')

    # Plot Noisy Human (Top Right)
    librosa.display.specshow(mel_h_noisy_db, sr=fs_h, x_axis='time',
                             y_axis='mel', ax=axs[0, 1], vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f'Human (Noisy, {noisy_snr_db} dB SNR)')

    # Plot Clean Synthetic (Bottom Left)
    librosa.display.specshow(mel_s_db, sr=fs_s, x_axis='time', y_axis='mel',
                             ax=axs[1, 0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Synthetic (Clean)')
    axs[1, 0].set_xlabel('Time [sec]')
    axs[1, 0].set_ylabel('Frequency [Hz]')

    # Plot Noisy Synthetic (Bottom Right)
    librosa.display.specshow(mel_s_noisy_db, sr=fs_s, x_axis='time',
                             y_axis='mel', ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_title(f'Synthetic (Noisy, {noisy_snr_db} dB SNR)')
    axs[1, 1].set_xlabel('Time [sec]')

    fig3.colorbar(img1, ax=axs, format='%+2.0f dB',
                  label='Intensity [dB]')
    plt.show()


def create_vcv_report(
    output_file: str, threshold_db: float = common.DEFAULT_TRIM_DB_THRESHOLD
):
  """
  Generates a single multi-page PDF with all diagnostic plots for each VCV pair.

  Each page of the PDF corresponds to one VCV and contains three analyses:
  1. Time-domain waveform comparison (Human vs. Synthetic).
  2. Power Spectral Density (PSD) comparison.
  3. Mel spectrogram comparison (Clean vs. Noisy for both types).

  Args:
    output_file: The path to save the generated PDF report.
    threshold_db: The dB threshold for detecting active signal parts.
  """
  stimuli_dir = 'stimuli'
  clean_dir = os.path.join(stimuli_dir, 'clean')
  synthetic_dir = os.path.join(stimuli_dir, 'synthetic')
  consonant_labels = [
      'aba', 'ada', 'aga', 'aka', 'ana', 'asa', 'asha', 'ata', 'ava', 'aza'
  ]
  all_human_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))

  with PdfPages(output_file) as pdf:
    print(f"Generating report, saving to '{output_file}'...")
    for i, clean_path in enumerate(all_human_files):
      base_name = os.path.basename(clean_path)
      found_label = next((l for l in consonant_labels if l in base_name), None)

      if not found_label:
        print(f'Skipping {base_name}, could not determine label.')
        continue

      synth_path = os.path.join(synthetic_dir, f'{found_label}.wav')
      if not os.path.exists(synth_path):
        print(f'Skipping {found_label}, synthetic file not found.')
        continue

      print(f'  ({i+1}/{len(all_human_files)}) '
            f'Processing: {found_label.upper()}')

      # --- 1. Data Processing ---
      # Waveform data
      clean_signal, fs, _ = common.read_wav_as_float(clean_path)
      synth_signal_raw, _, _ = common.read_wav_as_float(synth_path)
      gain = 10 ** (-calibration.SYNTHETIC_VCV_DB_SPL_OFFSET / 20.0)
      synth_signal = common.prepend_silence(
          synth_signal_raw * gain, fs,
          calibration.SYNTHETIC_VCV_LEADING_SILENCE_S
      )

      # Mel spectrogram data
      noisy_snr_db = -3.0
      clean_human, fs_h = _create_noisy_stimulus(clean_path, 'Human', None)
      noisy_human, _ = _create_noisy_stimulus(
          clean_path, 'Human', noisy_snr_db
      )
      clean_synth, fs_s = _create_noisy_stimulus(synth_path, 'Synthetic', None)
      noisy_synth, _ = _create_noisy_stimulus(
          synth_path, 'Synthetic', noisy_snr_db
      )

      # Mono conversion for librosa
      mono_clean_human = (np.mean(clean_human, axis=1)
                          if clean_human.ndim > 1 else clean_human)
      mono_noisy_human = (np.mean(noisy_human, axis=1)
                          if noisy_human.ndim > 1 else noisy_human)
      mono_clean_synth = (np.mean(clean_synth, axis=1)
                          if clean_synth.ndim > 1 else clean_synth)
      mono_noisy_synth = (np.mean(noisy_synth, axis=1)
                          if noisy_synth.ndim > 1 else noisy_synth)

      # Calculate Mel spectrograms in dB
      mel_h_db = librosa.power_to_db(librosa.feature.melspectrogram(
          y=mono_clean_human, sr=fs_h), ref=np.max)
      mel_h_noisy_db = librosa.power_to_db(librosa.feature.melspectrogram(
          y=mono_noisy_human, sr=fs_h), ref=np.max)
      mel_s_db = librosa.power_to_db(librosa.feature.melspectrogram(
          y=mono_clean_synth, sr=fs_s), ref=np.max)
      mel_s_noisy_db = librosa.power_to_db(librosa.feature.melspectrogram(
          y=mono_noisy_synth, sr=fs_s), ref=np.max)

      # --- 2. Page and Plot Layout ---
      fig = plt.figure(figsize=(12, 18), constrained_layout=True)
      fig.suptitle(f'VCV Diagnostic Report: "{found_label.upper()}"',
                   fontsize=18, weight='bold')
      gs_main = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 2])

      # --- 3. Populate Plots ---
      # a) Waveform Plots (in a nested GridSpec)
      gs_wave = gs_main[0].subgridspec(2, 1, hspace=0.1)

      ax_wh = fig.add_subplot(gs_wave[0])
      time_ax_h = np.linspace(0., len(clean_signal) / fs, len(clean_signal))
      start_h, end_h, thresh_h = _get_activity_boundaries(
          clean_signal, threshold_db)
      ax_wh.plot(time_ax_h, clean_signal, label='Waveform', color='blue')
      ax_wh.axhline(thresh_h, ls=':', color='gray',
                    label=f'{threshold_db} dB Thresh')
      ax_wh.axhline(-thresh_h, ls=':', color='gray')
      ax_wh.axvspan(0, start_h / fs, color='red', alpha=0.2, label='Silence')
      ax_wh.axvspan(end_h / fs, time_ax_h[-1], color='red', alpha=0.2)
      ax_wh.set_title(f'Human: {base_name}', fontsize=10)
      ax_wh.grid(True, which='both', linestyle='--', linewidth=0.5)
      ax_wh.legend(loc='upper right', fontsize='small')
      plt.setp(ax_wh.get_xticklabels(), visible=False)

      ax_ws = fig.add_subplot(gs_wave[1], sharex=ax_wh, sharey=ax_wh)
      time_ax_s = np.linspace(0., len(synth_signal) / fs, len(synth_signal))
      start_s, end_s, thresh_s = _get_activity_boundaries(
          synth_signal, threshold_db)
      ax_ws.plot(time_ax_s, synth_signal, label='Waveform', color='green')
      ax_ws.axhline(thresh_s, ls=':', color='gray',
                    label=f'{threshold_db} dB Thresh')
      ax_ws.axhline(-thresh_s, ls=':', color='gray')
      ax_ws.axvspan(0, start_s / fs, color='red', alpha=0.2, label='Silence')
      ax_ws.axvspan(end_s / fs, time_ax_s[-1], color='red', alpha=0.2)
      ax_ws.set_title(
          f'Synthetic (Calibrated): {os.path.basename(synth_path)}',
          fontsize=10)
      ax_ws.set_xlabel('Time (s)')
      ax_ws.grid(True, which='both', linestyle='--', linewidth=0.5)
      ax_ws.legend(loc='upper right', fontsize='small')


      # b) PSD Plot
      ax_psd = fig.add_subplot(gs_main[1])
      freqs_h, psd_h = signal.welch(clean_signal, fs, nperseg=1024)
      freqs_s, psd_s = signal.welch(synth_signal, fs, nperseg=1024)
      ax_psd.semilogx(freqs_h, 10 * np.log10(psd_h), label='Human')
      ax_psd.semilogx(freqs_s, 10 * np.log10(psd_s),
                      label='Synthetic (Calibrated)')
      ax_psd.set_title('Power Spectral Density Comparison')
      ax_psd.set_xlabel('Frequency (Hz)')
      ax_psd.set_ylabel('Power/Frequency (dB/Hz)')
      ax_psd.grid(True, which='both', linestyle='--', linewidth=0.5)
      ax_psd.legend()

      # c) Mel Spectrogram Plots
      gs_mel = gs_main[2].subgridspec(2, 2, wspace=0.05, hspace=0.05)
      all_mels = [mel_h_db.flatten(), mel_h_noisy_db.flatten(),
                  mel_s_db.flatten(), mel_s_noisy_db.flatten()]
      vmin = np.percentile(np.concatenate(all_mels), 5)
      vmax = np.percentile(np.concatenate(all_mels), 99)

      ax_m_hh = fig.add_subplot(gs_mel[0, 0])
      img1 = librosa.display.specshow(mel_h_db, sr=fs_h, x_axis='time',
                                      y_axis='mel', ax=ax_m_hh, vmin=vmin,
                                      vmax=vmax)
      ax_m_hh.set_title('Human (Clean)')
      ax_m_hh.set_ylabel('Frequency [Hz]')

      ax_m_hn = fig.add_subplot(gs_mel[0, 1], sharey=ax_m_hh)
      librosa.display.specshow(mel_h_noisy_db, sr=fs_h, x_axis='time',
                               y_axis='mel', ax=ax_m_hn, vmin=vmin,
                               vmax=vmax)
      ax_m_hn.set_title(f'Human (Noisy, {noisy_snr_db} dB SNR)')

      # The two top plots share an x-axis with the bottom plots, so hide labels.
      plt.setp(ax_m_hh.get_xticklabels(), visible=False)
      ax_m_hh.set_xlabel('')
      plt.setp(ax_m_hn.get_xticklabels(), visible=False)
      ax_m_hn.set_xlabel('')

      ax_m_sh = fig.add_subplot(gs_mel[1, 0], sharex=ax_m_hh)
      librosa.display.specshow(mel_s_db, sr=fs_s, x_axis='time', y_axis='mel',
                               ax=ax_m_sh, vmin=vmin, vmax=vmax)
      ax_m_sh.set_title('Synthetic (Clean)')
      ax_m_sh.set_xlabel('Time [sec]')
      ax_m_sh.set_ylabel('Frequency [Hz]')

      ax_m_sn = fig.add_subplot(gs_mel[1, 1], sharey=ax_m_sh, sharex=ax_m_hn)
      librosa.display.specshow(mel_s_noisy_db, sr=fs_s, x_axis='time',
                               y_axis='mel', ax=ax_m_sn, vmin=vmin,
                               vmax=vmax)
      ax_m_sn.set_title(f'Synthetic (Noisy, {noisy_snr_db} dB SNR)')
      ax_m_sn.set_xlabel('Time [sec]')

      fig.colorbar(img1, ax=[ax_m_hh, ax_m_hn, ax_m_sh, ax_m_sn],
                   format='%+2.0f dB', label='Intensity [dB]', aspect=40)

      # Save the completed page to the PDF file
      pdf.savefig(fig)
      plt.close(fig)

    print(f'\nReport generation complete. {len(all_human_files)} pages saved.')


def get_wav_rms(file_path: str) -> float:
  """Reads a WAV file and calculates its RMS value in float format."""
  try:
    signal_float, _, _ = common.read_wav_as_float(file_path)
    # If the signal is stereo, convert it to mono by averaging channels.
    if signal_float.ndim == 2:
      signal_float = np.mean(signal_float, axis=1)
    return common.get_active_signal_rms(signal_float)
  except (FileNotFoundError, TypeError, ValueError):
    print(f'Warning: Could not process file {file_path}, skipping.')
    return 0.0


def compare_vcv_loudness():
  """Compares the average loudness of VCVs in clean and synthetic dirs."""
  stimuli_dir = 'stimuli'
  clean_dir = os.path.join(stimuli_dir, 'clean')
  synthetic_dir = os.path.join(stimuli_dir, 'synthetic')

  clean_files = glob.glob(os.path.join(clean_dir, '*.wav'))
  synthetic_files = glob.glob(os.path.join(synthetic_dir, '*.wav'))

  if not clean_files or not synthetic_files:
    print('Warning: Could not find WAV files in one or both directories.')
    print(f'Clean files found: {len(clean_files)}')
    print(f'Synthetic files found: {len(synthetic_files)}')
    return

  clean_rms = [get_wav_rms(f) for f in clean_files]
  synthetic_rms = [get_wav_rms(f) for f in synthetic_files]

  avg_clean_rms = np.mean(clean_rms)
  avg_synthetic_rms = np.mean(synthetic_rms)

  # Use a small epsilon to avoid log(0) for silent files.
  epsilon = 1e-12
  avg_clean_db = 20 * np.log10(avg_clean_rms + epsilon)
  avg_synthetic_db = 20 * np.log10(avg_synthetic_rms + epsilon)

  db_difference = avg_clean_db - avg_synthetic_db

  print('Loudness Comparison of VCV Stimuli')
  print('-' * 40)
  print(f'Found {len(clean_files)} files in "{clean_dir}"')
  print(f'Found {len(synthetic_files)} files in "{synthetic_dir}"')
  print('\nAverage Loudness (dBFS):')
  print(f'  Human VCVs:     {avg_clean_db:.2f} dBFS')
  print(f'  Synthetic VCVs: {avg_synthetic_db:.2f} dBFS')
  print(f'\nAverage Difference: {abs(db_difference):.2f} dB')

  if db_difference > 0.1:
    print('Human VCVs are on average louder.')
  elif db_difference < -0.1:
    print('Synthetic VCVs are on average louder.')
  else:
    print('The average loudness is very similar.')


def plot_tones(directory: str):
  """
  Plots the waveforms of all WAV files in a specified directory, each on a
  separate plot.

  Args:
    directory: The path to the directory containing the WAV files.
  """
  wav_files = glob.glob(os.path.join(directory, '*.wav'))
  if not wav_files:
    print(f'Warning: Could not find any WAV files in "{directory}".')
    return

  for file_path in wav_files:
    plt.figure(figsize=(12, 8))
    signal_float, sample_rate, _ = common.read_wav_as_float(file_path)
    # If stereo, take the first channel.
    if signal_float.ndim == 2:
      signal_float = signal_float[:, 0]

    duration = len(signal_float) / sample_rate
    time = np.linspace(0.0, duration, len(signal_float))

    filename = os.path.basename(file_path)
    min_val = np.min(signal_float)
    max_val = np.max(signal_float)
    label = f'{filename}\nMin: {min_val:.4f}, Max: {max_val:.4f}'
    plt.plot(time, signal_float, label=label, alpha=0.7)
    plt.title(f'Waveform of "{filename}"')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


def equalize_synthetic_vcvs():
  """
  Designs and applies an EQ to synthetic VCVs to match the spectrum of
  human VCVs. This is an offline process that generates a new set of
  equalized WAV files.
  """
  # --- 1. Setup Paths ---
  stimuli_dir = 'stimuli'
  clean_dir = os.path.join(stimuli_dir, 'clean')
  synthetic_dir = os.path.join(stimuli_dir, 'synthetic')
  output_dir = os.path.join(stimuli_dir, 'synthetic_eq')
  os.makedirs(output_dir, exist_ok=True)

  print('Starting VCV equalization process...')
  print(f'Human stimuli directory: "{clean_dir}"')
  print(f'Synthetic stimuli directory: "{synthetic_dir}"')
  print(f'Output directory for EQ\'d files: "{output_dir}"')

  # --- 2. Calculate Average Spectra ---
  def _get_average_psd(
      directory: str,
  ) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Calculates the average Power Spectral Density from all WAVs in a dir."""
    all_psds = []
    freqs = None
    sample_rate_g = -1

    wav_files = sorted(glob.glob(os.path.join(directory, '*.wav')))
    if not wav_files:
      print(f'Warning: No WAV files found in "{directory}".')
      return None, None, None

    print(f'Analyzing {len(wav_files)} files in "{directory}"...')
    for file_path in wav_files:
      signal_float, sample_rate, _ = common.read_wav_as_float(file_path)
      if sample_rate_g == -1:
        sample_rate_g = sample_rate
      elif sample_rate_g != sample_rate:
        print(f'Warning: Mismatched sample rates in {file_path}. Skipping.')
        continue

      # Isolate active part of the signal to avoid silence biasing the PSD.
      start_idx, end_idx, _ = _get_activity_boundaries(
          signal_float, common.DEFAULT_TRIM_DB_THRESHOLD
      )
      active_signal = signal_float[start_idx:end_idx]

      if active_signal.size == 0:
        continue

      f, psd = signal.welch(active_signal, fs=sample_rate, nperseg=2048)
      if freqs is None:
        freqs = f
      all_psds.append(psd)

    if not all_psds:
      print('Could not calculate any PSDs.')
      return None, None, None

    # Average the PSDs.
    avg_psd = np.mean(all_psds, axis=0)
    return freqs, avg_psd, sample_rate_g

  freqs, avg_psd_human, sample_rate = _get_average_psd(clean_dir)
  _, avg_psd_synth, _ = _get_average_psd(synthetic_dir)

  if freqs is None or avg_psd_human is None or avg_psd_synth is None:
    print('Error: Could not compute average PSDs. Aborting.')
    return

  # --- 3. Determine and Smooth the EQ Curve ---
  # Use a small epsilon to avoid log(0) errors.
  epsilon = 1e-12
  eq_db_raw = 10 * np.log10(avg_psd_human + epsilon) - 10 * np.log10(
      avg_psd_synth + epsilon
  )

  # Smooth the curve to prevent overfitting to noise.
  # Window length must be odd and less than the number of points.
  savgol_win = min(101, len(freqs) - 1 if len(freqs) % 2 == 0 else len(freqs))
  if savgol_win % 2 == 0:
    savgol_win -=1 # Ensure it is odd
  eq_db_smoothed = signal.savgol_filter(
      eq_db_raw, window_length=savgol_win, polyorder=3
  )

  # --- 4. Plot the EQ Curve for Verification ---
  plt.figure(figsize=(12, 7))
  plt.semilogx(
      freqs,
      eq_db_raw,
      label='Raw EQ Curve',
      color='gray',
      alpha=0.5,
      linestyle=':',
  )
  plt.semilogx(freqs, eq_db_smoothed, label='Smoothed EQ Curve', color='blue')
  plt.title('Required EQ to Match Synthetic VCVs to Human VCVs')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Gain (dB)')
  plt.grid(True, which='both', linestyle='--', linewidth=0.5)
  plt.axhline(0, color='black', linewidth=1, linestyle='--')
  plt.legend()
  plt.show()

  # --- 5. Design FIR Filter and Apply to Synthetic VCVs ---
  # Convert smoothed dB curve to linear gain for the filter design.
  gain_linear = 10.0 ** (eq_db_smoothed / 20.0)

  # firwin2 requires frequencies to be normalized between 0 and 1 (Nyquist).
  nyquist = sample_rate / 2.0
  normalized_freqs = freqs / nyquist

  # Design the filter. A longer filter (more taps) gives better frequency
  # resolution but has a longer impulse response. 511 is a good starting point.
  num_taps = 511
  try:
    fir_taps = signal.firwin2(
        num_taps, normalized_freqs, gain_linear, window='hann'
    )
  except ValueError as e:
    print(f'Filter design failed: {e}')
    print('This can happen if frequency points are not unique or monotonic.')
    return

  # Apply the filter to each synthetic file.
  synthetic_files = sorted(glob.glob(os.path.join(synthetic_dir, '*.wav')))
  print(f'Designing filter and applying to {len(synthetic_files)} files...')

  for file_path in synthetic_files:
    signal_float, sample_rate, _ = common.read_wav_as_float(file_path)
    # Apply the filter.
    eq_signal = signal.lfilter(fir_taps, 1.0, signal_float)
    # Normalize to prevent clipping.
    eq_signal /= np.max(np.abs(eq_signal))
    # Convert back to 16-bit integer for saving.
    eq_signal_int16 = np.int16(eq_signal * 32767)
    # Save the new file.
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, base_name)
    write(output_path, sample_rate, eq_signal_int16)

  print('\nProcessing complete.')
  print(f'Equalized files have been saved to: "{output_dir}"')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run diagnostic scripts for the codebase.'
  )
  parser.add_argument(
      '--task',
      type=str,
      choices=['plot_noise', 'compare_vcvs', 'plot_tones', 'visualize_pairs',
               'create_report', 'equalize_vcvs'],
      default='plot_noise',
      help='Select the diagnostic task to run.'
  )
  parser.add_argument(
      '--dir',
      type=str,
      default='tone_generator_output',
      help='Specify the directory for tasks that need it (e.g., plot_tones).',
  )
  parser.add_argument(
      '--output-file',
      type=str,
      default='vcv_diagnostics_report.pdf',
      help='Specify the output file path for the report.',
  )
  args = parser.parse_args()

  if args.task == 'plot_noise':
    print('Running: Analyze and plot noise spectrums...')
    analyze_and_plot_noise_spectrums()
  elif args.task == 'compare_vcvs':
    print('Running: Compare VCV loudness...')
    compare_vcv_loudness()
  elif args.task == 'plot_tones':
    print(f'Running: Plotting tones from "{args.dir}"...')
    plot_tones(args.dir)
  elif args.task == 'visualize_pairs':
    print('Running: Visualize VCV pairs with activity detection...')
    visualize_vcv_pairs()
  elif args.task == 'create_report':
    create_vcv_report(output_file=args.output_file)
  elif args.task == 'equalize_vcvs':
    print('Running: Equalize synthetic VCVs to match human VCVs...')
    equalize_synthetic_vcvs()
