"""Functions shared across multiple demos in the app."""

import base64
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import os
import smtplib
import subprocess
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import streamlit as st

# Email account for sending results (must have app-specific password generated).
# Sender email is read from Streamlit secrets (cloud) or an environment
# variable (local) to avoid hardcoding it in public source code.
# Set SENDER_EMAIL in .streamlit/secrets.toml or as an env var.
# Version string for the app GUI and log files.
DEMO_UPDATED = 'Version 7.1.2, 19 June 2026'
# Determine the preferred stimuli directory.
PREFERRED_STIMULI_DIR = ('local_stimuli' if
                         os.path.isdir('local_stimuli') else 'stimuli')
MAX_16_BIT_INT = 32767
MAX_32_BIT_INT = 2147483647
SETTINGS_TITLE = 'Settings'
SETTINGS_STRING = (
  'If you are participating in a study, the following options are locked. '
  'Otherwise, you can adjust them to change the test behavior.'
)
VOL_SETTING_URL = ('https://github.com/Australian-Future-Hearing-Initiative'
                   '/afhi-hearing-tests-public/blob/main/docs'
                   '/volume_calibration.md')
MERGE_LR_HELP = ('Combine left and right channels in hearing tests. This '
                 'will halve the test time, but affects the clinical '
                 'relevance of the results.')
DEFAULT_INITIAL_EAR = 'left'  # Default starting ear for all hearing tests.
DEFAULT_TRIM_DB_THRESHOLD = -40.0 # Threshold for silence detection.


# PTA / PIP test shared configuration constants.
STANDARD_FREQS_HZ = [1000, 2000, 3000, 4000, 6000, 8000, 500, 250]
PTA_START_LEVEL_DB_HL = 50
PTA_MIN_LEVEL_DB_HL = -5
PTA_MAX_LEVEL_DB_HL = 70
PTA_MAX_TRIALS_PER_EAR = 40  # Stopping condition for adaptive logic.

# Supported headphone devices for calibrated hearing tests.
DEVICE_PIXEL_BUDS = 'Google Pixel Buds Pro 2'
DEVICE_AIRPODS_PRO2 = 'Apple AirPods Pro 2'
DEVICE_OTHER = 'Other (Untested Calibration)'
SUPPORTED_DEVICES = [DEVICE_PIXEL_BUDS, DEVICE_AIRPODS_PRO2, DEVICE_OTHER]


@st.cache_data(ttl=86400) # Cache for 24 hours
def get_autoeq_index() -> list[dict]:
  """Downloads and parses the full AutoEq results index from GitHub.
  
  Returns a list of dicts, each with keys: 'name', 'path', 'source', 'rig'
  """
  import re
  import urllib.request
  url = "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/INDEX.md"
  headers = {
      "User-Agent": "HearingTestCalibrationAgent/1.0"
  }
  try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
      content = response.read().decode('utf-8')
      
    # Regex to parse the markdown index lines
    # Example: - [1MORE Piston Fit](./Rtings/HMS%20II.3%20in-ear/1MORE%20Piston%20Fit) by Rtings on HMS II.3
    pattern = re.compile(
        r'^-\s+\[(?P<name>[^\]]+)\]\(\./(?P<path>.*?)\)\s+by\s+(?P<source>.*?)(?:\s+on\s+(?P<rig>[^\n]+))?$',
        re.MULTILINE
    )
    
    entries = []
    for match in pattern.finditer(content):
      gd = match.groupdict()
      entries.append({
          "name": gd["name"],
          "path": gd["path"],
          "source": gd["source"],
          "rig": gd["rig"] if gd["rig"] else ""
      })
    return entries
  except Exception as e:
    print(f"Failed to fetch AutoEq index: {e}")
    return []


def get_target_audience() -> str:
  """Reads the target audience from secrets or env vars, defaulting to ALL."""
  audience = 'ALL' # Default.
  try:
    # Prefer Streamlit secrets if available.
    secret_value = st.secrets['APP_TARGET_AUDIENCE']
    if secret_value:
      audience = str(secret_value).upper()
      print(f'Target audience read from secrets: {audience}')
  except (AttributeError, KeyError, st.errors.StreamlitSecretNotFoundError):
    # Fallback to environment variable.
    env_value = os.environ.get('APP_TARGET_AUDIENCE')
    if env_value:
      audience = env_value.upper()
      print(f'Target audience read from env var: {audience}')
    else:
      # Default if not set anywhere.
      print('Target audience not set, defaulting to ALL.')
  # Validate and return.
  if audience not in ['UX', 'NAL', 'ALL']:
    print('Warning: Invalid APP_TARGET_AUDIENCE. Defaulting to ALL.')
    return 'ALL'
  return audience

def display_preparation():
  st.subheader('Prepare for the test')
  st.markdown(
      f"""
      * **Find a quiet environment:**  Background noise can interfere with the
      test and lead to inaccurate results.
      * **Use headphones:** Use headphones to ensure that the sound
      is directed to your left/right ears individually and is not affected by
      the surrounding environment.
      * **Calibrate volume (IMPORTANT):** Set your laptop volume to 50%
      ([instructions here]({VOL_SETTING_URL})). Do not
      change the volume during the test.
      """
  )

def autoplay_audio(file_path: str):
  """Plays an audio file without displaying an audio player.

  Args:
    file_path: The path to the audio file to play.

  Raises:
    ValueError: If the file extension is not WAV or MP3.
  """
  file_extension = file_path.split('.')[-1]
  if file_extension == 'wav':
    audio_type = 'wav'
  elif file_extension == 'mp3':
    audio_type = 'mpeg'
  else:
    raise ValueError('Only WAV and MP3 files are supported.')
  with open(file_path, 'rb') as f:
    data = f.read()
    b64 = base64.b64encode(data).decode()
    html = f"""
            <audio autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/{audio_type}">
            </audio>
            """
    with st.session_state.audio_container:
      st.components.v1.html(html, height=0)

def generate_zip_bytes(files: list[tuple[str, any]]) -> bytes:
  """Generates the bytes content of a zip file in memory.

  Args:
    files: List of tuples containing (filename_in_zip, content) pairs.
      Content can be a string (for text files), a matplotlib Figure
      (for plots), or bytes (for binary files such as WAV audio).

  Returns:
    Bytes representing the zip file content.
  """
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
    for filename, content in files:
      if isinstance(content, str):
        zip_file.writestr(filename, content)
      elif isinstance(content, bytes):
        zip_file.writestr(filename, content)
      elif isinstance(content, plt.Figure):
        # Save plot to a temporary png file within the zip context.
        # Use delete=True (default) as we only need it temporarily.
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_img_file:
          content.savefig(temp_img_file.name, dpi=300, bbox_inches='tight')
          # Add the saved image file to the zip archive using arcname.
          zip_file.write(temp_img_file.name, arcname=filename)
      else:
        print('Warning: Skipping unsupported content type '
              f'for {filename}: {type(content)}')
  zip_buffer.seek(0)
  zip_data = zip_buffer.getvalue() # getvalue() returns bytes.
  return zip_data

def send_results_email(
    recipient: str,
    subject: str,
    body_html: str,
    attachment_bytes: bytes,
    attachment_filename: str
) -> bool:
  """Sends an HTML email with zip attachment using the configured Gmail account.

  Args:
    recipient: The email address to send the results to.
    subject: The subject line for the email.
    body_html: The HTML content of the email body.
    attachment_bytes: The raw bytes of the file to attach (e.g., zip file).
    attachment_filename: The desired filename for the attachment.

  Returns:
    True if the email was sent successfully, False otherwise.
  """
  # Get sender email and password securely.
  # Prefer Streamlit secrets if available (for cloud), fall back to env var.
  try:
    sender_email = st.secrets['SENDER_EMAIL']
  except (AttributeError, KeyError):
    sender_email = os.environ.get('SENDER_EMAIL')

  if not sender_email:
    st.error('Sender email not configured. Cannot send email.')
    print('ERROR: SENDER_EMAIL secret/environment variable not set.')
    return False

  try:
    gmail_password = st.secrets['GMAIL_APP_PASSWORD']
  except (AttributeError, KeyError):
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD')

  if not gmail_password:
    st.error('Email password not configured. Cannot send email.')
    print('ERROR: GMAIL_APP_PASSWORD secret/environment variable not set.')
    return False

  # Create the container email message.
  message = MIMEMultipart()
  message['Subject'] = subject
  message['From'] = sender_email
  message['To'] = recipient
  # Attach the HTML body.
  message.attach(MIMEText(body_html, 'html'))
  # Create the attachment part.
  part = MIMEApplication(
      attachment_bytes,
      Name=attachment_filename
  )
  # Add header to make it downloadble.
  part['Content-Disposition'] = f'attachment; filename="{attachment_filename}"'
  message.attach(part)

  server = None
  try:
    print(f'Attempting to send email from {sender_email} to {recipient}...')
    # Connect to Gmail SMTP server using SSL
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo() # Identify ourselves using EHLO for the ESMTP session.
    print('Logging into Gmail...')
    server.login(sender_email, gmail_password)
    print('Login successful. Sending email...')
    server.sendmail(sender_email, recipient, message.as_string())
    print('Email sent successfully.')
    return True
  except smtplib.SMTPAuthenticationError:
    st.error('Email server authentication failed. Check App Password.')
    print('ERROR: SMTP Authentication Error. Check GMAIL_APP_PASSWORD.')
    return False
  except smtplib.SMTPException as e:
    st.error(f'An SMTP error occurred while sending the email: {e}')
    print(f'Warning: Failed to send email cleanly: {e}')
    return False
  except OSError as e:
    st.error(f'A network error occurred while sending the email: {e}')
    print(f'ERROR: Failed to send email due to network error: {e}')
    return False
  finally:
    if server:
      try:
        print('Closing SMTP server connection.')
        server.quit()
      except smtplib.SMTPException as e:
        # Non-critical error, just log it to console.
        print(f'Warning: Failed to close SMTP connection cleanly: {e}')

def display_email_results_form(
    test_name: str,
    files_for_zip: list[tuple[str, any]],
    zip_filename_prefix: str):
  """Displays a Streamlit form to email the test results.

  Args:
    test_name: The name of the test (e.g., 'Pure-Tone Audiometry').
    files_for_zip: The list of (filename, content) tuples needed to
      generate the zip file attachment. Content can be str or Figure.
    zip_filename_prefix: The prefix for the generated zip filename
      (e.g., 'pta_results').
  """
  # Check audience before displaying anything.
  audience = st.session_state.get('app_target_audience', 'ALL')
  if audience == 'NAL':
    print('Email form skipped for NAL audience.')
    return # Do not display the form for NAL.

  st.subheader('Email Results')
  with st.form(key=f'email_form_{zip_filename_prefix}'):
    recipient_email = st.text_input('Recipient Email Address:')
    participant_id = st.text_input('Participant ID:')
    submitted = st.form_submit_button('Send Email')

    if submitted:
      valid_input = True
      if not recipient_email or '@' not in recipient_email:
        st.warning('Please enter a valid recipient email address.')
        valid_input = False
      if not participant_id:
        st.warning('Please enter a Participant ID.')
        valid_input = False

      if valid_input:
        # Proceed only if inputs are valid.
        print(f'Generating zip file for {zip_filename_prefix}...')
        try:
          attachment_bytes = generate_zip_bytes(files_for_zip)
          print(f'Zip file generated ({len(attachment_bytes)} bytes).')
        except (OSError, zipfile.BadZipFile, TypeError) as e:
          st.error(f'Failed to generate results zip file: {e}')
          print(f'ERROR generating zip: {e}')
          return

        # Generate timestamped filename.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        attachment_filename = f'UTC{timestamp}_{zip_filename_prefix}.zip'

        # Construct subject.
        subject = f'{test_name} Results'
        if participant_id:
          subject += f' - Participant {participant_id}'

        # Construct HTML body.
        body_html = f"""
        <html><body>
        <p>Test results for the <b>{test_name}</b> are attached.</p>
        """
        if participant_id:
          body_html += f"<p>Participant ID: {participant_id}</p>"
        body_html += """
        <p>Please download and save the attached zip file.</p>
        </body></html>
        """

        print(f'Attempting to email results to {recipient_email}...')
        success = send_results_email(
            recipient=recipient_email,
            subject=subject,
            body_html=body_html,
            attachment_bytes=attachment_bytes,
            attachment_filename=attachment_filename
        )

        if success:
          st.success(f'Results successfully sent to {recipient_email}!')
          print('Email form submission successful.')
        else:
          print('Email form submission failed.')

def save_local_backup(zip_data: bytes, zip_filename: str):
  """Saves the provided zip data to a local backup directory.

  Creates the './local_results' directory if it doesn't exist.
  Only intended for use in local development/testing environments.

  Args:
    zip_data: The bytes content of the zip file.
    zip_filename: The desired filename for the backup.
  """
  backup_dir = './local_results'
  try:
    # Ensure the backup directory exists.
    os.makedirs(backup_dir, exist_ok=True)
    # Construct the full path.
    file_path = os.path.join(backup_dir, zip_filename)
    # Write the zip data to the file.
    with open(file_path, 'wb') as f:
      f.write(zip_data)
    print(f'Local backup saved successfully to: {file_path}')
  except (OSError, IOError) as e:
    print(f'Error saving local backup to {backup_dir}: {e}')

def get_macos_system_volume():
  """Gets the macOS system output volume using osascript.

  Returns:
      str: The volume percentage as a string (e.g., '75') if successful,
           otherwise returns the string 'unknown'.
  """
  unknown_volume = 'unknown'
  try:
    command = ['osascript', '-e', 'output volume of (get volume settings)']
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=1  # Wait for 1 second max.
    )
    volume_str = result.stdout.strip()
    if volume_str:
      # Validate it's an integer and clamp between 0-100
      volume_int = int(volume_str)
      volume_clamped = max(0, min(100, volume_int))
      return str(volume_clamped)
    else:
      return unknown_volume  # Command ran, but no output.
  # pylint: disable=broad-exception-caught
  except (FileNotFoundError, subprocess.CalledProcessError,
          subprocess.TimeoutExpired, ValueError, Exception):
    # Any errors, just return 'unknown'.
    return unknown_volume


def read_wav_as_float(file_path: str) -> tuple[np.ndarray, int, np.dtype]:
  """Reads a WAV file and returns its data as a normalized float array.

  Args:
    file_path: The path to the WAV file.

  Returns:
    A tuple of (float_audio_data, sample_rate, original_dtype).

  Raises:
    TypeError: If the audio data type is unsupported.
    FileNotFoundError: If the file_path does not exist.
  """
  fs, data = wavfile.read(file_path)
  original_dtype = data.dtype

  # If the signal is stereo, convert it to mono by averaging channels.
  if data.ndim == 2:
    data = np.mean(data, axis=1)

  if np.issubdtype(original_dtype, np.integer):
    max_val = np.iinfo(original_dtype).max
    data_float = data.astype(np.float32) / max_val
  elif np.issubdtype(original_dtype, np.floating):
    data_float = data.astype(np.float32)
  else:
    raise TypeError(f'Unsupported audio data type: {original_dtype}')

  return data_float, fs, original_dtype


def get_active_signal_rms(
    audio_data: np.ndarray, threshold_db: float = DEFAULT_TRIM_DB_THRESHOLD
) -> float:
  """Calculates the RMS of the active part of an audio signal.

  'Active' is defined as the portion of the signal between the first and last
  sample that exceeds a threshold defined relative to the peak amplitude.

  Args:
    audio_data: A NumPy array containing the audio signal. The function assumes
      the data is already normalized to a float format (e.g., [-1.0, 1.0]).
    threshold_db: The threshold in dBFS below the peak amplitude to consider a
      sample as 'active'.

  Returns:
    The RMS value of the active portion of the signal as a float.
  """
  # Ensure audio_data is in float format for calculations.
  audio_data = audio_data.astype(np.float64)

  peak_amplitude = np.max(np.abs(audio_data))
  if peak_amplitude == 0:
    return 0.0  # The signal is completely silent.

  # Convert the dB threshold to a linear amplitude value.
  threshold_linear = peak_amplitude * (10 ** (threshold_db / 20.0))

  # Find indices of all samples that are above the threshold.
  active_indices = np.where(np.abs(audio_data) >= threshold_linear)[0]

  if len(active_indices) == 0:
    return 0.0  # No samples were above the threshold.

  # Find the start and end of the active segment.
  start_index = active_indices[0]
  end_index = active_indices[-1]

  # Extract the active portion of the signal.
  active_segment = audio_data[start_index : end_index + 1]

  # Calculate and return the RMS of just the active segment.
  return np.sqrt(np.mean(active_segment**2))


def prepend_silence(
    audio_data: np.ndarray, sample_rate: int, duration_s: float
) -> np.ndarray:
  """Prepends a period of silence to a NumPy audio array.

  Args:
    audio_data: The NumPy array containing the audio data.
    sample_rate: The sample rate of the audio.
    duration_s: The duration of the silence to prepend, in seconds.

  Returns:
    A new NumPy array with the silence prepended.
  """
  silence_samples = int(sample_rate * duration_s)
  # Ensure silence has the same number of channels as the audio data.
  if audio_data.ndim == 2:
    num_channels = audio_data.shape[1]
    silence = np.zeros((silence_samples, num_channels), dtype=audio_data.dtype)
  else:
    silence = np.zeros(silence_samples, dtype=audio_data.dtype)

  return np.concatenate([silence, audio_data])


def get_scaled_vcv_data(
    file_path: str, target_db_spl: float, ref_db_spl: float
) -> tuple[np.ndarray, int]:
  """Reads and scales a VCV WAV file, returning the audio data.

  Args:
    file_path: Path to the WAV audio file.
    target_db_spl: The target sound pressure level in dB.
    ref_db_spl: The reference sound pressure level in dB.

  Returns:
    A tuple of (scaled_audio_data, sample_rate).

  Raises:
    ValueError: If file_path is not a .wav file.
    TypeError: If the audio data type is unsupported.
    FileNotFoundError: If the file_path does not exist.
  """
  if not file_path.lower().endswith('.wav'):
    raise ValueError('get_scaled_vcv_data expects a .wav file path.')

  try:
    delta_db = target_db_spl - ref_db_spl
    gain = 10 ** (delta_db / 20.0)

    data_float, fs, original_dtype = read_wav_as_float(file_path)

    scaled_float = data_float * gain

    if np.any(scaled_float > 1.0) or np.any(scaled_float < -1.0):
      print(f'WARNING: Clipping values in {os.path.basename(file_path)}')
      scaled_float = np.clip(scaled_float, -1.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
      max_val = np.iinfo(original_dtype).max
      scaled_final = (scaled_float * max_val).astype(original_dtype)
    else:
      scaled_final = scaled_float

    return scaled_final, fs
  except FileNotFoundError:
    st.error(f'Audio file not found: {file_path}')
    print(f'ERROR: File not found for scaling: {file_path}')
    raise
  except Exception as e:
    st.error(f'Error processing audio file {os.path.basename(file_path)}: {e}')
    print(f'ERROR: Failed to process/scale {file_path}: {e}')
    raise


def play_scaled_vcv_wav(file_path: str,
                        target_db_spl: float, ref_db_spl: float):
  """Plays a WAV file after scaling it based on VCV calibration constants.

  This is a wrapper around get_scaled_vcv_data that handles playback.

  Args:
    file_path: Path to the WAV audio file.
    target_db_spl: The target sound pressure level in dB.
    ref_db_spl: The reference sound pressure level in dB.
  """
  try:
    scaled_data, fs = get_scaled_vcv_data(file_path, target_db_spl, ref_db_spl)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
      wavfile.write(tmpf.name, fs, scaled_data)
      temp_file_path = tmpf.name
    print(f'  Playing scaled temp file: {os.path.basename(temp_file_path)}')
    autoplay_audio(temp_file_path)

  except (ValueError, TypeError, FileNotFoundError):
    # Errors are already logged by get_scaled_vcv_data, so just pass.
    pass
