""" Integration test for email functionality. """
import os
import sys
import argparse
import io
import zipfile

import common

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Test AFHI email sending functionality.'
  )
  parser.add_argument(
      'recipient',
      help='Email address to send the test email to.'
  )
  args = parser.parse_args()
  recipient_email = args.recipient
  # Check that credentials are available. The send_results_email function reads
  # from st.secrets first, then falls back to env vars. We mirror that here so
  # the pre-flight check passes if either source is configured.
  import streamlit as st  # pylint: disable=import-outside-toplevel

  def _get_secret(key):
    """Returns a secret from st.secrets or os.environ, or None."""
    try:
      return st.secrets[key]
    except (AttributeError, KeyError):
      return os.environ.get(key)

  if not _get_secret('SENDER_EMAIL'):
    print('ERROR: SENDER_EMAIL is not set.')
    print('Set it in .streamlit/secrets.toml or as an environment variable.')
    exit(1)
  else:
    print('SENDER_EMAIL found.')

  if not _get_secret('GMAIL_APP_PASSWORD'):
    print('ERROR: GMAIL_APP_PASSWORD is not set.')
    print('Set it in .streamlit/secrets.toml or as an environment variable.')
    exit(1)
  else:
    print('GMAIL_APP_PASSWORD found.')

  test_subject = 'AFHI Demo - Email Test'
  # Use body_html and provide dummy attachment details for the test call.
  test_html_body = (
      '<html><body><p>This is a test <b>HTML</b> email sent from '
      'the AFHI Demo integration test.</p></body></html>'
  )
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
    zip_file.writestr('dummy_file.txt', 'This is content inside the zip.')
  zip_buffer.seek(0)
  attachment_bytes = zip_buffer.getvalue()
  attachment_filename = 'dummy_results.zip'

  print('\nCalling common.send_results_email...')
  # Note: This test script doesn't use Streamlit, so st.error won't display,
  # but the print statements and return value will work.
  success = common.send_results_email(
      recipient=recipient_email,
      subject=test_subject,
      body_html=test_html_body,
      attachment_bytes=attachment_bytes,
      attachment_filename=attachment_filename
  )

  print('\n--- Test Complete ---')
  if success:
    print(f'Email sent successfully to {recipient_email}.')
    print('   Please check the inbox (and spam folder).')
  else:
    print(f'Failed to send email to {recipient_email}.')
    print('   Check console output above for error details.')
