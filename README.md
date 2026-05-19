# AFHI Web Demos
Python demos of the hearing tests developed by the AFHI.

A live instance is hosted at [afhi-hearing-tests.streamlit.app](https://afhi-hearing-tests.streamlit.app/).

## Disclaimers
This application contains hearing tests currently being evaluated for equivalence to clinical audiometry. Please read the **Medical & Hardware Disclaimer** in the [LICENSE](LICENSE) file.

## Recommended installation procedure
1. Clone this repository: `git clone <repository-url>`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment:
    * `source .venv/bin/activate` (Linux/macOS)
    * `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
To run the Streamlit app locally:
`streamlit run web_app.py`

## Feature Control for Different Audiences

Certain features (like the "Email results" option) may be enabled or disabled 
depending on the target audience or deployment environment (e.g., for UX 
research vs. clinical research).

This is controlled using the `APP_TARGET_AUDIENCE` configuration setting, which 
can be set either as a **Streamlit Secret** (for cloud deployments) or an 
**environment variable** (for local testing).

The code reads this setting and adjusts the displayed features accordingly. 
Supported values are:

*   `"UX"`: Enables features specific to UX researchers (currently includes the email results form).
*   `"NAL"`: Disables features not required for clinical researchers (currently hides the email results form).
*   `"ALL"` (Default): If the variable is not set, or set to `"ALL"`, all features will be enabled.

This allows different Streamlit Cloud apps to present slightly different user 
experiences based solely on their configured Secrets, without requiring 
divergent code branches.

## Contributing

### Development workflow
1. Create a feature branch from `main`.
2. Increment the version number in `web_app.py`.
3. Run all tests and ensure they pass before opening a pull request.

### Testing
Discover and run all unit tests:\
`pytest`

Aim for 100% coverage with unit tests for new, non-UI code. Use\
`coverage run -m pytest`\
`coverage report -m --omit='*_test.py'`

### Python style
Use pylint:\
`pylint *.py`\
Code should be rated 10/10. Pylint uses the pylintrc file from
[here](https://google.github.io/styleguide/pylintrc) for Google style. This file is included in this repository.\
If there are many pylint issues (e.g., a lot of new code where the indentation is wrong), consider using pyink to first
autoformat with the following options:\
`pyink --pyink-indentation 2  --pyink-use-majority-quotes  <filename>.py'`\
Do not run pyink on code that already passes pylint (it sometimes produces a worse result than human-edited code).

## License
Please see the [LICENSE](LICENSE) file for details on the licensing of this project.

Note: The VCV audio stimuli files in this repository are sourced from the external [qVCV project](https://github.com/BoysTownOrg/qVCV).
