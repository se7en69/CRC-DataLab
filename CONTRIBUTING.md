# Contributing to CRC-DataLab

Thanks for your interest in contributing! This file explains how to get the project, run it locally, and submit fixes or new features.

## Code of conduct

Please follow respectful and collaborative behavior. Keep discussions constructive and evidence-based.

## How to contribute

1. Fork the repository and create a topic branch:

   - git clone https://github.com/your-username/CRC-DataLab.git
   - git checkout -b fix/feature-descriptive-name

2. Make changes in your branch. Keep commits small and focused. Include tests where possible.

3. Run the app locally to verify your change:

   - python -m venv .venv; .\.venv\Scripts\Activate.ps1
   - pip install -r requirements.txt
   - streamlit run CC-app.py

4. Push your branch and create a pull request explaining the change and any testing performed.

## Development guidelines

- Keep UI changes accessible and documented.
- Avoid large binary files in the repo; store only images necessary for the demo.
- If adding new dependencies, prefer well-established packages and update `requirements.txt`.
- For any code that processes data, prefer writing small, testable functions in separate modules.

## Testing

- Add pytest tests for any data-transformation functions.
- Run linters (black, ruff/flake8) before opening PRs.

## Review

- Pull requests will be reviewed for functionality, style, and tests.
- Maintain backward compatibility where possible.

## License and CLA

By contributing, you agree your contributions will be licensed under the project's Apache 2.0 license.
