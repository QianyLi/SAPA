# Contributing

Thanks for your interest in PersonalWAB/SAPA. Please open an issue before
large changes so that implementation and benchmark changes can be discussed.

## Development setup

1. Create a Python 3.11 virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add API keys only for local runs.
4. Keep datasets, model checkpoints, indexes, and experiment outputs local;
   the repository's `.gitignore` excludes these artifacts by design.

## Pull requests

- Keep changes focused and document new command-line options.
- Run the relevant smoke test or script before opening a pull request.
- Do not include credentials, private user data, downloaded model weights, or
  generated result dumps.
