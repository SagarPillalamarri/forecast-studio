# Forecast Studio: setup on another computer

This source package includes the cleaned datasets. It excludes installed dependencies, Python virtual environments, compiled frontend files, raw downloads, and generated reports.

## Requirements

- Python 3.11 (the version used by the current project)
- Node.js and npm
- pnpm 11.25.0 (matches `frontend/package.json`)

## First-time setup (macOS / Linux)

Open a terminal in the extracted project directory or cloned repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm install -g pnpm@11.25.0
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
chmod +x run.sh
./run.sh
```

If your Python 3.11 executable is named `python3`, use that instead of `python3.11`.

Open http://localhost:3000. Leave the terminal running; Ctrl+C stops the app. On later starts, use just `./run.sh` from the project root. If an older app is already running, stop it in its terminal before restarting.

The Node server starts the Python service automatically. The UI uses port 3000 and the Python API uses port 8765. Both bind to localhost. Model-fitting results are cached in memory.

After changing React code, rebuild with `cd frontend`, `pnpm build`, then `cd ..`. After changing Python code, restart the app.

## Optional original Streamlit interface

```bash
source .venv/bin/activate
streamlit run app.py
```

## Checks

Run from the project root:

```bash
.venv/bin/python -m unittest discover -s tests
node frontend/test-capacity.mjs
```

See `frontend/README.md` for the analysis, future forecasting, capacity planning, and backtesting features. See `GITHUB_UPLOAD.md` for GitHub upload instructions.
