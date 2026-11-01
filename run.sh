#!/bin/sh
# Start Forecast Studio from any working directory.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v node >/dev/null 2>&1; then
    node_bin=$(command -v node)
else
    node_bin="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
fi
if [ ! -x "$node_bin" ]; then
    echo "Node.js is required. Install Node.js, then run ./run.sh again." >&2
    exit 1
fi
python_bin="${ATSA_PYTHON:-$project_dir/.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
    echo "Python environment missing: $python_bin" >&2
    echo "Create the project's .venv and install requirements.txt first." >&2
    exit 1
fi
if [ ! -f "$project_dir/frontend/dist/index.html" ]; then
    echo "Frontend build missing. Run pnpm build inside frontend first." >&2
    exit 1
fi
cd "$project_dir/frontend"
exec "$node_bin" launch.mjs
