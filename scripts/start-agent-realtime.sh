#!/bin/bash
# Start the LiveKit voice assistant agent using vLLM-Omni's realtime API.
# Expects .env.local in agent/ directory.
set -e
cd "$(dirname "$0")/../agent"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -e .
fi

exec .venv/bin/python src/agent_realtime.py dev
