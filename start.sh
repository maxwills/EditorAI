#!/usr/bin/env bash
# Usage: ./start.sh
# Stop with: ./stop.sh  (or close this terminal — Ctrl+C is unreliable on Windows/Git Bash)
set -e

PID_FILE="$(dirname "$0")/.uvicorn.pid"

source "$(dirname "$0")/venv/Scripts/activate"
cd "$(dirname "$0")/cad_prediction_service"

# Write PID so stop.sh can kill it cleanly.
uvicorn app.main:app --reload --reload-dir app --port 8001 &
echo $! > "$PID_FILE"
wait
