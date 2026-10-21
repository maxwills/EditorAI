#!/usr/bin/env bash
# Usage: ./start.sh
# Stop with: ./stop.sh  (or close this terminal — Ctrl+C is unreliable on Windows/Git Bash)
set -e

PID_FILE="$(dirname "$0")/.uvicorn.pid"

source "$(dirname "$0")/venv/Scripts/activate"
cd "$(dirname "$0")/cad_prediction_service"

if python -c "import socket,sys; s=socket.socket(); s.settimeout(.2); sys.exit(s.connect_ex(('127.0.0.1',8001)) != 0)"; then
    echo "Port 8001 is already in use. Run ./stop.sh before starting another server."
    exit 1
fi

# Write PID so stop.sh can kill it cleanly.
uvicorn app.main:app --reload --reload-dir app --port 8001 &
echo $! > "$PID_FILE"
wait
