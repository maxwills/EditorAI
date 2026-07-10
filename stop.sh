#!/usr/bin/env bash
# Kills the uvicorn server started by start.sh.
PID_FILE="$(dirname "$0")/.uvicorn.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found — server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping uvicorn (PID $PID)..."
    # SIGTERM works on Windows/Git Bash where SIGINT does not.
    kill -SIGTERM "$PID" 2>/dev/null || taskkill //F //PID "$PID" 2>/dev/null
    echo "Done."
else
    echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
