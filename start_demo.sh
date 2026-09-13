#!/usr/bin/env bash
# Marinedetect — macOS / Linux launcher.  chmod +x start_demo.sh, then ./start_demo.sh
set -u
cd "$(dirname "$0")"

echo
echo "  Marinedetect"
echo "  Side-scan sonar detection and shadow-geometry measurement"
echo "  ---------------------------------------------------------"
echo

PY=""
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  for c in python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -m pip --version >/dev/null 2>&1; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "  Python 3.10+ was not found. Install it, then run this again."
    exit 1
  fi
  echo "  First-time setup. Runs once, 5-15 minutes, mostly downloading PyTorch."
  echo
  "$PY" scripts/setup.py || { echo; echo "  Setup did not complete — see above."; exit 1; }
  PY=".venv/bin/python"
fi

# free port 8000 if something is on it
if command -v lsof >/dev/null 2>&1; then
  PID=$(lsof -ti :8000 2>/dev/null || true)
  [ -n "$PID" ] && { echo "  Port 8000 busy — stopping old process."; kill -9 $PID 2>/dev/null || true; sleep 1; }
fi

echo
echo "  Starting at http://127.0.0.1:8000 — leave this window open, Ctrl+C to stop."
echo
( sleep 6; (command -v open >/dev/null && open http://127.0.0.1:8000) || \
           (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000) ) >/dev/null 2>&1 &
exec "$PY" -m uvicorn backend.main:app --app-dir . --host 127.0.0.1 --port 8000
