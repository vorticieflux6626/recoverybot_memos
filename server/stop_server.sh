#!/bin/bash
# Stop memOS Server
# Usage: ./stop_server.sh [--force]

FORCE=false
PORT=8001

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "🛑 Stopping memOS Server..."

# Find server processes
PIDS=$(pgrep -f "uvicorn main:app.*--port $PORT" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "   No memOS server running on port $PORT"
    # Clean up stale PID file
    rm -f /tmp/memos_server.pid
    exit 0
fi

# Kill processes
for PID in $PIDS; do
    if [ "$FORCE" = true ]; then
        kill -9 $PID 2>/dev/null
    else
        kill $PID 2>/dev/null
    fi
    echo "   Stopped process: $PID"
done

# Wait for processes to terminate
sleep 2

# Verify stopped
REMAINING=$(pgrep -f "uvicorn main:app.*--port $PORT" 2>/dev/null)
if [ -z "$REMAINING" ]; then
    echo "✅ memOS Server stopped"
    rm -f /tmp/memos_server.pid
else
    echo "⚠️  Some processes still running. Use --force to kill them."
    exit 1
fi
