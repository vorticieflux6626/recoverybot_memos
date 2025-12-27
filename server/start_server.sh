#!/bin/bash
# Start memOS Server
# Usage: ./start_server.sh [--fg] [--debug]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
FOREGROUND=false
DEBUG=""
PORT=8001
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --fg|--foreground)
            FOREGROUND=true
            shift
            ;;
        --debug)
            DEBUG="--log-level debug"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fg] [--debug] [--port PORT]"
            exit 1
            ;;
    esac
done

# Check if already running
if pgrep -f "uvicorn main:app.*--port $PORT" > /dev/null 2>&1; then
    echo "⚠️  memOS server already running on port $PORT"
    echo "   Use ./stop_server.sh to stop it first"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found at venv/"
    echo "   Run: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Log file
LOG_FILE="/tmp/memos_server.log"

echo "🚀 Starting memOS Server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log:  $LOG_FILE"

if [ "$FOREGROUND" = true ]; then
    echo "   Mode: Foreground (Ctrl+C to stop)"
    echo ""
    python -m uvicorn main:app --host "$HOST" --port "$PORT" --reload $DEBUG
else
    # Start in background
    nohup python -m uvicorn main:app --host "$HOST" --port "$PORT" --reload $DEBUG > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > /tmp/memos_server.pid

    # Wait for startup
    sleep 3

    # Check if running
    if kill -0 $PID 2>/dev/null; then
        echo "✅ memOS Server started (PID: $PID)"
        echo ""
        # Quick health check
        HEALTH=$(curl -s http://localhost:$PORT/health 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)
        if [ "$HEALTH" = "healthy" ]; then
            echo "   Health: ✅ $HEALTH"
        else
            echo "   Health: ⚠️  Checking... (server may still be initializing)"
        fi
        echo ""
        echo "   API Docs: http://localhost:$PORT/docs"
        echo "   Health:   http://localhost:$PORT/health"
    else
        echo "❌ Server failed to start. Check log: $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
fi
