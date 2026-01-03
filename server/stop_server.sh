#!/bin/bash
# Stop memOS Server and Docker Services
# Usage: ./stop_server.sh [--force] [--docker] [--all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCE=false
PORT=8001
STOP_DOCKER=false
STOP_ALL=false

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
        --docker)
            STOP_DOCKER=true
            shift
            ;;
        --all)
            STOP_ALL=true
            STOP_DOCKER=true
            shift
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

# Stop Docker services if requested
if [ "$STOP_DOCKER" = true ]; then
    echo ""
    echo "🐳 Stopping Docker services..."

    if [ "$STOP_ALL" = true ]; then
        # Stop all services including data stores
        docker compose --profile docling down 2>/dev/null
        echo "   ✅ All Docker services stopped"
    else
        # Stop only Docling (keep postgres/redis running for other uses)
        docker compose --profile docling stop docling 2>/dev/null
        echo "   ✅ Docling stopped"
        echo "   ℹ️  PostgreSQL and Redis still running (use --all to stop them)"
    fi
fi
