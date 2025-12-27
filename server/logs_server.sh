#!/bin/bash
# View memOS Server Logs
# Usage: ./logs_server.sh [--follow] [--lines N] [--errors]

FOLLOW=false
LINES=50
ERRORS_ONLY=false
LOG_FILE="/tmp/memos_server.log"

while [[ $# -gt 0 ]]; do
    case $1 in
        --follow|-f)
            FOLLOW=true
            shift
            ;;
        --lines|-n)
            LINES="$2"
            shift 2
            ;;
        --errors|-e)
            ERRORS_ONLY=true
            shift
            ;;
        --clear)
            echo "Clearing log file..."
            > "$LOG_FILE"
            echo "✅ Log cleared"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Server may not have been started yet."
    exit 1
fi

echo "📋 memOS Server Logs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "File: $LOG_FILE"
echo ""

if [ "$ERRORS_ONLY" = true ]; then
    if [ "$FOLLOW" = true ]; then
        tail -f "$LOG_FILE" | grep --line-buffered -E "(ERROR|WARN|Exception|Traceback)"
    else
        grep -E "(ERROR|WARN|Exception|Traceback)" "$LOG_FILE" | tail -n "$LINES"
    fi
else
    if [ "$FOLLOW" = true ]; then
        tail -f "$LOG_FILE"
    else
        tail -n "$LINES" "$LOG_FILE"
    fi
fi
