#!/bin/bash
# Check memOS Server Status
# Usage: ./status_server.sh [--verbose]

VERBOSE=false
PORT=8001

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
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

echo "📊 memOS Server Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check process
PIDS=$(pgrep -f "uvicorn main:app.*--port $PORT" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "Status:  ❌ Not Running"
    echo ""
    exit 1
fi

echo "Status:  ✅ Running"
echo "PIDs:    $PIDS"

# Get health check
HEALTH_JSON=$(curl -s --max-time 5 http://localhost:$PORT/health 2>/dev/null)

if [ -n "$HEALTH_JSON" ]; then
    STATUS=$(echo "$HEALTH_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)
    VERSION=$(echo "$HEALTH_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('version','unknown'))" 2>/dev/null)
    ENV=$(echo "$HEALTH_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('environment','unknown'))" 2>/dev/null)

    echo "Health:  $STATUS"
    echo "Version: $VERSION"
    echo "Env:     $ENV"

    if [ "$VERBOSE" = true ]; then
        echo ""
        echo "Services:"
        echo "$HEALTH_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
services = d.get('services', {})
for name, info in services.items():
    status = info.get('status', 'unknown') if isinstance(info, dict) else info
    icon = '✅' if status in ['healthy', 'enabled'] else '⚠️' if status == 'degraded' else '❌'
    print(f'  {icon} {name}: {status}')
" 2>/dev/null
    fi
else
    echo "Health:  ⚠️  Unable to connect"
fi

echo ""
echo "Endpoints:"
echo "  API Docs:  http://localhost:$PORT/docs"
echo "  Health:    http://localhost:$PORT/health"
echo "  Models:    http://localhost:$PORT/api/v1/models/local"
