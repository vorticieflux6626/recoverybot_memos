#!/bin/bash
# Check memOS Server and Docker Services Status
# Usage: ./status_server.sh [--verbose]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

echo "📊 memOS System Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Docker Services
echo ""
echo "🐳 Docker Services:"

# PostgreSQL
if docker ps --filter "name=memos-postgres" --filter "status=running" -q 2>/dev/null | grep -q .; then
    PG_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' memos-postgres 2>/dev/null || echo "unknown")
    if [ "$PG_HEALTH" = "healthy" ]; then
        echo "   ✅ PostgreSQL   :5432 (healthy)"
    else
        echo "   ⚠️  PostgreSQL   :5432 (status: $PG_HEALTH)"
    fi
else
    echo "   ❌ PostgreSQL   :5432 (not running)"
fi

# Redis
if docker ps --filter "name=memos-redis" --filter "status=running" -q 2>/dev/null | grep -q .; then
    REDIS_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' memos-redis 2>/dev/null || echo "unknown")
    if [ "$REDIS_HEALTH" = "healthy" ]; then
        echo "   ✅ Redis        :6379 (healthy)"
    else
        echo "   ⚠️  Redis        :6379 (status: $REDIS_HEALTH)"
    fi
else
    echo "   ❌ Redis        :6379 (not running)"
fi

# Docling
if docker ps --filter "name=memos-docling" --filter "status=running" -q 2>/dev/null | grep -q .; then
    DOCLING_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' memos-docling 2>/dev/null || echo "unknown")
    if [ "$DOCLING_HEALTH" = "healthy" ]; then
        echo "   ✅ Docling      :8003 (healthy)"
    elif [ "$DOCLING_HEALTH" = "starting" ]; then
        echo "   ⏳ Docling      :8003 (starting...)"
    else
        echo "   ⚠️  Docling      :8003 (status: $DOCLING_HEALTH)"
    fi
else
    echo "   ⚪ Docling      :8003 (not running - optional)"
fi

echo ""
echo "🖥️  memOS Server:"
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
