#!/bin/bash
# Test memOS Server Endpoints
# Usage: ./test_endpoint.sh [endpoint] [method] [data]

PORT=8001
BASE_URL="http://localhost:$PORT"

# Default endpoint
ENDPOINT="${1:-/health}"
METHOD="${2:-GET}"
DATA="$3"

echo "🧪 Testing memOS Endpoint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "URL:    $BASE_URL$ENDPOINT"
echo "Method: $METHOD"
if [ -n "$DATA" ]; then
    echo "Data:   $DATA"
fi
echo ""

# Build curl command
CURL_CMD="curl -s -X $METHOD"

if [ -n "$DATA" ]; then
    CURL_CMD="$CURL_CMD -H 'Content-Type: application/json' -d '$DATA'"
fi

CURL_CMD="$CURL_CMD '$BASE_URL$ENDPOINT'"

# Execute and format output
RESPONSE=$(eval $CURL_CMD 2>&1)

if [ -n "$RESPONSE" ]; then
    # Try to pretty print JSON
    echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo "❌ No response (server may be down)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Quick examples
if [ "$ENDPOINT" = "/health" ] && [ "$METHOD" = "GET" ]; then
    echo ""
    echo "Other useful endpoints:"
    echo ""
    echo "  Models:"
    echo "    ./test_endpoint.sh /api/v1/models/local"
    echo "    ./test_endpoint.sh /api/v1/models/refresh POST"
    echo "    ./test_endpoint.sh /api/v1/models/specs"
    echo ""
    echo "  TTS Engines:"
    echo "    ./test_endpoint.sh /api/tts/engines"
    echo "    ./test_endpoint.sh /api/tts/emotivoice/emotions"
    echo "    ./test_endpoint.sh /api/tts/openvoice/styles"
    echo ""
    echo "  For full TTS testing, use: ./test_tts.sh"
fi
