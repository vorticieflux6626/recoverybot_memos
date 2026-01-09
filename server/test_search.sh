#!/bin/bash
#
# test_search.sh - Reliable search pipeline testing with intelligent timeouts
#
# Usage:
#   ./test_search.sh "query" [preset] [--json] [--verbose]
#
# Arguments:
#   query     - Search query (required)
#   preset    - MINIMAL|BALANCED|ENHANCED|RESEARCH|FULL (default: BALANCED)
#   --json    - Output raw JSON response
#   --verbose - Show progress during execution
#
# Examples:
#   ./test_search.sh "What is injection molding?"
#   ./test_search.sh "SRVO-063 encoder error" ENHANCED
#   ./test_search.sh "short shots PA66 mold temperature" RESEARCH --verbose
#
# Timeout calculation:
#   MINIMAL:   60s base + 10s per 10 words
#   BALANCED:  120s base + 15s per 10 words
#   ENHANCED:  180s base + 20s per 10 words
#   RESEARCH:  300s base + 30s per 10 words
#   FULL:      420s base + 40s per 10 words

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRESET="BALANCED"
JSON_OUTPUT=false
VERBOSE=false
MEMOSURL="${MEMOS_URL:-http://localhost:8001}"

# Parse arguments
QUERY=""
for arg in "$@"; do
    case $arg in
        --json)
            JSON_OUTPUT=true
            ;;
        --verbose|-v)
            VERBOSE=true
            ;;
        MINIMAL|BALANCED|ENHANCED|RESEARCH|FULL)
            PRESET="$arg"
            ;;
        *)
            if [ -z "$QUERY" ]; then
                QUERY="$arg"
            fi
            ;;
    esac
done

# Validate query
if [ -z "$QUERY" ]; then
    echo -e "${RED}Error: Query is required${NC}"
    echo "Usage: $0 \"query\" [preset] [--json] [--verbose]"
    exit 1
fi

# Count words in query
WORD_COUNT=$(echo "$QUERY" | wc -w)

# Calculate timeout based on preset and query length
calculate_timeout() {
    local preset=$1
    local words=$2
    local base_timeout=0
    local per_10_words=0

    case $preset in
        MINIMAL)
            base_timeout=60
            per_10_words=10
            ;;
        BALANCED)
            base_timeout=120
            per_10_words=15
            ;;
        ENHANCED)
            base_timeout=180
            per_10_words=20
            ;;
        RESEARCH)
            base_timeout=300
            per_10_words=30
            ;;
        FULL)
            base_timeout=420
            per_10_words=40
            ;;
    esac

    # Calculate: base + (words/10 * per_10_words)
    local extra=$((($words / 10 + 1) * $per_10_words))
    echo $(($base_timeout + $extra))
}

TIMEOUT=$(calculate_timeout "$PRESET" "$WORD_COUNT")

# Check if server is running
if ! curl -s --max-time 5 "${MEMOSURL}/api/v1/health/" > /dev/null 2>&1; then
    echo -e "${RED}Error: memOS server not responding at ${MEMOSURL}${NC}"
    echo "Start the server with: ./start_server.sh"
    exit 1
fi

# Display test info
if [ "$JSON_OUTPUT" = false ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}memOS Search Pipeline Test${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Query:   ${YELLOW}${QUERY:0:60}${NC}$([ ${#QUERY} -gt 60 ] && echo '...')"
    echo -e "Preset:  ${GREEN}${PRESET}${NC}"
    echo -e "Timeout: ${TIMEOUT}s (${WORD_COUNT} words)"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

# Create temp file for response
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

# Build JSON payload
PAYLOAD=$(cat << EOF
{
    "query": $(echo "$QUERY" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'),
    "preset": "$PRESET",
    "max_iterations": 2,
    "include_sources": true
}
EOF
)

# Run the search with progress indicator
START_TIME=$(date +%s)

if [ "$VERBOSE" = true ]; then
    echo -e "\n${YELLOW}Running search...${NC}"

    # Run curl in background
    curl -sL --max-time "$TIMEOUT" -X POST \
        "${MEMOSURL}/api/v1/search/universal" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" > "$TMPFILE" 2>&1 &
    CURL_PID=$!

    # Show progress
    while kill -0 $CURL_PID 2>/dev/null; do
        ELAPSED=$(($(date +%s) - $START_TIME))
        printf "\r  Elapsed: %ds / %ds timeout " $ELAPSED $TIMEOUT

        # Check server logs for progress (last non-healthcheck line)
        PROGRESS=$(tail -20 /tmp/memos.log 2>/dev/null | grep -v healthz | grep -E "PHASE|Scraping|Scraped|Graph:" | tail -1)
        if [ -n "$PROGRESS" ]; then
            # Extract just the key part
            SHORT_PROGRESS=$(echo "$PROGRESS" | sed 's/.*\] //' | cut -c1-50)
            printf "│ %s" "$SHORT_PROGRESS"
        fi

        sleep 2
    done
    printf "\n"

    wait $CURL_PID
    CURL_EXIT=$?
else
    # Silent mode
    curl -sL --max-time "$TIMEOUT" -X POST \
        "${MEMOSURL}/api/v1/search/universal" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" > "$TMPFILE" 2>&1
    CURL_EXIT=$?
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Check if request succeeded
if [ $CURL_EXIT -ne 0 ]; then
    echo -e "${RED}Error: Request failed (exit code $CURL_EXIT)${NC}"
    if [ $CURL_EXIT -eq 28 ]; then
        echo -e "${YELLOW}Request timed out after ${TIMEOUT}s. Try a simpler query or MINIMAL preset.${NC}"
    fi
    exit 1
fi

# Check if response is valid JSON
if ! python3 -c "import json; json.load(open('$TMPFILE'))" 2>/dev/null; then
    echo -e "${RED}Error: Invalid JSON response${NC}"
    head -c 500 "$TMPFILE"
    exit 1
fi

# Output results
if [ "$JSON_OUTPUT" = true ]; then
    cat "$TMPFILE"
else
    # Parse and display results
    python3 << PYEOF
import json
import sys

with open('$TMPFILE') as f:
    result = json.load(f)

success = result.get('success', False)
data = result.get('data', {})

if not success:
    errors = result.get('errors', [])
    print(f"\033[0;31m❌ Search failed\033[0m")
    for e in errors:
        print(f"   {e.get('code', 'ERR')}: {e.get('message', 'Unknown error')}")
    sys.exit(1)

confidence = data.get('confidence_score', 0)
features = data.get('features_used', [])
sources = data.get('sources', [])
synthesis = data.get('synthesis', data.get('answer', ''))

# Status
print(f"\n\033[0;32m✅ Search completed in ${DURATION}s\033[0m")
print(f"Confidence: {confidence:.1%}")

# Features
if features:
    tech_docs = '✅' if 'technical_docs' in features else '  '
    print(f"\nFeatures ({len(features)}): {tech_docs} technical_docs {'in use' if 'technical_docs' in features else ''}")
    print(f"   {', '.join(features[:8])}")
    if len(features) > 8:
        print(f"   ... and {len(features) - 8} more")

# Sources
if sources:
    print(f"\nTop Sources ({len(sources)} total):")
    for i, src in enumerate(sources[:5], 1):
        title = src.get('title', 'N/A')[:50]
        url = src.get('url', '')[:60]
        print(f"  {i}. {title}")
        print(f"     {url}")

# Synthesis
if synthesis:
    print(f"\n{'━' * 60}")
    print(f"SYNTHESIS ({len(synthesis)} chars):")
    print(f"{'━' * 60}")
    # Truncate if too long
    if len(synthesis) > 2000:
        print(synthesis[:2000])
        print(f"\n... [truncated, {len(synthesis) - 2000} more chars]")
    else:
        print(synthesis)

# IMM-specific check
imm_terms = ['shot size', 'adjust', 'mold temperature', 'short shot', 'defect', 'procedure']
found = [t for t in imm_terms if t.lower() in synthesis.lower()]
if found:
    print(f"\n\033[0;32m✅ IMM terms found in synthesis: {found}\033[0m")
PYEOF
fi

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
