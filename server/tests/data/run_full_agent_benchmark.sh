#!/bin/bash
# Comprehensive Agent Benchmark Runner
# Runs benchmarks for all agents with multiple model variants

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Activate virtual environment
source venv/bin/activate

# Models by tier
FAST_MODELS="gemma3:4b,qwen3:8b,ministral-3:3b,llama3.2:3b"
MEDIUM_MODELS="qwen3:8b,qwen3:14b,gemma3:12b,mistral:7b"
THINKING_MODELS="deepseek-r1:8b,deepseek-r1:14b,cogito:8b,qwq:32b"

# Parse arguments
TIER="${1:-fast}"
MAX_CONTEXTS="${2:-3}"
AGENT="${3:-all}"

case $TIER in
    fast)
        MODELS="$FAST_MODELS"
        ;;
    medium)
        MODELS="$FAST_MODELS,$MEDIUM_MODELS"
        ;;
    full)
        MODELS="$FAST_MODELS,$MEDIUM_MODELS,$THINKING_MODELS"
        ;;
    thinking)
        MODELS="$THINKING_MODELS"
        ;;
    *)
        MODELS="$TIER"  # Custom model list
        ;;
esac

echo "========================================"
echo "Agent Benchmark Suite"
echo "========================================"
echo "Tier: $TIER"
echo "Models: $MODELS"
echo "Max contexts: $MAX_CONTEXTS"
echo "Agent: $AGENT"
echo "========================================"

if [ "$AGENT" = "all" ]; then
    AGENTS="analyzer crag_evaluator verifier self_reflection url_relevance_filter"
else
    AGENTS="$AGENT"
fi

LOG_DIR="/tmp/agent_benchmarks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

for agent in $AGENTS; do
    echo ""
    echo "Starting benchmark: $agent"
    python tests/data/agent_benchmarks.py \
        --agent "$agent" \
        --models "$MODELS" \
        --max-contexts "$MAX_CONTEXTS" \
        2>&1 | tee "$LOG_DIR/${agent}.log"
done

echo ""
echo "========================================"
echo "All benchmarks complete!"
echo "Logs saved to: $LOG_DIR"
echo "========================================"

# Show final rankings
echo ""
python tests/data/agent_benchmarks.py --rankings
