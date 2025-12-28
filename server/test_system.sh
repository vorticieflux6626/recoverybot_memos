#!/bin/bash
#
# memOS System Test Suite
# Comprehensive testing script for all agentic search components
#
# Usage:
#   ./test_system.sh              # Run all tests
#   ./test_system.sh quick        # Quick tests only (no LLM calls)
#   ./test_system.sh hybrid       # Test hybrid retrieval only
#   ./test_system.sh hyde         # Test HyDE only
#   ./test_system.sh ragas        # Test RAGAS only
#   ./test_system.sh api          # Test API endpoints only
#   ./test_system.sh search       # Test full agentic search
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API base URL
API_BASE="http://localhost:8001/api/v1/search"

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_test() {
    echo -e "${YELLOW}► Testing: $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ PASSED: $1${NC}"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}✗ FAILED: $1${NC}"
    ((FAILED++))
}

print_skip() {
    echo -e "${YELLOW}○ SKIPPED: $1${NC}"
    ((SKIPPED++))
}

check_server() {
    if curl -s "$API_BASE/../health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found at venv/bin/activate${NC}"
        exit 1
    fi
}

# ============================================================================
# Test: Module Imports
# ============================================================================
test_imports() {
    print_header "Module Import Tests"

    print_test "Core agentic imports"
    if python3 -c "from agentic import AgenticOrchestrator, __version__; print(f'Version: {__version__}')" 2>/dev/null; then
        print_pass "Core imports"
    else
        print_fail "Core imports"
    fi

    print_test "BGE-M3 Hybrid imports"
    if python3 -c "from agentic import BGEM3HybridRetriever, BM25Index, RetrievalMode" 2>/dev/null; then
        print_pass "BGE-M3 Hybrid imports"
    else
        print_fail "BGE-M3 Hybrid imports"
    fi

    print_test "HyDE imports"
    if python3 -c "from agentic import HyDEExpander, HyDEMode, HyDEResult" 2>/dev/null; then
        print_pass "HyDE imports"
    else
        print_fail "HyDE imports"
    fi

    print_test "RAGAS imports"
    if python3 -c "from agentic import RAGASEvaluator, RAGASResult, EvaluationMetric" 2>/dev/null; then
        print_pass "RAGAS imports"
    else
        print_fail "RAGAS imports"
    fi

    print_test "Mixed Precision imports"
    if python3 -c "from agentic import MixedPrecisionEmbeddingService, QWEN3_EMBEDDING_MODELS" 2>/dev/null; then
        print_pass "Mixed Precision imports"
    else
        print_fail "Mixed Precision imports"
    fi
}

# ============================================================================
# Test: BM25 Sparse Index (No LLM)
# ============================================================================
test_bm25() {
    print_header "BM25 Sparse Index Tests"

    print_test "BM25 indexing and search"
    python3 << 'EOF'
from agentic.bge_m3_hybrid import BM25Index

index = BM25Index()
docs = [
    ("doc1", "FANUC robot servo alarm SRVO-001"),
    ("doc2", "Raspberry Pi GPIO voltage"),
    ("doc3", "Robot motor overcurrent error"),
]
for doc_id, content in docs:
    index.add_document(doc_id, content)

results = index.search("robot motor", top_k=3)
assert len(results) >= 2, "Should find at least 2 results"
assert results[0][0] in ["doc1", "doc3"], f"Top result should be robot-related, got {results[0][0]}"
print(f"  Indexed: {index.get_stats()['documents']} docs, {index.get_stats()['vocabulary_size']} terms")
print(f"  Search 'robot motor': top={results[0][0]} (score={results[0][1]:.3f})")
EOF
    if [ $? -eq 0 ]; then
        print_pass "BM25 indexing and search"
    else
        print_fail "BM25 indexing and search"
    fi
}

# ============================================================================
# Test: BGE-M3 Hybrid Retrieval
# ============================================================================
test_hybrid() {
    print_header "BGE-M3 Hybrid Retrieval Tests"

    print_test "Dense embeddings via Ollama"
    python3 << 'EOF'
import asyncio
from agentic.bge_m3_hybrid import BGEM3HybridRetriever
import numpy as np

async def test():
    retriever = BGEM3HybridRetriever(db_path="/tmp/test_hybrid.db")
    emb = await retriever.get_embedding("Test embedding")
    await retriever.close()
    assert len(emb) == 1024, f"Expected 1024 dims, got {len(emb)}"
    assert np.linalg.norm(emb) > 0, "Embedding should not be zero"
    print(f"  Dimensions: {len(emb)}, Norm: {np.linalg.norm(emb):.4f}")

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Dense embeddings"
    else
        print_fail "Dense embeddings"
    fi

    print_test "Hybrid search (dense + sparse)"
    python3 << 'EOF'
import asyncio
import os
from agentic.bge_m3_hybrid import BGEM3HybridRetriever, RetrievalMode

async def test():
    db_path = "/tmp/test_hybrid_search.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    retriever = BGEM3HybridRetriever(db_path=db_path)

    # Index documents
    await retriever.add_document("d1", "FANUC robot servo alarm reset procedure")
    await retriever.add_document("d2", "Raspberry Pi overheating solution")
    await retriever.add_document("d3", "Robot motor troubleshooting guide")

    # Hybrid search
    results = await retriever.search("robot alarm", top_k=3, mode=RetrievalMode.HYBRID)
    await retriever.close()

    assert len(results) >= 2, "Should return results"
    print(f"  Indexed: 3 docs")
    print(f"  Query: 'robot alarm'")
    print(f"  Top result: {results[0].doc_id} (combined={results[0].combined_score:.4f})")

    os.remove(db_path)

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Hybrid search"
    else
        print_fail "Hybrid search"
    fi
}

# ============================================================================
# Test: HyDE Query Expansion
# ============================================================================
test_hyde() {
    print_header "HyDE Query Expansion Tests"

    print_test "Hypothetical document generation"
    python3 << 'EOF'
import asyncio
from agentic.hyde import HyDEExpander, HyDEMode

async def test():
    expander = HyDEExpander()
    result = await expander.expand(
        query="How do I reset a robot alarm?",
        mode=HyDEMode.SINGLE
    )
    await expander.close()

    assert len(result.hypothetical_documents) >= 1, "Should generate hypothetical"
    assert len(result.hypothetical_documents[0]) > 50, "Hypothetical should be substantial"
    assert result.fused_embedding is not None, "Should have fused embedding"

    print(f"  Generated: {len(result.hypothetical_documents)} hypothetical(s)")
    print(f"  Length: {len(result.hypothetical_documents[0])} chars")
    print(f"  Gen time: {result.generation_time_ms:.0f}ms")
    print(f"  Emb time: {result.embedding_time_ms:.0f}ms")

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Hypothetical generation"
    else
        print_fail "Hypothetical generation"
    fi

    print_test "Multi-hypothetical generation"
    python3 << 'EOF'
import asyncio
from agentic.hyde import HyDEExpander, HyDEMode

async def test():
    expander = HyDEExpander()
    result = await expander.expand(
        query="What causes overheating?",
        mode=HyDEMode.MULTI,
        num_hypotheticals=3
    )
    await expander.close()

    assert len(result.hypothetical_documents) >= 2, "Should generate multiple hypotheticals"
    print(f"  Generated: {len(result.hypothetical_documents)} hypotheticals")

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Multi-hypothetical generation"
    else
        print_fail "Multi-hypothetical generation"
    fi
}

# ============================================================================
# Test: RAGAS Evaluation
# ============================================================================
test_ragas() {
    print_header "RAGAS Evaluation Tests"

    print_test "Claim extraction"
    python3 << 'EOF'
import asyncio
from agentic.ragas import RAGASEvaluator

async def test():
    evaluator = RAGASEvaluator()
    claims = await evaluator.extract_claims(
        "The FANUC robot uses servo motors. The alarm code SRVO-001 indicates overcurrent."
    )
    await evaluator.close()

    assert len(claims) >= 1, "Should extract at least one claim"
    print(f"  Extracted: {len(claims)} claims")
    for i, claim in enumerate(claims[:3], 1):
        print(f"    {i}. {claim[:60]}...")

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Claim extraction"
    else
        print_fail "Claim extraction"
    fi

    print_test "Full RAGAS evaluation"
    python3 << 'EOF'
import asyncio
from agentic.ragas import RAGASEvaluator

async def test():
    evaluator = RAGASEvaluator()

    result = await evaluator.evaluate(
        question="How do I reset a robot alarm?",
        answer="Press the RESET button on the teach pendant. Check the error log for history.",
        contexts=[
            "To reset alarms, press RESET on the teach pendant.",
            "The alarm log contains error history and diagnostics."
        ]
    )
    await evaluator.close()

    print(f"  Faithfulness: {result.faithfulness:.2f}")
    print(f"  Answer Relevancy: {result.answer_relevancy:.2f}")
    print(f"  Context Relevancy: {result.context_relevancy:.2f}")
    print(f"  Overall: {result.overall_score:.2f}")
    print(f"  Time: {result.evaluation_time_ms:.0f}ms")

asyncio.run(test())
EOF
    if [ $? -eq 0 ]; then
        print_pass "Full RAGAS evaluation"
    else
        print_fail "Full RAGAS evaluation"
    fi
}

# ============================================================================
# Test: API Endpoints
# ============================================================================
test_api() {
    print_header "API Endpoint Tests"

    if ! check_server; then
        print_skip "Server not running - skipping API tests"
        return
    fi

    print_test "Hybrid stats endpoint"
    RESPONSE=$(curl -s "$API_BASE/hybrid/stats")
    if echo "$RESPONSE" | grep -q '"success":true'; then
        print_pass "Hybrid stats endpoint"
        echo "  Response: $(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"docs={d.get(\"data\",{}).get(\"documents_indexed\",0)}")')"
    else
        print_fail "Hybrid stats endpoint"
    fi

    print_test "HyDE stats endpoint"
    RESPONSE=$(curl -s "$API_BASE/hyde/stats")
    if echo "$RESPONSE" | grep -q '"success":true'; then
        print_pass "HyDE stats endpoint"
    else
        print_fail "HyDE stats endpoint"
    fi

    print_test "RAGAS stats endpoint"
    RESPONSE=$(curl -s "$API_BASE/ragas/stats")
    if echo "$RESPONSE" | grep -q '"success":true'; then
        print_pass "RAGAS stats endpoint"
    else
        print_fail "RAGAS stats endpoint"
    fi

    print_test "HyDE expand endpoint"
    RESPONSE=$(curl -s -X POST "$API_BASE/hyde/expand" \
        -H "Content-Type: application/json" \
        -d '{"query": "How to reset alarm?", "mode": "single"}')
    if echo "$RESPONSE" | grep -q '"hypothetical_documents"'; then
        print_pass "HyDE expand endpoint"
    else
        print_fail "HyDE expand endpoint"
    fi

    print_test "Hybrid index endpoint"
    RESPONSE=$(curl -s -X POST "$API_BASE/hybrid/index" \
        -H "Content-Type: application/json" \
        -d '{"documents": [{"doc_id": "test1", "content": "Test document for hybrid indexing"}]}')
    if echo "$RESPONSE" | grep -q '"indexed"'; then
        print_pass "Hybrid index endpoint"
    else
        print_fail "Hybrid index endpoint"
    fi

    print_test "Hybrid search endpoint"
    RESPONSE=$(curl -s -X POST "$API_BASE/hybrid/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "test document", "top_k": 5, "mode": "hybrid"}')
    if echo "$RESPONSE" | grep -q '"results"'; then
        print_pass "Hybrid search endpoint"
    else
        print_fail "Hybrid search endpoint"
    fi
}

# ============================================================================
# Test: Full Agentic Search
# ============================================================================
test_search() {
    print_header "Full Agentic Search Tests"

    if ! check_server; then
        print_skip "Server not running - skipping search tests"
        return
    fi

    print_test "Agentic search pipeline"
    RESPONSE=$(curl -s -X POST "$API_BASE/agentic" \
        -H "Content-Type: application/json" \
        -d '{"query": "What is a FANUC robot?", "max_iterations": 2}' \
        --max-time 120)
    if echo "$RESPONSE" | grep -q '"synthesized_context"'; then
        print_pass "Agentic search pipeline"
        echo "  Confidence: $(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("data",{}).get("confidence_score","N/A"))')"
    else
        print_fail "Agentic search pipeline"
    fi
}

# ============================================================================
# Test: Quick Tests (No LLM)
# ============================================================================
test_quick() {
    test_imports
    test_bm25
}

# ============================================================================
# Main
# ============================================================================
main() {
    print_header "memOS System Test Suite"
    echo "Testing agentic search components..."
    echo ""

    activate_venv

    case "${1:-all}" in
        quick)
            test_quick
            ;;
        imports)
            test_imports
            ;;
        bm25)
            test_bm25
            ;;
        hybrid)
            test_hybrid
            ;;
        hyde)
            test_hyde
            ;;
        ragas)
            test_ragas
            ;;
        api)
            test_api
            ;;
        search)
            test_search
            ;;
        all)
            test_imports
            test_bm25
            test_hybrid
            test_hyde
            test_ragas
            test_api
            test_search
            ;;
        *)
            echo "Usage: $0 {all|quick|imports|bm25|hybrid|hyde|ragas|api|search}"
            exit 1
            ;;
    esac

    # Summary
    print_header "Test Summary"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""

    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    fi
}

main "$@"
