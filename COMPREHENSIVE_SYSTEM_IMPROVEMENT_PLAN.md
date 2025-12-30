# Comprehensive System Improvement Plan

**Generated:** December 29, 2025
**Scope:** PDF_Extraction_Tools, memOS, and SearXNG Integration
**Methodology:** Multi-agent audit + NLP best practices research

---

## Executive Summary

This plan synthesizes findings from three codebase audits and two NLP best practices research agents. The systems collectively process FANUC robotics documentation with 8,449 error codes across 105 categories.

### Key Findings

| System | Status | Critical Issues |
|--------|--------|-----------------|
| **PDF_Extraction_Tools** | Functional, gaps | 4 critical, 7 high priority |
| **memOS** | Complex, fragmented | 60+ modules, no test suite |
| **SearXNG Integration** | BROKEN | Missing `node_id` field |

### Top Priority Fixes

1. **CRITICAL:** Fix memOS integration (missing `node_id` field)
2. **CRITICAL:** Fix path traversal vulnerability
3. **HIGH:** Integrate semantic search (embeddings exist but unused)
4. **HIGH:** Implement RRF ranking (already coded, not enabled)
5. **HIGH:** Add entity normalization for error codes

---

## Phase 1: Critical Security & Integration Fixes (Week 1)

### 1.1 Fix memOS SearchResult Field Names

**Problem:** memOS expects `node_id`, PDF API returns `doc_id`
**Impact:** All memOS → PDF integration is broken
**Files:** `pdf_extractor/api/models.py`, `pdf_extractor/models/core.py`

**Action:**
```python
# Already fixed in models/core.py - verify deployment
@dataclass
class SearchResult:
    node_id: str          # Changed from document_id
    score: float          # Changed from relevance_score
    content_preview: str  # Changed from snippet

    # Backwards-compatible aliases
    @property
    def document_id(self) -> str:
        return self.node_id
```

### 1.2 Fix Path Traversal Vulnerability

**Problem:** `ingest.py` accepts arbitrary file paths
**Risk:** Read any file on system via `../../etc/passwd`
**File:** `pdf_extractor/api/routes/ingest.py`

**Action:**
```python
from pathlib import Path
import os

ALLOWED_BASE_DIRS = [
    Path(os.environ.get("PDF_STORAGE_DIR", "./documents")).resolve()
]

def validate_path(file_path: str) -> Path:
    """Validate path is within allowed directories."""
    resolved = Path(file_path).resolve()
    for base_dir in ALLOWED_BASE_DIRS:
        if resolved.is_relative_to(base_dir):
            return resolved
    raise HTTPException(403, "Access denied: path outside allowed directories")
```

### 1.3 Fix CORS Misconfiguration

**Problem:** `allow_origins=["*"]` with `allow_credentials=True` is invalid
**File:** `pdf_extractor/api/main.py`

**Action:**
```python
# Replace with explicit origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.4 Fix Bare Exception Handlers

**Problem:** `except:` catches all errors silently
**Files:** `hierarchical_text_extractor.py:299`, others

**Action:**
```python
# Replace bare except with specific types
try:
    # operation
except (ValueError, TypeError) as e:
    logger.warning(f"Handled error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

---

## Phase 2: Semantic Search Integration (Week 2)

### 2.1 Enable RRF Ranking

**Problem:** RRFHybridRanker is implemented but not used
**Impact:** Suboptimal search ranking
**Files:** `pdf_extractor/search/search_engine.py`, `ranking.py`

**Current State:**
- `RRFHybridRanker` class exists with k=60
- `HybridRanker` used instead (has placeholder methods)

**Action:**
```python
# In search_engine.py, switch to RRF
class TechnicalSearchEngine:
    def __init__(self, ...):
        # Replace:
        # self.hybrid_ranker = HybridRanker(...)
        # With:
        self.hybrid_ranker = RRFHybridRanker(
            config=RRFConfig(k=60, weights={
                'bm25f': 1.0,
                'semantic': 1.0,
                'error_code': 1.5
            })
        )
```

### 2.2 Complete Hybrid Ranker Feature Extractors

**Problem:** 4 methods return hardcoded values
**File:** `pdf_extractor/search/ranking.py`

**Placeholder Methods to Fix:**
- `_calculate_term_coverage()` → returns 0.5
- `_calculate_structural_proximity()` → returns 0.5
- `_get_section_type_score()` → returns 0.5
- `_get_document_content()` → returns None

**Action:** Implement actual calculations (see `docs/NLP_RAG_BEST_PRACTICES.md`)

### 2.3 Integrate Embeddings into Search Pipeline

**Problem:** OllamaEmbeddings exist but aren't used for search
**Impact:** No semantic similarity matching

**Action:**
```python
# Add semantic branch to hybrid search
def hybrid_search(query: str, ...) -> List[SearchResult]:
    # Branch 1: BM25F (existing)
    bm25f_results = self.bm25f_ranker.rank(query, candidates)

    # Branch 2: Semantic (NEW)
    query_embedding = self.embedding_service.encode(query)
    semantic_results = self._semantic_rank(query_embedding, candidates)

    # Fuse with RRF
    return self.rrf_ranker.fuse({
        'bm25f': bm25f_results,
        'semantic': semantic_results,
        'error_code': self._error_code_rank(query)
    })
```

---

## Phase 3: Entity Normalization (Week 3)

### 3.1 Implement Error Code Normalizer

**Problem:** SRVO-063, SRVO063, srvo-063 treated as different entities
**Impact:** Missed matches, duplicate results

**Action:** Create `pdf_extractor/entities/normalizer.py` (see research doc)

### 3.2 Build Entity Linking System

**Problem:** No cross-reference resolution across documents
**Action:** Create `pdf_extractor/entities/linker.py`

### 3.3 Update Entity Index with Normalized Forms

**File:** `pdf_extractor/graph/unified_graph.py`

**Action:**
```python
def add_entity(self, entity_text: str, node_id: str):
    # Normalize first
    normalized = self.normalizer.normalize(entity_text)
    if normalized:
        canonical = normalized.canonical
        self.entity_index[canonical].add(node_id)
        # Also index aliases
        for alias in normalized.aliases:
            self._alias_to_canonical[alias] = canonical
```

---

## Phase 4: BGE-M3 Migration (Week 4)

### 4.1 Install Dependencies

```bash
pip install FlagEmbedding torch
```

### 4.2 Implement BGE-M3 Service

**Rationale:** BGE-M3 (72% accuracy) vs mxbai-embed-large (59.25%)
**File:** `pdf_extractor/embeddings/bge_m3.py` (from research doc)

### 4.3 Generate New Embeddings

```python
# Re-embed all 8,449 error code entities
service = BGEM3EmbeddingService()
for entity in graph.get_entities():
    embedding = service.encode(
        f"{entity.canonical}: {entity.cause}. {entity.remedy}",
        return_dense=True,
        return_sparse=True
    )
    entity.embedding = embedding.dense
    entity.sparse_embedding = embedding.sparse
```

### 4.4 Update HSEA Export

**File:** `scripts/export_to_hsea.py`

Use BGE-M3 embeddings for π₃ Substantive stratum (66% of HSEA).

---

## Phase 5: Query Enhancement (Week 5)

### 5.1 Implement Query Intent Detection

```python
def detect_query_intent(query: str) -> str:
    """Classify query for strategy selection."""
    if re.search(r'[A-Z]{2,4}-?\d{3,4}', query.upper()):
        return 'error_code'
    if query.lower().startswith(('how', 'why', 'what')):
        return 'question'
    if re.search(r'\d+\s*(mm|deg|rpm)', query, re.I):
        return 'parameter'
    return 'general'
```

### 5.2 Implement Query Expansion (HyDE-style)

```python
def expand_query(query: str, intent: str) -> str:
    """Expand query based on intent."""
    if intent == 'error_code':
        code = extract_error_code(query)
        return f"What causes {code}? How to fix {code}? {code} troubleshooting steps."
    return query
```

### 5.3 Add Fuzzy Matching

```python
# For typo tolerance
from rapidfuzz import fuzz

def fuzzy_error_match(query_code: str, canonical_codes: List[str]) -> List[str]:
    """Find error codes similar to query."""
    matches = []
    for code in canonical_codes:
        if fuzz.ratio(query_code.upper(), code) > 85:
            matches.append(code)
    return matches
```

---

## Phase 6: memOS Consolidation (Week 6)

### 6.1 Audit memOS Module Dependencies

**Problem:** 60+ agentic modules with unclear interactions

**Action:** Create dependency graph and consolidate:
- Multiple embedding aggregators → single `EmbeddingServiceManager`
- Multiple cache modules → unified cache with Redis
- Multiple orchestrators → single `UniversalOrchestrator`

### 6.2 Implement Test Suite for memOS

**Problem:** No unit tests exist

**Priority Test Targets:**
1. HSEA three-stratum retrieval
2. BGE-M3 hybrid search
3. HyDE query expansion
4. Error code normalization
5. memOS → PDF API integration

### 6.3 Document API Contracts

Create OpenAPI schemas for:
- `/api/v1/search/hsea/search`
- `/api/v1/search/hsea/troubleshoot/{code}`
- `/api/v1/search/hsea/similar/{code}`

---

## Phase 7: Testing & Validation (Week 7-8)

### 7.1 Create Integration Test Suite

```python
# tests/integration/test_memos_pdf_integration.py
async def test_error_code_search_flow():
    """End-to-end: memOS -> PDF API -> results."""
    # 1. Query memOS
    response = await memos_client.search("SRVO-063")

    # 2. Verify PDF results included
    assert any(r.node_type == "error_code" for r in response.results)

    # 3. Verify field names match
    for result in response.results:
        assert hasattr(result, 'node_id')
        assert hasattr(result, 'content_preview')
        assert hasattr(result, 'score')
```

### 7.2 Benchmark Search Quality

**Metrics:**
- NDCG@10 for ranking quality
- MRR for first relevant result
- Recall@K for coverage
- Latency P50/P95/P99

**Baseline:** Current keyword search
**Target:** 20% improvement with hybrid search

### 7.3 Validate HSEA Integration

```python
# Verify three-stratum retrieval
async def test_hsea_strata():
    results = await hsea.search("SRVO-063", include_strata=True)

    assert results.pi1_systemic.count >= 0  # Category anchors
    assert results.pi2_structural.count >= 0  # Relationships
    assert results.pi3_substantive.count >= 1  # Content
```

---

## Implementation Summary

### Priority Order

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Security & Integration | Fix path traversal, CORS, node_id field |
| 2 | Semantic Search | Enable RRF, fix hybrid ranker, embed search |
| 3 | Entity Normalization | Normalizer, linker, updated indices |
| 4 | BGE-M3 Migration | New embeddings, HSEA update |
| 5 | Query Enhancement | Intent detection, expansion, fuzzy match |
| 6 | memOS Consolidation | Unified services, test suite, docs |
| 7-8 | Testing | Integration tests, benchmarks, validation |

### Resource Requirements

| Resource | Purpose | Notes |
|----------|---------|-------|
| BGE-M3 model | Embeddings | ~2GB GPU memory or CPU |
| Redis | Unified caching | Optional but recommended |
| pytest | Test suite | Already in requirements |
| rapidfuzz | Fuzzy matching | Lightweight dependency |

### Success Metrics

1. **Integration:** memOS → PDF flow works end-to-end
2. **Search Quality:** 20% NDCG improvement
3. **Latency:** P95 < 200ms for search
4. **Coverage:** All 8,449 error codes normalized
5. **Test Coverage:** 80% for critical paths

---

## References

- `docs/NLP_RAG_BEST_PRACTICES.md` - Detailed implementation code
- `PROJECT_AUDIT_2025-12-29.md` - PDF tools audit details
- `DUAL_SYSTEM_ENGINEERING_GUIDE.md` - Architecture overview
- `MEMOS_INTEGRATION_PLAN.md` - Integration specifications

---

*Plan generated by multi-agent analysis - December 2025*
