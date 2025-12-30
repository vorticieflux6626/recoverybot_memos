# PDF Extraction Tools Integration Audit Report

**Generated:** December 29, 2025
**Purpose:** Integration audit for memOS ↔ PDF_Extraction_Tools
**Agent Analysis:** 5 specialized agents (3 audits + 2 NLP research)

---

## Executive Summary

This report documents the comprehensive multi-agent audit of PDF_Extraction_Tools integration with memOS. The audit identified critical integration issues and provides actionable recommendations based on NLP/ML best practices research.

### Key Statistics

| Metric | Value |
|--------|-------|
| Error Codes Indexed | 8,449 |
| Unique Categories | 105 |
| Node Types | 9 |
| Edge Types | 17 |
| Critical Issues Found | 4 |
| High Priority Issues | 7 |

### Critical Finding

**memOS → PDF Integration is BROKEN**

The `node_id` field expected by memOS is returned as `document_id` by the PDF API, causing all integration queries to fail silently.

---

## 1. Integration Status

### 1.1 Field Name Mismatch (CRITICAL)

memOS expects:
```python
@dataclass
class SearchResult:
    node_id: str          # EXPECTED
    score: float
    content_preview: str
```

PDF API returns:
```python
@dataclass
class SearchResult:
    document_id: str      # ACTUAL
    relevance_score: float
    snippet: str
```

**Fix Applied:**
```python
# pdf_extractor/models/core.py
@dataclass
class SearchResult:
    node_id: str
    score: float
    content_preview: str

    # Backwards-compatible aliases
    @property
    def document_id(self) -> str:
        return self.node_id

    @property
    def relevance_score(self) -> float:
        return self.score
```

### 1.2 HSEA Export Integration

The `scripts/export_to_hsea.py` bridge script maps:

| PDF Graph | HSEA Stratum | Purpose |
|-----------|--------------|---------|
| Category anchors | π₁ Systemic (17%) | High-level navigation |
| Entity relationships | π₂ Structural (17%) | Relationship traversal |
| Error code content | π₃ Substantive (66%) | Full-precision search |

---

## 2. Entity Index Statistics

### 2.1 Category Distribution (Top 10)

| Category | Count | Description |
|----------|-------|-------------|
| CVIS | 726 | Vision system errors |
| SVGN | 471 | Servo gain/tuning |
| INTP | 460 | Interpreter errors |
| MOTN | 455 | Motion control |
| SRVO | 429 | Servo motor/drive |
| PRIO | 367 | Priority/scheduling |
| IBSS | 301 | Internal bus |
| SEAL | 294 | Sealing operations |
| SYST | 293 | System-level |
| FORC | 244 | Force control |

### 2.2 Entity Types in Graph

| Node Type | Count | Edge Types |
|-----------|-------|------------|
| document | 1 | contains, parent_of |
| section | 105 | contains, categorizes |
| chunk | ~200 | contains, reference |
| entity | 8,449 | related_to, causes |
| concept | varies | concept_of |

---

## 3. Search Pipeline Status

### 3.1 Current Implementation

| Component | Status | Issue |
|-----------|--------|-------|
| BM25F Ranking | Implemented | Missing per-field `b` params |
| RRF Fusion | **Implemented** | k=60, ready to enable |
| Semantic Search | Partial | Embeddings unused |
| Error Code Tokenization | **Implemented** | Preserves SRVO-063 |
| Entity Normalization | **Implemented** | 105 categories |

### 3.2 Recommended Improvements

Based on NLP research agent findings:

1. **HyDE Query Expansion**
   - For symptom-based queries (e.g., "robot arm jerking")
   - Generates hypothetical error code entry
   - Bridges semantic gap to technical docs

2. **Multi-Query Retrieval (DMQR-RAG)**
   - General rewrite, keyword extraction, component chains
   - 14.46% precision improvement on complex queries

3. **FLARE Active Retrieval**
   - Retrieve only when model is uncertain
   - Confidence-based triggers for troubleshooting

4. **Int8 Quantization**
   - 75% memory reduction
   - 99.99% search accuracy preserved

---

## 4. API Endpoints for memOS

### 4.1 Required Endpoints

| Endpoint | Method | Status |
|----------|--------|--------|
| `/search` | POST | Working |
| `/health` | GET | Working |
| `/ingest/file` | POST | Path vuln fix needed |
| `/graph/node/{id}` | GET | Working |

### 4.2 HSEA-Specific Endpoints (memOS)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/hsea/search` | POST | Multi-stratum search |
| `/api/v1/search/hsea/troubleshoot/{code}` | GET | Troubleshooting context |
| `/api/v1/search/hsea/similar/{code}` | GET | Similar error codes |
| `/api/v1/search/hsea/index/batch` | POST | Batch indexing |

---

## 5. Security Issues

### 5.1 Path Traversal (CRITICAL)

**File:** `pdf_extractor/api/routes/ingest.py`

**Vulnerability:** Accepts arbitrary file paths
```python
# BEFORE (vulnerable)
file_path = request.json["file_path"]
with open(file_path) as f:  # Can read /etc/passwd
```

**Fix:**
```python
ALLOWED_BASE_DIRS = [
    Path(os.environ.get("PDF_STORAGE_DIR", "./documents")).resolve()
]

def validate_path(file_path: str) -> Path:
    resolved = Path(file_path).resolve()
    for base_dir in ALLOWED_BASE_DIRS:
        if resolved.is_relative_to(base_dir):
            return resolved
    raise HTTPException(403, "Access denied")
```

### 5.2 CORS Misconfiguration

**File:** `pdf_extractor/api/main.py`

**Issue:** `allow_origins=["*"]` with `allow_credentials=True` is invalid per CORS spec.

**Fix:** Use explicit origin whitelist via environment variable.

---

## 6. Implementation Roadmap

### Phase 1: Critical Fixes (Immediate)

1. ✅ Fix `node_id` field names for memOS compatibility
2. ⬜ Deploy path traversal fix
3. ⬜ Fix CORS configuration
4. ⬜ Fix bare exception handlers

### Phase 2: Search Quality (Week 2)

1. ✅ Enable RRF ranking (implemented, needs activation)
2. ⬜ Integrate embeddings into search pipeline
3. ⬜ Add HyDE for symptom queries
4. ⬜ Complete hybrid ranker feature extractors

### Phase 3: Entity Enhancement (Week 3)

1. ✅ Entity normalization (SRVO-063 variants handled)
2. ⬜ Build entity linking for cross-references
3. ⬜ Update entity index with normalized forms

### Phase 4: Embeddings Upgrade (Week 4)

1. ⬜ Evaluate BGE-M3 vs mxbai-embed-large
2. ⬜ Implement int8 quantization for caching
3. ⬜ Add multi-tier cache (memory + Redis)

---

## 7. Troubleshooting Patterns

The system includes 7 built-in patterns for memOS HSEA:

| Pattern ID | Name | Trigger Categories |
|------------|------|-------------------|
| 1 | encoder_replacement | SRVO (encoder alarms) |
| 2 | calibration | MOTN, SRVO (position errors) |
| 3 | communication_reset | HOST, COMM (network errors) |
| 4 | parameter_adjustment | SYST (parameter errors) |
| 5 | safety_interlock | SAFE (safety faults) |
| 6 | servo_power_cycle | SRVO (amplifier errors) |
| 7 | vision_calibration | CVIS (camera errors) |

---

## 8. Files Reference

### PDF_Extraction_Tools

| File | Purpose |
|------|---------|
| `pdf_extractor/models/core.py` | Unified type definitions |
| `pdf_extractor/search/rrf_ranker.py` | RRF fusion implementation |
| `pdf_extractor/entities/normalizer.py` | Error code normalization |
| `pdf_extractor/api/dependencies.py` | FastAPI DI + ML services |
| `scripts/export_to_hsea.py` | memOS bridge script |

### Documentation

| File | Purpose |
|------|---------|
| `docs/NLP_RAG_BEST_PRACTICES.md` | Research implementation guide |
| `COMPREHENSIVE_SYSTEM_IMPROVEMENT_PLAN.md` | 8-week improvement plan |
| `DUAL_SYSTEM_ENGINEERING_GUIDE.md` | Architecture overview |

---

## 9. Testing Status

### 9.1 Test Results

```
59 passed, 4 skipped (optional dependencies)
```

### 9.2 Test Coverage

| Module | Status |
|--------|--------|
| Chunking | ✅ 8 tests |
| Column Detection | ✅ 4 tests |
| Models | ✅ 7 tests |
| Integration | ⬜ Needs implementation |

---

## 10. Next Steps

1. **Verify memOS Integration**
   ```bash
   curl -X POST http://localhost:8001/api/v1/search/hsea/search \
     -H "Content-Type: application/json" \
     -d '{"query": "SRVO-063"}'
   ```

2. **Run Export Script**
   ```bash
   python scripts/export_to_hsea.py --dry-run --sample 10
   ```

3. **Monitor Search Quality**
   - Track NDCG@10 for ranking quality
   - Track MRR for first relevant result
   - Target: 20% improvement with hybrid search

---

*Report generated by multi-agent analysis - December 2025*
