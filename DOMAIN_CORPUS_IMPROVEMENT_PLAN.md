# memOS Domain Corpus & Abstraction Improvement Plan

## Based on: 6-Agent Parallel Analysis (December 2025)
## Date: 2025-12-29
## Status: COMPREHENSIVE AUDIT COMPLETE

---

## Executive Summary

This plan synthesizes findings from 6 parallel sub-agents:
- **3 Codebase Review Agents**: Domain corpus, embedding/retrieval, agentic pipeline
- **3 Research Agents**: NLP best practices, hybrid retrieval, agent orchestration

### Key Findings

| Area | Current State | Gap Identified | Impact |
|------|---------------|----------------|--------|
| **Domain Corpus** | 316 trusted + 62 premium domains | Query-type dynamic boosts missing | +5-10% precision |
| **Hybrid Retrieval** | BGE-M3 dense+sparse working | ColBERT disabled, sparse-only mode unused | +30% recall potential |
| **Agent Coordination** | Scratchpad defined | Write-only (agents don't read findings) | -25% token waste |
| **Multi-Agent** | Phase exists | Results orphaned, never merged | 0% utilization |
| **Quality Control** | CRAG + Self-RAG + RAGAS | Triple redundancy, no consensus | +15% execution time |

---

## PART A: CRITICAL FIXES (Priority 0-1)

### Fix 1: Implement Scratchpad Finding Cache
**Priority**: P0 - CRITICAL
**Effort**: 1 day
**Token Savings**: 25-30%

**Problem**: Agents write to scratchpad but never read it before searching.

**Current** (`orchestrator_universal.py:1835`):
```python
# SEARCH: Don't check for already-answered questions
raw_results = await self.searcher.search(queries_list)
```

**Fix**:
```python
# Before search: check scratchpad for answered questions
answered = scratchpad.get_answered_questions()
queries_to_search = [q for q in queries_list if q not in answered]

if queries_to_search:
    raw_results = await self.searcher.search(queries_to_search)
else:
    # Reuse cached findings
    raw_results = scratchpad.get_findings_for_questions(queries_list)
```

**Files to Modify**:
- `server/agentic/orchestrator_universal.py:1835`
- `server/agentic/scratchpad.py` - Add `get_answered_questions()` method

---

### Fix 2: Enable ColBERT Late Interaction
**Priority**: P0 - CRITICAL
**Effort**: 2-3 days
**Recall Improvement**: +30%

**Problem**: ColBERT is disabled (`bge_m3_hybrid.py:821`).

**Current**:
```python
class ColBERTReranker:
    enabled: bool = False  # Disabled!
```

**Root Cause**: Ollama's embedding API doesn't return token-level embeddings.

**Solution Options**:

| Option | Effort | Quality |
|--------|--------|---------|
| A. Use Jina-ColBERT-v2 via API | 1 day | +6.5% over ColBERTv2 |
| B. Self-host BGE-M3 via FlagEmbedding | 2-3 days | Native token embeddings |
| C. Use cross-encoder reranking instead | 1 day | +28% NDCG over bi-encoder |

**Recommendation**: Option C (cross-encoder) for immediate gains, then Option B for long-term.

**Files to Modify**:
- `server/agentic/bge_m3_hybrid.py:807-857`
- Add new `server/agentic/cross_encoder_reranker.py`

---

### Fix 3: Remove RAGAS Duplication
**Priority**: P1 - HIGH
**Effort**: 0.5 day
**Execution Time Savings**: 15%

**Problem**: Triple quality control with no consensus:
1. CRAG (pre-synthesis) - evaluates retrieval quality
2. Self-RAG (post-synthesis) - evaluates factuality
3. RAGAS (post-synthesis) - duplicates Self-RAG

**Current** (`orchestrator_universal.py:2163-2230`):
```python
reflection_result = await self._phase_self_reflection(...)  # Phase 6
ragas_result = await self._phase_ragas_evaluation(...)      # Phase 7 - DUPLICATE
```

**Fix**: Keep CRAG + Self-RAG, use RAGAS only as tiebreaker when Self-RAG confidence < 0.70.

**Files to Modify**:
- `server/agentic/orchestrator_universal.py:2163-2230`

---

### Fix 4: Integrate Multi-Agent Results
**Priority**: P1 - HIGH
**Effort**: 1 day
**Current Utilization**: 0%

**Problem**: Multi-agent results stored in metadata but never used (`orchestrator_universal.py:2765-2772`).

**Current**:
```python
multi_agent_results = await self._phase_multi_agent_execution(...)
if multi_agent_results:
    enhancement_metadata["agents_used"] = multi_agent_results.get("agents", [])
    # Results stored in metadata, NOT used in synthesis!
```

**Fix**:
```python
multi_agent_results = await self._phase_multi_agent_execution(...)
if multi_agent_results:
    # Merge agent findings into main results
    state.raw_results.extend(multi_agent_results.get("findings", []))
    state.scraped_contents.extend(multi_agent_results.get("contents", []))
    enhancement_metadata["agents_used"] = multi_agent_results.get("agents", [])
```

**Files to Modify**:
- `server/agentic/orchestrator_universal.py:2765-2772`
- `server/agentic/multi_agent.py` - Return structured findings

---

## PART B: DOMAIN CORPUS OPTIMIZATION (Priority 2)

### Improvement 1: Query-Type Dynamic Domain Boosts
**Effort**: 3-4 hours
**Precision Improvement**: +5-10%

**Problem**: Fixed boosts (0.25/0.15) regardless of query type.

**Current** (`searcher.py:1507-1523`):
```python
boost = 0.25 if is_premium_domain else 0.15  # Fixed!
```

**Fix**: Dynamic boosts based on query-domain alignment.

```python
def get_dynamic_domain_boost(self, query_type: str, domain: str) -> float:
    """Return context-aware boost value."""
    domain_category = self._get_domain_category(domain)

    # Query-domain alignment boosts
    alignment_matrix = {
        ("fanuc", "fanuc"): 0.35,      # FANUC query + FANUC domain
        ("fanuc", "industrial"): 0.25,
        ("academic", "academic"): 0.30,
        ("technical", "documentation"): 0.28,
        ("imm", "imm_manufacturer"): 0.30,
        ("imm", "plastics"): 0.25,
    }

    key = (query_type, domain_category)
    if key in alignment_matrix:
        return alignment_matrix[key]

    # Default tier-based boost
    return 0.25 if domain in PREMIUM_DOMAINS else 0.15
```

**Files to Modify**:
- `server/agentic/searcher.py:1507-1523`
- Add `_get_domain_category()` method

---

### Improvement 2: Domain Category Awareness
**Effort**: 2-3 hours
**Impact**: Better result diversity

**Problem**: 19 domain categories defined but not used during aggregation.

**Fix**: Add category matching bonus.

```python
def apply_category_matching(self, query: str, result: SearchResult) -> float:
    """Boost results from query-aligned categories."""
    query_categories = self.detect_query_categories(query)  # e.g., ["fanuc", "robotics"]
    result_category = self._get_domain_category(result.source)

    if result_category in query_categories:
        return 0.05  # Extra boost for category match
    return 0.0
```

---

### Improvement 3: Domain Performance Analytics
**Effort**: 2-3 hours
**Impact**: Data-driven domain optimization

**Problem**: No tracking of which domains produce best results.

**Fix**: Add domain performance metrics.

```python
class DomainPerformanceTracker:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "boost_applied": 0,
            "final_selection_rate": 0.0,
            "avg_synthesis_confidence": 0.0,
            "appearances": 0,
            "in_final_answer": 0
        })

    def record_result(self, domain: str, was_selected: bool, confidence: float):
        self.stats[domain]["appearances"] += 1
        if was_selected:
            self.stats[domain]["in_final_answer"] += 1
        # Update rolling averages...
```

---

## PART C: HYBRID RETRIEVAL OPTIMIZATION (Priority 2)

### Improvement 1: Sparse-Only Mode for Technical Queries
**Effort**: 2-3 hours
**Precision for Exact Matches**: +15%

**Problem**: BM25 mode exists but never activated.

**Current** (`bge_m3_hybrid.py:54-59`):
```python
class RetrievalMode(Enum):
    DENSE_ONLY = "dense_only"    # Never used
    SPARSE_ONLY = "sparse_only"  # Never used
    HYBRID = "hybrid"            # Always used
```

**Fix**: Route technical queries to sparse-only.

```python
def select_retrieval_mode(self, query_type: str) -> RetrievalMode:
    """Select optimal mode based on query characteristics."""
    if query_type in ["error_code", "part_number", "exact_match"]:
        return RetrievalMode.SPARSE_ONLY  # BM25 for exact terms
    elif query_type in ["conceptual", "semantic"]:
        return RetrievalMode.DENSE_ONLY   # Embeddings for concepts
    else:
        return RetrievalMode.HYBRID       # Default hybrid
```

---

### Improvement 2: RRF Parameter Tuning
**Effort**: 1-2 hours
**Potential**: +2-5% NDCG

**Current**: k=60 (default).

**Research Finding**: Optimal k varies by domain. For technical/precision domains, k=40-50 may be better.

**Fix**: Add configurable k with domain-based defaults.

```python
RRF_K_BY_DOMAIN = {
    "technical": 46,   # Higher influence to top results
    "academic": 55,    # Balanced
    "general": 60,     # Default
    "legal": 40,       # High precision
}
```

---

### Improvement 3: Enable Hybrid Search in MINIMAL/BALANCED Presets
**Effort**: 0.5 hours
**Impact**: Better quality for default users

**Problem**: `enable_hybrid_reranking=False` in MINIMAL/BALANCED.

**Fix**: Enable for BALANCED at minimum.

```python
# orchestrator_universal.py:429
OrchestratorPreset.BALANCED: FeatureConfig(
    enable_hybrid_reranking=True,  # Was False
    # ...
)
```

---

## PART D: AGENT ORCHESTRATION OPTIMIZATION (Priority 2-3)

### Improvement 1: Buffer of Thoughts Integration
**Effort**: 3-4 days
**Token Reduction**: 88% vs tree/graph methods

**Research Finding**: BoT stores reusable thought-templates, achieving Llama3-8B + BoT > Llama3-70B.

**Implementation**:
```python
class MetaBuffer:
    """Store successful reasoning templates for reuse."""

    def __init__(self):
        self.templates = {}  # query_pattern -> thought_template

    def find_template(self, query: str) -> Optional[ThoughtTemplate]:
        """Find matching template for query."""
        query_embedding = self.embed(query)
        for pattern, template in self.templates.items():
            if self.similarity(query_embedding, pattern) > 0.85:
                return template
        return None

    def store_template(self, query: str, reasoning_chain: list):
        """Store successful reasoning for reuse."""
        template = self.extract_template(reasoning_chain)
        self.templates[self.embed(query)] = template
```

**Note**: `enable_meta_buffer` exists but is underutilized.

---

### Improvement 2: Chain-of-Draft for DeepSeek R1
**Effort**: 0.5 hours
**Token Reduction**: 68-92%

**Research Finding**: Limiting each reasoning step to ~5 words dramatically reduces tokens.

**Fix**: Update system prompt for DeepSeek R1.

```python
CHAIN_OF_DRAFT_PROMPT = """
Think step by step, but only keep a minimum draft for each thinking step,
with 5 words at most. Focus on key reasoning, not verbose explanation.
"""
```

---

### Improvement 3: Remove Dead Features
**Effort**: 1 day
**Complexity Reduction**: -200 LOC

**Features to Remove** (based on audit):
- `enable_sufficient_context` (2 refs, dead code path)
- `enable_prefetching` (3 refs, minimal benefit)
- `enable_actor_factory` (2 refs, creates objects never used)
- `enable_memory_tiers` (2 refs, never called)

**Files to Modify**:
- `server/agentic/orchestrator_universal.py`
- Remove corresponding phase methods

---

## PART E: NLP/ENTITY IMPROVEMENTS (Priority 3)

### Improvement 1: Domain-Adaptive Pre-Training
**Effort**: 2-3 weeks
**Impact**: +10-15% domain relevance

**Research Finding**: DAPT on industrial documentation corpus significantly improves retrieval.

**Steps**:
1. Collect FANUC manuals, robotics literature (already have PDF tools)
2. Continue pre-training BGE-M3 on corpus (2-3 epochs)
3. Apply LM-Cocktail to merge with base model

---

### Improvement 2: Acronym Dictionary
**Effort**: 1-2 days
**Impact**: Better query understanding

**Problem**: Acronyms (FANUC, SRVO, IMM, MOTN) not expanded consistently.

**Fix**: Build domain-specific acronym dictionary.

```python
INDUSTRIAL_ACRONYMS = {
    "SRVO": "Servo Alarm",
    "MOTN": "Motion Alarm",
    "SYST": "System Alarm",
    "IMM": "Injection Molding Machine",
    "VFD": "Variable Frequency Drive",
    "PLC": "Programmable Logic Controller",
    "FANUC": "Factory Automation Numerical Control",
    # ... 100+ more
}

def expand_acronyms(query: str) -> str:
    """Expand known acronyms in query."""
    for acronym, expansion in INDUSTRIAL_ACRONYMS.items():
        if acronym in query.upper():
            query = query.replace(acronym, f"{acronym} ({expansion})")
    return query
```

---

### Improvement 3: Intent Classification
**Effort**: 2-3 days
**Impact**: Better query routing

**Research Finding**: Distinguishing troubleshooting vs informational queries improves results.

**Fix**: Add intent classifier.

```python
class QueryIntentClassifier:
    INTENTS = ["troubleshooting", "informational", "procedural", "comparison"]

    async def classify(self, query: str) -> str:
        """Classify query intent."""
        prompt = f"""Classify this query intent:
        Query: {query}

        Options: troubleshooting, informational, procedural, comparison
        Return only the intent label."""

        return await self.llm.generate(prompt)
```

---

## PART F: EVALUATION & METRICS (Priority 3)

### Improvement 1: Domain-Specific Test Set
**Effort**: 3-4 days
**Impact**: Reliable quality measurement

**Current**: No systematic benchmarking.

**Fix**: Build FANUC troubleshooting test set.

```python
FANUC_BENCHMARK = [
    {
        "query": "SRVO-063 alarm causes",
        "expected_entities": ["SRVO-063", "encoder", "calibration"],
        "expected_domains": ["fanucamerica.com", "robot-forum.com"]
    },
    {
        "query": "motor overcurrent fault",
        "expected_entities": ["SRVO", "overcurrent", "amplifier"],
        "expected_domains": ["rockwellautomation.com", "plctalk.net"]
    },
    # ... 50+ test cases
]
```

---

### Improvement 2: Technical Accuracy Metric
**Effort**: 1-2 days
**Impact**: Separate from semantic similarity

**Research Finding**: Presence of correct answer isn't sufficient; must assess understandability, logical structure.

**Fix**: Add technical accuracy scorer.

```python
class TechnicalAccuracyScorer:
    def score(self, answer: str, ground_truth: str) -> dict:
        return {
            "entity_coverage": self._entity_overlap(answer, ground_truth),
            "procedure_completeness": self._procedure_check(answer),
            "safety_warnings_present": self._safety_check(answer),
            "technical_term_accuracy": self._term_check(answer)
        }
```

---

## Implementation Roadmap

### Week 1: Critical Fixes (P0-P1)
| Day | Task | Effort | Impact |
|-----|------|--------|--------|
| 1 | Scratchpad finding cache | 1 day | -25% tokens |
| 2-3 | Cross-encoder reranking | 2 days | +28% NDCG |
| 4 | Remove RAGAS duplication | 0.5 day | -15% time |
| 4 | Integrate multi-agent results | 0.5 day | Enable FULL preset |
| 5 | Testing & validation | 1 day | Verify improvements |

### Week 2: Domain Optimization (P2)
| Day | Task | Effort | Impact |
|-----|------|--------|--------|
| 1 | Query-type dynamic boosts | 0.5 day | +5-10% precision |
| 1 | Domain category awareness | 0.5 day | Better diversity |
| 2 | Sparse-only mode routing | 0.5 day | +15% exact match |
| 2 | RRF parameter tuning | 0.5 day | +2-5% NDCG |
| 3 | Enable hybrid in BALANCED | 0.5 day | Default quality |
| 3 | Domain performance tracking | 0.5 day | Analytics |
| 4-5 | Testing & benchmarking | 2 days | Validate |

### Week 3-4: Advanced Features (P3)
| Task | Effort | Impact |
|------|--------|--------|
| Buffer of Thoughts | 3-4 days | 88% token reduction |
| Acronym dictionary | 1-2 days | Better understanding |
| Intent classification | 2-3 days | Better routing |
| Domain test set | 3-4 days | Reliable metrics |
| Remove dead features | 1 day | -200 LOC |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Token usage per query | Baseline | -25% | Scratchpad cache |
| NDCG@10 | Baseline | +15-20% | Cross-encoder + hybrid |
| Execution time | Baseline | -15% | Remove RAGAS duplicate |
| Domain match rate | 50-80% | 70-90% | Dynamic boosts |
| FULL preset utilization | 0% | 100% | Multi-agent fix |
| Test coverage | 0% | 80%+ | Domain test set |

---

## Risk Mitigation

1. **Git branches**: Create `fix/domain-optimization` branch
2. **Incremental commits**: One feature per commit
3. **A/B testing**: Compare new vs old for each change
4. **Rollback plan**: Feature flags for all new features
5. **Monitoring**: Add metrics before/after each change

---

## References

### Codebase Audit Reports
- Domain Corpus Integration: Agent a88ba86
- Embedding/Retrieval System: Agent a4b6faf
- Agentic Pipeline Architecture: Agent a6e2c30

### Research Reports
- Domain-Specific NLP Best Practices: Agent aa2ced3
- Hybrid Retrieval Best Practices: Agent a25b69f
- AI Agent Orchestration Patterns: Agent a0e489c

### Key Papers
- BGE-M3: Multi-Functionality Multi-Linguality Multi-Granularity (BAAI, 2024)
- CRAG: Corrective Retrieval Augmented Generation (arXiv:2401.15884)
- Self-RAG: Learning to Retrieve, Generate, and Critique (ICLR 2024)
- Buffer of Thoughts (NeurIPS 2024 Spotlight)
- Chain-of-Draft (Zoom AI, 2025)
- Adaptive-RAG (NAACL 2024)

---

*Plan generated by 6-agent parallel analysis - December 29, 2025*
