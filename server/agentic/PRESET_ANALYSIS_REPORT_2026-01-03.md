# Agentic Pipeline Preset Analysis Report

**Date**: 2026-01-03
**Author**: Claude Code Analysis

---

## Executive Summary

This report analyzes the memOS agentic search pipeline across different preset configurations (MINIMAL, BALANCED, ENHANCED, RESEARCH, FULL) with complex industrial automation domain queries. Testing revealed several critical issues affecting response quality and latency, along with specific improvement recommendations based on current research.

---

## 1. Observability Findings

### 1.1 Feature Configuration by Preset

| Preset | Feature Count | Key Features |
|--------|---------------|--------------|
| MINIMAL | 4 | content_cache, query_analysis, scratchpad, verification |
| BALANCED | 19 | + crag_evaluation, hybrid_reranking, self_reflection, hsea_context |
| ENHANCED | 30 | + cross_encoder, hyde, entity_tracking, deep_reading, technical_docs |
| RESEARCH | 39 | + dynamic_planning, meta_buffer, semantic_cache, iteration_bandit |
| FULL | 54 | + flare_retrieval, reasoning_dag, entropy_halting, vision_analysis |

### 1.2 Pipeline Graph Tracking

The observability layer shows real-time pipeline progress:
```
[A✓]→[P✓]→[S✓]→[E✓]→[W✓]→[V✓]→[Σ✓]→[R]→[✓]
```
Where: A=Analyze, P=Plan, S=Search, E=Evaluate, W=Web scrape, V=Verify, Σ=Synthesize, R=Reflect

---

## 2. Critical Issues Identified

### 2.1 VRAM Exhaustion (CRITICAL)

**Issue**: Running 4 parallel pipeline tests caused CUDA OOM errors.
```
FlagEmbedding encoding failed: CUDA out of memory. Tried to allocate 490.00 MiB.
GPU 0 has a total capacity of 23.45 GiB of which 270.81 MiB is free.
```

**Root Cause**:
- qwen3:8b synthesis: ~6-7GB per pipeline
- BGE-M3 hybrid retriever: ~4GB
- Vision model (qwen2.5vl:7b): ~8GB
- Total with 4 pipelines: >24GB available

**Recommendation**:
- Limit concurrent pipelines to 2 maximum
- Implement VRAM budget tracking in orchestrator
- Consider q4_0 quantization for embeddings (50% VRAM reduction)

### 2.2 LLM Call Failures

**Issue**: Empty responses from retrieval_evaluator and entity_tracker
```
[ERROR] agentic.retrieval_evaluator: LLM call failed:
[ERROR] agentic.entity_tracker: Entity extraction error:
```

**Root Cause**: LLM context exhaustion or timeout during high load

**Recommendation**:
- Add retry logic with exponential backoff
- Implement fallback to simpler prompts on failure
- Add circuit breaker pattern for LLM calls

### 2.3 Dynamic Planner JSON Parsing Errors

**Issue**: Planner output malformed
```
[WARNING] agentic.dynamic_planner: Failed to parse planner output:
Expecting ':' delimiter: line 57 column 14 (char 2365)
```

**Recommendation**:
- Add JSON schema validation with graceful fallback
- Use structured output (Pydantic models) for planner responses
- Implement repair logic for common JSON malformations

### 2.4 Off-Topic Result Contamination

**Issue**: Medical sites appearing in industrial queries
```
[SCRAPE] ✓ mayoclinic.org: 2,816 chars (704ms)
```

**Root Cause**: Wikipedia/general engines returning medical "injection" results for IMM queries

**Recommendation**:
- Already fixed: Removed wikipedia from IMM engine group
- Add domain blocklist for clearly off-topic domains
- Strengthen relevance filtering threshold

### 2.5 Brave Rate Limiting

**Issue**: Brave engine frequently rate-limited
```
[WARNING] brave failed: Suspended: too many requests, backing off 15s
```

**Status**: FIXED - Implemented automatic fallback with result aggregation
```
[WARNING] Brave rate-limited (35 initial results), retrying with fallback engines
```

---

## 3. Research-Based Improvement Recommendations

### 3.1 Cross-Encoder Reranking Optimization

**Research Finding**: Cross-encoder reranking improves accuracy by 20-35% ([Source](https://www.pinecone.io/learn/series/rag/rerankers/))

**Current Implementation**: Using MiniLM-based cross-encoder (~200-300ms latency)

**Recommendations**:
- Implement adaptive reranking depth (3-50 candidates based on query complexity)
- Consider FlashRank for faster inference
- Optimal candidate count: 50-75 for best NDCG@10

### 3.2 CRAG Corrective Retrieval

**Research Finding**: CRAG uses confidence scoring to trigger corrective actions ([arXiv:2401.15884](https://arxiv.org/abs/2401.15884))

**Current Implementation**: CRAG evaluator with CORRECT/INCORRECT/AMBIGUOUS classification

**Recommendations**:
- Fine-tune T5-large evaluator on industrial domain
- Add web search fallback for INCORRECT classifications
- Implement decompose-then-recompose for complex queries

### 3.3 Self-RAG Reflection Tokens

**Research Finding**: ISREL, ISSUP, ISUSE tokens enable self-correction ([ICLR 2024](https://arxiv.org/abs/2310.11511))

**Current Implementation**: Self-reflection stage post-synthesis

**Recommendations**:
- Implement adaptive retrieval threshold (default 0.2)
- Add segment-level ISSUP scoring for factuality
- Weight: w_rel=1.0, w_sup=1.0, w_use=0.5 (validated defaults)

### 3.4 Query Decomposition Best Practices

**Research Finding**: Azure AI Search shows subqueries running in parallel with semantic reranking improve coverage ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview))

**Current Implementation**: Query tree decoder for sub-query generation

**Recommendations**:
- Include redundancy in queries (both sub-queries and parent query)
- Add structured output for Questions with intermediate answers
- Implement final reasoning step combining sub-answers

### 3.5 Hallucination Detection

**Research Finding**: LLM-as-Judge paradigm outperforms traditional metrics ([HalluLens ACL 2025](https://aclanthology.org/2025.acl-long.1176.pdf))

**Current Implementation**: Verification agent with source cross-checking

**Recommendations**:
- Add semantic uncertainty quantification via density matrix
- Implement HaluCheck-style DPO-aligned detector
- Use curriculum learning for harder negatives

---

## 4. Confidence Scoring Calibration

### 4.1 Current Weight Distribution

| Signal | Current Weight | Recommended |
|--------|----------------|-------------|
| Verification Score | 40% | 35% |
| Source Diversity | 25% | 25% |
| Content Depth | 20% | 20% |
| Synthesis Quality | 15% | 20% |

### 4.2 Calibration Recommendations

- Add retrieval confidence as explicit signal (per CRAG)
- Implement log probability extraction for uncertainty quantification
- Add consistency checking against external knowledge base
- Route uncertain cases (confidence < 0.6) to human review

---

## 5. Latency Optimization

### 5.1 Current Bottlenecks

| Stage | Avg Latency | Target |
|-------|-------------|--------|
| LLM Synthesis | 60-120s | <30s |
| Web Scraping | 0.5-143s | <10s |
| Cross-Encoder | 200-300ms | <100ms |
| Total Pipeline | 120-300s | <60s |

### 5.2 Optimization Strategies

1. **Chain-of-Draft Prompting**: 50-80% token reduction for thinking models
2. **KV Cache Quantization**: q8_0 for 50% VRAM savings
3. **Parallel Execution**: Run independent stages concurrently
4. **Scraping Timeout**: Cap at 10s with fallback to cached content
5. **Model Persistence**: OLLAMA_KEEP_ALIVE=30m

---

## 6. Action Items

### P0 - Critical (This Week)

1. [ ] Add VRAM budget tracking to prevent OOM
2. [ ] Implement retry logic for LLM call failures
3. [ ] Add JSON repair/fallback for planner output
4. [ ] Limit concurrent pipelines to 2

### P1 - High (This Month)

5. [ ] Implement adaptive reranking depth
6. [ ] Add domain blocklist for off-topic filtering
7. [ ] Fine-tune CRAG evaluator on industrial domain
8. [ ] Add scraping timeout (10s max)

### P2 - Medium (Next Quarter)

9. [ ] Implement HaluCheck-style hallucination detection
10. [ ] Add semantic uncertainty quantification
11. [ ] Optimize cross-encoder with FlashRank
12. [ ] Implement query decomposition with redundancy

---

## 7. Sources

- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/abs/2501.09136)
- [CRAG: Corrective Retrieval (arXiv 2401.15884)](https://arxiv.org/abs/2401.15884)
- [Self-RAG (ICLR 2024)](https://arxiv.org/abs/2310.11511)
- [Cross-Encoder Reranking (Pinecone)](https://www.pinecone.io/learn/series/rag/rerankers/)
- [RAG Evaluation Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [HalluLens Benchmark (ACL 2025)](https://aclanthology.org/2025.acl-long.1176.pdf)
- [Azure Agentic Retrieval (Microsoft)](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview)
- [Query Decomposition (Haystack)](https://haystack.deepset.ai/blog/query-decomposition)

---

## 8. Test Results (Session 2026-01-03 17:00 UTC)

### 8.1 Completed Tests

#### BALANCED Preset - IMM Heater Diagnostic
| Metric | Value | Notes |
|--------|-------|-------|
| **Query** | Engel Victory 500 barrel zone 3 heater oscillating +/-15°C | Complex industrial troubleshooting |
| **Duration** | 0.1s | Cache hit (L1 Redis) |
| **Confidence** | 58% | Below 60% target |
| **Sources** | 10 | Mixed relevance |

**Response Quality**: Good synthesis despite source noise - correctly identified:
- Contactor wear diagnosis (multimeter test)
- SSR degradation symptoms (flickering, arcing)
- Heater band failure (resistance measurement)

**Source Issues**: Irrelevant sources included:
- tenforums.com (Windows diagnostics)
- Wikipedia (Common rapper entry)
- Cambridge Dictionary

### 8.2 In-Progress Tests (Incomplete)

| Test | Preset | Query Domain | Status |
|------|--------|--------------|--------|
| FANUC SRVO-023 | ENHANCED | Arc welding J2 axis overload | Timeout (>300s) |
| Allen-Bradley 1756-L73 | RESEARCH | I/O chassis fault 1:13 | Server conflicts |
| Siemens S7-1500 | MINIMAL | ProfiNET IO failure | Server conflicts |

**Root Cause**: Concurrent test agents competed for server port 8001, causing restart loops.

### 8.3 Resource Contention Analysis

During testing, observed severe resource contention:

```
GPU VRAM Timeline:
00:00 - Start: 3GB used (idle)
02:00 - Synthesis: 21GB used (deepseek-r1:14b loaded)
03:00 - VL Scraping: Warning "No models found within 0.5GB"
05:00 - Cross-encoder: Failed (VRAM exhausted)
```

**VL Scraper VRAM Starvation**:
```
[WARNING] services.model_selector: No models found for capability
ModelCapability.VISION within 0.5048828125GB
```

When deepseek-r1:14b (~20GB) is loaded for synthesis, vision models cannot load.

### 8.4 Off-Topic Result Contamination Evidence

Logged during ControlLogix query (PLC troubleshooting):
| URL | Domain | Relevance |
|-----|--------|-----------|
| mayoclinic.org/healthy-lifestyle/nutrition... | Medical | 0% |
| reddit.com/r/AITAH/... | Social drama | 0% |
| forum.manjaro.org/kernel-6-1-1-1 | Linux kernel | 0% |
| wordsmyth.net/?ent=common | Dictionary | 0% |

**Impact**: VL scraper wasted ~90s on irrelevant pages while VRAM-starved.

### 8.5 VL Scraper Cascade Failures

```
[INFO] academy.fanucamerica.com: 15 chars (89905ms) ← 90 second scrape!
[WARNING] VL scraper failed: Extraction failed: No vision model available
```

JS-heavy sites (FANUC Academy, Rockwell Knowledgebase) require VL extraction but synthesis model blocks VRAM.

---

## 9. Updated Action Items

### P0 - Critical (Blocking Tests)

| # | Item | Status | Impact |
|---|------|--------|--------|
| 1 | Serial VL scraping queue | NEW | Prevents VRAM exhaustion |
| 2 | Pre-synthesis relevance filter | NEW | Skip off-topic URLs before scrape |
| 3 | Model unload before VL stage | NEW | Free VRAM for vision models |
| 4 | Limit concurrent pipelines to 2 | VALIDATED | Prevents OOM crashes |

### P1 - High (Quality Issues)

| # | Item | Status |
|---|------|--------|
| 5 | Domain blocklist (medical, social) | NEW |
| 6 | Relevance scoring before VL fallback | NEW |
| 7 | Scrape timeout 10s (currently 90s+) | VALIDATED |
| 8 | Cross-encoder graceful degradation | NEW |

---

## 10. Conclusions

1. **Cache Performance**: Semantic cache (L1 Redis) working well - 0.1s for repeated queries
2. **Synthesis Quality**: Response content is accurate despite noisy sources
3. **Source Selection**: Critical weakness - off-topic results waste time and VRAM
4. **VRAM Management**: No coordination between synthesis and VL stages
5. **Concurrency**: Single server cannot handle multiple parallel pipeline requests

**Recommended Next Step**: Implement pre-scrape relevance scoring to filter URLs before expensive VL extraction.

---

*Report updated: 2026-01-03 17:27 UTC*
*Generated by Claude Code analysis of memOS agentic pipeline*
