# Max Tokens Configuration Audit Report

**Date**: 2026-01-11
**Auditor**: Claude Code
**Scope**: `/home/sparkone/sdd/Recovery_Bot/memOS/server/config/llm_models.yaml` and related agentic code

---

## Executive Summary

The current max_tokens configuration has **several issues** that could impact output quality:

| Severity | Issue | Affected Components |
|----------|-------|---------------------|
| **HIGH** | Synthesizer has only 4096 max_tokens but outputs can exceed 2000 tokens | Synthesis truncation |
| **HIGH** | Thinking model has 8192 but DeepSeek R1 thinking can use 10K+ tokens | Reasoning truncation |
| **MEDIUM** | Several agents have 256-512 tokens when JSON output needs 500-1000 | JSON parsing failures |
| **LOW** | Planner/Analyzer at 1024 may truncate complex plans | Plan incompleteness |

---

## Current Configuration Summary

### Pipeline Models (Primary)

| Agent | Model | max_tokens | Context Window | Issue |
|-------|-------|------------|----------------|-------|
| analyzer | qwen3:8b | 1024 | 40,960 | May truncate complex analysis |
| url_evaluator | ministral-3:3b | 2048 | 32,000 | OK |
| coverage_evaluator | qwen3:8b | 1024 | 40,960 | OK for simple evaluation |
| planner | qwen3:8b | 1024 | 40,960 | **May truncate multi-phase plans** |
| synthesizer | ministral-3:3b | 4096 | 32,000 | **Measured output: 2043 tokens - needs 6K+** |
| thinking | ministral-3:3b | 8192 | 32,000 | **DeepSeek R1 thinking needs 12K+** |
| retrieval_evaluator | qwen3:8b | 1024 | 40,960 | OK |
| self_reflection | qwen3:8b | 1024 | 40,960 | OK |
| verifier | qwen3:8b | 1024 | 40,960 | OK |

### Utility Models (Secondary)

| Agent | Model | max_tokens | Issue |
|-------|-------|------------|-------|
| entity_extractor | qwen3:8b | 2048 | OK |
| query_decomposer | gemma3:4b | 1024 | OK |
| relevance_scorer | gemma3:4b | **512** | **Too low for JSON with reasoning** |
| uncertainty_detector | qwen3:8b | **512** | **Too low if detection includes context** |
| experience_distiller | qwen3:8b | 2048 | OK |
| prompt_compressor | qwen2.5:0.5b | **256** | OK (just scores) |
| raptor_summarizer | qwen3:8b | 1024 | May truncate long summaries |
| graph_extractor | qwen3:8b | 2048 | OK |
| reasoning_dag | qwen3:8b | 2048 | **May truncate multi-path reasoning** |
| hyde_generator | gemma3:4b | 1024 | OK |
| flare_detector | qwen3:8b | **512** | OK (just detection) |
| information_bottleneck | gemma3:4b | **512** | **Too low for analysis output** |
| cross_encoder | qwen3:8b | **512** | OK (just scores) |
| ragas_judge | gemma3:4b | **512** | **May truncate detailed evaluation** |
| entropy_monitor | qwen3:8b | **512** | OK (just monitoring) |
| scraper_analyzer | qwen3:14b | 4096 | OK |

---

## Measured Output Sizes

From actual test runs:

| Output Type | Measured Size | Current Limit | Status |
|-------------|---------------|---------------|--------|
| FANUC v2 synthesis | ~2,043 tokens | 4,096 | **Marginal** - complex queries need 4K+ |
| FANUC v1 synthesis | ~703 tokens | 4,096 | OK |
| Detailed troubleshooting | ~3,000-4,000 tokens (est.) | 4,096 | **At risk** |
| DeepSeek R1 thinking | 5,000-15,000 tokens | 8,192 | **TRUNCATION RISK** |

---

## Recommendations

### Priority 1: Critical Fixes

```yaml
pipeline:
  synthesizer:
    max_tokens: 8192  # Was 4096 - synthesis can be 4K+ for detailed troubleshooting

  thinking:
    max_tokens: 16384  # Was 8192 - DeepSeek R1 thinking routinely uses 10K+

  planner:
    max_tokens: 2048  # Was 1024 - multi-phase plans need more space
```

### Priority 2: Important Fixes

```yaml
utility:
  reasoning_dag:
    max_tokens: 4096  # Was 2048 - multi-path reasoning generates substantial output

  information_bottleneck:
    max_tokens: 1024  # Was 512 - analysis needs more detail

  ragas_judge:
    max_tokens: 1024  # Was 512 - detailed evaluation metrics
```

### Priority 3: Minor Fixes

```yaml
utility:
  relevance_scorer:
    max_tokens: 768  # Was 512 - marginal increase for JSON safety

  uncertainty_detector:
    max_tokens: 768  # Was 512 - marginal increase
```

---

## Hardcoded Values in Code

Several files had hardcoded `num_predict` values. The following have been fixed to use centralized config:

| File | Status | Original Value | Fix Applied |
|------|--------|----------------|-------------|
| `url_relevance_filter.py` | **FIXED** | 1024 | Now uses `get_config_for_task("url_evaluator")` |
| `planner.py` | **FIXED** | 256 (too low) | Now uses `get_config_for_task("planner")` → 2048 |
| `retrieval_evaluator.py` | **FIXED** | 256-512 | Now uses `get_config_for_task("retrieval_evaluator")` → 1024 |

Remaining files with acceptable hardcoded values:

| File | Line | Value | Status |
|------|------|-------|--------|
| `synthesizer.py` | 214 | 1024 (fallback) | OK - already has config override |
| `self_reflection.py` | 706 | 128-2048 | OK - context-dependent values |
| `verifier.py` | 168 | 256 | OK for simple verification |
| `query_classifier.py` | 120 | 256 | OK for classification |
| `hoprag.py` | 402, 432 | 1024 | OK |
| `entity_tracker.py` | 394 | 2048 | OK |
| `dynamic_planner.py` | 482 | 2048 | OK |
| `orchestrator_universal.py` | 1739, 1837 | 4000 | OK |

---

## Model Context Window Reference

| Model | Context Window | Notes |
|-------|----------------|-------|
| qwen3:8b | 40,960 | Primary model, good balance |
| qwen3:14b | 40,960 | Larger, slower |
| ministral-3:3b | 32,000 | Fast, good for simple tasks |
| gemma3:4b | 131,072 | Very large context! |
| deepseek-r1:14b | 32,000 | Thinking model |

---

## Token Budget Analysis

For a typical complex query (e.g., FANUC troubleshooting):

```
Input Context Breakdown:
- System prompt:        ~500 tokens
- User query:           ~50 tokens
- Scraped content:      ~15,000 tokens (5 sources × 3000 each)
- Domain knowledge:     ~3,000 tokens
- Scratchpad:           ~1,000 tokens
- Instructions:         ~500 tokens
---------------------------------
Total Input:            ~20,050 tokens

Available for Output:
- qwen3:8b context:     40,960 tokens
- After input:          ~20,900 tokens available
- Current max_tokens:   4,096 (artificially limiting)

Recommendation: max_tokens should be at least 8192 for synthesis
```

---

## Implementation Priority

1. **Immediate**: Update `llm_models.yaml` with recommended values
2. **Short-term**: Audit hardcoded values in code and replace with config lookups
3. **Medium-term**: Add dynamic max_tokens based on query complexity
4. **Long-term**: Implement token budget tracking per request

---

## Proposed Config Changes

```yaml
# llm_models.yaml changes

pipeline:
  synthesizer:
    max_tokens: 8192  # Increased from 4096

  thinking:
    max_tokens: 16384  # Increased from 8192

  planner:
    max_tokens: 2048  # Increased from 1024

  analyzer:
    max_tokens: 2048  # Increased from 1024

utility:
  reasoning_dag:
    max_tokens: 4096  # Increased from 2048

  information_bottleneck:
    max_tokens: 1024  # Increased from 512

  ragas_judge:
    max_tokens: 1024  # Increased from 512

  relevance_scorer:
    max_tokens: 768  # Increased from 512
```

---

## Appendix: Token Estimation Rules

- **1 token ≈ 4 characters** (English text)
- **1 token ≈ 0.75 words** (average)
- **JSON overhead**: +20-30% for structure
- **Thinking tokens**: 3-5x final output for reasoning models
- **Safety margin**: Always reserve 10-20% buffer

---

*Report generated by Claude Code audit tool*
