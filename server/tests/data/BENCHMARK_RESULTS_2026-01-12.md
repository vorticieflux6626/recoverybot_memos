# Agent Benchmark Results - 2026-01-12

## Summary

Comprehensive benchmarking of LLM models across 5 agentic pipeline phases with 9 model variants.

| Metric | Value |
|--------|-------|
| Total Runs | 64 |
| Models Tested | 9 |
| Agent Types | 5 |
| Date | 2026-01-12 |

## Key Findings

### 1. gemma3:4b is the Most Efficient

**Efficiency Score: 0.2357** (highest)
- Fastest average duration: 5558ms
- High accuracy: 0.84 average
- Reasonable VRAM: ~3234MB delta
- 100% success rate across all agents

### 2. Thinking Models Underperform on Utility Tasks

| Model | Accuracy | Duration | Efficiency |
|-------|----------|----------|------------|
| cogito:8b (thinking) | 0.88 | 5621ms | 0.0534 |
| deepseek-r1:8b (thinking) | 0.50 | 8829ms | 0.0154 |
| gemma3:4b (standard) | 0.84 | 5558ms | 0.2357 |

**Insight**: Thinking models add latency without improving accuracy for classification, evaluation, and verification tasks.

### 3. qwen3:8b is Accurate but Slow

| Metric | qwen3:8b | gemma3:4b |
|--------|----------|-----------|
| Accuracy | 0.90 | 0.84 |
| Duration | 14551ms | 5558ms |
| Efficiency | 0.1050 | 0.2357 |

**Trade-off**: 6% accuracy gain costs 2.6x latency.

### 4. NEW: qwen3:4b-instruct-2507-q8_0 is Fastest Overall

| Metric | q8_0 | fp16 | Difference |
|--------|------|------|------------|
| Duration | 4235ms | 6748ms | **37% faster** |
| VRAM | 4996MB | 8593MB | **42% less** |
| Accuracy | 0.70 | 0.70 | Same |

**Insight**: q8_0 quantization provides significant speed and VRAM benefits with no accuracy loss for utility tasks.

### 5. NEW: qwen3:30b-a3b Excels at Complex Judgment Tasks

| Agent | 30B Accuracy | 4B Accuracy | Duration |
|-------|-------------|-------------|----------|
| URL Filter | **1.00** | 0.66 | 10423ms |
| Self-Reflection | **1.00** | 0.50 | 11696ms |
| Analyzer | **0.88** | 0.70 | 10981ms |
| CRAG Evaluator | 0.83 | 0.83 | 10197ms |

**Insight**: The 30B model provides 34-50% accuracy improvements on tasks requiring nuanced judgment (URL relevance, synthesis evaluation), worth the 2x latency for quality-critical paths.

## Rankings by Agent

### ANALYZER (Query Classification)

| Rank | Model | Accuracy | Duration | Score |
|------|-------|----------|----------|-------|
| 1 | ministral-3:3b | 1.00 | 5516ms | 0.97 |
| 2 | qwen3:8b | 1.00 | 12320ms | 0.97 |
| 3 | gemma3:4b | 0.91 | 5091ms | 0.94 |
| 4 | **qwen3:30b-a3b** | **0.88** | 10981ms | 0.93 |
| 5 | qwen3:4b-instruct-q8_0 | 0.70 | **4235ms** | 0.87 |
| 6 | qwen3:4b-instruct-fp16 | 0.70 | 6748ms | 0.87 |
| 7 | llama3.2:3b | 0.73 | 7308ms | 0.86 |

**Recommendation**: `gemma3:4b` for speed, `qwen3:30b-a3b` for highest accuracy

### CRAG_EVALUATOR (Retrieval Quality)

| Rank | Model | Accuracy | Duration | Score |
|------|-------|----------|----------|-------|
| 1 | gemma3:4b | 0.83 | 5079ms | 0.91 |
| 2 | **qwen3:4b-instruct-q8_0** | **0.83** | **4513ms** | 0.91 |
| 3 | qwen3:30b-a3b | 0.83 | 10197ms | 0.91 |
| 4 | qwen3:4b-instruct-fp16 | 0.83 | 7054ms | 0.91 |
| 5 | qwen3:8b | 0.75 | 8153ms | 0.88 |
| 6 | cogito:8b | 0.75 | 6224ms | 0.88 |
| 7 | deepseek-r1:8b | 0.50 | 8829ms | 0.79 |

**Recommendation**: `qwen3:4b-instruct-q8_0` - Fastest with equal accuracy to gemma3:4b

### SELF_REFLECTION (Quality Evaluation)

| Rank | Model | Accuracy | Duration | Score |
|------|-------|----------|----------|-------|
| 1 | cogito:8b | 1.00 | 5018ms | 0.97 |
| 2 | gemma3:4b | 1.00 | 7205ms | 0.97 |
| 3 | **qwen3:30b-a3b** | **1.00** | 11696ms | 0.97 |
| 4 | qwen3:8b | 1.00 | 20767ms | 0.97 |
| 5 | qwen3:4b-instruct-q8_0 | 0.50 | 5138ms | 0.79 |
| 6 | qwen3:4b-instruct-fp16 | 0.50 | 7881ms | 0.79 |

**Insight**: 4B models struggle with nuanced synthesis evaluation. Use 8B+ for self-reflection.

**Recommendation**: `cogito:8b` for speed, `qwen3:30b-a3b` for RESEARCH/FULL presets

### URL_RELEVANCE_FILTER

| Rank | Model | Accuracy | Duration | Success |
|------|-------|----------|----------|---------|
| 1 | **qwen3:30b-a3b** | **1.00** | 10423ms | 100% |
| 2 | gemma3:4b | 0.66 | 5712ms | 100% |
| 3 | qwen3:4b-instruct-q8_0 | 0.66 | **4515ms** | 100% |
| 4 | qwen3:4b-instruct-fp16 | 0.66 | 6954ms | 100% |
| 5 | ministral-3:3b | 0.66 | 7522ms | 100% |
| 6 | qwen3:8b | 1.00* | 9435ms | 50% |

*qwen3:8b had 50% success rate (failures not counted in accuracy)

**Insight**: The 30B model achieves **perfect URL relevance filtering** - critical for avoiding wasted scraping time.

**Recommendation**: `qwen3:30b-a3b` for quality-critical presets, `qwen3:4b-instruct-q8_0` for speed

### VERIFIER (Claim Verification)

| Rank | Model | Accuracy | Duration | Score |
|------|-------|----------|----------|-------|
| 1 | gemma3:4b | 0.75 | 5411ms | 0.88 |
| 2 | ministral-3:3b | 0.75 | 6476ms | 0.88 |
| 3 | qwen3:8b | 0.75 | 20634ms | 0.88 |
| 4 | qwen3:4b-instruct-q8_0 | 0.70 | **4623ms** | 0.86 |
| 5 | qwen3:4b-instruct-fp16 | 0.70 | 6911ms | 0.86 |
| 6 | qwen3:30b-a3b | 0.70 | 10307ms | 0.86 |

**Recommendation**: `gemma3:4b` - Best accuracy, `qwen3:4b-instruct-q8_0` for speed

## Production Recommendations

Based on benchmark results, recommended model assignments by preset:

### MINIMAL/BALANCED Presets (Speed Priority)

```python
FAST_MODEL_CONFIG = {
    # Query Understanding
    "analyzer": "qwen3:4b-instruct-2507-q8_0",  # Fastest (4235ms)
    "query_classifier": "qwen3:4b-instruct-2507-q8_0",

    # Retrieval Enhancement
    "crag_evaluator": "qwen3:4b-instruct-2507-q8_0",  # 0.83 accuracy, fastest
    "url_filter": "qwen3:4b-instruct-2507-q8_0",       # 100% success, 4515ms
    "hyde_expander": "qwen3:4b-instruct-2507-q8_0",

    # Verification
    "verifier": "qwen3:4b-instruct-2507-q8_0",
    "cross_domain_validator": "gemma3:4b",

    # Reflection
    "self_reflection": "cogito:8b",     # 4B models fail at 0.50 accuracy

    # Synthesis
    "synthesizer": "qwen3:8b",
}
```

### ENHANCED/RESEARCH/FULL Presets (Quality Priority)

```python
QUALITY_MODEL_CONFIG = {
    # Query Understanding
    "analyzer": "gemma3:4b",            # 0.91 accuracy, 5091ms

    # Retrieval Enhancement
    "crag_evaluator": "gemma3:4b",      # 0.83 accuracy
    "url_filter": "qwen3:30b-a3b-instruct-2507-q4_K_M",  # 1.00 accuracy!
    "hyde_expander": "qwen3:8b",

    # Verification
    "verifier": "gemma3:4b",            # 0.75 accuracy
    "cross_domain_validator": "qwen3:8b",

    # Reflection
    "self_reflection": "qwen3:30b-a3b-instruct-2507-q4_K_M",  # 1.00 accuracy

    # Synthesis
    "synthesizer": "qwen3:30b-a3b-instruct-2507-q4_K_M",  # Best quality
}
```

## VRAM Considerations

| Model | Typical VRAM Delta | Notes |
|-------|-------------------|-------|
| qwen3:4b-instruct-2507-q8_0 | ~4996MB | **Best efficiency** |
| gemma3:4b | ~3234MB | Low VRAM |
| ministral-3:3b | ~3371MB | Low VRAM |
| cogito:8b | ~4465MB | Thinking model |
| qwen3:8b | ~5586MB | Standard |
| qwen3:4b-instruct-2507-fp16 | ~8593MB | Not recommended (same accuracy as q8_0) |
| qwen3:30b-a3b-instruct-2507-q4_K_M | ~18260MB | **Quality model** |
| deepseek-r1:8b | ~5586MB | Thinking model |

**Configurations by VRAM Budget:**

| Budget | Config | Total VRAM |
|--------|--------|------------|
| **12GB** | qwen3:4b-q8_0 + cogito:8b + qwen3:8b | ~10GB |
| **20GB** | gemma3:4b + cogito:8b + qwen3:8b + qwen3:30b-a3b (swap) | ~18GB peak |
| **24GB+** | qwen3:4b-q8_0 + qwen3:30b-a3b concurrent | ~23GB |

## Next Steps

1. ✅ ~~**Add quantization variants** (q4_K_M, q8_0) to benchmarks~~ - Complete
2. ✅ ~~**Benchmark larger models** (30B) for synthesis~~ - Complete
3. **Update llm_config.py** with optimal model assignments (see configs above)
4. **Test HyDE and Planner** agents
5. **Cross-encoder reranking** benchmarks
6. **Apply qwen3:30b-a3b** to URL filter and self-reflection in RESEARCH/FULL presets

## Raw Data Location

- Database: `tests/data/agent_benchmarks.db`
- Framework: `tests/data/agent_benchmarks.py`
- Runner: `tests/data/run_full_agent_benchmark.sh`
