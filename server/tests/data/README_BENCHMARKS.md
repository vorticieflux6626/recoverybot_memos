# LLM Benchmarking Suite

Comprehensive benchmarking framework for all LLM calls in the agentic pipeline.

## Overview

This suite benchmarks:
- **24+ agent types** across the agentic pipeline
- **30+ model variants** (different sizes, quantizations, thinking vs instruct)
- **Multiple quality metrics** per agent task
- **Performance metrics** (TTFS, duration, VRAM usage)

## Databases

| Database | Purpose | Tables |
|----------|---------|--------|
| `model_benchmarks.db` | General synthesis benchmarks | benchmark_runs, test_contexts, model_rankings |
| `vision_benchmarks.db` | VL model extraction benchmarks | vision_benchmark_runs, vision_test_cases, vision_model_rankings |
| `synthesis_contexts.db` | Synthesis test contexts | synthesis_contexts, synthesis_benchmarks |
| `agent_benchmarks.db` | Agent-specific benchmarks | agent_benchmark_runs, agent_test_contexts, agent_model_rankings |

## Agent Types Benchmarked

### Query Understanding Phase
- **Analyzer**: Query classification, topic extraction, complexity assessment
- **QueryClassifier**: Query type categorization (troubleshooting, conceptual, procedural)
- **Planner**: Query decomposition into sub-queries

### Retrieval Enhancement Phase
- **HyDE Expander**: Hypothetical document generation for better retrieval
- **URL Relevance Filter**: Filter irrelevant URLs before scraping
- **CRAG Evaluator**: Evaluate retrieval quality (correct/ambiguous/incorrect)

### Verification Phase
- **Verifier**: Extract and verify claims from content
- **Cross-Domain Validator**: Detect hallucinated cross-domain claims
- **Entity Grounding**: Verify entities against knowledge base

### Synthesis & Reflection Phase
- **Synthesizer**: Generate final response (benchmarked separately)
- **Self-Reflection**: ISREL/ISSUP/ISUSE evaluation
- **RAGAS Evaluator**: Alternative quality evaluation

### Learning Phase
- **Experience Distiller**: Extract templates from successful searches
- **Entropy Monitor**: Uncertainty estimation

## Model Variants

### Utility Models (Fast, 1-8B)
```
gemma3:1b, gemma3:4b, gemma3:4b-it-q4_K_M, gemma3:4b-it-q8_0
qwen3:1.7b, qwen3:4b, qwen3:8b, qwen3:8b-q4_K_M, qwen3:8b-q8_0
llama3.2:1b, llama3.2:3b
ministral-3:3b
phi4-mini:3.8b
```

### Thinking Models (Reasoning-focused)
```
deepseek-r1:1.5b, deepseek-r1:7b, deepseek-r1:8b, deepseek-r1:14b, deepseek-r1:32b
qwq:32b
cogito:3b, cogito:8b, cogito:14b
phi4-reasoning:14b
openthinker:7b, openthinker:32b
```

### Standard Models (General purpose)
```
qwen3:8b, qwen3:14b, qwen3:30b-a3b
gemma3:12b
llama3.3:70b
mistral:7b, mistral-nemo:12b
```

## Metrics Tracked

### Performance Metrics
| Metric | Description |
|--------|-------------|
| `total_duration_ms` | Total inference time |
| `ttfs_ms` | Time to first token (streaming latency) |
| `vram_before_mb` | GPU memory before loading model |
| `vram_after_mb` | GPU memory after inference |
| `vram_peak_mb` | Peak GPU memory usage |
| `input_tokens` | Prompt token count |
| `output_tokens` | Response token count |
| `thinking_tokens` | Tokens in thinking blocks (for reasoning models) |

### Quality Metrics
| Metric | Description |
|--------|-------------|
| `accuracy_score` | How correct was the output (0-1) |
| `completeness_score` | Did output cover all expected aspects (0-1) |
| `format_score` | Was output well-structured/parseable (0-1) |
| `efficiency_score` | Quality per time per VRAM |
| `composite_score` | Weighted average of all quality metrics |

## Usage

### Quick Test (2-3 models)
```bash
python tests/data/agent_benchmarks.py --agent analyzer --quick
```

### Specific Agent + Models
```bash
python tests/data/agent_benchmarks.py \
  --agent crag_evaluator \
  --models "gemma3:4b,qwen3:8b,deepseek-r1:8b"
```

### Full Benchmark Suite
```bash
./tests/data/run_full_agent_benchmark.sh fast    # Fast models only
./tests/data/run_full_agent_benchmark.sh medium  # + Medium models
./tests/data/run_full_agent_benchmark.sh full    # + Thinking models
```

### View Rankings
```bash
python tests/data/agent_benchmarks.py --rankings
python tests/data/agent_benchmarks.py --rankings-agent analyzer
```

### Synthesis Benchmarks
```bash
python tests/data/synthesis_model_benchmarker.py --rankings
python tests/data/synthesis_model_benchmarker.py --models "qwen3:8b,gemma3:12b"
```

### Vision Model Benchmarks
```bash
python tests/data/vision_benchmark.py --models qwen3-vl:2b --quick
```

## Test Contexts

Each agent has multiple test contexts with varying difficulty:

| Difficulty | Description |
|------------|-------------|
| `easy` | Simple, single-topic queries |
| `medium` | Multi-step, technical queries |
| `hard` | Complex, multi-domain queries |
| `expert` | Edge cases, ambiguous queries |

### Example Test Context (Analyzer)
```python
AgentTestContext(
    context_id="analyzer_fanuc_servo",
    agent_type="analyzer",
    name="FANUC Servo Error Analysis",
    difficulty="medium",
    query="FANUC R-30iB SRVO-062 alarm on J2 axis during welding cycle",
    expected_fields=["query_type", "requires_search", "complexity"],
    expected_values={"query_type": "troubleshooting"},
    validation_keywords=["FANUC", "servo", "alarm", "troubleshoot"]
)
```

## Current Results Summary

### Top Performers by Agent

**Analyzer** (Query Analysis):
- Best Speed: gemma3:4b (5091ms)
- Best Accuracy: qwen3:8b, ministral-3:3b (1.0)
- Best Overall: ministral-3:3b (0.97 score, fast)

**CRAG Evaluator** (Retrieval Quality):
- Best Speed: gemma3:4b (5079ms)
- Best Accuracy: gemma3:4b (0.83)
- Best Overall: gemma3:4b (0.91 score)

**Thinking Models**: Slower but not necessarily more accurate for utility tasks.

## Architecture

```
tests/data/
├── agent_benchmarks.py          # Main agent benchmark framework
├── model_benchmarks.py          # General model benchmarking
├── synthesis_model_benchmarker.py # Synthesis-specific benchmarks
├── vision_benchmark.py          # VL model benchmarks
├── run_full_agent_benchmark.sh  # Full suite runner
├── README_BENCHMARKS.md         # This file
└── *.db                         # SQLite result databases
```

## Adding New Benchmarks

### Add New Agent Type
1. Add to `AgentType` enum in `agent_benchmarks.py`
2. Define test contexts in `_define_test_contexts()`
3. Add prompt builder in `_build_agent_prompt()`
4. Map to model categories in `AGENT_MODEL_MAP`

### Add New Model Variant
1. Add to appropriate category in `MODEL_VARIANTS`
2. Run benchmark: `python agent_benchmarks.py --agent analyzer --models "new_model:tag"`

### Add New Test Context
1. Create `AgentTestContext` with expected outputs
2. Add to contexts list in `_define_test_contexts()`
