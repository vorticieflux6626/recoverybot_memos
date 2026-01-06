# Vision Model Effectiveness Report

> **Generated**: 2026-01-05T12:52:40.737685
> **Ollama Version**: 0.13.5
> **Test Duration**: 33.4s
> **Test Image**: `/home/sparkone/sdd/Recovery_Bot/memOS/server/tests/data/test_screenshot.png`

---

## Summary

| Metric | Value |
|--------|-------|
| Models Tested | 1/1 |
| Avg Latency | 11049ms |
| Avg JSON Parse Rate | 66.7% |
| Avg Content Richness | 0.61 |

---

## Results by Model

| Model | Family | Size | Latency (avg) | JSON Rate | Richness | Status |
|-------|--------|------|---------------|-----------|----------|--------|
| qwen3-vl:2b | qwen3-vl | 0.0GB | 11049ms | 67% | 0.61 | PASS |

---

## Top Performers

### Fastest (with >50% parse rate)

| Rank | Model | Latency | Parse Rate |
|------|-------|---------|------------|
| 1 | qwen3-vl:2b | 11049ms | 67% |

### Highest Quality (parse rate × richness)

| Rank | Model | Parse Rate | Richness | Score |
|------|-------|------------|----------|-------|
| 1 | qwen3-vl:2b | 67% | 0.61 | 0.40 |

---

## Family Comparison

| Family | Models Tested | Avg Latency | Avg Parse Rate | Avg Richness |
|--------|---------------|-------------|----------------|--------------|
| qwen3-vl | 1 | 11049ms | 67% | 0.61 |

---

## Recommendations

- **Fastest**: `qwen3-vl:2b` (11049ms)
- **Highest Quality**: `qwen3-vl:2b` (67% parse, 0.61 richness)
- **Best <5GB**: `qwen3-vl:2b` (0.0GB)

---

*Report generated on 2026-01-05T12:52:40.737685*