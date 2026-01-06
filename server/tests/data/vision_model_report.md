# Vision Model Effectiveness Report

> **Generated**: 2026-01-05T13:05:20.414734
> **Ollama Version**: 0.13.5
> **Test Duration**: 172.2s
> **Test Image**: `/home/sparkone/sdd/Recovery_Bot/memOS/server/tests/data/test_screenshot.png`

---

## Summary

| Metric | Value |
|--------|-------|
| Models Tested | 8/8 |
| Avg Latency | 4013ms |
| Avg JSON Parse Rate | 37.5% |
| Avg Content Richness | 0.31 |

---

## Results by Model

| Model | Family | Size | Context | Latency (avg) | JSON Rate | Richness | Status |
|-------|--------|------|---------|---------------|-----------|----------|--------|
| qwen2.5vl:7b | qwen2.5vl | 0.0GB | 125K📏 | 6772ms | 100% | 0.50 | PASS |
| granite3.2-vision:2b | granite3.2-vision | 0.0GB | 16K📏 | 2978ms | 100% | 0.35 | PASS |
| llama3.2-vision:11b | llama3.2-vision | 0.0GB | 128K📏 | 6963ms | 33% | 1.00 | PASS |
| qwen3-vl:2b | qwen3-vl | 0.0GB | 256K | 11584ms | 67% | 0.61 | PASS |
| llava:latest | llava | 0.0GB | 32K📏 | 0ms | 0% | 0.00 | FAIL |
| llava-llama3:8b | llava | 0.0GB | 8K📏 | 0ms | 0% | 0.00 | FAIL |
| minicpm-v:8b | minicpm-v | 0.0GB | 32K📏 | 0ms | 0% | 0.00 | FAIL |
| deepseek-ocr:3b | deepseek-ocr | 0.0GB | 8K📏 | 3804ms | 0% | 0.00 | FAIL |

---

## Top Performers

### Fastest (with >50% parse rate)

| Rank | Model | Latency | Parse Rate |
|------|-------|---------|------------|
| 1 | granite3.2-vision:2b | 2978ms | 100% |
| 2 | qwen2.5vl:7b | 6772ms | 100% |
| 3 | qwen3-vl:2b | 11584ms | 67% |

### Highest Quality (parse rate × richness)

| Rank | Model | Parse Rate | Richness | Score |
|------|-------|------------|----------|-------|
| 1 | qwen2.5vl:7b | 100% | 0.50 | 0.50 |
| 2 | qwen3-vl:2b | 67% | 0.61 | 0.40 |
| 3 | granite3.2-vision:2b | 100% | 0.35 | 0.35 |
| 4 | llama3.2-vision:11b | 33% | 1.00 | 0.33 |
| 5 | llava:latest | 0% | 0.00 | 0.00 |

---

## Family Comparison

| Family | Models Tested | Avg Latency | Avg Parse Rate | Avg Richness |
|--------|---------------|-------------|----------------|--------------|
| deepseek-ocr | 1 | 3804ms | 0% | 0.00 |
| granite3.2-vision | 1 | 2978ms | 100% | 0.35 |
| llama3.2-vision | 1 | 6963ms | 33% | 1.00 |
| llava | 2 | 0ms | 0% | 0.00 |
| minicpm-v | 1 | 0ms | 0% | 0.00 |
| qwen2.5vl | 1 | 6772ms | 100% | 0.50 |
| qwen3-vl | 1 | 11584ms | 67% | 0.61 |

---

## Errors

### llava:latest
- `general: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`
- `technical: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource li...`
- `contact: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`

### llava-llama3:8b
- `general: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`
- `technical: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource li...`
- `contact: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`

### minicpm-v:8b
- `general: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`
- `technical: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource li...`
- `contact: HTTP 500: {"error":"model runner has unexpectedly stopped, this may be due to resource limi...`


---

## Recommendations

- **Fastest**: `granite3.2-vision:2b` (2978ms)
- **Highest Quality**: `qwen2.5vl:7b` (100% parse, 0.50 richness)
- **Best <5GB**: `qwen2.5vl:7b` (0.0GB)

---

*Report generated on 2026-01-05T13:05:20.414734*