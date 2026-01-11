# Vision Model Benchmark Analysis

**Date**: 2026-01-09 (Updated with Docling fix)
**Test Cases**: 4 (Simple Static, Technical Documentation, Wikipedia, Industrial Equipment)
**Models Tested**: 11 (8 VL models, Pandoc, Docling)

## Executive Summary

**Key Finding**: **Docling achieves 95% quality** at 2s latency, surpassing both Pandoc (74%) and VL models (64-68%). For general HTML extraction, use **Docling > Pandoc > qwen2.5vl**. VL models should be reserved for image-heavy pages.

## Rankings (Updated 2026-01-09)

| Rank | Model | Avg Duration | Quality | Success Rate | Use Case |
|------|-------|-------------|---------|--------------|----------|
| 1 | **docling** | 2,021ms | 95% | 100% | Best overall - tables, structure |
| 2 | **pandoc** | 742ms | 74% | 100% | Fastest - simple HTML |
| 3 | qwen2.5vl:7b | 52,678ms | 68% | 100% | VL fallback |
| 4 | qwen3-vl:2b | 5,229ms | 68% | 100% | Fast VL option |
| 5 | qwen3-vl:8b | 8,514ms | 68% | 100% | Medium VL |
| 6 | qwen2.5vl:7b-q8_0 | 54,789ms | 64% | 100% | Quality VL |
| 7 | qwen3-vl:4b | 6,432ms | 53% | 100% | Balanced VL |
| 8 | llama3.2-vision:11b | 106,006ms | 48% | 100% | Avoid - too slow |
| 9 | granite3.2-vision:2b | 18,081ms | 42% | 100% | VRAM-constrained |

*Note: minicpm-v and llava excluded due to 0% success rate*

## Key Insights

### 1. Docling is the New Champion
- **95% quality** with excellent table extraction (IBM TableFormer)
- **2 seconds** average latency - 3x slower than Pandoc but 26x faster than VL models
- **100% success rate** across all test cases
- Perfect for complex documents with tables, nested structures

### 2. Pandoc Remains Fastest
- **742ms** average - fastest extraction method
- **74% quality** - good for simple HTML
- Best for speed-critical applications where structure isn't complex

### 3. VL Models for Visual Content Only
- qwen2.5vl/qwen3-vl: 64-68% quality at 5-55s latency
- Reserve for: JavaScript-rendered pages, image-heavy content, screenshots
- **llama3.2-vision too slow** (106s) for practical use

### 4. Models to Avoid
- **llava:latest**: 0% success rate
- **minicpm-v (all variants)**: 0% success rate
- These fail to extract meaningful content from web pages

## Docling Test Results

| Test Case | Duration | Quality |
|-----------|----------|---------|
| Simple Static Page | 2017ms | 83% |
| Technical Documentation | 2016ms | **98%** |
| Wikipedia Article | 2023ms | **100%** |
| Industrial Equipment | 2029ms | **98%** |

**Notable**: Docling achieved **98-100% quality** on complex pages with tables and nested structure.

## Recommended Extraction Strategy

```
Primary: Docling (best quality/speed tradeoff)
├── Speed: 2s average
├── Quality: 95%
├── When: Any document with tables, technical docs, structured content
└── Features: IBM TableFormer, OCR, structure recognition

Secondary: Pandoc (fastest)
├── Speed: 742ms average
├── Quality: 74%
└── When: Simple HTML, speed-critical, no complex tables

Tertiary: qwen3-vl:2b (VL fallback)
├── Speed: 5s average
├── Quality: 68%
└── When: JavaScript-rendered, image-heavy pages

Emergency: granite3.2-vision:2b (VRAM-constrained)
├── Speed: 18s average
├── Quality: 42%
└── When: Limited GPU memory
```

## Docling Fix Applied

**Issue**: Request format was incorrect
- **Before (broken)**: `{"source": url, "options": {"to_format": "md"}}`
- **After (fixed)**: `{"sources": [{"url": url, "kind": "http"}], "options": {"to_formats": ["md"]}}`

**Location**: `tests/data/vision_benchmark.py:extract_with_docling()`

## Database Location

Results stored in: `tests/data/vision_benchmarks.db`

Tables:
- `vision_benchmark_runs` - Individual test results
- `vision_model_rankings` - Aggregated rankings

## Next Steps

1. [x] ~~Investigate Docling API compatibility~~ FIXED
2. [ ] Update VL scraper to use Docling as primary extractor
3. [ ] Implement fallback chain: Docling -> Pandoc -> VL model
4. [ ] Add content quality validation to trigger fallback
5. [ ] Remove non-working models (llava, minicpm) from available pool
