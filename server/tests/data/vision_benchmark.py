#!/usr/bin/env python3
"""
Vision Model Benchmarking Framework - Systematic testing of VL models for web scraping.

Extends the existing ModelBenchmark framework to support:
- Vision-language model comparison (qwen3-vl, llama3.2-vision, minicpm-v, etc.)
- Quantization level comparison (q4, q8, fp16, bf16)
- Alternative extraction methods (pandoc, docling)
- Screenshot-based content extraction quality

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/data/vision_benchmark.py

    # Or from Python:
    from tests.data.vision_benchmark import VisionBenchmark
    bench = VisionBenchmark()
    results = await bench.run_full_benchmark()
"""

import asyncio
import sqlite3
import json
import time
import subprocess
import base64
import tempfile
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import httpx

# Import Playwright for screenshots
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Install with: pip install playwright && playwright install chromium")


@dataclass
class VisionBenchmarkResult:
    """Result from a single vision benchmark run."""
    model: str
    operation: str  # vl_extraction, pandoc, docling
    quantization: str  # q4, q8, fp16, bf16, default, n/a
    model_size_gb: float

    # Timing metrics
    ttfs_ms: Optional[float]  # Time to first token (VL models only)
    total_duration_ms: float
    screenshot_time_ms: Optional[float]  # Time to capture screenshot

    # Resource metrics
    vram_before_mb: int
    vram_after_mb: int
    vram_peak_mb: int

    # Quality metrics
    success: bool
    extraction_length: int  # chars extracted
    content_coverage: float  # 0-1 how much expected content was found
    noise_ratio: float  # 0-1 ratio of irrelevant content
    structure_preserved: float  # 0-1 did it preserve headings/lists/tables
    overall_quality: float  # weighted average

    # Content
    source_url: str
    extracted_preview: str  # first 500 chars
    expected_keywords: List[str] = field(default_factory=list)
    found_keywords: List[str] = field(default_factory=list)

    # Metadata
    error_message: Optional[str] = None
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkTestCase:
    """A test case for vision benchmarking."""
    id: str
    name: str
    url: str
    expected_keywords: List[str]
    expected_structure: List[str]  # h1, h2, table, list, etc.
    difficulty: str  # easy, medium, hard
    category: str  # static, js_heavy, pdf, table_heavy, form
    notes: str = ""


# Default test cases covering different website types
DEFAULT_TEST_CASES = [
    BenchmarkTestCase(
        id="static_simple",
        name="Simple Static Page",
        url="https://example.com",
        expected_keywords=["example", "domain", "illustrative"],
        expected_structure=["h1", "p"],
        difficulty="easy",
        category="static"
    ),
    BenchmarkTestCase(
        id="docs_technical",
        name="Technical Documentation",
        url="https://docs.python.org/3/library/json.html",
        expected_keywords=["json", "encode", "decode", "dump", "load"],
        expected_structure=["h1", "h2", "code", "list"],
        difficulty="medium",
        category="static"
    ),
    BenchmarkTestCase(
        id="wiki_article",
        name="Wikipedia Article",
        url="https://en.wikipedia.org/wiki/Injection_molding",
        expected_keywords=["injection", "mold", "plastic", "polymer", "process"],
        expected_structure=["h1", "h2", "table", "list", "p"],
        difficulty="medium",
        category="static"
    ),
    BenchmarkTestCase(
        id="industrial_manual",
        name="Industrial Equipment Page",
        url="https://www.fanucamerica.com/products/robots/series",
        expected_keywords=["robot", "fanuc", "industrial", "automation"],
        expected_structure=["h1", "h2", "list"],
        difficulty="hard",
        category="js_heavy"
    ),
]


class VisionBenchmark:
    """Benchmark framework for vision model comparison."""

    VISION_MODELS = [
        # Small models (1-3GB)
        "qwen3-vl:2b",
        "granite3.2-vision:2b",
        "qwen3-vl:4b",

        # Medium models (4-6GB)
        "llava:latest",
        "minicpm-v:8b",
        "qwen2.5vl:7b",
        "qwen3-vl:8b",

        # Large models (7-12GB)
        "llama3.2-vision:11b",
        "llava:7b-v1.6-mistral-q8_0",

        # Quantization variants
        "granite3.2-vision:2b-q8_0",
        "granite3.2-vision:2b-fp16",
        "qwen2.5vl:7b-q8_0",
        "qwen2.5vl:7b-fp16",
        "minicpm-v:8b-2.6-q8_0",
        "minicpm-v:8b-2.6-fp16",
        "llama3.2-vision:11b-instruct-q8_0",
    ]

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "vision_benchmarks.db"
        self.db_path = Path(db_path)
        self.ollama_url = "http://localhost:11434"
        self.docling_url = "http://localhost:8003"
        self._init_db()

    def _init_db(self):
        """Initialize vision benchmark database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS vision_benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    quantization TEXT,
                    model_size_gb REAL,

                    -- Timing
                    ttfs_ms REAL,
                    total_duration_ms REAL NOT NULL,
                    screenshot_time_ms REAL,

                    -- Resources
                    vram_before_mb INTEGER,
                    vram_after_mb INTEGER,
                    vram_peak_mb INTEGER,

                    -- Quality
                    success INTEGER NOT NULL,
                    extraction_length INTEGER,
                    content_coverage REAL,
                    noise_ratio REAL,
                    structure_preserved REAL,
                    overall_quality REAL,

                    -- Content
                    source_url TEXT,
                    extracted_preview TEXT,
                    expected_keywords TEXT,
                    found_keywords TEXT,

                    -- Meta
                    error_message TEXT,
                    notes TEXT,
                    test_case_id TEXT
                );

                CREATE TABLE IF NOT EXISTS vision_test_cases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    expected_keywords TEXT,
                    expected_structure TEXT,
                    difficulty TEXT,
                    category TEXT,
                    notes TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS vision_model_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    updated_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    quantization TEXT,
                    avg_duration_ms REAL,
                    avg_ttfs_ms REAL,
                    avg_vram_mb REAL,
                    avg_quality REAL,
                    success_rate REAL,
                    run_count INTEGER,
                    efficiency_score REAL,
                    rank INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_vision_runs_model ON vision_benchmark_runs(model);
                CREATE INDEX IF NOT EXISTS idx_vision_runs_timestamp ON vision_benchmark_runs(timestamp);
            """)

    def get_gpu_stats(self) -> Dict[str, int]:
        """Get current GPU memory stats."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "used_mb": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2])
                }
        except Exception:
            pass
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

    def parse_model_info(self, model_name: str) -> Tuple[str, float]:
        """Parse model name to extract quantization and size estimate."""
        quant_patterns = ['q8_0', 'q4_k_m', 'q4_0', 'q5_k_m', 'q6_k', 'fp16', 'bf16', 'f16']
        quantization = "default"
        for q in quant_patterns:
            if q in model_name.lower():
                quantization = q.upper()
                break

        # Get actual size from Ollama
        size_gb = 0.0
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.ollama_url}/api/show", "-d", json.dumps({"name": model_name})],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                size_gb = data.get("size", 0) / (1024**3)
        except:
            pass

        return quantization, size_gb

    async def capture_screenshot(self, url: str, width: int = 1280, height: int = 800) -> Tuple[bytes, float]:
        """Capture screenshot of URL using Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")

        start = time.time()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": width, "height": height})

            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                # Scroll to load lazy content
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)
                await page.evaluate("window.scrollTo(0, 0)")

                screenshot = await page.screenshot(full_page=True, type="png")
            finally:
                await browser.close()

        duration_ms = (time.time() - start) * 1000
        return screenshot, duration_ms

    async def extract_with_vl_model(
        self,
        model: str,
        screenshot: bytes,
        prompt: str = "Extract all text content from this screenshot. Preserve structure (headings, lists, tables)."
    ) -> Tuple[str, float, float, int, int]:
        """Extract content using vision-language model."""

        vram_before = self.get_gpu_stats()["used_mb"]
        start = time.time()
        ttfs_ms = None
        output_text = ""

        # Encode image as base64
        image_b64 = base64.b64encode(screenshot).decode("utf-8")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": True,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 4096
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if first_token_time is None and data.get("response"):
                                    first_token_time = time.time()
                                    ttfs_ms = (first_token_time - start) * 1000

                                output_text += data.get("response", "")
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            raise RuntimeError(f"VL extraction failed: {e}")

        total_duration_ms = (time.time() - start) * 1000
        vram_after = self.get_gpu_stats()["used_mb"]

        return output_text, total_duration_ms, ttfs_ms, vram_before, vram_after

    async def extract_with_pandoc(self, url: str) -> Tuple[str, float]:
        """Extract content using pandoc (fetch HTML first, then convert)."""
        start = time.time()

        try:
            # Fetch HTML
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, follow_redirects=True)
                html = resp.text

            # Convert with pandoc
            result = subprocess.run(
                ["pandoc", "-f", "html", "-t", "markdown", "--wrap=none"],
                input=html,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"Pandoc failed: {result.stderr}")

            output_text = result.stdout

        except Exception as e:
            raise RuntimeError(f"Pandoc extraction failed: {e}")

        duration_ms = (time.time() - start) * 1000
        return output_text, duration_ms

    async def extract_with_docling(self, url: str) -> Tuple[str, float]:
        """Extract content using Docling.

        Request format per Docling OpenAPI spec:
        {
            "sources": [{"url": "...", "kind": "http"}],
            "options": {"to_formats": ["md"]}
        }
        """
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Correct request format per Docling OpenAPI spec
                payload = {
                    "sources": [{"url": url, "kind": "http"}],
                    "options": {
                        "to_formats": ["md"],
                        "do_ocr": True,
                        "table_mode": "accurate"  # Use IBM TableFormer for better table extraction
                    }
                }

                resp = await client.post(
                    f"{self.docling_url}/v1/convert/source",
                    json=payload
                )

                if resp.status_code != 200:
                    error_detail = resp.text[:200] if resp.text else "No details"
                    raise RuntimeError(f"Docling returned {resp.status_code}: {error_detail}")

                data = resp.json()
                # Response format: {"document": {"filename": "...", "md_content": "..."}, "status": "success"}
                output_text = data.get("document", {}).get("md_content", "")

                if not output_text and data.get("errors"):
                    raise RuntimeError(f"Docling errors: {data.get('errors')}")

        except Exception as e:
            raise RuntimeError(f"Docling extraction failed: {e}")

        duration_ms = (time.time() - start) * 1000
        return output_text, duration_ms

    def calculate_quality_metrics(
        self,
        extracted: str,
        expected_keywords: List[str],
        expected_structure: List[str]
    ) -> Tuple[float, float, float, float, List[str]]:
        """Calculate quality metrics for extracted content."""
        extracted_lower = extracted.lower()

        # Content coverage - what keywords were found
        found_keywords = [kw for kw in expected_keywords if kw.lower() in extracted_lower]
        content_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0

        # Noise ratio - estimate based on common noise patterns
        noise_patterns = [
            r"cookie", r"privacy policy", r"terms of service",
            r"subscribe", r"newsletter", r"advertisement",
            r"javascript", r"enable javascript", r"browser"
        ]
        noise_count = sum(1 for p in noise_patterns if re.search(p, extracted_lower))
        noise_ratio = min(1.0, noise_count / 10)

        # Structure preservation
        structure_markers = {
            "h1": r"^#\s",
            "h2": r"^##\s",
            "h3": r"^###\s",
            "list": r"^[-*]\s|^\d+\.\s",
            "table": r"\|.*\|",
            "code": r"```|`[^`]+`",
            "p": r"\n\n"
        }
        found_structures = 0
        for struct in expected_structure:
            if struct in structure_markers:
                if re.search(structure_markers[struct], extracted, re.MULTILINE):
                    found_structures += 1
        structure_preserved = found_structures / len(expected_structure) if expected_structure else 0.5

        # Overall quality (weighted)
        overall_quality = (
            content_coverage * 0.5 +
            (1 - noise_ratio) * 0.2 +
            structure_preserved * 0.3
        )

        return content_coverage, noise_ratio, structure_preserved, overall_quality, found_keywords

    async def benchmark_model(
        self,
        model: str,
        test_case: BenchmarkTestCase,
        unload_first: bool = True
    ) -> VisionBenchmarkResult:
        """Run benchmark for a single VL model on a test case."""

        print(f"  Testing {model} on {test_case.name}...")

        # Optionally unload models first
        if unload_first:
            await self.unload_all_models()
            await asyncio.sleep(2)

        quantization, model_size_gb = self.parse_model_info(model)
        vram_before = self.get_gpu_stats()["used_mb"]

        error_msg = None
        extracted = ""
        screenshot_time_ms = None
        ttfs_ms = None
        total_duration_ms = 0
        vram_after = vram_before

        try:
            # Capture screenshot
            screenshot, screenshot_time_ms = await self.capture_screenshot(test_case.url)

            # Extract with VL model
            extracted, total_duration_ms, ttfs_ms, _, vram_after = await self.extract_with_vl_model(
                model, screenshot
            )

        except Exception as e:
            error_msg = str(e)

        # Calculate quality metrics
        content_coverage, noise_ratio, structure_preserved, overall_quality, found_keywords = \
            self.calculate_quality_metrics(extracted, test_case.expected_keywords, test_case.expected_structure)

        return VisionBenchmarkResult(
            model=model,
            operation="vl_extraction",
            quantization=quantization,
            model_size_gb=model_size_gb,
            ttfs_ms=ttfs_ms,
            total_duration_ms=total_duration_ms,
            screenshot_time_ms=screenshot_time_ms,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
            vram_peak_mb=max(vram_before, vram_after),
            success=len(extracted) > 100 and error_msg is None,
            extraction_length=len(extracted),
            content_coverage=content_coverage,
            noise_ratio=noise_ratio,
            structure_preserved=structure_preserved,
            overall_quality=overall_quality,
            source_url=test_case.url,
            extracted_preview=extracted[:500] if extracted else "",
            expected_keywords=test_case.expected_keywords,
            found_keywords=found_keywords,
            error_message=error_msg,
            notes=f"Difficulty: {test_case.difficulty}, Category: {test_case.category}"
        )

    async def benchmark_pandoc(self, test_case: BenchmarkTestCase) -> VisionBenchmarkResult:
        """Benchmark pandoc extraction."""
        print(f"  Testing pandoc on {test_case.name}...")

        error_msg = None
        extracted = ""
        total_duration_ms = 0

        try:
            extracted, total_duration_ms = await self.extract_with_pandoc(test_case.url)
        except Exception as e:
            error_msg = str(e)

        content_coverage, noise_ratio, structure_preserved, overall_quality, found_keywords = \
            self.calculate_quality_metrics(extracted, test_case.expected_keywords, test_case.expected_structure)

        return VisionBenchmarkResult(
            model="pandoc",
            operation="pandoc",
            quantization="n/a",
            model_size_gb=0,
            ttfs_ms=None,
            total_duration_ms=total_duration_ms,
            screenshot_time_ms=None,
            vram_before_mb=0,
            vram_after_mb=0,
            vram_peak_mb=0,
            success=len(extracted) > 100 and error_msg is None,
            extraction_length=len(extracted),
            content_coverage=content_coverage,
            noise_ratio=noise_ratio,
            structure_preserved=structure_preserved,
            overall_quality=overall_quality,
            source_url=test_case.url,
            extracted_preview=extracted[:500] if extracted else "",
            expected_keywords=test_case.expected_keywords,
            found_keywords=found_keywords,
            error_message=error_msg,
            notes=f"Difficulty: {test_case.difficulty}, Category: {test_case.category}"
        )

    async def benchmark_docling(self, test_case: BenchmarkTestCase) -> VisionBenchmarkResult:
        """Benchmark Docling extraction."""
        print(f"  Testing docling on {test_case.name}...")

        error_msg = None
        extracted = ""
        total_duration_ms = 0

        try:
            extracted, total_duration_ms = await self.extract_with_docling(test_case.url)
        except Exception as e:
            error_msg = str(e)

        content_coverage, noise_ratio, structure_preserved, overall_quality, found_keywords = \
            self.calculate_quality_metrics(extracted, test_case.expected_keywords, test_case.expected_structure)

        return VisionBenchmarkResult(
            model="docling",
            operation="docling",
            quantization="n/a",
            model_size_gb=0,
            ttfs_ms=None,
            total_duration_ms=total_duration_ms,
            screenshot_time_ms=None,
            vram_before_mb=0,
            vram_after_mb=0,
            vram_peak_mb=0,
            success=len(extracted) > 100 and error_msg is None,
            extraction_length=len(extracted),
            content_coverage=content_coverage,
            noise_ratio=noise_ratio,
            structure_preserved=structure_preserved,
            overall_quality=overall_quality,
            source_url=test_case.url,
            extracted_preview=extracted[:500] if extracted else "",
            expected_keywords=test_case.expected_keywords,
            found_keywords=found_keywords,
            error_message=error_msg,
            notes=f"Difficulty: {test_case.difficulty}, Category: {test_case.category}"
        )

    async def unload_all_models(self):
        """Unload all models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={"model": model["name"], "keep_alive": 0}
                        )
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Warning: Could not unload models: {e}")

    def save_result(self, result: VisionBenchmarkResult, test_case_id: str):
        """Save benchmark result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO vision_benchmark_runs (
                    timestamp, model, operation, quantization, model_size_gb,
                    ttfs_ms, total_duration_ms, screenshot_time_ms,
                    vram_before_mb, vram_after_mb, vram_peak_mb,
                    success, extraction_length, content_coverage, noise_ratio,
                    structure_preserved, overall_quality,
                    source_url, extracted_preview, expected_keywords, found_keywords,
                    error_message, notes, test_case_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp, result.model, result.operation, result.quantization,
                result.model_size_gb, result.ttfs_ms, result.total_duration_ms,
                result.screenshot_time_ms, result.vram_before_mb, result.vram_after_mb,
                result.vram_peak_mb, int(result.success), result.extraction_length,
                result.content_coverage, result.noise_ratio, result.structure_preserved,
                result.overall_quality, result.source_url, result.extracted_preview,
                json.dumps(result.expected_keywords), json.dumps(result.found_keywords),
                result.error_message, result.notes, test_case_id
            ))

    async def run_full_benchmark(
        self,
        models: Optional[List[str]] = None,
        test_cases: Optional[List[BenchmarkTestCase]] = None,
        include_pandoc: bool = True,
        include_docling: bool = True
    ) -> List[VisionBenchmarkResult]:
        """Run full benchmark suite."""

        if models is None:
            # Filter to available models
            models = await self.get_available_vision_models()

        if test_cases is None:
            test_cases = DEFAULT_TEST_CASES

        results = []
        total = len(models) * len(test_cases)
        if include_pandoc:
            total += len(test_cases)
        if include_docling:
            total += len(test_cases)

        print(f"Running vision benchmark: {len(models)} models x {len(test_cases)} test cases")
        print(f"Including pandoc: {include_pandoc}, docling: {include_docling}")
        print(f"Total benchmarks: {total}")
        print("=" * 60)

        # Benchmark VL models
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Model: {model}")
            for tc in test_cases:
                try:
                    result = await self.benchmark_model(model, tc)
                    results.append(result)
                    self.save_result(result, tc.id)
                    status = "✓" if result.success else "✗"
                    print(f"    {status} {tc.name}: {result.total_duration_ms:.0f}ms, quality={result.overall_quality:.2f}")
                except Exception as e:
                    print(f"    ✗ {tc.name}: Error - {e}")

        # Benchmark pandoc
        if include_pandoc:
            print(f"\n[Pandoc]")
            for tc in test_cases:
                try:
                    result = await self.benchmark_pandoc(tc)
                    results.append(result)
                    self.save_result(result, tc.id)
                    status = "✓" if result.success else "✗"
                    print(f"    {status} {tc.name}: {result.total_duration_ms:.0f}ms, quality={result.overall_quality:.2f}")
                except Exception as e:
                    print(f"    ✗ {tc.name}: Error - {e}")

        # Benchmark docling
        if include_docling:
            print(f"\n[Docling]")
            for tc in test_cases:
                try:
                    result = await self.benchmark_docling(tc)
                    results.append(result)
                    self.save_result(result, tc.id)
                    status = "✓" if result.success else "✗"
                    print(f"    {status} {tc.name}: {result.total_duration_ms:.0f}ms, quality={result.overall_quality:.2f}")
                except Exception as e:
                    print(f"    ✗ {tc.name}: Error - {e}")

        # Update rankings
        self.update_rankings()

        print("\n" + "=" * 60)
        print("Benchmark complete! Results saved to vision_benchmarks.db")

        return results

    async def get_available_vision_models(self) -> List[str]:
        """Get list of available vision models from Ollama."""
        available = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    all_models = [m["name"] for m in data.get("models", [])]

                    vision_keywords = ['vision', 'vl', 'llava', 'bakllava', 'moondream',
                                      'minicpm-v', 'granite3.2-vision']
                    for m in all_models:
                        if any(v in m.lower() for v in vision_keywords):
                            available.append(m)
        except Exception as e:
            print(f"Warning: Could not get model list: {e}")

        return sorted(available)

    def update_rankings(self):
        """Update model rankings based on benchmark results."""
        with sqlite3.connect(self.db_path) as conn:
            # Clear old rankings
            conn.execute("DELETE FROM vision_model_rankings")

            # Calculate rankings
            results = conn.execute("""
                SELECT
                    model,
                    quantization,
                    AVG(total_duration_ms) as avg_duration,
                    AVG(ttfs_ms) as avg_ttfs,
                    AVG(vram_after_mb - vram_before_mb) as avg_vram_delta,
                    AVG(overall_quality) as avg_quality,
                    SUM(success) * 1.0 / COUNT(*) as success_rate,
                    COUNT(*) as run_count
                FROM vision_benchmark_runs
                GROUP BY model, quantization
                ORDER BY avg_quality DESC, avg_duration ASC
            """).fetchall()

            for rank, row in enumerate(results, 1):
                model, quant, avg_dur, avg_ttfs, avg_vram, avg_qual, success_rate, run_count = row

                # Efficiency score: quality per second per GB VRAM
                efficiency = 0
                if avg_dur and avg_vram and avg_vram > 0:
                    efficiency = (avg_qual * 1000000) / (avg_dur * max(1, avg_vram))

                conn.execute("""
                    INSERT INTO vision_model_rankings (
                        updated_at, model, quantization, avg_duration_ms, avg_ttfs_ms,
                        avg_vram_mb, avg_quality, success_rate, run_count, efficiency_score, rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(), model, quant, avg_dur, avg_ttfs,
                    avg_vram, avg_qual, success_rate, run_count, efficiency, rank
                ))

    def print_rankings(self):
        """Print current model rankings."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT rank, model, quantization, avg_duration_ms, avg_quality,
                       success_rate, efficiency_score, run_count
                FROM vision_model_rankings
                ORDER BY rank
            """).fetchall()

        print("\n" + "=" * 80)
        print("VISION MODEL RANKINGS")
        print("=" * 80)
        print(f"{'Rank':<5} {'Model':<35} {'Quant':<8} {'Duration':<10} {'Quality':<8} {'Success':<8} {'Runs':<5}")
        print("-" * 80)

        for row in results:
            rank, model, quant, dur, qual, success, eff, runs = row
            print(f"{rank:<5} {model:<35} {quant or 'def':<8} {dur:>8.0f}ms {qual:>7.2f} {success:>7.1%} {runs:>5}")


async def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision Model Benchmark")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer models")
    parser.add_argument("--no-pandoc", action="store_true", help="Skip pandoc benchmark")
    parser.add_argument("--no-docling", action="store_true", help="Skip docling benchmark")
    parser.add_argument("--rankings", action="store_true", help="Just print rankings")
    args = parser.parse_args()

    bench = VisionBenchmark()

    if args.rankings:
        bench.print_rankings()
        return

    models = args.models
    if args.quick and not models:
        # Quick test with small models
        models = ["qwen3-vl:2b", "granite3.2-vision:2b", "llava:latest"]

    results = await bench.run_full_benchmark(
        models=models,
        include_pandoc=not args.no_pandoc,
        include_docling=not args.no_docling
    )

    bench.print_rankings()


if __name__ == "__main__":
    asyncio.run(main())
