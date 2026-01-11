"""
Content Extractor with Intelligent Fallback Chain

Implements a multi-tier extraction strategy:
1. Direct Docling (URL fetch) - 2s, 95% quality - best for static HTML
2. Screenshot + Docling OCR - 24s, 95% quality - for JS-rendered pages
3. Screenshot + VL Model - 5-8s, 68% quality - for image-heavy pages

Based on benchmark results from 2026-01-09:
- Docling: 95% quality, 2s latency (tables, structure)
- Pandoc: 74% quality, 742ms latency (simple HTML)
- qwen3-vl:2b: 68% quality, 5s latency (fast VL)
- qwen2.5vl:7b: 68% quality, 53s latency (quality VL)
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# VRAM-Aware Semaphores for Parallel Scraping
# =============================================================================
# These module-level semaphores control concurrent access to VRAM-intensive operations.
# They are shared across all ContentExtractor instances to prevent VRAM exhaustion.
#
# Architecture:
#   CPU Queue (Semaphore 8): HTTP GET, Pandoc - 0 VRAM, high parallelism
#   GPU Queue (Semaphore 2): Docling OCR - ~4GB VRAM each
#   VL Queue (Semaphore 1): VL Models - 2-11GB VRAM, serialized for safety
#
# Total VRAM ceiling: ~15GB (1 VL + 2 Docling) or ~11GB (1 large VL model)

_cpu_semaphore: Optional[asyncio.Semaphore] = None
_gpu_semaphore: Optional[asyncio.Semaphore] = None
_vl_semaphore: Optional[asyncio.Semaphore] = None


def _get_cpu_semaphore() -> asyncio.Semaphore:
    """Get or create CPU semaphore (high parallelism for HTTP/Pandoc)."""
    global _cpu_semaphore
    if _cpu_semaphore is None:
        _cpu_semaphore = asyncio.Semaphore(8)
    return _cpu_semaphore


def _get_gpu_semaphore() -> asyncio.Semaphore:
    """Get or create GPU semaphore (limited parallelism for Docling OCR)."""
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(2)
    return _gpu_semaphore


def _get_vl_semaphore() -> asyncio.Semaphore:
    """Get or create VL semaphore (single-threaded for VL models)."""
    global _vl_semaphore
    if _vl_semaphore is None:
        _vl_semaphore = asyncio.Semaphore(1)
    return _vl_semaphore


class ExtractionMethod(Enum):
    """Extraction methods in order of preference."""
    DOCLING_URL = "docling_url"      # Direct URL fetch via Docling
    PANDOC = "pandoc"                 # Fast HTML conversion
    DOCLING_OCR = "docling_ocr"       # Screenshot + Docling OCR
    VL_MODEL = "vl_model"             # Screenshot + Vision-Language model


class ExtractionQuality(Enum):
    """Quality levels for extraction results."""
    EXCELLENT = "excellent"   # >90% keyword coverage, good structure
    GOOD = "good"             # >70% keyword coverage
    ACCEPTABLE = "acceptable" # >50% keyword coverage
    POOR = "poor"             # <50% keyword coverage
    FAILED = "failed"         # Extraction failed


@dataclass
class ExtractionResult:
    """Result of content extraction."""
    success: bool
    content: str
    method: ExtractionMethod
    quality: ExtractionQuality
    duration_ms: float
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    fallback_chain: List[str] = field(default_factory=list)  # Methods attempted


@dataclass
class ExtractorConfig:
    """Configuration for content extractor."""
    # Docling settings
    docling_url: str = "http://localhost:8003"
    docling_timeout: float = 120.0
    docling_ocr_engine: str = "easyocr"
    docling_table_mode: str = "accurate"

    # Pandoc settings (if available)
    pandoc_enabled: bool = True
    pandoc_timeout: float = 30.0

    # VL Model settings
    vl_model_primary: str = "qwen3-vl:2b"
    vl_model_fallback: str = "granite3.2-vision:2b"
    vl_timeout: float = 60.0
    ollama_url: str = "http://localhost:11434"

    # Screenshot settings
    viewport_width: int = 1280
    viewport_height: int = 1024
    screenshot_timeout: float = 15.0  # Reduced from 30s - most pages load in <10s

    # Quality thresholds
    min_content_length: int = 100  # Minimum chars to consider successful
    min_quality_for_acceptance: ExtractionQuality = ExtractionQuality.ACCEPTABLE

    # Fallback behavior
    enable_fallback: bool = True
    max_fallback_attempts: int = 3


class ContentExtractor:
    """
    Intelligent content extractor with multi-tier fallback chain.

    Extraction Strategy:
    1. Try Direct Docling (fastest, best for static HTML with tables)
    2. If fails/empty → Try Pandoc (very fast, good for simple HTML)
    3. If fails/empty → Try Screenshot + Docling OCR (slower, handles JS)
    4. If quality low → Try Screenshot + VL Model (best for complex visuals)

    Usage:
        extractor = ContentExtractor()
        result = await extractor.extract("https://example.com")

        if result.success:
            print(f"Extracted {result.char_count} chars via {result.method.value}")
            print(f"Quality: {result.quality.value}")
            print(f"Fallback chain: {result.fallback_chain}")
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._playwright = None
        self._browser = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is available."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.config.docling_timeout)
        return self._http_client

    async def _ensure_browser(self):
        """Ensure Playwright browser is available."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch()
                logger.info("Playwright browser initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright: {e}")
                raise
        return self._browser

    async def close(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def extract(
        self,
        url: str,
        keywords: Optional[List[str]] = None,
        force_method: Optional[ExtractionMethod] = None
    ) -> ExtractionResult:
        """
        Extract content from URL using intelligent fallback chain.

        Args:
            url: URL to extract content from
            keywords: Optional keywords to validate extraction quality
            force_method: Force a specific extraction method (skip fallback)

        Returns:
            ExtractionResult with content and metadata
        """
        fallback_chain = []

        if force_method:
            return await self._extract_with_method(url, force_method, keywords, fallback_chain)

        # Tier 1: Try Direct Docling (best for static HTML)
        result = await self._extract_with_method(
            url, ExtractionMethod.DOCLING_URL, keywords, fallback_chain
        )
        if self._is_acceptable(result):
            return result

        # Tier 2: Try Pandoc (fastest for simple HTML)
        if self.config.pandoc_enabled:
            result = await self._extract_with_method(
                url, ExtractionMethod.PANDOC, keywords, fallback_chain
            )
            if self._is_acceptable(result):
                return result

        if not self.config.enable_fallback:
            return result

        # Skip screenshot-based fallbacks if we got HTTP errors (403, 404, etc.)
        # These indicate the URL is inaccessible, so screenshots won't help
        if result.error and any(code in result.error for code in ["403", "404", "401", "500", "502", "503"]):
            logger.info(f"Skipping screenshot fallbacks due to HTTP error: {result.error[:50]}")
            return result

        # Tier 3: Try Screenshot + Docling OCR (handles JS-rendered)
        result = await self._extract_with_method(
            url, ExtractionMethod.DOCLING_OCR, keywords, fallback_chain
        )
        if self._is_acceptable(result):
            return result

        # Tier 4: Try Screenshot + VL Model (last resort)
        result = await self._extract_with_method(
            url, ExtractionMethod.VL_MODEL, keywords, fallback_chain
        )

        return result

    def _is_acceptable(self, result: ExtractionResult) -> bool:
        """Check if extraction result meets quality threshold."""
        if not result.success:
            return False
        if result.char_count < self.config.min_content_length:
            return False
        if result.quality == ExtractionQuality.FAILED:
            return False
        # Allow POOR quality to trigger fallback
        if result.quality == ExtractionQuality.POOR:
            return False
        return True

    async def _extract_with_method(
        self,
        url: str,
        method: ExtractionMethod,
        keywords: Optional[List[str]],
        fallback_chain: List[str]
    ) -> ExtractionResult:
        """Execute extraction with specific method.

        Uses VRAM-aware semaphores to control concurrent access:
        - CPU semaphore (8): DOCLING_URL, PANDOC - no VRAM usage
        - GPU semaphore (2): DOCLING_OCR - ~4GB VRAM each
        - VL semaphore (1): VL_MODEL - 2-11GB VRAM, serialized
        """
        fallback_chain.append(method.value)
        start_time = time.time()

        try:
            # Select appropriate semaphore based on VRAM requirements
            if method in (ExtractionMethod.DOCLING_URL, ExtractionMethod.PANDOC):
                semaphore = _get_cpu_semaphore()
            elif method == ExtractionMethod.DOCLING_OCR:
                semaphore = _get_gpu_semaphore()
            elif method == ExtractionMethod.VL_MODEL:
                semaphore = _get_vl_semaphore()
            else:
                raise ValueError(f"Unknown extraction method: {method}")

            # Execute extraction under the appropriate semaphore
            async with semaphore:
                if method == ExtractionMethod.DOCLING_URL:
                    content, metadata = await self._extract_docling_url(url)
                elif method == ExtractionMethod.PANDOC:
                    content, metadata = await self._extract_pandoc(url)
                elif method == ExtractionMethod.DOCLING_OCR:
                    content, metadata = await self._extract_docling_ocr(url)
                elif method == ExtractionMethod.VL_MODEL:
                    content, metadata = await self._extract_vl_model(url)

            duration_ms = (time.time() - start_time) * 1000
            quality = self._assess_quality(content, keywords)

            return ExtractionResult(
                success=True,
                content=content,
                method=method,
                quality=quality,
                duration_ms=duration_ms,
                char_count=len(content),
                metadata=metadata,
                fallback_chain=fallback_chain.copy()
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Extraction failed with {method.value}: {e}")
            return ExtractionResult(
                success=False,
                content="",
                method=method,
                quality=ExtractionQuality.FAILED,
                duration_ms=duration_ms,
                char_count=0,
                error=str(e),
                fallback_chain=fallback_chain.copy()
            )

    async def _extract_docling_url(self, url: str) -> Tuple[str, Dict]:
        """Extract via direct Docling URL fetch."""
        client = await self._ensure_http_client()

        payload = {
            "sources": [{"url": url, "kind": "http"}],
            "options": {
                "to_formats": ["md"],
                "do_ocr": True,
                "table_mode": self.config.docling_table_mode
            }
        }

        resp = await client.post(
            f"{self.config.docling_url}/v1/convert/source",
            json=payload,
            timeout=self.config.docling_timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Docling returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        content = data.get("document", {}).get("md_content", "")

        if not content:
            raise RuntimeError("Docling returned empty content")

        metadata = {
            "status": data.get("status"),
            "processing_time": data.get("processing_time"),
            "errors": data.get("errors", [])
        }

        return content, metadata

    async def _extract_pandoc(self, url: str) -> Tuple[str, Dict]:
        """Extract via Pandoc HTML-to-Markdown conversion."""
        import subprocess

        # Fetch HTML first
        client = await self._ensure_http_client()
        resp = await client.get(url, timeout=self.config.pandoc_timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch URL: {resp.status_code}")

        html_content = resp.text

        # Convert via Pandoc
        proc = await asyncio.create_subprocess_exec(
            "pandoc", "-f", "html", "-t", "markdown",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=html_content.encode()),
            timeout=self.config.pandoc_timeout
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Pandoc failed: {stderr.decode()[:200]}")

        content = stdout.decode()
        metadata = {"html_size": len(html_content)}

        return content, metadata

    async def _extract_docling_ocr(self, url: str) -> Tuple[str, Dict]:
        """Extract via Screenshot + Docling OCR."""
        # Take screenshot
        screenshot_bytes, screenshot_meta = await self._take_screenshot(url)

        # Send to Docling for OCR
        client = await self._ensure_http_client()
        b64_data = base64.b64encode(screenshot_bytes).decode("utf-8")

        payload = {
            "sources": [{
                "base64_string": b64_data,
                "filename": "screenshot.png",
                "kind": "file"
            }],
            "options": {
                "to_formats": ["md"],
                "do_ocr": True,
                "ocr_engine": self.config.docling_ocr_engine,
                "table_mode": self.config.docling_table_mode
            }
        }

        resp = await client.post(
            f"{self.config.docling_url}/v1/convert/source",
            json=payload,
            timeout=self.config.docling_timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Docling OCR returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        content = data.get("document", {}).get("md_content", "")

        # Filter out embedded base64 images from markdown
        lines = content.split("\n")
        filtered_lines = [l for l in lines if not l.startswith("![Image](data:")]
        content = "\n".join(filtered_lines)

        if not content.strip():
            raise RuntimeError("Docling OCR returned empty content after filtering")

        metadata = {
            "screenshot_size": len(screenshot_bytes),
            "page_title": screenshot_meta.get("title"),
            "ocr_engine": self.config.docling_ocr_engine
        }

        return content, metadata

    async def _extract_vl_model(self, url: str) -> Tuple[str, Dict]:
        """Extract via Screenshot + VL Model."""
        # Take screenshot
        screenshot_bytes, screenshot_meta = await self._take_screenshot(url)
        b64_data = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Try primary VL model
        content, model_used = await self._call_vl_model(
            b64_data,
            self.config.vl_model_primary
        )

        # Fallback to secondary if primary fails
        if not content and self.config.vl_model_fallback:
            content, model_used = await self._call_vl_model(
                b64_data,
                self.config.vl_model_fallback
            )

        if not content:
            raise RuntimeError("VL model extraction failed")

        metadata = {
            "model_used": model_used,
            "screenshot_size": len(screenshot_bytes),
            "page_title": screenshot_meta.get("title")
        }

        return content, metadata

    async def _take_screenshot(self, url: str) -> Tuple[bytes, Dict]:
        """Take full-page screenshot using Playwright."""
        browser = await self._ensure_browser()

        page = await browser.new_page(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            }
        )

        try:
            await page.goto(url, wait_until="networkidle", timeout=int(self.config.screenshot_timeout * 1000))

            # Scroll to trigger lazy loading
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)
            await page.evaluate("window.scrollTo(0, 0)")

            title = await page.title()
            screenshot = await page.screenshot(full_page=True)

            return screenshot, {"title": title}

        finally:
            await page.close()

    async def _call_vl_model(self, image_b64: str, model: str) -> Tuple[str, str]:
        """Call VL model via Ollama."""
        client = await self._ensure_http_client()

        prompt = """Extract all text content from this webpage screenshot.
Focus on:
- Main content and headings
- Navigation items
- Important data and tables
- Contact information

Return the extracted text in a clean, readable format."""

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }],
            "stream": False
        }

        try:
            resp = await client.post(
                f"{self.config.ollama_url}/api/chat",
                json=payload,
                timeout=self.config.vl_timeout
            )

            if resp.status_code != 200:
                logger.warning(f"VL model {model} returned {resp.status_code}")
                return "", model

            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return content, model

        except Exception as e:
            logger.warning(f"VL model {model} failed: {e}")
            return "", model

    def _assess_quality(self, content: str, keywords: Optional[List[str]] = None) -> ExtractionQuality:
        """Assess extraction quality based on content and keywords."""
        if not content or len(content) < 50:
            return ExtractionQuality.FAILED

        if keywords:
            content_lower = content.lower()
            found = sum(1 for k in keywords if k.lower() in content_lower)
            coverage = found / len(keywords)

            if coverage >= 0.9:
                return ExtractionQuality.EXCELLENT
            elif coverage >= 0.7:
                return ExtractionQuality.GOOD
            elif coverage >= 0.5:
                return ExtractionQuality.ACCEPTABLE
            else:
                return ExtractionQuality.POOR

        # Without keywords, assess based on content characteristics
        if len(content) > 5000:
            return ExtractionQuality.EXCELLENT
        elif len(content) > 1000:
            return ExtractionQuality.GOOD
        elif len(content) > 200:
            return ExtractionQuality.ACCEPTABLE
        else:
            return ExtractionQuality.POOR


# Singleton instance for convenience
_extractor_instance: Optional[ContentExtractor] = None


def get_content_extractor(config: Optional[ExtractorConfig] = None) -> ContentExtractor:
    """Get or create singleton ContentExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ContentExtractor(config)
    return _extractor_instance


async def extract_content(
    url: str,
    keywords: Optional[List[str]] = None,
    force_method: Optional[ExtractionMethod] = None
) -> ExtractionResult:
    """Convenience function for one-off extraction."""
    extractor = get_content_extractor()
    return await extractor.extract(url, keywords, force_method)
