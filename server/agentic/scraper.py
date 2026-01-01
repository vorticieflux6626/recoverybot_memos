"""
Content Scraper for Agentic Search

Fetches and extracts content from discovered URLs including:
- HTML pages (with readability extraction)
- PDF documents (text extraction)
- Technical documentation

Provides clean text for deep analysis by reasoning models.

Phase 2 Optimization:
- Content hash cache to avoid re-scraping (saves ~10s per cached URL)
- Integrated with agentic/content_cache.py
"""

import asyncio
import base64
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import io

import httpx
import time

from .content_cache import get_content_cache
from .metrics import get_performance_metrics
from .context_limits import get_model_context_window
from .search_metrics import get_search_metrics

# VL Scraper for JS-heavy pages (lazy import to avoid circular dependencies)
_vl_scraper_instance = None

def get_vl_scraper():
    """Get or create VLScraper instance (lazy loaded)."""
    global _vl_scraper_instance
    if _vl_scraper_instance is None:
        try:
            from services.vl_scraper import VLScraper
            _vl_scraper_instance = VLScraper()
            logger.info("VLScraper initialized for JS-heavy page extraction")
        except ImportError as e:
            logger.warning(f"VLScraper not available: {e}")
            _vl_scraper_instance = False  # Mark as unavailable
    return _vl_scraper_instance if _vl_scraper_instance else None


# Domains known to require JavaScript rendering
JS_HEAVY_DOMAINS = {
    # Single-page applications
    "angular.io", "react.dev", "vuejs.org",
    # Dynamic content sites
    "twitter.com", "x.com", "linkedin.com", "facebook.com",
    # E-commerce with heavy JS
    "amazon.com", "ebay.com", "alibaba.com",
    # Industrial manufacturer portals (often use React/Angular)
    "rockwellautomation.com", "siemens.com", "fanucamerica.com",
    # Forums with infinite scroll
    "reddit.com", "quora.com",
}

# Optional image handling imports
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger("agentic.scraper")


class ContentScraper:
    """
    Scrapes and extracts content from web pages and documents.

    Features:
    - HTML content extraction with boilerplate removal
    - PDF text extraction
    - Content chunking for LLM context limits
    - Parallel fetching with rate limiting
    """

    # Max content length per page (chars)
    MAX_CONTENT_LENGTH = 50000

    # Timeout for fetching
    FETCH_TIMEOUT = 30.0

    # User agent for requests
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    # JS detection patterns in HTML content
    JS_FRAMEWORK_PATTERNS = [
        r'<script[^>]*\bsrc=["\'][^"\']*(?:react|angular|vue|next|nuxt)',
        r'<script[^>]*>.*(?:React\.createElement|angular\.module|Vue\.createApp)',
        r'id=["\'](?:root|app|__next)["\']',  # Common SPA mount points
        r'<noscript>.*(?:JavaScript|enable|required)',  # Noscript warnings
    ]

    def __init__(self, enable_vl_scraper: bool = True):
        """
        Initialize ContentScraper.

        Args:
            enable_vl_scraper: Whether to use VL scraper for JS-heavy pages (default True)
        """
        self.session: Optional[httpx.AsyncClient] = None
        self.enable_vl_scraper = enable_vl_scraper
        self._vl_scraper = None  # Lazy loaded

    def _get_vl_scraper_instance(self):
        """Get VL scraper instance (lazy loaded)."""
        if self._vl_scraper is None and self.enable_vl_scraper:
            self._vl_scraper = get_vl_scraper()
        return self._vl_scraper

    def _is_js_heavy_page(self, url: str, html: Optional[str] = None) -> bool:
        """
        Detect if a page requires JavaScript rendering for proper extraction.

        Args:
            url: The URL being scraped
            html: Optional HTML content to analyze

        Returns:
            True if the page likely requires JS rendering
        """
        domain = urlparse(url).netloc.lower().replace("www.", "")

        # Check known JS-heavy domains
        for js_domain in JS_HEAVY_DOMAINS:
            if js_domain in domain:
                logger.debug(f"Domain {domain} is in JS_HEAVY_DOMAINS")
                return True

        # If we have HTML content, analyze it for JS frameworks
        if html:
            # Check for minimal content (possible JS-rendered page)
            visible_text = re.sub(r'<[^>]+>', '', html)
            visible_text = re.sub(r'\s+', ' ', visible_text).strip()

            # Very little visible text suggests JS-rendered content
            if len(visible_text) < 500 and len(html) > 5000:
                logger.debug(f"Minimal visible text ({len(visible_text)} chars) suggests JS rendering")
                return True

            # Check for JS framework patterns
            for pattern in self.JS_FRAMEWORK_PATTERNS:
                if re.search(pattern, html, re.IGNORECASE | re.DOTALL):
                    logger.debug(f"JS framework pattern detected in HTML")
                    return True

        return False

    async def _scrape_with_vl(self, url: str) -> Dict[str, Any]:
        """
        Scrape a URL using the VL (Vision-Language) scraper.

        Uses Playwright to capture screenshots and VL models to extract content.
        """
        vl_scraper = self._get_vl_scraper_instance()
        if not vl_scraper:
            logger.warning("VL scraper not available, falling back to standard extraction")
            return None

        try:
            logger.info(f"Using VL scraper for JS-heavy page: {url[:60]}...")

            # Use the VL scraper's scrape method
            result = await vl_scraper.scrape(url)

            if result and result.success:
                return {
                    "url": url,
                    "title": result.title or "",
                    "content": result.content or "",
                    "content_type": "vl_extracted",
                    "success": True,
                    "error": None,
                    "extraction_type": result.extraction_type.value if result.extraction_type else "general",
                    "vl_model": result.model_used
                }
            else:
                logger.warning(f"VL scraper failed for {url}: {result.error if result else 'No result'}")
                return None

        except Exception as e:
            logger.error(f"VL scraper error for {url}: {e}")
            return None

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session"""
        if self.session is None or self.session.is_closed:
            self.session = httpx.AsyncClient(
                timeout=self.FETCH_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": self.USER_AGENT}
            )
        return self.session

    async def scrape_urls(
        self,
        urls: List[str],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Scrape content from multiple URLs concurrently.

        Returns list of:
        {
            "url": str,
            "title": str,
            "content": str,
            "content_type": str,  # html, pdf, error
            "success": bool,
            "error": Optional[str]
        }
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_limit(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.scrape_url(url)

        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                scraped.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "content_type": "error",
                    "success": False,
                    "error": str(result)
                })
            else:
                scraped.append(result)

        return scraped

    async def scrape_url(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scrape content from a single URL.

        Args:
            url: URL to scrape
            use_cache: Whether to check/update cache (default True)

        Returns:
            Dict with url, title, content, content_type, success, error
        """
        metrics = get_search_metrics()
        domain = urlparse(url).netloc.replace("www.", "")
        start_time = time.time()

        # Check if domain should be skipped based on failure history
        should_skip, skip_reason = metrics.should_skip_domain(domain)
        if should_skip:
            logger.info(f"Skipping {domain}: {skip_reason}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_type": "skipped",
                "success": False,
                "error": f"Domain skipped: {skip_reason}"
            }

        # Phase 2 Optimization: Check cache first
        if use_cache:
            cache = get_content_cache()
            cached = cache.get_content(url)
            if cached:
                logger.info(f"Cache hit for {url[:60]}...")
                return cached

        # Phase K.1: Check if domain is known to be JS-heavy (VL scraper first)
        if self.enable_vl_scraper and self._is_js_heavy_page(url):
            vl_result = await self._scrape_with_vl(url)
            if vl_result:
                # Record metrics and cache
                duration_ms = (time.time() - start_time) * 1000
                content_length = len(vl_result.get("content", ""))
                metrics.record_scrape(
                    domain=domain,
                    success=True,
                    content_length=content_length,
                    duration_ms=duration_ms
                )
                if use_cache:
                    cache = get_content_cache()
                    cache.set_content(
                        url=url,
                        title=vl_result.get("title", ""),
                        content=vl_result.get("content", ""),
                        content_type=vl_result.get("content_type", "vl_extracted"),
                        success=True,
                        error=None
                    )
                return vl_result
            # If VL scraper failed, fall through to standard extraction

        try:
            session = await self._get_session()

            # HEAD request first to check content type
            try:
                head_response = await session.head(url, timeout=10.0)
                content_type = head_response.headers.get("content-type", "").lower()
            except (httpx.HTTPError, httpx.TimeoutException):
                content_type = ""  # HEAD failed, will detect from GET response

            # Fetch content
            response = await session.get(url)
            response.raise_for_status()

            actual_content_type = response.headers.get("content-type", "").lower()

            if "pdf" in actual_content_type or url.lower().endswith(".pdf"):
                result = await self._extract_pdf(url, response.content)
            else:
                html_text = response.text
                result = self._extract_html(url, html_text)

                # Phase K.1: Check if minimal content suggests JS rendering
                if self.enable_vl_scraper and result.get("success"):
                    content = result.get("content", "")
                    # If very little content extracted, try VL scraper as fallback
                    if len(content) < 500 and self._is_js_heavy_page(url, html_text):
                        logger.info(f"Minimal content ({len(content)} chars), trying VL scraper as fallback")
                        vl_result = await self._scrape_with_vl(url)
                        if vl_result and len(vl_result.get("content", "")) > len(content):
                            logger.info(f"VL scraper extracted more content: {len(vl_result.get('content', ''))} chars")
                            result = vl_result

            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            content_length = len(result.get("content", ""))
            metrics.record_scrape(
                domain=domain,
                success=result.get("success", False),
                content_length=content_length,
                duration_ms=duration_ms
            )

            # Cache the result
            if use_cache and result.get("success"):
                cache = get_content_cache()
                cache.set_content(
                    url=url,
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    content_type=result.get("content_type", "html"),
                    success=result.get("success", False),
                    error=result.get("error")
                )

            return result

        except httpx.HTTPStatusError as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            metrics.record_scrape(
                domain=domain,
                success=False,
                duration_ms=duration_ms,
                failure_reason=error_msg
            )
            logger.warning(f"HTTP error scraping {url}: {e.response.status_code}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_type": "error",
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = type(e).__name__
            metrics.record_scrape(
                domain=domain,
                success=False,
                duration_ms=duration_ms,
                failure_reason=error_msg
            )
            logger.warning(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_type": "error",
                "success": False,
                "error": str(e)
            }

    def _extract_html(self, url: str, html: str) -> Dict[str, Any]:
        """Extract readable content from HTML"""
        try:
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            title = re.sub(r'<[^>]+>', '', title)  # Remove any nested tags

            # Try to find main content areas
            content = html

            # Remove scripts, styles, nav, header, footer
            for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']:
                content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', content, flags=re.IGNORECASE | re.DOTALL)

            # Try to extract main content areas
            # Domain-specific patterns first (more specific = earlier match)
            main_patterns = [
                # StackOverflow/Stack Exchange patterns
                r'<div[^>]*id="question"[^>]*>(.*?)</div>',
                r'<div[^>]*id="answers"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*s-prose[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*js-post-body[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*post-text[^"]*"[^>]*>(.*?)</div>',
                # GitHub patterns
                r'<div[^>]*class="[^"]*markdown-body[^"]*"[^>]*>(.*?)</div>',
                r'<article[^>]*class="[^"]*markdown-body[^"]*"[^>]*>(.*?)</article>',
                # Generic patterns (fallback)
                r'<main[^>]*>(.*?)</main>',
                r'<article[^>]*>(.*?)</article>',
                r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*post[^"]*"[^>]*>(.*?)</div>',
            ]

            extracted = None

            # First try: collect all matches from multi-content patterns (SO answers, GH comments)
            multi_content_patterns = [
                r'<div[^>]*class="[^"]*s-prose[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*js-post-body[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*post-text[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*comment-body[^"]*"[^>]*>(.*?)</div>',
            ]

            for pattern in multi_content_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                if matches:
                    combined = '\n\n'.join(matches)
                    if len(combined) > 500:
                        extracted = combined
                        break

            # Second try: single-match patterns
            if not extracted:
                for pattern in main_patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match and len(match.group(1)) > 500:
                        extracted = match.group(1)
                        break

            if extracted:
                content = extracted

            # Remove remaining HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)

            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

            # Truncate if too long
            if len(content) > self.MAX_CONTENT_LENGTH:
                content = content[:self.MAX_CONTENT_LENGTH] + "... [truncated]"

            return {
                "url": url,
                "title": title,
                "content": content,
                "content_type": "html",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"HTML extraction error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_type": "error",
                "success": False,
                "error": str(e)
            }

    async def _extract_pdf(
        self,
        url: str,
        pdf_bytes: bytes,
        extract_images: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text and optionally images from PDF document.

        Returns:
        {
            "url": str,
            "title": str,
            "content": str,
            "content_type": "pdf",
            "success": bool,
            "error": Optional[str],
            "images": List[Dict]  # Optional, if extract_images=True
        }
        """
        images = []

        try:
            # Try PyMuPDF (fitz) first - best quality and image extraction
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                try:
                    text_parts = []

                    for page_num in range(min(doc.page_count, 50)):  # Limit to 50 pages
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

                        # Extract images from this page
                        if extract_images and PILLOW_AVAILABLE:
                            try:
                                image_list = page.get_images(full=True)
                                for img_idx, img_info in enumerate(image_list[:5]):  # Max 5 images per page
                                    xref = img_info[0]
                                    try:
                                        base_image = doc.extract_image(xref)
                                        image_bytes = base_image["image"]
                                        image_ext = base_image["ext"]

                                        # Convert to PIL Image for processing
                                        img = Image.open(io.BytesIO(image_bytes))

                                        # Skip very small images (likely icons/bullets)
                                        if img.width < 100 or img.height < 100:
                                            continue

                                        # Convert to base64 for vision model
                                        buffered = io.BytesIO()
                                        img_format = "PNG" if image_ext.lower() in ["png"] else "JPEG"
                                        if img.mode in ("RGBA", "P"):
                                            img = img.convert("RGB")
                                        img.save(buffered, format=img_format, quality=85)
                                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                                        images.append({
                                            "page": page_num + 1,
                                            "index": img_idx,
                                            "width": img.width,
                                            "height": img.height,
                                            "format": img_format.lower(),
                                            "base64": img_base64,
                                            "description": f"Image from page {page_num + 1}"
                                        })
                                    except Exception as e:
                                        logger.debug(f"Failed to extract image {img_idx} from page {page_num + 1}: {e}")
                            except Exception as e:
                                logger.debug(f"Failed to get images from page {page_num + 1}: {e}")
                finally:
                    doc.close()
                content = "\n\n".join(text_parts)

                # Truncate if too long
                if len(content) > self.MAX_CONTENT_LENGTH:
                    content = content[:self.MAX_CONTENT_LENGTH] + "\n... [truncated]"

                result = {
                    "url": url,
                    "title": f"PDF: {urlparse(url).path.split('/')[-1]}",
                    "content": content,
                    "content_type": "pdf",
                    "success": True,
                    "error": None
                }

                if images:
                    result["images"] = images
                    logger.info(f"Extracted {len(images)} images from PDF: {url}")

                return result

            except ImportError:
                logger.warning("PyMuPDF not installed, trying pypdf (no image extraction)")

            # Fallback to pypdf (text only, no image extraction)
            try:
                from pypdf import PdfReader

                pdf_file = io.BytesIO(pdf_bytes)
                reader = PdfReader(pdf_file)

                text_parts = []
                for page_num, page in enumerate(reader.pages[:50]):  # Limit to 50 pages
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

                content = "\n\n".join(text_parts)

                if len(content) > self.MAX_CONTENT_LENGTH:
                    content = content[:self.MAX_CONTENT_LENGTH] + "\n... [truncated]"

                return {
                    "url": url,
                    "title": f"PDF: {urlparse(url).path.split('/')[-1]}",
                    "content": content,
                    "content_type": "pdf",
                    "success": True,
                    "error": None,
                    "images": []  # pypdf doesn't support image extraction easily
                }

            except ImportError:
                logger.warning("pypdf not installed")
                return {
                    "url": url,
                    "title": f"PDF: {urlparse(url).path.split('/')[-1]}",
                    "content": "[PDF content - extraction libraries not available]",
                    "content_type": "pdf",
                    "success": False,
                    "error": "PDF extraction libraries not installed (pip install pymupdf or pypdf)"
                }

        except Exception as e:
            logger.error(f"PDF extraction error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_type": "error",
                "success": False,
                "error": str(e)
            }

    async def scrape_image(self, url: str) -> Dict[str, Any]:
        """
        Fetch and process an image URL for vision model analysis.

        Returns:
        {
            "url": str,
            "success": bool,
            "image_base64": str,
            "width": int,
            "height": int,
            "format": str,
            "error": Optional[str]
        }
        """
        if not PILLOW_AVAILABLE:
            return {
                "url": url,
                "success": False,
                "error": "Pillow not installed for image processing"
            }

        try:
            session = await self._get_session()
            response = await session.get(url)
            response.raise_for_status()

            # Load image
            img = Image.open(io.BytesIO(response.content))

            # Convert to RGB if needed
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize if too large (max 2048 on longest side for vision models)
            max_dim = 2048
            if max(img.width, img.height) > max_dim:
                ratio = max_dim / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                "url": url,
                "success": True,
                "image_base64": img_base64,
                "width": img.width,
                "height": img.height,
                "format": "jpeg",
                "error": None
            }

        except Exception as e:
            logger.error(f"Image fetch error for {url}: {e}")
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.is_closed:
            await self.session.aclose()


class VisionAnalyzer:
    """
    Analyzes images using vision-capable LLMs.

    Handles:
    - Chart and graph interpretation
    - Diagram analysis
    - Screenshot understanding
    - Technical figure extraction

    Automatically selects the best available vision model.
    """

    # Preferred vision models in order of capability
    PREFERRED_VISION_MODELS = [
        "qwen3-vl:32b",
        "qwen3-vl:8b",
        "qwen3-vl:4b",
        "qwen2.5vl:32b",
        "qwen2.5vl:7b",
        "llama3.2-vision:11b",
        "llama3.2-vision:latest",
        "granite3.2-vision:2b",
        "minicpm-v:8b",
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: Optional[str] = None
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 90.0  # Vision models can be slow

    async def get_available_vision_models(self) -> List[str]:
        """Get list of available vision models from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Filter to vision-capable models
                    vision_models = []
                    for m in models:
                        m_lower = m.lower()
                        if any(v in m_lower for v in ["vision", "-vl", "vl:", "minicpm-v", "granite3.2-vision"]):
                            vision_models.append(m)
                    return vision_models
        except Exception as e:
            logger.error(f"Failed to get vision models: {e}")
        return []

    async def select_best_vision_model(self) -> Optional[str]:
        """
        Select the best available vision model based on real-time GPU status.

        Uses:
        1. Real-time GPU VRAM availability from nvidia-smi
        2. Database-backed model specs for VRAM requirements
        3. Fallback to hardcoded preferences
        """
        if self.model:
            return self.model

        available = await self.get_available_vision_models()
        if not available:
            logger.warning("No vision models available")
            return None

        # Get real-time GPU status
        free_vram_gb = 20.0  # Default assumption
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                gpu_response = await client.get("http://localhost:8001/api/v1/models/gpu/status")
                if gpu_response.status_code == 200:
                    gpu_data = gpu_response.json().get("data", {})
                    free_vram_gb = gpu_data.get("free_vram_gb", 20.0)
                    loaded_models = gpu_data.get("loaded_models", [])
                    logger.info(f"GPU status: {free_vram_gb:.1f}GB free, {len(loaded_models)} models loaded")
        except Exception as e:
            logger.debug(f"GPU status unavailable: {e}")

        # Try to get model specs from database for VRAM-aware selection
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "http://localhost:8001/api/v1/models/specs",
                    params={"capability": "vision"}
                )
                if response.status_code == 200:
                    vision_specs = response.json()

                    # Filter models that fit in available VRAM (with 2GB buffer)
                    usable_vram = free_vram_gb - 2.0
                    fitting_models = [
                        spec for spec in vision_specs
                        if spec.get("model_name") in available
                        and spec.get("vram_min_gb", 100) <= usable_vram
                    ]

                    if fitting_models:
                        # Sort by capability (prefer larger models that still fit)
                        fitting_models.sort(key=lambda x: x.get("vram_min_gb", 0), reverse=True)
                        best = fitting_models[0]
                        self.model = best.get("model_name")
                        vram = best.get("vram_min_gb", 0)
                        logger.info(f"Selected vision model: {self.model} "
                                   f"(VRAM: {vram:.1f}GB, available: {free_vram_gb:.1f}GB)")
                        return self.model
                    else:
                        logger.warning(f"No vision models fit in {usable_vram:.1f}GB available VRAM")

        except Exception as e:
            logger.debug(f"Database model selection unavailable: {e}")

        # Fallback: Check preferred models
        for preferred in self.PREFERRED_VISION_MODELS:
            if preferred in available:
                self.model = preferred
                logger.info(f"Selected vision model (fallback): {preferred}")
                return preferred

        # Fallback to first available
        self.model = available[0]
        logger.info(f"Using fallback vision model: {self.model}")
        return self.model

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str = "Describe this image in detail. If it's a chart or graph, extract the key data points and trends.",
        context: Optional[str] = None,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze an image using a vision model.

        Args:
            image_base64: Base64-encoded image data
            prompt: Analysis prompt
            context: Optional context about what to look for

        Returns:
        {
            "success": bool,
            "description": str,
            "model_used": str,
            "error": Optional[str]
        }
        """
        model = await self.select_best_vision_model()
        if not model:
            return {
                "success": False,
                "description": "",
                "model_used": None,
                "error": "No vision model available"
            }

        # Build the full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "images": [image_base64],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for factual extraction
                            "num_predict": 1024
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    description = data.get("response", "")

                    # Track context utilization for vision analysis
                    if request_id and description:
                        metrics = get_performance_metrics()
                        metrics.record_context_utilization(
                            request_id=request_id,
                            agent_name="vision_analyzer",
                            model_name=model,
                            input_text=full_prompt,
                            output_text=description,
                            context_window=get_model_context_window(model)
                        )

                    return {
                        "success": True,
                        "description": description,
                        "model_used": model,
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "description": "",
                        "model_used": model,
                        "error": f"API error: {response.status_code}"
                    }

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {
                "success": False,
                "description": "",
                "model_used": model,
                "error": str(e)
            }

    async def analyze_chart(
        self,
        image_base64: str,
        chart_context: Optional[str] = None,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Specialized analysis for charts and graphs.

        Returns extracted data points, trends, and insights.
        """
        prompt = """Analyze this chart or graph image. Please provide:

1. **Chart Type**: (bar chart, line graph, pie chart, etc.)
2. **Title/Subject**: What is this chart about?
3. **Axes/Labels**: What are the X and Y axes measuring? What are the categories?
4. **Key Data Points**: List the specific values shown if readable
5. **Trends**: What patterns or trends are visible?
6. **Key Insights**: What are the main takeaways?

Format your response clearly with these sections."""

        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            context=chart_context,
            request_id=request_id
        )

    async def analyze_diagram(
        self,
        image_base64: str,
        diagram_context: Optional[str] = None,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Specialized analysis for technical diagrams.

        Returns component descriptions, relationships, and flow.
        """
        prompt = """Analyze this technical diagram. Please provide:

1. **Diagram Type**: (flowchart, architecture diagram, UML, network diagram, etc.)
2. **Components**: List all visible components/elements
3. **Relationships**: How are the components connected or related?
4. **Flow/Process**: If applicable, describe the flow or process shown
5. **Technical Details**: Any specific technical information visible

Format your response clearly with these sections."""

        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            context=diagram_context,
            request_id=request_id
        )

    async def extract_text_from_image(
        self,
        image_base64: str,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Extract text content from an image (OCR-like functionality).
        """
        prompt = """Extract all readable text from this image.

If this is a screenshot or document:
- List all visible text content
- Preserve the structure/formatting if possible
- Note any important headers or labels

If this is a chart or figure:
- Extract the title, axis labels, and legend text
- List any data labels or annotations"""

        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            request_id=request_id
        )

    async def analyze_images_batch(
        self,
        images: List[Dict[str, Any]],
        question: Optional[str] = None,
        request_id: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images, optionally in context of a question.

        Args:
            images: List of image dicts with 'base64' key
            question: Optional question to answer using the images
            request_id: Request ID for metrics tracking

        Returns:
            List of analysis results
        """
        results = []

        for i, img_data in enumerate(images[:10]):  # Limit to 10 images
            if "base64" not in img_data:
                continue

            context = None
            if question:
                context = f"Analyzing image {i+1} to help answer: {question}"

            # Determine if it looks like a chart/graph based on context
            page_info = img_data.get("description", "")
            if any(kw in page_info.lower() for kw in ["chart", "graph", "figure", "diagram"]):
                result = await self.analyze_chart(img_data["base64"], context, request_id=request_id)
            else:
                result = await self.analyze_image(
                    img_data["base64"],
                    prompt="Describe this image. If it contains data, charts, or technical information, extract the key details.",
                    context=context,
                    request_id=request_id
                )

            result["image_index"] = i
            result["page"] = img_data.get("page")
            results.append(result)

            # Small delay between vision requests
            await asyncio.sleep(0.5)

        return results


class DeepReader:
    """
    Deep content analysis using LLM reasoning.

    Takes scraped content and uses a powerful model to:
    - Extract specific answers to questions
    - Identify relevant technical details
    - Synthesize information from multiple sources
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:14b"  # Use a capable reasoning model
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 120.0  # Longer timeout for deep analysis

    async def analyze_content(
        self,
        question: str,
        scraped_content: List[Dict[str, Any]],
        max_context_chars: int = 30000,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze scraped content to answer a specific question.

        Returns:
        {
            "answer": str,
            "confidence": float,
            "sources_used": List[str],
            "key_findings": List[str],
            "limitations": str
        }
        """
        # Build context from scraped content
        context_parts = []
        sources_used = []
        total_chars = 0

        for item in scraped_content:
            if not item.get("success") or not item.get("content"):
                continue

            content = item["content"]
            url = item["url"]

            # Check if we have room
            if total_chars + len(content) > max_context_chars:
                # Truncate this piece
                remaining = max_context_chars - total_chars
                if remaining > 1000:
                    content = content[:remaining] + "... [truncated]"
                else:
                    break

            context_parts.append(f"=== SOURCE: {url} ===\n{content}\n")
            sources_used.append(url)
            total_chars += len(content)

        if not context_parts:
            return {
                "answer": "No content could be scraped from the sources.",
                "confidence": 0.0,
                "sources_used": [],
                "key_findings": [],
                "limitations": "All scraping attempts failed."
            }

        context = "\n".join(context_parts)

        # Build analysis prompt
        prompt = f"""You are an expert technical analyst. Analyze the following documentation to answer the user's question.

QUESTION: {question}

DOCUMENTATION:
{context}

Provide a comprehensive answer based ONLY on the documentation provided. Include:
1. Direct answer to the question with specific details
2. Relevant technical specifications, settings, or procedures mentioned
3. Any environment variables, configuration options, or commands found
4. Limitations or caveats mentioned in the documentation

If the documentation doesn't contain a direct answer, explain what related information was found.

Format your response as:

ANSWER:
[Your detailed answer here]

KEY FINDINGS:
- [Finding 1]
- [Finding 2]
- [etc.]

CONFIDENCE: [HIGH/MEDIUM/LOW]
[Explain why]

LIMITATIONS:
[What information was not found or unclear]"""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,  # Low temp for factual analysis
                            "num_predict": 2000
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

                answer_text = result.get("response", "")

                # Track context utilization for deep reading
                if request_id and answer_text:
                    metrics = get_performance_metrics()
                    metrics.record_context_utilization(
                        request_id=request_id,
                        agent_name="deep_reader",
                        model_name=self.model,
                        input_text=prompt,
                        output_text=answer_text,
                        context_window=get_model_context_window(self.model)
                    )

                # Parse the response
                return self._parse_analysis_response(answer_text, sources_used)

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return {
                "answer": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "sources_used": sources_used,
                "key_findings": [],
                "limitations": str(e)
            }

    def _parse_analysis_response(
        self,
        response: str,
        sources_used: List[str]
    ) -> Dict[str, Any]:
        """Parse structured response from LLM"""

        # Extract answer section
        answer = response
        if "ANSWER:" in response:
            parts = response.split("ANSWER:", 1)
            if len(parts) > 1:
                answer_part = parts[1]
                if "KEY FINDINGS:" in answer_part:
                    answer = answer_part.split("KEY FINDINGS:")[0].strip()
                elif "CONFIDENCE:" in answer_part:
                    answer = answer_part.split("CONFIDENCE:")[0].strip()
                else:
                    answer = answer_part.strip()

        # Extract key findings
        key_findings = []
        if "KEY FINDINGS:" in response:
            findings_part = response.split("KEY FINDINGS:", 1)[1]
            if "CONFIDENCE:" in findings_part:
                findings_part = findings_part.split("CONFIDENCE:")[0]

            for line in findings_part.strip().split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    key_findings.append(line[1:].strip())

        # Extract confidence
        confidence = 0.5
        if "CONFIDENCE:" in response:
            conf_part = response.split("CONFIDENCE:", 1)[1].split("\n")[0].upper()
            if "HIGH" in conf_part:
                confidence = 0.9
            elif "MEDIUM" in conf_part:
                confidence = 0.6
            elif "LOW" in conf_part:
                confidence = 0.3

        # Extract limitations
        limitations = ""
        if "LIMITATIONS:" in response:
            limitations = response.split("LIMITATIONS:", 1)[1].strip()

        return {
            "answer": answer,
            "confidence": confidence,
            "sources_used": sources_used,
            "key_findings": key_findings,
            "limitations": limitations
        }

    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError) as e:
            logger.debug(f"Failed to get available models: {e}")
        return []

    async def select_best_model(self) -> str:
        """Select the best available reasoning model"""
        available = await self.get_available_models()

        # Preferred models in order (larger/smarter first)
        preferred = [
            "qwen3:14b", "qwen3:8b", "qwen3:4b",
            "llama3.2:8b", "llama3.2:3b",
            "gemma3:9b", "gemma3:4b",
            "mistral:7b", "mistral:latest",
            "phi3:14b", "phi3:medium",
        ]

        for model in preferred:
            if model in available:
                logger.info(f"Selected reasoning model: {model}")
                return model

        # Fallback to any available model
        if available:
            logger.info(f"Using fallback model: {available[0]}")
            return available[0]

        return "gemma3:4b"  # Default
