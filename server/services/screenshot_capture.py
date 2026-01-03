#!/usr/bin/env python3
"""
Screenshot Capture Module - Lightweight web page screenshot capture using Playwright.

This module provides:
1. Full page screenshots for VL model processing
2. Viewport screenshots for specific sections
3. Element-specific screenshots
4. PDF capture for document pages
5. Scroll-and-capture for long pages
"""

import asyncio
import logging
import base64
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import playwright
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")


@dataclass
class ScreenshotResult:
    """Result of a screenshot capture operation"""
    success: bool
    image_data: Optional[bytes] = None  # Raw PNG bytes
    image_base64: Optional[str] = None  # Base64 encoded
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    viewport_size: Optional[Tuple[int, int]] = None
    full_page_size: Optional[Tuple[int, int]] = None
    capture_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CaptureConfig:
    """Configuration for screenshot capture"""
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = True
    wait_for_load: bool = True
    wait_timeout: int = 30000  # ms
    wait_for_network_idle: bool = True
    quality: int = 90  # JPEG quality (ignored for PNG)
    format: str = "png"  # png or jpeg
    scale: float = 1.0  # Device scale factor
    scroll_delay: int = 500  # ms between scroll captures
    max_height: int = 16384  # Max screenshot height
    block_resources: List[str] = None  # Resource types to block (e.g., ['image', 'font'])
    user_agent: Optional[str] = None
    extra_http_headers: Dict[str, str] = None

    def __post_init__(self):
        if self.block_resources is None:
            self.block_resources = []
        if self.extra_http_headers is None:
            self.extra_http_headers = {}


class ScreenshotCapture:
    """
    Lightweight screenshot capture using Playwright.

    Optimized for VL model processing:
    - Takes full-page screenshots
    - Handles dynamic content loading
    - Provides scroll-and-capture for very long pages
    - Blocks unnecessary resources for faster loading
    """

    def __init__(self, config: Optional[CaptureConfig] = None):
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed")

        self.config = config or CaptureConfig()
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._playwright = None

    async def start(self):
        """Start the browser instance"""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--single-process',
            ]
        )

        self._context = await self._browser.new_context(
            viewport={'width': self.config.viewport_width, 'height': self.config.viewport_height},
            device_scale_factor=self.config.scale,
            user_agent=self.config.user_agent,
            extra_http_headers=self.config.extra_http_headers or {}
        )

        # Block unnecessary resources if configured
        if self.config.block_resources:
            await self._context.route("**/*", self._block_resources)

        logger.info("Screenshot capture browser started")

    def _is_browser_healthy(self) -> bool:
        """Check if browser and context are still connected and usable"""
        try:
            if self._browser is None or self._context is None:
                return False
            # Check if browser is still connected
            if not self._browser.is_connected():
                return False
            return True
        except Exception:
            return False

    async def _ensure_browser(self):
        """Ensure browser is running and healthy, restart if needed"""
        if not self._is_browser_healthy():
            logger.warning("Browser context unhealthy, restarting...")
            await self.stop()
            await self.start()

    async def stop(self):
        """Stop the browser instance"""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Screenshot capture browser stopped")

    async def _block_resources(self, route):
        """Block specified resource types"""
        if route.request.resource_type in self.config.block_resources:
            await route.abort()
        else:
            await route.continue_()

    async def capture(
        self,
        url: str,
        config: Optional[CaptureConfig] = None
    ) -> ScreenshotResult:
        """
        Capture a screenshot of a web page.

        Args:
            url: URL to capture
            config: Optional override config for this capture

        Returns:
            ScreenshotResult with image data and metadata
        """
        cfg = config or self.config
        start_time = datetime.now(timezone.utc)

        try:
            # Ensure browser is running and healthy
            await self._ensure_browser()

            # Create new page with retry on context failure
            try:
                page = await self._context.new_page()
            except Exception as e:
                if "closed" in str(e).lower():
                    logger.warning(f"Browser context closed, restarting: {e}")
                    await self.stop()
                    await self.start()
                    page = await self._context.new_page()
                else:
                    raise

            try:
                # Navigate to URL
                logger.info(f"Capturing screenshot of {url}")

                response = await page.goto(
                    url,
                    wait_until='domcontentloaded',
                    timeout=cfg.wait_timeout
                )

                # Wait for network to be idle if configured
                if cfg.wait_for_network_idle:
                    try:
                        await page.wait_for_load_state('networkidle', timeout=cfg.wait_timeout // 2)
                    except Exception:
                        # Network idle timeout is not critical
                        logger.debug("Network idle timeout, continuing anyway")

                # Additional wait for JavaScript rendering
                await asyncio.sleep(0.5)

                # Scroll to trigger lazy loading if full page
                if cfg.full_page:
                    await self._scroll_page(page)

                # Get page info
                page_title = await page.title()
                page_url = page.url

                # Get page dimensions
                dimensions = await page.evaluate("""
                    () => ({
                        viewportWidth: window.innerWidth,
                        viewportHeight: window.innerHeight,
                        fullWidth: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth),
                        fullHeight: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)
                    })
                """)

                # Take screenshot
                screenshot_options = {
                    'type': cfg.format,
                    'full_page': cfg.full_page,
                }

                if cfg.format == 'jpeg':
                    screenshot_options['quality'] = cfg.quality

                image_data = await page.screenshot(**screenshot_options)

                return ScreenshotResult(
                    success=True,
                    image_data=image_data,
                    image_base64=base64.b64encode(image_data).decode('utf-8'),
                    page_title=page_title,
                    page_url=page_url,
                    viewport_size=(dimensions['viewportWidth'], dimensions['viewportHeight']),
                    full_page_size=(dimensions['fullWidth'], dimensions['fullHeight']),
                    capture_time=start_time,
                    metadata={
                        'status_code': response.status if response else None,
                        'capture_duration_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                        'config': {
                            'full_page': cfg.full_page,
                            'viewport': f"{cfg.viewport_width}x{cfg.viewport_height}",
                            'format': cfg.format,
                        }
                    }
                )

            finally:
                await page.close()

        except Exception as e:
            logger.error(f"Screenshot capture failed for {url}: {e}")
            return ScreenshotResult(
                success=False,
                error=str(e),
                capture_time=start_time
            )

    async def _scroll_page(self, page: Page):
        """Scroll through the page to trigger lazy loading"""
        try:
            # Get page height
            page_height = await page.evaluate("document.body.scrollHeight")
            viewport_height = self.config.viewport_height

            # Scroll in steps
            current_position = 0
            while current_position < page_height:
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await asyncio.sleep(self.config.scroll_delay / 1000)
                current_position += viewport_height

                # Update page height in case lazy loading added content
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height > page_height:
                    page_height = new_height

                # Safety limit
                if current_position > self.config.max_height:
                    break

            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.2)

        except Exception as e:
            logger.warning(f"Error during page scroll: {e}")

    async def capture_element(
        self,
        url: str,
        selector: str,
        config: Optional[CaptureConfig] = None
    ) -> ScreenshotResult:
        """
        Capture a screenshot of a specific element.

        Args:
            url: URL to capture
            selector: CSS selector for the element
            config: Optional override config

        Returns:
            ScreenshotResult with element screenshot
        """
        cfg = config or self.config
        start_time = datetime.now(timezone.utc)

        try:
            await self._ensure_browser()

            try:
                page = await self._context.new_page()
            except Exception as e:
                if "closed" in str(e).lower():
                    logger.warning(f"Browser context closed, restarting: {e}")
                    await self.stop()
                    await self.start()
                    page = await self._context.new_page()
                else:
                    raise

            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=cfg.wait_timeout)

                if cfg.wait_for_network_idle:
                    try:
                        await page.wait_for_load_state('networkidle', timeout=cfg.wait_timeout // 2)
                    except Exception:
                        pass

                # Wait for element
                element = await page.wait_for_selector(selector, timeout=cfg.wait_timeout)

                if not element:
                    return ScreenshotResult(
                        success=False,
                        error=f"Element not found: {selector}",
                        capture_time=start_time
                    )

                # Screenshot the element
                image_data = await element.screenshot(type=cfg.format)

                return ScreenshotResult(
                    success=True,
                    image_data=image_data,
                    image_base64=base64.b64encode(image_data).decode('utf-8'),
                    page_title=await page.title(),
                    page_url=page.url,
                    capture_time=start_time,
                    metadata={'element_selector': selector}
                )

            finally:
                await page.close()

        except Exception as e:
            logger.error(f"Element capture failed: {e}")
            return ScreenshotResult(
                success=False,
                error=str(e),
                capture_time=start_time
            )

    async def capture_multiple_viewports(
        self,
        url: str,
        scroll_steps: int = 3
    ) -> List[ScreenshotResult]:
        """
        Capture multiple viewport screenshots by scrolling.

        Useful for very long pages where a single full-page screenshot
        would be too large for VL model processing.

        Args:
            url: URL to capture
            scroll_steps: Number of viewport-height scroll steps

        Returns:
            List of ScreenshotResults, one per viewport
        """
        results = []
        cfg = self.config

        try:
            await self._ensure_browser()

            try:
                page = await self._context.new_page()
            except Exception as e:
                if "closed" in str(e).lower():
                    logger.warning(f"Browser context closed, restarting: {e}")
                    await self.stop()
                    await self.start()
                    page = await self._context.new_page()
                else:
                    raise

            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=cfg.wait_timeout)

                if cfg.wait_for_network_idle:
                    try:
                        await page.wait_for_load_state('networkidle', timeout=cfg.wait_timeout // 2)
                    except Exception:
                        pass

                page_height = await page.evaluate("document.body.scrollHeight")
                viewport_height = cfg.viewport_height
                actual_steps = min(scroll_steps, (page_height // viewport_height) + 1)

                for i in range(actual_steps):
                    scroll_position = i * viewport_height
                    await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                    await asyncio.sleep(cfg.scroll_delay / 1000)

                    # Capture viewport (not full page)
                    image_data = await page.screenshot(type=cfg.format, full_page=False)

                    results.append(ScreenshotResult(
                        success=True,
                        image_data=image_data,
                        image_base64=base64.b64encode(image_data).decode('utf-8'),
                        page_title=await page.title(),
                        page_url=page.url,
                        viewport_size=(cfg.viewport_width, cfg.viewport_height),
                        capture_time=datetime.now(timezone.utc),
                        metadata={
                            'viewport_index': i,
                            'scroll_position': scroll_position,
                            'total_viewports': actual_steps
                        }
                    ))

            finally:
                await page.close()

        except Exception as e:
            logger.error(f"Multi-viewport capture failed: {e}")
            results.append(ScreenshotResult(
                success=False,
                error=str(e),
                capture_time=datetime.now(timezone.utc)
            ))

        return results

    async def save_screenshot(
        self,
        result: ScreenshotResult,
        output_path: str
    ) -> bool:
        """Save a screenshot result to disk"""
        if not result.success or not result.image_data:
            return False

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'wb') as f:
                f.write(result.image_data)

            logger.info(f"Screenshot saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return False


# Convenience function for one-off captures
async def capture_screenshot(
    url: str,
    full_page: bool = True,
    viewport_width: int = 1920,
    viewport_height: int = 1080
) -> ScreenshotResult:
    """
    Convenience function for one-off screenshot captures.

    Args:
        url: URL to capture
        full_page: Whether to capture the full page or just viewport
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels

    Returns:
        ScreenshotResult with image data
    """
    config = CaptureConfig(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        full_page=full_page
    )

    capture = ScreenshotCapture(config)
    try:
        await capture.start()
        return await capture.capture(url)
    finally:
        await capture.stop()


async def main():
    """Test the screenshot capture"""
    url = "https://www.samhsa.gov/find-help/national-helpline"

    print(f"Capturing screenshot of {url}...")

    result = await capture_screenshot(url, full_page=True)

    if result.success:
        print(f"Success!")
        print(f"Title: {result.page_title}")
        print(f"Viewport: {result.viewport_size}")
        print(f"Full page: {result.full_page_size}")
        print(f"Image size: {len(result.image_data)} bytes")
        print(f"Duration: {result.metadata.get('capture_duration_ms', 0):.0f}ms")

        # Save to file
        output_path = "/tmp/screenshot_test.png"
        with open(output_path, 'wb') as f:
            f.write(result.image_data)
        print(f"Saved to {output_path}")
    else:
        print(f"Failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
