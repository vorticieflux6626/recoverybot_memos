"""
memOS Services Module

Provides various services for the Recovery Bot system:
- Model selection and capability detection
- Screenshot capture using Playwright
- Vision-Language based web scraping
- GPU monitoring
- Model scraping
"""

from .model_selector import (
    ModelSelector,
    ModelCapability,
    ModelInfo,
    get_model_selector
)

from .screenshot_capture import (
    ScreenshotCapture,
    ScreenshotResult,
    CaptureConfig,
    capture_screenshot
)

from .vl_scraper import (
    VLScraper,
    ExtractionType,
    ExtractionResult,
    ScrapingResult,
    scrape_url
)

__all__ = [
    # Model selector
    'ModelSelector',
    'ModelCapability',
    'ModelInfo',
    'get_model_selector',
    # Screenshot capture
    'ScreenshotCapture',
    'ScreenshotResult',
    'CaptureConfig',
    'capture_screenshot',
    # VL Scraper
    'VLScraper',
    'ExtractionType',
    'ExtractionResult',
    'ScrapingResult',
    'scrape_url',
]
