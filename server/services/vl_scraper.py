#!/usr/bin/env python3
"""
Vision-Language Scraper - Extract structured data from web pages using VL models.

This module provides:
1. Screenshot capture of JavaScript-rendered pages
2. VL model extraction using the best available model
3. Relevance evaluation to filter out irrelevant results
4. Structured data output for the Recovery Bot knowledge base

The key advantage over traditional scraping is the ability to:
- Process JavaScript-rendered content
- Handle complex layouts and visual elements
- Extract information from images, maps, and canvas elements
- Work with sites that actively block traditional scrapers
"""

import asyncio
import logging
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx

from .model_selector import ModelSelector, ModelCapability, ModelInfo, get_model_selector
from .screenshot_capture import (
    ScreenshotCapture, ScreenshotResult, CaptureConfig, capture_screenshot
)

logger = logging.getLogger(__name__)


class ExtractionType(Enum):
    """Types of data extraction"""
    RECOVERY_SERVICE = "recovery_service"
    CONTACT_INFO = "contact_info"
    GENERAL_INFO = "general_info"
    MEETING_SCHEDULE = "meeting_schedule"
    RESOURCE_LIST = "resource_list"


@dataclass
class ExtractionResult:
    """Result of VL extraction"""
    success: bool
    extraction_type: ExtractionType
    data: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None
    relevance_score: float = 0.0
    relevance_reasoning: Optional[str] = None
    model_used: Optional[str] = None
    extraction_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScrapingResult:
    """Complete result of a VL scraping operation"""
    success: bool
    url: str
    extraction: Optional[ExtractionResult] = None
    screenshot: Optional[ScreenshotResult] = None
    is_relevant: bool = False
    total_time_ms: float = 0.0
    error: Optional[str] = None


# Extraction prompts for different types
EXTRACTION_PROMPTS = {
    ExtractionType.RECOVERY_SERVICE: """Analyze this webpage screenshot and extract information about recovery services, treatment centers, or support resources.

Extract the following information in JSON format:
{
    "name": "Organization/facility name",
    "type": "Type of service (e.g., treatment center, support group, hotline)",
    "address": "Full street address if visible",
    "city": "City name",
    "state": "State abbreviation",
    "zip": "ZIP code",
    "phone": "Phone number(s)",
    "email": "Email address if visible",
    "website": "Website URL if visible",
    "hours": "Operating hours if visible",
    "services": ["List of services offered"],
    "description": "Brief description of the organization",
    "insurance": "Insurance/payment information if visible",
    "additional_info": "Any other relevant details"
}

If any field is not visible or not applicable, use null.
Return ONLY the JSON object, no additional text.""",

    ExtractionType.CONTACT_INFO: """Extract all contact information visible on this webpage screenshot.

Return in JSON format:
{
    "organization_name": "Name of organization",
    "phone_numbers": ["List of phone numbers"],
    "addresses": ["List of addresses"],
    "emails": ["List of email addresses"],
    "websites": ["List of website URLs"],
    "social_media": {"platform": "url"},
    "hours_of_operation": "Operating hours if visible"
}

Return ONLY the JSON object.""",

    ExtractionType.MEETING_SCHEDULE: """Extract meeting schedule information from this webpage screenshot.

Return in JSON format:
{
    "organization": "Organization name",
    "meeting_type": "Type of meeting (AA, NA, etc.)",
    "meetings": [
        {
            "day": "Day of week",
            "time": "Meeting time",
            "location": "Meeting location/address",
            "name": "Meeting name if given",
            "format": "In-person/Online/Hybrid",
            "notes": "Any additional notes"
        }
    ],
    "contact_info": "Contact information for questions"
}

Return ONLY the JSON object.""",

    ExtractionType.RESOURCE_LIST: """Extract the list of resources, organizations, or services shown on this webpage.

Return in JSON format:
{
    "page_title": "Title of the page",
    "resources": [
        {
            "name": "Resource/organization name",
            "type": "Type of resource",
            "description": "Brief description",
            "contact": "Contact info if visible",
            "link": "URL if visible"
        }
    ],
    "total_count": "Number of resources found"
}

Return ONLY the JSON object.""",

    ExtractionType.GENERAL_INFO: """Extract all relevant information from this webpage screenshot.

Return in JSON format:
{
    "page_title": "Title of the page",
    "main_topic": "Main topic or purpose of the page",
    "key_information": ["List of key facts or information points"],
    "organizations_mentioned": ["List of any organizations mentioned"],
    "contact_details": "Any contact information visible",
    "locations": ["Any locations/addresses mentioned"],
    "additional_notes": "Any other relevant information"
}

Return ONLY the JSON object."""
}

# Relevance evaluation prompt
RELEVANCE_PROMPT = """You are evaluating whether extracted information is relevant to recovery services, addiction treatment, mental health support, or community assistance resources.

Extracted Information:
{extracted_data}

Original Query/Context:
{context}

Evaluate the relevance on a scale of 0.0 to 1.0:
- 1.0: Highly relevant (directly about recovery services, treatment, support groups)
- 0.7-0.9: Moderately relevant (related healthcare, social services, community resources)
- 0.4-0.6: Somewhat relevant (general health info, tangentially related)
- 0.1-0.3: Low relevance (unrelated but might have some useful info)
- 0.0: Not relevant (completely unrelated, ads, errors, etc.)

Return in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "reasoning": "Brief explanation of the score",
    "useful_for": "What this information could be useful for, if anything",
    "recommendation": "keep" or "discard"
}}

Return ONLY the JSON object."""


class VLScraper:
    """
    Vision-Language based web scraper.

    Uses screenshots and VL models to extract structured data from
    JavaScript-rendered web pages.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        capture_config: Optional[CaptureConfig] = None
    ):
        self.ollama_host = ollama_host
        self.model_selector = get_model_selector(ollama_host)
        self.capture = ScreenshotCapture(capture_config or CaptureConfig())
        self._started = False

    async def start(self):
        """Initialize the scraper"""
        if self._started:
            return

        await self.capture.start()
        await self.model_selector.refresh_models()
        self._started = True
        logger.info("VL Scraper initialized")

    async def stop(self):
        """Shutdown the scraper"""
        await self.capture.stop()
        self._started = False
        logger.info("VL Scraper stopped")

    async def _call_ollama(
        self,
        model: str,
        prompt: str,
        image_base64: Optional[str] = None,
        timeout: float = 120.0
    ) -> Tuple[bool, str]:
        """Make a call to Ollama API"""
        try:
            messages = [{
                "role": "user",
                "content": prompt
            }]

            if image_base64:
                messages[0]["images"] = [image_base64]

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for structured extraction
                            "num_predict": 4096
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()

            content = data.get("message", {}).get("content", "")
            return True, content

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return False, str(e)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from model response, handling markdown code blocks"""
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            response = json_match.group(1)

        # Clean up the response
        response = response.strip()

        # Try to parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(response[start:end + 1])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
        return None

    async def extract_from_screenshot(
        self,
        screenshot: ScreenshotResult,
        extraction_type: ExtractionType = ExtractionType.GENERAL_INFO,
        custom_prompt: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract information from a screenshot using VL model.

        Args:
            screenshot: ScreenshotResult with image data
            extraction_type: Type of extraction to perform
            custom_prompt: Optional custom prompt override

        Returns:
            ExtractionResult with extracted data
        """
        start_time = datetime.utcnow()

        if not screenshot.success or not screenshot.image_base64:
            return ExtractionResult(
                success=False,
                extraction_type=extraction_type,
                error="Invalid or failed screenshot"
            )

        # Select the best vision model
        vision_model = await self.model_selector.select_vision_model(min_quality=3)
        if not vision_model:
            # Fall back to any vision model
            vision_model = await self.model_selector.select_vision_model(min_quality=1)

        if not vision_model:
            return ExtractionResult(
                success=False,
                extraction_type=extraction_type,
                error="No vision model available"
            )

        # Get the prompt
        prompt = custom_prompt or EXTRACTION_PROMPTS.get(
            extraction_type,
            EXTRACTION_PROMPTS[ExtractionType.GENERAL_INFO]
        )

        # Call the VL model
        logger.info(f"Extracting with {vision_model.name} ({extraction_type.value})")
        success, response = await self._call_ollama(
            vision_model.name,
            prompt,
            screenshot.image_base64
        )

        if not success:
            return ExtractionResult(
                success=False,
                extraction_type=extraction_type,
                error=f"VL model call failed: {response}",
                model_used=vision_model.name
            )

        # Parse the response
        parsed_data = self._parse_json_response(response)

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExtractionResult(
            success=parsed_data is not None,
            extraction_type=extraction_type,
            data=parsed_data or {},
            raw_response=response,
            model_used=vision_model.name,
            extraction_time_ms=elapsed_ms,
            metadata={
                'page_title': screenshot.page_title,
                'page_url': screenshot.page_url,
                'screenshot_size': len(screenshot.image_data) if screenshot.image_data else 0
            }
        )

    async def evaluate_relevance(
        self,
        extraction: ExtractionResult,
        context: str = "Recovery services, addiction treatment, mental health support"
    ) -> Tuple[float, str, str]:
        """
        Evaluate the relevance of extracted data.

        Args:
            extraction: ExtractionResult to evaluate
            context: Context/query for relevance evaluation

        Returns:
            Tuple of (relevance_score, reasoning, recommendation)
        """
        if not extraction.success or not extraction.data:
            return 0.0, "No data to evaluate", "discard"

        # Select a text model for relevance evaluation (doesn't need to be huge)
        text_model = await self.model_selector.select_text_model(
            max_size_gb=10,  # Use a smaller model for this simple task
            min_quality=2
        )

        if not text_model:
            # If no text model, assume relevant
            logger.warning("No text model for relevance check, assuming relevant")
            return 0.7, "No model available for evaluation", "keep"

        # Format the prompt
        prompt = RELEVANCE_PROMPT.format(
            extracted_data=json.dumps(extraction.data, indent=2),
            context=context
        )

        # Call the model
        success, response = await self._call_ollama(text_model.name, prompt)

        if not success:
            return 0.5, f"Evaluation failed: {response}", "keep"

        # Parse the response
        parsed = self._parse_json_response(response)

        if parsed:
            score = float(parsed.get("relevance_score", 0.5))
            reasoning = parsed.get("reasoning", "")
            recommendation = parsed.get("recommendation", "keep")
            return score, reasoning, recommendation

        return 0.5, "Could not parse evaluation response", "keep"

    async def scrape(
        self,
        url: str,
        extraction_type: ExtractionType = ExtractionType.RECOVERY_SERVICE,
        evaluate_relevance: bool = True,
        relevance_threshold: float = 0.4,
        context: str = "Recovery services and addiction support"
    ) -> ScrapingResult:
        """
        Complete scraping pipeline: capture screenshot, extract data, evaluate relevance.

        Args:
            url: URL to scrape
            extraction_type: Type of data to extract
            evaluate_relevance: Whether to evaluate relevance
            relevance_threshold: Minimum relevance score to keep
            context: Context for relevance evaluation

        Returns:
            ScrapingResult with all data
        """
        start_time = datetime.utcnow()

        if not self._started:
            await self.start()

        try:
            # Step 1: Capture screenshot
            logger.info(f"Scraping {url}")
            screenshot = await self.capture.capture(url)

            if not screenshot.success:
                return ScrapingResult(
                    success=False,
                    url=url,
                    screenshot=screenshot,
                    error=f"Screenshot failed: {screenshot.error}"
                )

            # Step 2: Extract data using VL model
            extraction = await self.extract_from_screenshot(
                screenshot,
                extraction_type
            )

            if not extraction.success:
                return ScrapingResult(
                    success=False,
                    url=url,
                    screenshot=screenshot,
                    extraction=extraction,
                    error=f"Extraction failed: {extraction.error}"
                )

            # Step 3: Evaluate relevance (optional)
            is_relevant = True
            if evaluate_relevance:
                score, reasoning, recommendation = await self.evaluate_relevance(
                    extraction,
                    context
                )
                extraction.relevance_score = score
                extraction.relevance_reasoning = reasoning
                is_relevant = score >= relevance_threshold

                logger.info(f"Relevance: {score:.2f} ({recommendation}) - {reasoning}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ScrapingResult(
                success=True,
                url=url,
                screenshot=screenshot,
                extraction=extraction,
                is_relevant=is_relevant,
                total_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return ScrapingResult(
                success=False,
                url=url,
                error=str(e)
            )

    async def scrape_multiple(
        self,
        urls: List[str],
        extraction_type: ExtractionType = ExtractionType.RECOVERY_SERVICE,
        max_concurrent: int = 3,
        **kwargs
    ) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            extraction_type: Type of data to extract
            max_concurrent: Maximum concurrent scrapes
            **kwargs: Additional arguments for scrape()

        Returns:
            List of ScrapingResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url: str) -> ScrapingResult:
            async with semaphore:
                return await self.scrape(url, extraction_type, **kwargs)

        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)


# Convenience function
async def scrape_url(
    url: str,
    extraction_type: ExtractionType = ExtractionType.RECOVERY_SERVICE,
    ollama_host: str = "http://localhost:11434"
) -> ScrapingResult:
    """
    Convenience function for one-off VL scraping.

    Args:
        url: URL to scrape
        extraction_type: Type of data to extract
        ollama_host: Ollama API host

    Returns:
        ScrapingResult
    """
    scraper = VLScraper(ollama_host)
    try:
        await scraper.start()
        return await scraper.scrape(url, extraction_type)
    finally:
        await scraper.stop()


async def main():
    """Test the VL scraper"""
    # Test URL - SAMHSA helpline page
    url = "https://www.samhsa.gov/find-help/national-helpline"

    print(f"\n{'='*60}")
    print(f"VL Scraper Test")
    print(f"{'='*60}")
    print(f"URL: {url}")

    scraper = VLScraper()

    try:
        await scraper.start()

        print("\n--- Model Selection ---")
        vision_model = await scraper.model_selector.select_vision_model()
        text_model = await scraper.model_selector.select_text_model()
        print(f"Vision model: {vision_model.name if vision_model else 'None'}")
        print(f"Text model: {text_model.name if text_model else 'None'}")

        print("\n--- Scraping ---")
        result = await scraper.scrape(
            url,
            ExtractionType.RECOVERY_SERVICE,
            evaluate_relevance=True
        )

        print(f"\nSuccess: {result.success}")
        print(f"Total time: {result.total_time_ms:.0f}ms")

        if result.screenshot:
            print(f"\nScreenshot:")
            print(f"  Title: {result.screenshot.page_title}")
            print(f"  Size: {len(result.screenshot.image_data or b'')} bytes")

        if result.extraction:
            print(f"\nExtraction:")
            print(f"  Model: {result.extraction.model_used}")
            print(f"  Time: {result.extraction.extraction_time_ms:.0f}ms")
            print(f"  Relevance: {result.extraction.relevance_score:.2f}")
            print(f"  Reasoning: {result.extraction.relevance_reasoning}")
            print(f"\nExtracted Data:")
            print(json.dumps(result.extraction.data, indent=2))

        if result.error:
            print(f"\nError: {result.error}")

    finally:
        await scraper.stop()


if __name__ == "__main__":
    asyncio.run(main())
