"""
Centralized User-Agent Configuration for Web Scraping

Provides standardized User-Agent strings for all scrapers in the ecosystem.
Using a consistent format improves transparency and allows target sites
to identify and manage our scraping activity appropriately.

Format: RecoveryBot/1.0 (Component/1.0; Purpose; +https://recoverybot.app)

Based on scraping audit recommendations (2026-01-13).
"""

from typing import Dict

# Base information
BOT_NAME = "RecoveryBot"
BOT_VERSION = "1.0"
BOT_URL = "https://recoverybot.app"

# Standard User-Agent template
# Format follows robots.txt friendly conventions
USER_AGENT_TEMPLATE = "{bot}/{version} ({component}/{comp_version}; {purpose}; +{url})"


def build_user_agent(
    component: str,
    purpose: str,
    component_version: str = "1.0"
) -> str:
    """
    Build a standardized User-Agent string.

    Args:
        component: Name of the component (e.g., "Scraper", "CorpusBuilder")
        purpose: Brief description of purpose (e.g., "Content Extraction")
        component_version: Version of the component

    Returns:
        Formatted User-Agent string
    """
    return USER_AGENT_TEMPLATE.format(
        bot=BOT_NAME,
        version=BOT_VERSION,
        component=component,
        comp_version=component_version,
        purpose=purpose,
        url=BOT_URL
    )


# Pre-defined User-Agents for each component
class UserAgents:
    """Centralized User-Agent strings for all ecosystem components."""

    # Main content scraper
    CONTENT_SCRAPER = build_user_agent(
        "Scraper", "Content Extraction"
    )

    # Vision-Language scraper
    VL_SCRAPER = build_user_agent(
        "VLScraper", "Visual Content Analysis"
    )

    # Content extractor (multi-tier)
    CONTENT_EXTRACTOR = build_user_agent(
        "Extractor", "Document Processing"
    )

    # Model scraper (Ollama library)
    MODEL_SCRAPER = build_user_agent(
        "ModelScraper", "Model Discovery"
    )

    # PLC corpus builder
    PLC_CORPUS_BUILDER = build_user_agent(
        "CorpusBuilder", "Industrial Automation Research"
    )

    # RJG corpus builder
    RJG_CORPUS_BUILDER = build_user_agent(
        "CorpusBuilder", "Scientific Molding Research"
    )

    # Search provider (DuckDuckGo, etc.)
    SEARCH_PROVIDER = build_user_agent(
        "Searcher", "Web Search"
    )

    # Rate limiter default
    RATE_LIMITER = build_user_agent(
        "Scraper", "Rate Limited Fetch"
    )

    # Generic fallback
    DEFAULT = build_user_agent(
        "Agent", "General Purpose"
    )


# Mapping of old User-Agents to new standardized ones
# Use this for gradual migration and compatibility
MIGRATION_MAP: Dict[str, str] = {
    # Old patterns -> New standardized
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36": UserAgents.CONTENT_SCRAPER,
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36": UserAgents.SEARCH_PROVIDER,
    "RJG-Corpus-Builder/1.0 (Educational/Research)": UserAgents.RJG_CORPUS_BUILDER,
    "PLC-Corpus-Builder/1.0 (Educational/Research)": UserAgents.PLC_CORPUS_BUILDER,
    "RecoveryBot-ModelScraper/1.0": UserAgents.MODEL_SCRAPER,
    "RecoveryBot/1.0 (Recovery Services Directory)": UserAgents.DEFAULT,
    "RecoveryBot/1.0": UserAgents.DEFAULT,
}


def get_user_agent(component: str = None) -> str:
    """
    Get the appropriate User-Agent for a component.

    Args:
        component: Component name (optional). If None, returns DEFAULT.

    Returns:
        User-Agent string
    """
    if component is None:
        return UserAgents.DEFAULT

    component_lower = component.lower()

    # Map component names to User-Agents
    mapping = {
        "scraper": UserAgents.CONTENT_SCRAPER,
        "content_scraper": UserAgents.CONTENT_SCRAPER,
        "contentscraper": UserAgents.CONTENT_SCRAPER,
        "vl_scraper": UserAgents.VL_SCRAPER,
        "vlscraper": UserAgents.VL_SCRAPER,
        "extractor": UserAgents.CONTENT_EXTRACTOR,
        "content_extractor": UserAgents.CONTENT_EXTRACTOR,
        "model_scraper": UserAgents.MODEL_SCRAPER,
        "modelscraper": UserAgents.MODEL_SCRAPER,
        "plc": UserAgents.PLC_CORPUS_BUILDER,
        "plc_corpus": UserAgents.PLC_CORPUS_BUILDER,
        "rjg": UserAgents.RJG_CORPUS_BUILDER,
        "rjg_corpus": UserAgents.RJG_CORPUS_BUILDER,
        "search": UserAgents.SEARCH_PROVIDER,
        "searcher": UserAgents.SEARCH_PROVIDER,
    }

    return mapping.get(component_lower, UserAgents.DEFAULT)


# Browser-like User-Agent for sites that block bots
# Use sparingly and only when necessary
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def get_browser_user_agent() -> str:
    """
    Get a browser-like User-Agent for sites that block bots.

    Use sparingly - prefer transparent bot identification when possible.
    """
    return BROWSER_USER_AGENT
