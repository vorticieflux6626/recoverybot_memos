"""
Scraper Infrastructure Integration Tests

Tests the Phase 1-3 scraping consolidation components:
- UnifiedRateLimiter (aiometer-based)
- RedisCacheService (with circuit breaker)
- BaseCorpusScraper (unified base class)
- PLC/RJG scrapers (derived implementations)

Run with: python -m pytest server/tests/test_scraper_infrastructure.py -v
Or directly: python server/tests/test_scraper_infrastructure.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_scraper_infra")


# ============================================
# TEST RESULTS COLLECTOR
# ============================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            logger.info(f"  PASS: {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            logger.error(f"  FAIL: {name} - {error}")

    def summary(self):
        total = self.passed + self.failed
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            logger.info("Failures:")
            for name, error in self.errors:
                logger.info(f"  - {name}: {error}")
        logger.info(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# ============================================
# RATE LIMITER TESTS
# ============================================

async def test_rate_limiter_import():
    """Test rate limiter can be imported."""
    try:
        from agentic.rate_limiter import UnifiedRateLimiter, get_rate_limiter, RateLimitedClient
        results.record("rate_limiter_import", True)
        return True
    except Exception as e:
        results.record("rate_limiter_import", False, str(e))
        return False


async def test_rate_limiter_singleton():
    """Test rate limiter singleton creation."""
    try:
        from agentic.rate_limiter import get_rate_limiter
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2, "Singleton should return same instance"
        results.record("rate_limiter_singleton", True)
        return True
    except Exception as e:
        results.record("rate_limiter_singleton", False, str(e))
        return False


async def test_rate_limiter_domain_config():
    """Test rate limiter has domain configurations."""
    try:
        from agentic.rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()

        # Check some expected domain configs exist
        assert "github.com" in limiter.domain_configs, "github.com should have config"
        assert "stackoverflow.com" in limiter.domain_configs, "stackoverflow.com should have config"

        # Check config structure
        github_config = limiter.domain_configs["github.com"]
        assert hasattr(github_config, "max_per_second"), "Config should have max_per_second"
        assert hasattr(github_config, "max_concurrent"), "Config should have max_concurrent"

        results.record("rate_limiter_domain_config", True)
        return True
    except Exception as e:
        results.record("rate_limiter_domain_config", False, str(e))
        return False


async def test_rate_limiter_fetch():
    """Test rate limiter can fetch a URL."""
    try:
        from agentic.rate_limiter import get_rate_limiter, RateLimitedClient
        limiter = get_rate_limiter()

        # Use a local endpoint for testing
        test_url = "http://localhost:8001/api/v1/system/health"

        async with RateLimitedClient(limiter) as client:
            result = await client.get(test_url)

        # Should get a result (success or failure is OK, we're testing the limiter works)
        assert result is not None, "Should return a result"
        assert hasattr(result, "success"), "Result should have success attribute"
        assert hasattr(result, "url"), "Result should have url attribute"

        results.record("rate_limiter_fetch", True)
        return True
    except Exception as e:
        results.record("rate_limiter_fetch", False, str(e))
        return False


async def test_rate_limiter_stats():
    """Test rate limiter stats collection."""
    try:
        from agentic.rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()

        stats = limiter.get_stats()

        assert "total_requests" in stats, "Should have total_requests"
        assert "successful_requests" in stats, "Should have successful_requests"
        assert "failed_requests" in stats, "Should have failed_requests"

        results.record("rate_limiter_stats", True)
        return True
    except Exception as e:
        results.record("rate_limiter_stats", False, str(e))
        return False


# ============================================
# REDIS CACHE TESTS
# ============================================

async def test_redis_cache_import():
    """Test Redis cache service can be imported."""
    try:
        from agentic.redis_cache_service import RedisCacheService, get_redis_cache_service
        results.record("redis_cache_import", True)
        return True
    except Exception as e:
        results.record("redis_cache_import", False, str(e))
        return False


async def test_redis_cache_connect():
    """Test Redis cache can connect."""
    try:
        from agentic.redis_cache_service import get_redis_cache_service
        cache = get_redis_cache_service()

        await cache.connect()

        # Check connection status
        assert cache._connected or cache._fallback_mode, "Should be connected or in fallback mode"

        results.record("redis_cache_connect", True)
        return True
    except Exception as e:
        results.record("redis_cache_connect", False, str(e))
        return False


async def test_redis_cache_set_get():
    """Test Redis cache set/get operations."""
    try:
        from agentic.redis_cache_service import get_redis_cache_service
        cache = get_redis_cache_service()

        await cache.connect()

        # Test set
        test_url = "http://test.example.com/test-page"
        test_content = "This is test content for cache verification."

        await cache.set_content(
            url=test_url,
            title="Test Page",
            content=test_content,
            content_type="html",
            success=True
        )

        # Test get
        result = await cache.get_content(test_url)

        assert result is not None, "Should retrieve cached content"
        assert result.get("content") == test_content, "Content should match"
        assert result.get("title") == "Test Page", "Title should match"

        results.record("redis_cache_set_get", True)
        return True
    except Exception as e:
        results.record("redis_cache_set_get", False, str(e))
        return False


async def test_redis_cache_stats():
    """Test Redis cache stats."""
    try:
        from agentic.redis_cache_service import get_redis_cache_service
        cache = get_redis_cache_service()

        stats = await cache.get_stats()

        assert "connected" in stats, "Should have connected"
        assert "circuit_state" in stats, "Should have circuit_state"
        assert "content_cache" in stats, "Should have content_cache"

        results.record("redis_cache_stats", True)
        return True
    except Exception as e:
        results.record("redis_cache_stats", False, str(e))
        return False


async def test_redis_cache_circuit_breaker():
    """Test Redis cache circuit breaker state."""
    try:
        from agentic.redis_cache_service import get_redis_cache_service
        cache = get_redis_cache_service()

        # Check circuit breaker exists
        assert hasattr(cache, '_circuit'), "Should have circuit breaker"

        # Check stats report circuit state
        stats = await cache.get_stats()
        assert stats.get("circuit_state") in ["closed", "open", "half_open"], \
            f"Circuit state should be valid, got {stats.get('circuit_state')}"

        results.record("redis_cache_circuit_breaker", True)
        return True
    except Exception as e:
        results.record("redis_cache_circuit_breaker", False, str(e))
        return False


# ============================================
# USER AGENT TESTS
# ============================================

async def test_user_agent_import():
    """Test user agent config can be imported."""
    try:
        from agentic.user_agent_config import UserAgents, get_user_agent
        results.record("user_agent_import", True)
        return True
    except Exception as e:
        results.record("user_agent_import", False, str(e))
        return False


async def test_user_agent_values():
    """Test user agent values are defined."""
    try:
        from agentic.user_agent_config import UserAgents

        # Check required agents exist
        assert UserAgents.PLC_CORPUS_BUILDER, "PLC_CORPUS_BUILDER should exist"
        assert UserAgents.RJG_CORPUS_BUILDER, "RJG_CORPUS_BUILDER should exist"
        assert UserAgents.CONTENT_SCRAPER, "CONTENT_SCRAPER should exist"

        # Check format (should contain RecoveryBot)
        assert "RecoveryBot" in UserAgents.PLC_CORPUS_BUILDER, \
            "User agent should identify as RecoveryBot"

        results.record("user_agent_values", True)
        return True
    except Exception as e:
        results.record("user_agent_values", False, str(e))
        return False


# ============================================
# BASE CORPUS SCRAPER TESTS
# ============================================

async def test_base_scraper_import():
    """Test base corpus scraper can be imported."""
    try:
        from agentic.base_corpus_scraper import BaseCorpusScraper, ScrapeResult, ScraperMetrics
        results.record("base_scraper_import", True)
        return True
    except Exception as e:
        results.record("base_scraper_import", False, str(e))
        return False


async def test_plc_scraper_import():
    """Test PLC scraper can be imported."""
    try:
        from agentic.plc_corpus_scraper import PLCCorpusScraper, get_plc_scraper
        results.record("plc_scraper_import", True)
        return True
    except Exception as e:
        results.record("plc_scraper_import", False, str(e))
        return False


async def test_rjg_scraper_import():
    """Test RJG scraper can be imported."""
    try:
        from agentic.rjg_corpus_scraper import RJGCorpusScraper, get_rjg_scraper
        results.record("rjg_scraper_import", True)
        return True
    except Exception as e:
        results.record("rjg_scraper_import", False, str(e))
        return False


async def test_plc_scraper_inheritance():
    """Test PLC scraper properly inherits from base class."""
    try:
        from agentic.base_corpus_scraper import BaseCorpusScraper
        from agentic.plc_corpus_scraper import PLCCorpusScraper

        assert issubclass(PLCCorpusScraper, BaseCorpusScraper), \
            "PLCCorpusScraper should inherit from BaseCorpusScraper"

        results.record("plc_scraper_inheritance", True)
        return True
    except Exception as e:
        results.record("plc_scraper_inheritance", False, str(e))
        return False


async def test_rjg_scraper_inheritance():
    """Test RJG scraper properly inherits from base class."""
    try:
        from agentic.base_corpus_scraper import BaseCorpusScraper
        from agentic.rjg_corpus_scraper import RJGCorpusScraper

        assert issubclass(RJGCorpusScraper, BaseCorpusScraper), \
            "RJGCorpusScraper should inherit from BaseCorpusScraper"

        results.record("rjg_scraper_inheritance", True)
        return True
    except Exception as e:
        results.record("rjg_scraper_inheritance", False, str(e))
        return False


async def test_plc_scraper_schema():
    """Test PLC scraper creates valid schema."""
    try:
        from agentic.plc_corpus_scraper import PLCCorpusScraper

        # Create scraper (don't need full init for schema test)
        scraper = PLCCorpusScraper()

        # Check schema is created
        assert scraper.schema is not None, "Schema should be created"
        assert scraper.schema.domain_id == "plc_automation", "Domain ID should match"
        assert len(scraper.schema.entity_types) > 0, "Should have entity types"

        # Check entity types
        entity_type_names = [e.entity_type for e in scraper.schema.entity_types]
        assert "fault_code" in entity_type_names, "Should have fault_code entity"
        assert "module" in entity_type_names, "Should have module entity"

        results.record("plc_scraper_schema", True)
        return True
    except Exception as e:
        results.record("plc_scraper_schema", False, str(e))
        return False


async def test_rjg_scraper_schema():
    """Test RJG scraper creates valid schema."""
    try:
        from agentic.rjg_corpus_scraper import RJGCorpusScraper

        scraper = RJGCorpusScraper()

        assert scraper.schema is not None, "Schema should be created"
        assert scraper.schema.domain_id == "rjg_scientific_molding", "Domain ID should match"

        entity_type_names = [e.entity_type for e in scraper.schema.entity_types]
        assert "defect" in entity_type_names, "Should have defect entity"
        assert "process_variable" in entity_type_names, "Should have process_variable entity"

        results.record("rjg_scraper_schema", True)
        return True
    except Exception as e:
        results.record("rjg_scraper_schema", False, str(e))
        return False


async def test_scraper_has_rate_limiter():
    """Test scrapers are configured with rate limiter."""
    try:
        from agentic.plc_corpus_scraper import PLCCorpusScraper

        scraper = PLCCorpusScraper()

        assert scraper._use_rate_limiter is True, "Should use rate limiter"
        assert scraper._rate_limiter is not None, "Rate limiter should be initialized"

        results.record("scraper_has_rate_limiter", True)
        return True
    except Exception as e:
        results.record("scraper_has_rate_limiter", False, str(e))
        return False


async def test_scraper_has_redis_cache():
    """Test scrapers are configured with Redis cache."""
    try:
        from agentic.plc_corpus_scraper import PLCCorpusScraper

        scraper = PLCCorpusScraper()

        assert scraper._use_redis_cache is True, "Should use Redis cache"
        assert scraper._redis_cache is not None, "Redis cache should be initialized"

        results.record("scraper_has_redis_cache", True)
        return True
    except Exception as e:
        results.record("scraper_has_redis_cache", False, str(e))
        return False


async def test_scraper_metrics():
    """Test scraper metrics tracking."""
    try:
        from agentic.plc_corpus_scraper import PLCCorpusScraper

        scraper = PLCCorpusScraper()

        # Get initial metrics
        metrics = scraper.get_metrics()

        assert metrics is not None, "Should return metrics"
        assert hasattr(metrics, "urls_scraped"), "Should have urls_scraped"
        assert hasattr(metrics, "cache_hits"), "Should have cache_hits"

        # Test metrics to_dict
        metrics_dict = metrics.to_dict()
        assert "success_rate" in metrics_dict, "Should have success_rate"
        assert "cache_hit_rate" in metrics_dict, "Should have cache_hit_rate"

        results.record("scraper_metrics", True)
        return True
    except Exception as e:
        results.record("scraper_metrics", False, str(e))
        return False


# ============================================
# END-TO-END SCRAPE TEST
# ============================================

async def test_live_scrape():
    """Test a live scrape operation (uses httpbin for safe testing)."""
    try:
        from agentic.base_corpus_scraper import BaseCorpusScraper, ScrapeResult
        from agentic.domain_corpus import DomainSchema, DomainEntityDef

        # Create a minimal test scraper
        class TestScraper(BaseCorpusScraper):
            def create_schema(self):
                return DomainSchema(
                    domain_id="test",
                    domain_name="Test Domain",
                    description="Test",
                    entity_types=[
                        DomainEntityDef(
                            entity_type="test_entity",
                            description="Test entity",
                            extraction_patterns=[],
                            examples=["test"],
                            attributes=[]
                        )
                    ],
                    relationships=[]
                )

            def get_seed_urls(self):
                return []

            def get_article_urls(self):
                return []

            def get_user_agent(self):
                return "TestBot/1.0"

        scraper = TestScraper(
            domain_id="test_scraper",
            use_rate_limiter=True,
            use_redis_cache=True
        )

        # Scrape a simple test URL (httpbin returns HTML)
        result = await scraper.scrape_url(
            url="https://httpbin.org/html",
            source_type="test",
            extract_entities=False  # Skip entity extraction for speed
        )

        assert isinstance(result, ScrapeResult), "Should return ScrapeResult"
        assert result.url == "https://httpbin.org/html", "URL should match"

        # httpbin/html should succeed
        if result.success:
            assert result.word_count > 0, "Should have extracted some words"
            assert result.duration_ms > 0, "Should have duration"

        results.record("live_scrape", True)
        return True
    except Exception as e:
        results.record("live_scrape", False, str(e))
        return False


# ============================================
# MAIN TEST RUNNER
# ============================================

async def run_all_tests():
    """Run all tests in sequence."""
    logger.info("="*60)
    logger.info("SCRAPER INFRASTRUCTURE INTEGRATION TESTS")
    logger.info("="*60)

    # Rate Limiter Tests
    logger.info("\n[Rate Limiter Tests]")
    await test_rate_limiter_import()
    await test_rate_limiter_singleton()
    await test_rate_limiter_domain_config()
    await test_rate_limiter_fetch()
    await test_rate_limiter_stats()

    # Redis Cache Tests
    logger.info("\n[Redis Cache Tests]")
    await test_redis_cache_import()
    await test_redis_cache_connect()
    await test_redis_cache_set_get()
    await test_redis_cache_stats()
    await test_redis_cache_circuit_breaker()

    # User Agent Tests
    logger.info("\n[User Agent Tests]")
    await test_user_agent_import()
    await test_user_agent_values()

    # Base Scraper Tests
    logger.info("\n[Base Corpus Scraper Tests]")
    await test_base_scraper_import()
    await test_plc_scraper_import()
    await test_rjg_scraper_import()
    await test_plc_scraper_inheritance()
    await test_rjg_scraper_inheritance()
    await test_plc_scraper_schema()
    await test_rjg_scraper_schema()
    await test_scraper_has_rate_limiter()
    await test_scraper_has_redis_cache()
    await test_scraper_metrics()

    # End-to-End Test
    logger.info("\n[End-to-End Tests]")
    await test_live_scrape()

    # Print summary
    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
