"""
Scraper Infrastructure Integration Tests

Tests the Phase 1-4 scraping consolidation components:
- Phase 1-2: UnifiedRateLimiter, RedisCacheService
- Phase 3: BaseCorpusScraper, PLC/RJG scrapers
- Phase 4: ProxyManager, RetryStrategy, CrossEncoderReranker

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
# PHASE 4: PROXY MANAGER TESTS
# ============================================

async def test_proxy_manager_import():
    """Test proxy manager can be imported."""
    try:
        from agentic.proxy_manager import ProxyManager, get_proxy_manager, RotationStrategy
        results.record("proxy_manager_import", True)
        return True
    except Exception as e:
        results.record("proxy_manager_import", False, str(e))
        return False


async def test_proxy_manager_singleton():
    """Test proxy manager singleton creation."""
    try:
        from agentic.proxy_manager import get_proxy_manager
        pm1 = get_proxy_manager()
        pm2 = get_proxy_manager()
        assert pm1 is pm2, "Singleton should return same instance"
        results.record("proxy_manager_singleton", True)
        return True
    except Exception as e:
        results.record("proxy_manager_singleton", False, str(e))
        return False


async def test_proxy_manager_no_proxies():
    """Test proxy manager works without proxies configured."""
    try:
        from agentic.proxy_manager import get_proxy_manager
        pm = get_proxy_manager()

        # Should work even without proxies
        has_proxies = pm.has_proxies()
        assert isinstance(has_proxies, bool), "has_proxies should return bool"

        # If no proxies, get_proxy should return None
        if not has_proxies:
            proxy = await pm.get_proxy()
            assert proxy is None, "Should return None when no proxies"

        results.record("proxy_manager_no_proxies", True)
        return True
    except Exception as e:
        results.record("proxy_manager_no_proxies", False, str(e))
        return False


async def test_proxy_manager_stats():
    """Test proxy manager stats collection."""
    try:
        from agentic.proxy_manager import get_proxy_manager
        pm = get_proxy_manager()

        stats = pm.get_stats()

        assert "total_proxies" in stats, "Should have total_proxies"
        assert "active_proxies" in stats, "Should have active_proxies"
        assert "rotation_strategy" in stats, "Should have rotation_strategy"

        results.record("proxy_manager_stats", True)
        return True
    except Exception as e:
        results.record("proxy_manager_stats", False, str(e))
        return False


async def test_proxy_manager_rotation_strategies():
    """Test proxy manager has all rotation strategies."""
    try:
        from agentic.proxy_manager import RotationStrategy

        # Check all strategies exist
        assert RotationStrategy.ROUND_ROBIN, "Should have ROUND_ROBIN"
        assert RotationStrategy.RANDOM, "Should have RANDOM"
        assert RotationStrategy.WEIGHTED, "Should have WEIGHTED"
        assert RotationStrategy.LEAST_USED, "Should have LEAST_USED"

        results.record("proxy_manager_rotation_strategies", True)
        return True
    except Exception as e:
        results.record("proxy_manager_rotation_strategies", False, str(e))
        return False


# ============================================
# PHASE 4: RETRY STRATEGY TESTS
# ============================================

async def test_retry_strategy_import():
    """Test retry strategy can be imported."""
    try:
        from agentic.retry_strategy import (
            UnifiedRetryStrategy, get_retry_strategy, RetryContext,
            CircuitBreaker, CircuitState, with_retry
        )
        results.record("retry_strategy_import", True)
        return True
    except Exception as e:
        results.record("retry_strategy_import", False, str(e))
        return False


async def test_retry_strategy_singleton():
    """Test retry strategy singleton creation."""
    try:
        from agentic.retry_strategy import get_retry_strategy
        rs1 = get_retry_strategy()
        rs2 = get_retry_strategy()
        assert rs1 is rs2, "Singleton should return same instance"
        results.record("retry_strategy_singleton", True)
        return True
    except Exception as e:
        results.record("retry_strategy_singleton", False, str(e))
        return False


async def test_retry_strategy_domain_available():
    """Test retry strategy domain availability check."""
    try:
        from agentic.retry_strategy import get_retry_strategy
        strategy = get_retry_strategy()

        # New domain should be available (circuit closed)
        available = strategy.is_domain_available("example.com")
        assert available is True, "New domain should be available"

        results.record("retry_strategy_domain_available", True)
        return True
    except Exception as e:
        results.record("retry_strategy_domain_available", False, str(e))
        return False


async def test_retry_strategy_circuit_breaker():
    """Test retry strategy circuit breaker state transitions."""
    try:
        from agentic.retry_strategy import get_retry_strategy, CircuitState

        strategy = get_retry_strategy()

        # Get circuit for a test domain
        circuit = strategy._get_circuit("test-circuit-domain.com")

        # Initially should be closed
        assert circuit.state == CircuitState.CLOSED, "Initial state should be CLOSED"

        # Record some failures (not enough to open)
        for i in range(3):
            circuit.record_failure(Exception("test"))

        # Should still be closed (threshold is 5 by default)
        assert circuit.state == CircuitState.CLOSED, "Should still be CLOSED after 3 failures"

        results.record("retry_strategy_circuit_breaker", True)
        return True
    except Exception as e:
        results.record("retry_strategy_circuit_breaker", False, str(e))
        return False


async def test_retry_strategy_stats():
    """Test retry strategy stats collection."""
    try:
        from agentic.retry_strategy import get_retry_strategy
        strategy = get_retry_strategy()

        stats = strategy.get_stats()

        assert "retry_config" in stats, "Should have retry_config"
        assert "circuit_config" in stats, "Should have circuit_config"
        assert "circuits" in stats, "Should have circuits"

        # Check retry config structure
        retry_config = stats["retry_config"]
        assert "base_delay" in retry_config, "Should have base_delay"
        assert "max_delay" in retry_config, "Should have max_delay"
        assert "jitter_factor" in retry_config, "Should have jitter_factor"

        results.record("retry_strategy_stats", True)
        return True
    except Exception as e:
        results.record("retry_strategy_stats", False, str(e))
        return False


async def test_retry_strategy_context():
    """Test retry strategy context manager."""
    try:
        from agentic.retry_strategy import get_retry_strategy

        strategy = get_retry_strategy()

        # Use retry context
        async with strategy.context("context-test-domain.com") as ctx:
            attempt_count = 0
            for attempt in ctx.attempts(max_retries=3):
                attempt_count += 1
                if attempt.is_first:
                    assert attempt.attempt_number == 1, "First attempt should be 1"
                if attempt_count >= 2:
                    ctx.record_success()
                    break
                ctx.record_failure(Exception("test"))
                # Don't actually wait in tests
                # await attempt.wait()

        assert attempt_count == 2, "Should have done 2 attempts"

        results.record("retry_strategy_context", True)
        return True
    except Exception as e:
        results.record("retry_strategy_context", False, str(e))
        return False


# ============================================
# PHASE 4: CROSS-ENCODER RERANKER TESTS
# ============================================

async def test_reranker_import():
    """Test cross-encoder reranker can be imported."""
    try:
        from agentic.cross_encoder_reranker import (
            CrossEncoderReranker, RerankedResult, RerankerStats
        )
        results.record("reranker_import", True)
        return True
    except Exception as e:
        results.record("reranker_import", False, str(e))
        return False


async def test_reranker_initialization():
    """Test cross-encoder reranker initialization."""
    try:
        from agentic.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Check attributes
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3", "Should have default model"
        assert reranker.batch_size == 32, "Should have default batch size"

        results.record("reranker_initialization", True)
        return True
    except Exception as e:
        results.record("reranker_initialization", False, str(e))
        return False


async def test_reranker_availability():
    """Test cross-encoder reranker availability check."""
    try:
        from agentic.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # is_available returns False if FlagReranker isn't installed
        available = reranker.is_available()
        assert isinstance(available, bool), "is_available should return bool"

        # Note: We don't assert True because FlagReranker may not be installed

        results.record("reranker_availability", True)
        return True
    except Exception as e:
        results.record("reranker_availability", False, str(e))
        return False


async def test_reranker_fallback():
    """Test cross-encoder reranker fallback when model unavailable."""
    try:
        from agentic.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Test documents
        documents = [
            {"doc_id": "1", "content": "Test document one", "score": 0.9},
            {"doc_id": "2", "content": "Test document two", "score": 0.8},
            {"doc_id": "3", "content": "Test document three", "score": 0.7},
        ]

        # Rerank - should work even if model not available (returns original order)
        reranked, stats = await reranker.rerank(
            query="test query",
            documents=documents,
            top_k=2
        )

        assert len(reranked) <= 2, "Should return at most top_k results"
        assert stats.input_count == 3, "Should have correct input count"

        results.record("reranker_fallback", True)
        return True
    except Exception as e:
        results.record("reranker_fallback", False, str(e))
        return False


# ============================================
# PHASE 4: SEARCHER RERANKING INTEGRATION TESTS
# ============================================

async def test_searcher_reranking_enabled():
    """Test searcher has reranking configuration."""
    try:
        from agentic.searcher import SearcherAgent

        agent = SearcherAgent()

        # Check reranking attributes exist
        assert hasattr(agent, "_reranking_enabled"), "Should have _reranking_enabled"
        assert hasattr(agent, "_get_reranker"), "Should have _get_reranker method"
        assert hasattr(agent, "_apply_cross_encoder_reranking"), "Should have reranking method"

        results.record("searcher_reranking_enabled", True)
        return True
    except Exception as e:
        results.record("searcher_reranking_enabled", False, str(e))
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

    # Phase 4: Proxy Manager Tests
    logger.info("\n[Phase 4: Proxy Manager Tests]")
    await test_proxy_manager_import()
    await test_proxy_manager_singleton()
    await test_proxy_manager_no_proxies()
    await test_proxy_manager_stats()
    await test_proxy_manager_rotation_strategies()

    # Phase 4: Retry Strategy Tests
    logger.info("\n[Phase 4: Retry Strategy Tests]")
    await test_retry_strategy_import()
    await test_retry_strategy_singleton()
    await test_retry_strategy_domain_available()
    await test_retry_strategy_circuit_breaker()
    await test_retry_strategy_stats()
    await test_retry_strategy_context()

    # Phase 4: Cross-Encoder Reranker Tests
    logger.info("\n[Phase 4: Cross-Encoder Reranker Tests]")
    await test_reranker_import()
    await test_reranker_initialization()
    await test_reranker_availability()
    await test_reranker_fallback()

    # Phase 4: Searcher Integration Tests
    logger.info("\n[Phase 4: Searcher Integration Tests]")
    await test_searcher_reranking_enabled()

    # Print summary
    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
