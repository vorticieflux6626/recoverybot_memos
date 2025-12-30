"""
Shared pytest fixtures for memOS server tests.

This module provides common fixtures used across unit and integration tests.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

# Add server directory to path
SERVER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_DIR))


# ============================================
# Event Loop Fixtures
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================
# Configuration Fixtures
# ============================================

@pytest.fixture
def settings():
    """Get application settings."""
    from config.settings import get_settings
    return get_settings()


@pytest.fixture
def mock_settings():
    """Create mock settings for isolated tests."""
    mock = MagicMock()
    mock.environment = "test"
    mock.database_url = "sqlite:///test.db"
    mock.ollama_base_url = "http://localhost:11434"
    mock.pdf_api_url = "http://localhost:8002"
    mock.pdf_api_enabled = True
    mock.pdf_api_timeout = 30
    return mock


# ============================================
# HTTP Client Fixtures
# ============================================

@pytest.fixture
async def http_client():
    """Create an async HTTP client for API testing."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
async def memos_client():
    """Create HTTP client for memOS API testing."""
    import httpx
    async with httpx.AsyncClient(
        base_url="http://localhost:8001",
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
async def pdf_api_client():
    """Create HTTP client for PDF API testing."""
    import httpx
    async with httpx.AsyncClient(
        base_url="http://localhost:8002",
        timeout=30.0
    ) as client:
        yield client


# ============================================
# Mock Service Fixtures
# ============================================

@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    mock = AsyncMock()
    mock.generate.return_value = {
        "response": "Mock LLM response",
        "model": "test-model",
        "done": True
    }
    mock.embeddings.return_value = {
        "embedding": [0.1] * 1024
    }
    return mock


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = AsyncMock()
    mock.encode.return_value = [0.1] * 1024
    mock.encode_batch.return_value = [[0.1] * 1024]
    return mock


# ============================================
# Sample Data Fixtures
# ============================================

@pytest.fixture
def sample_fanuc_text():
    """Sample FANUC technical text for testing."""
    return """
    SRVO-063 RCAL Alarm (Group: 1 Axis: 1)

    Cause: The calibration data for axis 1 is invalid.
    Check encoder cable connections.

    Remedy:
    1. Check encoder cable for damage
    2. Re-calibrate the robot using RCAL menu
    3. If error persists, replace encoder

    See also: SRVO-062, SRVO-064
    Related: Motor J1 encoder
    """


@pytest.fixture
def sample_error_codes():
    """Sample error codes for testing."""
    return [
        "SRVO-063",
        "MOTN-023",
        "SYST-001",
        "HOST-005",
        "INTP-100"
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "node_id": "node_001",
            "score": 0.95,
            "title": "SRVO-063 RCAL Alarm",
            "content_preview": "The calibration data for axis 1 is invalid...",
            "node_type": "error_code",
            "metadata": {"category": "SRVO"}
        },
        {
            "node_id": "node_002",
            "score": 0.87,
            "title": "Encoder Calibration",
            "content_preview": "Steps to calibrate encoder...",
            "node_type": "procedure",
            "metadata": {"category": "MAINTENANCE"}
        }
    ]


# ============================================
# Orchestrator Fixtures
# ============================================

@pytest.fixture
def orchestrator_presets():
    """Preset configurations for orchestrator testing."""
    return {
        "minimal": {
            "feature_count": 8,
            "expected_features": ["content_cache", "query_analysis", "scratchpad", "verification"]
        },
        "balanced": {
            "feature_count": 18,
            "expected_features": ["hyde", "bm25", "error_code_extraction"]
        },
        "enhanced": {
            "feature_count": 28,
            "expected_features": ["crag_evaluation", "self_reflection", "context_curation"]
        },
        "research": {
            "feature_count": 39,
            "expected_features": ["dynamic_planning", "reasoning_composer", "meta_buffer"]
        },
        "full": {
            "feature_count": 42,
            "expected_features": ["multi_agent", "entity_tracking", "all_features"]
        }
    }


# ============================================
# Entity/Normalizer Fixtures
# ============================================

@pytest.fixture
def error_code_normalizer():
    """Create an ErrorCodeNormalizer instance."""
    # Import from PDF_Extraction_Tools if available
    try:
        sys.path.insert(0, "/home/sparkone/sdd/PDF_Extraction_Tools")
        from pdf_extractor.entities import ErrorCodeNormalizer
        return ErrorCodeNormalizer()
    except ImportError:
        pytest.skip("PDF_Extraction_Tools not available")


@pytest.fixture
def entity_linker(error_code_normalizer):
    """Create an EntityLinker instance."""
    try:
        sys.path.insert(0, "/home/sparkone/sdd/PDF_Extraction_Tools")
        from pdf_extractor.entities import EntityLinker
        return EntityLinker(normalizer=error_code_normalizer)
    except ImportError:
        pytest.skip("PDF_Extraction_Tools not available")


# ============================================
# Cleanup Fixtures
# ============================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up any test files after each test."""
    yield
    # Cleanup logic here if needed
    import glob
    for f in glob.glob("/tmp/test_*.db"):
        try:
            Path(f).unlink()
        except OSError:
            pass


# ============================================
# Server Fixtures (for integration tests)
# ============================================

@pytest.fixture
async def check_server_running():
    """Check if memOS server is running."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
async def check_pdf_api_running():
    """Check if PDF API is running."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False
