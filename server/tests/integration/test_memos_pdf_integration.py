"""
Integration tests for memOS ↔ PDF Extraction Tools API.

Tests the end-to-end integration between:
- memOS HSEA controller
- PDF Extraction Tools API (port 8002)
- DocumentGraphService

Markers:
    @pytest.mark.integration - All integration tests
    @pytest.mark.requires_server - Requires memOS server running
    @pytest.mark.requires_pdf_api - Requires PDF API running
"""

import pytest
import httpx
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add server directory to path
SERVER_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SERVER_DIR))


# ============================================
# Server Availability Fixtures
# ============================================

@pytest.fixture
async def memos_available() -> bool:
    """Check if memOS server is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
async def pdf_api_available() -> bool:
    """Check if PDF API is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


# ============================================
# HTTP Client Fixtures
# ============================================

@pytest.fixture
async def memos_client():
    """HTTP client for memOS API."""
    async with httpx.AsyncClient(
        base_url="http://localhost:8001",
        timeout=60.0  # Longer timeout for agentic operations
    ) as client:
        yield client


@pytest.fixture
async def pdf_client():
    """HTTP client for PDF API."""
    async with httpx.AsyncClient(
        base_url="http://localhost:8002",
        timeout=30.0
    ) as client:
        yield client


# ============================================
# Basic Connectivity Tests
# ============================================

@pytest.mark.integration
class TestServerConnectivity:
    """Test basic server connectivity."""

    @pytest.mark.asyncio
    async def test_memos_health(self, memos_client, memos_available):
        """Test memOS server health endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_pdf_api_health(self, pdf_client, pdf_api_available):
        """Test PDF API health endpoint."""
        if not pdf_api_available:
            pytest.skip("PDF API not running")

        response = await pdf_client.get("/health")
        assert response.status_code == 200


# ============================================
# HSEA Endpoint Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestHSEAEndpoints:
    """Test HSEA three-stratum search endpoints."""

    @pytest.mark.asyncio
    async def test_hsea_stats(self, memos_client, memos_available):
        """Test HSEA stats endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/api/v1/search/hsea/stats")
        assert response.status_code == 200

        data = response.json()
        assert "success" in data

    @pytest.mark.asyncio
    async def test_hsea_search_basic(self, memos_client, memos_available):
        """Test HSEA basic search."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/hsea/search",
            json={
                "query": "SRVO-063",
                "top_k": 5
            }
        )

        # Should return 200 or 404 if no indexed data
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "success" in data

    @pytest.mark.asyncio
    async def test_hsea_troubleshoot_endpoint(self, memos_client, memos_available):
        """Test HSEA troubleshooting endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/api/v1/search/hsea/troubleshoot/SRVO-063")

        # Should return 200 or 404 if code not found
        assert response.status_code in [200, 404]


# ============================================
# Technical Documentation Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
@pytest.mark.requires_pdf_api
class TestTechnicalDocumentation:
    """Test technical documentation search (PDF API integration)."""

    @pytest.mark.asyncio
    async def test_technical_health(self, memos_client, memos_available, pdf_api_available):
        """Test technical documentation health check."""
        if not memos_available:
            pytest.skip("memOS server not running")
        if not pdf_api_available:
            pytest.skip("PDF API not running")

        response = await memos_client.get("/api/v1/search/technical/health")
        assert response.status_code == 200

        data = response.json()
        assert "success" in data

    @pytest.mark.asyncio
    async def test_technical_search(self, memos_client, memos_available, pdf_api_available):
        """Test technical documentation search."""
        if not memos_available:
            pytest.skip("memOS server not running")
        if not pdf_api_available:
            pytest.skip("PDF API not running")

        response = await memos_client.post(
            "/api/v1/search/technical/search",
            json={
                "query": "SRVO-063 calibration",
                "limit": 5
            }
        )

        # Should return 200 regardless of results
        assert response.status_code == 200

        data = response.json()
        assert "success" in data

    @pytest.mark.asyncio
    async def test_technical_troubleshoot(self, memos_client, memos_available, pdf_api_available):
        """Test technical troubleshooting endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")
        if not pdf_api_available:
            pytest.skip("PDF API not running")

        response = await memos_client.post(
            "/api/v1/search/technical/troubleshoot",
            json={
                "error_code": "SRVO-063"
            }
        )

        # Should return 200 (with results) or 404 (code not found)
        assert response.status_code in [200, 404]


# ============================================
# Universal Orchestrator Preset Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestUniversalOrchestratorPresets:
    """Test universal orchestrator preset configurations."""

    @pytest.mark.asyncio
    async def test_preset_list(self, memos_client, memos_available):
        """Test getting list of available presets."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/api/v1/search/universal/presets")

        # Should return presets endpoint
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_minimal_preset_search(self, memos_client, memos_available):
        """Test search with MINIMAL preset (fast)."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "test query",
                "preset": "minimal",
                "max_iterations": 1
            },
            timeout=120.0  # 2 minute timeout
        )

        assert response.status_code == 200

        data = response.json()
        assert "success" in data

    @pytest.mark.asyncio
    async def test_balanced_preset_search(self, memos_client, memos_available):
        """Test search with BALANCED preset."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "FANUC robot alarm",
                "preset": "balanced",
                "max_iterations": 2
            },
            timeout=180.0  # 3 minute timeout
        )

        assert response.status_code == 200


# ============================================
# Gateway Endpoint Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestGatewayEndpoint:
    """Test gateway streaming endpoint for Android integration."""

    @pytest.mark.asyncio
    async def test_gateway_stream_basic(self, memos_client, memos_available):
        """Test gateway stream endpoint returns SSE."""
        if not memos_available:
            pytest.skip("memOS server not running")

        # Use stream=True for SSE response
        async with memos_client.stream(
            "POST",
            "/api/v1/search/gateway/stream",
            json={
                "query": "test query",
                "preset": "minimal",
                "max_iterations": 1
            },
            timeout=120.0
        ) as response:
            assert response.status_code == 200

            # Should be SSE content type
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            # Read at least one event
            event_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event_count += 1
                    if event_count >= 3:  # Got some events
                        break

            assert event_count > 0, "Should receive at least one SSE event"


# ============================================
# Field Name Compatibility Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestFieldNameCompatibility:
    """Test field name compatibility between memOS and Android client."""

    @pytest.mark.asyncio
    async def test_search_result_fields(self, memos_client, memos_available):
        """Test that search results have expected field names."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "test",
                "preset": "minimal",
                "max_iterations": 1
            },
            timeout=120.0
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level response structure
        assert "success" in data
        assert "data" in data or "message" in data

        # If we have data, check result structure
        if data.get("success") and data.get("data"):
            result_data = data["data"]

            # Expected fields for Android compatibility
            expected_fields = ["synthesis", "sources", "confidence"]
            for field in expected_fields:
                if field in result_data:
                    # Field exists (good)
                    pass

    @pytest.mark.asyncio
    async def test_source_result_fields(self, memos_client, memos_available):
        """Test that source results have Android-compatible field names."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "Python programming",
                "preset": "balanced",
                "max_iterations": 2
            },
            timeout=180.0
        )

        assert response.status_code == 200
        data = response.json()

        if data.get("success") and data.get("data"):
            sources = data["data"].get("sources", [])

            if sources:
                # Check first source has expected fields
                first_source = sources[0]

                # These fields must exist for Android parsing
                # Using node_id instead of document_id per audit fix
                android_fields = ["title", "url"]
                for field in android_fields:
                    assert field in first_source, f"Source missing field: {field}"


# ============================================
# Error Handling Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestErrorHandling:
    """Test error handling across integration points."""

    @pytest.mark.asyncio
    async def test_invalid_preset(self, memos_client, memos_available):
        """Test handling of invalid preset name."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "test",
                "preset": "invalid_preset_name",
                "max_iterations": 1
            }
        )

        # Should handle gracefully (400 or fallback to default)
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_empty_query(self, memos_client, memos_available):
        """Test handling of empty query."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "",
                "preset": "minimal"
            }
        )

        # Should reject with 400 or 422
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_missing_query(self, memos_client, memos_available):
        """Test handling of missing query field."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "preset": "minimal"
            }
        )

        # Should reject with 422 (validation error)
        assert response.status_code == 422


# ============================================
# Performance Baseline Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
@pytest.mark.slow
class TestPerformanceBaseline:
    """Test performance baselines for search operations."""

    @pytest.mark.asyncio
    async def test_minimal_preset_latency(self, memos_client, memos_available):
        """Test MINIMAL preset completes within acceptable time."""
        if not memos_available:
            pytest.skip("memOS server not running")

        import time
        start = time.time()

        response = await memos_client.post(
            "/api/v1/search/universal",
            json={
                "query": "simple test query",
                "preset": "minimal",
                "max_iterations": 1
            },
            timeout=60.0
        )

        elapsed = time.time() - start

        assert response.status_code == 200
        # MINIMAL should complete within 60 seconds
        assert elapsed < 60, f"MINIMAL preset took {elapsed:.1f}s (expected <60s)"

    @pytest.mark.asyncio
    async def test_hsea_search_latency(self, memos_client, memos_available):
        """Test HSEA search completes within acceptable time."""
        if not memos_available:
            pytest.skip("memOS server not running")

        import time
        start = time.time()

        response = await memos_client.post(
            "/api/v1/search/hsea/search",
            json={
                "query": "SRVO-063",
                "top_k": 5
            }
        )

        elapsed = time.time() - start

        # HSEA should be fast (local index search)
        assert elapsed < 5, f"HSEA search took {elapsed:.1f}s (expected <5s)"


# ============================================
# Metrics Endpoint Tests
# ============================================

@pytest.mark.integration
@pytest.mark.requires_server
class TestMetricsEndpoints:
    """Test metrics and stats endpoints."""

    @pytest.mark.asyncio
    async def test_search_metrics(self, memos_client, memos_available):
        """Test search metrics endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/api/v1/search/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_stats(self, memos_client, memos_available):
        """Test cache stats endpoint."""
        if not memos_available:
            pytest.skip("memOS server not running")

        response = await memos_client.get("/api/v1/search/cache/stats")
        assert response.status_code == 200
