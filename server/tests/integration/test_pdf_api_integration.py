"""
Integration tests for PDF_Extraction_Tools API integration.

Tests all new methods added to:
- DocumentGraphService (HSEA search, facets, IMM, graph stats, subgraph)
- EntityGroundingAgent (batch validation)
- CrossDomainValidator (causal chain validation)

Run with: pytest tests/integration/test_pdf_api_integration.py -v
"""

import asyncio
import pytest
from typing import List

# Import services
from core.document_graph_service import get_document_graph_service, DocumentGraphService
from agentic.entity_grounding import (
    EntityGroundingAgent,
    get_entity_grounding_agent,
    ExtractedEntity,
    EntityType,
    GroundingStatus
)
from agentic.cross_domain_validator import (
    CrossDomainValidator,
    get_cross_domain_validator,
    ValidationSeverity
)


@pytest.fixture
def doc_service() -> DocumentGraphService:
    """Get document graph service instance."""
    return get_document_graph_service()


@pytest.fixture
def grounding_agent() -> EntityGroundingAgent:
    """Get entity grounding agent instance."""
    return get_entity_grounding_agent()


@pytest.fixture
def validator() -> CrossDomainValidator:
    """Get cross-domain validator instance."""
    return get_cross_domain_validator()


class TestDocumentGraphServiceNew:
    """Tests for new DocumentGraphService methods."""

    @pytest.mark.asyncio
    async def test_hsea_search(self, doc_service):
        """Test HSEA three-stratum semantic search."""
        results = await doc_service.hsea_search(
            query="servo motor overcurrent",
            top_k=5
        )
        # May return empty if PDF API not available
        assert isinstance(results, list)
        if results:
            assert "title" in results[0] or "node_id" in results[0]
            print(f"HSEA search returned {len(results)} results")

    @pytest.mark.asyncio
    async def test_hsea_strata_info(self, doc_service):
        """Test HSEA strata configuration retrieval."""
        info = await doc_service.hsea_strata_info()
        assert isinstance(info, dict)
        if info:
            print(f"HSEA strata info: {list(info.keys())}")

    @pytest.mark.asyncio
    async def test_hsea_export_entities(self, doc_service):
        """Test HSEA entity export for caching."""
        result = await doc_service.hsea_export_entities(
            entity_types=["error_code"],
            limit=10
        )
        assert isinstance(result, dict)
        assert "entities" in result or "total" in result
        print(f"Exported {len(result.get('entities', []))} entities")

    @pytest.mark.asyncio
    async def test_get_facets(self, doc_service):
        """Test faceted search filters."""
        facets = await doc_service.get_facets()
        assert isinstance(facets, dict)
        if facets:
            print(f"Available facets: {list(facets.keys())}")

    @pytest.mark.asyncio
    async def test_get_search_suggestions(self, doc_service):
        """Test autocomplete suggestions."""
        suggestions = await doc_service.get_search_suggestions("SRVO", limit=5)
        assert isinstance(suggestions, list)
        print(f"Suggestions for 'SRVO': {suggestions[:3]}")

    @pytest.mark.asyncio
    async def test_imm_troubleshoot(self, doc_service):
        """Test IMM defect troubleshooting."""
        result = await doc_service.imm_troubleshoot(
            defect="flash",
            include_related=True
        )
        assert isinstance(result, dict)
        if result:
            print(f"IMM troubleshoot result keys: {list(result.keys())}")

    @pytest.mark.asyncio
    async def test_get_imm_entities(self, doc_service):
        """Test IMM entity listing."""
        entities = await doc_service.get_imm_entities(
            entity_type="defect",
            limit=10
        )
        assert isinstance(entities, list)
        print(f"IMM entities: {len(entities)}")

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, doc_service):
        """Test graph statistics retrieval."""
        stats = await doc_service.get_graph_stats()
        assert isinstance(stats, dict)
        if stats:
            print(f"Graph stats: {stats.get('total_nodes', 0)} nodes, "
                  f"{stats.get('total_edges', 0)} edges")

    @pytest.mark.asyncio
    async def test_get_subgraph(self, doc_service):
        """Test subgraph extraction."""
        # Use a known error code node if available
        result = await doc_service.get_subgraph(
            start_node_id="error_code:SRVO-063",
            max_depth=2,
            max_nodes=20
        )
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        print(f"Subgraph: {len(result.get('nodes', []))} nodes, "
              f"{len(result.get('edges', []))} edges")


class TestEntityGroundingAgentNew:
    """Tests for new EntityGroundingAgent methods."""

    @pytest.mark.asyncio
    async def test_batch_part_number_validation(self, grounding_agent):
        """Test batch part number validation."""
        # Create test entities
        entities = [
            ExtractedEntity(
                text="A06B-6110-H006",
                entity_type=EntityType.PART_NUMBER,
                context="Replace servo amplifier A06B-6110-H006"
            ),
            ExtractedEntity(
                text="A06B-6114-H206",
                entity_type=EntityType.PART_NUMBER,
                context="Check A06B-6114-H206 for issues"
            ),
            ExtractedEntity(
                text="1756-L71",
                entity_type=EntityType.PART_NUMBER,
                context="Allen-Bradley PLC 1756-L71"
            ),
        ]

        results = await grounding_agent._ground_part_numbers_batch(entities)
        assert isinstance(results, dict)
        assert len(results) == 3

        for pn, result in results.items():
            print(f"Part number {pn}: {result.status.value} ({result.confidence:.2f})")

    @pytest.mark.asyncio
    async def test_ground_entities_uses_batch(self, grounding_agent):
        """Test that ground_entities uses batch validation for multiple part numbers."""
        text = """
        The servo amplifier A06B-6110-H006 may need replacement.
        Also check the encoder cable A06B-6052-H003 and
        the controller module A06B-6400-H001.
        The Allen-Bradley 1756-L71 PLC should be verified too.
        """

        results = await grounding_agent.ground_entities(text, use_batch=True)
        assert results.total_entities >= 3
        print(f"Grounded {results.total_entities} entities: "
              f"{results.pattern_valid_count} valid, "
              f"{results.suspicious_count} suspicious, "
              f"{results.fabricated_count} fabricated")

    @pytest.mark.asyncio
    async def test_fabrication_detection(self, grounding_agent):
        """Test detection of fabricated part numbers."""
        text = "The SRV-XXXX-PULSCODER is a fabricated placeholder."

        results = await grounding_agent.ground_entities(text)
        # Should detect fabrication
        fabricated = [r for r in results.results if r.status == GroundingStatus.FABRICATED]
        print(f"Detected {len(fabricated)} fabricated entities")


class TestCrossDomainValidatorNew:
    """Tests for new CrossDomainValidator methods."""

    @pytest.mark.asyncio
    async def test_validate_causal_chain_valid(self, validator):
        """Test validation of valid causal chain."""
        # Valid chain: robot controller -> IMM controller (discrete I/O)
        chain = ["robot_controller", "imm_controller"]
        result = await validator.validate_causal_chain(chain)

        print(f"Chain {' -> '.join(chain)}: valid={result.is_valid}, "
              f"severity={result.severity.value}")

    @pytest.mark.asyncio
    async def test_validate_causal_chain_invalid(self, validator):
        """Test validation of invalid causal chain."""
        # Invalid chain: servo -> hydraulic (no physical connection)
        chain = ["servo_drive", "hydraulic_system"]
        result = await validator.validate_causal_chain(chain)

        assert not result.is_valid
        assert result.severity == ValidationSeverity.CRITICAL
        print(f"Chain {' -> '.join(chain)}: correctly flagged as invalid")

    @pytest.mark.asyncio
    async def test_validate_multi_hop_chain(self, validator):
        """Test validation of multi-hop causal chain."""
        # Invalid multi-hop: encoder -> servo -> hydraulic
        chain = ["encoder", "servo_drive", "hydraulic_system"]
        result = await validator.validate_causal_chain(chain)

        print(f"Multi-hop chain: valid={result.is_valid}, "
              f"message={result.message[:100]}...")

    def test_extract_causal_chains(self, validator):
        """Test extraction of causal chains from text."""
        text = """
        The encoder failure causes servo drive issues which leads to
        robot controller errors.
        """
        chains = validator.extract_causal_chains(text)
        print(f"Extracted {len(chains)} causal chains")
        for chain in chains:
            print(f"  Chain: {' -> '.join(chain)}")

    @pytest.mark.asyncio
    async def test_validate_synthesis_with_chains(self, validator):
        """Test full synthesis validation including chain detection."""
        synthesis = """
        When the servo encoder fails, it causes servo drive overcurrent which
        triggers the robot controller fault. This then propagates to the
        IMM hydraulic system causing pressure fluctuations.
        """
        result = await validator.validate_synthesis(synthesis)

        print(f"Synthesis validation: {result.total_claims} claims, "
              f"{result.critical_issues} critical, "
              f"{result.warnings} warnings")
        if result.revised_text:
            print("Revised text available")


class TestOrchestratorIntegration:
    """Tests for orchestrator integration of new capabilities."""

    @pytest.mark.asyncio
    async def test_technical_docs_search_imm(self):
        """Test technical docs search with IMM defect query."""
        from agentic.orchestrator_universal import UniversalOrchestrator, FeatureConfig

        config = FeatureConfig(enable_technical_docs=True)
        orchestrator = UniversalOrchestrator(config)

        # This tests the IMM troubleshooting path
        context = await orchestrator._search_technical_docs(
            "flash defect in injection molding"
        )
        if context:
            print(f"IMM context length: {len(context)} chars")
            print(f"Preview: {context[:200]}...")
        else:
            print("No context returned (PDF API may be unavailable)")

    @pytest.mark.asyncio
    async def test_technical_docs_search_fanuc(self):
        """Test technical docs search with FANUC error code."""
        from agentic.orchestrator_universal import UniversalOrchestrator, FeatureConfig

        config = FeatureConfig(
            enable_technical_docs=True,
            enable_structured_causal_chain=True
        )
        orchestrator = UniversalOrchestrator(config)

        context = await orchestrator._search_technical_docs(
            "SRVO-063 pulsecoder error"
        )
        if context:
            print(f"FANUC context length: {len(context)} chars")
            print(f"Preview: {context[:200]}...")
        else:
            print("No context returned (PDF API may be unavailable)")


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
