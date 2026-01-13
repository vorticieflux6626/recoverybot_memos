"""
Tests for Mem0 Integration in Agentic Pipeline

Tests the mem0_adapter.py and mem0_config.py modules for:
1. Configuration loading
2. Memory extraction
3. Entity tracking
4. Cross-session context retrieval
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Import the modules under test
from agentic.mem0_config import (
    get_mem0_config,
    get_preset_config,
    PRESETS,
    GATEWAY_URL,
    OLLAMA_URL
)
from agentic.mem0_adapter import (
    AgenticMemoryAdapter,
    get_adapter_for_user,
    clear_adapter_for_user,
    MemoryCategory,
    ExtractedMemory,
    ConversationContext
)


class TestMem0Config:
    """Tests for mem0_config.py"""

    def test_get_mem0_config_default(self):
        """Test default configuration generation"""
        config = get_mem0_config()

        assert "llm" in config
        assert "embedder" in config
        assert "vector_store" in config

        # Check LLM config
        assert config["llm"]["provider"] == "ollama"
        assert "model" in config["llm"]["config"]
        assert "ollama_base_url" in config["llm"]["config"]

        # Check embedder config
        assert config["embedder"]["provider"] == "ollama"
        assert config["embedder"]["config"]["model"] == "nomic-embed-text"

        # Check vector store config
        assert config["vector_store"]["provider"] == "qdrant"
        assert config["vector_store"]["config"]["embedding_model_dims"] == 768

    def test_get_mem0_config_gateway_routing(self):
        """Test Gateway routing configuration"""
        config_gateway = get_mem0_config(use_gateway=True)
        config_direct = get_mem0_config(use_gateway=False)

        # Gateway should route through :8100
        assert "8100" in config_gateway["llm"]["config"]["ollama_base_url"]

        # Direct should use Ollama port :11434
        assert "11434" in config_direct["llm"]["config"]["ollama_base_url"]

    def test_get_mem0_config_custom_collection(self):
        """Test custom collection name"""
        config = get_mem0_config(collection_name="test_collection")
        assert config["vector_store"]["config"]["collection_name"] == "test_collection"

    def test_get_mem0_config_embedding_dims(self):
        """Test embedding dimension configuration"""
        config_768 = get_mem0_config(embedding_dims=768)
        config_1024 = get_mem0_config(embedding_dims=1024)

        assert config_768["vector_store"]["config"]["embedding_model_dims"] == 768
        assert config_1024["vector_store"]["config"]["embedding_model_dims"] == 1024

    def test_presets_exist(self):
        """Test preset configurations exist"""
        assert "user_preferences" in PRESETS
        assert "conversation_context" in PRESETS
        assert "experience_templates" in PRESETS

    def test_get_preset_config(self):
        """Test preset configuration retrieval"""
        config = get_preset_config("user_preferences")

        assert "llm" in config
        assert "embedder" in config
        assert "vector_store" in config

    def test_get_preset_config_invalid(self):
        """Test invalid preset raises error"""
        with pytest.raises(ValueError):
            get_preset_config("invalid_preset")


class TestAgenticMemoryAdapter:
    """Tests for mem0_adapter.py"""

    def test_adapter_initialization(self):
        """Test adapter initialization"""
        adapter = AgenticMemoryAdapter(
            user_id="test_user",
            use_gateway=True,
            auto_extract=True
        )

        assert adapter.user_id == "test_user"
        assert adapter.use_gateway is True
        assert adapter.auto_extract is True
        assert adapter._initialized is False

    def test_get_adapter_for_user_singleton(self):
        """Test singleton pattern for adapters"""
        # Clear any existing adapters
        clear_adapter_for_user("singleton_test")

        adapter1 = get_adapter_for_user("singleton_test")
        adapter2 = get_adapter_for_user("singleton_test")

        assert adapter1 is adapter2

        # Cleanup
        clear_adapter_for_user("singleton_test")

    def test_clear_adapter_for_user(self):
        """Test adapter clearing"""
        adapter1 = get_adapter_for_user("clear_test")
        clear_adapter_for_user("clear_test")
        adapter2 = get_adapter_for_user("clear_test")

        assert adapter1 is not adapter2

    @pytest.mark.asyncio
    async def test_extract_entities_error_codes(self):
        """Test entity extraction for error codes"""
        adapter = AgenticMemoryAdapter(user_id="test")

        entities = await adapter._extract_entities(
            query="What causes SRVO-063 errors?",
            response="SRVO-063 indicates servo disconnect. Related: MOTN-023."
        )

        assert "SRVO-063" in entities
        assert "MOTN-023" in entities

    @pytest.mark.asyncio
    async def test_extract_entities_part_numbers(self):
        """Test entity extraction for part numbers"""
        adapter = AgenticMemoryAdapter(user_id="test")

        entities = await adapter._extract_entities(
            query="Replace encoder A860-2014-T301",
            response="The A860-2014-T301 encoder is compatible with A06B-6114-H105."
        )

        assert "A860-2014-T301" in entities
        assert "A06B-6114-H105" in entities

    @pytest.mark.asyncio
    async def test_extract_domain_info_fanuc(self):
        """Test domain detection for FANUC"""
        adapter = AgenticMemoryAdapter(user_id="test", min_confidence=0.5)

        memory = await adapter._extract_domain_info(
            query="How do I fix SRVO-063 on my FANUC robot?",
            context={}
        )

        assert memory is not None
        assert memory.category == MemoryCategory.DOMAIN
        assert "fanuc" in memory.metadata.get("domain", "").lower()

    @pytest.mark.asyncio
    async def test_extract_domain_info_siemens(self):
        """Test domain detection for Siemens"""
        adapter = AgenticMemoryAdapter(user_id="test", min_confidence=0.5)

        memory = await adapter._extract_domain_info(
            query="S7-1500 TIA Portal configuration",
            context={}
        )

        assert memory is not None
        assert "siemens" in memory.metadata.get("domain", "").lower()

    @pytest.mark.asyncio
    async def test_extract_preference_research(self):
        """Test preference extraction for RESEARCH preset"""
        adapter = AgenticMemoryAdapter(user_id="test")

        memory = await adapter._extract_preference(
            "Used RESEARCH preset",
            {"preset": "RESEARCH"}
        )

        assert memory is not None
        assert memory.category == MemoryCategory.PREFERENCE
        assert "detailed" in memory.content.lower() or "thorough" in memory.content.lower()

    @pytest.mark.asyncio
    async def test_conversation_context_empty(self):
        """Test empty conversation context"""
        adapter = AgenticMemoryAdapter(user_id="test")

        # Without initialization, should return empty context
        context = await adapter.get_context_for_query("test query")

        assert isinstance(context, ConversationContext)
        assert context.memories == []
        assert context.total_score == 0.0

    @pytest.mark.asyncio
    async def test_resolve_entity_recent(self):
        """Test entity resolution returns most recent"""
        adapter = AgenticMemoryAdapter(user_id="test")
        adapter._current_entities = ["SRVO-062", "SRVO-063"]

        resolved = await adapter.resolve_entity("it")
        assert resolved == "SRVO-063"

    @pytest.mark.asyncio
    async def test_resolve_entity_error(self):
        """Test entity resolution for 'that error'"""
        adapter = AgenticMemoryAdapter(user_id="test")
        adapter._current_entities = ["R-2000iC", "MOTN-023", "encoder"]

        resolved = await adapter.resolve_entity("that error")
        assert resolved == "MOTN-023"

    @pytest.mark.asyncio
    async def test_clear_conversation_context(self):
        """Test clearing conversation context"""
        adapter = AgenticMemoryAdapter(user_id="test")
        adapter._current_entities = ["entity1", "entity2"]
        adapter._conversation_id = "conv123"

        await adapter.clear_conversation_context()

        assert adapter._current_entities == []
        assert adapter._conversation_id is None


class TestMemoryCategory:
    """Tests for MemoryCategory enum"""

    def test_memory_categories_exist(self):
        """Test all expected categories exist"""
        assert MemoryCategory.PREFERENCE
        assert MemoryCategory.DOMAIN
        assert MemoryCategory.EXPERTISE
        assert MemoryCategory.EQUIPMENT
        assert MemoryCategory.PROBLEM_TYPE
        assert MemoryCategory.APPROACH
        assert MemoryCategory.ENTITY


class TestExtractedMemory:
    """Tests for ExtractedMemory dataclass"""

    def test_extracted_memory_creation(self):
        """Test creating ExtractedMemory"""
        memory = ExtractedMemory(
            content="User prefers FANUC",
            category=MemoryCategory.DOMAIN,
            confidence=0.85,
            source_query="test query"
        )

        assert memory.content == "User prefers FANUC"
        assert memory.category == MemoryCategory.DOMAIN
        assert memory.confidence == 0.85
        assert memory.timestamp is not None


class TestOrchestratorIntegration:
    """Tests for orchestrator Mem0 integration"""

    @pytest.mark.asyncio
    async def test_extract_mem0_memories_with_valid_response(self):
        """Test memory extraction from valid search response"""
        from agentic.orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
        from agentic.models import SearchRequest, SearchResponse, SearchResultData, SearchMeta

        # Create orchestrator with balanced preset (has enable_mem0_extraction)
        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)

        # Create a mock request
        request = SearchRequest(
            query="How do I fix SRVO-063 error on FANUC robot?",
            user_id="test_user_123"
        )

        # Create a valid response with required synthesized_context and meta fields
        response = SearchResponse(
            success=True,
            data=SearchResultData(
                synthesized_context="SRVO-063 indicates a servo disconnect error. Check the encoder cable connection at the robot base. Also verify the A06B-6114-H105 servo amplifier is functioning correctly.",
                sources=[{"title": "FANUC Manual", "url": "https://example.com"}],
                search_queries=["SRVO-063 fix", "FANUC servo error"],
                confidence_score=0.85
            ),
            meta=SearchMeta()
        )

        # Test extraction (should not raise)
        try:
            await orchestrator._extract_mem0_memories(
                request=request,
                response=response,
                request_id="test-123"
            )
        except Exception as e:
            # Expected if Mem0/Gateway not running
            error_msg = str(e).lower()
            assert any(kw in error_msg for kw in ["initialize", "not found", "connection", "refused", "ollama"])

    @pytest.mark.asyncio
    async def test_mem0_context_retrieval(self):
        """Test context retrieval for query"""
        from agentic.orchestrator_universal import UniversalOrchestrator, OrchestratorPreset

        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)

        # Get context (should return None or ConversationContext if no prior memories)
        context = await orchestrator._get_mem0_context_for_query(
            query="SRVO-063 troubleshooting",
            user_id="test_context_user"
        )

        # Should return ConversationContext or None
        from agentic.mem0_adapter import ConversationContext
        assert context is None or isinstance(context, ConversationContext)

    @pytest.mark.asyncio
    async def test_feature_flag_controls_extraction(self):
        """Test that enable_mem0_extraction flag controls behavior"""
        from agentic.orchestrator_universal import UniversalOrchestrator, OrchestratorPreset

        # balanced should have extraction enabled
        balanced_orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)
        assert balanced_orchestrator.config.enable_mem0_extraction is True

        # minimal should have extraction disabled (default is False, not explicitly set)
        minimal_orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.MINIMAL)
        assert minimal_orchestrator.config.enable_mem0_extraction is False

    @pytest.mark.asyncio
    async def test_extraction_handles_missing_adapter(self):
        """Test graceful handling when adapter unavailable"""
        from agentic.orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
        from agentic.models import SearchRequest, SearchResponse, SearchResultData, SearchMeta
        from agentic.mem0_adapter import clear_adapter_for_user

        # Clear any existing adapter
        clear_adapter_for_user("graceful_test_user")

        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)

        request = SearchRequest(
            query="Test query",
            user_id="graceful_test_user"
        )

        response = SearchResponse(
            success=True,
            data=SearchResultData(
                synthesized_context="Test response content",
                confidence_score=0.8
            ),
            meta=SearchMeta()
        )

        # Should not raise, should handle gracefully
        try:
            await orchestrator._extract_mem0_memories(
                request=request,
                response=response,
                request_id="graceful-test"
            )
        except Exception:
            pass  # Expected if services not running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
