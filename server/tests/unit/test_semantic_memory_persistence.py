"""
Tests for A-MEM Semantic Memory Persistence (G.6.1)

Tests cross-session persistence functionality for the SemanticMemoryNetwork.
"""

import asyncio
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import AsyncMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic.semantic_memory import (
    SemanticMemoryNetwork,
    Memory,
    MemoryType,
    MemoryConnection,
    ConnectionType,
)


class TestSemanticMemoryPersistence:
    """Test suite for A-MEM cross-session persistence."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding generation to avoid LLM calls."""
        # Return a simple 10-dim embedding for testing
        return [0.1] * 10

    def test_database_initialization(self, temp_db_path):
        """Test that database is properly initialized."""
        network = SemanticMemoryNetwork(
            db_path=temp_db_path,
            persist=True,
            auto_connect=False  # Skip embedding for this test
        )

        # Check that database file was created
        assert os.path.exists(temp_db_path)

        # Check tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "memories" in tables
        assert "connections" in tables

        conn.close()

    def test_persistence_disabled(self):
        """Test that persistence can be disabled."""
        # Use a path that doesn't exist
        non_existent_path = "/tmp/test_amem_no_persist_should_not_exist.db"
        if os.path.exists(non_existent_path):
            os.unlink(non_existent_path)

        network = SemanticMemoryNetwork(
            db_path=non_existent_path,
            persist=False,
            auto_connect=False
        )

        # Database file should not be created
        assert not os.path.exists(non_existent_path)

    @pytest.mark.asyncio
    async def test_add_memory_persists(self, temp_db_path, mock_embedding):
        """Test that add_memory saves to database."""
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Add a memory
            memory = await network.add_memory(
                content="SRVO-063 encoder error troubleshooting",
                memory_type=MemoryType.FINDING,
                attributes={"category": "fanuc", "severity": "high"}
            )

            # Check it's in the database
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, memory_type FROM memories WHERE id = ?", (memory.id,))
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            assert row[0] == memory.id
            assert row[1] == "SRVO-063 encoder error troubleshooting"
            assert row[2] == "finding"

    @pytest.mark.asyncio
    async def test_memory_survives_restart(self, temp_db_path, mock_embedding):
        """Test that memories persist across network restarts."""
        memory_id = None

        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            # First session - add memory
            network1 = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            memory = await network1.add_memory(
                content="Test cross-session persistence",
                memory_type=MemoryType.OBSERVATION,
                attributes={"test": True}
            )
            memory_id = memory.id

            # Verify it's in network1
            assert memory_id in network1.memories
            initial_access_count = network1.memories[memory_id].access_count

        # Second session - load from database
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network2 = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Memory should be loaded from database
            assert memory_id in network2.memories
            loaded_memory = network2.memories[memory_id]
            assert loaded_memory.content == "Test cross-session persistence"
            assert loaded_memory.memory_type == MemoryType.OBSERVATION
            assert loaded_memory.attributes.get("test") is True

    @pytest.mark.asyncio
    async def test_get_memory_updates_access(self, temp_db_path, mock_embedding):
        """Test that get_memory updates access statistics in database."""
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Add a memory
            memory = await network.add_memory(
                content="Access test memory",
                memory_type=MemoryType.FINDING
            )
            memory_id = memory.id
            initial_access = memory.access_count

            # Access it multiple times
            network.get_memory(memory_id)
            network.get_memory(memory_id)
            network.get_memory(memory_id)

            # Check database has updated access count
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT access_count FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            assert row[0] == initial_access + 3

    @pytest.mark.asyncio
    async def test_boost_memory_persists(self, temp_db_path, mock_embedding):
        """Test that boost_memory updates are persisted."""
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Add a memory
            memory = await network.add_memory(
                content="Boost test memory",
                memory_type=MemoryType.FINDING
            )
            memory_id = memory.id
            initial_access = memory.access_count

            # Boost it
            network.boost_memory(memory_id, boost_amount=0.5)  # +5 access count

            # Check database has updated access count
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT access_count FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            assert row[0] == initial_access + 5

    @pytest.mark.asyncio
    async def test_consolidate_deletes_from_db(self, temp_db_path, mock_embedding):
        """Test that consolidate_memories removes memories from database."""
        import time

        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Add a memory with old timestamp
            memory = await network.add_memory(
                content="Old weak memory to consolidate",
                memory_type=MemoryType.FINDING
            )
            memory_id = memory.id

            # Manually make it old (8 days ago)
            network.memories[memory_id].created_at = time.time() - (8 * 86400)
            network.memories[memory_id].last_accessed = time.time() - (8 * 86400)
            network.memories[memory_id].access_count = 0

            # Consolidate with high threshold (should remove this memory)
            result = network.consolidate_memories(
                strength_threshold=0.5,  # High threshold
                min_age_days=7.0,
                dry_run=False
            )

            assert result["removed"] >= 1

            # Check it's deleted from database
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()

            assert row is None

    @pytest.mark.asyncio
    async def test_connections_persisted(self, temp_db_path, mock_embedding):
        """Test that memory connections are persisted."""
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Add first memory
            memory1 = await network.add_memory(
                content="First memory",
                memory_type=MemoryType.FINDING
            )

            # Add second memory with explicit connection to first
            memory2 = await network.add_memory(
                content="Second memory related to first",
                memory_type=MemoryType.FINDING,
                explicit_connections=[(memory1.id, ConnectionType.SUPPORTS)]
            )

            # Check connections in database
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT source_id, target_id, connection_type FROM connections WHERE source_id = ?",
                (memory2.id,)
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            assert row[1] == memory1.id
            assert row[2] == "supports"

    @pytest.mark.asyncio
    async def test_connections_loaded_on_restart(self, temp_db_path, mock_embedding):
        """Test that connections are restored on network restart."""
        memory1_id = None
        memory2_id = None

        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            # First session
            network1 = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            memory1 = await network1.add_memory(
                content="Source memory",
                memory_type=MemoryType.SOURCE
            )
            memory1_id = memory1.id

            memory2 = await network1.add_memory(
                content="Finding from source",
                memory_type=MemoryType.FINDING,
                explicit_connections=[(memory1.id, ConnectionType.DERIVED_FROM)]
            )
            memory2_id = memory2.id

        # Second session
        with patch.object(
            SemanticMemoryNetwork,
            "_get_embedding",
            new_callable=AsyncMock,
            return_value=mock_embedding
        ):
            network2 = SemanticMemoryNetwork(
                db_path=temp_db_path,
                persist=True,
                auto_connect=False
            )

            # Check connections were restored
            loaded_memory2 = network2.memories[memory2_id]
            assert len(loaded_memory2.connections) >= 1

            connection = loaded_memory2.connections[0]
            assert connection.target_id == memory1_id
            assert connection.connection_type == ConnectionType.DERIVED_FROM

    def test_stats_include_persistence_info(self, temp_db_path):
        """Test that get_stats returns persistence information."""
        network = SemanticMemoryNetwork(
            db_path=temp_db_path,
            persist=True,
            auto_connect=False
        )

        stats = network.get_stats()
        assert "total_memories" in stats
        assert "total_connections" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
