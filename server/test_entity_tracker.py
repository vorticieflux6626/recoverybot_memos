#!/usr/bin/env python3
"""
Phase 2 Test Suite: GSW-Style Entity Tracking

Tests:
1. EntityTracker - Entity extraction, reconciliation, summaries
2. Scratchpad Integration - Entity tracking in working memory
3. Analyzer Integration - Entity extraction during content analysis

Usage:
    python test_entity_tracker.py           # Run all tests
    python test_entity_tracker.py --full    # Include LLM-based extraction test
"""

import asyncio
import sys
import json
import time
from datetime import datetime, timezone

# Add parent directory for imports
sys.path.insert(0, "/home/sparkone/sdd/Recovery_Bot/memOS/server")

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def log(msg: str, color: str = RESET) -> None:
    """Log with color"""
    print(f"{color}{msg}{RESET}")


async def test_entity_tracker_basic() -> bool:
    """Test 1: EntityTracker basic operations (no LLM)"""
    log("\n=== Test 1: EntityTracker Basic Operations ===", BLUE)

    from agentic.entity_tracker import (
        EntityTracker, EntityState, EntityType, RoleType, RelationType,
        RoleAssignment, EntityEvent, EntityRelation
    )

    tracker = EntityTracker(ollama_url="http://localhost:11434")

    # Create test entities manually
    entity1 = EntityState(
        name="FastAPI",
        entity_type=EntityType.TECHNOLOGY
    )
    entity1.add_alias("fastapi")
    entity1.add_attribute("version", "0.100.0")
    entity1.add_attribute("language", "Python")
    entity1.add_role(RoleType.FEATURE, "web framework", confidence=0.9)
    entity1.add_event("Released version 0.100.0", event_type="observation")

    entity2 = EntityState(
        name="Sebastian Ramirez",
        entity_type=EntityType.PERSON
    )
    entity2.add_alias("tiangolo")
    entity2.add_role(RoleType.CREATOR, "FastAPI", confidence=0.95)

    # Test reconciliation
    tracker.reconcile([entity1, entity2])

    # Verify entities were added
    assert len(tracker.entities) == 2, f"Expected 2 entities, got {len(tracker.entities)}"
    log(f"  ✓ Added 2 entities", GREEN)

    # Test entity lookup
    found = tracker.find_entity_by_name("FastAPI")
    assert found is not None, "Failed to find FastAPI entity"
    assert found.name == "FastAPI", "Wrong entity name"
    log(f"  ✓ Entity lookup works", GREEN)

    # Test alias lookup
    found_alias = tracker.find_entity_by_name("tiangolo")
    assert found_alias is not None, "Failed to find entity by alias"
    assert found_alias.name == "Sebastian Ramirez", "Wrong entity for alias"
    log(f"  ✓ Alias lookup works", GREEN)

    # Test summary generation
    summary = entity1.generate_summary()
    assert "FastAPI" in summary, "Summary missing entity name"
    assert "technology" in summary.lower(), "Summary missing entity type"
    log(f"  ✓ Summary generation works: {summary[:60]}...", GREEN)

    # Test workspace context generation
    context = tracker.generate_workspace_context("python web framework")
    assert len(context) > 0, "Empty workspace context"
    log(f"  ✓ Workspace context generation works ({len(context)} chars)", GREEN)

    # Test entity merging (add same entity again)
    entity1_dup = EntityState(name="FastAPI", entity_type=EntityType.TECHNOLOGY)
    entity1_dup.add_attribute("async", True)
    tracker.reconcile([entity1_dup])

    # Should have merged, not added
    assert len(tracker.entities) == 2, f"Expected 2 entities after merge, got {len(tracker.entities)}"
    fastapi = tracker.find_entity_by_name("FastAPI")
    assert fastapi.attributes.get("async") == True, "Merge didn't update attributes"
    log(f"  ✓ Entity merging works", GREEN)

    # Test stats
    stats = tracker.get_stats()
    assert stats["total_entities"] == 2, "Wrong entity count in stats"
    assert stats["entities_merged"] >= 1, "Merge not counted in stats"
    log(f"  ✓ Stats: {stats}", GREEN)

    log("Test 1: PASS", GREEN)
    return True


async def test_scratchpad_integration() -> bool:
    """Test 2: Scratchpad entity tracking integration"""
    log("\n=== Test 2: Scratchpad Entity Integration ===", BLUE)

    from agentic.scratchpad import AgenticScratchpad

    # Create scratchpad
    scratchpad = AgenticScratchpad(
        original_query="What is FastAPI?",
        session_id="test-session"
    )

    # Test adding entities to scratchpad
    entity_data = {
        "id": "e1",
        "name": "FastAPI",
        "entity_type": "technology",
        "aliases": ["fastapi"],
        "roles": [{"role": "feature", "context": "web development", "confidence": 0.9}],
        "states": ["popular", "maintained"],
        "attributes": {"version": "0.100.0"},
        "mention_count": 5
    }

    scratchpad.add_entity(entity_data)
    assert len(scratchpad.tracked_entities) == 1, "Entity not added to scratchpad"
    log(f"  ✓ Added entity to scratchpad", GREEN)

    # Test entity retrieval
    retrieved = scratchpad.get_entity("e1")
    assert retrieved is not None, "Failed to retrieve entity"
    assert retrieved["name"] == "FastAPI", "Wrong entity data"
    log(f"  ✓ Entity retrieval works", GREEN)

    # Test find by name
    found = scratchpad.find_entity_by_name("FastAPI")
    assert found is not None, "Failed to find by name"
    log(f"  ✓ Find by name works", GREEN)

    # Test alias search
    found_alias = scratchpad.find_entity_by_name("fastapi")
    assert found_alias is not None, "Failed to find by alias"
    log(f"  ✓ Find by alias works", GREEN)

    # Test entity relations
    relation_data = {
        "relation_id": "r1",
        "source_entity_id": "e1",
        "target_entity_id": "e2",
        "relation_type": "depends_on",
        "description": "FastAPI depends on Starlette"
    }
    scratchpad.add_entity_relation(relation_data)
    assert len(scratchpad.entity_relations) == 1, "Relation not added"
    log(f"  ✓ Entity relations work", GREEN)

    # Add second entity for relevance testing
    entity2_data = {
        "id": "e2",
        "name": "Starlette",
        "entity_type": "technology",
        "aliases": [],
        "roles": [{"role": "dependency", "context": "ASGI framework", "confidence": 0.85}],
        "states": ["stable"],
        "attributes": {},
        "mention_count": 3
    }
    scratchpad.add_entity(entity2_data)

    # Test relevance scoring
    relevant = scratchpad.get_relevant_entities("FastAPI web framework", limit=5)
    assert len(relevant) > 0, "No relevant entities found"
    assert relevant[0]["name"] == "FastAPI", "FastAPI should be most relevant"
    log(f"  ✓ Relevance scoring works", GREEN)

    # Test entity context generation
    context = scratchpad.generate_entity_context("python async web")
    assert len(context) > 0, "Empty entity context"
    assert "FastAPI" in context, "FastAPI missing from context"
    log(f"  ✓ Entity context generation: {len(context)} chars", GREEN)

    # Test entity stats
    stats = scratchpad.get_entity_stats()
    assert stats["total_entities"] == 2, "Wrong entity count"
    assert stats["total_relations"] == 1, "Wrong relation count"
    log(f"  ✓ Entity stats: {stats}", GREEN)

    log("Test 2: PASS", GREEN)
    return True


async def test_analyzer_integration() -> bool:
    """Test 3: Analyzer entity extraction integration"""
    log("\n=== Test 3: Analyzer Entity Integration ===", BLUE)

    from agentic.analyzer import QueryAnalyzer
    from agentic.entity_tracker import EntityTracker

    # Create analyzer without entity tracker
    analyzer = QueryAnalyzer(ollama_url="http://localhost:11434")

    # Verify entity extraction is disabled by default
    assert analyzer._entity_extraction_enabled == False, "Entity extraction should be disabled by default"
    log(f"  ✓ Entity extraction disabled by default", GREEN)

    # Enable entity extraction
    tracker = EntityTracker(ollama_url="http://localhost:11434")
    analyzer.enable_entity_extraction(tracker)

    assert analyzer._entity_extraction_enabled == True, "Entity extraction should be enabled"
    assert analyzer.entity_tracker is not None, "Tracker should be set"
    log(f"  ✓ Entity extraction enabled", GREEN)

    # Test entity stats retrieval
    stats = analyzer.get_entity_stats()
    assert stats["enabled"] == True, "Stats should show enabled"
    log(f"  ✓ Entity stats accessible: {stats}", GREEN)

    # Test entity context generation (empty tracker)
    context = analyzer.generate_entity_context("test query")
    assert "No entities tracked" in context, "Should indicate no entities"
    log(f"  ✓ Empty context handled correctly", GREEN)

    # Add some entities manually for context test
    from agentic.entity_tracker import EntityState, EntityType
    entity = EntityState(name="Python", entity_type=EntityType.TECHNOLOGY)
    entity.add_attribute("version", "3.12")
    tracker.reconcile([entity])

    context = analyzer.generate_entity_context("programming language")
    assert len(context) > 0, "Context should not be empty"
    log(f"  ✓ Context generation with entities: {len(context)} chars", GREEN)

    # Test disable
    analyzer.disable_entity_extraction()
    assert analyzer._entity_extraction_enabled == False, "Should be disabled"
    log(f"  ✓ Entity extraction can be disabled", GREEN)

    log("Test 3: PASS", GREEN)
    return True


async def test_llm_extraction(skip: bool = False) -> bool:
    """Test 4: LLM-based entity extraction (requires Ollama)"""
    if skip:
        log("\n=== Test 4: LLM Entity Extraction (SKIPPED) ===", YELLOW)
        log("  Use --full flag to run this test", YELLOW)
        return True

    log("\n=== Test 4: LLM Entity Extraction ===", BLUE)

    from agentic.entity_tracker import EntityTracker

    tracker = EntityTracker(
        ollama_url="http://localhost:11434",
        extraction_model="qwen3:8b"
    )

    # Test content for extraction
    test_content = """
    FastAPI is a modern, fast web framework for building APIs with Python 3.7+.
    It was created by Sebastian Ramirez (also known as tiangolo) in 2018.
    FastAPI is built on top of Starlette for the web parts and Pydantic for data validation.
    It provides automatic OpenAPI documentation and is one of the fastest Python frameworks available.
    Key features include dependency injection, async support, and excellent IDE support.
    """

    log("  Extracting entities from test content...", YELLOW)
    start = time.time()

    try:
        entities = await tracker.extract_entities(
            content=test_content,
            source_url="https://example.com/fastapi"
        )
        duration = time.time() - start

        if entities:
            log(f"  ✓ Extracted {len(entities)} entities in {duration:.1f}s", GREEN)
            for entity in entities[:5]:
                log(f"    - {entity.name} ({entity.entity_type.value})", GREEN)

            # Reconcile entities
            tracker.reconcile(entities)

            # Generate workspace context
            context = tracker.generate_workspace_context("python web framework")
            log(f"  ✓ Generated workspace context ({len(context)} chars)", GREEN)
            log(f"    Preview: {context[:200]}...", GREEN)

            stats = tracker.get_stats()
            log(f"  ✓ Final stats: {stats}", GREEN)

            log("Test 4: PASS", GREEN)
            return True
        else:
            log(f"  ✗ No entities extracted (LLM may have failed)", RED)
            log("Test 4: FAIL (no entities)", RED)
            return False

    except Exception as e:
        log(f"  ✗ Extraction failed: {e}", RED)
        log("Test 4: FAIL (exception)", RED)
        return False


async def test_module_imports() -> bool:
    """Test 0: Module imports and exports"""
    log("\n=== Test 0: Module Imports ===", BLUE)

    try:
        from agentic import (
            EntityTracker,
            EntityState,
            EntityType,
            RoleType,
            RelationType,
            RoleAssignment,
            EntityEvent,
            EntityRelation,
            VerbFrame,
            create_entity_tracker,
            __version__
        )

        log(f"  ✓ All entity_tracker exports imported successfully", GREEN)
        log(f"  ✓ Module version: {__version__}", GREEN)

        # Test factory function
        tracker = create_entity_tracker()
        assert isinstance(tracker, EntityTracker), "Factory should return EntityTracker"
        log(f"  ✓ create_entity_tracker() factory works", GREEN)

        log("Test 0: PASS", GREEN)
        return True

    except ImportError as e:
        log(f"  ✗ Import failed: {e}", RED)
        log("Test 0: FAIL", RED)
        return False


async def run_all_tests(full: bool = False) -> None:
    """Run all Phase 2 tests"""
    log("\n" + "=" * 60, BLUE)
    log("  Phase 2 Test Suite: GSW-Style Entity Tracking", BLUE)
    log("=" * 60, BLUE)

    results = {}

    # Test 0: Module imports
    results["imports"] = await test_module_imports()

    # Test 1: Basic EntityTracker
    results["entity_tracker"] = await test_entity_tracker_basic()

    # Test 2: Scratchpad integration
    results["scratchpad"] = await test_scratchpad_integration()

    # Test 3: Analyzer integration
    results["analyzer"] = await test_analyzer_integration()

    # Test 4: LLM extraction (optional)
    results["llm_extraction"] = await test_llm_extraction(skip=not full)

    # Summary
    log("\n" + "=" * 60, BLUE)
    log("  Test Summary", BLUE)
    log("=" * 60, BLUE)

    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        log(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    log(f"\nPassed: {passed}/{passed + failed}")

    if failed > 0:
        log(f"\n{RED}Some tests failed!{RESET}")
        sys.exit(1)
    else:
        log(f"\n{GREEN}All tests passed!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    full = "--full" in sys.argv
    asyncio.run(run_all_tests(full=full))
