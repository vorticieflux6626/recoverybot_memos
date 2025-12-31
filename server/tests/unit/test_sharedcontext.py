"""
Tests for G.6.3 SharedContext Blackboard Pattern

Tests the LbMAS-inspired agent selection and coordination features.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic.scratchpad import (
    AgenticScratchpad,
    QuestionProgress,
    QuestionStatus,
    ScratchpadFinding,
    FindingType,
    AgentNote,
)


class TestSharedContextAgentSelection:
    """Test suite for G.6.3 SharedContext agent selection."""

    @pytest.fixture
    def scratchpad(self):
        """Create a test scratchpad."""
        return AgenticScratchpad.create(
            query="What are the benefits of solar energy?",
            request_id="test-123"
        )

    def test_select_next_agent_default_analyzer(self, scratchpad):
        """Empty scratchpad should select analyzer first."""
        agent = scratchpad.select_next_agent()
        assert agent == "analyzer"

    def test_select_next_agent_with_contradictions(self, scratchpad):
        """Contradictions should trigger verifier selection."""
        scratchpad.contradictions.append({
            "finding_a": "solar is expensive",
            "finding_b": "solar is cheap",
            "severity": "high"
        })
        agent = scratchpad.select_next_agent()
        assert agent == "verifier"

    def test_select_next_agent_with_search_action(self, scratchpad):
        """Pending search action should select searcher."""
        scratchpad.next_actions.append({
            "type": "search",
            "query": "solar energy benefits"
        })
        agent = scratchpad.select_next_agent()
        assert agent == "searcher"

    def test_select_next_agent_with_scrape_action(self, scratchpad):
        """Pending scrape action should select scraper."""
        scratchpad.next_actions.append({
            "type": "scrape",
            "url": "https://example.com/solar"
        })
        agent = scratchpad.select_next_agent()
        assert agent == "scraper"

    def test_select_next_agent_unanswered_questions(self, scratchpad):
        """Unanswered questions should trigger searcher."""
        scratchpad.questions["q1"] = QuestionProgress(
            question_id="q1",
            question_text="What are cost benefits?",
            completion_criteria="List 3 benefits",
            status=QuestionStatus.UNANSWERED
        )
        agent = scratchpad.select_next_agent()
        assert agent == "searcher"

    def test_select_next_agent_partial_questions(self, scratchpad):
        """Partial answers should trigger scraper for more depth."""
        scratchpad.questions["q1"] = QuestionProgress(
            question_id="q1",
            question_text="What are cost benefits?",
            completion_criteria="List 3 benefits",
            status=QuestionStatus.PARTIAL
        )
        agent = scratchpad.select_next_agent()
        assert agent == "scraper"

    def test_select_next_agent_low_confidence_verifier(self, scratchpad):
        """Low confidence with findings should trigger verifier."""
        scratchpad.overall_confidence = 0.4
        scratchpad.findings["f1"] = ScratchpadFinding(
            question_id="q1",
            content="Solar saves money",
            source_url="https://example.com",
            confidence=0.5
        )
        agent = scratchpad.select_next_agent()
        assert agent == "verifier"

    def test_select_next_agent_ready_to_synthesize(self, scratchpad):
        """High confidence with findings should trigger synthesizer."""
        scratchpad.overall_confidence = 0.8
        scratchpad.findings["f1"] = ScratchpadFinding(
            question_id="q1",
            content="Solar saves money",
            source_url="https://example.com",
            confidence=0.9
        )
        agent = scratchpad.select_next_agent()
        assert agent == "synthesizer"

    def test_select_next_agent_with_recommendation(self, scratchpad):
        """Agent note recommendation should be followed."""
        scratchpad.agent_notes.append(AgentNote(
            agent="analyzer",
            action_taken="decomposed query",
            observation="found 3 sub-questions",
            recommendation="search for each",
            for_agent="searcher"
        ))
        scratchpad.questions["q1"] = QuestionProgress(
            question_id="q1",
            question_text="What are benefits?",
            completion_criteria="List benefits",
            status=QuestionStatus.ANSWERED  # Not unanswered to avoid searcher trigger
        )
        scratchpad.overall_confidence = 0.5  # Medium to avoid synthesizer
        agent = scratchpad.select_next_agent()
        assert agent == "searcher"

    def test_get_agent_state_summary(self, scratchpad):
        """State summary should contain all required fields."""
        # Setup some state
        scratchpad.questions["q1"] = QuestionProgress(
            question_id="q1",
            question_text="Test question",
            completion_criteria="Answer it",
            status=QuestionStatus.ANSWERED
        )
        scratchpad.overall_confidence = 0.75
        scratchpad.sources_consulted = 5

        summary = scratchpad.get_agent_state_summary()

        assert "questions_total" in summary
        assert "questions_answered" in summary
        assert "findings_count" in summary
        assert "overall_confidence" in summary
        assert summary["questions_total"] == 1
        assert summary["questions_answered"] == 1
        assert summary["overall_confidence"] == 0.75

    def test_notify_agent_update(self, scratchpad):
        """Agent updates should be written to public space."""
        scratchpad.notify_agent_update(
            agent_id="searcher",
            update_type="search_complete",
            data={"results": 5}
        )

        # Check public space has the update
        update = scratchpad.read_public("last_update_searcher")
        assert update is not None
        assert update["type"] == "search_complete"
        assert update["data"]["results"] == 5

    def test_custom_available_agents(self, scratchpad):
        """Should respect custom agent list."""
        # With limited agents, should pick from available
        scratchpad.contradictions.append({"test": "data"})
        agent = scratchpad.select_next_agent(
            available_agents=["searcher", "synthesizer"]
        )
        # Verifier not in list, so should fall through to other logic
        # Since there are contradictions but verifier unavailable,
        # it checks next_actions, then questions, etc.
        assert agent in ["searcher", "synthesizer"]


class TestPublicPrivateSpaces:
    """Test public/private space functionality."""

    @pytest.fixture
    def scratchpad(self):
        return AgenticScratchpad.create(query="Test query")

    def test_write_read_public(self, scratchpad):
        """Test public space read/write."""
        scratchpad.write_public("agent1", "shared_data", {"key": "value"})
        result = scratchpad.read_public("shared_data")
        assert result == {"key": "value"}

    def test_write_read_private(self, scratchpad):
        """Test private space read/write."""
        scratchpad.write_private("agent1", "secret", "my_secret")
        result = scratchpad.read_private("agent1", "secret")
        assert result == "my_secret"

    def test_private_isolation(self, scratchpad):
        """Private spaces should be isolated per agent."""
        scratchpad.write_private("agent1", "data", "agent1_data")
        scratchpad.write_private("agent2", "data", "agent2_data")

        assert scratchpad.read_private("agent1", "data") == "agent1_data"
        assert scratchpad.read_private("agent2", "data") == "agent2_data"
        assert scratchpad.read_private("agent3", "data") is None

    def test_get_agent_context(self, scratchpad):
        """Agent context should include public and private."""
        scratchpad.write_public("system", "config", {"timeout": 30})
        scratchpad.write_private("searcher", "cache", {"results": []})

        context = scratchpad.get_agent_context("searcher")

        assert "public" in context
        assert "private" in context
        assert context["public"]["config"] == {"timeout": 30}
        assert context["private"]["cache"] == {"results": []}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
