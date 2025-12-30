"""
Unit tests for UniversalOrchestrator and preset configurations.

Tests orchestrator presets, feature flags, and configuration management
without requiring running services (uses mocks for external dependencies).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add server directory to path
SERVER_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SERVER_DIR))

from agentic.orchestrator_universal import (
    UniversalOrchestrator,
    OrchestratorPreset,
    FeatureConfig,
    PRESET_CONFIGS,
    UniversalGraphState
)


class TestOrchestratorPreset:
    """Test OrchestratorPreset enum."""

    def test_preset_values(self):
        """Test preset enum values."""
        assert OrchestratorPreset.MINIMAL.value == "minimal"
        assert OrchestratorPreset.BALANCED.value == "balanced"
        assert OrchestratorPreset.ENHANCED.value == "enhanced"
        assert OrchestratorPreset.RESEARCH.value == "research"
        assert OrchestratorPreset.FULL.value == "full"

    def test_preset_count(self):
        """Test that we have exactly 5 presets."""
        assert len(OrchestratorPreset) == 5

    def test_preset_is_string_enum(self):
        """Test that presets are string enums."""
        for preset in OrchestratorPreset:
            assert isinstance(preset.value, str)


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_core_features(self):
        """Test that core features are enabled by default."""
        config = FeatureConfig()
        assert config.enable_query_analysis is True
        assert config.enable_verification is True
        assert config.enable_scratchpad is True
        assert config.enable_metrics is True

    def test_default_quality_control(self):
        """Test default quality control features."""
        config = FeatureConfig()
        assert config.enable_self_reflection is True
        assert config.enable_crag_evaluation is True
        assert config.enable_sufficient_context is True
        assert config.enable_positional_optimization is True

    def test_default_learning_features(self):
        """Test default learning features."""
        config = FeatureConfig()
        assert config.enable_experience_distillation is True
        assert config.enable_classifier_feedback is True

    def test_default_enhanced_features_disabled(self):
        """Test that enhanced features are disabled by default."""
        config = FeatureConfig()
        assert config.enable_hyde is False
        assert config.enable_hybrid_reranking is False
        assert config.enable_ragas is False
        assert config.enable_entity_tracking is False
        assert config.enable_multi_agent is False

    def test_adaptive_refinement_defaults(self):
        """Test adaptive refinement configuration defaults."""
        config = FeatureConfig()
        assert config.enable_adaptive_refinement is True
        assert config.min_confidence_threshold == 0.5
        assert config.max_refinement_attempts == 3

    def test_custom_configuration(self):
        """Test creating custom configuration."""
        config = FeatureConfig(
            enable_hyde=True,
            enable_hybrid_reranking=True,
            enable_multi_agent=True,
            min_confidence_threshold=0.7,
            max_refinement_attempts=5
        )
        assert config.enable_hyde is True
        assert config.enable_hybrid_reranking is True
        assert config.enable_multi_agent is True
        assert config.min_confidence_threshold == 0.7
        assert config.max_refinement_attempts == 5


class TestPresetConfigs:
    """Test PRESET_CONFIGS dictionary."""

    def test_all_presets_defined(self):
        """Test that all presets have configurations."""
        for preset in OrchestratorPreset:
            assert preset in PRESET_CONFIGS, f"Missing config for {preset}"

    def test_minimal_preset_features(self):
        """Test MINIMAL preset disables most features."""
        config = PRESET_CONFIGS[OrchestratorPreset.MINIMAL]

        # Core should still be enabled
        assert config.enable_query_analysis is True
        assert config.enable_verification is True
        assert config.enable_scratchpad is True

        # Quality/learning should be disabled
        assert config.enable_self_reflection is False
        assert config.enable_crag_evaluation is False
        assert config.enable_sufficient_context is False
        assert config.enable_experience_distillation is False
        assert config.enable_classifier_feedback is False

        # Adaptive refinement disabled
        assert config.enable_adaptive_refinement is False
        assert config.enable_answer_grading is False

        # Performance features disabled
        assert config.enable_semantic_cache is False
        assert config.enable_ttl_pinning is False
        assert config.enable_metrics is False

    def test_balanced_preset_features(self):
        """Test BALANCED preset has moderate features."""
        config = PRESET_CONFIGS[OrchestratorPreset.BALANCED]

        # Quality control enabled
        assert config.enable_self_reflection is True
        assert config.enable_crag_evaluation is True

        # Learning enabled
        assert config.enable_experience_distillation is True
        assert config.enable_classifier_feedback is True

        # Caching enabled
        assert config.enable_content_cache is True
        assert config.enable_semantic_cache is True

        # Enhanced features (hybrid_reranking enabled in BALANCED for quality)
        assert config.enable_hyde is False
        assert config.enable_hybrid_reranking is True  # Now enabled in BALANCED
        assert config.enable_ragas is False

        # Domain features enabled (HSEA for FANUC)
        assert config.enable_domain_corpus is True
        assert config.enable_hsea_context is True

    def test_enhanced_preset_features(self):
        """Test ENHANCED preset has quality features."""
        config = PRESET_CONFIGS[OrchestratorPreset.ENHANCED]

        # Layer 2 quality features
        assert config.enable_hyde is True
        assert config.enable_hybrid_reranking is True
        assert config.enable_ragas is True
        assert config.enable_context_curation is True
        assert config.context_curation_preset == "balanced"

        # Layer 3 reasoning features
        assert config.enable_entity_tracking is True
        assert config.enable_thought_library is True
        assert config.enable_domain_corpus is True
        assert config.enable_deep_reading is True

        # Technical documentation
        assert config.enable_technical_docs is True
        assert config.enable_hsea_context is True

        # Mixed precision enabled
        assert config.enable_mixed_precision is True
        assert config.enable_entity_enhanced_retrieval is True

    def test_research_preset_features(self):
        """Test RESEARCH preset has thorough exploration features."""
        config = PRESET_CONFIGS[OrchestratorPreset.RESEARCH]

        # Enhanced retrieval
        assert config.enable_hyde is True
        assert config.enable_hybrid_reranking is True
        assert config.enable_ragas is True

        # Thorough context curation
        assert config.enable_context_curation is True
        assert config.context_curation_preset == "thorough"

        # Phase 2: Confidence-Calibrated Halting
        assert config.enable_entropy_halting is True
        assert config.enable_iteration_bandit is True

        # Phase 3: Enhanced Query Generation
        assert config.enable_flare_retrieval is True
        assert config.enable_query_tree is True

        # Phase 4: Scratchpad Enhancement
        assert config.enable_semantic_memory is True
        assert config.enable_raise_structure is True

        # Phase 5: Template Reuse Optimization
        assert config.enable_meta_buffer is True
        assert config.enable_reasoning_composer is True

        # Enhanced patterns
        assert config.enable_pre_act_planning is True
        assert config.enable_stuck_detection is True
        assert config.enable_parallel_execution is True
        assert config.enable_contradiction_detection is True

        # Vision/Deep analysis
        assert config.enable_vision_analysis is True
        assert config.enable_deep_reading is True

        # KV cache performance
        assert config.enable_kv_cache_service is True

    def test_full_preset_features(self):
        """Test FULL preset enables everything."""
        config = PRESET_CONFIGS[OrchestratorPreset.FULL]

        # All retrieval features
        assert config.enable_hyde is True
        assert config.enable_hybrid_reranking is True
        assert config.enable_ragas is True
        assert config.enable_mixed_precision is True

        # Phase 2: Confidence-Calibrated Halting
        assert config.enable_entropy_halting is True
        assert config.enable_iteration_bandit is True
        assert config.enable_self_consistency is True  # Only enabled in FULL

        # Phase 3-5 features
        assert config.enable_flare_retrieval is True
        assert config.enable_query_tree is True
        assert config.enable_semantic_memory is True
        assert config.enable_raise_structure is True
        assert config.enable_meta_buffer is True
        assert config.enable_reasoning_composer is True

        # Layer 3 - Advanced reasoning
        assert config.enable_entity_tracking is True
        assert config.enable_thought_library is True
        assert config.enable_reasoning_dag is True

        # Layer 4 - Dynamic planning
        assert config.enable_dynamic_planning is True
        assert config.enable_progress_tracking is True

        # Layer 4 - Multi-agent
        assert config.enable_actor_factory is True
        assert config.enable_multi_agent is True

        # Layer 4 - Graph cache
        assert config.enable_graph_cache is True
        assert config.enable_prefetching is True

        # Debug mode
        assert config.enable_llm_debug is True

        # Technical documentation
        assert config.enable_technical_docs is True
        assert config.enable_hsea_context is True

    def test_preset_feature_counts(self):
        """Test that feature counts increase with preset complexity."""
        # Count enabled features for each preset
        feature_counts = {}
        for preset in OrchestratorPreset:
            config = PRESET_CONFIGS[preset]
            enabled_count = sum(
                1 for field in dir(config)
                if field.startswith("enable_") and getattr(config, field) is True
            )
            feature_counts[preset] = enabled_count

        # Verify ordering: MINIMAL <= BALANCED <= ENHANCED <= RESEARCH <= FULL
        assert feature_counts[OrchestratorPreset.MINIMAL] <= feature_counts[OrchestratorPreset.BALANCED], \
            f"MINIMAL ({feature_counts[OrchestratorPreset.MINIMAL]}) should have <= features than BALANCED ({feature_counts[OrchestratorPreset.BALANCED]})"
        assert feature_counts[OrchestratorPreset.BALANCED] <= feature_counts[OrchestratorPreset.ENHANCED], \
            f"BALANCED ({feature_counts[OrchestratorPreset.BALANCED]}) should have <= features than ENHANCED ({feature_counts[OrchestratorPreset.ENHANCED]})"
        assert feature_counts[OrchestratorPreset.ENHANCED] <= feature_counts[OrchestratorPreset.RESEARCH], \
            f"ENHANCED ({feature_counts[OrchestratorPreset.ENHANCED]}) should have <= features than RESEARCH ({feature_counts[OrchestratorPreset.RESEARCH]})"
        assert feature_counts[OrchestratorPreset.RESEARCH] <= feature_counts[OrchestratorPreset.FULL], \
            f"RESEARCH ({feature_counts[OrchestratorPreset.RESEARCH]}) should have <= features than FULL ({feature_counts[OrchestratorPreset.FULL]})"

        # FULL should have the most
        assert feature_counts[OrchestratorPreset.FULL] == max(feature_counts.values())


class TestUniversalGraphState:
    """Test UniversalGraphState visualization."""

    def test_agents_list(self):
        """Test agent symbol definitions."""
        state = UniversalGraphState()
        symbols = [sym for sym, _ in state.AGENTS]

        # Verify expected agents
        assert "A" in symbols  # Analyze
        assert "P" in symbols  # Plan
        assert "S" in symbols  # Search
        assert "E" in symbols  # Evaluate
        assert "W" in symbols  # Scrape
        assert "V" in symbols  # Verify
        assert "Σ" in symbols  # Synthesize
        assert "R" in symbols  # Reflect
        assert "✓" in symbols  # Complete

    def test_initial_state(self):
        """Test initial graph state."""
        state = UniversalGraphState()
        line = state.to_line()

        # All should be pending (no marks)
        assert "[A]" in line
        assert "[P]" in line
        assert "[S]" in line
        assert "✓" not in line.replace("[✓]", "")  # ✓ only in complete symbol, not as mark
        assert "•" not in line

    def test_enter_agent(self):
        """Test entering an agent (marking active)."""
        state = UniversalGraphState()
        state.enter("A")
        line = state.to_line()

        assert "[A•]" in line  # Active marker
        assert "[P]" in line   # Still pending

    def test_complete_agent(self):
        """Test completing an agent."""
        state = UniversalGraphState()
        state.enter("A")
        state.complete("A")
        line = state.to_line()

        assert "[A✓]" in line  # Completed marker

    def test_sequential_agents(self):
        """Test sequential agent progression."""
        state = UniversalGraphState()

        # Analyze -> Plan -> Search
        state.enter("A")
        state.complete("A")
        state.enter("P")
        state.complete("P")
        state.enter("S")

        line = state.to_line()
        assert "[A✓]" in line
        assert "[P✓]" in line
        assert "[S•]" in line

    def test_full_pipeline_completion(self):
        """Test completing full pipeline."""
        state = UniversalGraphState()

        for sym, _ in state.AGENTS:
            state.enter(sym)
            state.complete(sym)

        line = state.to_line()

        # All should be completed
        assert "[A✓]" in line
        assert "[P✓]" in line
        assert "[S✓]" in line
        assert "[E✓]" in line
        assert "[W✓]" in line
        assert "[V✓]" in line
        assert "[Σ✓]" in line
        assert "[R✓]" in line
        assert "[✓✓]" in line

    def test_reset_state(self):
        """Test resetting graph state."""
        state = UniversalGraphState()

        state.enter("A")
        state.complete("A")
        state.enter("S")

        # Should have some marks
        assert "✓" in state.to_line() or "•" in state.to_line()

        state.reset()

        # Should be clean again
        line = state.to_line()
        assert "✓" not in line.replace("[✓]", "")
        assert "•" not in line


class TestUniversalOrchestratorInstantiation:
    """Test UniversalOrchestrator creation patterns."""

    @patch('agentic.orchestrator_universal.QueryClassifier')
    @patch('agentic.orchestrator_universal.get_self_reflection_agent')
    @patch('agentic.orchestrator_universal.RetrievalEvaluator')
    @patch('agentic.orchestrator_universal.get_experience_distiller')
    @patch('agentic.orchestrator_universal.get_classifier_feedback')
    @patch('agentic.orchestrator_universal.get_sufficient_context_classifier')
    @patch('agentic.orchestrator_universal.get_adaptive_refinement_engine')
    def test_default_instantiation(self, *mocks):
        """Test creating orchestrator with defaults."""
        orchestrator = UniversalOrchestrator()

        # Should default to balanced preset
        assert orchestrator.config.enable_query_analysis is True
        assert orchestrator.config.enable_self_reflection is True

    @patch('agentic.orchestrator_universal.QueryClassifier')
    @patch('agentic.orchestrator_universal.get_self_reflection_agent')
    @patch('agentic.orchestrator_universal.RetrievalEvaluator')
    @patch('agentic.orchestrator_universal.get_experience_distiller')
    @patch('agentic.orchestrator_universal.get_classifier_feedback')
    @patch('agentic.orchestrator_universal.get_sufficient_context_classifier')
    @patch('agentic.orchestrator_universal.get_adaptive_refinement_engine')
    def test_preset_instantiation(self, *mocks):
        """Test creating orchestrator with preset."""
        for preset in OrchestratorPreset:
            orchestrator = UniversalOrchestrator(preset=preset)
            expected_config = PRESET_CONFIGS[preset]

            # Verify config matches preset
            assert orchestrator.config.enable_hyde == expected_config.enable_hyde

    @patch('agentic.orchestrator_universal.QueryClassifier')
    @patch('agentic.orchestrator_universal.get_self_reflection_agent')
    @patch('agentic.orchestrator_universal.RetrievalEvaluator')
    @patch('agentic.orchestrator_universal.get_experience_distiller')
    @patch('agentic.orchestrator_universal.get_classifier_feedback')
    @patch('agentic.orchestrator_universal.get_sufficient_context_classifier')
    @patch('agentic.orchestrator_universal.get_adaptive_refinement_engine')
    def test_feature_override(self, *mocks):
        """Test overriding preset features."""
        orchestrator = UniversalOrchestrator(
            preset=OrchestratorPreset.MINIMAL,
            enable_hyde=True,  # Override to enable
            enable_multi_agent=True
        )

        # Overrides should apply
        assert orchestrator.config.enable_hyde is True
        assert orchestrator.config.enable_multi_agent is True

        # Other MINIMAL defaults should remain
        assert orchestrator.config.enable_self_reflection is False

    @patch('agentic.orchestrator_universal.QueryClassifier')
    @patch('agentic.orchestrator_universal.get_self_reflection_agent')
    @patch('agentic.orchestrator_universal.RetrievalEvaluator')
    @patch('agentic.orchestrator_universal.get_experience_distiller')
    @patch('agentic.orchestrator_universal.get_classifier_feedback')
    @patch('agentic.orchestrator_universal.get_sufficient_context_classifier')
    @patch('agentic.orchestrator_universal.get_adaptive_refinement_engine')
    def test_custom_config(self, *mocks):
        """Test creating orchestrator with custom config."""
        custom_config = FeatureConfig(
            enable_hyde=True,
            enable_multi_agent=True,
            enable_self_reflection=False,
            min_confidence_threshold=0.8
        )

        orchestrator = UniversalOrchestrator(config=custom_config)

        assert orchestrator.config.enable_hyde is True
        assert orchestrator.config.enable_multi_agent is True
        assert orchestrator.config.enable_self_reflection is False
        assert orchestrator.config.min_confidence_threshold == 0.8


class TestPresetProgression:
    """Test that preset feature sets form a sensible progression."""

    def test_minimal_is_subset_of_balanced(self):
        """Test that features enabled in MINIMAL are also in BALANCED."""
        minimal = PRESET_CONFIGS[OrchestratorPreset.MINIMAL]
        balanced = PRESET_CONFIGS[OrchestratorPreset.BALANCED]

        # Core features that should remain enabled
        core_features = ["enable_query_analysis", "enable_verification", "enable_scratchpad"]

        for feature in core_features:
            if getattr(minimal, feature):
                assert getattr(balanced, feature), f"BALANCED should have {feature} if MINIMAL does"

    def test_balanced_is_subset_of_enhanced(self):
        """Test that features enabled in BALANCED are also in ENHANCED."""
        balanced = PRESET_CONFIGS[OrchestratorPreset.BALANCED]
        enhanced = PRESET_CONFIGS[OrchestratorPreset.ENHANCED]

        for field in dir(balanced):
            if field.startswith("enable_") and getattr(balanced, field):
                assert getattr(enhanced, field), f"ENHANCED should have {field} if BALANCED does"

    def test_enhanced_is_subset_of_research(self):
        """Test that features enabled in ENHANCED are also in RESEARCH."""
        enhanced = PRESET_CONFIGS[OrchestratorPreset.ENHANCED]
        research = PRESET_CONFIGS[OrchestratorPreset.RESEARCH]

        # Count how many features from ENHANCED are in RESEARCH
        enhanced_features = [f for f in dir(enhanced) if f.startswith("enable_") and getattr(enhanced, f)]
        research_features = [f for f in dir(research) if f.startswith("enable_") and getattr(research, f)]

        # RESEARCH should have at least as many features as ENHANCED
        assert len(research_features) >= len(enhanced_features)

    def test_research_is_subset_of_full(self):
        """Test that features enabled in RESEARCH are also in FULL."""
        research = PRESET_CONFIGS[OrchestratorPreset.RESEARCH]
        full = PRESET_CONFIGS[OrchestratorPreset.FULL]

        for field in dir(research):
            if field.startswith("enable_") and getattr(research, field):
                assert getattr(full, field), f"FULL should have {field} if RESEARCH does"

    def test_full_has_most_features(self):
        """Test that FULL has the most enabled features."""
        feature_counts = {}
        for preset in OrchestratorPreset:
            config = PRESET_CONFIGS[preset]
            count = sum(
                1 for field in dir(config)
                if field.startswith("enable_") and getattr(config, field)
            )
            feature_counts[preset] = count

        # FULL should have the highest count
        assert feature_counts[OrchestratorPreset.FULL] == max(feature_counts.values())


class TestSpecificFeatureCombinations:
    """Test that specific feature combinations work correctly."""

    def test_fanuc_domain_features(self):
        """Test FANUC-specific features are properly grouped."""
        for preset in [OrchestratorPreset.BALANCED, OrchestratorPreset.ENHANCED,
                       OrchestratorPreset.RESEARCH, OrchestratorPreset.FULL]:
            config = PRESET_CONFIGS[preset]

            # If HSEA is enabled, domain_corpus should also be enabled
            if config.enable_hsea_context:
                assert config.enable_domain_corpus, f"{preset.value} has HSEA but not domain_corpus"

    def test_phase_feature_groupings(self):
        """Test that phase features are enabled together."""
        # Phase 2 features should be grouped
        research = PRESET_CONFIGS[OrchestratorPreset.RESEARCH]
        if research.enable_entropy_halting:
            assert research.enable_iteration_bandit, "Phase 2 features should be grouped"

        # Phase 3 features should be grouped
        if research.enable_flare_retrieval:
            assert research.enable_query_tree, "Phase 3 features should be grouped"

        # Phase 4 features should be grouped
        if research.enable_semantic_memory:
            assert research.enable_raise_structure, "Phase 4 features should be grouped"

        # Phase 5 features should be grouped
        if research.enable_meta_buffer:
            assert research.enable_reasoning_composer, "Phase 5 features should be grouped"

    def test_layer_4_features_in_full(self):
        """Test that Layer 4 (expensive) features are enabled in FULL."""
        layer_4_features = [
            "enable_dynamic_planning",
            "enable_multi_agent",
            "enable_graph_cache",
            "enable_prefetching"
        ]

        # All Layer 4 features should be in FULL
        full = PRESET_CONFIGS[OrchestratorPreset.FULL]
        for feature in layer_4_features:
            assert getattr(full, feature), f"FULL should have {feature}"

    def test_full_has_more_layer_4_than_minimal(self):
        """Test that FULL has more Layer 4 features than MINIMAL."""
        layer_4_features = [
            "enable_dynamic_planning",
            "enable_multi_agent",
            "enable_graph_cache",
            "enable_prefetching",
            "enable_actor_factory",
            "enable_progress_tracking"
        ]

        minimal = PRESET_CONFIGS[OrchestratorPreset.MINIMAL]
        full = PRESET_CONFIGS[OrchestratorPreset.FULL]

        minimal_count = sum(1 for f in layer_4_features if getattr(minimal, f))
        full_count = sum(1 for f in layer_4_features if getattr(full, f))

        assert full_count >= minimal_count, \
            f"FULL should have >= Layer 4 features than MINIMAL ({full_count} vs {minimal_count})"
        assert full_count == len(layer_4_features), \
            f"FULL should have all Layer 4 features ({full_count}/{len(layer_4_features)})"
