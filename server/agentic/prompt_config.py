"""
Central Prompt Configuration Loader for memOS Agentic Pipeline.

This module provides typed access to all prompts defined in config/prompts.yaml.
Similar to llm_config.py but for prompt templates.

Usage:
    from agentic.prompt_config import get_prompt_config

    config = get_prompt_config()
    system_prompt = config.agent_prompts.synthesizer.system
    template = config.templates.url_relevance.format(url=url, query=query)

Author: Claude Code
Date: January 2026
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Dataclasses for Typed Access
# =============================================================================

@dataclass
class Instructions:
    """Shared instruction snippets."""
    chain_of_draft: str = ""
    chain_of_draft_short: str = ""
    no_think_suffix: str = " /no_think"
    json_output: str = ""
    industrial_expertise: str = ""
    citation_requirement: str = ""
    cross_domain_constraints: str = ""


@dataclass
class SystemPrompts:
    """Core system-level prompts."""
    core_prefix: str = ""
    industrial_prefix: str = ""


@dataclass
class AnalyzerPrompts:
    """Analyzer agent prompts."""
    system: str = ""
    url_evaluation: str = ""
    content_adequacy: str = ""


@dataclass
class PlannerPrompts:
    """Planner agent prompts."""
    system: str = ""
    query_generation: str = ""


@dataclass
class SynthesizerPrompts:
    """Synthesizer agent prompts."""
    system: str = ""
    main: str = ""
    thinking_instruction: str = ""


@dataclass
class VerifierPrompts:
    """Verifier agent prompts."""
    system: str = ""
    verification: str = ""


@dataclass
class QueryClassifierPrompts:
    """Query classifier prompts."""
    system: str = ""
    classification: str = ""


@dataclass
class RetrievalEvaluatorPrompts:
    """Retrieval evaluator (CRAG) prompts."""
    system: str = ""
    document_scoring: str = ""
    query_refinement: str = ""
    fallback_queries: str = ""
    query_decomposition: str = ""


@dataclass
class SelfReflectionPrompts:
    """Self-reflection agent prompts."""
    system: str = ""
    relevance_check: str = ""
    support_check: str = ""


@dataclass
class EntityTrackerPrompts:
    """Entity tracker prompts."""
    extraction: str = ""


@dataclass
class CrossEncoderPrompts:
    """Cross-encoder reranker prompts."""
    rerank: str = ""


@dataclass
class HyDEPrompts:
    """HyDE (Hypothetical Document Embeddings) prompts."""
    answer: str = ""
    passage: str = ""
    explanation: str = ""
    technical: str = ""
    multi: str = ""


@dataclass
class ReasoningDAGPrompts:
    """Reasoning DAG (Graph-of-Thought) prompts."""
    branch: str = ""
    aggregate: str = ""
    critique: str = ""
    refine: str = ""


@dataclass
class RAGASPrompts:
    """RAGAS evaluation prompts."""
    claim_extraction: str = ""
    claim_verification: str = ""
    answer_relevancy: str = ""
    context_relevancy: str = ""
    question_generation: str = ""


@dataclass
class SufficientContextPrompts:
    """Sufficient context evaluator prompts."""
    evaluation: str = ""


@dataclass
class AdaptiveRefinementPrompts:
    """Adaptive refinement prompts."""
    gap_analysis: str = ""
    adequacy_grading: str = ""


@dataclass
class EnhancedReasoningPrompts:
    """Enhanced reasoning prompts."""
    planning: str = ""
    qa: str = ""
    contradiction_analysis: str = ""


@dataclass
class FLAREPrompts:
    """FLARE retrieval prompts."""
    uncertainty_detection: str = ""
    continuation: str = ""


@dataclass
class InformationBottleneckPrompts:
    """Information bottleneck prompts."""
    analysis: str = ""


@dataclass
class EntropyMonitorPrompts:
    """Entropy monitor prompts."""
    confidence: str = ""


@dataclass
class DomainCorpusPrompts:
    """Domain corpus extraction prompts."""
    extraction: str = ""


@dataclass
class URLRelevanceFilterPrompts:
    """URL relevance filter prompts."""
    system: str = ""
    evaluation: str = ""


@dataclass
class AgentPrompts:
    """Container for all agent-specific prompts."""
    analyzer: AnalyzerPrompts = field(default_factory=AnalyzerPrompts)
    planner: PlannerPrompts = field(default_factory=PlannerPrompts)
    synthesizer: SynthesizerPrompts = field(default_factory=SynthesizerPrompts)
    verifier: VerifierPrompts = field(default_factory=VerifierPrompts)
    query_classifier: QueryClassifierPrompts = field(default_factory=QueryClassifierPrompts)
    retrieval_evaluator: RetrievalEvaluatorPrompts = field(default_factory=RetrievalEvaluatorPrompts)
    self_reflection: SelfReflectionPrompts = field(default_factory=SelfReflectionPrompts)
    entity_tracker: EntityTrackerPrompts = field(default_factory=EntityTrackerPrompts)
    cross_encoder: CrossEncoderPrompts = field(default_factory=CrossEncoderPrompts)
    hyde: HyDEPrompts = field(default_factory=HyDEPrompts)
    reasoning_dag: ReasoningDAGPrompts = field(default_factory=ReasoningDAGPrompts)
    ragas: RAGASPrompts = field(default_factory=RAGASPrompts)
    sufficient_context: SufficientContextPrompts = field(default_factory=SufficientContextPrompts)
    adaptive_refinement: AdaptiveRefinementPrompts = field(default_factory=AdaptiveRefinementPrompts)
    enhanced_reasoning: EnhancedReasoningPrompts = field(default_factory=EnhancedReasoningPrompts)
    flare: FLAREPrompts = field(default_factory=FLAREPrompts)
    information_bottleneck: InformationBottleneckPrompts = field(default_factory=InformationBottleneckPrompts)
    entropy_monitor: EntropyMonitorPrompts = field(default_factory=EntropyMonitorPrompts)
    domain_corpus: DomainCorpusPrompts = field(default_factory=DomainCorpusPrompts)
    url_relevance_filter: URLRelevanceFilterPrompts = field(default_factory=URLRelevanceFilterPrompts)


@dataclass
class Templates:
    """Reusable prompt templates."""
    search_query_generation: str = ""
    url_relevance: str = ""
    content_summary: str = ""
    gap_detection: str = ""
    synthesis_with_sources: str = ""


@dataclass
class PromptConfig:
    """Root configuration object containing all prompts."""
    instructions: Instructions = field(default_factory=Instructions)
    system_prompts: SystemPrompts = field(default_factory=SystemPrompts)
    agent_prompts: AgentPrompts = field(default_factory=AgentPrompts)
    templates: Templates = field(default_factory=Templates)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_agent_prompt(self, agent: str, prompt_type: str) -> str:
        """
        Get a specific agent prompt by name.

        Args:
            agent: Agent name (e.g., 'synthesizer', 'analyzer')
            prompt_type: Prompt type (e.g., 'system', 'main')

        Returns:
            The prompt string, or empty string if not found
        """
        agent_prompts = getattr(self.agent_prompts, agent, None)
        if agent_prompts:
            return getattr(agent_prompts, prompt_type, "")
        return ""

    def get_instruction(self, name: str) -> str:
        """Get a shared instruction by name."""
        return getattr(self.instructions, name, "")

    def get_template(self, name: str) -> str:
        """Get a template by name."""
        return getattr(self.templates, name, "")

    def format_prompt(self, agent: str, prompt_type: str, **kwargs) -> str:
        """
        Get and format a prompt with variable substitution.

        Args:
            agent: Agent name
            prompt_type: Prompt type
            **kwargs: Variables to substitute in the prompt

        Returns:
            Formatted prompt string
        """
        prompt = self.get_agent_prompt(agent, prompt_type)
        if prompt and kwargs:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing variable in prompt {agent}.{prompt_type}: {e}")
                return prompt
        return prompt


# =============================================================================
# Configuration Loading
# =============================================================================

def _get_config_path() -> Path:
    """Get the path to prompts.yaml."""
    # Check environment variable first
    env_path = os.environ.get("MEMOS_PROMPTS_CONFIG")
    if env_path:
        return Path(env_path)

    # Default path relative to this file
    this_dir = Path(__file__).parent
    return this_dir.parent / "config" / "prompts.yaml"


def _load_agent_prompts(data: Dict[str, Any]) -> AgentPrompts:
    """Load agent prompts from YAML data."""
    agent_prompts = AgentPrompts()

    if not data:
        return agent_prompts

    # Map YAML keys to dataclass attributes
    agent_mapping = {
        "analyzer": (AnalyzerPrompts, "analyzer"),
        "planner": (PlannerPrompts, "planner"),
        "synthesizer": (SynthesizerPrompts, "synthesizer"),
        "verifier": (VerifierPrompts, "verifier"),
        "query_classifier": (QueryClassifierPrompts, "query_classifier"),
        "retrieval_evaluator": (RetrievalEvaluatorPrompts, "retrieval_evaluator"),
        "self_reflection": (SelfReflectionPrompts, "self_reflection"),
        "entity_tracker": (EntityTrackerPrompts, "entity_tracker"),
        "cross_encoder": (CrossEncoderPrompts, "cross_encoder"),
        "hyde": (HyDEPrompts, "hyde"),
        "reasoning_dag": (ReasoningDAGPrompts, "reasoning_dag"),
        "ragas": (RAGASPrompts, "ragas"),
        "sufficient_context": (SufficientContextPrompts, "sufficient_context"),
        "adaptive_refinement": (AdaptiveRefinementPrompts, "adaptive_refinement"),
        "enhanced_reasoning": (EnhancedReasoningPrompts, "enhanced_reasoning"),
        "flare": (FLAREPrompts, "flare"),
        "information_bottleneck": (InformationBottleneckPrompts, "information_bottleneck"),
        "entropy_monitor": (EntropyMonitorPrompts, "entropy_monitor"),
        "domain_corpus": (DomainCorpusPrompts, "domain_corpus"),
        "url_relevance_filter": (URLRelevanceFilterPrompts, "url_relevance_filter"),
    }

    for yaml_key, (dataclass_type, attr_name) in agent_mapping.items():
        if yaml_key in data:
            agent_data = data[yaml_key]
            prompt_obj = dataclass_type(**{k: v for k, v in agent_data.items() if hasattr(dataclass_type, k)})
            setattr(agent_prompts, attr_name, prompt_obj)

    return agent_prompts


def load_prompt_config(config_path: Optional[Path] = None) -> PromptConfig:
    """
    Load prompt configuration from YAML file.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        PromptConfig object with all prompts loaded
    """
    path = config_path or _get_config_path()

    if not path.exists():
        logger.warning(f"Prompt config not found at {path}, using defaults")
        return PromptConfig()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        config = PromptConfig()

        # Load instructions
        if "instructions" in data:
            instr_data = data["instructions"]
            config.instructions = Instructions(**{k: v for k, v in instr_data.items() if hasattr(Instructions, k)})

        # Load system prompts
        if "system_prompts" in data:
            sys_data = data["system_prompts"]
            config.system_prompts = SystemPrompts(**{k: v for k, v in sys_data.items() if hasattr(SystemPrompts, k)})

        # Load agent prompts
        if "agent_prompts" in data:
            config.agent_prompts = _load_agent_prompts(data["agent_prompts"])

        # Load templates
        if "templates" in data:
            tmpl_data = data["templates"]
            config.templates = Templates(**{k: v for k, v in tmpl_data.items() if hasattr(Templates, k)})

        # Load metadata
        if "metadata" in data:
            config.metadata = data["metadata"]

        logger.info(f"Loaded prompt config from {path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load prompt config from {path}: {e}")
        return PromptConfig()


# =============================================================================
# Singleton Pattern
# =============================================================================

_prompt_config: Optional[PromptConfig] = None


def get_prompt_config() -> PromptConfig:
    """
    Get the global prompt configuration singleton.

    Returns:
        PromptConfig instance with all prompts loaded from config/prompts.yaml
    """
    global _prompt_config

    if _prompt_config is None:
        _prompt_config = load_prompt_config()

    return _prompt_config


def reload_prompt_config() -> PromptConfig:
    """
    Force reload the prompt configuration.

    Useful for development or when config file has been updated.

    Returns:
        Newly loaded PromptConfig instance
    """
    global _prompt_config
    _prompt_config = load_prompt_config()
    return _prompt_config


# =============================================================================
# Convenience Functions
# =============================================================================

def get_instruction(name: str) -> str:
    """Get a shared instruction by name."""
    return get_prompt_config().get_instruction(name)


def get_agent_prompt(agent: str, prompt_type: str) -> str:
    """Get a specific agent prompt."""
    return get_prompt_config().get_agent_prompt(agent, prompt_type)


def get_template(name: str) -> str:
    """Get a template by name."""
    return get_prompt_config().get_template(name)


def format_prompt(agent: str, prompt_type: str, **kwargs) -> str:
    """Get and format a prompt with variable substitution."""
    return get_prompt_config().format_prompt(agent, prompt_type, **kwargs)
