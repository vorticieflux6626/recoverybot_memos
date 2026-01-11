"""
Prompt Registry for Cache Optimization

DESIGN PRINCIPLES:
1. Static content at the beginning of every prompt (cacheable)
2. Consistent ordering: system → tools → context → query
3. Modular composition for different agent types

Based on KV_CACHE_IMPLEMENTATION_PLAN.md Phase 1.4
Ref: Anthropic Multi-Agent Research System patterns

NOTE: All prompts are now loaded from config/prompts.yaml via prompt_config.py.
This file provides backward-compatible access to the central configuration.
"""

from typing import Optional, Dict

# Import central prompt configuration
from .prompt_config import get_prompt_config


# =============================================================================
# Backward-Compatible Exports (read from central config)
# =============================================================================

def _get_core_system_prefix() -> str:
    """Get core system prefix from central config."""
    return get_prompt_config().system_prompts.core_prefix


def _get_chain_of_draft() -> str:
    """Get chain-of-draft instruction from central config."""
    return get_prompt_config().instructions.chain_of_draft


# Lazy-loaded properties for backward compatibility
# These read from prompts.yaml on first access
class _PromptProxy:
    """Proxy object that loads prompts from central config on access."""

    @property
    def CORE_SYSTEM_PREFIX(self) -> str:
        return get_prompt_config().system_prompts.core_prefix

    @property
    def CHAIN_OF_DRAFT_INSTRUCTION(self) -> str:
        return get_prompt_config().instructions.chain_of_draft

    @property
    def ANALYZER_SUFFIX(self) -> str:
        config = get_prompt_config()
        return f"\n{config.agent_prompts.analyzer.system}"

    @property
    def PLANNER_SUFFIX(self) -> str:
        config = get_prompt_config()
        return f"\n{config.agent_prompts.planner.system}"

    @property
    def SYNTHESIZER_SUFFIX(self) -> str:
        config = get_prompt_config()
        return f"\n{config.agent_prompts.synthesizer.system}"

    @property
    def VERIFIER_SUFFIX(self) -> str:
        config = get_prompt_config()
        return f"\n{config.agent_prompts.verifier.system}"

    @property
    def COVERAGE_SUFFIX(self) -> str:
        # Coverage evaluator uses sufficient_context prompts
        config = get_prompt_config()
        return f"\n{config.agent_prompts.sufficient_context.evaluation}"

    @property
    def URL_EVALUATOR_SUFFIX(self) -> str:
        config = get_prompt_config()
        return f"\n{config.agent_prompts.analyzer.url_evaluation}"

_proxy = _PromptProxy()

# For backward compatibility, expose as module-level constants via __getattr__
# This allows lazy loading from prompts.yaml on first access
def __getattr__(name: str):
    """Module-level attribute access for backward compatibility."""
    if name == "CORE_SYSTEM_PREFIX":
        return _proxy.CORE_SYSTEM_PREFIX
    elif name == "CHAIN_OF_DRAFT_INSTRUCTION":
        return _proxy.CHAIN_OF_DRAFT_INSTRUCTION
    elif name == "ANALYZER_SUFFIX":
        return _proxy.ANALYZER_SUFFIX
    elif name == "PLANNER_SUFFIX":
        return _proxy.PLANNER_SUFFIX
    elif name == "SYNTHESIZER_SUFFIX":
        return _proxy.SYNTHESIZER_SUFFIX
    elif name == "VERIFIER_SUFFIX":
        return _proxy.VERIFIER_SUFFIX
    elif name == "COVERAGE_SUFFIX":
        return _proxy.COVERAGE_SUFFIX
    elif name == "URL_EVALUATOR_SUFFIX":
        return _proxy.URL_EVALUATOR_SUFFIX
    elif name == "AGENT_SUFFIXES":
        return _get_agent_suffixes()
    elif name == "TEMPLATES":
        return _get_templates()
    raise AttributeError(f"module 'prompts' has no attribute '{name}'")


def _get_agent_suffixes() -> Dict[str, str]:
    """Build agent suffixes dict from central config."""
    return {
        "analyzer": _proxy.ANALYZER_SUFFIX,
        "planner": _proxy.PLANNER_SUFFIX,
        "synthesizer": _proxy.SYNTHESIZER_SUFFIX,
        "verifier": _proxy.VERIFIER_SUFFIX,
        "coverage": _proxy.COVERAGE_SUFFIX,
        "url_evaluator": _proxy.URL_EVALUATOR_SUFFIX,
    }

# Agent type to suffix mapping is now provided via __getattr__


def build_prompt(
    agent_type: str,
    dynamic_context: str = "",
    use_chain_of_draft: bool = True,
    custom_suffix: Optional[str] = None
) -> str:
    """
    Build prompt with consistent prefix structure for cache optimization.

    The CORE_SYSTEM_PREFIX is always first, ensuring maximum KV cache
    reuse across different requests with the same model.

    Args:
        agent_type: One of: analyzer, planner, synthesizer, verifier, coverage, url_evaluator
        dynamic_context: Variable content (user query, search results, etc.)
        use_chain_of_draft: Include CoD instruction for thinking models
        custom_suffix: Override the default suffix for this agent type

    Returns:
        Complete prompt with static prefix + agent suffix + dynamic content
    """
    config = get_prompt_config()

    # Static prefix (highest cache hit potential)
    prompt = config.system_prompts.core_prefix

    # Agent-specific suffix
    agent_suffixes = _get_agent_suffixes()
    suffix = custom_suffix or agent_suffixes.get(agent_type, "")
    if suffix:
        prompt += suffix

    # Chain-of-Draft instruction (for thinking models)
    if use_chain_of_draft and agent_type in ["synthesizer", "verifier"]:
        prompt += f"\n\n{config.instructions.chain_of_draft}"

    # Dynamic context (lowest cache hit potential - always different)
    if dynamic_context:
        prompt += f"\n\n{dynamic_context}"

    return prompt


def get_system_prompt(agent_type: str) -> str:
    """
    Get just the system prompt portion (prefix + suffix) without dynamic content.

    Use this for creating consistent system prompts that can be cached.
    """
    config = get_prompt_config()
    prompt = config.system_prompts.core_prefix
    agent_suffixes = _get_agent_suffixes()
    suffix = agent_suffixes.get(agent_type, "")
    if suffix:
        prompt += suffix
    return prompt


def estimate_prefix_tokens(agent_type: str) -> int:
    """
    Estimate token count for static prefix (for cache planning).

    Rough estimate: ~1.3 tokens per word, ~4 chars per token
    """
    prefix = get_system_prompt(agent_type)
    # Conservative estimate
    return len(prefix) // 4


def _get_templates() -> Dict[str, str]:
    """Get templates from central config."""
    config = get_prompt_config()
    return {
        "search_query_generation": config.templates.search_query_generation,
        "url_relevance": config.templates.url_relevance,
        "content_summary": config.templates.content_summary,
        "gap_detection": config.templates.gap_detection,
        "synthesis_with_sources": config.templates.synthesis_with_sources,
    }


# Prompt templates for specific operations are provided via __getattr__


def get_template(template_name: str, **kwargs) -> str:
    """
    Get a prompt template with variables filled in.

    Args:
        template_name: Name of the template
        **kwargs: Variables to fill in the template

    Returns:
        Filled template string
    """
    templates = _get_templates()
    template = templates.get(template_name)
    if not template:
        # Fallback to central config lookup
        template = get_prompt_config().get_template(template_name)
    if not template:
        raise ValueError(f"Unknown template: {template_name}")

    return template.format(**kwargs)


# Token budget constants for context management
TOKEN_BUDGETS = {
    "analyzer": {
        "max_input": 4096,
        "max_output": 512,
        "prefix_reserve": 500,  # Reserved for system prompt
    },
    "synthesizer": {
        "max_input": 32768,  # Large context for content synthesis
        "max_output": 4096,
        "prefix_reserve": 600,
    },
    "coverage": {
        "max_input": 8192,
        "max_output": 1024,
        "prefix_reserve": 500,
    },
    "verifier": {
        "max_input": 8192,
        "max_output": 1024,
        "prefix_reserve": 500,
    },
}


def get_token_budget(agent_type: str) -> Dict[str, int]:
    """Get token budget configuration for an agent type."""
    return TOKEN_BUDGETS.get(agent_type, {
        "max_input": 8192,
        "max_output": 2048,
        "prefix_reserve": 500,
    })
