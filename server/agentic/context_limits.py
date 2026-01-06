"""
Context Limits and Model-Aware Utilization Calculator

Dynamically calculates optimal limits for URLs, content, and synthesis
based on the actual model's context window. This ensures maximum context
utilization for each model call in the agentic pipeline.

Key Principle: Fill the context window to give the model the best chance
of finding relevant information, then let the model trim to pertinent content.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from .model_specs import OLLAMA_MODEL_SPECS
import logging

logger = logging.getLogger(__name__)


# Token-to-character ratio (approximate)
# Most tokenizers average ~4 chars/token for English text
CHARS_PER_TOKEN = 4


@dataclass
class ContextBudget:
    """Budget allocation for a specific model call"""
    model_name: str
    context_window_tokens: int
    context_window_chars: int

    # Reserved space
    system_prompt_chars: int
    response_reserve_chars: int

    # Available for content
    available_chars: int

    # Recommended limits
    max_urls_to_scrape: int
    max_content_per_source: int
    max_total_source_content: int
    max_snippets: int
    snippet_length: int

    # Utilization targets
    target_utilization: float = 0.85  # Aim for 85% context utilization


@dataclass
class PipelineContextConfig:
    """Context configuration for the entire agentic pipeline"""

    # Analyzer model (query analysis, URL evaluation)
    analyzer_model: str = "qwen3:8b"
    analyzer_context: int = 40960  # 40K tokens (from ollama.com)

    # Planner model (search planning)
    planner_model: str = "qwen3:8b"
    planner_context: int = 40960  # 40K tokens

    # Synthesizer model (content synthesis) - ministral-3:3b scored 0.848 overall
    # Benchmark: 93.3% analysis, 17s duration, only 3GB VRAM (fits with PDF Tools)
    synthesizer_model: str = "ministral-3:3b"
    synthesizer_context: int = 32000  # 32K tokens (ministral has 128K context)

    # Thinking model (complex reasoning) - ministral-3:3b best practical option
    # phi4-reasoning:14b (0.893) too VRAM-heavy (11GB) for production use
    thinking_model: str = "ministral-3:3b"
    thinking_context: int = 32000  # 32K tokens

    # Fast evaluation model
    evaluator_model: str = "qwen3:8b"  # Upgraded from gemma3:4b for better evaluation quality
    evaluator_context: int = 131072  # 128K tokens

    # Verification model
    verifier_model: str = "qwen3:8b"
    verifier_context: int = 40960  # 40K tokens


# Default pipeline configuration
DEFAULT_PIPELINE_CONFIG = PipelineContextConfig()


def get_model_context_window(model_name: str) -> int:
    """
    Get the context window size for a model in tokens.

    First tries to match from our scraped database, then falls back to
    the static OLLAMA_MODEL_SPECS dictionary.

    Args:
        model_name: Model identifier (e.g., "qwen3:8b", "deepseek-r1:14b")

    Returns:
        Context window size in tokens
    """
    # Try to get from live database first (populated by model scraper)
    # This is loaded lazily to avoid circular imports
    try:
        from .model_selector import get_model_selector
        selector = get_model_selector()
        if selector and hasattr(selector, '_model_specs'):
            specs = selector._model_specs
            if model_name in specs:
                ctx = specs[model_name].get("context_window")
                if ctx:
                    return ctx
            # Try partial match
            for spec_name, spec in specs.items():
                if model_name in spec_name or spec_name in model_name:
                    ctx = spec.get("context_window")
                    if ctx:
                        return ctx
    except Exception as e:
        logger.debug(f"Could not query model selector: {e}")

    # Fall back to static OLLAMA_MODEL_SPECS
    normalized = model_name.lower().replace(":", "").replace("-", "").replace("_", "")

    for spec_name, spec in OLLAMA_MODEL_SPECS.items():
        spec_normalized = spec_name.lower().replace(":", "").replace("-", "").replace("_", "")
        if spec_normalized in normalized or normalized in spec_normalized:
            return spec.get("context_window", 8192)

    # Known model defaults (updated from ollama.com scrape results)
    model_defaults = {
        # Qwen3 models (40K native)
        "qwen3:8b": 40960,
        "qwen3:4b": 40960,
        "qwen3:14b": 40960,
        "qwen3:32b": 40960,
        # DeepSeek R1 models (128K)
        "deepseek-r1:8b": 128000,
        "deepseek-r1:14b": 128000,
        "deepseek-r1:14b-qwen-distill-q8_0": 128000,
        "deepseek-r1:32b": 128000,
        # Gemma3 models (128K)
        "gemma3:4b": 131072,
        "gemma3:12b": 131072,
        "gemma3:27b": 131072,
        # Other common models
        "llama3.2:3b": 131072,
        "llama3.3:70b": 131072,
        "phi4-mini:3.8b": 131072,
        "mistral-small3.2:24b": 131072,
        # Ministral models (128K)
        "ministral-3:3b": 131072,
        "ministral-3:8b": 131072,
        "ministral-3:14b": 131072,
        # Vision models (256K)
        "qwen3-vl:8b": 256000,
        "qwen3-vl:32b": 256000,
    }

    for key, value in model_defaults.items():
        key_normalized = key.lower().replace(":", "").replace("-", "").replace("_", "")
        if key_normalized in normalized or normalized in key_normalized:
            return value

    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default 32K context")
    return 32768  # Reasonable default for modern models


def calculate_context_budget(
    model_name: str,
    system_prompt_chars: int = 2000,
    response_reserve_chars: int = 4000,
    target_utilization: float = 0.85
) -> ContextBudget:
    """
    Calculate the context budget for a specific model.

    Args:
        model_name: Model identifier
        system_prompt_chars: Space reserved for system prompt
        response_reserve_chars: Space reserved for model response
        target_utilization: Target context window utilization (0-1)

    Returns:
        ContextBudget with calculated limits
    """
    context_tokens = get_model_context_window(model_name)
    context_chars = context_tokens * CHARS_PER_TOKEN

    # Calculate available space
    reserved = system_prompt_chars + response_reserve_chars
    available = int((context_chars - reserved) * target_utilization)

    # Calculate optimal limits based on available space
    # Heuristic: Aim for 15-25 sources with balanced content each

    if available >= 400000:  # 100K+ token models (gemma3, etc.)
        max_urls = 30
        max_per_source = 15000
        max_snippets = 50
        snippet_length = 500
    elif available >= 160000:  # 40K+ token models (qwen3:8b)
        max_urls = 25
        max_per_source = 12000
        max_snippets = 40
        snippet_length = 400
    elif available >= 80000:  # 20K+ token models
        max_urls = 20
        max_per_source = 10000
        max_snippets = 30
        snippet_length = 350
    elif available >= 40000:  # 10K+ token models
        max_urls = 15
        max_per_source = 8000
        max_snippets = 25
        snippet_length = 300
    elif available >= 20000:  # 5K+ token models
        max_urls = 10
        max_per_source = 6000
        max_snippets = 20
        snippet_length = 250
    else:  # Small context models
        max_urls = 8
        max_per_source = 4000
        max_snippets = 15
        snippet_length = 200

    # Ensure total doesn't exceed available
    max_total = min(available, max_urls * max_per_source)

    return ContextBudget(
        model_name=model_name,
        context_window_tokens=context_tokens,
        context_window_chars=context_chars,
        system_prompt_chars=system_prompt_chars,
        response_reserve_chars=response_reserve_chars,
        available_chars=available,
        max_urls_to_scrape=max_urls,
        max_content_per_source=max_per_source,
        max_total_source_content=max_total,
        max_snippets=max_snippets,
        snippet_length=snippet_length,
        target_utilization=target_utilization
    )


def get_analyzer_limits(model_name: Optional[str] = None) -> Dict[str, int]:
    """
    Get optimal limits for the analyzer/evaluator model.

    Returns limits for URL evaluation, content analysis, etc.
    """
    model = model_name or DEFAULT_PIPELINE_CONFIG.analyzer_model
    budget = calculate_context_budget(
        model,
        system_prompt_chars=3000,  # Analyzer has detailed instructions
        response_reserve_chars=2000
    )

    return {
        "max_urls_to_evaluate": budget.max_snippets,
        "snippet_length": budget.snippet_length,
        "max_content_for_analysis": budget.available_chars,
        "num_ctx": budget.context_window_tokens,
    }


def get_synthesizer_limits(
    model_name: Optional[str] = None,
    is_thinking_model: bool = False
) -> Dict[str, int]:
    """
    Get optimal limits for the synthesizer model.

    Args:
        model_name: Model to use for synthesis
        is_thinking_model: Whether this is a thinking/reasoning model

    Returns limits for content synthesis.
    """
    if is_thinking_model:
        model = model_name or DEFAULT_PIPELINE_CONFIG.thinking_model
        # Thinking models need more response space for chain-of-thought
        response_reserve = 8000
    else:
        model = model_name or DEFAULT_PIPELINE_CONFIG.synthesizer_model
        response_reserve = 4000

    budget = calculate_context_budget(
        model,
        system_prompt_chars=2500,
        response_reserve_chars=response_reserve,
        target_utilization=0.80  # Slightly lower for synthesis stability
    )

    return {
        "max_urls_to_scrape": budget.max_urls_to_scrape,
        "max_content_per_source": budget.max_content_per_source,
        "max_total_content": budget.max_total_source_content,
        "max_snippets_if_no_scrape": budget.max_snippets,
        "num_ctx": budget.context_window_tokens,
    }


def get_dynamic_source_allocation(
    total_sources: int,
    model_name: Optional[str] = None,
    is_thinking_model: bool = False
) -> Tuple[int, int]:
    """
    Calculate dynamic per-source content allocation.

    Distributes available context evenly across sources,
    with a minimum floor per source.

    Args:
        total_sources: Number of sources to allocate for
        model_name: Model being used
        is_thinking_model: Whether thinking model is being used

    Returns:
        Tuple of (chars_per_source, max_sources_to_use)
    """
    limits = get_synthesizer_limits(model_name, is_thinking_model)
    max_total = limits["max_total_content"]

    # Minimum per source to be useful
    MIN_PER_SOURCE = 2000
    # Maximum per source to ensure diversity
    MAX_PER_SOURCE = 15000

    if total_sources <= 0:
        return (MAX_PER_SOURCE, 1)

    # Calculate ideal allocation
    chars_per_source = max_total // total_sources

    # Apply bounds
    chars_per_source = max(MIN_PER_SOURCE, min(MAX_PER_SOURCE, chars_per_source))

    # Calculate how many sources we can fit with this allocation
    max_sources = max_total // chars_per_source

    return (chars_per_source, min(total_sources, max_sources))


def get_search_result_limits(model_name: Optional[str] = None) -> Dict[str, int]:
    """
    Get limits for initial search results (before scraping).

    These limits apply to the search phase when we're collecting
    snippets and URLs, not full page content.
    """
    model = model_name or DEFAULT_PIPELINE_CONFIG.analyzer_model
    budget = calculate_context_budget(model)

    return {
        "max_results_per_query": min(15, budget.max_snippets // 3),
        "max_total_results": budget.max_snippets,
        "snippet_length": budget.snippet_length,
        "max_queries_per_iteration": 5,
    }


def get_verification_limits(model_name: Optional[str] = None) -> Dict[str, int]:
    """
    Get limits for the verification/CRAG evaluation phase.
    """
    model = model_name or DEFAULT_PIPELINE_CONFIG.verifier_model
    budget = calculate_context_budget(
        model,
        system_prompt_chars=2000,
        response_reserve_chars=2000
    )

    return {
        "max_claims_to_verify": 20,
        "max_content_per_claim": budget.available_chars // 20,
        "max_sources_per_verification": 10,
        "num_ctx": budget.context_window_tokens,
    }


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text.

    Uses simple character-based estimation.
    For more accuracy, use actual tokenizer.
    """
    return len(text) // CHARS_PER_TOKEN


def format_context_utilization_report(
    model_name: str,
    actual_input_chars: int,
    actual_output_chars: int = 0
) -> str:
    """
    Generate a human-readable context utilization report.
    """
    context_tokens = get_model_context_window(model_name)
    context_chars = context_tokens * CHARS_PER_TOKEN

    input_tokens = estimate_token_count(str(actual_input_chars))
    output_tokens = estimate_token_count(str(actual_output_chars))

    total_used = actual_input_chars + actual_output_chars
    utilization_pct = (total_used / context_chars) * 100 if context_chars > 0 else 0

    return (
        f"Model: {model_name}\n"
        f"Context Window: {context_tokens:,} tokens ({context_chars:,} chars)\n"
        f"Input: ~{input_tokens:,} tokens ({actual_input_chars:,} chars)\n"
        f"Output: ~{output_tokens:,} tokens ({actual_output_chars:,} chars)\n"
        f"Utilization: {utilization_pct:.1f}%"
    )


# ============================================================================
# Lost-in-Middle Mitigation
# ============================================================================
# LLMs attend less to content in the middle of the context window.
# This function reorders documents to place most relevant at start and end.
# Research: https://arxiv.org/abs/2307.03172 (Liu et al., 2023)
# ============================================================================

from typing import List, TypeVar, Callable, Any

T = TypeVar('T')


def reorder_for_attention(
    items: List[T],
    score_fn: Optional[Callable[[T], float]] = None,
    score_key: Optional[str] = None,
    reverse: bool = True
) -> List[T]:
    """
    Reorder items to mitigate lost-in-the-middle problem.

    Places most important items at the START and END of the list,
    with less important items in the middle. This improves LLM attention
    to critical content.

    Pattern: Sorts by score, then interleaves front/back placement:
    - Most relevant → position 0 (start)
    - 2nd most relevant → last position (end)
    - 3rd most relevant → position 1
    - 4th most relevant → second-to-last
    - ... continues alternating

    Args:
        items: List of items to reorder
        score_fn: Function to extract score from item (higher = more relevant)
        score_key: Alternative: attribute/key name to extract score
        reverse: If True, higher scores = more relevant (default)

    Returns:
        Reordered list with important items at extremes

    Example:
        >>> docs = [{"title": "A", "score": 0.9}, {"title": "B", "score": 0.5}, ...]
        >>> reordered = reorder_for_attention(docs, score_key="score")
        # Result: highest score at start, 2nd highest at end, etc.
    """
    if not items or len(items) <= 2:
        return items

    # Determine scoring function
    if score_fn is not None:
        get_score = score_fn
    elif score_key is not None:
        def get_score(item: T) -> float:
            if isinstance(item, dict):
                return item.get(score_key, 0) or 0
            return getattr(item, score_key, 0) or 0
    else:
        # Default: assume items are already sortable or use index
        def get_score(item: T) -> float:
            return 0

    # Sort by score
    try:
        sorted_items = sorted(items, key=get_score, reverse=reverse)
    except TypeError:
        # Items not sortable, return as-is
        logger.warning("Items not sortable for lost-in-middle reordering")
        return items

    # Interleave: front/back alternating
    reordered = [None] * len(sorted_items)
    front_idx = 0
    back_idx = len(sorted_items) - 1

    for i, item in enumerate(sorted_items):
        if i % 2 == 0:
            # Even indices go to front (advancing forward)
            reordered[front_idx] = item
            front_idx += 1
        else:
            # Odd indices go to back (advancing backward)
            reordered[back_idx] = item
            back_idx -= 1

    logger.debug(f"Reordered {len(items)} items for attention (front/back placement)")
    return reordered


def reorder_search_results(
    results: List[Dict[str, Any]],
    score_key: str = "relevance_score"
) -> List[Dict[str, Any]]:
    """
    Reorder search results for optimal LLM attention.

    Convenience wrapper for reorder_for_attention with common defaults.

    Args:
        results: List of search result dicts
        score_key: Key containing relevance score (default: "relevance_score")

    Returns:
        Reordered results with most relevant at start and end
    """
    return reorder_for_attention(results, score_key=score_key)


def reorder_sources_for_synthesis(
    sources: List[Any],
    score_attr: str = "combined_score"
) -> List[Any]:
    """
    Reorder sources for synthesis prompt construction.

    Works with both dicts and objects (WebSearchResult, etc.)

    Args:
        sources: List of source objects or dicts
        score_attr: Attribute/key containing relevance score

    Returns:
        Reordered sources with most relevant at start and end
    """
    def get_score(item) -> float:
        if isinstance(item, dict):
            return item.get(score_attr, 0) or item.get("relevance_score", 0) or 0
        # Try multiple common attributes
        for attr in [score_attr, "relevance_score", "score", "combined_score"]:
            val = getattr(item, attr, None)
            if val is not None:
                return float(val)
        return 0

    return reorder_for_attention(sources, score_fn=get_score)


# Pre-calculated limits for common pipeline stages
ANALYZER_LIMITS = get_analyzer_limits()
SYNTHESIZER_LIMITS = get_synthesizer_limits()
THINKING_SYNTHESIZER_LIMITS = get_synthesizer_limits(is_thinking_model=True)
SEARCH_LIMITS = get_search_result_limits()
VERIFICATION_LIMITS = get_verification_limits()


# Log configured limits on module load
logger.info(f"Context limits initialized:")
logger.info(f"  Analyzer: {ANALYZER_LIMITS['num_ctx']} tokens, {ANALYZER_LIMITS['max_urls_to_evaluate']} URLs")
logger.info(f"  Synthesizer: {SYNTHESIZER_LIMITS['num_ctx']} tokens, {SYNTHESIZER_LIMITS['max_urls_to_scrape']} URLs")
logger.info(f"  Thinking: {THINKING_SYNTHESIZER_LIMITS['num_ctx']} tokens, {THINKING_SYNTHESIZER_LIMITS['max_urls_to_scrape']} URLs")
logger.info(f"  Search: {SEARCH_LIMITS['max_total_results']} results, {SEARCH_LIMITS['snippet_length']} chars/snippet")
