"""
Prompt Registry for Cache Optimization

DESIGN PRINCIPLES:
1. Static content at the beginning of every prompt (cacheable)
2. Consistent ordering: system → tools → context → query
3. Modular composition for different agent types

Based on KV_CACHE_IMPLEMENTATION_PLAN.md Phase 1.4
Ref: Anthropic Multi-Agent Research System patterns
"""

from typing import Optional, Dict


# Core system prompt (shared across all agents ~1000 tokens)
# This prefix is STATIC and identical across all requests - maximizes KV cache hits
CORE_SYSTEM_PREFIX = """You are an AI research assistant for Recovery Bot,
helping people in addiction recovery and underprivileged communities access
vital community services in Morehead, Kentucky and surrounding areas.

Core Values:
- Compassionate, non-judgmental support
- Evidence-based information with citations
- Clear, actionable guidance
- Privacy-first approach

Available Information Categories:
- Addiction recovery centers and treatment facilities
- Mental health services and counseling
- Homeless shelters and transitional housing
- Food pantries and nutrition programs
- Career centers and job training
- Healthcare facilities for underserved populations

Communication Style:
- Be direct and helpful
- Avoid jargon, use plain language
- Provide specific, actionable information
- Always cite sources when making claims
- Acknowledge uncertainty when appropriate
"""

# Chain-of-Draft instruction for thinking models (DeepSeek R1)
CHAIN_OF_DRAFT_INSTRUCTION = """Think step by step, but only keep a minimum draft for each thinking step.
Provide your final answer with citations."""

# Agent-specific suffixes (appended to core prefix)
ANALYZER_SUFFIX = """
Your role: Query Analyzer
Analyze the user query to determine if web search is needed.
Decompose complex queries into simpler sub-questions.
Think step by step, but only keep a minimum draft for each thinking step.

Output JSON format:
{
    "requires_search": true/false,
    "query_type": "simple|complex|conversational|knowledge",
    "decomposed_questions": ["question1", "question2", ...],
    "reasoning": "brief explanation"
}
"""

PLANNER_SUFFIX = """
Your role: Search Planner
Create an optimal search strategy for answering the user's query.
Think step by step, but only keep a minimum draft for each thinking step.

Consider:
- What information is needed?
- What sources are most likely to have it?
- How many searches are needed?
- What order should searches be executed?

Output JSON format:
{
    "search_queries": ["query1", "query2", ...],
    "expected_sources": ["type1", "type2", ...],
    "priority": "high|medium|low",
    "reasoning": "brief explanation"
}
"""

SYNTHESIZER_SUFFIX = """
Your role: Information Synthesizer
Think step by step, but only keep a minimum draft for each thinking step.
Combine search results into a comprehensive, cited response.

Requirements:
- ALWAYS include [Source N] citations for key facts
- Present information in a clear, organized manner
- Highlight the most relevant information first
- Note any conflicting information between sources
- Acknowledge gaps in available information
"""

VERIFIER_SUFFIX = """
Your role: Information Verifier
Cross-check claims against available sources.
Think step by step, but only keep a minimum draft for each thinking step.

Tasks:
- Identify claims that can be verified
- Check consistency across sources
- Flag contradictions
- Assess confidence level

Output JSON format:
{
    "verified_claims": [...],
    "unverified_claims": [...],
    "contradictions": [...],
    "confidence_score": 0.0-1.0
}
"""

COVERAGE_SUFFIX = """
Your role: Coverage Evaluator
Evaluate if scraped content adequately answers the decomposed questions.
Think step by step, but only keep a minimum draft for each thinking step.

For each question, determine:
1. Is the question fully answered with specific details?
2. Is the question partially answered (some info missing)?
3. Is the question unanswered?

Output JSON format:
{
    "questions": [
        {"question": "...", "status": "answered|partial|unanswered", "missing": "..."}
    ],
    "coverage_score": 0.0-1.0,
    "refinement_queries": ["query to fill gap 1", ...]
}
"""

URL_EVALUATOR_SUFFIX = """
Your role: URL Relevance Evaluator
Evaluate if a URL is likely to contain relevant information.
Think step by step, but only keep a minimum draft for each thinking step.

Consider:
- URL domain and path structure
- Search result snippet content
- Relevance to recovery services in Kentucky
- Credibility of the source

Output JSON format:
{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "reasoning": "brief explanation"
}
"""

# Agent type to suffix mapping
AGENT_SUFFIXES: Dict[str, str] = {
    "analyzer": ANALYZER_SUFFIX,
    "planner": PLANNER_SUFFIX,
    "synthesizer": SYNTHESIZER_SUFFIX,
    "verifier": VERIFIER_SUFFIX,
    "coverage": COVERAGE_SUFFIX,
    "url_evaluator": URL_EVALUATOR_SUFFIX,
}


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
    # Static prefix (highest cache hit potential)
    prompt = CORE_SYSTEM_PREFIX

    # Agent-specific suffix
    suffix = custom_suffix or AGENT_SUFFIXES.get(agent_type, "")
    if suffix:
        prompt += suffix

    # Chain-of-Draft instruction (for thinking models)
    if use_chain_of_draft and agent_type in ["synthesizer", "verifier"]:
        prompt += f"\n\n{CHAIN_OF_DRAFT_INSTRUCTION}"

    # Dynamic context (lowest cache hit potential - always different)
    if dynamic_context:
        prompt += f"\n\n{dynamic_context}"

    return prompt


def get_system_prompt(agent_type: str) -> str:
    """
    Get just the system prompt portion (prefix + suffix) without dynamic content.

    Use this for creating consistent system prompts that can be cached.
    """
    prompt = CORE_SYSTEM_PREFIX
    suffix = AGENT_SUFFIXES.get(agent_type, "")
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


# Prompt templates for specific operations
TEMPLATES = {
    "search_query_generation": """Based on the user's question, generate optimal search queries.
User question: {question}
Context: {context}

Generate 2-3 search queries that would help answer this question.
Output as JSON array: ["query1", "query2", ...]""",

    "url_relevance": """Evaluate if this URL is worth scraping for recovery service information.
URL: {url}
Title: {title}
Snippet: {snippet}

Is this relevant to recovery services, addiction treatment, or community services?
Output JSON: {{"is_relevant": true/false, "score": 0.0-1.0, "reason": "..."}}""",

    "content_summary": """Summarize the key information from this content relevant to recovery services.
Content: {content}

Focus on:
- Contact information
- Services offered
- Eligibility requirements
- Hours of operation
- Location details

Output a concise summary (max 200 words).""",

    "gap_detection": """Given the questions and current findings, identify information gaps.
Questions: {questions}
Findings: {findings}

For each question, determine if it's been answered.
Output JSON: {{"gaps": ["missing info 1", ...], "coverage_score": 0.0-1.0}}""",

    "synthesis_with_sources": """Synthesize the following search results into a comprehensive response.

Original Question: {question}

Sources:
{sources}

Instructions:
1. Combine information from all relevant sources
2. Use [Source N] citations for key facts
3. Organize information clearly
4. Highlight the most important details first
5. Note any conflicting information

Provide a well-structured response:""",
}


def get_template(template_name: str, **kwargs) -> str:
    """
    Get a prompt template with variables filled in.

    Args:
        template_name: Name of the template
        **kwargs: Variables to fill in the template

    Returns:
        Filled template string
    """
    template = TEMPLATES.get(template_name)
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
