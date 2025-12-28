"""
Prefix-Optimized Prompt Architecture for Maximum KV Cache Reuse.

Based on research from:
- SGLang RadixAttention: Tree-structured KV cache with automatic prefix sharing
- vLLM Automatic Prefix Caching: Block-level hashing for prefix matching
- llm-d: Distributed KV cache routing (87% cache hit rate)

Key principles:
1. Static content at the beginning (system prompts, agent roles)
2. Stable prefixes before dynamic content
3. Append-only context growth
4. Deterministic serialization for cache hits

Prompt structure (most stable → least stable):
1. SYSTEM_PREFIX (100% cache hit potential across all agents)
2. Agent role prefix (per-agent cache hit)
3. Scratchpad context (incremental reuse)
4. Current task (always recomputed)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib


# =============================================================================
# Level 1: System-wide Static Prefix (shared across ALL agents)
# =============================================================================

SYSTEM_PREFIX = """You are part of an intelligent agentic search system designed for research, troubleshooting, and engineering solutions.

CORE PRINCIPLES:
- Accuracy and source attribution are paramount
- Provide technically accurate, verifiable information
- Prefer authoritative sources (official docs, peer-reviewed, standards bodies)
- Be direct and solution-focused in all responses
- Cite sources using [Source N] notation

AVAILABLE TOOLS:
- web_search: Execute web searches via SearXNG metasearch engine
- web_scrape: Extract detailed content from URLs using Playwright
- vl_scrape: Vision-language screenshot analysis for JS-heavy pages
- rag_search: Search the local knowledge base
- memory_store: Save important findings for future reference

RESPONSE GUIDELINES:
- Always structure responses with clear headers
- Include confidence scores when making claims
- Acknowledge information gaps honestly
- Provide actionable next steps and solutions when appropriate
- Match technical depth to the query complexity"""


# =============================================================================
# Level 2: Agent-Specific Role Prefixes (per-agent cache hits)
# =============================================================================

AGENT_ROLE_PREFIXES = {
    "analyzer": """
---
ROLE: Query Analyzer Agent
RESPONSIBILITY: Analyze user queries to determine search requirements and complexity.

TASKS:
1. Classify query type (factual, comparative, procedural, exploratory)
2. Assess complexity (low, medium, high)
3. Determine if web search is required
4. Identify key entities and concepts

OUTPUT FORMAT (JSON):
{
  "requires_search": true/false,
  "query_type": "factual|comparative|procedural|exploratory",
  "complexity": "low|medium|high",
  "key_entities": ["entity1", "entity2"],
  "reasoning": "Brief explanation of analysis"
}
---""",

    "planner": """
---
ROLE: Query Planner Agent
RESPONSIBILITY: Decompose complex queries into searchable sub-questions with clear completion criteria.

TASKS:
1. Break down complex queries into atomic, searchable sub-questions
2. Define explicit completion criteria for each sub-question
3. Assign priority (1=highest) to order search efforts
4. Estimate phases needed for comprehensive coverage

OUTPUT FORMAT (JSON):
{
  "decomposed_questions": [
    {
      "question": "Specific searchable question",
      "criteria": "What constitutes a complete answer",
      "priority": 1
    }
  ],
  "phases": 1-3,
  "initial_queries": ["search query 1", "search query 2"],
  "reasoning": "Brief explanation of decomposition strategy"
}
---""",

    "searcher": """
---
ROLE: Web Searcher Agent
RESPONSIBILITY: Execute web searches and evaluate URL relevance before scraping.

TASKS:
1. Generate effective search queries from sub-questions
2. Evaluate URL relevance (0-1 score) before committing to scrape
3. Track search coverage across sub-questions
4. Avoid redundant searches using search history

OUTPUT FORMAT (JSON):
{
  "search_queries": ["query1", "query2"],
  "url_evaluations": [
    {"url": "...", "relevance": 0.85, "should_scrape": true, "reasoning": "..."}
  ],
  "coverage_progress": {"question_1": 0.6, "question_2": 0.3}
}
---""",

    "scraper": """
---
ROLE: Content Scraper Agent
RESPONSIBILITY: Extract and structure relevant content from web pages.

TASKS:
1. Extract content relevant to the current sub-questions
2. Structure extracted information with source attribution
3. Identify information gaps requiring additional searches
4. Calculate relevance scores for extracted content

OUTPUT FORMAT (JSON):
{
  "extracted_content": [
    {
      "content": "Extracted text...",
      "source_url": "https://...",
      "relevance_score": 0.9,
      "addresses_question": "question_id"
    }
  ],
  "identified_gaps": ["Gap 1", "Gap 2"],
  "scrape_quality": 0.85
}
---""",

    "verifier": """
---
ROLE: Claim Verifier Agent
RESPONSIBILITY: Cross-check facts across sources and assess information confidence.

TASKS:
1. Identify specific claims that can be verified
2. Cross-check claims against multiple sources
3. Detect contradictions between sources
4. Calculate confidence scores based on source agreement

OUTPUT FORMAT (JSON):
{
  "verified_claims": [
    {"claim": "...", "verified": true, "sources": ["url1", "url2"], "confidence": 0.9}
  ],
  "contradictions": [
    {"claim": "...", "source_a": "...", "source_b": "...", "resolution": "..."}
  ],
  "overall_confidence": 0.85,
  "verification_notes": "..."
}
---""",

    "synthesizer": """
---
ROLE: Response Synthesizer Agent
RESPONSIBILITY: Combine verified findings into a coherent, well-cited response.

TASKS:
1. Synthesize information from all verified sources
2. Add inline citations using [Source N] notation
3. Structure response with clear headers for readability
4. Include confidence level and any caveats
5. Provide source list at the end

OUTPUT FORMAT (Markdown):
### Main Topic

Content with [Source 1] inline citations...

#### Subtopic
More content with [Source 2] citations...

**Confidence Level:** Medium/High
**Caveats:** Any limitations or uncertainties

**Sources:**
- [Source 1]: Title - URL
- [Source 2]: Title - URL
---""",

    "coverage_evaluator": """
---
ROLE: Coverage Evaluator Agent
RESPONSIBILITY: Assess whether scraped content sufficiently answers the decomposed questions.

TASKS:
1. Evaluate each sub-question against gathered content
2. Calculate coverage score (0-1) for each question
3. Identify specific information gaps
4. Recommend refinement queries if coverage is insufficient

OUTPUT FORMAT (JSON):
{
  "coverage_score": 0.75,
  "is_sufficient": true/false,
  "question_coverage": {
    "question_1": {"score": 0.9, "gaps": []},
    "question_2": {"score": 0.6, "gaps": ["Missing cost info"]}
  },
  "refinement_queries": ["query1", "query2"],
  "reasoning": "Explanation of coverage assessment"
}
---"""
}


# =============================================================================
# Level 3: Scratchpad Context Builder (grows incrementally, append-only)
# =============================================================================

def build_scratchpad_context(scratchpad_state: Dict[str, Any],
                              max_length: int = 8000) -> str:
    """
    Build scratchpad context in deterministic, append-only format.

    The context is structured to maximize prefix cache hits:
    - Mission statement (stable once set)
    - Sub-questions (stable once decomposed)
    - Findings (append-only, newest last)
    - Gaps and notes (append-only)

    Args:
        scratchpad_state: Current scratchpad state dictionary
        max_length: Maximum context length in characters

    Returns:
        Formatted scratchpad context string
    """
    sections = []

    # Mission (stable once set)
    if scratchpad_state.get('mission'):
        sections.append(f"MISSION: {scratchpad_state['mission']}")

    # Sub-questions (stable once decomposed)
    if scratchpad_state.get('sub_questions'):
        questions = []
        for i, q in enumerate(scratchpad_state['sub_questions']):
            if isinstance(q, dict):
                questions.append(f"  Q{i+1}: {q.get('question', q)}")
                if q.get('criteria'):
                    questions.append(f"       Criteria: {q['criteria']}")
            else:
                questions.append(f"  Q{i+1}: {q}")
        sections.append("SUB-QUESTIONS:\n" + "\n".join(questions))

    # Search history (append-only)
    if scratchpad_state.get('search_history'):
        searches = scratchpad_state['search_history'][-10:]  # Last 10
        search_list = "\n".join(f"  - {s}" for s in searches)
        sections.append(f"RECENT SEARCHES:\n{search_list}")

    # Findings (append-only, newest last)
    if scratchpad_state.get('findings'):
        findings = scratchpad_state['findings'][-15:]  # Last 15
        finding_list = []
        for f in findings:
            if isinstance(f, dict):
                source = f.get('source', 'Unknown')
                summary = f.get('summary', str(f))[:200]
                finding_list.append(f"  [{source}]: {summary}")
            else:
                finding_list.append(f"  - {str(f)[:200]}")
        sections.append("FINDINGS:\n" + "\n".join(finding_list))

    # Coverage status
    if scratchpad_state.get('coverage'):
        coverage = scratchpad_state['coverage']
        coverage_text = f"COVERAGE: {coverage.get('score', 0):.0%}"
        if coverage.get('gaps'):
            coverage_text += f"\n  Gaps: {', '.join(coverage['gaps'][:5])}"
        sections.append(coverage_text)

    # Contradictions (if any)
    if scratchpad_state.get('contradictions'):
        contradictions = scratchpad_state['contradictions'][:3]
        contra_list = "\n".join(f"  - {c}" for c in contradictions)
        sections.append(f"CONTRADICTIONS NOTED:\n{contra_list}")

    # Agent notes (append-only communication)
    if scratchpad_state.get('agent_notes'):
        notes = scratchpad_state['agent_notes'][-5:]
        note_list = "\n".join(f"  [{n.get('from', 'Agent')}]: {n.get('note', '')}"
                             for n in notes if isinstance(n, dict))
        if note_list:
            sections.append(f"AGENT NOTES:\n{note_list}")

    context = "\n\n".join(sections)

    # Truncate if needed, preserving most recent content
    if len(context) > max_length:
        context = "...[truncated]...\n" + context[-(max_length - 20):]

    return context


# =============================================================================
# Level 4: Current Task Builder (dynamic portion at end)
# =============================================================================

def build_current_task(agent_type: str, task_details: Dict[str, Any]) -> str:
    """
    Build current task prompt (dynamic portion at end of prompt).

    This section changes with each invocation, so it's placed last
    to maximize prefix cache hits on the stable portions above.

    Args:
        agent_type: Type of agent being invoked
        task_details: Task-specific details including query, iteration, etc.

    Returns:
        Formatted task instruction string
    """
    instruction = task_details.get('instruction', 'Process the above context.')
    query = task_details.get('query', '')
    iteration = task_details.get('iteration', 1)

    task_text = f"""
---
CURRENT TASK (Iteration {iteration}):
{instruction}

USER QUERY: {query}

Respond in the required format for your role ({agent_type}).
---"""

    return task_text


# =============================================================================
# Full Prompt Builder
# =============================================================================

def build_full_prompt(agent_type: str,
                      scratchpad_state: Dict[str, Any],
                      task_details: Dict[str, Any]) -> str:
    """
    Build full prompt with prefix optimization for maximum KV cache reuse.

    Structure (most stable → least stable):
    1. SYSTEM_PREFIX (100% cache hit potential)
    2. Agent role prefix (per-agent cache hit)
    3. Scratchpad context (incremental reuse)
    4. Current task (always recomputed)

    Args:
        agent_type: Type of agent (analyzer, planner, etc.)
        scratchpad_state: Current scratchpad/blackboard state
        task_details: Current task details

    Returns:
        Complete prompt string optimized for prefix caching
    """
    role_prefix = AGENT_ROLE_PREFIXES.get(agent_type, "")
    scratchpad_context = build_scratchpad_context(scratchpad_state)
    current_task = build_current_task(agent_type, task_details)

    return "\n\n".join([
        SYSTEM_PREFIX,
        role_prefix,
        scratchpad_context,
        current_task
    ])


def get_prefix_for_warming(agent_type: str) -> str:
    """
    Get the static prefix for a specific agent type.

    This is used for KV cache warming - pre-computing the attention
    for static portions that will be reused across multiple requests.

    Args:
        agent_type: Type of agent

    Returns:
        Static prefix string (system + role)
    """
    role_prefix = AGENT_ROLE_PREFIXES.get(agent_type, "")
    return SYSTEM_PREFIX + "\n\n" + role_prefix


def estimate_prefix_reuse(agent_type: str, scratchpad_size: int) -> Dict[str, Any]:
    """
    Estimate potential KV cache reuse for a prompt configuration.

    Args:
        agent_type: Type of agent
        scratchpad_size: Number of items in scratchpad

    Returns:
        Dictionary with cache reuse estimates
    """
    # Rough token estimation (words * 1.3)
    system_tokens = len(SYSTEM_PREFIX.split()) * 1.3
    role_prefix = AGENT_ROLE_PREFIXES.get(agent_type, "")
    role_tokens = len(role_prefix.split()) * 1.3

    # Estimate scratchpad tokens
    scratchpad_tokens = scratchpad_size * 50  # ~50 tokens per finding/item

    total_cacheable = system_tokens + role_tokens
    total_prompt = total_cacheable + scratchpad_tokens + 100  # ~100 for task

    return {
        'system_prefix_tokens': int(system_tokens),
        'role_prefix_tokens': int(role_tokens),
        'total_cacheable_prefix': int(total_cacheable),
        'estimated_scratchpad_tokens': int(scratchpad_tokens),
        'estimated_total_tokens': int(total_prompt),
        'prefix_cache_ratio': total_cacheable / max(1, total_prompt),
        'estimated_cache_hit_rate': min(0.95, 0.5 + (total_cacheable / max(1, total_prompt)) * 0.5)
    }


def compute_prompt_hash(prompt: str) -> str:
    """
    Compute a hash for prompt prefix caching.

    This can be used to identify when prompts share common prefixes
    and enable cache routing decisions.

    Args:
        prompt: Full or partial prompt string

    Returns:
        SHA256 hash truncated to 16 characters
    """
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# =============================================================================
# Chain-of-Draft Optimization for DeepSeek R1
# =============================================================================

CHAIN_OF_DRAFT_INSTRUCTION = """Think step by step, but only output minimal drafts of each step in ~5 words. After all drafts, output your final answer.

Format:
Draft 1: [~5 word summary]
Draft 2: [~5 word summary]
...
Final: [Complete answer]"""


def apply_chain_of_draft(prompt: str, model_name: str = "") -> str:
    """
    Apply Chain-of-Draft instruction for thinking models.

    This reduces thinking tokens by 50-80% for models like DeepSeek R1.

    Args:
        prompt: Original prompt
        model_name: Model name to check if CoD should be applied

    Returns:
        Modified prompt with CoD instruction if applicable
    """
    thinking_models = ['deepseek', 'r1', 'o1', 'o3']

    if any(m in model_name.lower() for m in thinking_models):
        # Insert CoD instruction after system prefix but before task
        return prompt.replace(
            "Respond in the required format",
            f"{CHAIN_OF_DRAFT_INSTRUCTION}\n\nRespond in the required format"
        )

    return prompt


# =============================================================================
# Prompt Registry for Common Patterns
# =============================================================================

@dataclass
class PromptTemplate:
    """Registered prompt template with caching metadata"""
    name: str
    template: str
    prefix_hash: str
    estimated_tokens: int
    warm_priority: int  # 1=high priority for warming


class PromptRegistry:
    """
    Registry of prompt templates for KV cache optimization.

    Tracks which prompts are frequently used to prioritize warming.
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.usage_counts: Dict[str, int] = {}

        # Pre-register common agent prompts
        self._register_agent_prompts()

    def _register_agent_prompts(self):
        """Register all agent prompt prefixes"""
        for agent_type, role_prefix in AGENT_ROLE_PREFIXES.items():
            full_prefix = SYSTEM_PREFIX + "\n\n" + role_prefix
            self.templates[agent_type] = PromptTemplate(
                name=agent_type,
                template=full_prefix,
                prefix_hash=compute_prompt_hash(full_prefix),
                estimated_tokens=int(len(full_prefix.split()) * 1.3),
                warm_priority=1 if agent_type in ['analyzer', 'synthesizer'] else 2
            )

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a registered template by name"""
        if name in self.templates:
            self.usage_counts[name] = self.usage_counts.get(name, 0) + 1
        return self.templates.get(name)

    def get_high_priority_templates(self, limit: int = 5) -> List[PromptTemplate]:
        """Get templates that should be warmed first based on priority and usage"""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: (t.warm_priority, -self.usage_counts.get(t.name, 0))
        )
        return sorted_templates[:limit]

    def register_custom(self, name: str, template: str, priority: int = 3):
        """Register a custom prompt template"""
        self.templates[name] = PromptTemplate(
            name=name,
            template=template,
            prefix_hash=compute_prompt_hash(template),
            estimated_tokens=int(len(template.split()) * 1.3),
            warm_priority=priority
        )


# Singleton instance
_prompt_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """Get or create the singleton PromptRegistry instance"""
    global _prompt_registry
    if _prompt_registry is None:
        _prompt_registry = PromptRegistry()
    return _prompt_registry
