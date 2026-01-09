"""
Query Analyzer Agent for Agentic Search System

Analyzes user queries using LLM to determine:
1. Whether web search would be beneficial
2. Query type and complexity
3. Key topics to research
4. Initial search strategy

Phase 2 Enhancement (GSW Entity Tracking):
- Extracts entities from scraped content using EntityTracker
- Maintains entity-centric memory for 51% token reduction
- Generates query-relevant entity summaries

B.10 Enhancement (LLM Gateway Integration):
- Optionally routes LLM calls through Gateway service (port 8100)
- Automatic fallback to direct Ollama when gateway unavailable
- Unified routing with VRAM management
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

from .models import QueryAnalysis, SearchPlan
from .context_limits import get_analyzer_limits, ANALYZER_LIMITS, get_model_context_window
from .metrics import get_performance_metrics
from .acronym_dictionary import expand_acronyms, get_acronym_info, get_related_terms
from .gateway_client import get_gateway_client, LogicalModel, GatewayResponse
from .llm_config import get_llm_config, get_config_for_task


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract a JSON object from text using proper brace matching.
    Handles nested objects and arrays correctly.
    """
    # Find the first { character
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Track brace depth to find matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i + 1]

    return None  # No matching closing brace found

if TYPE_CHECKING:
    from .entity_tracker import EntityTracker, EntityState

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes user queries to determine if web search is needed
    and creates an initial search strategy.

    Phase 2 Enhancement: GSW-style entity tracking for 51% token reduction.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        entity_tracker: Optional["EntityTracker"] = None,
        enable_acronym_expansion: bool = True
    ):
        # Load from central config if not provided
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.model = model or llm_config.pipeline.analyzer.model
        self.timeout = llm_config.ollama.default_timeout

        # GSW Entity Tracking (Phase 2)
        self._entity_tracker = entity_tracker
        self._entity_extraction_enabled = entity_tracker is not None

        # Acronym Expansion (Part E.1)
        self._enable_acronym_expansion = enable_acronym_expansion

    def _expand_query_acronyms(self, query: str) -> tuple[str, List[str]]:
        """
        Expand known industrial acronyms in query.

        Returns:
            Tuple of (expanded_query, list_of_related_terms)
        """
        if not self._enable_acronym_expansion:
            return query, []

        # Expand acronyms inline (e.g., "SRVO-063" -> "SRVO (Servo Alarm)-063")
        expanded = expand_acronyms(query, inline=True)

        # Collect related terms for query expansion
        related = []
        words = query.upper().split()
        for word in words:
            # Strip punctuation for matching
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                info = get_acronym_info(clean_word)
                if info:
                    related.extend(get_related_terms(clean_word))

        return expanded, related

    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        request_id: str = "",
        use_gateway: bool = False
    ) -> QueryAnalysis:
        """
        Analyze the user query to determine if web search is beneficial.

        Args:
            query: The user's query text
            context: Optional context (conversation history, user preferences)
            request_id: Request ID for tracking
            use_gateway: If True, route through LLM Gateway service

        Returns QueryAnalysis with:
        - requires_search: bool
        - search_reasoning: explanation
        - query_type: factual, opinion, local_service, crisis, etc.
        - key_topics: list of topics to research
        - suggested_queries: initial search queries
        - estimated_complexity: low, medium, high
        """
        logger.info(f"Analyzing query (gateway={use_gateway}): {query[:100]}...")

        # Expand industrial acronyms for better understanding
        expanded_query, related_terms = self._expand_query_acronyms(query)
        if expanded_query != query:
            logger.debug(f"Expanded query: {expanded_query[:100]}...")

        # Build analysis prompt with expanded query
        prompt = self._build_analysis_prompt(expanded_query, context)

        try:
            if use_gateway:
                result = await self._call_via_gateway(prompt, request_id)
            else:
                result = await self._call_ollama(prompt, request_id)
            analysis = self._parse_analysis(result, query)  # Use original for response

            # Add related terms from acronym expansion to key_topics
            if related_terms:
                analysis.key_topics = list(set(analysis.key_topics + related_terms))

            logger.info(f"Query analysis: requires_search={analysis.requires_search}, "
                       f"type={analysis.query_type}, complexity={analysis.estimated_complexity}")
            return analysis
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Default to requiring search on error
            return QueryAnalysis(
                requires_search=True,
                search_reasoning=f"Analysis failed, defaulting to search: {str(e)}",
                query_type="unknown",
                key_topics=[query],
                suggested_queries=[query],
                estimated_complexity="medium",
                confidence=0.3
            )

    async def create_search_plan(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]] = None,
        request_id: str = "",
        use_gateway: bool = False
    ) -> SearchPlan:
        """
        Create a comprehensive search plan based on query analysis.

        Args:
            query: The original query
            analysis: QueryAnalysis from analyze()
            context: Optional context dictionary
            request_id: Request ID for tracking
            use_gateway: If True, route through LLM Gateway service

        The plan includes:
        - Decomposed questions to answer
        - Search phases (initial, refinement, verification)
        - Priority order for queries
        - Fallback strategies
        """
        logger.info(f"Creating search plan (gateway={use_gateway}) for: {query[:100]}...")

        prompt = self._build_plan_prompt(query, analysis, context)

        try:
            if use_gateway:
                result = await self._call_via_gateway(prompt, request_id)
            else:
                result = await self._call_ollama(prompt, request_id)
            plan = self._parse_plan(result, query, analysis)
            logger.info(f"Search plan created: {len(plan.decomposed_questions)} questions, "
                       f"{len(plan.search_phases)} phases")
            return plan
        except Exception as e:
            logger.error(f"Search plan creation failed: {e}")
            # Create basic plan from analysis
            return SearchPlan(
                original_query=query,
                decomposed_questions=[query],
                search_phases=[
                    {"phase": "initial", "queries": analysis.suggested_queries or [query]},
                    {"phase": "refinement", "queries": []},
                ],
                priority_order=list(range(len(analysis.suggested_queries or [query]))),
                fallback_strategies=["broaden search terms", "try alternative phrasing"],
                estimated_iterations=3 if analysis.estimated_complexity == "medium" else (
                    2 if analysis.estimated_complexity == "low" else 5
                ),
                reasoning=f"Basic plan from analysis: {str(e)}"
            )

    def _build_analysis_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Build prompt for query analysis"""
        context_info = ""
        if context and context.get("conversation_history"):
            recent = context["conversation_history"][-3:]
            context_info = "\n".join([f"- {m.get('role', 'user')}: {m.get('content', '')[:100]}"
                                      for m in recent])
            context_info = f"\n\nRecent conversation:\n{context_info}"

        # Role-based prompt with thorough reasoning for industrial troubleshooting accuracy
        return f"""<role>INDUSTRIAL QUERY ANALYZER for manufacturing automation</role>
<expertise>FANUC robotics, Allen-Bradley PLCs, Siemens automation, servo systems, industrial troubleshooting</expertise>

Think through this step by step, providing thorough reasoning for each consideration.

Analyze this user query to determine if web search would be helpful and what level of reasoning is required.

Query: "{query}"{context_info}

Respond with a JSON object containing:
{{
    "requires_search": true/false,
    "search_reasoning": "explanation of why search is or isn't needed",
    "query_type": "one of: factual, opinion, local_service, crisis, how_to, definition, current_events, personal, creative, troubleshooting, technical_research, comparative_analysis, debugging, problem_solving",
    "key_topics": ["topic1", "topic2", ...],
    "suggested_queries": ["search query 1", "search query 2", ...],
    "priority_domains": ["domain1.com", "domain2.org", ...],
    "estimated_complexity": "low/medium/high",
    "confidence": 0.0-1.0,
    "requires_thinking_model": true/false,
    "reasoning_complexity": "simple/moderate/complex/expert",
    "thinking_model_reasoning": "why thinking model is or isn't needed"
}}

Guidelines:
- requires_search=true if: asking for facts, current info, technical details, specific data, troubleshooting
- requires_search=false if: casual conversation, personal opinion requested, creative writing, simple greeting
- For technical topics: prioritize official documentation, GitHub, Stack Overflow
- For research: prioritize arxiv, IEEE, ACM, academic sources
- For engineering: include specifications, standards, and manufacturer docs

THINKING MODEL CLASSIFICATION:
- requires_thinking_model=TRUE for queries involving:
  * Diagnosing WHY something isn't working (root cause analysis)
  * Intermittent or conditional failures (e.g., "only happens when warm")
  * Multi-step problem solving requiring reasoning chains
  * Comparative analysis between multiple options
  * Complex procedures requiring judgment and sequencing
  * Questions requiring synthesis across multiple technical domains

- requires_thinking_model=FALSE for:
  * Simple fact lookup (what is X, what does Y mean)
  * Error code MEANING/DEFINITION lookup (e.g., "What does SRVO-063 mean?")
  * Part number lookup or identification
  * Parameter definitions or settings lookup
  * Basic definitions and terminology
  * Current events/news
  * Straightforward how-to questions with known steps

EXAMPLES:
  - "What does FANUC SRVO-063 alarm mean?" → FALSE (just looking up definition)
  - "FANUC R-30iB J1 motor part number" → FALSE (just looking up part number)
  - "What is $PARAM_GROUP[1].$PAYLOAD?" → FALSE (parameter definition)
  - "Robot intermittently loses encoder position after warm-up, what's causing this?" → TRUE (diagnosis)
  - "Compare R-2000iC vs M-900iB for spot welding" → TRUE (comparative analysis)
  - "Why does my robot drift after warm restart but not cold start?" → TRUE (root cause)

- reasoning_complexity levels:
  * simple: direct fact retrieval, definitions, error code meanings, part numbers
  * moderate: synthesis of multiple sources, basic summarization
  * complex: multi-step reasoning, comparing alternatives, weighing trade-offs
  * expert: technical troubleshooting, debugging, domain-specific analysis

Return ONLY the JSON object, no other text."""

    def _build_plan_prompt(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for search planning"""
        return f"""Create a comprehensive web search plan for this query.

Original Query: "{query}"
Query Type: {analysis.query_type}
Complexity: {analysis.estimated_complexity}
Key Topics: {', '.join(analysis.key_topics)}

Create a multi-phase search plan to thoroughly research this topic. The plan should:
1. Break down the query into sub-questions that need answering
2. Define search phases (initial broad search, then targeted refinement)
3. Prioritize which aspects to search first
4. Include fallback strategies if initial searches don't yield results

Respond with a JSON object:
{{
    "decomposed_questions": [
        "What is X?",
        "How does X relate to Y?",
        "Where can I find X near location?"
    ],
    "search_phases": [
        {{"phase": "initial", "queries": ["broad query 1", "broad query 2"], "goal": "gather overview"}},
        {{"phase": "detail", "queries": ["specific query 1"], "goal": "get specific information"}},
        {{"phase": "verify", "queries": ["verification query"], "goal": "cross-check facts"}}
    ],
    "priority_order": [0, 1, 2],
    "fallback_strategies": [
        "broaden terms if no results",
        "try synonyms",
        "search for related topics"
    ],
    "estimated_iterations": 3-10,
    "reasoning": "explanation of the plan"
}}

For complex queries, plan more iterations. For simple queries, fewer.
Return ONLY the JSON object, no other text."""

    async def _call_ollama(
        self,
        prompt: str,
        request_id: str = "",
        num_predict: int = 1024,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Call Ollama API for LLM inference with context utilization tracking

        Args:
            prompt: The prompt to send to the LLM
            request_id: Request ID for tracking context utilization
            num_predict: Maximum tokens to generate (default 1024, use 2048 for larger outputs)
            model: Optional model override (uses self.model if not provided)
            temperature: Optional temperature override (default 0.3)
        """
        use_model = model or self.model
        use_temp = temperature if temperature is not None else 0.3

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": use_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": use_temp,
                        "num_predict": num_predict
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "")

            # Track context utilization
            if request_id:
                metrics = get_performance_metrics()
                metrics.record_context_utilization(
                    request_id=request_id,
                    agent_name="analyzer",
                    model_name=use_model,
                    input_text=prompt,
                    output_text=result,
                    context_window=get_model_context_window(use_model)
                )

            return result

    async def _call_via_gateway(self, prompt: str, request_id: str = "") -> str:
        """
        Call LLM via Gateway service with automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            request_id: Request ID for tracking context utilization

        Returns:
            LLM response text
        """
        try:
            gateway = get_gateway_client()

            response: GatewayResponse = await gateway.generate(
                prompt=prompt,
                model=LogicalModel.ANALYZER,
                timeout=self.timeout,
                options={
                    "temperature": 0.3,
                    "num_predict": 1024,
                }
            )

            result = response.content

            # Track context utilization
            if request_id and result:
                metrics = get_performance_metrics()
                metrics.record_context_utilization(
                    request_id=request_id,
                    agent_name="analyzer",
                    model_name=response.model,
                    input_text=prompt,
                    output_text=result,
                    context_window=get_model_context_window(response.model)
                )

            if response.fallback_used:
                logger.info(f"Gateway analyzer used fallback to direct Ollama (model: {response.model})")

            return result

        except Exception as e:
            logger.error(f"Gateway analyzer call failed: {e}, falling back to direct Ollama")
            return await self._call_ollama(prompt, request_id)

    def _parse_analysis(self, response: str, query: str) -> QueryAnalysis:
        """Parse LLM response into QueryAnalysis"""
        try:
            # Extract JSON from response using proper brace matching
            json_str = extract_json_object(response)
            logger.info(f"[ANALYZER DEBUG] Response length: {len(response)}, JSON extracted: {json_str is not None}")
            if not json_str:
                # Log more details to debug the issue
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                logger.warning(f"[ANALYZER DEBUG] No JSON found. start_idx={start_idx}, end_idx={end_idx}")
                logger.warning(f"[ANALYZER DEBUG] First 500 chars: {response[:500]}")
                logger.warning(f"[ANALYZER DEBUG] Last 200 chars: {response[-200:]}")
            if json_str:
                data = json.loads(json_str)

                # Determine thinking model requirement
                requires_thinking = data.get("requires_thinking_model", False)
                reasoning_complexity = data.get("reasoning_complexity", "moderate")
                query_type = data.get("query_type", "informational")
                logger.debug(f"[ANALYZER DEBUG] Parsed query_type from LLM: {query_type}")

                # FIX 5: Force industrial classification when error code/equipment patterns detected
                # This ensures HSEA domain knowledge is queried and prioritized
                import re
                industrial_patterns = [
                    # FANUC robotics error codes
                    r'\bSRVO[-_]?\d{3}\b',  # SRVO-023, SRVO_023
                    r'\bMOTN[-_]?\d{3}\b',  # MOTN-023, MOTN_023
                    r'\bSYST[-_]?\d{3}\b',  # SYST-023
                    r'\bINTP[-_]?\d{3}\b',  # INTP-023
                    r'\bHOST[-_]?\d{3}\b',  # HOST-023
                    r'\bR-30i[AB]\b',       # R-30iA, R-30iB
                    r'\bLR\s?Mate\b',       # LR Mate
                    # Allen-Bradley / Rockwell
                    r'\bControlLogix\b',    # Allen-Bradley
                    r'\b1756[-_]L\d+\b',    # 1756-L71
                    r'\bCompactLogix\b',
                    r'\bPLC[-_]?\d\b',
                    # Injection Molding Machines (IMM)
                    r'\binjection\s+mold',  # injection molding, injection molder
                    r'\bbarrel\s+(heater|temperature|zone)',  # barrel heater, barrel temperature
                    r'\b(plasticizing|plastici[sz]er)\b',
                    r'\bscrew\s+(speed|rpm|rotation)',
                    r'\b(clamp|clamping)\s+(force|tonnage|pressure)',
                    r'\bplaten\b',
                    r'\bnozzle\s+(temperature|heater)',
                    r'\bhopper\s+(dryer|temperature)',
                    r'\bmold\s+(temperature|cooling)',
                    # IMM manufacturers
                    r'\b(Engel|Arburg|Husky|Nissei|Toshiba|Sumitomo|Milacron|Haitian|JSW)\b',
                    # Siemens automation
                    r'\bSinamics\b',
                    r'\bS7[-_]?\d{3,4}\b',  # S7-1200, S7-1500
                ]
                query_upper = query.upper()
                is_industrial_error = any(re.search(pattern, query, re.IGNORECASE) for pattern in industrial_patterns)
                # Force search when industrial error code detected - domain knowledge is critical
                force_search = False
                if is_industrial_error:
                    query_type = "industrial_troubleshooting"
                    requires_thinking = True
                    reasoning_complexity = "expert"
                    force_search = True  # Industrial errors MUST use search to get HSEA domain knowledge
                    logger.info(f"FIX 5: Detected industrial error code pattern - forcing query_type=industrial_troubleshooting, requires_search=True")

                # Apply heuristics for thinking model if LLM didn't classify
                thinking_query_types = {
                    "troubleshooting", "technical_research", "comparative_analysis",
                    "debugging", "problem_solving", "industrial_troubleshooting"
                }
                if query_type in thinking_query_types and not requires_thinking:
                    requires_thinking = True
                    reasoning_complexity = "complex" if reasoning_complexity == "moderate" else reasoning_complexity

                # Also check for keywords indicating complex reasoning
                thinking_keywords = [
                    "troubleshoot", "debug", "fix", "configure", "optimize",
                    "compare", "evaluate", "analyze", "diagnose", "root cause",
                    "architecture", "system", "integration", "trade-off"
                ]
                query_lower = query.lower()
                if any(kw in query_lower for kw in thinking_keywords):
                    if not requires_thinking:
                        requires_thinking = True
                        reasoning_complexity = "complex"

                # Use force_search for industrial errors, otherwise use LLM classification
                final_requires_search = force_search or data.get("requires_search", True)

                analysis = QueryAnalysis(
                    requires_search=final_requires_search,
                    search_reasoning=data.get("search_reasoning", ""),
                    query_type=query_type,
                    key_topics=data.get("key_topics", [query]),
                    suggested_queries=data.get("suggested_queries", [query]),
                    priority_domains=data.get("priority_domains", []),
                    estimated_complexity=data.get("estimated_complexity", "medium"),
                    confidence=data.get("confidence", 0.7),
                    requires_thinking_model=requires_thinking,
                    reasoning_complexity=reasoning_complexity,
                    thinking_model_reasoning=data.get("thinking_model_reasoning", "")
                )

                if requires_thinking:
                    logger.info(f"Query classified as requiring THINKING MODEL: "
                               f"type={query_type}, complexity={reasoning_complexity}")

                return analysis

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")

        # Fallback: analyze response text and apply keyword heuristics
        requires_search = "true" in response.lower() or "yes" in response.lower()

        # Check keywords for thinking model
        thinking_keywords = [
            "troubleshoot", "debug", "fix", "configure", "optimize",
            "compare", "evaluate", "analyze", "diagnose", "robot", "controller",
            "architecture", "system", "integration"
        ]
        query_lower = query.lower()
        requires_thinking = any(kw in query_lower for kw in thinking_keywords)

        return QueryAnalysis(
            requires_search=requires_search,
            search_reasoning=response[:200],
            query_type="unknown",
            key_topics=[query],
            suggested_queries=[query],
            estimated_complexity="medium",
            confidence=0.5,
            requires_thinking_model=requires_thinking,
            reasoning_complexity="complex" if requires_thinking else "moderate",
            thinking_model_reasoning="Keyword-based classification" if requires_thinking else ""
        )

    def _parse_plan(
        self,
        response: str,
        query: str,
        analysis: QueryAnalysis
    ) -> SearchPlan:
        """Parse LLM response into SearchPlan"""
        try:
            # Extract JSON from response using proper brace matching
            json_str = extract_json_object(response)
            if json_str:
                data = json.loads(json_str)
                # Ensure decomposed_questions is a list of strings
                decomposed = data.get("decomposed_questions", [query])
                if isinstance(decomposed, str):
                    # If LLM returned a string, wrap it in a list
                    decomposed = [decomposed]
                elif not isinstance(decomposed, list):
                    decomposed = [query]
                # Ensure each element is a string
                decomposed = [str(q) for q in decomposed if q]
                if not decomposed:
                    decomposed = [query]
                return SearchPlan(
                    original_query=query,
                    decomposed_questions=decomposed,
                    search_phases=data.get("search_phases", []),
                    priority_order=data.get("priority_order", [0]),
                    fallback_strategies=data.get("fallback_strategies", []),
                    estimated_iterations=data.get("estimated_iterations", 3),
                    reasoning=data.get("reasoning", "")
                )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan JSON: {e}")

        # Fallback: create basic plan
        return SearchPlan(
            original_query=query,
            decomposed_questions=[query],
            search_phases=[
                {"phase": "initial", "queries": analysis.suggested_queries or [query]}
            ],
            priority_order=[0],
            fallback_strategies=["broaden search"],
            estimated_iterations=3,
            reasoning="Fallback plan due to parsing error"
        )

    async def evaluate_urls_for_scraping(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_urls: int = None,  # Will use dynamic default if None
        model: Optional[str] = None  # Override model (uses url_evaluator config by default)
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to evaluate which URLs are worth scraping for the query.

        Args:
            query: User's search query
            search_results: List of search result dicts with title, url, snippet, domain
            max_urls: Maximum URLs to evaluate (default from ANALYZER_LIMITS)
            model: Optional model override (default: url_evaluator from llm_config)

        Returns list of URLs with relevance scores and reasoning.
        Only returns URLs that are likely to contain useful information.
        """
        if not search_results:
            return []

        # Get url_evaluator config from central config
        url_eval_config = get_config_for_task("url_evaluator")
        eval_model = model or url_eval_config.model
        eval_temp = url_eval_config.temperature
        eval_max_tokens = url_eval_config.max_tokens

        # Use dynamic limit based on model context window
        if max_urls is None:
            max_urls = ANALYZER_LIMITS["max_urls_to_evaluate"]

        logger.debug(f"URL evaluation using model: {eval_model}")

        # Format search results for LLM evaluation
        results_summary = []
        for i, result in enumerate(search_results[:max_urls]):  # Evaluate up to max_urls
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            snippet = result.get("snippet", "")[:200]
            domain = result.get("source_domain", result.get("domain", ""))

            results_summary.append(
                f"{i+1}. **{title}**\n"
                f"   URL: {url}\n"
                f"   Domain: {domain}\n"
                f"   Snippet: {snippet}"
            )

        results_text = "\n\n".join(results_summary)

        prompt = f"""Evaluate which of these search results are worth scraping to answer the user's question.

USER'S QUESTION: {query}

SEARCH RESULTS:
{results_text}

For each result, determine if it's likely to contain USEFUL INFORMATION to answer the question.

Consider:
- Does the title/snippet suggest relevant content?
- Is it from a credible/authoritative source for this topic?
- Will scraping this page likely provide actionable information?
- Avoid: generic landing pages, login walls, paywalls, irrelevant topics

Return a JSON array of objects for URLs WORTH SCRAPING (only include relevant ones):
[
  {{"index": 1, "url": "...", "relevance": "high/medium", "reason": "why this is relevant"}},
  {{"index": 3, "url": "...", "relevance": "high/medium", "reason": "why this is relevant"}}
]

Only include URLs with HIGH or MEDIUM relevance. Skip low relevance or irrelevant results.
Return ONLY the JSON array, no other text. /no_think"""

        try:
            result = await self._call_ollama(
                prompt,
                num_predict=eval_max_tokens,
                model=eval_model,
                temperature=eval_temp
            )

            # Parse JSON response
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                evaluated = json.loads(json_match.group())

                # Map back to full result data
                relevant_urls = []
                for item in evaluated[:max_urls]:
                    idx = item.get("index", 0) - 1
                    if 0 <= idx < len(search_results):
                        original = search_results[idx]
                        relevant_urls.append({
                            "url": original.get("url", item.get("url", "")),
                            "title": original.get("title", ""),
                            "relevance": item.get("relevance", "medium"),
                            "reason": item.get("reason", ""),
                            "domain": original.get("source_domain", original.get("domain", ""))
                        })

                logger.info(f"URL evaluation: {len(relevant_urls)}/{len(search_results)} URLs deemed relevant")
                return relevant_urls

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse URL evaluation JSON: {e}")
        except Exception as e:
            logger.error(f"URL evaluation failed: {e}")

        # Fallback: return top results by position
        logger.warning("Falling back to position-based URL selection")
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "relevance": "medium",
                "reason": "fallback selection",
                "domain": r.get("source_domain", r.get("domain", ""))
            }
            for r in search_results[:5]
        ]

    async def evaluate_content_coverage(
        self,
        query: str,
        decomposed_questions: List[str],
        scraped_content: List[Dict[str, Any]],
        max_content_chars: int = 20000
    ) -> Dict[str, Any]:
        """
        Evaluate if scraped content sufficiently answers the decomposed questions.

        This is called AFTER URL scraping to determine if additional searches
        are needed to fill information gaps.

        Args:
            query: Original user query
            decomposed_questions: Questions that need to be answered
            scraped_content: List of scraped page content
            max_content_chars: Max chars to include in prompt

        Returns:
            {
                "is_sufficient": bool,
                "coverage_score": float (0-1),
                "answered_questions": [str],
                "unanswered_questions": [str],
                "information_gaps": [str],
                "suggested_queries": [str],
                "reasoning": str
            }
        """
        if not scraped_content:
            return {
                "is_sufficient": False,
                "coverage_score": 0.0,
                "answered_questions": [],
                "unanswered_questions": decomposed_questions,
                "information_gaps": ["No content was scraped"],
                "suggested_queries": [query],
                "reasoning": "No content available to evaluate"
            }

        logger.info(f"Evaluating content coverage for {len(decomposed_questions)} questions "
                   f"across {len(scraped_content)} scraped sources")

        # Build content summary for evaluation
        content_summary = []
        total_chars = 0
        for idx, content in enumerate(scraped_content, 1):
            title = content.get("title", "Unknown")
            text = content.get("content", "")[:4000]  # Limit per source

            if total_chars + len(text) > max_content_chars:
                text = text[:max(500, max_content_chars - total_chars)]

            content_summary.append(f"[Source {idx}: {title}]\n{text}")
            total_chars += len(text)

            if total_chars >= max_content_chars:
                break

        content_text = "\n\n---\n\n".join(content_summary)
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(decomposed_questions)])

        # Use central config for coverage evaluation model
        llm_config = get_llm_config()
        analysis_model = llm_config.pipeline.coverage_evaluator.model

        prompt = f"""Analyze if the scraped content adequately answers the user's questions.

ORIGINAL QUERY: {query}

QUESTIONS TO ANSWER:
{questions_text}

SCRAPED CONTENT:
{content_text}

TASK: Evaluate which questions are answered by the content and identify gaps.

For each question, determine:
- Is it FULLY answered (specific, actionable information found)?
- Is it PARTIALLY answered (some info but incomplete)?
- Is it UNANSWERED (no relevant information)?

Respond with JSON:
{{
    "coverage_score": 0.0-1.0,
    "question_status": [
        {{"question": "...", "status": "fully_answered|partially_answered|unanswered", "evidence": "brief quote or summary"}},
        ...
    ],
    "information_gaps": [
        "specific missing info 1",
        "specific missing info 2"
    ],
    "suggested_queries": [
        "new search query to fill gap 1",
        "new search query to fill gap 2"
    ],
    "reasoning": "overall assessment"
}}

Be specific about what information is missing. Generate targeted search queries that would fill the gaps.
Return ONLY the JSON object. /no_think"""

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": analysis_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "10m",  # OPTIMIZATION: Keep coverage model loaded
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 1024,
                            "num_ctx": ANALYZER_LIMITS["num_ctx"]  # Dynamic based on model context window
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

                # Track context utilization for coverage evaluation
                metrics = get_performance_metrics()
                metrics.record_context_utilization(
                    request_id=f"coverage_{hash(query) % 10000}",
                    agent_name="analyzer_coverage",
                    model_name=analysis_model,
                    input_text=prompt,
                    output_text=result,
                    context_window=get_model_context_window(analysis_model)
                )

            # Parse JSON response - use more robust extraction
            # Find the outermost balanced braces
            result_stripped = result.strip()
            start_idx = result_stripped.find('{')
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(result_stripped[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                json_str = result_stripped[start_idx:end_idx]
                data = json.loads(json_str)

                # Extract answered/unanswered questions
                answered = []
                unanswered = []
                for qs in data.get("question_status", []):
                    q = qs.get("question", "")
                    status = qs.get("status", "unanswered")
                    if status == "fully_answered":
                        answered.append(q)
                    elif status in ["partially_answered", "unanswered"]:
                        unanswered.append(q)

                coverage = data.get("coverage_score", 0.5)
                is_sufficient = coverage >= 0.7 and len(unanswered) == 0

                result = {
                    "is_sufficient": is_sufficient,
                    "coverage_score": coverage,
                    "answered_questions": answered,
                    "unanswered_questions": unanswered,
                    "information_gaps": data.get("information_gaps", []),
                    "suggested_queries": data.get("suggested_queries", []),
                    "reasoning": data.get("reasoning", "")
                }

                logger.info(f"Content coverage: {coverage:.0%}, sufficient={is_sufficient}, "
                           f"gaps={len(result['information_gaps'])}, new_queries={len(result['suggested_queries'])}")

                return result
            else:
                logger.warning(f"No JSON found in coverage evaluation response")
                raise ValueError("No JSON object found in response")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse coverage evaluation JSON: {e}")
        except Exception as e:
            logger.error(f"Content coverage evaluation failed: {e}")

        # Fallback: assume sufficient if we have content
        return {
            "is_sufficient": len(scraped_content) >= 3,
            "coverage_score": 0.5,
            "answered_questions": [],
            "unanswered_questions": decomposed_questions,
            "information_gaps": ["Evaluation failed"],
            "suggested_queries": [],
            "reasoning": f"Fallback evaluation: {len(scraped_content)} sources scraped"
        }

    async def should_continue_search(
        self,
        query: str,
        current_results: List[Dict[str, Any]],
        search_plan: SearchPlan,
        iteration: int
    ) -> tuple[bool, str, List[str]]:
        """
        Determine if search should continue based on current results.

        Returns:
        - should_continue: bool
        - reason: str
        - new_queries: List[str] (if continuing)
        """
        if not current_results:
            return True, "No results yet", search_plan.decomposed_questions[:3]

        # Build prompt to evaluate results
        results_summary = "\n".join([
            f"- {r.get('title', 'Unknown')}: {r.get('snippet', '')[:100]}"
            for r in current_results[:5]
        ])

        prompt = f"""Evaluate if we have enough information to answer this query.

Original Query: "{query}"
Questions to Answer: {search_plan.decomposed_questions}

Current Results ({len(current_results)} found):
{results_summary}

Iteration: {iteration} of max {search_plan.estimated_iterations}

Respond with JSON:
{{
    "information_sufficient": true/false,
    "reason": "explanation",
    "unanswered_questions": ["question1", "question2"],
    "new_search_queries": ["query1", "query2"] (if continuing)
}}

Return ONLY the JSON object."""

        try:
            response = await self._call_ollama(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sufficient = data.get("information_sufficient", False)
                reason = data.get("reason", "")
                new_queries = data.get("new_search_queries", [])
                return not sufficient, reason, new_queries
        except Exception as e:
            logger.warning(f"Continue check failed: {e}")

        # Default: continue if under iteration limit
        return iteration < search_plan.estimated_iterations, "Default continuation", []

    # ============================================================
    # GSW-STYLE ENTITY TRACKING (Phase 2 Enhancement)
    # ============================================================

    @property
    def entity_tracker(self) -> Optional["EntityTracker"]:
        """Get the entity tracker instance"""
        return self._entity_tracker

    @entity_tracker.setter
    def entity_tracker(self, tracker: "EntityTracker") -> None:
        """Set the entity tracker instance"""
        self._entity_tracker = tracker
        self._entity_extraction_enabled = tracker is not None

    def enable_entity_extraction(self, tracker: "EntityTracker") -> None:
        """Enable GSW entity extraction with the given tracker"""
        self._entity_tracker = tracker
        self._entity_extraction_enabled = True
        logger.info("GSW entity extraction enabled")

    def disable_entity_extraction(self) -> None:
        """Disable GSW entity extraction"""
        self._entity_extraction_enabled = False
        logger.info("GSW entity extraction disabled")

    async def extract_entities_from_content(
        self,
        content: str,
        source_url: str = "",
        auto_reconcile: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from content using GSW EntityTracker.

        This enables 51% token reduction by tracking entities rather than
        storing full document content.

        Args:
            content: Text content to extract entities from
            source_url: URL where content was found
            auto_reconcile: Whether to automatically merge with existing entities

        Returns:
            List of extracted entity dictionaries
        """
        if not self._entity_extraction_enabled or not self._entity_tracker:
            logger.debug("Entity extraction not enabled, skipping")
            return []

        try:
            # Extract entities using the tracker
            entities = await self._entity_tracker.extract_entities(
                content=content,
                source_url=source_url
            )

            if not entities:
                return []

            # Optionally reconcile with existing entities
            if auto_reconcile:
                self._entity_tracker.reconcile(entities)

            # Return as dictionaries for scratchpad storage
            return [entity.to_dict() for entity in entities]

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def extract_entities_from_scraped_content(
        self,
        scraped_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract entities from multiple scraped pages.

        Args:
            scraped_content: List of scraped page content dictionaries

        Returns:
            Summary with entity counts and key entities found
        """
        if not self._entity_extraction_enabled or not self._entity_tracker:
            return {"enabled": False, "entities": [], "count": 0}

        all_entities = []
        for page in scraped_content:
            content = page.get("content", "")
            url = page.get("url", "")

            if content:
                entities = await self.extract_entities_from_content(
                    content=content,
                    source_url=url,
                    auto_reconcile=True
                )
                all_entities.extend(entities)

        # Get tracker stats
        stats = self._entity_tracker.get_stats()

        return {
            "enabled": True,
            "new_entities_extracted": len(all_entities),
            "total_entities": stats.get("total_entities", 0),
            "total_relations": stats.get("total_relations", 0),
            "entities_merged": stats.get("entities_merged", 0),
            "top_entities": all_entities[:10]  # Return top 10 for summary
        }

    def generate_entity_context(
        self,
        query: str,
        max_entities: int = 10,
        max_length: int = 2000
    ) -> str:
        """
        Generate entity-centric context for LLM synthesis.

        Instead of full document retrieval, generates focused summaries
        for entities most relevant to the query. Key to 51% token reduction.

        Args:
            query: The query to generate context for
            max_entities: Maximum number of entities to include
            max_length: Maximum total context length

        Returns:
            Formatted entity context string
        """
        if not self._entity_extraction_enabled or not self._entity_tracker:
            return ""

        return self._entity_tracker.generate_workspace_context(
            query=query,
            max_entities=max_entities,
            max_length=max_length
        )

    def get_entity_stats(self) -> Dict[str, Any]:
        """Get entity tracking statistics"""
        if not self._entity_tracker:
            return {"enabled": False}

        stats = self._entity_tracker.get_stats()
        stats["enabled"] = self._entity_extraction_enabled
        return stats
