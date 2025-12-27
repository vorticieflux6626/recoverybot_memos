"""
Query Analyzer Agent for Agentic Search System

Analyzes user queries using LLM to determine:
1. Whether web search would be beneficial
2. Query type and complexity
3. Key topics to research
4. Initial search strategy
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from .models import QueryAnalysis, SearchPlan

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes user queries to determine if web search is needed
    and creates an initial search strategy.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b"
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 60.0

    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """
        Analyze the user query to determine if web search is beneficial.

        Returns QueryAnalysis with:
        - requires_search: bool
        - search_reasoning: explanation
        - query_type: factual, opinion, local_service, crisis, etc.
        - key_topics: list of topics to research
        - suggested_queries: initial search queries
        - estimated_complexity: low, medium, high
        """
        logger.info(f"Analyzing query: {query[:100]}...")

        # Build analysis prompt
        prompt = self._build_analysis_prompt(query, context)

        try:
            result = await self._call_ollama(prompt)
            analysis = self._parse_analysis(result, query)
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
        context: Optional[Dict[str, Any]] = None
    ) -> SearchPlan:
        """
        Create a comprehensive search plan based on query analysis.

        The plan includes:
        - Decomposed questions to answer
        - Search phases (initial, refinement, verification)
        - Priority order for queries
        - Fallback strategies
        """
        logger.info(f"Creating search plan for: {query[:100]}...")

        prompt = self._build_plan_prompt(query, analysis, context)

        try:
            result = await self._call_ollama(prompt)
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

        return f"""Analyze this user query to determine if web search would be helpful and what level of reasoning is required.

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
- requires_search=true if: asking for facts, current info, services, locations, specific data
- requires_search=false if: casual conversation, personal opinion requested, creative writing, simple greeting
- For recovery/health topics: prioritize SAMHSA, NIH, CDC, Mayo Clinic
- For local services: search should include location terms
- For crisis: include crisis hotlines and immediate resources

THINKING MODEL CLASSIFICATION:
- requires_thinking_model=TRUE for queries involving:
  * Technical troubleshooting (fixing systems, debugging issues)
  * Multi-step problem solving (complex procedures, workflows)
  * Comparative analysis (evaluating options, trade-offs)
  * Expert domain knowledge (industrial equipment, medical procedures, engineering)
  * Root cause analysis (why something isn't working)
  * System architecture or configuration questions
  * Questions requiring synthesis across multiple technical domains

- requires_thinking_model=FALSE for:
  * Simple fact lookup (what, when, where)
  * Basic definitions
  * Local service finding
  * Current events/news
  * Straightforward how-to questions

- reasoning_complexity levels:
  * simple: direct fact retrieval, definitions
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

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for LLM inference"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 512
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    def _parse_analysis(self, response: str, query: str) -> QueryAnalysis:
        """Parse LLM response into QueryAnalysis"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Determine thinking model requirement
                requires_thinking = data.get("requires_thinking_model", False)
                reasoning_complexity = data.get("reasoning_complexity", "moderate")
                query_type = data.get("query_type", "informational")

                # Apply heuristics for thinking model if LLM didn't classify
                thinking_query_types = {
                    "troubleshooting", "technical_research", "comparative_analysis",
                    "debugging", "problem_solving"
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

                analysis = QueryAnalysis(
                    requires_search=data.get("requires_search", True),
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
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return SearchPlan(
                    original_query=query,
                    decomposed_questions=data.get("decomposed_questions", [query]),
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
        max_urls: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to evaluate which URLs are worth scraping for the query.

        Returns list of URLs with relevance scores and reasoning.
        Only returns URLs that are likely to contain useful information.
        """
        if not search_results:
            return []

        # Format search results for LLM evaluation
        results_summary = []
        for i, result in enumerate(search_results[:15]):  # Evaluate top 15
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
Return ONLY the JSON array, no other text."""

        try:
            result = await self._call_ollama(prompt)

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

        # OPTIMIZATION: Use gemma3:4b for faster coverage evaluation (4B vs 8B params)
        # Task is simple classification - doesn't need full reasoning power
        analysis_model = "gemma3:4b"

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
                            "num_ctx": 16384  # OPTIMIZATION: Reduced from 32K (sufficient for coverage)
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            # Parse JSON response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

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
