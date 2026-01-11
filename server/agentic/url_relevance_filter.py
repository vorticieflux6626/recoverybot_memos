"""
LLM-Based URL Relevance Filter

Evaluates search result URLs for relevance BEFORE scraping, preventing wasted time
on irrelevant sources like language.stackexchange.com for "FANUC servo" queries.

This replaces hardcoded domain blocking with intelligent, query-aware filtering.

Usage:
    filter = URLRelevanceFilter(ollama_url)
    filtered_results = await filter.filter_urls(query, search_results, max_urls=15)

Performance:
    - Uses qwen3:8b for fast evaluation (~3-5s for 20 URLs)
    - Single LLM call evaluates all URLs at once (batch efficiency)
    - Saves 60s+ per irrelevant URL that would have been scraped

Observability:
    - Integrates with llm_logger for LLM call tracking
    - Integrates with decision_logger for filtering decisions
    - Uses centralized prompts from config/prompts.yaml
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import httpx

from .prompt_config import get_prompt_config
from .llm_logger import get_llm_logger
from .decision_logger import get_decision_logger, AgentName, DecisionType

logger = logging.getLogger("agentic.url_relevance_filter")


@dataclass
class SearchResult:
    """Represents a search result for filtering."""
    url: str
    title: str
    snippet: str
    rank: int  # Original search rank


class URLRelevanceFilter:
    """
    LLM-based URL relevance evaluation before scraping.

    Intelligently filters search results to only scrape URLs that are
    actually relevant to the user's query, based on title, snippet, and domain.
    """

    # Fast model for URL evaluation
    MODEL = "qwen3:8b"

    # Timeout for LLM call
    TIMEOUT = 30.0

    # Maximum URLs to evaluate in one call
    MAX_EVALUATE = 30

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize URL relevance filter.

        Args:
            ollama_url: Base URL for Ollama API
        """
        self.ollama_url = ollama_url.rstrip("/")
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.TIMEOUT)
        return self._http_client

    async def filter_urls(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_urls: int = 15,
        request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter URLs using LLM relevance evaluation.

        Args:
            query: Original user query
            search_results: List of search results with 'url', 'title', 'snippet' keys
            max_urls: Maximum number of URLs to return after filtering
            request_id: Optional request ID for observability tracking

        Returns:
            Filtered list of search results (preserves original dict structure)
        """
        if not search_results:
            return []

        start_time = time.time()

        # Limit evaluation batch size
        results_to_evaluate = search_results[:self.MAX_EVALUATE]

        # Build compact representation for LLM
        url_descriptions = self._build_url_descriptions(results_to_evaluate)

        # Generate LLM prompt
        prompt = self._build_prompt(query, url_descriptions)

        try:
            # Call LLM for evaluation
            response = await self._call_llm(prompt, request_id)

            # Parse response to get relevant indices
            relevant_indices = self._parse_response(response, len(results_to_evaluate))

            duration_ms = int((time.time() - start_time) * 1000)

            # Handle three cases:
            # 1. None = parse failure → fall back to original results
            # 2. Empty list = LLM found no relevant URLs → return top 3 by search rank
            # 3. Non-empty list = use filtered results
            if relevant_indices is None:
                logger.warning("URL filter parse failed, falling back to first results")
                decision_made = "fallback_parse_failure"
                filtered_urls = results_to_evaluate[:max_urls]
            elif len(relevant_indices) == 0:
                # LLM found no relevant URLs - this is unusual, fall back to top 3
                fallback_count = min(3, max_urls, len(results_to_evaluate))
                logger.warning(f"URL filter found 0 relevant URLs, falling back to top {fallback_count} by search rank")
                decision_made = f"fallback_no_relevant_found"
                filtered_urls = results_to_evaluate[:fallback_count]
            else:
                filtered_urls = [results_to_evaluate[i] for i in relevant_indices if i < len(results_to_evaluate)]
                filtered_urls = filtered_urls[:max_urls]
                logger.info(f"URL filter: {len(results_to_evaluate)} -> {len(filtered_urls)} relevant URLs")
                decision_made = f"filtered_{len(filtered_urls)}_of_{len(results_to_evaluate)}"

            # Log decision for observability
            if request_id:
                try:
                    decision_logger = get_decision_logger(request_id)
                    await decision_logger.log_decision(
                        agent_name=AgentName.URL_RELEVANCE_FILTER,
                        decision_type=DecisionType.EVALUATION,
                        decision_made=decision_made,
                        reasoning=f"Evaluated {len(results_to_evaluate)} URLs, kept {len(filtered_urls)} relevant",
                        alternatives=[f"url_{i}" for i in (relevant_indices or [])[:5]],
                        confidence=0.8 if relevant_indices else 0.4,
                        metadata={
                            "query": query[:100],
                            "urls_evaluated": len(results_to_evaluate),
                            "urls_passed": len(filtered_urls),
                            "duration_ms": duration_ms,
                            "relevant_indices": relevant_indices[:10] if relevant_indices else [],
                            "filtered_urls": [u.get("url", "")[:80] for u in filtered_urls[:5]]
                        }
                    )
                except Exception as log_err:
                    logger.debug(f"Decision logging failed (non-fatal): {log_err}")

            return filtered_urls

        except Exception as e:
            logger.error(f"URL relevance filter error: {e}")
            # On error, return original results (don't block scraping)
            return results_to_evaluate[:max_urls]

    def _build_url_descriptions(self, results: List[Dict[str, Any]]) -> List[str]:
        """Build compact URL descriptions for LLM evaluation."""
        descriptions = []
        for i, result in enumerate(results):
            url = result.get("url", "")
            title = result.get("title", "")[:80] or "No title"
            snippet = result.get("snippet", "")[:120] or ""

            # Extract domain for context
            try:
                domain = urlparse(url).netloc.replace("www.", "")
            except Exception:
                domain = "unknown"

            desc = f"{i}. [{domain}] {title}"
            if snippet:
                desc += f"\n   {snippet}"
            descriptions.append(desc)

        return descriptions

    def _build_prompt(self, query: str, url_descriptions: List[str]) -> str:
        """Build the LLM prompt for URL relevance evaluation using centralized config."""
        try:
            config = get_prompt_config()
            # Use centralized prompt template from config/prompts.yaml
            template = config.agent_prompts.url_relevance_filter.evaluation
            prompt = template.format(
                query=query,
                url_descriptions=chr(10).join(url_descriptions)
            )
            return prompt
        except Exception as e:
            # Fallback to hardcoded prompt if config fails
            logger.warning(f"Failed to load prompt from config, using fallback: {e}")
            return f'''Query: "{query}"

Which URL indices are relevant to this query?
Relevant = discusses the specific topic/equipment/technology mentioned.
Irrelevant = dictionary definitions, celebrities, unrelated topics.

{chr(10).join(url_descriptions)}

Output ONLY the JSON below. No explanation. No thinking. Just the JSON.
{{"relevant": [indices here]}}'''

    async def _call_llm(self, prompt: str, request_id: Optional[str] = None) -> str:
        """Call Ollama for URL evaluation with LLM logging."""
        client = await self._ensure_http_client()

        payload = {
            "model": self.MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for consistent filtering
                "num_predict": 1024  # Allow space for thinking + array of indices
            }
        }

        # Track LLM call for observability
        llm_logger = None
        if request_id:
            try:
                llm_logger = get_llm_logger(request_id)
            except Exception as e:
                logger.debug(f"Could not get LLM logger: {e}")

        if llm_logger:
            async with llm_logger.track_call(
                agent_name=AgentName.URL_RELEVANCE_FILTER,
                operation="evaluation",
                model=self.MODEL,
                prompt=prompt,
                prompt_template="url_relevance_filter.evaluation",
                metadata={"prompt_length": len(prompt)}
            ) as call:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload
                )

                if response.status_code != 200:
                    call.error = f"Ollama returned {response.status_code}"
                    raise RuntimeError(f"Ollama returned {response.status_code}: {response.text[:200]}")

                data = response.json()
                llm_response = data.get("response", "")
                call.output = llm_response[:500] if llm_response else ""

                # Extract token counts from Ollama response if available
                if "eval_count" in data:
                    call.output_tokens = data.get("eval_count", 0)
                if "prompt_eval_count" in data:
                    call.input_tokens = data.get("prompt_eval_count", 0)
        else:
            # No logging - just make the call
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama returned {response.status_code}: {response.text[:200]}")

            data = response.json()
            llm_response = data.get("response", "")

        # Log the actual response for debugging
        if not llm_response:
            logger.warning(f"LLM returned empty response. Full data: {data}")
        else:
            logger.info(f"URL filter LLM response ({len(llm_response)} chars): {llm_response[:200]}")

        return llm_response

    def _parse_response(self, response: str, max_index: int) -> Optional[List[int]]:
        """Parse LLM response to extract relevant URL indices.

        Returns:
            List[int]: Valid indices of relevant URLs (may be empty if LLM found none)
            None: If parsing failed (caller should fall back to original results)
        """
        # Strip thinking tags that qwen3 might add even with /no_think
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        cleaned = cleaned.strip()

        logger.debug(f"URL filter raw response: {response[:500]}")

        # Try to find JSON in response - handle nested braces
        # Look for {"relevant": [...]} pattern
        json_match = re.search(r'\{[^{}]*"relevant"\s*:\s*\[[^\]]*\][^{}]*\}', cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group())
                indices = data.get("relevant", [])
                logger.debug(f"URL filter parsed indices: {indices}")
                # Validate indices - empty list is valid (means no relevant URLs)
                if isinstance(indices, list):
                    valid = [i for i in indices if isinstance(i, int) and 0 <= i < max_index]
                    # Empty list means LLM found no relevant URLs - this is valid
                    # Caller will decide whether to fall back
                    logger.info(f"URL filter: {len(valid)} relevant URLs identified")
                    return valid
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error: {e}")

        # Try simpler JSON pattern
        json_match = re.search(r'\{[^}]+\}', cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group())
                indices = data.get("relevant", [])
                if isinstance(indices, list):
                    valid = [i for i in indices if isinstance(i, int) and 0 <= i < max_index]
                    return valid
            except json.JSONDecodeError:
                pass

        # Try to extract array directly
        array_match = re.search(r'\[[\d,\s]+\]', cleaned)
        if array_match:
            try:
                indices = json.loads(array_match.group())
                if isinstance(indices, list):
                    valid = [i for i in indices if isinstance(i, int) and 0 <= i < max_index]
                    return valid
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract numbers from response
        numbers = re.findall(r'\b(\d+)\b', cleaned)
        indices = [int(n) for n in numbers if int(n) < max_index]
        if indices:
            logger.debug(f"URL filter using fallback number extraction: {indices}")
            return sorted(set(indices))  # Deduplicate and sort

        logger.warning(f"URL filter could not parse response: {cleaned[:200]}")
        return None  # Signal parse failure

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Module-level singleton
_url_filter_instance: Optional[URLRelevanceFilter] = None


def get_url_relevance_filter(ollama_url: str = "http://localhost:11434") -> URLRelevanceFilter:
    """Get or create URL relevance filter singleton."""
    global _url_filter_instance
    if _url_filter_instance is None:
        _url_filter_instance = URLRelevanceFilter(ollama_url)
    return _url_filter_instance
