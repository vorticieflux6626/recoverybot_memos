"""
Synthesizer Agent - Result Combination and Formatting

Combines search results, verifications, and context into
coherent, well-structured answers.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

import httpx

from .models import (
    WebSearchResult,
    VerificationResult,
    SearchState,
    ConfidenceLevel
)
from .context_limits import (
    get_synthesizer_limits,
    get_dynamic_source_allocation,
    get_model_context_window,
    SYNTHESIZER_LIMITS,
    THINKING_SYNTHESIZER_LIMITS,
)
from .metrics import get_performance_metrics

logger = logging.getLogger("agentic.synthesizer")

# Chain-of-Draft instruction to reduce thinking tokens by up to 80%
# Source: https://www.helicone.ai/blog/prompt-thinking-models
CHAIN_OF_DRAFT_INSTRUCTION = """Think step by step, but only keep a minimum draft for each thinking step.
Provide your final answer with citations."""

# Thinking models for complex reasoning tasks
# Updated with validated sampling parameters from DeepSeek API docs
THINKING_MODELS = {
    "deepseek-r1:14b-qwen-distill-q8_0": {
        "vram_gb": 15,
        "context_window": 16384,  # Reduced to fit in VRAM with KV cache
        "max_tokens": 4096,
        "temperature": 0.6,  # VALIDATED: Prevents repetition, maintains coherence
        "top_p": 0.95,       # VALIDATED: Good diversity while filtering improbable tokens
        "description": "Qwen-distilled 14B thinking model - best balance of speed and reasoning quality"
    },
    "deepseek-r1:32b": {
        "vram_gb": 19,
        "context_window": 64000,
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "description": "Largest thinking model - best for complex technical troubleshooting"
    },
    "deepseek-r1:8b": {
        "vram_gb": 5,
        "context_window": 32000,
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.95,
        "description": "Lightweight thinking model for moderate complexity tasks"
    }
}

# Default thinking model - 14b distill offers best speed/quality balance for TITAN RTX 24GB
DEFAULT_THINKING_MODEL = "deepseek-r1:14b-qwen-distill-q8_0"


class SynthesizerAgent:
    """
    Combines search results into coherent answers.

    Features:
    - Structures information logically
    - Cites sources appropriately
    - Highlights conflicts or uncertainties
    - Adapts depth to query complexity
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        model: str = "qwen3:8b"  # Larger model for better synthesis
    ):
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url
        self.model = model
        self.mcp_available = False

    async def check_mcp_available(self) -> bool:
        """Check if MCP Node Editor is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.mcp_url}/api/status")
                self.mcp_available = response.status_code == 200
                return self.mcp_available
        except Exception:
            self.mcp_available = False
            return False

    async def synthesize(
        self,
        query: str,
        search_results: List[WebSearchResult],
        verifications: Optional[List[VerificationResult]] = None,
        context: Optional[Dict[str, Any]] = None,
        request_id: str = ""
    ) -> str:
        """
        Synthesize search results into a coherent answer.

        Args:
            query: Original user query
            search_results: Web search results
            verifications: Optional verification results
            context: Optional conversation context

        Returns:
            Synthesized answer text
        """
        # Format search results
        if search_results:
            results_text = self._format_results(search_results)
        else:
            results_text = "No web search results available."

        # Format verifications if present
        verification_text = ""
        if verifications:
            verified = [v for v in verifications if v.verified]
            conflicts = [v for v in verifications if v.conflicts]

            if verified:
                verification_text += f"\n\nVerified facts: {len(verified)}/{len(verifications)}"
            if conflicts:
                conflict_notes = "; ".join(
                    f"{v.claim[:50]}... has conflicts" for v in conflicts
                )
                verification_text += f"\nPotential conflicts: {conflict_notes}"

        # Build synthesis prompt
        prompt = f"""You are a research synthesizer providing accurate, well-structured answers.
Based on the search results provided, create a comprehensive answer to the user's question.

Original Question: {query}

Search Results:
{results_text}
{verification_text}

Instructions:
1. Provide accurate, technically correct information from the sources
2. Structure your answer with clear sections if appropriate
3. Be direct and solution-focused
4. Include specific details, examples, and references when relevant
5. If information is limited, acknowledge what's known and unknown
6. Focus on practical, actionable guidance
7. Use [Source N] citations for key facts and claims

Your synthesized answer:"""

        try:
            if self.mcp_available:
                synthesis = await self._synthesize_via_mcp(prompt)
            else:
                synthesis = await self._synthesize_via_ollama(prompt, request_id)

            return synthesis or self._fallback_synthesis(query, search_results)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(query, search_results)

    async def _synthesize_via_ollama(self, prompt: str, request_id: str = "") -> str:
        """Execute synthesis via direct Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": 1024
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    synthesis = data.get("response", "")

                    # Track context utilization
                    if request_id and synthesis:
                        metrics = get_performance_metrics()
                        metrics.record_context_utilization(
                            request_id=request_id,
                            agent_name="synthesizer",
                            model_name=self.model,
                            input_text=prompt,
                            output_text=synthesis,
                            context_window=get_model_context_window(self.model)
                        )

                    return synthesis

        except Exception as e:
            logger.error(f"Ollama synthesis failed: {e}")

        return ""

    async def _synthesize_via_mcp(self, prompt: str) -> str:
        """Execute synthesis via MCP Node Editor pipeline"""
        pipeline = {
            "nodes": [
                {
                    "id": 0, "type": "input", "title": "Prompt",
                    "x": 100, "y": 100,
                    "properties": {"source_type": "text", "text": prompt},
                    "inputs": [], "outputs": [{"name": "output", "type": "text"}]
                },
                {
                    "id": 1, "type": "model", "title": "Synthesizer",
                    "x": 300, "y": 100,
                    "properties": {"model": self.model, "temperature": 0.5, "max_tokens": 1024},
                    "inputs": [{"name": "prompt", "type": "text"}],
                    "outputs": [{"name": "response", "type": "text"}]
                },
                {
                    "id": 2, "type": "output", "title": "Answer",
                    "x": 500, "y": 100,
                    "properties": {"format": "text"},
                    "inputs": [{"name": "input", "type": "text"}], "outputs": []
                }
            ],
            "connections": [
                {"id": 0, "source": {"node_id": 0, "port_index": 0, "port_type": "output"}, "target": {"node_id": 1, "port_index": 0}},
                {"id": 1, "source": {"node_id": 1, "port_index": 0, "port_type": "output"}, "target": {"node_id": 2, "port_index": 0}}
            ],
            "settings": {"max_iterations_per_node": 10, "execution_timeout": 120}
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.mcp_url}/api/execute", json=pipeline)
                if response.status_code != 200:
                    return ""

                data = response.json()
                pipeline_id = data.get("pipeline_id")

                # Poll for result
                for _ in range(60):
                    result_response = await client.get(f"{self.mcp_url}/api/result/{pipeline_id}")
                    result = result_response.json()

                    if result.get("status") == "completed":
                        outputs = result.get("outputs", {})
                        for value in outputs.values():
                            if isinstance(value, dict) and "value" in value:
                                return value["value"]
                            elif isinstance(value, str):
                                return value
                        return ""

                    elif result.get("status") == "failed":
                        return ""

                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"MCP synthesis failed: {e}")

        return ""

    def _format_results(self, results: List[WebSearchResult]) -> str:
        """Format search results for the synthesis prompt - use all results to maximize context utilization"""
        formatted = []
        for i, result in enumerate(results[:15], 1):
            formatted.append(
                f"[{i}] **{result.title}**\n"
                f"Source: {result.source_domain}\n"
                f"{result.snippet}\n"
            )
        return "\n---\n".join(formatted)

    def _fallback_synthesis(
        self,
        query: str,
        results: List[WebSearchResult]
    ) -> str:
        """Generate a basic synthesis when LLM fails"""
        if not results:
            return f"""I wasn't able to find specific information for your question: "{query}"

Try these approaches:
- Refine your search query with more specific terms
- Check official documentation or authoritative sources
- Break down complex questions into smaller parts

Please try rephrasing your question or providing more context."""

        # Basic compilation of results - use all available results
        synthesis = f"Here's what I found regarding your question: \"{query}\"\n\n"

        for i, result in enumerate(results[:10], 1):
            synthesis += f"**[{i}] {result.title}**\n{result.snippet}\n\n"

        synthesis += "\nFor more detailed information, consult the original sources linked above."

        return synthesis

    async def synthesize_with_content(
        self,
        query: str,
        search_results: List[WebSearchResult],
        scraped_content: List[Dict[str, Any]],
        verifications: Optional[List[VerificationResult]] = None,
        context: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        request_id: str = ""
    ) -> str:
        """
        Synthesize an answer using full scraped content from web pages.

        This method uses the actual page content (not just snippets) to provide
        more detailed and accurate answers to user questions.

        Args:
            query: Original user query
            search_results: Web search results (snippets)
            scraped_content: Full content scraped from top sources
            verifications: Optional verification results
            context: Optional conversation context
            model_override: Optional model to use instead of default (e.g., thinking model)

        Returns:
            Synthesized answer text that directly addresses the question
        """
        # Determine which model to use
        synthesis_model = model_override or self.model
        is_thinking_model = synthesis_model in THINKING_MODELS

        if is_thinking_model:
            model_config = THINKING_MODELS[synthesis_model]
            logger.info(f"Using THINKING MODEL for synthesis: {synthesis_model} "
                       f"(context: {model_config['context_window']}, max_tokens: {model_config['max_tokens']})")
        else:
            model_config = {
                "context_window": 32768,
                "max_tokens": 2048,
                "temperature": 0.4
            }
        logger.info(f"synthesize_with_content called with {len(scraped_content)} scraped sources")
        for i, sc in enumerate(scraped_content):
            content_len = len(sc.get("content", ""))
            logger.info(f"  Source {i+1}: {sc.get('url', 'unknown')[:50]}... ({content_len} chars)")

        # If no scraped content, fall back to regular synthesis
        if not scraped_content:
            logger.warning("No scraped content available, falling back to snippet-based synthesis")
            return await self.synthesize(query, search_results, verifications, context)

        # Format scraped content for the prompt
        # Use all sources that have meaningful content, prioritize by content length
        valid_sources = [
            s for s in scraped_content
            if s.get("content") and len(s.get("content", "")) > 100
        ]
        # Sort by content length descending (best sources first)
        valid_sources.sort(key=lambda x: len(x.get("content", "")), reverse=True)

        content_sections = []
        total_chars = 0

        # Get dynamic limits based on model context window
        limits = THINKING_SYNTHESIZER_LIMITS if is_thinking_model else SYNTHESIZER_LIMITS
        max_total_chars = limits["max_total_content"]
        max_sources = limits["max_urls_to_scrape"]
        per_source_limit_base = limits["max_content_per_source"]

        logger.info(f"Synthesis context budget: {max_total_chars} chars, {max_sources} sources, {per_source_limit_base} chars/source")

        for i, content in enumerate(valid_sources, 1):
            title = content.get("title", "Source")
            url = content.get("url", "")
            text = content.get("content", "")

            # Calculate how much we can use from this source
            remaining = max_total_chars - total_chars
            if remaining < 500:
                break  # No more room

            # Dynamic per-source allocation: larger budget if fewer sources
            per_source_limit = min(per_source_limit_base, remaining)
            text = text[:per_source_limit]

            content_sections.append(
                f"=== SOURCE [{i}]: {title} ===\n"
                f"URL: {url}\n\n"
                f"{text}\n"
            )
            total_chars += len(text)

            if i >= max_sources:
                break

        full_content = "\n\n".join(content_sections)
        num_sources = len(content_sections)

        # Format verifications if present
        verification_text = ""
        if verifications:
            verified = [v for v in verifications if v.verified]
            if verified:
                verification_text = f"\n\nNote: {len(verified)}/{len(verifications)} claims have been verified."

        # Build a comprehensive synthesis prompt that asks the model to ANSWER the question
        # OPTIMIZATION: Chain-of-Draft instruction reduces thinking tokens by up to 80%
        cod_prefix = CHAIN_OF_DRAFT_INSTRUCTION + "\n\n" if is_thinking_model else ""

        prompt = f"""{cod_prefix}You are an expert research assistant. Your task is to carefully read the source content below and provide a detailed, accurate answer to the user's question.

USER'S QUESTION: {query}

IMPORTANT: You must directly answer the question using information from the sources. You MUST cite your sources using [Source X] notation.

I have provided {num_sources} sources for you to analyze:

{full_content}
{verification_text}

INSTRUCTIONS:
1. Read ALL source content carefully
2. Extract information that directly answers the question
3. **ALWAYS cite sources** using [Source 1], [Source 2], etc. after each fact
4. Be specific - include names, dates, times, addresses, phone numbers, and other details
5. If sources disagree, note the discrepancy with citations
6. If the sources don't contain enough information, clearly state what is known and what remains unclear
7. Use a clear, organized format with headings if the answer is complex

Example citation format:
"AA meetings are held on Mondays at 6:00 PM at the Community Center [Source 1]. The Wednesday meeting is at 7:00 PM [Source 2]."

YOUR DETAILED ANSWER (with citations):"""

        try:
            # Log prompt length for debugging
            logger.info(f"Synthesis prompt length: {len(prompt)} chars")

            # Thinking models need much longer timeout for chain-of-thought reasoning
            # DeepSeek R1 can take 5-10+ minutes for complex reasoning
            request_timeout = 600.0 if is_thinking_model else 180.0

            # Use a larger context window for synthesis with content
            # OPTIMIZATION: keep_alive=5m keeps thinking model in VRAM for 5 min (faster subsequent calls)
            # OPTIMIZATION: top_p=0.95 validated for DeepSeek R1 reasoning quality
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": synthesis_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30m" if is_thinking_model else "5m",  # Keep model loaded (matches OLLAMA_KEEP_ALIVE)
                        "options": {
                            "temperature": model_config.get("temperature", 0.4),
                            "top_p": model_config.get("top_p", 0.95),  # OPTIMIZATION: Validated for R1
                            "num_predict": model_config.get("max_tokens", 2048),
                            "num_ctx": model_config.get("context_window", 32768)
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    synthesis = data.get("response", "")
                    logger.info(f"Ollama synthesis response: {len(synthesis)} chars")

                    # Track context utilization for synthesizer (critical - highest context usage)
                    if request_id and synthesis:
                        metrics = get_performance_metrics()
                        context_window = model_config.get("context_window", 32768)
                        metrics.record_context_utilization(
                            request_id=request_id,
                            agent_name="synthesizer_content",
                            model_name=synthesis_model,
                            input_text=prompt,
                            output_text=synthesis,
                            context_window=context_window
                        )

                    if synthesis:
                        # Add source references at the end matching the [Source X] citations
                        sources_list = "\n".join([
                            f"- **[Source {i+1}]**: [{c.get('title', 'Source')}]({c.get('url', '')})"
                            for i, c in enumerate(valid_sources[:num_sources])
                        ])
                        final_result = f"{synthesis}\n\n**Sources consulted:**\n{sources_list}"
                        logger.info(f"Final synthesis with sources: {len(final_result)} chars")
                        return final_result
                    else:
                        logger.warning("Ollama returned empty synthesis, falling back")
                else:
                    logger.warning(f"Ollama returned status {response.status_code}")

        except Exception as e:
            logger.error(f"Content synthesis failed: {e}")

        # Fallback to regular synthesis
        return await self.synthesize(query, search_results, verifications, context, request_id)

    def determine_confidence_level(
        self,
        verification_results: Optional[List[VerificationResult]],
        source_count: int
    ) -> ConfidenceLevel:
        """Determine overall confidence level"""
        if not verification_results:
            if source_count >= 5:
                return ConfidenceLevel.MEDIUM
            elif source_count >= 2:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.UNKNOWN

        verified_ratio = sum(1 for v in verification_results if v.verified) / len(verification_results)

        if verified_ratio >= 0.8 and source_count >= 3:
            return ConfidenceLevel.HIGH
        elif verified_ratio >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
