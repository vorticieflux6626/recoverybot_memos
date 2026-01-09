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
    reorder_sources_for_synthesis,
    SYNTHESIZER_LIMITS,
    THINKING_SYNTHESIZER_LIMITS,
)
from .metrics import get_performance_metrics
from .gateway_client import get_gateway_client, LogicalModel, GatewayResponse
from .prompt_config import get_prompt_config

# Lazy settings import to avoid circular dependencies
_settings = None
def _get_settings():
    global _settings
    if _settings is None:
        from config.settings import get_settings
        _settings = get_settings()
    return _settings

logger = logging.getLogger("agentic.synthesizer")


def _get_thinking_instruction() -> str:
    """Get thinking model instruction from central config."""
    return get_prompt_config().agent_prompts.synthesizer.thinking_instruction


# Thorough reasoning instruction for thinking models (loaded from config)
# Full reasoning preferred for industrial troubleshooting accuracy over token reduction
THINKING_MODEL_REASONING_INSTRUCTION = property(lambda self: _get_thinking_instruction())

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

# Default thinking model - ministral-3:3b scored 0.848 overall (best speed/quality/VRAM)
# Benchmark: 93.3% analysis coverage, 17s duration, ~3GB VRAM - fits alongside other models
# phi4-reasoning:14b (0.893) requires 11GB which conflicts with PDF Tools vision models
DEFAULT_THINKING_MODEL = "ministral-3:3b"


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
        ollama_url: Optional[str] = None,
        mcp_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        settings = _get_settings()
        self.ollama_url = ollama_url or settings.ollama_base_url
        self.mcp_url = mcp_url or settings.mcp_url
        self.model = model or settings.synthesizer_model
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
        request_id: str = "",
        use_gateway: bool = False,
        model_override: Optional[str] = None
    ) -> str:
        """
        Synthesize search results into a coherent answer.

        Args:
            query: Original user query
            search_results: Web search results
            verifications: Optional verification results
            context: Optional conversation context
            request_id: Request ID for metrics tracking
            use_gateway: If True, route LLM calls through gateway service
            model_override: Optional model to use instead of default (for thinking model selection)

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

        # Build synthesis prompt with mandatory citation requirements
        # Load prompts from central config
        prompt_config = get_prompt_config()
        synth_prompts = prompt_config.agent_prompts.synthesizer
        instructions = prompt_config.instructions

        # Format the main synthesis prompt with variables
        prompt = synth_prompts.main.format(
            query=query,
            results_text=results_text,
            verification_text=verification_text,
            citation_requirement=instructions.citation_requirement,
            cross_domain_constraints=instructions.cross_domain_constraints
        )

        try:
            if use_gateway:
                # Route through LLM Gateway for unified routing and VRAM management
                synthesis = await self._synthesize_via_gateway(prompt, request_id)
            elif self.mcp_available:
                synthesis = await self._synthesize_via_mcp(prompt)
            else:
                synthesis = await self._synthesize_via_ollama(prompt, request_id, model_override=model_override)

            return synthesis or self._fallback_synthesis(query, search_results)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(query, search_results)

    async def _synthesize_via_ollama(
        self,
        prompt: str,
        request_id: str = "",
        model_override: Optional[str] = None
    ) -> str:
        """Execute synthesis via direct Ollama API

        Args:
            prompt: The synthesis prompt
            request_id: Request ID for metrics tracking
            model_override: Optional model to use instead of default (for thinking model selection)
        """
        # Use override model if provided, otherwise use default
        model_to_use = model_override or self.model

        # Get model-specific options if using a thinking model
        options = {"temperature": 0.5, "num_predict": 1024}
        if model_override and model_override in THINKING_MODELS:
            model_config = THINKING_MODELS[model_override]
            options["temperature"] = model_config.get("temperature", 0.6)
            options["num_predict"] = model_config.get("num_predict", 4096)
            options["top_p"] = model_config.get("top_p", 0.95)
            logger.info(f"[{request_id}] Using thinking model config for {model_override}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_to_use,
                        "prompt": prompt,
                        "stream": False,
                        "options": options
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    synthesis = data.get("response", "")

                    # Track context utilization with the actual model used
                    if request_id and synthesis:
                        metrics = get_performance_metrics()
                        metrics.record_context_utilization(
                            request_id=request_id,
                            agent_name="synthesizer",
                            model_name=model_to_use,
                            input_text=prompt,
                            output_text=synthesis,
                            context_window=get_model_context_window(model_to_use)
                        )

                    return synthesis

        except Exception as e:
            logger.error(f"Ollama synthesis failed: {e}")

        return ""

    async def _synthesize_via_gateway(
        self,
        prompt: str,
        request_id: str = "",
        is_thinking_model: bool = False,
        model_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute synthesis via LLM Gateway service with automatic fallback.

        The gateway provides unified routing, priority queuing, and VRAM management.
        Falls back to direct Ollama if gateway is unavailable.

        Args:
            prompt: The synthesis prompt
            request_id: Request ID for metrics tracking
            is_thinking_model: Whether to use a thinking model (DeepSeek R1)
            model_config: Optional model configuration overrides

        Returns:
            Synthesized text response
        """
        try:
            gateway = get_gateway_client()

            # Select logical model based on context
            if is_thinking_model:
                logical_model = LogicalModel.THINKING
            else:
                logical_model = LogicalModel.SYNTHESIZER

            # Build generation options
            options = {}
            if model_config:
                options["temperature"] = model_config.get("temperature", 0.4)
                options["top_p"] = model_config.get("top_p", 0.95)
                options["num_predict"] = model_config.get("max_tokens", 2048)
                options["num_ctx"] = model_config.get("context_window", 32768)
            else:
                options = {
                    "temperature": 0.5,
                    "num_predict": 1024
                }

            # Request timeout depends on model type
            timeout = 600.0 if is_thinking_model else 180.0

            # Call gateway
            response: GatewayResponse = await gateway.generate(
                prompt=prompt,
                model=logical_model,
                timeout=timeout,
                options=options
            )

            synthesis = response.content

            # Track context utilization if we have metrics
            if request_id and synthesis:
                metrics = get_performance_metrics()
                context_window = options.get("num_ctx", 32768) if model_config else get_model_context_window(self.model)
                agent_name = "synthesizer_gateway_thinking" if is_thinking_model else "synthesizer_gateway"
                metrics.record_context_utilization(
                    request_id=request_id,
                    agent_name=agent_name,
                    model_name=response.model,
                    input_text=prompt,
                    output_text=synthesis,
                    context_window=context_window
                )

            # Log if fallback was used
            if response.fallback_used:
                logger.info(f"Gateway synthesis used fallback to direct Ollama (model: {response.model})")

            return synthesis

        except Exception as e:
            logger.error(f"Gateway synthesis failed: {e}, falling back to direct Ollama")
            # Fallback to direct Ollama on gateway failure
            return await self._synthesize_via_ollama(prompt, request_id)

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
        """Format search results for the synthesis prompt - use relevant results for accurate synthesis.

        Applies lost-in-middle mitigation: reorders sources so most relevant are at
        the start and end of the list, where LLMs attend best.
        """
        # Apply lost-in-middle mitigation: reorder for optimal LLM attention
        # Most relevant at start, 2nd most at end, alternating
        reordered = reorder_sources_for_synthesis(results[:15])

        formatted = []
        for i, result in enumerate(reordered, 1):
            formatted.append(
                f"[Source {i}] **{result.title}**\n"
                f"URL: {result.source_domain}\n"
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

        # Basic compilation of results - use available relevant results
        synthesis = f"Here's what I found regarding your question: \"{query}\"\n\n"

        for i, result in enumerate(results[:10], 1):
            synthesis += f"**[Source {i}] {result.title}**\n{result.snippet}\n\n"

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
        request_id: str = "",
        use_gateway: bool = False
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
            request_id: Request ID for metrics tracking
            use_gateway: If True, route LLM calls through gateway service

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
            return await self.synthesize(query, search_results, verifications, context, request_id, use_gateway)

        # Format scraped content for the prompt
        # Use relevant sources that have meaningful content, prioritize by content length
        valid_sources = [
            s for s in scraped_content
            if s.get("content") and len(s.get("content", "")) > 100
        ]
        # Sort by content length descending (best sources first)
        valid_sources.sort(key=lambda x: len(x.get("content", "")), reverse=True)

        # Apply lost-in-middle mitigation: reorder so best sources are at start and end
        # This improves LLM attention to the most important content
        valid_sources = reorder_sources_for_synthesis(
            valid_sources,
            score_attr="relevance_score"  # If available, otherwise falls back to order
        )

        content_sections = []
        total_chars = 0

        # Get dynamic limits based on model context window
        limits = THINKING_SYNTHESIZER_LIMITS if is_thinking_model else SYNTHESIZER_LIMITS
        max_total_chars = limits["max_total_content"]
        max_sources = limits["max_urls_to_scrape"]
        per_source_limit_base = limits["max_content_per_source"]

        # FIX 2: Cap sources when domain knowledge exists to improve signal-to-noise ratio
        # When HSEA provides authoritative domain knowledge, limit web sources to supplementary role
        domain_knowledge_chars = len(context.get("additional_context", "")) if context else 0
        if domain_knowledge_chars > 100:  # Has meaningful domain knowledge
            # Reduce sources to improve signal-to-noise ratio
            # Domain knowledge should be primary; web sources are supplementary
            original_max_sources = max_sources
            max_sources = min(max_sources, 10)  # Cap at 10 sources when domain knowledge present
            max_total_chars = min(max_total_chars, 80000)  # Cap total web content
            logger.info(f"Domain knowledge detected ({domain_knowledge_chars} chars) - capping sources: {original_max_sources} → {max_sources}")

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
                f"=== [Source {i}]: {title} ===\n"
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
        # Thorough reasoning instruction added for thinking models to improve accuracy
        reasoning_prefix = _get_thinking_instruction() + "\n\n" if is_thinking_model else ""

        # Extract domain knowledge from context if provided (e.g., HSEA FANUC error codes)
        # Domain knowledge is AUTHORITATIVE and takes PRIORITY over web sources
        domain_knowledge_text = ""
        if context and context.get("additional_context"):
            domain_ctx = context["additional_context"]
            if domain_ctx.strip():
                domain_knowledge_text = f"""
⚠️ AUTHORITATIVE DOMAIN KNOWLEDGE - USE THIS AS YOUR PRIMARY SOURCE ⚠️
The following information comes from official technical documentation and MUST be used as the authoritative source for your answer. If web sources conflict with this information, prefer the domain knowledge.

{domain_ctx}

CRITICAL: For any error codes mentioned above (e.g., SRVO-xxx, MOTN-xxx), you MUST use the exact cause and remedy from this domain knowledge. Do NOT substitute with information about different error codes from your training data.
---
"""
                logger.info(f"Synthesizer including domain knowledge: {len(domain_ctx)} chars")

        # FIX 1: Recency Bias - Domain knowledge now placed AFTER sources for better attention
        # LLMs pay more attention to content near the end of prompts (recency bias)
        # Structure: Query → Sources → Domain Knowledge → Final Instructions

        # Build domain knowledge section marker for citations if present
        has_domain_knowledge = bool(domain_knowledge_text.strip())
        citation_instruction = """CITATION REQUIREMENTS:
- Every factual claim MUST be cited with [Source X] notation
- For general research: Use [Source 1], [Source 2], etc. for all facts""" if not has_domain_knowledge else """CITATION HIERARCHY (CRITICAL):
1. **[Domain Knowledge]** - Use this FIRST for error codes, causes, remedies (authoritative official documentation)
2. **[Source X]** - Use for supplementary context, user experiences, troubleshooting tips
3. If web sources conflict with Domain Knowledge, the Domain Knowledge is CORRECT"""

        prompt = f"""{reasoning_prefix}<role>EXPERT RESEARCH SYNTHESIZER for industrial automation</role>
<expertise>Combine information from multiple sources into accurate, actionable answers for FANUC robotics, Allen-Bradley PLCs, Siemens automation, servo systems, and industrial troubleshooting. Every claim MUST be cited. Use technical terminology correctly.</expertise>

**FOCUS REQUIREMENT**: Your answer MUST directly address the specific topic in the USER'S QUESTION below. Do NOT answer about tangentially related topics that appear in search results. Stay focused on exactly what was asked.

USER'S QUESTION: {query}

I have provided {num_sources} web sources for you to analyze:

{full_content}
{verification_text}
{domain_knowledge_text}
{citation_instruction}

CRITICAL REQUIREMENTS:
1. **STAY ON TOPIC**: Answer ONLY about the specific error code, component, or procedure mentioned in the USER'S QUESTION. Ignore unrelated content in search results.
2. **MANDATORY CITATIONS**: Every factual claim MUST have a citation. Answers without citations are INCOMPLETE.
3. **TERM COVERAGE**: Use the key technical terms from the question in your answer (e.g., error codes, component names, procedures, part numbers).
4. Read all source content carefully and extract ONLY information that directly answers the question.
5. Be specific - include part numbers, error codes, parameter names, and other technical details.
6. If sources disagree, note the discrepancy: "Source 1 says X [Source 1], but Source 2 says Y [Source 2]."
7. If the sources don't contain enough information, clearly state what is known and what remains unclear.

CITATION FORMAT EXAMPLES:
- "SRVO-023 indicates 'Stop error excess' - the servo positional error exceeded a specified value [Domain Knowledge]."
- "A user on the forum reported success by inspecting the external axis coupling [Source 3]."
- "The R-30iB Plus controller manual recommends verifying encoder connections [Source 1]."

WARNING: Responses about a DIFFERENT topic than asked will be rejected. Responses without citations are INCOMPLETE.

YOUR DETAILED ANSWER (with citations):"""

        try:
            # Log prompt length for debugging
            logger.info(f"Synthesis prompt length: {len(prompt)} chars")

            # Thinking models need much longer timeout for chain-of-thought reasoning
            # DeepSeek R1 can take 5-10+ minutes for complex reasoning
            request_timeout = 600.0 if is_thinking_model else 180.0

            synthesis = ""

            if use_gateway:
                # Route through LLM Gateway for unified routing and VRAM management
                logger.info("Using LLM Gateway for synthesis_with_content")
                synthesis = await self._synthesize_via_gateway(
                    prompt=prompt,
                    request_id=request_id,
                    is_thinking_model=is_thinking_model,
                    model_config=model_config
                )
                logger.info(f"Gateway synthesis response: {len(synthesis)} chars")
            else:
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
                    else:
                        logger.warning(f"Ollama returned status {response.status_code}")

            # Add source references if synthesis was successful
            if synthesis:
                sources_list = "\n".join([
                    f"- **[Source {i+1}]**: [{c.get('title', 'Source')}]({c.get('url', '')})"
                    for i, c in enumerate(valid_sources[:num_sources])
                ])
                final_result = f"{synthesis}\n\n**Sources consulted:**\n{sources_list}"
                logger.info(f"Final synthesis with sources: {len(final_result)} chars")
                return final_result
            else:
                logger.warning("Synthesis returned empty, falling back")

        except Exception as e:
            logger.error(f"Content synthesis failed: {e}")

        # Fallback to regular synthesis
        return await self.synthesize(query, search_results, verifications, context, request_id, use_gateway)

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
