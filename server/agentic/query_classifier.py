"""
Query Classification using DeepSeek-R1 14B Q8 thinking model.

Analyzes user intent and routes to appropriate pipeline.
Uses Chain-of-Draft prompting for efficient classification.

Based on the Implementation Plan for DeepSeek-R1 Query Classification.
"""

import json
import logging
import aiohttp
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from .metrics import get_performance_metrics
from .context_limits import get_model_context_window
from .gateway_client import get_gateway_client, LogicalModel, GatewayResponse

logger = logging.getLogger("agentic.query_classifier")


class QueryCategory(str, Enum):
    """Types of user queries"""
    RESEARCH = "research"           # Information gathering, learning about topics
    PROBLEM_SOLVING = "problem_solving"  # Debugging, troubleshooting, finding solutions
    FACTUAL = "factual"             # Direct questions with verifiable answers
    CREATIVE = "creative"           # Open-ended brainstorming, ideation
    TECHNICAL = "technical"         # Code, engineering, scientific analysis
    COMPARATIVE = "comparative"     # Evaluating options, comparing alternatives
    HOW_TO = "how_to"              # Step-by-step guidance, tutorials


class RecommendedPipeline(str, Enum):
    """Pipeline routing recommendations"""
    DIRECT_ANSWER = "direct_answer"     # Simple LLM response, no search needed
    WEB_SEARCH = "web_search"           # Basic web search + synthesis
    AGENTIC_SEARCH = "agentic_search"   # Full multi-agent pipeline
    CODE_ASSISTANT = "code_assistant"   # Technical/code analysis mode


class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"       # Single-step, straightforward
    MODERATE = "moderate"   # Multi-step, some reasoning
    COMPLEX = "complex"     # Multi-faceted, significant reasoning
    EXPERT = "expert"       # Domain expertise required


@dataclass
class QueryClassification:
    """Result of query classification"""
    category: QueryCategory
    capabilities: List[str]
    complexity: QueryComplexity
    urgency: str  # low, medium, high
    use_thinking_model: bool
    recommended_pipeline: RecommendedPipeline
    reasoning: str
    raw_response: Optional[str] = None


# Classification prompt using Chain-of-Draft for efficiency
QUERY_CLASSIFIER_PROMPT = """You are a query classifier for an AI research and problem-solving assistant.
Analyze the user's query and determine the best approach to answer it.

Think step by step, but only keep a minimum draft for each thinking step.

Query Categories:
- research: Information gathering, learning about topics, exploring concepts
- problem_solving: Debugging, troubleshooting, finding solutions to issues (includes industrial error codes)
- factual: Direct questions with verifiable answers
- creative: Open-ended brainstorming, ideation, creative writing
- technical: Code, engineering, scientific analysis
- comparative: Evaluating options, comparing alternatives
- how_to: Step-by-step guidance, tutorials, procedures

Required Capabilities (select all that apply):
- web_search: Need current/external information
- code_analysis: Involves programming or technical implementation
- data_processing: Requires analyzing data or documents
- reasoning: Complex multi-step logical reasoning
- memory_retrieval: May benefit from prior conversation context

Complexity Indicators:
- simple: Single fact or straightforward answer
- moderate: Requires some explanation or multiple facts
- complex: Multi-faceted question requiring synthesis
- expert: Requires deep domain knowledge (FANUC/PLC/industrial troubleshooting, robotics)

IMPORTANT - Industrial Troubleshooting Rules:
- Error codes (SRVO-xxx, MOTN-xxx, SYST-xxx, INTP-xxx, HOST-xxx) = problem_solving + expert + agentic_search + use_thinking_model=true
- Diagnostic queries (intermittent issues, encoder problems, servo alarms) = problem_solving + expert + agentic_search + use_thinking_model=true
- Procedural queries (mastering, calibration, backup) = how_to + complex + agentic_search
- Robot comparisons and technical evaluations = comparative + expert + agentic_search + use_thinking_model=true

Pipeline Selection:
- direct_answer: For conversational, simple factual, or when answer is in model knowledge
- web_search: For current events, specific facts, or verifiable information
- agentic_search: For complex research, troubleshooting, comparisons, or when multiple sources needed
- code_assistant: For programming, debugging, or technical implementation

User Query: {query}

Context (if available): {context}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"category": "research|problem_solving|factual|creative|technical|comparative|how_to", "capabilities": ["web_search", "reasoning", ...], "complexity": "simple|moderate|complex|expert", "urgency": "low|medium|high", "use_thinking_model": true|false, "recommended_pipeline": "direct_answer|web_search|agentic_search|code_assistant", "reasoning": "Brief explanation of classification"}}"""


# Default model for classification - use a FAST model (not thinking model)
# Classification is simple JSON output, doesn't need reasoning
DEFAULT_CLASSIFIER_MODEL = "qwen3:8b"  # Fast 8B model for classification

# Parameters optimized for fast classification
CLASSIFIER_PARAMS = {
    "temperature": 0.2,      # Low temp for consistent classification
    "top_p": 0.9,            # Focused output
    "num_predict": 256,      # Classification is very concise
}


class QueryClassifier:
    """
    Classifies user queries to determine optimal processing pipeline.

    Uses DeepSeek-R1 14B with Chain-of-Draft prompting for efficient
    classification with reasoning capabilities.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_CLASSIFIER_MODEL
    ):
        self.ollama_url = ollama_url
        self.model = model
        self._available_models: Optional[List[str]] = None

    async def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        if self._available_models is not None:
            return self._available_models

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._available_models = [m["name"] for m in data.get("models", [])]
                        return self._available_models
        except Exception as e:
            logger.warning(f"Failed to get Ollama models: {e}")

        return []

    async def _select_classifier_model(self) -> str:
        """Select the best available model for classification"""
        available = await self._get_available_models()

        # Prioritized list of FAST classification models
        # Classification is a simple task - use fast models, not thinking models
        # IMPORTANT: Avoid embedding models - they can't be used for text generation
        preferred_models = [
            "qwen3:8b",           # Fast 8B model - good balance
            "gemma3:4b",          # Very fast 4B model
            "llama3.2:3b",        # Lightweight fallback
            "qwen3:4b",           # Another fast option
        ]

        # Models to skip (embedding models, vision-only models)
        # Comprehensive list to prevent using non-generative models
        skip_patterns = [
            "embedding", "embed",           # General embedding models
            "bge-", "bge_",                 # BGE embedding models
            "minilm",                       # All-minilm models
            "arctic-embed", "snowflake",    # Snowflake embedding
            "nomic-embed",                  # Nomic embedding
            "functiongemma", "embeddinggemma",  # Gemma embedding variants
            "vision", "-vl",                # Vision-only models
            "mxbai-embed",                  # MixedBread embedding
        ]

        for model in preferred_models:
            # Check for exact match first
            if model in available:
                logger.info(f"Selected classifier model: {model}")
                return model

            # Check for versioned match (e.g., qwen3:8b-q4_0)
            for available_model in available:
                # Skip embedding and vision models
                if any(pattern in available_model.lower() for pattern in skip_patterns):
                    continue

                # Match base name without quantization suffix
                if available_model.startswith(model):
                    logger.info(f"Selected classifier model: {available_model}")
                    return available_model

        # Fallback to default (make sure it's not an embedding model)
        logger.warning(f"No preferred classifier model found, using default: {self.model}")
        return self.model

    async def _classify_via_gateway(
        self,
        prompt: str,
        query: str,
        request_id: str = ""
    ) -> QueryClassification:
        """
        Classify query via LLM Gateway service with automatic fallback.

        Args:
            prompt: The formatted classification prompt
            query: Original user query (for metrics and fallback)
            request_id: Request ID for tracking

        Returns:
            QueryClassification result
        """
        try:
            gateway = get_gateway_client()

            # Use CLASSIFIER logical model
            response: GatewayResponse = await gateway.generate(
                prompt=prompt,
                model=LogicalModel.CLASSIFIER,
                timeout=60.0,
                options={
                    "temperature": CLASSIFIER_PARAMS["temperature"],
                    "top_p": CLASSIFIER_PARAMS["top_p"],
                    "num_predict": CLASSIFIER_PARAMS["num_predict"],
                }
            )

            response_text = response.content

            # Track context utilization
            if request_id and response_text:
                metrics = get_performance_metrics()
                metrics.record_context_utilization(
                    request_id=request_id,
                    agent_name="query_classifier",
                    model_name=response.model,
                    input_text=prompt,
                    output_text=response_text,
                    context_window=get_model_context_window(response.model)
                )

            if response.fallback_used:
                logger.info(f"Gateway classification used fallback to direct Ollama (model: {response.model})")

            # Parse the JSON response
            return self._parse_classification(response_text, query)

        except Exception as e:
            logger.error(f"Gateway classification failed: {e}, falling back to direct Ollama")
            return await self._classify_via_ollama(prompt, query, request_id)

    async def _classify_via_ollama(
        self,
        prompt: str,
        query: str,
        request_id: str = ""
    ) -> QueryClassification:
        """
        Classify query via direct Ollama API call.

        Args:
            prompt: The formatted classification prompt
            query: Original user query (for metrics and fallback)
            request_id: Request ID for tracking

        Returns:
            QueryClassification result
        """
        model = await self._select_classifier_model()
        logger.info(f"Classifying query with {model} via direct Ollama: {query[:50]}...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        **CLASSIFIER_PARAMS
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama API error: {resp.status}")
                        return self._default_classification(query)

                    data = await resp.json()
                    response_text = data.get("response", "")

                    # Track context utilization
                    if request_id:
                        metrics = get_performance_metrics()
                        metrics.record_context_utilization(
                            request_id=request_id,
                            agent_name="query_classifier",
                            model_name=model,
                            input_text=prompt,
                            output_text=response_text,
                            context_window=get_model_context_window(model)
                        )

                    # Parse the JSON response
                    return self._parse_classification(response_text, query)

        except Exception as e:
            logger.error(f"Direct Ollama classification failed: {e}")
            return self._default_classification(query)

    async def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        request_id: str = "",
        use_gateway: bool = False
    ) -> QueryClassification:
        """
        Classify a user query to determine processing approach.

        Args:
            query: The user's query text
            context: Optional context (conversation history, user preferences)
            request_id: Request ID for tracking
            use_gateway: If True, route through LLM Gateway service

        Returns:
            QueryClassification with category, capabilities, and pipeline recommendation
        """
        # Build the classification prompt
        context_str = json.dumps(context) if context else "None"
        prompt = QUERY_CLASSIFIER_PROMPT.format(
            query=query,
            context=context_str
        )

        # Generate request_id if not provided
        if not request_id:
            request_id = f"classify_{hash(query) % 10000}"

        logger.info(f"Classifying query (gateway={use_gateway}): {query[:50]}...")

        try:
            if use_gateway:
                # Route through LLM Gateway for unified routing and VRAM management
                return await self._classify_via_gateway(prompt, query, request_id)
            else:
                # Direct Ollama API call
                return await self._classify_via_ollama(prompt, query, request_id)

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._default_classification(query)

    def _parse_classification(
        self,
        response_text: str,
        query: str
    ) -> QueryClassification:
        """Parse the LLM's classification response"""
        try:
            # Clean the response (remove any markdown or extra text)
            json_str = response_text.strip()

            # Find JSON object in response
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx]

            result = json.loads(json_str)

            classification = QueryClassification(
                category=QueryCategory(result.get("category", "research")),
                capabilities=result.get("capabilities", []),
                complexity=QueryComplexity(result.get("complexity", "moderate")),
                urgency=result.get("urgency", "medium"),
                use_thinking_model=result.get("use_thinking_model", False),
                recommended_pipeline=RecommendedPipeline(
                    result.get("recommended_pipeline", "web_search")
                ),
                reasoning=result.get("reasoning", ""),
                raw_response=response_text
            )

            logger.info(
                f"Classification result: category={classification.category.value}, "
                f"pipeline={classification.recommended_pipeline.value}, "
                f"complexity={classification.complexity.value}"
            )

            return classification

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return self._default_classification(query, response_text)

    def _default_classification(
        self,
        query: str,
        raw_response: Optional[str] = None
    ) -> QueryClassification:
        """Generate a default classification based on query heuristics"""
        query_lower = query.lower()

        # Simple heuristic-based classification
        if any(word in query_lower for word in ["code", "debug", "error", "function", "class", "implement"]):
            category = QueryCategory.TECHNICAL
            pipeline = RecommendedPipeline.CODE_ASSISTANT
            capabilities = ["code_analysis", "reasoning"]
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference", "better"]):
            category = QueryCategory.COMPARATIVE
            pipeline = RecommendedPipeline.WEB_SEARCH
            capabilities = ["web_search", "reasoning"]
        elif any(word in query_lower for word in ["how to", "how do", "tutorial", "guide", "steps"]):
            category = QueryCategory.HOW_TO
            pipeline = RecommendedPipeline.WEB_SEARCH
            capabilities = ["web_search"]
        elif any(word in query_lower for word in ["what is", "define", "explain"]):
            category = QueryCategory.FACTUAL
            pipeline = RecommendedPipeline.WEB_SEARCH
            capabilities = ["web_search"]
        elif any(word in query_lower for word in ["latest", "current", "news", "recent", "today"]):
            category = QueryCategory.RESEARCH
            pipeline = RecommendedPipeline.AGENTIC_SEARCH
            capabilities = ["web_search", "reasoning"]
        else:
            category = QueryCategory.RESEARCH
            pipeline = RecommendedPipeline.WEB_SEARCH
            capabilities = ["web_search"]

        return QueryClassification(
            category=category,
            capabilities=capabilities,
            complexity=QueryComplexity.MODERATE,
            urgency="medium",
            use_thinking_model=False,
            recommended_pipeline=pipeline,
            reasoning="Default classification based on query keywords",
            raw_response=raw_response
        )


# Singleton instance for reuse
_classifier_instance: Optional[QueryClassifier] = None


def get_query_classifier(
    ollama_url: str = "http://localhost:11434"
) -> QueryClassifier:
    """Get or create singleton QueryClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier(ollama_url=ollama_url)
    return _classifier_instance


# Convenience function for quick classification
async def classify_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    ollama_url: str = "http://localhost:11434",
    request_id: str = "",
    use_gateway: bool = False
) -> QueryClassification:
    """
    Quick utility function to classify a query.

    Args:
        query: The user's query
        context: Optional context dictionary
        ollama_url: Ollama API URL
        request_id: Request ID for tracking
        use_gateway: If True, route through LLM Gateway service

    Returns:
        QueryClassification result
    """
    classifier = get_query_classifier(ollama_url)
    return await classifier.classify(query, context, request_id, use_gateway)
