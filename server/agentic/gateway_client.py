"""
LLM Gateway Client for memOS Agents

Async client for routing LLM requests through the unified gateway service.
Provides fallback to direct Ollama calls when gateway is unavailable.

Usage:
    from agentic.gateway_client import GatewayClient, get_gateway_client

    # Get singleton client
    client = get_gateway_client()

    # Generate completion
    response = await client.generate(
        prompt="Analyze this text",
        model="analyzer",  # Logical model name
        system="You are an expert analyst"
    )

    # Get embeddings
    embeddings = await client.embed(texts=["text1", "text2"])
"""

import asyncio
import httpx
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class LogicalModel(str, Enum):
    """Logical model names mapped to physical models by gateway."""

    # Synthesis models
    SYNTHESIZER = "synthesizer"
    SYNTHESIZER_FAST = "synthesizer_fast"

    # Analysis models
    ANALYZER = "analyzer"
    CLASSIFIER = "classifier"

    # Reasoning models
    THINKING = "thinking"
    VERIFIER = "verifier"
    REFLECTOR = "reflector"

    # Vision models
    VISION = "vision"

    # Embedding models
    EMBEDDINGS = "embeddings"
    EMBEDDINGS_SEMANTIC = "embeddings_semantic"


@dataclass
class GatewayResponse:
    """Response from gateway generation request."""

    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    fallback_used: bool = False
    thinking: Optional[str] = None
    raw_response: dict = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from gateway embedding request."""

    embeddings: list[list[float]]
    model: str
    dimensions: int = 0
    tokens: int = 0
    latency_ms: float = 0.0


class GatewayClient:
    """
    Async client for LLM Gateway service.

    Features:
    - Automatic fallback to direct Ollama when gateway unavailable
    - Logical model name abstraction
    - Request priority and source identification
    - Circuit breaker pattern for reliability
    - Streaming support
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:8100",
        ollama_url: str = "http://localhost:11434",
        source_system: str = "memos",
        timeout: float = 600.0,
        enable_fallback: bool = True,
        max_retries: int = 2,
    ):
        self.gateway_url = gateway_url
        self.ollama_url = ollama_url
        self.source_system = source_system
        self.timeout = timeout
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries

        # Circuit breaker state
        self._gateway_healthy = True
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # seconds
        self._consecutive_failures = 0
        self._failure_threshold = 3

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Model mapping for direct Ollama fallback
        self._model_mapping = {
            LogicalModel.SYNTHESIZER: "qwen3:8b",
            LogicalModel.SYNTHESIZER_FAST: "qwen3:8b",
            LogicalModel.ANALYZER: "gemma3:4b",
            LogicalModel.CLASSIFIER: "deepseek-r1:14b-qwen-distill-q8_0",
            LogicalModel.THINKING: "deepseek-r1:14b-qwen-distill-q8_0",
            LogicalModel.VERIFIER: "gemma3:4b",
            LogicalModel.REFLECTOR: "gemma3:4b",
            LogicalModel.VISION: "qwen3-vl:7b",
            LogicalModel.EMBEDDINGS: "mxbai-embed-large",
            LogicalModel.EMBEDDINGS_SEMANTIC: "nomic-embed-text",
        }

        # Fallback chains for direct Ollama
        self._fallback_chains = {
            LogicalModel.SYNTHESIZER: ["qwen3:8b", "qwen3:30b-a3b", "llama3.2:3b"],
            LogicalModel.ANALYZER: ["gemma3:4b", "qwen3:8b"],
            LogicalModel.THINKING: ["deepseek-r1:14b-qwen-distill-q8_0", "qwen3:8b"],
            LogicalModel.EMBEDDINGS: ["mxbai-embed-large", "nomic-embed-text"],
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def check_health(self) -> bool:
        """Check if gateway is healthy."""
        now = time.time()

        # Use cached result if recent
        if now - self._last_health_check < self._health_check_interval:
            return self._gateway_healthy

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.gateway_url}/health",
                timeout=5.0
            )
            self._gateway_healthy = response.status_code == 200
            self._consecutive_failures = 0
        except Exception as e:
            logger.warning(f"Gateway health check failed: {e}")
            self._gateway_healthy = False

        self._last_health_check = now
        return self._gateway_healthy

    def _record_failure(self):
        """Record a gateway failure for circuit breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._gateway_healthy = False
            logger.warning(
                f"Gateway circuit breaker opened after {self._consecutive_failures} failures"
            )

    def _record_success(self):
        """Record a gateway success."""
        self._consecutive_failures = 0
        self._gateway_healthy = True

    def _resolve_model(self, model: str | LogicalModel) -> str:
        """Resolve logical model name to physical model for fallback."""
        if isinstance(model, LogicalModel):
            return self._model_mapping.get(model, "qwen3:8b")
        return model

    def _get_fallback_chain(self, model: str | LogicalModel) -> list[str]:
        """Get fallback chain for a model."""
        if isinstance(model, LogicalModel):
            return self._fallback_chains.get(model, [self._resolve_model(model)])
        return [model]

    async def generate(
        self,
        prompt: str,
        model: str | LogicalModel = LogicalModel.SYNTHESIZER,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        priority: str = "normal",
        timeout_ms: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs
    ) -> GatewayResponse:
        """
        Generate completion via gateway.

        Args:
            prompt: The prompt to complete
            model: Logical model name or physical model
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            priority: Request priority (low, normal, high, critical)
            timeout_ms: Request timeout in milliseconds
            options: Additional Ollama options
            **kwargs: Additional arguments

        Returns:
            GatewayResponse with completion and metadata
        """
        start_time = time.time()
        model_str = model.value if isinstance(model, LogicalModel) else model

        # Check gateway health
        gateway_available = await self.check_health()

        if gateway_available:
            try:
                response = await self._generate_via_gateway(
                    prompt=prompt,
                    model=model_str,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    priority=priority,
                    timeout_ms=timeout_ms,
                    options=options,
                    **kwargs
                )
                self._record_success()
                return response
            except Exception as e:
                logger.warning(f"Gateway generate failed: {e}")
                self._record_failure()

        # Fallback to direct Ollama
        if self.enable_fallback:
            logger.info("Falling back to direct Ollama")
            return await self._generate_via_ollama(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
                start_time=start_time,
                **kwargs
            )

        raise RuntimeError("Gateway unavailable and fallback disabled")

    async def _generate_via_gateway(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        priority: str,
        timeout_ms: Optional[int],
        options: Optional[dict],
        **kwargs
    ) -> GatewayResponse:
        """Generate via gateway service."""
        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **(options or {})
            }
        }

        if system:
            payload["system"] = system

        headers = {
            "X-Source-System": self.source_system,
            "X-Request-Priority": priority,
        }

        timeout = (timeout_ms / 1000) if timeout_ms else self.timeout

        response = await client.post(
            f"{self.gateway_url}/api/generate",
            json=payload,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()

        data = response.json()

        return GatewayResponse(
            content=data.get("response", ""),
            model=data.get("model", model),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            latency_ms=data.get("total_duration", 0) / 1_000_000,  # ns to ms
            from_cache=data.get("from_cache", False),
            thinking=data.get("thinking"),
            raw_response=data
        )

    async def _generate_via_ollama(
        self,
        prompt: str,
        model: str | LogicalModel,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        options: Optional[dict],
        start_time: float,
        **kwargs
    ) -> GatewayResponse:
        """Generate directly via Ollama (fallback)."""
        client = await self._get_client()

        fallback_chain = self._get_fallback_chain(model)
        last_error = None

        for physical_model in fallback_chain:
            try:
                payload = {
                    "model": physical_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        **(options or {})
                    }
                }

                if system:
                    payload["system"] = system

                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                return GatewayResponse(
                    content=data.get("response", ""),
                    model=physical_model,
                    tokens_in=data.get("prompt_eval_count", 0),
                    tokens_out=data.get("eval_count", 0),
                    latency_ms=latency_ms,
                    fallback_used=True,
                    raw_response=data
                )

            except Exception as e:
                logger.warning(f"Ollama model {physical_model} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All fallback models failed: {last_error}")

    async def chat(
        self,
        messages: list[dict],
        model: str | LogicalModel = LogicalModel.SYNTHESIZER,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        priority: str = "normal",
        timeout_ms: Optional[int] = None,
        **kwargs
    ) -> GatewayResponse:
        """
        Chat completion via gateway.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            model: Logical model name or physical model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            priority: Request priority
            timeout_ms: Request timeout in milliseconds

        Returns:
            GatewayResponse with completion and metadata
        """
        start_time = time.time()
        model_str = model.value if isinstance(model, LogicalModel) else model

        gateway_available = await self.check_health()

        if gateway_available:
            try:
                response = await self._chat_via_gateway(
                    messages=messages,
                    model=model_str,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    priority=priority,
                    timeout_ms=timeout_ms,
                    **kwargs
                )
                self._record_success()
                return response
            except Exception as e:
                logger.warning(f"Gateway chat failed: {e}")
                self._record_failure()

        if self.enable_fallback:
            return await self._chat_via_ollama(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                start_time=start_time,
                **kwargs
            )

        raise RuntimeError("Gateway unavailable and fallback disabled")

    async def _chat_via_gateway(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        priority: str,
        timeout_ms: Optional[int],
        **kwargs
    ) -> GatewayResponse:
        """Chat via gateway service (OpenAI format)."""
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "X-Source-System": self.source_system,
            "X-Request-Priority": priority,
        }

        timeout = (timeout_ms / 1000) if timeout_ms else self.timeout

        response = await client.post(
            f"{self.gateway_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return GatewayResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=data.get("latency_ms", 0),
            from_cache=data.get("from_cache", False),
            raw_response=data
        )

    async def _chat_via_ollama(
        self,
        messages: list[dict],
        model: str | LogicalModel,
        temperature: float,
        max_tokens: int,
        start_time: float,
        **kwargs
    ) -> GatewayResponse:
        """Chat directly via Ollama (fallback)."""
        client = await self._get_client()

        fallback_chain = self._get_fallback_chain(model)
        last_error = None

        for physical_model in fallback_chain:
            try:
                payload = {
                    "model": physical_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }

                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                message = data.get("message", {})
                latency_ms = (time.time() - start_time) * 1000

                return GatewayResponse(
                    content=message.get("content", ""),
                    model=physical_model,
                    tokens_in=data.get("prompt_eval_count", 0),
                    tokens_out=data.get("eval_count", 0),
                    latency_ms=latency_ms,
                    fallback_used=True,
                    raw_response=data
                )

            except Exception as e:
                logger.warning(f"Ollama model {physical_model} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All fallback models failed: {last_error}")

    async def embed(
        self,
        texts: list[str],
        model: str | LogicalModel = LogicalModel.EMBEDDINGS,
        truncate: bool = True,
        priority: str = "normal",
    ) -> EmbeddingResponse:
        """
        Generate embeddings via gateway.

        Args:
            texts: List of texts to embed
            model: Logical model name or physical model
            truncate: Whether to truncate long texts
            priority: Request priority

        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        start_time = time.time()
        model_str = model.value if isinstance(model, LogicalModel) else model

        gateway_available = await self.check_health()

        if gateway_available:
            try:
                response = await self._embed_via_gateway(
                    texts=texts,
                    model=model_str,
                    truncate=truncate,
                    priority=priority,
                )
                self._record_success()
                return response
            except Exception as e:
                logger.warning(f"Gateway embed failed: {e}")
                self._record_failure()

        if self.enable_fallback:
            return await self._embed_via_ollama(
                texts=texts,
                model=model,
                start_time=start_time,
            )

        raise RuntimeError("Gateway unavailable and fallback disabled")

    async def _embed_via_gateway(
        self,
        texts: list[str],
        model: str,
        truncate: bool,
        priority: str,
    ) -> EmbeddingResponse:
        """Embed via gateway service."""
        client = await self._get_client()

        payload = {
            "model": model,
            "input": texts,
            "truncate": truncate,
        }

        headers = {
            "X-Source-System": self.source_system,
            "X-Request-Priority": priority,
        }

        response = await client.post(
            f"{self.gateway_url}/api/embed",
            json=payload,
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings", [])
        if embeddings:
            dimensions = len(embeddings[0])
        else:
            dimensions = 0

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", model),
            dimensions=dimensions,
            tokens=data.get("total_tokens", 0),
            latency_ms=data.get("latency_ms", 0),
        )

    async def _embed_via_ollama(
        self,
        texts: list[str],
        model: str | LogicalModel,
        start_time: float,
    ) -> EmbeddingResponse:
        """Embed directly via Ollama (fallback)."""
        client = await self._get_client()

        physical_model = self._resolve_model(model)

        # Ollama embed endpoint
        payload = {
            "model": physical_model,
            "input": texts,
        }

        response = await client.post(
            f"{self.ollama_url}/api/embed",
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()

        data = response.json()
        latency_ms = (time.time() - start_time) * 1000

        embeddings = data.get("embeddings", [])
        if embeddings:
            dimensions = len(embeddings[0])
        else:
            dimensions = 0

        return EmbeddingResponse(
            embeddings=embeddings,
            model=physical_model,
            dimensions=dimensions,
            tokens=data.get("prompt_eval_count", 0),
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        prompt: str,
        model: str | LogicalModel = LogicalModel.SYNTHESIZER,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation via gateway.

        Args:
            prompt: The prompt to complete
            model: Logical model name or physical model
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Token strings as they are generated
        """
        model_str = model.value if isinstance(model, LogicalModel) else model
        gateway_available = await self.check_health()

        if gateway_available:
            try:
                async for token in self._stream_via_gateway(
                    prompt=prompt,
                    model=model_str,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield token
                self._record_success()
                return
            except Exception as e:
                logger.warning(f"Gateway stream failed: {e}")
                self._record_failure()

        if self.enable_fallback:
            async for token in self._stream_via_ollama(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield token
            return

        raise RuntimeError("Gateway unavailable and fallback disabled")

    async def _stream_via_gateway(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream via gateway service."""
        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        headers = {
            "X-Source-System": self.source_system,
        }

        async with client.stream(
            "POST",
            f"{self.gateway_url}/api/generate",
            json=payload,
            headers=headers,
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue

    async def _stream_via_ollama(
        self,
        prompt: str,
        model: str | LogicalModel,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream directly via Ollama (fallback)."""
        client = await self._get_client()
        physical_model = self._resolve_model(model)

        payload = {
            "model": physical_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        async with client.stream(
            "POST",
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue


# Singleton instance
_gateway_client: Optional[GatewayClient] = None


def get_gateway_client() -> GatewayClient:
    """Get or create the singleton gateway client."""
    global _gateway_client

    if _gateway_client is None:
        gateway_url = os.environ.get("LLM_GATEWAY_URL", "http://localhost:8100")
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        _gateway_client = GatewayClient(
            gateway_url=gateway_url,
            ollama_url=ollama_url,
            source_system="memos",
            enable_fallback=True,
        )

    return _gateway_client


async def close_gateway_client():
    """Close the singleton gateway client."""
    global _gateway_client

    if _gateway_client is not None:
        await _gateway_client.close()
        _gateway_client = None
