"""
HyDE (Hypothetical Document Embeddings) for Query Expansion.

HyDE improves retrieval by generating a hypothetical document that would
answer the query, then using that document's embedding for search.

Key Insight:
- Queries are short and abstract
- Documents are long and detailed
- Query embeddings don't match document embeddings well
- Hypothetical documents bridge this gap

Pipeline:
1. Query → LLM generates hypothetical answer document
2. Hypothetical document → Embedding
3. Embedding → Vector search against corpus
4. Return real documents matching hypothetical

Research:
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023)
- arXiv:2212.10496

Benefits:
- 10-20% improvement in recall@10 on benchmarks
- Works without domain-specific training
- Composable with existing embedding systems
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import httpx

from .llm_config import get_llm_config
from .prompt_config import get_prompt_config

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Models
# =============================================================================

class HyDEMode(Enum):
    """HyDE generation modes."""
    SINGLE = "single"           # Generate one hypothetical document
    MULTI = "multi"             # Generate multiple hypothetical documents
    CONTRASTIVE = "contrastive" # Generate positive + negative examples


class DocumentType(Enum):
    """Types of hypothetical documents to generate."""
    ANSWER = "answer"           # Direct answer to query
    PASSAGE = "passage"         # Relevant passage from hypothetical source
    EXPLANATION = "explanation" # Detailed explanation
    SUMMARY = "summary"         # Summary of relevant information
    TECHNICAL = "technical"     # Technical documentation style


@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""
    mode: HyDEMode = HyDEMode.SINGLE
    document_type: DocumentType = DocumentType.PASSAGE
    num_hypotheticals: int = 1      # For MULTI mode
    temperature: float = 0.7        # Generation temperature
    max_length: int = 300           # Max hypothetical document length
    chain_of_draft: bool = True     # Use Chain-of-Draft prompting
    embedding_model: str = "bge-m3" # Embedding model


@dataclass
class HyDEResult:
    """Result of HyDE expansion."""
    original_query: str
    hypothetical_documents: List[str]
    hypothetical_embeddings: List[np.ndarray]
    fused_embedding: np.ndarray
    generation_time_ms: float
    embedding_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "hypothetical_documents": self.hypothetical_documents,
            "num_hypotheticals": len(self.hypothetical_documents),
            "fused_embedding_norm": float(np.linalg.norm(self.fused_embedding)),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "embedding_time_ms": round(self.embedding_time_ms, 2),
            "metadata": self.metadata
        }


# =============================================================================
# HyDE Prompts (loaded from central config)
# =============================================================================

def _get_hyde_prompts() -> Dict[DocumentType, str]:
    """Get HyDE prompts from central config."""
    config = get_prompt_config()
    hyde = config.agent_prompts.hyde
    return {
        DocumentType.ANSWER: hyde.answer,
        DocumentType.PASSAGE: hyde.passage,
        DocumentType.EXPLANATION: hyde.explanation,
        DocumentType.SUMMARY: hyde.explanation,  # Reuse explanation for summary
        DocumentType.TECHNICAL: hyde.technical,
    }


def _get_multi_hyde_prompt() -> str:
    """Get multi-HyDE prompt from central config."""
    return get_prompt_config().agent_prompts.hyde.multi


def _get_hyde_chain_of_draft() -> str:
    """Get chain-of-draft instruction from central config."""
    return get_prompt_config().instructions.chain_of_draft


# Backward-compatible module-level access
HYDE_PROMPTS = property(lambda self: _get_hyde_prompts())
MULTI_HYDE_PROMPT = property(lambda self: _get_multi_hyde_prompt())
CHAIN_OF_DRAFT_INSTRUCTION = property(lambda self: _get_hyde_chain_of_draft())


# =============================================================================
# HyDE Expander
# =============================================================================

class HyDEExpander:
    """
    Hypothetical Document Embeddings expander.

    Generates hypothetical documents for queries and creates fused embeddings
    that better match real document embeddings.
    """

    def __init__(
        self,
        ollama_url: str = None,
        generation_model: str = None,
        embedding_model: str = "bge-m3",
        config: Optional[HyDEConfig] = None
    ):
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.generation_model = generation_model or llm_config.utility.hyde_generator.model
        self.embedding_model = embedding_model
        self.config = config or HyDEConfig()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Cache for hypothetical documents
        self._cache: Dict[str, HyDEResult] = {}

        # Embedding dimension (detected on first use)
        self._embedding_dim: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def generate_hypothetical(
        self,
        query: str,
        document_type: Optional[DocumentType] = None
    ) -> str:
        """
        Generate a single hypothetical document for a query.

        Args:
            query: The search query
            document_type: Type of hypothetical to generate

        Returns:
            Generated hypothetical document text
        """
        doc_type = document_type or self.config.document_type
        hyde_prompts = _get_hyde_prompts()
        prompt_template = hyde_prompts.get(doc_type, hyde_prompts[DocumentType.PASSAGE])
        prompt = prompt_template.format(query=query)

        if self.config.chain_of_draft:
            prompt = _get_hyde_chain_of_draft() + "\n\n" + prompt

        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.generation_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_length
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except Exception as e:
            logger.error(f"HyDE generation error: {e}")
            # Fallback: expand query with keywords
            return f"Document about: {query}. This document contains detailed information related to {query}."

    async def generate_multi_hypotheticals(
        self,
        query: str,
        n: int = 3
    ) -> List[str]:
        """
        Generate multiple hypothetical documents for diverse coverage.

        Args:
            query: The search query
            n: Number of hypotheticals to generate

        Returns:
            List of hypothetical documents
        """
        prompt = _get_multi_hyde_prompt().format(query=query, n=n)

        if self.config.chain_of_draft:
            prompt = _get_hyde_chain_of_draft() + "\n\n" + prompt

        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.generation_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_length * n
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "")

            # Parse numbered passages
            passages = []
            for i in range(1, n + 1):
                # Find text after "1.", "2.", etc.
                import re
                pattern = f"{i}[.):]\\s*"
                parts = re.split(pattern, text)
                if len(parts) > 1:
                    # Get text until next number or end
                    passage = parts[1].split(f"{i+1}")[0].strip()
                    if passage:
                        passages.append(passage)

            # If parsing failed, split by newlines
            if len(passages) < n:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                passages = paragraphs[:n]

            # Ensure we have at least one
            if not passages:
                passages = [text.strip()]

            return passages

        except Exception as e:
            logger.error(f"Multi-HyDE generation error: {e}")
            return [await self.generate_hypothetical(query)]

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using configured embedding model."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)

            if self._embedding_dim is None:
                self._embedding_dim = len(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            dim = self._embedding_dim or 1024
            return np.zeros(dim, dtype=np.float32)

    def fuse_embeddings(
        self,
        embeddings: List[np.ndarray],
        method: str = "mean"
    ) -> np.ndarray:
        """
        Fuse multiple embeddings into one.

        Methods:
        - mean: Average of all embeddings (default for HyDE)
        - max: Element-wise maximum (captures diverse features)
        - weighted: Weighted average (first embedding has more weight)
        """
        if not embeddings:
            dim = self._embedding_dim or 1024
            return np.zeros(dim, dtype=np.float32)

        if len(embeddings) == 1:
            return embeddings[0]

        stacked = np.stack(embeddings)

        if method == "mean":
            fused = np.mean(stacked, axis=0)
        elif method == "max":
            fused = np.max(stacked, axis=0)
        elif method == "weighted":
            # First embedding (most relevant) gets more weight
            weights = np.array([1 / (i + 1) for i in range(len(embeddings))])
            weights = weights / weights.sum()
            fused = np.average(stacked, axis=0, weights=weights)
        else:
            fused = np.mean(stacked, axis=0)

        # Normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused

    async def expand(
        self,
        query: str,
        mode: Optional[HyDEMode] = None,
        num_hypotheticals: int = 1,
        include_query_embedding: bool = True,
        cache: bool = True
    ) -> HyDEResult:
        """
        Expand a query using HyDE.

        Args:
            query: The search query
            mode: HyDE mode (SINGLE or MULTI)
            num_hypotheticals: Number of hypothetical documents (for MULTI mode)
            include_query_embedding: Also include original query embedding in fusion
            cache: Use caching for repeated queries

        Returns:
            HyDEResult with hypothetical documents and fused embedding
        """
        # Check cache
        cache_key = hashlib.md5(
            f"{query}:{mode}:{num_hypotheticals}:{include_query_embedding}".encode()
        ).hexdigest()

        if cache and cache_key in self._cache:
            logger.debug(f"HyDE cache hit for query: {query[:50]}...")
            return self._cache[cache_key]

        mode = mode or self.config.mode
        n = num_hypotheticals if mode == HyDEMode.MULTI else 1

        # Generate hypothetical documents
        gen_start = time.time()

        if n == 1:
            hypotheticals = [await self.generate_hypothetical(query)]
        else:
            hypotheticals = await self.generate_multi_hypotheticals(query, n)

        gen_time = (time.time() - gen_start) * 1000

        # Get embeddings for all hypotheticals
        emb_start = time.time()

        embeddings = []
        for hypo in hypotheticals:
            emb = await self.get_embedding(hypo)
            embeddings.append(emb)

        # Optionally include original query embedding
        if include_query_embedding:
            query_emb = await self.get_embedding(query)
            embeddings.insert(0, query_emb)

        emb_time = (time.time() - emb_start) * 1000

        # Fuse embeddings
        fused = self.fuse_embeddings(embeddings)

        result = HyDEResult(
            original_query=query,
            hypothetical_documents=hypotheticals,
            hypothetical_embeddings=embeddings,
            fused_embedding=fused,
            generation_time_ms=gen_time,
            embedding_time_ms=emb_time,
            metadata={
                "mode": mode.value,
                "num_hypotheticals": len(hypotheticals),
                "include_query": include_query_embedding,
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model
            }
        )

        # Cache result
        if cache:
            self._cache[cache_key] = result

        logger.info(
            f"HyDE expansion: {len(hypotheticals)} docs, "
            f"gen={gen_time:.0f}ms, emb={emb_time:.0f}ms"
        )

        return result

    async def search_with_hyde(
        self,
        query: str,
        corpus_embeddings: Dict[str, np.ndarray],
        corpus_contents: Dict[str, str],
        top_k: int = 10,
        mode: HyDEMode = HyDEMode.SINGLE,
        num_hypotheticals: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Perform search using HyDE-expanded query.

        Args:
            query: Search query
            corpus_embeddings: Dict of doc_id -> embedding
            corpus_contents: Dict of doc_id -> content
            top_k: Number of results
            mode: HyDE mode
            num_hypotheticals: Number of hypotheticals for MULTI mode

        Returns:
            List of search results with scores
        """
        # Expand query with HyDE
        hyde_result = await self.expand(
            query=query,
            mode=mode,
            num_hypotheticals=num_hypotheticals
        )

        # Search using fused embedding
        query_emb = hyde_result.fused_embedding
        query_norm = np.linalg.norm(query_emb)

        scores = []
        for doc_id, doc_emb in corpus_embeddings.items():
            doc_norm = np.linalg.norm(doc_emb)
            if query_norm > 0 and doc_norm > 0:
                sim = float(np.dot(query_emb, doc_emb) / (query_norm * doc_norm))
            else:
                sim = 0.0
            scores.append({
                "doc_id": doc_id,
                "content": corpus_contents.get(doc_id, "")[:500],
                "score": sim,
                "hyde_metadata": hyde_result.to_dict()
            })

        # Sort by score
        scores.sort(key=lambda x: x["score"], reverse=True)

        return scores[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get expander statistics."""
        return {
            "cache_size": len(self._cache),
            "embedding_dim": self._embedding_dim,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "config": {
                "mode": self.config.mode.value,
                "document_type": self.config.document_type.value,
                "temperature": self.config.temperature,
                "max_length": self.config.max_length
            }
        }

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Singleton and Factory
# =============================================================================

_hyde_expander: Optional[HyDEExpander] = None


def get_hyde_expander(
    ollama_url: str = None,
    generation_model: str = None,
    embedding_model: str = "bge-m3"
) -> HyDEExpander:
    """Get or create the global HyDE expander instance (config from llm_models.yaml)."""
    global _hyde_expander

    if _hyde_expander is None:
        _hyde_expander = HyDEExpander(
            ollama_url=ollama_url,
            generation_model=generation_model,
            embedding_model=embedding_model
        )

    return _hyde_expander


async def create_hyde_expander(
    ollama_url: str = None,
    generation_model: str = None,
    embedding_model: str = "bge-m3",
    config: Optional[HyDEConfig] = None
) -> HyDEExpander:
    """Create a new HyDE expander instance (config from llm_models.yaml)."""
    return HyDEExpander(
        ollama_url=ollama_url,
        generation_model=generation_model,
        embedding_model=embedding_model,
        config=config
    )
