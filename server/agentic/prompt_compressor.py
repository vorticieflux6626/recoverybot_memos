"""
LLMLingua-2 Style Prompt Compression.

Part of G.5.2: Advanced RAG Techniques - 3-6x context compression.

Based on Microsoft LLMLingua-2 (ACL 2024):
- Token-level compression using small classifier
- Force preservation of domain-specific tokens
- Maintains semantic quality while reducing token count
- Compatible with any downstream LLM

Key Benefits:
- 3-6x context compression ratio
- Preserves critical domain terms
- Reduces LLM inference cost
- Maintains answer quality (within 5% of uncompressed)

Research Basis:
- LLMLingua-2 (Microsoft, ACL 2024)
- Selective Context (arXiv 2023)
- Token pruning literature

Usage:
    from agentic.prompt_compressor import (
        PromptCompressor,
        CompressionConfig,
        get_prompt_compressor
    )

    compressor = get_prompt_compressor()
    result = await compressor.compress(
        context="Long context text...",
        query="User question",
        target_ratio=0.25,  # 4x compression
        force_tokens=["SRVO-063", "FANUC"]
    )
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

logger = logging.getLogger("agentic.prompt_compressor")

# Try to import LLMLingua if available
try:
    from llmlingua import PromptCompressor as LLMLinguaCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    logger.info("LLMLingua not available, using fallback compression")


class CompressionMethod(str, Enum):
    """Compression method to use."""
    LLMLINGUA = "llmlingua"  # Use LLMLingua library
    SENTENCE_IMPORTANCE = "sentence_importance"  # Score and filter sentences
    TOKEN_PROBABILITY = "token_probability"  # Use LLM token probabilities
    EXTRACTIVE = "extractive"  # Extract key sentences
    HYBRID = "hybrid"  # Combine multiple methods


class CompressionLevel(str, Enum):
    """Predefined compression levels."""
    LIGHT = "light"  # 2x compression (0.5 ratio)
    MODERATE = "moderate"  # 3x compression (0.33 ratio)
    AGGRESSIVE = "aggressive"  # 4x compression (0.25 ratio)
    EXTREME = "extreme"  # 6x compression (0.17 ratio)


@dataclass
class CompressionConfig:
    """Configuration for prompt compression."""
    # Compression settings
    method: CompressionMethod = CompressionMethod.SENTENCE_IMPORTANCE
    target_ratio: float = 0.25  # Target compression ratio (0.25 = 4x compression)
    min_length: int = 100  # Minimum output length in characters

    # Token preservation
    force_tokens: List[str] = field(default_factory=list)
    preserve_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Z]{3,5}-\d{3,4}\b',  # Error codes like SRVO-063
        r'\$[A-Z_]+\.\$?[A-Z_]+',  # Parameters like $PARAM_GROUP
        r'\b\d+(?:\.\d+)?(?:mm|cm|m|kg|lb|°|%)\b',  # Measurements
    ])

    # Sentence scoring
    query_weight: float = 0.4  # Weight for query relevance
    position_weight: float = 0.2  # Weight for position (beginning/end preferred)
    length_weight: float = 0.1  # Weight for sentence length
    keyword_weight: float = 0.3  # Weight for keyword density

    # LLM settings (for token probability method)
    ollama_url: str = "http://localhost:11434"
    scoring_model: str = "qwen2.5:0.5b"  # Small model for scoring

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 600


@dataclass
class CompressionResult:
    """Result of prompt compression."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    preserved_tokens: List[str]
    method_used: CompressionMethod
    compression_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original_text),
            "compressed_length": len(self.compressed_text),
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 3),
            "preserved_tokens_count": len(self.preserved_tokens),
            "method": self.method_used.value,
            "time_ms": round(self.compression_time_ms, 2),
        }


class PromptCompressor:
    """
    LLMLingua-2 style prompt compression.

    Reduces context length while preserving semantic meaning and
    critical domain-specific tokens.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
    ):
        """
        Initialize prompt compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()

        # Initialize LLMLingua if available and requested
        self._llmlingua = None
        if LLMLINGUA_AVAILABLE and self.config.method == CompressionMethod.LLMLINGUA:
            try:
                self._llmlingua = LLMLinguaCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
                )
                logger.info("LLMLingua initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLMLingua: {e}")

        # Compile preservation patterns
        self._preserve_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.preserve_patterns
        ]

        # Statistics
        self._total_compressions = 0
        self._total_ratio = 0.0
        self._cache_hits = 0

        # Cache
        self._cache: Dict[str, Tuple[CompressionResult, float]] = {}

        logger.info(f"PromptCompressor initialized with method={self.config.method.value}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return max(1, len(text) // 4)

    def _extract_preserved_tokens(self, text: str) -> Set[str]:
        """Extract tokens that must be preserved."""
        preserved = set()

        # Add force tokens
        for token in self.config.force_tokens:
            if token.lower() in text.lower():
                preserved.add(token)

        # Add pattern matches
        for pattern in self._preserve_patterns:
            matches = pattern.findall(text)
            preserved.update(matches)

        return preserved

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(
        self,
        sentence: str,
        query: Optional[str],
        position: int,
        total_sentences: int,
        preserved_tokens: Set[str]
    ) -> float:
        """Score a sentence for importance."""
        score = 0.0

        # Query relevance
        if query:
            query_terms = set(query.lower().split())
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms & sentence_terms)
            query_score = overlap / max(1, len(query_terms))
            score += self.config.query_weight * query_score

        # Position score (favor beginning and end)
        if total_sentences > 1:
            relative_pos = position / (total_sentences - 1)
            # U-shaped curve: high at start and end
            position_score = 1 - 4 * (relative_pos - 0.5) ** 2
            score += self.config.position_weight * max(0, position_score)

        # Length score (moderate length preferred)
        words = len(sentence.split())
        if 10 <= words <= 30:
            length_score = 1.0
        elif words < 10:
            length_score = words / 10
        else:
            length_score = max(0.3, 1 - (words - 30) / 50)
        score += self.config.length_weight * length_score

        # Keyword score (contains preserved tokens)
        keyword_count = sum(
            1 for token in preserved_tokens
            if token.lower() in sentence.lower()
        )
        keyword_score = min(1.0, keyword_count / max(1, len(preserved_tokens)))
        score += self.config.keyword_weight * keyword_score

        # Boost sentences with numbers, lists, or technical content
        if re.search(r'\d+', sentence):
            score *= 1.1
        if re.search(r'^\s*[-•*]\s*', sentence):
            score *= 1.2
        if re.search(r'step\s*\d|first|second|third|finally', sentence, re.I):
            score *= 1.15

        return min(1.0, score)

    def _compress_sentence_importance(
        self,
        text: str,
        query: Optional[str],
        target_ratio: float,
        preserved_tokens: Set[str]
    ) -> str:
        """Compress using sentence importance scoring."""
        sentences = self._split_sentences(text)

        if len(sentences) <= 2:
            return text  # Too short to compress

        # Score each sentence
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(
                sentence, query, i, len(sentences), preserved_tokens
            )
            scored.append((score, i, sentence))

        # Calculate how many sentences to keep
        target_count = max(1, int(len(sentences) * target_ratio))

        # Always keep sentences with preserved tokens
        must_keep = set()
        for score, i, sentence in scored:
            if any(t.lower() in sentence.lower() for t in preserved_tokens):
                must_keep.add(i)

        # Select top sentences by score
        sorted_by_score = sorted(scored, key=lambda x: x[0], reverse=True)
        selected_indices = must_keep.copy()

        for score, i, sentence in sorted_by_score:
            if len(selected_indices) >= target_count:
                break
            selected_indices.add(i)

        # Ensure first and last sentences are included if room
        if len(selected_indices) < target_count:
            selected_indices.add(0)
        if len(selected_indices) < target_count:
            selected_indices.add(len(sentences) - 1)

        # Reconstruct in original order
        selected = sorted(selected_indices)
        compressed = " ".join(sentences[i] for i in selected)

        return compressed

    def _compress_extractive(
        self,
        text: str,
        query: Optional[str],
        target_ratio: float,
        preserved_tokens: Set[str]
    ) -> str:
        """Compress using extractive summarization approach."""
        sentences = self._split_sentences(text)

        if len(sentences) <= 3:
            return text

        # Simple TF-IDF-like scoring
        word_freq: Dict[str, int] = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Score sentences by word importance
        scored = []
        for i, sentence in enumerate(sentences):
            words = [re.sub(r'[^\w]', '', w.lower()) for w in sentence.split()]
            words = [w for w in words if len(w) > 2]

            if not words:
                score = 0
            else:
                # Sum of inverse document frequency approximation
                score = sum(1 / word_freq.get(w, 1) for w in words) / len(words)

            # Boost for preserved tokens
            for token in preserved_tokens:
                if token.lower() in sentence.lower():
                    score *= 1.5

            scored.append((score, i, sentence))

        # Select top sentences
        target_count = max(2, int(len(sentences) * target_ratio))
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
        selected_indices = sorted([s[1] for s in sorted_scored[:target_count]])

        return " ".join(sentences[i] for i in selected_indices)

    def _compress_hybrid(
        self,
        text: str,
        query: Optional[str],
        target_ratio: float,
        preserved_tokens: Set[str]
    ) -> str:
        """Combine multiple compression methods."""
        # First pass: sentence importance
        intermediate = self._compress_sentence_importance(
            text, query, min(0.6, target_ratio * 1.5), preserved_tokens
        )

        # Second pass: extractive on the result if still too long
        target_len = len(text) * target_ratio
        if len(intermediate) > target_len * 1.2:
            return self._compress_extractive(
                intermediate, query, target_ratio / 0.6, preserved_tokens
            )

        return intermediate

    async def _compress_with_llmlingua(
        self,
        text: str,
        query: Optional[str],
        target_ratio: float,
        force_tokens: List[str]
    ) -> str:
        """Compress using actual LLMLingua library."""
        if not self._llmlingua:
            raise ValueError("LLMLingua not initialized")

        result = self._llmlingua.compress_prompt(
            context=[text],
            instruction=query or "",
            rate=target_ratio,
            force_tokens=force_tokens,
            drop_consecutive=True,
        )

        return result.get("compressed_prompt", text)

    async def compress(
        self,
        context: str,
        query: Optional[str] = None,
        target_ratio: Optional[float] = None,
        force_tokens: Optional[List[str]] = None,
        level: Optional[CompressionLevel] = None
    ) -> CompressionResult:
        """
        Compress a context/prompt.

        Args:
            context: Text to compress
            query: Optional query for relevance scoring
            target_ratio: Target compression ratio (0.25 = 4x compression)
            force_tokens: Additional tokens to preserve
            level: Predefined compression level

        Returns:
            CompressionResult with compressed text and metadata
        """
        start_time = time.perf_counter()

        # Determine target ratio
        if level:
            ratio_map = {
                CompressionLevel.LIGHT: 0.5,
                CompressionLevel.MODERATE: 0.33,
                CompressionLevel.AGGRESSIVE: 0.25,
                CompressionLevel.EXTREME: 0.17,
            }
            target_ratio = ratio_map[level]
        else:
            target_ratio = target_ratio or self.config.target_ratio

        # Check cache
        cache_key = self._compute_cache_key(context, query, target_ratio)
        if self.config.enable_cache and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl_seconds:
                self._cache_hits += 1
                return cached_result

        # Merge force tokens
        all_force_tokens = list(self.config.force_tokens)
        if force_tokens:
            all_force_tokens.extend(force_tokens)

        # Extract preserved tokens
        preserved = self._extract_preserved_tokens(context)
        preserved.update(all_force_tokens)

        original_tokens = self._estimate_tokens(context)

        # Skip compression if text is short
        if len(context) < self.config.min_length * 2:
            result = CompressionResult(
                original_text=context,
                compressed_text=context,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                preserved_tokens=list(preserved),
                method_used=self.config.method,
                compression_time_ms=0,
                metadata={"skipped": "text_too_short"}
            )
            return result

        # Compress based on method
        method = self.config.method

        if method == CompressionMethod.LLMLINGUA and self._llmlingua:
            compressed = await self._compress_with_llmlingua(
                context, query, target_ratio, list(preserved)
            )
        elif method == CompressionMethod.EXTRACTIVE:
            compressed = self._compress_extractive(
                context, query, target_ratio, preserved
            )
        elif method == CompressionMethod.HYBRID:
            compressed = self._compress_hybrid(
                context, query, target_ratio, preserved
            )
        else:  # Default: SENTENCE_IMPORTANCE
            compressed = self._compress_sentence_importance(
                context, query, target_ratio, preserved
            )

        # Ensure minimum length
        if len(compressed) < self.config.min_length and len(context) >= self.config.min_length:
            # Fall back to less aggressive compression
            compressed = self._compress_sentence_importance(
                context, query, min(0.5, target_ratio * 2), preserved
            )

        compressed_tokens = self._estimate_tokens(compressed)
        actual_ratio = compressed_tokens / max(1, original_tokens)

        compression_time = (time.perf_counter() - start_time) * 1000

        result = CompressionResult(
            original_text=context,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=actual_ratio,
            preserved_tokens=list(preserved),
            method_used=method if method != CompressionMethod.LLMLINGUA or self._llmlingua
                        else CompressionMethod.SENTENCE_IMPORTANCE,
            compression_time_ms=compression_time,
            metadata={
                "target_ratio": target_ratio,
                "force_tokens_count": len(all_force_tokens),
            }
        )

        # Update statistics
        self._total_compressions += 1
        self._total_ratio = (
            (self._total_ratio * (self._total_compressions - 1) + actual_ratio)
            / self._total_compressions
        )

        # Cache result
        if self.config.enable_cache:
            self._cache[cache_key] = (result, time.time())

        return result

    async def compress_batch(
        self,
        contexts: List[str],
        query: Optional[str] = None,
        target_ratio: Optional[float] = None
    ) -> List[CompressionResult]:
        """Compress multiple contexts in parallel."""
        tasks = [
            self.compress(ctx, query, target_ratio)
            for ctx in contexts
        ]
        return await asyncio.gather(*tasks)

    def _compute_cache_key(
        self,
        context: str,
        query: Optional[str],
        target_ratio: float
    ) -> str:
        """Compute cache key."""
        content = f"{context}:{query or ''}:{target_ratio}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "total_compressions": self._total_compressions,
            "average_ratio": round(self._total_ratio, 3),
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "llmlingua_available": LLMLINGUA_AVAILABLE,
            "llmlingua_initialized": self._llmlingua is not None,
            "config": {
                "method": self.config.method.value,
                "target_ratio": self.config.target_ratio,
                "force_tokens": len(self.config.force_tokens),
                "preserve_patterns": len(self.config.preserve_patterns),
            }
        }

    def clear_cache(self) -> int:
        """Clear compression cache. Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count


# Preset configurations
LIGHT_COMPRESSION = CompressionConfig(
    target_ratio=0.5,
    method=CompressionMethod.SENTENCE_IMPORTANCE,
)

MODERATE_COMPRESSION = CompressionConfig(
    target_ratio=0.33,
    method=CompressionMethod.HYBRID,
)

AGGRESSIVE_COMPRESSION = CompressionConfig(
    target_ratio=0.25,
    method=CompressionMethod.HYBRID,
)

FANUC_COMPRESSION = CompressionConfig(
    target_ratio=0.25,
    method=CompressionMethod.HYBRID,
    force_tokens=["SRVO", "MOTN", "SYST", "HOST", "FANUC", "robot"],
    preserve_patterns=[
        r'\b[A-Z]{3,5}-\d{3,4}\b',  # Error codes
        r'\$[A-Z_]+',  # Parameters
        r'J[1-9]|axis\s*[1-9]',  # Axis references
        r'\b(?:motor|encoder|servo|amplifier)\b',  # Components
    ],
)


# Global instance
_prompt_compressor: Optional[PromptCompressor] = None


def get_prompt_compressor(
    config: Optional[CompressionConfig] = None
) -> PromptCompressor:
    """Get or create global prompt compressor."""
    global _prompt_compressor
    if _prompt_compressor is None:
        _prompt_compressor = PromptCompressor(config)
    return _prompt_compressor


async def compress_prompt(
    context: str,
    query: Optional[str] = None,
    target_ratio: float = 0.25,
    force_tokens: Optional[List[str]] = None
) -> CompressionResult:
    """Convenience function for prompt compression."""
    return await get_prompt_compressor().compress(
        context, query, target_ratio, force_tokens
    )
