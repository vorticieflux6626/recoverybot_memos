"""
Mixed-Precision Embedding Service for Agentic Search.

Implements precision-stratified embedding retrieval based on:
- Matryoshka Representation Learning (MRL) for flexible dimensions
- ResQ: Mixed-precision quantization with low-rank residuals
- R2Q: Residual refinement quantization
- Binary/Scalar rescoring for efficient retrieval

Key Concepts:
1. Higher-precision embeddings (fp16) serve as "bounding hyperspace"
2. Lower-precision (int8/binary) for fast coarse retrieval
3. Semantic residuals capture what's lost in quantization
4. Three-stage retrieval: binary -> int8 -> fp16 rescoring

Memory Efficiency:
- Binary index: 32x compression (1-bit per dimension)
- Int8 index: 4x compression with 95-99% quality
- FP16 rescore: Full quality for final ranking

References:
- Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
- Saxena et al., "ResQ: Mixed-Precision Quantization" (arXiv 2024)
- "R2Q: Residual Refinement Quantization" (arXiv 2025)
- "Binary and Scalar Embedding Quantization" (HuggingFace 2024)
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Qwen3-Embedding Model Registry
# =============================================================================
# Model specifications for all available qwen3-embedding variants.
# Dimensions vary by model size, not quantization level.

@dataclass
class EmbeddingModelSpec:
    """Specification for an embedding model variant."""
    model_tag: str           # Ollama model tag
    dimensions: int          # Output embedding dimensions
    size_gb: float          # Approximate VRAM/disk size in GB
    quality_tier: str       # "high", "medium", "fast"
    quantization: str       # "fp16", "q8_0", "q4_K_M"
    description: str


# Registry of all qwen3-embedding model variants
QWEN3_EMBEDDING_MODELS: Dict[str, EmbeddingModelSpec] = {
    # 8B models - 4096 dimensions (highest quality)
    "qwen3-embedding:8b-fp16": EmbeddingModelSpec(
        model_tag="qwen3-embedding:8b-fp16",
        dimensions=4096,
        size_gb=15.0,
        quality_tier="high",
        quantization="fp16",
        description="8B full precision - highest quality, slowest"
    ),
    "qwen3-embedding:8b-q8_0": EmbeddingModelSpec(
        model_tag="qwen3-embedding:8b-q8_0",
        dimensions=4096,
        size_gb=8.0,
        quality_tier="high",
        quantization="q8_0",
        description="8B 8-bit quantized - near-fp16 quality, faster"
    ),
    "qwen3-embedding:8b-q4_K_M": EmbeddingModelSpec(
        model_tag="qwen3-embedding:8b-q4_K_M",
        dimensions=4096,
        size_gb=4.7,
        quality_tier="high",
        quantization="q4_K_M",
        description="8B 4-bit quantized - good quality, balanced speed"
    ),

    # 4B models - 2560 dimensions (balanced)
    "qwen3-embedding:4b-fp16": EmbeddingModelSpec(
        model_tag="qwen3-embedding:4b-fp16",
        dimensions=2560,
        size_gb=8.0,
        quality_tier="medium",
        quantization="fp16",
        description="4B full precision - balanced quality/speed"
    ),
    "qwen3-embedding:4b-q8_0": EmbeddingModelSpec(
        model_tag="qwen3-embedding:4b-q8_0",
        dimensions=2560,
        size_gb=4.3,
        quality_tier="medium",
        quantization="q8_0",
        description="4B 8-bit - good quality, faster"
    ),
    "qwen3-embedding:4b-q4_K_M": EmbeddingModelSpec(
        model_tag="qwen3-embedding:4b-q4_K_M",
        dimensions=2560,
        size_gb=2.5,
        quality_tier="medium",
        quantization="q4_K_M",
        description="4B 4-bit - balanced, efficient"
    ),

    # 0.6B models - 1024 dimensions (fast/lightweight)
    "qwen3-embedding:0.6b-fp16": EmbeddingModelSpec(
        model_tag="qwen3-embedding:0.6b-fp16",
        dimensions=1024,
        size_gb=1.2,
        quality_tier="fast",
        quantization="fp16",
        description="0.6B full precision - fastest, lightweight"
    ),
    "qwen3-embedding:0.6b-q8_0": EmbeddingModelSpec(
        model_tag="qwen3-embedding:0.6b-q8_0",
        dimensions=1024,
        size_gb=0.6,
        quality_tier="fast",
        quantization="q8_0",
        description="0.6B 8-bit - ultra-fast, minimal VRAM"
    ),

    # Aliases
    "qwen3-embedding": EmbeddingModelSpec(
        model_tag="qwen3-embedding",
        dimensions=4096,  # Default is 8b-q4_K_M
        size_gb=4.7,
        quality_tier="high",
        quantization="q4_K_M",
        description="Default (8b-q4_K_M)"
    ),
    "qwen3-embedding:latest": EmbeddingModelSpec(
        model_tag="qwen3-embedding:latest",
        dimensions=4096,
        size_gb=4.7,
        quality_tier="high",
        quantization="q4_K_M",
        description="Latest (8b-q4_K_M)"
    ),
}


# Model tiers for automatic selection based on use case
MODEL_TIERS = {
    "high": [
        "qwen3-embedding:8b-fp16",
        "qwen3-embedding:8b-q8_0",
        "qwen3-embedding:8b-q4_K_M",
    ],
    "medium": [
        "qwen3-embedding:4b-fp16",
        "qwen3-embedding:4b-q8_0",
        "qwen3-embedding:4b-q4_K_M",
    ],
    "fast": [
        "qwen3-embedding:0.6b-fp16",
        "qwen3-embedding:0.6b-q8_0",
    ],
}


def get_model_spec(model_tag: str) -> Optional[EmbeddingModelSpec]:
    """Get specification for a model tag."""
    return QWEN3_EMBEDDING_MODELS.get(model_tag)


def get_model_dimension(model_tag: str) -> int:
    """Get expected embedding dimension for a model."""
    spec = get_model_spec(model_tag)
    if spec:
        return spec.dimensions
    # Default fallback based on model name pattern
    if "8b" in model_tag.lower():
        return 4096
    elif "4b" in model_tag.lower():
        return 2560
    elif "0.6b" in model_tag.lower() or "600m" in model_tag.lower():
        return 1024
    return 4096  # Default to 8B dimensions


class PrecisionLevel(str, Enum):
    """Embedding precision levels."""
    BINARY = "binary"    # 1-bit per dimension (32x compression)
    INT4 = "int4"        # 4-bit quantized (8x compression)
    INT8 = "int8"        # 8-bit quantized (4x compression)
    FP16 = "fp16"        # 16-bit float (2x compression from fp32)
    FP32 = "fp32"        # Full precision


@dataclass
class QuantizedEmbedding:
    """An embedding stored at multiple precision levels."""
    doc_id: str
    binary: Optional[bytes] = None      # Packed binary embedding
    int8: Optional[np.ndarray] = None   # Int8 quantized
    fp16: Optional[np.ndarray] = None   # Full precision reference

    # Quantization metadata
    int8_scale: float = 1.0
    int8_zero_point: int = 0
    dimension: int = 4096

    # Semantic residual for precision correction
    residual: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Result from mixed-precision search."""
    doc_id: str
    score: float
    precision_used: PrecisionLevel
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalStats:
    """Statistics from three-stage retrieval."""
    binary_candidates: int = 0
    int8_candidates: int = 0
    final_results: int = 0
    binary_time_ms: float = 0
    int8_time_ms: float = 0
    fp16_time_ms: float = 0
    total_time_ms: float = 0


class MixedPrecisionEmbeddingService:
    """
    Implements precision-stratified embedding retrieval for RAG systems.

    Architecture:
    ```
    User Query
        |
        v
    [Qwen3-Embedding (4096-dim fp16)]
        |
        +---> [Binary Index] ---> Top-500 candidates
        |            |
        |            v
        +---> [Int8 Index] ----> Top-50 rescored
        |            |
        |            v
        +---> [FP16 Store] ----> Top-10 final
                     |
                     v
              Retrieved Documents
    ```
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "qwen3-embedding",
        db_path: Optional[str] = None,
        default_dimension: Optional[int] = None,
        quality_tier: str = "high"  # "high", "medium", "fast"
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.embedding_model = embedding_model
        self.quality_tier = quality_tier

        # Auto-detect dimension from model registry
        if default_dimension is None:
            default_dimension = get_model_dimension(embedding_model)
        self.default_dimension = default_dimension

        # Track detected dimensions per model (for multi-model support)
        self._detected_dimensions: Dict[str, int] = {}

        # Get model spec if available
        self.model_spec = get_model_spec(embedding_model)

        # Database for persistent storage
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "mixed_precision_embeddings.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # In-memory indices for fast access
        self.binary_index: Dict[str, bytes] = {}      # doc_id -> packed binary
        self.int8_index: Dict[str, np.ndarray] = {}   # doc_id -> int8 array
        self.fp16_store: Dict[str, np.ndarray] = {}   # doc_id -> fp16 array

        # Track dimension per document (for mixed-dimension indices)
        self.doc_dimensions: Dict[str, int] = {}      # doc_id -> dimension

        # Quantization metadata
        self.quant_metadata: Dict[str, Dict] = {}     # doc_id -> {scale, zero_point, model}

        # Semantic residuals for precision correction
        self.residuals: Dict[str, np.ndarray] = {}    # doc_id -> residual

        # Anchor embeddings for semantic guidance
        self.anchors: Dict[str, np.ndarray] = {}      # category -> fp16 embedding
        self.anchor_dimensions: Dict[str, int] = {}   # category -> dimension

        # Initialize database
        self._init_db()

        logger.info(
            f"MixedPrecisionEmbeddingService initialized: "
            f"model={embedding_model}, dim={default_dimension}, tier={quality_tier}"
        )

    def _init_db(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id TEXT PRIMARY KEY,
                    binary_data BLOB,
                    int8_data BLOB,
                    fp16_data BLOB,
                    scale REAL DEFAULT 1.0,
                    zero_point INTEGER DEFAULT 0,
                    dimension INTEGER,
                    model_tag TEXT,
                    residual_data BLOB,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anchors (
                    category TEXT PRIMARY KEY,
                    embedding BLOB,
                    dimension INTEGER,
                    model_tag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_created
                ON embeddings(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_dimension
                ON embeddings(dimension)
            """)
            conn.commit()

    async def get_available_models(self) -> List[str]:
        """Get list of available qwen3-embedding models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return [m for m in models if "qwen3-embedding" in m.lower()]
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
        return []

    async def select_model_for_tier(
        self,
        tier: str = "high",
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Select best available model for a quality tier.

        Args:
            tier: Quality tier ("high", "medium", "fast")
            available_models: Optional list of available models

        Returns:
            Model tag to use
        """
        if available_models is None:
            available_models = await self.get_available_models()

        tier_models = MODEL_TIERS.get(tier, MODEL_TIERS["high"])

        for model in tier_models:
            if model in available_models:
                return model

        # Fallback: try any available qwen3-embedding model
        for model in available_models:
            if "qwen3-embedding" in model:
                return model

        # Ultimate fallback
        return self.embedding_model

    def align_dimensions(
        self,
        embedding: np.ndarray,
        target_dim: int,
        method: str = "truncate"
    ) -> np.ndarray:
        """
        Align embedding to target dimension.

        Args:
            embedding: Source embedding
            target_dim: Target dimension
            method: "truncate" (MRL-style), "pad" (zero-pad), or "project"

        Returns:
            Aligned embedding
        """
        current_dim = len(embedding)

        if current_dim == target_dim:
            return embedding

        if current_dim > target_dim:
            # MRL-style truncation (preserve first N dimensions)
            truncated = embedding[:target_dim]
            # Re-normalize
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
            return truncated.astype(np.float16)

        else:  # current_dim < target_dim
            if method == "pad":
                # Zero-pad to target dimension
                padded = np.zeros(target_dim, dtype=np.float16)
                padded[:current_dim] = embedding
                return padded
            else:
                # For project method, would need learned projection matrix
                # Fall back to padding for now
                padded = np.zeros(target_dim, dtype=np.float16)
                padded[:current_dim] = embedding
                return padded

    async def get_embedding(
        self,
        text: str,
        instruction: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        model: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate fp16 embedding using Ollama.

        Args:
            text: Text to embed
            instruction: Optional task instruction (1-5% improvement)
            truncate_dim: Optional MRL truncation (e.g., 64, 256, 1024)
            model: Optional model override (e.g., "qwen3-embedding:4b-q8_0")

        Returns:
            Embedding as fp16 numpy array
        """
        prompt = text
        if instruction:
            prompt = f"Instruct: {instruction}\nQuery: {text}"

        # Use specified model or default
        use_model = model or self.embedding_model

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": use_model,
                        "prompt": prompt
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = np.array(data.get("embedding", []), dtype=np.float16)

                    # Auto-detect and cache dimension for this model
                    actual_dim = len(embedding)
                    if use_model not in self._detected_dimensions:
                        self._detected_dimensions[use_model] = actual_dim
                        logger.info(f"Detected dimension for {use_model}: {actual_dim}")

                    # MRL truncation if requested
                    if truncate_dim and truncate_dim < len(embedding):
                        embedding = embedding[:truncate_dim]
                        # Re-normalize after truncation
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                    return embedding
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    # Use detected dimension if available
                    fallback_dim = self._detected_dimensions.get(
                        use_model,
                        get_model_dimension(use_model)
                    )
                    return np.zeros(fallback_dim, dtype=np.float16)

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            fallback_dim = self._detected_dimensions.get(
                use_model,
                get_model_dimension(use_model)
            )
            return np.zeros(fallback_dim, dtype=np.float16)

    def quantize_to_binary(self, embedding: np.ndarray) -> bytes:
        """
        Convert fp16 embedding to binary (1-bit per dimension).

        Binary quantization: each dimension becomes 1 if > 0, else 0
        Packed into bytes for 32x compression.
        """
        binary_bits = (embedding > 0).astype(np.uint8)
        return np.packbits(binary_bits).tobytes()

    def dequantize_binary(self, binary_data: bytes, dimension: int) -> np.ndarray:
        """Convert binary back to approximate fp16."""
        unpacked = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        # Truncate to original dimension and convert to float
        return (unpacked[:dimension].astype(np.float16) * 2) - 1  # Map 0,1 to -1,+1

    def quantize_to_int8(
        self,
        embedding: np.ndarray
    ) -> Tuple[np.ndarray, float, int]:
        """
        Convert fp16 embedding to int8 with symmetric quantization.

        Returns:
            Tuple of (quantized_array, scale, zero_point)
        """
        # Symmetric quantization around zero
        max_val = np.abs(embedding).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0

        # Quantize
        quantized = np.clip(
            np.round(embedding / scale),
            -128, 127
        ).astype(np.int8)

        return quantized, scale, 0  # zero_point = 0 for symmetric

    def dequantize_int8(
        self,
        int8_data: np.ndarray,
        scale: float,
        zero_point: int = 0
    ) -> np.ndarray:
        """Convert int8 back to fp16."""
        return (int8_data.astype(np.float16) - zero_point) * scale

    def compute_residual(
        self,
        fp16_embedding: np.ndarray,
        int8_embedding: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        Compute semantic residual: what's lost in quantization.

        The residual can be used to correct operations on int8 embeddings.
        """
        dequantized = self.dequantize_int8(int8_embedding, scale)
        return fp16_embedding - dequantized

    async def index_document(
        self,
        doc_id: str,
        text: str,
        content: str = "",
        metadata: Optional[Dict] = None,
        instruction: Optional[str] = None,
        store_residual: bool = True
    ) -> QuantizedEmbedding:
        """
        Index a document at all precision levels.

        Args:
            doc_id: Unique document identifier
            text: Text to embed
            content: Original document content
            metadata: Optional metadata dict
            instruction: Optional embedding instruction
            store_residual: Whether to compute and store residual

        Returns:
            QuantizedEmbedding with all precision levels
        """
        # Generate fp16 embedding
        fp16_emb = await self.get_embedding(text, instruction)
        dimension = len(fp16_emb)

        # Quantize to int8
        int8_emb, scale, zero_point = self.quantize_to_int8(fp16_emb)

        # Quantize to binary
        binary_emb = self.quantize_to_binary(fp16_emb)

        # Compute residual if requested
        residual = None
        if store_residual:
            residual = self.compute_residual(fp16_emb, int8_emb, scale)

        # Store in memory
        self.binary_index[doc_id] = binary_emb
        self.int8_index[doc_id] = int8_emb
        self.fp16_store[doc_id] = fp16_emb
        self.quant_metadata[doc_id] = {"scale": scale, "zero_point": zero_point}
        if residual is not None:
            self.residuals[doc_id] = residual

        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings
                (doc_id, binary_data, int8_data, fp16_data, scale, zero_point,
                 dimension, residual_data, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                binary_emb,
                int8_emb.tobytes(),
                fp16_emb.tobytes(),
                scale,
                zero_point,
                dimension,
                residual.tobytes() if residual is not None else None,
                content,
                json.dumps(metadata or {})
            ))
            conn.commit()

        return QuantizedEmbedding(
            doc_id=doc_id,
            binary=binary_emb,
            int8=int8_emb,
            fp16=fp16_emb,
            int8_scale=scale,
            int8_zero_point=zero_point,
            dimension=dimension,
            residual=residual
        )

    def hamming_distance(self, a: bytes, b: bytes) -> int:
        """Compute Hamming distance between two binary embeddings."""
        # XOR and count bits
        xor_result = bytes(x ^ y for x, y in zip(a, b))
        return sum(bin(byte).count('1') for byte in xor_result)

    def cosine_similarity_int8(
        self,
        query: np.ndarray,
        doc: np.ndarray
    ) -> float:
        """Compute cosine similarity between int8 embeddings."""
        # Convert to float32 for accurate dot product
        q = query.astype(np.float32)
        d = doc.astype(np.float32)

        dot = np.dot(q, d)
        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)

        if norm_q == 0 or norm_d == 0:
            return 0.0
        return dot / (norm_q * norm_d)

    def cosine_similarity_fp16(
        self,
        query: np.ndarray,
        doc: np.ndarray
    ) -> float:
        """Compute cosine similarity between fp16 embeddings."""
        # Use float32 for accuracy
        q = query.astype(np.float32)
        d = doc.astype(np.float32)

        dot = np.dot(q, d)
        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)

        if norm_q == 0 or norm_d == 0:
            return 0.0
        return dot / (norm_q * norm_d)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        instruction: Optional[str] = None,
        binary_candidates: int = 500,
        int8_candidates: int = 50
    ) -> Tuple[List[SearchResult], RetrievalStats]:
        """
        Three-stage precision-stratified search.

        Stage 1: Binary search (ultra-fast, coarse)
        Stage 2: Int8 rescore (fast, medium precision)
        Stage 3: FP16 final rescore (slower, high precision)

        Args:
            query: Search query text
            top_k: Number of final results
            instruction: Optional embedding instruction
            binary_candidates: Number of binary stage candidates
            int8_candidates: Number of int8 stage candidates

        Returns:
            Tuple of (results, stats)
        """
        import time
        stats = RetrievalStats()
        start_time = time.time()

        # Generate query embedding at full precision
        query_fp16 = await self.get_embedding(query, instruction)

        # Stage 1: Binary search
        binary_start = time.time()
        query_binary = self.quantize_to_binary(query_fp16)

        binary_scores = []
        for doc_id, doc_binary in self.binary_index.items():
            # Hamming distance (lower = more similar)
            distance = self.hamming_distance(query_binary, doc_binary)
            # Convert to similarity score (higher = more similar)
            max_distance = len(query_binary) * 8  # bits
            similarity = 1.0 - (distance / max_distance)
            binary_scores.append((doc_id, similarity))

        # Sort by similarity and take top candidates
        binary_scores.sort(key=lambda x: x[1], reverse=True)
        stage1_candidates = [doc_id for doc_id, _ in binary_scores[:binary_candidates]]

        stats.binary_candidates = len(stage1_candidates)
        stats.binary_time_ms = (time.time() - binary_start) * 1000

        # Stage 2: Int8 rescore
        int8_start = time.time()
        query_int8, _, _ = self.quantize_to_int8(query_fp16)

        int8_scores = []
        for doc_id in stage1_candidates:
            if doc_id in self.int8_index:
                doc_int8 = self.int8_index[doc_id]
                similarity = self.cosine_similarity_int8(query_int8, doc_int8)
                int8_scores.append((doc_id, similarity))

        int8_scores.sort(key=lambda x: x[1], reverse=True)
        stage2_candidates = [doc_id for doc_id, _ in int8_scores[:int8_candidates]]

        stats.int8_candidates = len(stage2_candidates)
        stats.int8_time_ms = (time.time() - int8_start) * 1000

        # Stage 3: FP16 final rescore
        fp16_start = time.time()

        fp16_scores = []
        for doc_id in stage2_candidates:
            if doc_id in self.fp16_store:
                doc_fp16 = self.fp16_store[doc_id]
                similarity = self.cosine_similarity_fp16(query_fp16, doc_fp16)
                fp16_scores.append((doc_id, similarity))

        fp16_scores.sort(key=lambda x: x[1], reverse=True)

        stats.fp16_time_ms = (time.time() - fp16_start) * 1000
        stats.total_time_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for doc_id, score in fp16_scores[:top_k]:
            # Fetch content from database
            content = ""
            metadata = {}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT content, metadata FROM embeddings WHERE doc_id = ?",
                    (doc_id,)
                )
                row = cursor.fetchone()
                if row:
                    content = row[0] or ""
                    metadata = json.loads(row[1]) if row[1] else {}

            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                precision_used=PrecisionLevel.FP16,
                content=content,
                metadata=metadata
            ))

        stats.final_results = len(results)

        logger.info(
            f"Three-stage search: binary={stats.binary_candidates} "
            f"({stats.binary_time_ms:.1f}ms) -> int8={stats.int8_candidates} "
            f"({stats.int8_time_ms:.1f}ms) -> fp16={stats.final_results} "
            f"({stats.fp16_time_ms:.1f}ms), total={stats.total_time_ms:.1f}ms"
        )

        return results, stats

    async def mrl_hierarchical_search(
        self,
        query: str,
        top_k: int = 10,
        instruction: Optional[str] = None,
        stages: List[int] = [64, 256, 1024, 4096]
    ) -> Tuple[List[SearchResult], RetrievalStats]:
        """
        Matryoshka-style hierarchical search using dimension truncation.

        Early dimensions contain coarse semantics (fast filtering).
        Later dimensions add fine-grained semantics (precise ranking).

        Args:
            query: Search query
            top_k: Final number of results
            instruction: Optional embedding instruction
            stages: Dimension stages for progressive refinement

        Returns:
            Tuple of (results, stats)
        """
        import time
        stats = RetrievalStats()
        start_time = time.time()

        # Generate full query embedding
        query_full = await self.get_embedding(query, instruction)

        candidates = None
        k_at_stage = [1000, 200, 50, top_k]  # Progressively narrow

        for i, dim in enumerate(stages):
            if dim > len(query_full):
                dim = len(query_full)

            stage_start = time.time()
            query_truncated = query_full[:dim]

            # Normalize after truncation
            norm = np.linalg.norm(query_truncated)
            if norm > 0:
                query_truncated = query_truncated / norm

            if candidates is None:
                # Initial search over all documents
                scores = []
                for doc_id, doc_fp16 in self.fp16_store.items():
                    doc_truncated = doc_fp16[:dim]
                    # Normalize
                    doc_norm = np.linalg.norm(doc_truncated)
                    if doc_norm > 0:
                        doc_truncated = doc_truncated / doc_norm

                    similarity = self.cosine_similarity_fp16(
                        query_truncated, doc_truncated
                    )
                    scores.append((doc_id, similarity))

                scores.sort(key=lambda x: x[1], reverse=True)
                candidates = scores[:k_at_stage[i]]
            else:
                # Rescore existing candidates
                rescored = []
                for doc_id, _ in candidates:
                    if doc_id in self.fp16_store:
                        doc_fp16 = self.fp16_store[doc_id]
                        doc_truncated = doc_fp16[:dim]
                        doc_norm = np.linalg.norm(doc_truncated)
                        if doc_norm > 0:
                            doc_truncated = doc_truncated / doc_norm

                        similarity = self.cosine_similarity_fp16(
                            query_truncated, doc_truncated
                        )
                        rescored.append((doc_id, similarity))

                rescored.sort(key=lambda x: x[1], reverse=True)
                candidates = rescored[:k_at_stage[min(i, len(k_at_stage)-1)]]

            logger.debug(f"MRL stage dim={dim}: {len(candidates)} candidates")

        stats.total_time_ms = (time.time() - start_time) * 1000
        stats.final_results = len(candidates) if candidates else 0

        # Build results
        results = []
        if candidates:
            for doc_id, score in candidates[:top_k]:
                content = ""
                metadata = {}
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT content, metadata FROM embeddings WHERE doc_id = ?",
                        (doc_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        content = row[0] or ""
                        metadata = json.loads(row[1]) if row[1] else {}

                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    precision_used=PrecisionLevel.FP16,
                    content=content,
                    metadata=metadata
                ))

        return results, stats

    def register_anchor(
        self,
        category: str,
        embedding: np.ndarray
    ):
        """
        Register a high-precision anchor embedding for a category.

        Anchors provide semantic reference frames for operations
        on lower-precision embeddings.
        """
        self.anchors[category] = embedding.astype(np.float16)

        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO anchors (category, embedding, dimension)
                VALUES (?, ?, ?)
            """, (category, embedding.tobytes(), len(embedding)))
            conn.commit()

        logger.info(f"Registered anchor for category: {category}")

    async def create_anchor_from_examples(
        self,
        category: str,
        example_texts: List[str],
        instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Create an anchor embedding by averaging example embeddings.

        Args:
            category: Category name (e.g., "addiction_recovery")
            example_texts: List of representative texts
            instruction: Optional embedding instruction

        Returns:
            The anchor embedding
        """
        embeddings = []
        for text in example_texts:
            emb = await self.get_embedding(text, instruction)
            embeddings.append(emb)

        # Average and normalize
        anchor = np.mean(embeddings, axis=0).astype(np.float16)
        norm = np.linalg.norm(anchor)
        if norm > 0:
            anchor = anchor / norm

        self.register_anchor(category, anchor)
        return anchor

    def guided_interpolation(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        alpha: float,
        anchor_category: Optional[str] = None,
        correction_weight: float = 0.3
    ) -> np.ndarray:
        """
        Interpolate between embeddings with optional anchor guidance.

        The anchor provides a "valid semantic region" that the
        interpolation should stay within.

        Args:
            emb_a: First embedding
            emb_b: Second embedding
            alpha: Interpolation factor (0 = a, 1 = b)
            anchor_category: Optional category anchor for guidance
            correction_weight: How much to pull toward anchor

        Returns:
            Interpolated embedding
        """
        # Standard linear interpolation
        interp = (1 - alpha) * emb_a.astype(np.float32) + alpha * emb_b.astype(np.float32)

        if anchor_category and anchor_category in self.anchors:
            anchor = self.anchors[anchor_category].astype(np.float32)

            # Check similarity to anchor
            interp_norm = np.linalg.norm(interp)
            anchor_norm = np.linalg.norm(anchor)

            if interp_norm > 0 and anchor_norm > 0:
                similarity = np.dot(interp, anchor) / (interp_norm * anchor_norm)

                if similarity < 0.7:  # Threshold for semantic validity
                    # Pull toward anchor
                    correction = anchor - interp
                    interp = interp + correction_weight * correction

        # Normalize result
        norm = np.linalg.norm(interp)
        if norm > 0:
            interp = interp / norm

        return interp.astype(np.float16)

    def semantic_arithmetic(
        self,
        base: np.ndarray,
        add: np.ndarray,
        subtract: np.ndarray,
        anchor_category: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform semantic arithmetic: base - subtract + add.

        Example for services:
            base = "homeless shelter"
            add = "addiction recovery"
            subtract = "basic housing"
            → Result closer to "recovery center"

        Args:
            base: Base embedding
            add: Embedding to add
            subtract: Embedding to subtract
            anchor_category: Optional anchor for validation

        Returns:
            Resulting embedding
        """
        # Perform in fp32 for accuracy
        base_fp = base.astype(np.float32)
        add_fp = add.astype(np.float32)
        sub_fp = subtract.astype(np.float32)

        # Compute delta
        delta = add_fp - sub_fp

        # Apply to base
        result = base_fp + delta

        # Validate against anchor if provided
        if anchor_category and anchor_category in self.anchors:
            anchor = self.anchors[anchor_category].astype(np.float32)

            # Compute similarity
            result_norm = np.linalg.norm(result)
            anchor_norm = np.linalg.norm(anchor)

            if result_norm > 0 and anchor_norm > 0:
                similarity = np.dot(result, anchor) / (result_norm * anchor_norm)

                if similarity < 0.5:  # Result drifted too far
                    # Project onto subspace near anchor
                    result = result + 0.5 * (anchor - result)

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result.astype(np.float16)

    def load_from_db(self):
        """Load all embeddings from database into memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doc_id, binary_data, int8_data, fp16_data,
                       scale, zero_point, dimension, residual_data
                FROM embeddings
            """)

            count = 0
            for row in cursor:
                doc_id = row[0]

                if row[1]:  # binary
                    self.binary_index[doc_id] = row[1]

                if row[2]:  # int8
                    self.int8_index[doc_id] = np.frombuffer(row[2], dtype=np.int8)

                if row[3]:  # fp16
                    self.fp16_store[doc_id] = np.frombuffer(row[3], dtype=np.float16)

                self.quant_metadata[doc_id] = {
                    "scale": row[4],
                    "zero_point": row[5]
                }

                if row[7]:  # residual
                    self.residuals[doc_id] = np.frombuffer(row[7], dtype=np.float16)

                count += 1

            # Load anchors
            cursor = conn.execute("SELECT category, embedding, dimension FROM anchors")
            for row in cursor:
                self.anchors[row[0]] = np.frombuffer(row[1], dtype=np.float16)

            logger.info(f"Loaded {count} embeddings and {len(self.anchors)} anchors from database")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        # Estimate memory usage
        binary_memory = sum(len(b) for b in self.binary_index.values())
        int8_memory = sum(a.nbytes for a in self.int8_index.values())
        fp16_memory = sum(a.nbytes for a in self.fp16_store.values())

        # Collect dimension stats
        unique_dims = set(self.doc_dimensions.values()) if self.doc_dimensions else set()

        return {
            "embedding_model": self.embedding_model,
            "default_dimension": self.default_dimension,
            "quality_tier": self.quality_tier,
            "model_spec": {
                "dimensions": self.model_spec.dimensions if self.model_spec else None,
                "size_gb": self.model_spec.size_gb if self.model_spec else None,
                "quantization": self.model_spec.quantization if self.model_spec else None,
            } if self.model_spec else None,
            "detected_dimensions": dict(self._detected_dimensions),
            "indexed_documents": len(self.fp16_store),
            "unique_dimensions_in_index": list(unique_dims),
            "anchors_registered": len(self.anchors),
            "memory_usage": {
                "binary_index_bytes": binary_memory,
                "int8_index_bytes": int8_memory,
                "fp16_store_bytes": fp16_memory,
                "total_bytes": binary_memory + int8_memory + fp16_memory,
                "compression_ratio": (
                    fp16_memory / (binary_memory + 1) if binary_memory > 0 else 0
                )
            },
            "available_model_tiers": MODEL_TIERS,
            "database_path": self.db_path
        }


# Singleton instance
_mixed_precision_service: Optional[MixedPrecisionEmbeddingService] = None


def get_mixed_precision_service(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "qwen3-embedding"
) -> MixedPrecisionEmbeddingService:
    """Get or create singleton MixedPrecisionEmbeddingService."""
    global _mixed_precision_service
    if _mixed_precision_service is None:
        _mixed_precision_service = MixedPrecisionEmbeddingService(
            ollama_url=ollama_url,
            embedding_model=embedding_model
        )
    return _mixed_precision_service
