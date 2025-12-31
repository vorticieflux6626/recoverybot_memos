"""
Qdrant On-Disk Storage Configuration.

Part of G.2.4: Configure Qdrant on-disk storage for VRAM management.

Provides Qdrant vector storage with on-disk indexing to reduce VRAM usage.
Supports memory-mapped files and HNSW indexing for efficient retrieval.

Key Features:
- On-disk storage with mmap for VRAM efficiency
- HNSW indexing with configurable parameters
- Quantization support (scalar, binary, product)
- Collection management with automatic schema creation
- Batch indexing and searching

VRAM Management:
- On-disk storage: Vectors stored on SSD, only loaded on-demand
- Mmap mode: Memory-mapped access for large collections
- Scalar quantization: 4x compression with minimal accuracy loss
- Binary quantization: 32x compression for coarse retrieval

Research Basis:
- Qdrant documentation on storage optimization
- HNSW paper (Malkov & Yashunin, 2018)
- Quantization for vector search (various 2023-2024)

Usage:
    from agentic.qdrant_storage import QdrantStorage, StorageConfig

    storage = QdrantStorage(
        config=StorageConfig(
            storage_path="./data/qdrant",
            on_disk=True,
            quantization="scalar"
        )
    )

    await storage.create_collection("documents", dimension=1024)
    await storage.upsert("documents", [{"id": "1", "vector": [...], "payload": {...}}])
    results = await storage.search("documents", query_vector, top_k=10)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("agentic.qdrant_storage")

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        OptimizersConfigDiff,
        HnswConfigDiff,
        QuantizationConfig,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        BinaryQuantization,
        BinaryQuantizationConfig,
        ProductQuantization,
        ProductQuantizationConfig,
        CompressionRatio,
    )
    QDRANT_AVAILABLE = True
    logger.info("Qdrant client available")
except ImportError as e:
    QDRANT_AVAILABLE = False
    logger.warning(f"Qdrant client not available: {e}")
    # Create dummy types for type hints
    QdrantClient = None
    AsyncQdrantClient = None


class QuantizationType(str, Enum):
    """Quantization types for VRAM reduction."""
    NONE = "none"
    SCALAR = "scalar"      # 4x compression, minimal accuracy loss
    BINARY = "binary"      # 32x compression, for coarse retrieval
    PRODUCT = "product"    # Variable compression via PQ


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLID = "euclid"
    DOT = "dot"


@dataclass
class StorageConfig:
    """Configuration for Qdrant storage."""
    # Connection settings
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True

    # Storage settings
    storage_path: Optional[str] = None  # For local mode
    on_disk: bool = True               # Store vectors on disk
    use_mmap: bool = True              # Memory-mapped access

    # HNSW indexing
    hnsw_m: int = 16                   # Number of connections per node
    hnsw_ef_construct: int = 100       # Construction-time ef parameter
    hnsw_full_scan_threshold: int = 10000  # Threshold for full scan

    # Quantization
    quantization: QuantizationType = QuantizationType.SCALAR
    quantization_always_ram: bool = True  # Keep quantized in RAM

    # Performance
    indexing_threshold: int = 20000    # Threshold for indexing
    memmap_threshold: int = 50000      # Threshold for mmap
    max_optimization_threads: int = 2

    # Default collection settings
    default_distance: DistanceMetric = DistanceMetric.COSINE
    default_on_disk_payload: bool = True


@dataclass
class SearchResult:
    """Result from Qdrant search."""
    id: str
    score: float
    vector: Optional[List[float]] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionInfo:
    """Information about a Qdrant collection."""
    name: str
    vectors_count: int
    points_count: int
    indexed_vectors_count: int
    dimension: int
    distance: str
    on_disk: bool
    quantization: Optional[str] = None


class QdrantStorage:
    """
    Qdrant vector storage with on-disk VRAM optimization.

    Supports:
    - Local mode (embedded Qdrant)
    - Server mode (remote Qdrant)
    - On-disk storage with mmap
    - Various quantization methods
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize Qdrant storage.

        Args:
            config: Storage configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")

        self.config = config or StorageConfig()
        self._client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        self._collections: Dict[str, CollectionInfo] = {}

        # Create storage path if using local mode
        if self.config.storage_path:
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"QdrantStorage initialized: on_disk={self.config.on_disk}, "
            f"quantization={self.config.quantization.value}"
        )

    def _get_client(self) -> QdrantClient:
        """Get or create sync client."""
        if self._client is None:
            if self.config.storage_path:
                # Local mode
                self._client = QdrantClient(
                    path=self.config.storage_path,
                    prefer_grpc=self.config.prefer_grpc
                )
            else:
                # Server mode
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc
                )
        return self._client

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create async client."""
        if self._async_client is None:
            if self.config.storage_path:
                self._async_client = AsyncQdrantClient(
                    path=self.config.storage_path,
                    prefer_grpc=self.config.prefer_grpc
                )
            else:
                self._async_client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc
                )
        return self._async_client

    def _get_distance(self, metric: DistanceMetric) -> Distance:
        """Convert distance metric to Qdrant Distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLID: Distance.EUCLID,
            DistanceMetric.DOT: Distance.DOT,
        }
        return mapping.get(metric, Distance.COSINE)

    def _get_quantization_config(self) -> Optional[QuantizationConfig]:
        """Get quantization configuration based on settings."""
        if self.config.quantization == QuantizationType.NONE:
            return None

        if self.config.quantization == QuantizationType.SCALAR:
            return QuantizationConfig(
                scalar=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=self.config.quantization_always_ram
                    )
                )
            )

        if self.config.quantization == QuantizationType.BINARY:
            return QuantizationConfig(
                binary=BinaryQuantization(
                    binary=BinaryQuantizationConfig(
                        always_ram=self.config.quantization_always_ram
                    )
                )
            )

        if self.config.quantization == QuantizationType.PRODUCT:
            return QuantizationConfig(
                product=ProductQuantization(
                    product=ProductQuantizationConfig(
                        compression=CompressionRatio.X16,
                        always_ram=self.config.quantization_always_ram
                    )
                )
            )

        return None

    def _get_optimizer_config(self) -> OptimizersConfigDiff:
        """Get optimizer configuration for on-disk storage."""
        return OptimizersConfigDiff(
            indexing_threshold=self.config.indexing_threshold,
            memmap_threshold=self.config.memmap_threshold,
            max_optimization_threads=self.config.max_optimization_threads
        )

    def _get_hnsw_config(self) -> HnswConfigDiff:
        """Get HNSW configuration."""
        return HnswConfigDiff(
            m=self.config.hnsw_m,
            ef_construct=self.config.hnsw_ef_construct,
            full_scan_threshold=self.config.hnsw_full_scan_threshold,
            on_disk=self.config.on_disk
        )

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: Optional[DistanceMetric] = None,
        recreate: bool = False
    ) -> bool:
        """
        Create a collection with on-disk storage.

        Args:
            name: Collection name
            dimension: Vector dimension
            distance: Distance metric
            recreate: Whether to recreate if exists

        Returns:
            True if created successfully
        """
        client = await self._get_async_client()
        distance = distance or self.config.default_distance

        try:
            # Check if collection exists
            collections = await client.get_collections()
            exists = any(c.name == name for c in collections.collections)

            if exists:
                if recreate:
                    await client.delete_collection(name)
                    logger.info(f"Deleted existing collection: {name}")
                else:
                    logger.info(f"Collection already exists: {name}")
                    return True

            # Create collection with on-disk storage
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=self._get_distance(distance),
                    on_disk=self.config.on_disk
                ),
                hnsw_config=self._get_hnsw_config(),
                optimizers_config=self._get_optimizer_config(),
                quantization_config=self._get_quantization_config(),
                on_disk_payload=self.config.default_on_disk_payload
            )

            logger.info(
                f"Created collection '{name}': dim={dimension}, "
                f"on_disk={self.config.on_disk}, "
                f"quantization={self.config.quantization.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            return False

    async def upsert(
        self,
        collection: str,
        points: List[Dict[str, Any]]
    ) -> int:
        """
        Upsert vectors into collection.

        Args:
            collection: Collection name
            points: List of dicts with 'id', 'vector', and optional 'payload'

        Returns:
            Number of points upserted
        """
        client = await self._get_async_client()

        try:
            qdrant_points = [
                qdrant_models.PointStruct(
                    id=p["id"] if isinstance(p["id"], int) else hash(p["id"]) % (2**63),
                    vector=p["vector"],
                    payload=p.get("payload", {})
                )
                for p in points
            ]

            await client.upsert(
                collection_name=collection,
                points=qdrant_points
            )

            logger.debug(f"Upserted {len(points)} points to '{collection}'")
            return len(points)

        except Exception as e:
            logger.error(f"Failed to upsert to '{collection}': {e}")
            return 0

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            collection: Collection name
            query_vector: Query vector
            top_k: Number of results
            filter: Optional payload filter
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            score_threshold: Minimum score threshold

        Returns:
            List of SearchResult
        """
        client = await self._get_async_client()

        try:
            results = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_models.Filter(**filter) if filter else None,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold
            )

            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    vector=r.vector if with_vectors else None,
                    payload=r.payload if r.payload else {}
                )
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search failed in '{collection}': {e}")
            return []

    async def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        client = await self._get_async_client()

        try:
            info = await client.get_collection(name)
            config = info.config

            return CollectionInfo(
                name=name,
                vectors_count=info.vectors_count or 0,
                points_count=info.points_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0,
                dimension=config.params.vectors.size if hasattr(config.params.vectors, 'size') else 0,
                distance=str(config.params.vectors.distance if hasattr(config.params.vectors, 'distance') else "unknown"),
                on_disk=config.params.vectors.on_disk if hasattr(config.params.vectors, 'on_disk') else False,
                quantization=str(config.quantization_config) if config.quantization_config else None
            )

        except Exception as e:
            logger.error(f"Failed to get collection info for '{name}': {e}")
            return None

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        client = await self._get_async_client()

        try:
            await client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False

    async def count(self, collection: str) -> int:
        """Get point count in collection."""
        info = await self.get_collection_info(collection)
        return info.points_count if info else 0

    async def close(self):
        """Close connections."""
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "qdrant_available": QDRANT_AVAILABLE,
            "config": {
                "on_disk": self.config.on_disk,
                "use_mmap": self.config.use_mmap,
                "quantization": self.config.quantization.value,
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construct": self.config.hnsw_ef_construct,
                "storage_path": self.config.storage_path
            }
        }


# Default on-disk storage configurations
VRAM_EFFICIENT_CONFIG = StorageConfig(
    storage_path="./data/qdrant",
    on_disk=True,
    use_mmap=True,
    quantization=QuantizationType.SCALAR,
    quantization_always_ram=True,
    hnsw_m=16,
    hnsw_ef_construct=100,
    indexing_threshold=10000,
    memmap_threshold=20000
)

MAXIMUM_COMPRESSION_CONFIG = StorageConfig(
    storage_path="./data/qdrant",
    on_disk=True,
    use_mmap=True,
    quantization=QuantizationType.BINARY,
    quantization_always_ram=True,
    hnsw_m=12,  # Lower for binary
    hnsw_ef_construct=64,
    indexing_threshold=5000,
    memmap_threshold=10000
)

BALANCED_CONFIG = StorageConfig(
    storage_path="./data/qdrant",
    on_disk=True,
    use_mmap=True,
    quantization=QuantizationType.SCALAR,
    quantization_always_ram=True,
    hnsw_m=32,  # Higher for better recall
    hnsw_ef_construct=200,
    indexing_threshold=20000,
    memmap_threshold=50000
)


# Global instance
_storage: Optional[QdrantStorage] = None


def get_qdrant_storage(
    config: Optional[StorageConfig] = None
) -> QdrantStorage:
    """Get or create global Qdrant storage instance."""
    global _storage
    if _storage is None:
        _storage = QdrantStorage(config or VRAM_EFFICIENT_CONFIG)
    return _storage


async def get_qdrant_storage_async(
    config: Optional[StorageConfig] = None
) -> QdrantStorage:
    """Async wrapper for get_qdrant_storage."""
    return get_qdrant_storage(config)


def is_qdrant_available() -> bool:
    """Check if Qdrant is available."""
    return QDRANT_AVAILABLE
