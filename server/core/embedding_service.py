"""
Embedding Service for memOS Server
Handles text embeddings using Ollama for semantic search
"""

import asyncio
import logging
import httpx
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Text embedding service using Ollama
    Provides semantic embeddings for memory search and similarity
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(
            base_url=self.settings.ollama_base_url,
            timeout=30.0
        )
        self.embedding_model = self.settings.ollama_embedding_model
        self.embedding_dimensions = self.settings.embedding_dimensions
        
        # LRU cache for frequently used embeddings (OrderedDict maintains access order)
        self._embedding_cache: OrderedDict[int, List[float]] = OrderedDict()
        self._cache_max_size = 1000
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Ollama.
        Includes retry logic for transient 500 errors.
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dimensions

        # Check cache first (LRU: move to end on access)
        text_hash = hash(text.strip().lower())
        if text_hash in self._embedding_cache:
            # Move to end to mark as recently used
            self._embedding_cache.move_to_end(text_hash)
            return self._embedding_cache[text_hash]

        # Retry logic for transient 500 errors (Ollama under load)
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Call Ollama embeddings API
                response = await self.client.post(
                    "/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text.strip()
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])

                    if embedding:
                        # Normalize embedding vector
                        normalized_embedding = self._normalize_vector(embedding)

                        # Cache the result
                        self._cache_embedding(text_hash, normalized_embedding)

                        return normalized_embedding
                    else:
                        logger.warning(f"Empty embedding returned for text: {text[:50]}...")
                        return [0.0] * self.embedding_dimensions
                elif response.status_code == 500:
                    # Transient server error - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Ollama embedding 500 error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        import asyncio
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Ollama embedding API failed after {max_retries} retries: 500 Internal Server Error")
                        return [0.0] * self.embedding_dimensions
                else:
                    logger.error(f"Ollama embedding API error: {response.status_code} - {response.text[:200]}")
                    return [0.0] * self.embedding_dimensions

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Embedding generation error: {type(e).__name__}, retrying in {delay}s")
                    import asyncio
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Embedding generation failed after {max_retries} retries: {type(e).__name__}: {e}")
                return [0.0] * self.embedding_dimensions

        return [0.0] * self.embedding_dimensions
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self.generate_embedding(text) for text in batch],
                return_exceptions=True
            )
            
            # Handle exceptions and add valid embeddings
            for embedding in batch_embeddings:
                if isinstance(embedding, Exception):
                    logger.warning(f"Batch embedding failed: {embedding}")
                    embeddings.append([0.0] * self.embedding_dimensions)
                else:
                    embeddings.append(embedding)
        
        return embeddings
    
    async def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embedding vectors
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            logger.warning("Embedding dimensions mismatch")
            return 0.0
        
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            if method == "cosine":
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return float(dot_product / (norm1 * norm2))
            
            elif method == "euclidean":
                # Convert Euclidean distance to similarity (0-1)
                distance = np.linalg.norm(vec1 - vec2)
                max_distance = 2.0  # Maximum possible distance for normalized vectors
                return float(1.0 - (distance / max_distance))
            
            elif method == "dot_product":
                # Dot product similarity
                return float(np.dot(vec1, vec2))
            
            else:
                logger.warning(f"Unknown similarity method: {method}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def find_similar_memories(
        self,
        query_embedding: List[float],
        memory_embeddings: List[Tuple[str, List[float]]],
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar memories to query embedding
        Returns list of (memory_id, similarity_score) tuples
        """
        similarities = []
        
        for memory_id, memory_embedding in memory_embeddings:
            similarity = await self.calculate_similarity(
                query_embedding, memory_embedding
            )
            
            if similarity >= threshold:
                similarities.append((memory_id, similarity))
        
        # Sort by similarity score (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def cluster_embeddings(
        self,
        embeddings: List[List[float]],
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> List[int]:
        """
        Cluster embeddings for memory organization
        Returns cluster assignments for each embedding
        """
        if not embeddings:
            return []
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Standardize embeddings
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embedding_matrix)
            
            # Perform clustering
            if method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(scaled_embeddings)
                return cluster_labels.tolist()
            else:
                logger.warning(f"Unknown clustering method: {method}")
                return [0] * len(embeddings)
                
        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return [0] * len(embeddings)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [0] * len(embeddings)
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics
        """
        return {
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self._cache_max_size,
            "ollama_base_url": self.settings.ollama_base_url
        }
    
    async def health_check(self) -> bool:
        """
        Check if Ollama embedding service is available
        """
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                # Check if our embedding model is available
                return any(
                    self.embedding_model in name 
                    for name in model_names
                )
            return False
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize vector to unit length for consistent similarity calculations
        """
        try:
            vec = np.array(vector)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return vector  # Return original if zero vector
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            logger.warning(f"Vector normalization failed: {e}")
            return vector
    
    def _cache_embedding(self, text_hash: int, embedding: List[float]):
        """
        Cache embedding with true LRU eviction.

        Uses OrderedDict.popitem(last=False) to remove the least recently
        accessed item (first in order). New items are added at the end,
        and accessed items are moved to end via move_to_end().
        """
        if len(self._embedding_cache) >= self._cache_max_size:
            # Remove least recently used item (first in OrderedDict)
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[text_hash] = embedding
    
    async def clear_cache(self):
        """
        Clear embedding cache
        """
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()


# Utility functions for memory-specific embedding operations

async def generate_therapeutic_embedding(
    text: str,
    recovery_stage: Optional[str] = None,
    therapeutic_context: Optional[Dict[str, Any]] = None
) -> List[float]:
    """
    Generate embedding with therapeutic context enhancement
    """
    embedding_service = EmbeddingService()
    
    # Enhance text with therapeutic context
    enhanced_text = text
    
    if recovery_stage:
        enhanced_text = f"Recovery stage: {recovery_stage}. {enhanced_text}"
    
    if therapeutic_context:
        context_items = []
        for key, value in therapeutic_context.items():
            if value:
                context_items.append(f"{key}: {value}")
        
        if context_items:
            context_str = ". ".join(context_items)
            enhanced_text = f"{context_str}. {enhanced_text}"
    
    return await embedding_service.generate_embedding(enhanced_text)


async def calculate_therapeutic_similarity(
    query_embedding: List[float],
    memory_embedding: List[float],
    recovery_stage: Optional[str] = None,
    therapeutic_weight: float = 1.0
) -> float:
    """
    Calculate similarity with therapeutic weighting
    """
    embedding_service = EmbeddingService()
    
    # Base cosine similarity
    base_similarity = await embedding_service.calculate_similarity(
        query_embedding, memory_embedding, method="cosine"
    )
    
    # Apply therapeutic weighting
    stage_weights = {
        'detox': 1.2,
        'early_recovery': 1.1,
        'maintenance': 0.9,
        'relapse_prevention': 1.0
    }
    
    stage_multiplier = stage_weights.get(recovery_stage, 1.0)
    
    return base_similarity * therapeutic_weight * stage_multiplier


# Global embedding service instance
embedding_service = EmbeddingService()