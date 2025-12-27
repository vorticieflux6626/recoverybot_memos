"""
Memory-specific embeddings utilities
Wrapper functions for memory service embedding needs
"""

from typing import List, Optional
from core.embedding_service import EmbeddingService

# Create singleton instance
_embedding_service = EmbeddingService()

async def get_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for memory content"""
    try:
        return await _embedding_service.generate_embedding(text)
    except Exception as e:
        # Return None if embedding generation fails
        return None