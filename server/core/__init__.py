"""
memOS Server Core Components
HIPAA-compliant memory services for Recovery Bot
"""

# NOTE: Previously exported MemoryServiceFixed (Mem0 disabled) due to initialization bug.
# Bug fixed 2025-12-29: Import shadowing issue resolved in memory_service.py
# Now exporting the full MemoryService with Mem0 integration enabled.
from .memory_service import MemoryService
from .embedding_service import EmbeddingService
from .privacy_service import PrivacyService
from .encryption_service import EncryptionService

__all__ = [
    "MemoryService",
    "EmbeddingService",
    "PrivacyService",
    "EncryptionService"
]