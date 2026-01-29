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

# Troubleshooting Task Tracker (Phase 2: Quest System Re-implementation)
from .troubleshooting_service import TroubleshootingService, troubleshooting_service
from .troubleshooting_tracker import (
    TroubleshootingTaskTracker,
    create_tracker,
    track_pipeline_task,
    PIPELINE_HOOKS,
)

__all__ = [
    # Memory services
    "MemoryService",
    "EmbeddingService",
    "PrivacyService",
    "EncryptionService",
    # Troubleshooting services
    "TroubleshootingService",
    "troubleshooting_service",
    "TroubleshootingTaskTracker",
    "create_tracker",
    "track_pipeline_task",
    "PIPELINE_HOOKS",
]