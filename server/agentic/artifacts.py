"""
Filesystem-based Artifact Storage for Agent Communication

Based on Anthropic's multi-agent research system patterns:
- Subagents write to files, pass lightweight references
- Prevents information loss in multi-stage processing
- Reduces token overhead significantly (only reference IDs passed between agents)

Ref: KV_CACHE_IMPLEMENTATION_PLAN.md Phase 2.3
Ref: https://www.anthropic.com/engineering/multi-agent-research-system
"""

import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("agentic.artifacts")


class ArtifactType(str, Enum):
    """Types of artifacts that can be stored"""
    SEARCH_RESULTS = "search_results"       # Web search result list
    SCRAPED_CONTENT = "scraped_content"     # Full scraped page content
    SYNTHESIS = "synthesis"                  # Synthesized response
    ANALYSIS = "analysis"                    # Query analysis result
    COVERAGE = "coverage"                    # Coverage evaluation result
    VERIFICATION = "verification"            # Verification result
    FINDINGS = "findings"                    # Extracted findings/facts
    SCRATCHPAD = "scratchpad"               # Full scratchpad state snapshot


@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts"""
    artifact_id: str
    artifact_type: ArtifactType
    session_id: str
    created_at: str
    created_by: str  # Agent that created this artifact
    size_bytes: int
    content_hash: str
    ttl_minutes: int = 60  # Default 1 hour TTL
    tags: List[str] = field(default_factory=list)
    parent_artifact_id: Optional[str] = None  # For tracing lineage


class ArtifactStore:
    """
    Filesystem-based artifact storage for agent communication.

    Design:
    - Artifacts stored as JSON files in session directories
    - Lightweight references (artifact_id) passed between agents
    - Automatic cleanup of expired artifacts
    - Support for artifact lineage tracking

    Usage:
        store = ArtifactStore()

        # Store artifact
        artifact_id = store.store(
            session_id="search_123",
            artifact_type=ArtifactType.SCRAPED_CONTENT,
            content={"url": "...", "text": "..."},
            created_by="scraper_agent"
        )

        # Pass artifact_id to next agent (not the full content)
        # Next agent retrieves:
        artifact = store.retrieve(session_id, artifact_id)
    """

    def __init__(
        self,
        base_path: str = "/tmp/memos_artifacts",
        default_ttl_minutes: int = 60,
        max_artifact_size_mb: int = 50
    ):
        """
        Initialize artifact store.

        Args:
            base_path: Root directory for artifact storage
            default_ttl_minutes: Default time-to-live for artifacts
            max_artifact_size_mb: Maximum size for single artifact
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.default_ttl_minutes = default_ttl_minutes
        self.max_artifact_size_bytes = max_artifact_size_mb * 1024 * 1024

        # Index for fast lookups
        self._index: Dict[str, ArtifactMetadata] = {}

        # Stats
        self.stats = {
            "artifacts_stored": 0,
            "artifacts_retrieved": 0,
            "artifacts_expired": 0,
            "bytes_stored": 0,
        }

        logger.info(f"ArtifactStore initialized at {base_path}")

    def _generate_artifact_id(
        self,
        session_id: str,
        artifact_type: ArtifactType,
        content: Any
    ) -> str:
        """Generate unique artifact ID with content hash for dedup"""
        content_str = json.dumps(content, default=str, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S%f")[:10]
        return f"{artifact_type.value[:4]}_{timestamp}_{content_hash}"

    def _get_session_path(self, session_id: str) -> Path:
        """Get or create session directory"""
        session_path = self.base_path / session_id
        session_path.mkdir(exist_ok=True)
        return session_path

    def store(
        self,
        session_id: str,
        artifact_type: ArtifactType,
        content: Any,
        created_by: str = "unknown",
        metadata: Optional[Dict] = None,
        ttl_minutes: Optional[int] = None,
        parent_artifact_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store artifact and return lightweight reference.

        Args:
            session_id: Session/workflow identifier
            artifact_type: Type of artifact
            content: Artifact content (will be JSON serialized)
            created_by: Agent that created this artifact
            metadata: Optional additional metadata
            ttl_minutes: Time-to-live (overrides default)
            parent_artifact_id: ID of parent artifact (for lineage)
            tags: Optional tags for categorization

        Returns:
            artifact_id: Lightweight reference to stored artifact
        """
        # Serialize content
        content_str = json.dumps(content, default=str, indent=2)
        content_bytes = content_str.encode('utf-8')

        # Check size limit
        if len(content_bytes) > self.max_artifact_size_bytes:
            raise ValueError(
                f"Artifact too large: {len(content_bytes)} bytes "
                f"(max: {self.max_artifact_size_bytes})"
            )

        # Generate ID
        artifact_id = self._generate_artifact_id(session_id, artifact_type, content)

        # Calculate content hash
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Create metadata
        artifact_meta = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
            ttl_minutes=ttl_minutes or self.default_ttl_minutes,
            tags=tags or [],
            parent_artifact_id=parent_artifact_id
        )

        # Build full artifact with metadata
        full_artifact = {
            "metadata": asdict(artifact_meta),
            "content": content,
            "extra_metadata": metadata or {}
        }

        # Write to file
        session_path = self._get_session_path(session_id)
        artifact_path = session_path / f"{artifact_id}.json"
        artifact_path.write_text(json.dumps(full_artifact, default=str, indent=2))

        # Update index
        self._index[artifact_id] = artifact_meta

        # Update stats
        self.stats["artifacts_stored"] += 1
        self.stats["bytes_stored"] += len(content_bytes)

        logger.debug(
            f"Stored artifact {artifact_id} ({artifact_type.value}, "
            f"{len(content_bytes)} bytes) by {created_by}"
        )

        return artifact_id

    def retrieve(
        self,
        session_id: str,
        artifact_id: str,
        include_metadata: bool = False
    ) -> Optional[Any]:
        """
        Retrieve artifact by reference.

        Args:
            session_id: Session identifier
            artifact_id: Artifact reference
            include_metadata: If True, return full artifact with metadata

        Returns:
            Artifact content (or full artifact if include_metadata=True)
            None if not found
        """
        artifact_path = self.base_path / session_id / f"{artifact_id}.json"

        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_id}")
            return None

        try:
            full_artifact = json.loads(artifact_path.read_text())

            # Check expiry
            meta = full_artifact.get("metadata", {})
            created_at = datetime.fromisoformat(meta.get("created_at", ""))
            ttl_minutes = meta.get("ttl_minutes", self.default_ttl_minutes)
            expires_at = created_at + timedelta(minutes=ttl_minutes)

            if datetime.now(timezone.utc) > expires_at:
                logger.debug(f"Artifact expired: {artifact_id}")
                self._delete_artifact(session_id, artifact_id)
                self.stats["artifacts_expired"] += 1
                return None

            self.stats["artifacts_retrieved"] += 1

            if include_metadata:
                return full_artifact
            return full_artifact.get("content")

        except Exception as e:
            logger.error(f"Error retrieving artifact {artifact_id}: {e}")
            return None

    def retrieve_by_type(
        self,
        session_id: str,
        artifact_type: ArtifactType,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all artifacts of a specific type for a session.

        Returns list of {artifact_id, content, metadata}
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            return []

        results = []
        for artifact_path in session_path.glob("*.json"):
            try:
                full_artifact = json.loads(artifact_path.read_text())
                meta = full_artifact.get("metadata", {})

                if meta.get("artifact_type") == artifact_type.value:
                    results.append({
                        "artifact_id": meta.get("artifact_id"),
                        "content": full_artifact.get("content"),
                        "metadata": meta
                    })

                    if len(results) >= limit:
                        break

            except Exception as e:
                logger.warning(f"Error reading artifact {artifact_path}: {e}")

        return results

    def get_lineage(self, session_id: str, artifact_id: str) -> List[str]:
        """
        Get artifact lineage (chain of parent artifacts).

        Returns list of artifact IDs from root to current.
        """
        lineage = []
        current_id = artifact_id

        while current_id:
            artifact = self.retrieve(session_id, current_id, include_metadata=True)
            if not artifact:
                break

            lineage.insert(0, current_id)
            current_id = artifact.get("metadata", {}).get("parent_artifact_id")

        return lineage

    def _delete_artifact(self, session_id: str, artifact_id: str):
        """Delete a specific artifact"""
        artifact_path = self.base_path / session_id / f"{artifact_id}.json"
        if artifact_path.exists():
            artifact_path.unlink()

        if artifact_id in self._index:
            del self._index[artifact_id]

    def cleanup_session(self, session_id: str):
        """Clean up all artifacts for a session"""
        session_path = self.base_path / session_id
        if session_path.exists():
            shutil.rmtree(session_path)

        # Clean index
        to_remove = [aid for aid, meta in self._index.items()
                     if meta.session_id == session_id]
        for aid in to_remove:
            del self._index[aid]

        logger.info(f"Cleaned up session {session_id}")

    def cleanup_expired(self) -> int:
        """
        Clean up expired artifacts across all sessions.

        Returns number of artifacts cleaned.
        """
        cleaned = 0
        now = datetime.now(timezone.utc)

        for session_path in self.base_path.iterdir():
            if not session_path.is_dir():
                continue

            for artifact_path in session_path.glob("*.json"):
                try:
                    full_artifact = json.loads(artifact_path.read_text())
                    meta = full_artifact.get("metadata", {})

                    created_at = datetime.fromisoformat(meta.get("created_at", ""))
                    ttl_minutes = meta.get("ttl_minutes", self.default_ttl_minutes)
                    expires_at = created_at + timedelta(minutes=ttl_minutes)

                    if now > expires_at:
                        artifact_path.unlink()
                        cleaned += 1

                except Exception as e:
                    logger.warning(f"Error checking artifact {artifact_path}: {e}")

            # Remove empty session directories
            if not list(session_path.glob("*.json")):
                session_path.rmdir()

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired artifacts")
            self.stats["artifacts_expired"] += cleaned

        return cleaned

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of artifacts in a session"""
        session_path = self.base_path / session_id
        if not session_path.exists():
            return {"exists": False}

        artifacts_by_type: Dict[str, int] = {}
        total_bytes = 0

        for artifact_path in session_path.glob("*.json"):
            try:
                full_artifact = json.loads(artifact_path.read_text())
                meta = full_artifact.get("metadata", {})
                artifact_type = meta.get("artifact_type", "unknown")

                artifacts_by_type[artifact_type] = artifacts_by_type.get(artifact_type, 0) + 1
                total_bytes += meta.get("size_bytes", 0)

            except Exception:
                pass

        return {
            "exists": True,
            "session_id": session_id,
            "artifact_count": sum(artifacts_by_type.values()),
            "artifacts_by_type": artifacts_by_type,
            "total_bytes": total_bytes
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get artifact store statistics"""
        return {
            **self.stats,
            "index_size": len(self._index),
            "base_path": str(self.base_path)
        }


# Global artifact store instance
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance"""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


# Convenience functions for common patterns
def store_search_results(
    session_id: str,
    results: List[Dict],
    query: str,
    agent: str = "searcher"
) -> str:
    """Store search results with query context"""
    store = get_artifact_store()
    return store.store(
        session_id=session_id,
        artifact_type=ArtifactType.SEARCH_RESULTS,
        content={"query": query, "results": results, "count": len(results)},
        created_by=agent,
        tags=["search", query[:50]]
    )


def store_scraped_content(
    session_id: str,
    url: str,
    content: str,
    title: str = "",
    agent: str = "scraper"
) -> str:
    """Store scraped page content"""
    store = get_artifact_store()
    return store.store(
        session_id=session_id,
        artifact_type=ArtifactType.SCRAPED_CONTENT,
        content={"url": url, "title": title, "content": content},
        created_by=agent,
        tags=["scraped", url[:50]]
    )


def store_synthesis(
    session_id: str,
    synthesis: str,
    sources: List[str],
    agent: str = "synthesizer",
    parent_artifact_id: Optional[str] = None
) -> str:
    """Store synthesized response"""
    store = get_artifact_store()
    return store.store(
        session_id=session_id,
        artifact_type=ArtifactType.SYNTHESIS,
        content={"synthesis": synthesis, "sources": sources},
        created_by=agent,
        parent_artifact_id=parent_artifact_id,
        tags=["synthesis"]
    )


def get_session_artifacts(session_id: str) -> Dict[str, Any]:
    """Get all artifacts for a session organized by type"""
    store = get_artifact_store()
    summary = store.get_session_summary(session_id)

    if not summary.get("exists"):
        return summary

    # Get actual artifacts by type
    artifacts = {}
    for artifact_type in ArtifactType:
        type_artifacts = store.retrieve_by_type(session_id, artifact_type)
        if type_artifacts:
            artifacts[artifact_type.value] = type_artifacts

    return {
        **summary,
        "artifacts": artifacts
    }
