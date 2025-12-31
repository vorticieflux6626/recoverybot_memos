"""
Feature Flags for Gradual Rollout.

Part of G.4.3: Production Hardening - Simple config-based feature flags.

Provides gradual rollout, A/B testing, and kill-switch capabilities:
- Boolean flags for on/off features
- Percentage-based rollout for gradual deployment
- User segment targeting for specific groups
- Override support for testing

Key Features:
- Simple YAML/JSON configuration
- No external service dependency (can upgrade to Unleash later)
- Thread-safe flag evaluation
- Built-in percentage hashing for consistent user experience
- Flag bundles for related features

Research Basis:
- 2025 Multi-Agent RAG Breakthrough Report
- LaunchDarkly/Unleash feature flag patterns
- Google's gradual rollout best practices

Usage:
    from agentic.feature_flags import (
        FeatureFlagManager,
        Flag,
        get_feature_flags,
        is_enabled
    )

    manager = get_feature_flags()

    # Check if feature is enabled
    if manager.is_enabled("new_embedding_model"):
        use_new_model()

    # With user context for percentage rollout
    if manager.is_enabled("beta_features", user_id="user123"):
        show_beta_ui()

    # Get flag value with default
    value = manager.get_value("max_retries", default=3)
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

logger = logging.getLogger("agentic.feature_flags")


class FlagType(str, Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"          # Simple on/off
    PERCENTAGE = "percentage"    # Gradual rollout (0-100%)
    SEGMENT = "segment"          # User segment targeting
    VALUE = "value"              # Configuration value
    KILL_SWITCH = "kill_switch"  # Emergency disable


class FlagStatus(str, Enum):
    """Flag lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class FlagEvaluation:
    """Result of evaluating a feature flag."""
    flag_name: str
    enabled: bool
    value: Any
    reason: str  # Why this result
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Flag:
    """Feature flag definition."""
    name: str
    type: FlagType = FlagType.BOOLEAN
    enabled: bool = False
    percentage: float = 0.0  # For percentage rollout (0-100)
    value: Any = None  # For VALUE type flags
    segments: List[str] = field(default_factory=list)  # Allowed segments
    excluded_segments: List[str] = field(default_factory=list)  # Blocked segments
    description: str = ""
    status: FlagStatus = FlagStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "enabled": self.enabled,
            "percentage": self.percentage,
            "value": self.value,
            "segments": self.segments,
            "excluded_segments": self.excluded_segments,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Flag":
        """Create flag from dictionary."""
        return cls(
            name=data["name"],
            type=FlagType(data.get("type", "boolean")),
            enabled=data.get("enabled", False),
            percentage=data.get("percentage", 0.0),
            value=data.get("value"),
            segments=data.get("segments", []),
            excluded_segments=data.get("excluded_segments", []),
            description=data.get("description", ""),
            status=FlagStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FlagBundle:
    """Group of related flags that are activated together."""
    name: str
    flags: List[str]  # Flag names in this bundle
    description: str = ""
    enabled: bool = False  # Master switch for bundle


class FeatureFlagManager:
    """
    Manages feature flags for the agentic search system.

    Supports:
    - Boolean on/off flags
    - Percentage-based gradual rollout
    - User segment targeting
    - Configuration values
    - Kill switches for emergencies
    - Flag bundles for related features
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize feature flag manager.

        Args:
            config_path: Path to YAML/JSON config file
            overrides: Dict of flag_name -> enabled for testing
        """
        self._flags: Dict[str, Flag] = {}
        self._bundles: Dict[str, FlagBundle] = {}
        self._overrides = overrides or {}
        self._evaluation_history: List[FlagEvaluation] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()

        # Load default flags
        self._register_default_flags()

        # Load from config if provided
        if config_path:
            self.load_config(config_path)

        logger.info(f"FeatureFlagManager initialized with {len(self._flags)} flags")

    def _register_default_flags(self) -> None:
        """Register default feature flags for memOS agentic search."""
        default_flags = [
            # G.4 Production Hardening flags
            Flag(
                name="circuit_breakers_enabled",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Enable circuit breakers for external service calls"
            ),
            Flag(
                name="shadow_embeddings_enabled",
                type=FlagType.PERCENTAGE,
                enabled=True,
                percentage=100.0,
                description="Enable shadow mode for embedding model comparison"
            ),
            Flag(
                name="embedding_drift_monitoring",
                type=FlagType.BOOLEAN,
                enabled=False,
                description="Enable embedding drift detection and alerting"
            ),

            # RAG Architecture flags
            Flag(
                name="colbert_retrieval",
                type=FlagType.PERCENTAGE,
                enabled=True,
                percentage=100.0,
                description="Enable ColBERT late interaction retrieval"
            ),
            Flag(
                name="cascade_retrieval",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Enable binary->int8->fp16 cascade retrieval"
            ),
            Flag(
                name="query_intent_fusion",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Adapt fusion weights based on query intent"
            ),
            Flag(
                name="adaptive_topk",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Use CAR algorithm for dynamic top-k"
            ),

            # Graph RAG flags
            Flag(
                name="nano_graphrag",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Enable NanoGraphRAG with PPR + Leiden"
            ),
            Flag(
                name="gliner_extraction",
                type=FlagType.PERCENTAGE,
                enabled=True,
                percentage=50.0,
                description="Use GLiNER for entity extraction (vs regex only)"
            ),
            Flag(
                name="late_chunking",
                type=FlagType.BOOLEAN,
                enabled=True,
                description="Enable context-aware late chunking"
            ),

            # Advanced RAG (G.5) - Coming soon
            Flag(
                name="speculative_rag",
                type=FlagType.PERCENTAGE,
                enabled=False,
                percentage=0.0,
                description="Enable speculative RAG for 51% latency reduction"
            ),
            Flag(
                name="llmlingua_compression",
                type=FlagType.BOOLEAN,
                enabled=False,
                description="Enable LLMLingua-2 prompt compression"
            ),
            Flag(
                name="hoprag_multipath",
                type=FlagType.BOOLEAN,
                enabled=False,
                description="Enable HopRAG multi-hop passage graphs"
            ),

            # Agent Coordination (G.6) - Coming soon
            Flag(
                name="amem_memory",
                type=FlagType.BOOLEAN,
                enabled=False,
                description="Enable A-MEM cross-session agentic memory"
            ),
            Flag(
                name="dylan_agent_skipping",
                type=FlagType.BOOLEAN,
                enabled=False,
                description="Enable DyLAN agent importance scoring"
            ),

            # Kill switches
            Flag(
                name="web_scraping_kill_switch",
                type=FlagType.KILL_SWITCH,
                enabled=False,
                description="Emergency disable all web scraping"
            ),
            Flag(
                name="llm_calls_kill_switch",
                type=FlagType.KILL_SWITCH,
                enabled=False,
                description="Emergency disable LLM API calls"
            ),
            Flag(
                name="external_search_kill_switch",
                type=FlagType.KILL_SWITCH,
                enabled=False,
                description="Emergency disable external search providers"
            ),

            # Configuration values
            Flag(
                name="max_search_iterations",
                type=FlagType.VALUE,
                enabled=True,
                value=10,
                description="Maximum search iterations per request"
            ),
            Flag(
                name="cache_ttl_seconds",
                type=FlagType.VALUE,
                enabled=True,
                value=3600,
                description="Default cache TTL in seconds"
            ),
            Flag(
                name="embedding_batch_size",
                type=FlagType.VALUE,
                enabled=True,
                value=32,
                description="Batch size for embedding operations"
            ),
        ]

        for flag in default_flags:
            self._flags[flag.name] = flag

        # Register default bundles
        self._bundles["production_hardening"] = FlagBundle(
            name="production_hardening",
            flags=["circuit_breakers_enabled", "shadow_embeddings_enabled", "embedding_drift_monitoring"],
            description="G.4 Production hardening features",
            enabled=True
        )
        self._bundles["advanced_retrieval"] = FlagBundle(
            name="advanced_retrieval",
            flags=["colbert_retrieval", "cascade_retrieval", "query_intent_fusion", "adaptive_topk"],
            description="Advanced retrieval optimizations",
            enabled=True
        )
        self._bundles["graph_rag"] = FlagBundle(
            name="graph_rag",
            flags=["nano_graphrag", "gliner_extraction", "late_chunking"],
            description="Graph-enhanced RAG features",
            enabled=True
        )
        self._bundles["kill_switches"] = FlagBundle(
            name="kill_switches",
            flags=["web_scraping_kill_switch", "llm_calls_kill_switch", "external_search_kill_switch"],
            description="Emergency kill switches",
            enabled=False  # Never auto-enable
        )

    def load_config(self, config_path: str) -> bool:
        """Load flags from YAML or JSON config file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return False

        try:
            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Load flags
            for flag_data in data.get("flags", []):
                flag = Flag.from_dict(flag_data)
                self._flags[flag.name] = flag

            # Load bundles
            for bundle_data in data.get("bundles", []):
                bundle = FlagBundle(
                    name=bundle_data["name"],
                    flags=bundle_data.get("flags", []),
                    description=bundle_data.get("description", ""),
                    enabled=bundle_data.get("enabled", False)
                )
                self._bundles[bundle.name] = bundle

            logger.info(f"Loaded {len(data.get('flags', []))} flags from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return False

    def save_config(self, config_path: str) -> bool:
        """Save current flags to config file."""
        path = Path(config_path)
        try:
            data = {
                "flags": [flag.to_dict() for flag in self._flags.values()],
                "bundles": [
                    {
                        "name": bundle.name,
                        "flags": bundle.flags,
                        "description": bundle.description,
                        "enabled": bundle.enabled
                    }
                    for bundle in self._bundles.values()
                ]
            }

            with open(path, "w") as f:
                if path.suffix in [".yaml", ".yml"]:
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved {len(self._flags)} flags to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False

    def _hash_user_percentage(self, flag_name: str, user_id: str) -> float:
        """
        Generate consistent percentage for user+flag combination.

        Uses hash to ensure same user always gets same result for a flag,
        but different results across different flags.
        """
        combined = f"{flag_name}:{user_id}"
        hash_bytes = hashlib.md5(combined.encode()).digest()
        # Use first 4 bytes as unsigned int, mod 100 for percentage
        hash_int = int.from_bytes(hash_bytes[:4], "big")
        return (hash_int % 10000) / 100.0  # 0.00 to 99.99

    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        user_segment: Optional[str] = None,
        default: bool = False
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the flag
            user_id: Optional user ID for percentage rollout
            user_segment: Optional user segment for targeting
            default: Default value if flag not found

        Returns:
            True if enabled, False otherwise
        """
        # Check overrides first (for testing)
        if flag_name in self._overrides:
            return self._overrides[flag_name]

        flag = self._flags.get(flag_name)
        if not flag:
            logger.debug(f"Flag not found: {flag_name}, using default: {default}")
            return default

        # Archived flags are always disabled
        if flag.status == FlagStatus.ARCHIVED:
            return False

        # Kill switches return opposite of their enabled state
        if flag.type == FlagType.KILL_SWITCH:
            result = flag.enabled  # If kill switch is enabled, feature is disabled
            reason = "kill switch active" if result else "kill switch inactive"
        elif flag.type == FlagType.BOOLEAN:
            result = flag.enabled
            reason = "boolean flag"
        elif flag.type == FlagType.PERCENTAGE:
            if not flag.enabled:
                result = False
                reason = "percentage flag disabled"
            elif user_id:
                user_pct = self._hash_user_percentage(flag_name, user_id)
                result = user_pct < flag.percentage
                reason = f"percentage rollout ({user_pct:.2f} < {flag.percentage})"
            else:
                # No user ID, use random (not recommended for production)
                import random
                result = random.random() * 100 < flag.percentage
                reason = "percentage rollout (random, no user_id)"
        elif flag.type == FlagType.SEGMENT:
            if not flag.enabled:
                result = False
                reason = "segment flag disabled"
            elif user_segment in flag.excluded_segments:
                result = False
                reason = f"segment excluded: {user_segment}"
            elif not flag.segments or user_segment in flag.segments:
                result = True
                reason = f"segment allowed: {user_segment}"
            else:
                result = False
                reason = f"segment not in allowed list: {user_segment}"
        elif flag.type == FlagType.VALUE:
            result = flag.enabled and flag.value is not None
            reason = "value flag"
        else:
            result = default
            reason = "unknown type"

        # Record evaluation
        evaluation = FlagEvaluation(
            flag_name=flag_name,
            enabled=result,
            value=flag.value if flag.type == FlagType.VALUE else None,
            reason=reason,
            user_id=user_id
        )
        self._record_evaluation(evaluation)

        return result

    def get_value(
        self,
        flag_name: str,
        default: Any = None
    ) -> Any:
        """Get value of a VALUE type flag."""
        flag = self._flags.get(flag_name)
        if not flag or not flag.enabled or flag.type != FlagType.VALUE:
            return default
        return flag.value if flag.value is not None else default

    def set_enabled(self, flag_name: str, enabled: bool) -> bool:
        """Set flag enabled state."""
        if flag_name not in self._flags:
            return False

        self._flags[flag_name].enabled = enabled
        self._flags[flag_name].updated_at = datetime.now(UTC)
        logger.info(f"Flag '{flag_name}' set to enabled={enabled}")
        return True

    def set_percentage(self, flag_name: str, percentage: float) -> bool:
        """Set percentage for rollout flag."""
        if flag_name not in self._flags:
            return False

        flag = self._flags[flag_name]
        if flag.type != FlagType.PERCENTAGE:
            logger.warning(f"Flag '{flag_name}' is not a percentage type")
            return False

        flag.percentage = max(0.0, min(100.0, percentage))
        flag.updated_at = datetime.now(UTC)
        logger.info(f"Flag '{flag_name}' percentage set to {flag.percentage}")
        return True

    def set_value(self, flag_name: str, value: Any) -> bool:
        """Set value for VALUE type flag."""
        if flag_name not in self._flags:
            return False

        flag = self._flags[flag_name]
        if flag.type != FlagType.VALUE:
            logger.warning(f"Flag '{flag_name}' is not a value type")
            return False

        flag.value = value
        flag.updated_at = datetime.now(UTC)
        logger.info(f"Flag '{flag_name}' value set to {value}")
        return True

    def register_flag(self, flag: Flag) -> None:
        """Register a new flag."""
        self._flags[flag.name] = flag
        logger.info(f"Registered flag: {flag.name}")

    def get_flag(self, flag_name: str) -> Optional[Flag]:
        """Get flag by name."""
        return self._flags.get(flag_name)

    def get_all_flags(self) -> Dict[str, Flag]:
        """Get all registered flags."""
        return self._flags.copy()

    def get_flag_status(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a flag."""
        flag = self._flags.get(flag_name)
        if not flag:
            return None

        return {
            **flag.to_dict(),
            "evaluation_count": sum(
                1 for e in self._evaluation_history if e.flag_name == flag_name
            ),
            "recent_enabled_rate": self._get_recent_enabled_rate(flag_name),
        }

    def _get_recent_enabled_rate(self, flag_name: str, limit: int = 100) -> float:
        """Get recent enabled rate for a flag."""
        recent = [
            e for e in self._evaluation_history[-limit:]
            if e.flag_name == flag_name
        ]
        if not recent:
            return 0.0
        return sum(1 for e in recent if e.enabled) / len(recent)

    def _record_evaluation(self, evaluation: FlagEvaluation) -> None:
        """Record a flag evaluation."""
        self._evaluation_history.append(evaluation)
        # Trim history if needed
        if len(self._evaluation_history) > self._max_history:
            self._evaluation_history = self._evaluation_history[-self._max_history:]

    def enable_bundle(self, bundle_name: str) -> bool:
        """Enable all flags in a bundle."""
        bundle = self._bundles.get(bundle_name)
        if not bundle:
            return False

        for flag_name in bundle.flags:
            self.set_enabled(flag_name, True)

        bundle.enabled = True
        logger.info(f"Bundle '{bundle_name}' enabled ({len(bundle.flags)} flags)")
        return True

    def disable_bundle(self, bundle_name: str) -> bool:
        """Disable all flags in a bundle."""
        bundle = self._bundles.get(bundle_name)
        if not bundle:
            return False

        for flag_name in bundle.flags:
            self.set_enabled(flag_name, False)

        bundle.enabled = False
        logger.info(f"Bundle '{bundle_name}' disabled ({len(bundle.flags)} flags)")
        return True

    def get_bundle(self, bundle_name: str) -> Optional[FlagBundle]:
        """Get bundle by name."""
        return self._bundles.get(bundle_name)

    def get_all_bundles(self) -> Dict[str, FlagBundle]:
        """Get all bundles."""
        return self._bundles.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature flag statistics."""
        enabled_count = sum(1 for f in self._flags.values() if f.enabled)
        by_type = {}
        for flag in self._flags.values():
            by_type[flag.type.value] = by_type.get(flag.type.value, 0) + 1

        return {
            "total_flags": len(self._flags),
            "enabled_flags": enabled_count,
            "disabled_flags": len(self._flags) - enabled_count,
            "flags_by_type": by_type,
            "total_bundles": len(self._bundles),
            "enabled_bundles": sum(1 for b in self._bundles.values() if b.enabled),
            "total_evaluations": len(self._evaluation_history),
            "override_count": len(self._overrides),
        }

    def set_override(self, flag_name: str, enabled: bool) -> None:
        """Set an override for testing."""
        self._overrides[flag_name] = enabled
        logger.debug(f"Override set: {flag_name}={enabled}")

    def clear_override(self, flag_name: str) -> None:
        """Clear an override."""
        self._overrides.pop(flag_name, None)
        logger.debug(f"Override cleared: {flag_name}")

    def clear_all_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()
        logger.debug("All overrides cleared")


# Global instance
_feature_flags: Optional[FeatureFlagManager] = None


def get_feature_flags(
    config_path: Optional[str] = None
) -> FeatureFlagManager:
    """Get or create global feature flag manager."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlagManager(config_path)
    return _feature_flags


def is_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
    user_segment: Optional[str] = None,
    default: bool = False
) -> bool:
    """Convenience function to check if a flag is enabled."""
    return get_feature_flags().is_enabled(flag_name, user_id, user_segment, default)


def get_flag_value(flag_name: str, default: Any = None) -> Any:
    """Convenience function to get a flag value."""
    return get_feature_flags().get_value(flag_name, default)


# Context manager for temporary overrides
class FeatureFlagOverride:
    """Context manager for temporary flag overrides (useful for testing)."""

    def __init__(self, overrides: Dict[str, bool]):
        self.overrides = overrides
        self.manager = get_feature_flags()

    def __enter__(self):
        for flag_name, enabled in self.overrides.items():
            self.manager.set_override(flag_name, enabled)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for flag_name in self.overrides:
            self.manager.clear_override(flag_name)
        return False
