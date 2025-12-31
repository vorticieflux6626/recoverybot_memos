"""
Embedding Drift Monitoring.

Part of G.4.4: Production Hardening - Detect embedding model drift with alerting.

Monitors embedding distributions to detect:
- Model behavior changes (updates, degradation)
- Data distribution shifts
- Infrastructure issues affecting embeddings

Key Features:
- Statistical drift detection (KS test, KL divergence, cosine shift)
- Baseline management with rolling windows
- Alerting thresholds (warning, critical)
- Historical tracking for trend analysis
- Integration with shadow testing

Research Basis:
- 2025 Multi-Agent RAG Breakthrough Report
- ML Drift Detection (Alibi Detect, Evidently patterns)
- Production ML monitoring best practices

Usage:
    from agentic.embedding_drift import (
        EmbeddingDriftMonitor,
        DriftConfig,
        get_drift_monitor
    )

    monitor = get_drift_monitor()

    # Record embeddings during production
    monitor.record_batch(embeddings, source="production")

    # Check for drift
    result = monitor.check_drift()
    if result.is_drifted:
        alert(f"Embedding drift detected: {result.drift_type}")

    # Get drift report
    report = monitor.get_drift_report()
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger("agentic.embedding_drift")


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    DISTRIBUTION_SHIFT = "distribution_shift"  # Overall distribution changed
    MEAN_SHIFT = "mean_shift"  # Mean vector shifted
    VARIANCE_CHANGE = "variance_change"  # Variance increased/decreased
    NORM_CHANGE = "norm_change"  # Vector norms changed
    COSINE_SHIFT = "cosine_shift"  # Cosine similarity to baseline dropped


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    NONE = "none"
    INFO = "info"  # Minor change, informational
    WARNING = "warning"  # Significant change, needs attention
    CRITICAL = "critical"  # Major change, immediate action needed


@dataclass
class DriftConfig:
    """Configuration for drift monitoring."""
    # Window sizes
    baseline_window: int = 1000  # Number of embeddings for baseline
    detection_window: int = 100  # Number of embeddings for drift detection
    rolling_window: int = 500  # Rolling window for trend analysis

    # Statistical thresholds
    ks_warning_threshold: float = 0.1  # Kolmogorov-Smirnov warning
    ks_critical_threshold: float = 0.2  # KS critical
    mean_shift_warning: float = 0.05  # Mean shift warning (cosine distance)
    mean_shift_critical: float = 0.15  # Mean shift critical
    variance_change_warning: float = 0.2  # 20% variance change
    variance_change_critical: float = 0.5  # 50% variance change
    norm_change_warning: float = 0.1  # 10% norm change
    norm_change_critical: float = 0.3  # 30% norm change

    # Alerting
    alert_cooldown_seconds: int = 300  # Minimum time between alerts
    enable_alerts: bool = True

    # Sampling
    sample_rate: float = 1.0  # 1.0 = record all, 0.1 = 10%


@dataclass
class DriftMetrics:
    """Metrics from drift detection."""
    ks_statistic: float = 0.0  # Kolmogorov-Smirnov statistic
    ks_pvalue: float = 1.0
    mean_shift: float = 0.0  # Cosine distance of mean vectors
    variance_ratio: float = 1.0  # Current variance / baseline variance
    norm_ratio: float = 1.0  # Current mean norm / baseline mean norm
    dimension_drifts: List[int] = field(default_factory=list)  # Dimensions with drift


@dataclass
class DriftResult:
    """Result of drift detection."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_drifted: bool = False
    drift_type: DriftType = DriftType.NONE
    severity: DriftSeverity = DriftSeverity.NONE
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    baseline_size: int = 0
    detection_size: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_drifted": self.is_drifted,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metrics": {
                "ks_statistic": round(self.metrics.ks_statistic, 4),
                "ks_pvalue": round(self.metrics.ks_pvalue, 4),
                "mean_shift": round(self.metrics.mean_shift, 4),
                "variance_ratio": round(self.metrics.variance_ratio, 4),
                "norm_ratio": round(self.metrics.norm_ratio, 4),
                "dimension_drifts": self.metrics.dimension_drifts[:10],  # Top 10
            },
            "baseline_size": self.baseline_size,
            "detection_size": self.detection_size,
            "message": self.message,
        }


@dataclass
class AlertEvent:
    """Alert event from drift detection."""
    timestamp: datetime
    severity: DriftSeverity
    drift_type: DriftType
    message: str
    metrics: Dict[str, float]


class EmbeddingDriftMonitor:
    """
    Monitors embedding distributions for drift detection.

    Tracks embeddings over time and detects when the distribution
    changes significantly from the baseline.
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        dimensions: int = 1024,
        alert_callback: Optional[Callable[[AlertEvent], None]] = None
    ):
        """
        Initialize drift monitor.

        Args:
            config: Drift monitoring configuration
            dimensions: Expected embedding dimensions
            alert_callback: Callback for alerts
        """
        self.config = config or DriftConfig()
        self.dimensions = dimensions
        self.alert_callback = alert_callback

        # Baseline storage
        self._baseline_embeddings: deque = deque(maxlen=self.config.baseline_window)
        self._baseline_stats: Optional[Dict[str, np.ndarray]] = None
        self._baseline_locked: bool = False

        # Detection window
        self._detection_embeddings: deque = deque(maxlen=self.config.detection_window)

        # Rolling window for trends
        self._rolling_embeddings: deque = deque(maxlen=self.config.rolling_window)

        # History
        self._drift_history: List[DriftResult] = []
        self._alert_history: List[AlertEvent] = []
        self._last_alert_time: Optional[datetime] = None

        # Counters
        self._total_recorded: int = 0
        self._drifts_detected: int = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"EmbeddingDriftMonitor initialized: dim={dimensions}, "
            f"baseline_window={self.config.baseline_window}"
        )

    def record_embedding(
        self,
        embedding: np.ndarray,
        source: str = "production"
    ) -> None:
        """Record a single embedding for drift monitoring."""
        self.record_batch(embedding.reshape(1, -1), source)

    def record_batch(
        self,
        embeddings: np.ndarray,
        source: str = "production"
    ) -> int:
        """
        Record a batch of embeddings for drift monitoring.

        Args:
            embeddings: Array of shape (n, dimensions)
            source: Source identifier (for logging)

        Returns:
            Number of embeddings recorded
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Validate dimensions
        if embeddings.shape[1] != self.dimensions:
            logger.warning(
                f"Dimension mismatch: expected {self.dimensions}, got {embeddings.shape[1]}"
            )
            # Truncate or pad
            if embeddings.shape[1] > self.dimensions:
                embeddings = embeddings[:, :self.dimensions]
            else:
                padding = np.zeros((embeddings.shape[0], self.dimensions - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])

        # Sampling
        if self.config.sample_rate < 1.0:
            mask = np.random.random(len(embeddings)) < self.config.sample_rate
            embeddings = embeddings[mask]
            if len(embeddings) == 0:
                return 0

        # Record embeddings
        for emb in embeddings:
            if not self._baseline_locked:
                self._baseline_embeddings.append(emb)
            self._detection_embeddings.append(emb)
            self._rolling_embeddings.append(emb)

        self._total_recorded += len(embeddings)

        # Auto-lock baseline when full
        if (not self._baseline_locked and
            len(self._baseline_embeddings) >= self.config.baseline_window):
            self._compute_baseline_stats()
            self._baseline_locked = True
            logger.info(f"Baseline locked with {len(self._baseline_embeddings)} embeddings")

        return len(embeddings)

    def _compute_baseline_stats(self) -> None:
        """Compute and cache baseline statistics."""
        embeddings = np.array(list(self._baseline_embeddings))
        self._baseline_stats = {
            "mean": np.mean(embeddings, axis=0),
            "std": np.std(embeddings, axis=0),
            "variance": np.var(embeddings, axis=0),
            "norms": np.linalg.norm(embeddings, axis=1),
            "mean_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
        }

    def check_drift(self) -> DriftResult:
        """
        Check for drift between baseline and detection window.

        Returns:
            DriftResult with detection outcome
        """
        result = DriftResult()

        # Need both baseline and detection data
        if not self._baseline_locked:
            result.message = "Baseline not yet established"
            return result

        if len(self._detection_embeddings) < self.config.detection_window // 2:
            result.message = f"Insufficient detection data ({len(self._detection_embeddings)})"
            return result

        baseline = np.array(list(self._baseline_embeddings))
        detection = np.array(list(self._detection_embeddings))
        result.baseline_size = len(baseline)
        result.detection_size = len(detection)

        # Compute detection stats
        detection_mean = np.mean(detection, axis=0)
        detection_variance = np.var(detection, axis=0)
        detection_norms = np.linalg.norm(detection, axis=1)
        detection_mean_norm = np.mean(detection_norms)

        # 1. Mean shift (cosine distance)
        baseline_mean = self._baseline_stats["mean"]
        mean_shift = 1 - np.dot(baseline_mean, detection_mean) / (
            np.linalg.norm(baseline_mean) * np.linalg.norm(detection_mean) + 1e-9
        )
        result.metrics.mean_shift = float(mean_shift)

        # 2. Variance ratio
        baseline_var = np.mean(self._baseline_stats["variance"])
        detection_var = np.mean(detection_variance)
        variance_ratio = detection_var / (baseline_var + 1e-9)
        result.metrics.variance_ratio = float(variance_ratio)

        # 3. Norm ratio
        baseline_mean_norm = self._baseline_stats["mean_norm"]
        norm_ratio = detection_mean_norm / (baseline_mean_norm + 1e-9)
        result.metrics.norm_ratio = float(norm_ratio)

        # 4. KS test on norms (distribution test)
        baseline_norms = self._baseline_stats["norms"]
        ks_stat, ks_pvalue = stats.ks_2samp(baseline_norms, detection_norms)
        result.metrics.ks_statistic = float(ks_stat)
        result.metrics.ks_pvalue = float(ks_pvalue)

        # 5. Per-dimension drift (find drifting dimensions)
        dimension_drifts = []
        baseline_std = self._baseline_stats["std"]
        for dim in range(min(self.dimensions, 100)):  # Check first 100 dims
            dim_shift = abs(baseline_mean[dim] - detection_mean[dim])
            if dim_shift > 2 * baseline_std[dim]:  # 2-sigma rule
                dimension_drifts.append(dim)
        result.metrics.dimension_drifts = dimension_drifts

        # Determine drift type and severity
        result = self._evaluate_drift(result)

        # Record history
        self._drift_history.append(result)
        if result.is_drifted:
            self._drifts_detected += 1

        # Generate alert if needed
        if result.is_drifted and result.severity in [DriftSeverity.WARNING, DriftSeverity.CRITICAL]:
            self._maybe_alert(result)

        return result

    def _evaluate_drift(self, result: DriftResult) -> DriftResult:
        """Evaluate metrics and determine drift type/severity."""
        m = result.metrics
        cfg = self.config

        drift_reasons = []
        max_severity = DriftSeverity.NONE

        # Check mean shift
        if m.mean_shift >= cfg.mean_shift_critical:
            drift_reasons.append(("mean_shift", DriftSeverity.CRITICAL))
            max_severity = DriftSeverity.CRITICAL
        elif m.mean_shift >= cfg.mean_shift_warning:
            drift_reasons.append(("mean_shift", DriftSeverity.WARNING))
            if max_severity.value < DriftSeverity.WARNING.value:
                max_severity = DriftSeverity.WARNING

        # Check variance change
        var_change = abs(1 - m.variance_ratio)
        if var_change >= cfg.variance_change_critical:
            drift_reasons.append(("variance", DriftSeverity.CRITICAL))
            max_severity = DriftSeverity.CRITICAL
        elif var_change >= cfg.variance_change_warning:
            drift_reasons.append(("variance", DriftSeverity.WARNING))
            if max_severity.value < DriftSeverity.WARNING.value:
                max_severity = DriftSeverity.WARNING

        # Check norm change
        norm_change = abs(1 - m.norm_ratio)
        if norm_change >= cfg.norm_change_critical:
            drift_reasons.append(("norm", DriftSeverity.CRITICAL))
            max_severity = DriftSeverity.CRITICAL
        elif norm_change >= cfg.norm_change_warning:
            drift_reasons.append(("norm", DriftSeverity.WARNING))
            if max_severity.value < DriftSeverity.WARNING.value:
                max_severity = DriftSeverity.WARNING

        # Check KS statistic
        if m.ks_statistic >= cfg.ks_critical_threshold:
            drift_reasons.append(("distribution", DriftSeverity.CRITICAL))
            max_severity = DriftSeverity.CRITICAL
        elif m.ks_statistic >= cfg.ks_warning_threshold:
            drift_reasons.append(("distribution", DriftSeverity.WARNING))
            if max_severity.value < DriftSeverity.WARNING.value:
                max_severity = DriftSeverity.WARNING

        # Determine overall result
        if drift_reasons:
            result.is_drifted = True
            result.severity = max_severity

            # Determine primary drift type
            if any(r[0] == "mean_shift" for r in drift_reasons):
                result.drift_type = DriftType.MEAN_SHIFT
            elif any(r[0] == "variance" for r in drift_reasons):
                result.drift_type = DriftType.VARIANCE_CHANGE
            elif any(r[0] == "norm" for r in drift_reasons):
                result.drift_type = DriftType.NORM_CHANGE
            else:
                result.drift_type = DriftType.DISTRIBUTION_SHIFT

            reasons_str = ", ".join(f"{r[0]}({r[1].value})" for r in drift_reasons)
            result.message = f"Drift detected: {reasons_str}"
        else:
            result.message = "No significant drift detected"

        return result

    def _maybe_alert(self, result: DriftResult) -> None:
        """Generate an alert if cooldown allows."""
        if not self.config.enable_alerts:
            return

        now = datetime.now(UTC)
        if self._last_alert_time:
            cooldown = timedelta(seconds=self.config.alert_cooldown_seconds)
            if now - self._last_alert_time < cooldown:
                return

        alert = AlertEvent(
            timestamp=now,
            severity=result.severity,
            drift_type=result.drift_type,
            message=result.message,
            metrics={
                "ks_statistic": result.metrics.ks_statistic,
                "mean_shift": result.metrics.mean_shift,
                "variance_ratio": result.metrics.variance_ratio,
                "norm_ratio": result.metrics.norm_ratio,
            }
        )

        self._alert_history.append(alert)
        self._last_alert_time = now

        logger.warning(f"DRIFT ALERT [{result.severity.value}]: {result.message}")

        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def reset_baseline(self) -> None:
        """Reset baseline and start collecting new baseline data."""
        self._baseline_embeddings.clear()
        self._baseline_stats = None
        self._baseline_locked = False
        logger.info("Baseline reset")

    def lock_baseline(self) -> bool:
        """Manually lock the current baseline."""
        if len(self._baseline_embeddings) < self.config.detection_window:
            logger.warning("Cannot lock baseline: insufficient data")
            return False

        self._compute_baseline_stats()
        self._baseline_locked = True
        logger.info(f"Baseline manually locked with {len(self._baseline_embeddings)} embeddings")
        return True

    def get_drift_report(self) -> Dict[str, Any]:
        """Get comprehensive drift monitoring report."""
        recent_drifts = [r for r in self._drift_history[-100:] if r.is_drifted]

        return {
            "status": "monitoring" if self._baseline_locked else "collecting_baseline",
            "total_recorded": self._total_recorded,
            "baseline_size": len(self._baseline_embeddings),
            "detection_window_size": len(self._detection_embeddings),
            "baseline_locked": self._baseline_locked,
            "drifts_detected": self._drifts_detected,
            "recent_drifts": [r.to_dict() for r in recent_drifts[-10:]],
            "alert_count": len(self._alert_history),
            "last_check": (
                self._drift_history[-1].to_dict()
                if self._drift_history else None
            ),
            "config": {
                "baseline_window": self.config.baseline_window,
                "detection_window": self.config.detection_window,
                "sample_rate": self.config.sample_rate,
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get current drift monitoring statistics."""
        baseline_stats = {}
        if self._baseline_stats:
            baseline_stats = {
                "mean_norm": float(self._baseline_stats["mean_norm"]),
                "mean_std": float(np.mean(self._baseline_stats["std"])),
                "mean_variance": float(np.mean(self._baseline_stats["variance"])),
            }

        detection_stats = {}
        if len(self._detection_embeddings) > 10:
            detection = np.array(list(self._detection_embeddings))
            detection_stats = {
                "mean_norm": float(np.mean(np.linalg.norm(detection, axis=1))),
                "mean_std": float(np.mean(np.std(detection, axis=0))),
            }

        return {
            "dimensions": self.dimensions,
            "total_recorded": self._total_recorded,
            "baseline_size": len(self._baseline_embeddings),
            "detection_size": len(self._detection_embeddings),
            "rolling_size": len(self._rolling_embeddings),
            "baseline_locked": self._baseline_locked,
            "drifts_detected": self._drifts_detected,
            "total_checks": len(self._drift_history),
            "alert_count": len(self._alert_history),
            "baseline_stats": baseline_stats,
            "detection_stats": detection_stats,
        }

    def get_trend(self, window: int = 20) -> Dict[str, Any]:
        """Get drift trend from recent checks."""
        recent = self._drift_history[-window:] if len(self._drift_history) > 0 else []
        if not recent:
            return {"trend": "no_data", "checks": 0}

        drift_count = sum(1 for r in recent if r.is_drifted)
        drift_rate = drift_count / len(recent)

        avg_ks = np.mean([r.metrics.ks_statistic for r in recent])
        avg_mean_shift = np.mean([r.metrics.mean_shift for r in recent])

        # Determine trend
        if drift_rate > 0.5:
            trend = "deteriorating"
        elif drift_rate > 0.2:
            trend = "unstable"
        elif drift_rate > 0.05:
            trend = "minor_fluctuations"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "checks": len(recent),
            "drift_rate": round(drift_rate, 3),
            "avg_ks_statistic": round(avg_ks, 4),
            "avg_mean_shift": round(avg_mean_shift, 4),
        }


# Global instance
_drift_monitor: Optional[EmbeddingDriftMonitor] = None


def get_drift_monitor(
    dimensions: int = 1024,
    config: Optional[DriftConfig] = None
) -> EmbeddingDriftMonitor:
    """Get or create global drift monitor."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = EmbeddingDriftMonitor(config, dimensions)
    return _drift_monitor


def record_embedding_for_drift(
    embedding: np.ndarray,
    source: str = "production"
) -> None:
    """Convenience function to record embedding for drift monitoring."""
    get_drift_monitor(embedding.shape[-1]).record_embedding(embedding, source)


def check_embedding_drift() -> DriftResult:
    """Convenience function to check for drift."""
    return get_drift_monitor().check_drift()
