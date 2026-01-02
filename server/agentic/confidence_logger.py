"""
Confidence Logger - Multi-signal confidence breakdown for calibration debugging.

Part of P2 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Provides detailed logging for confidence score calculations:
- Individual signal scores with weights
- Weighted contribution breakdown
- Threshold comparisons
- Calibration drift detection
- SSE emission for real-time visibility

Based on AGENTIC_IMPROVEMENT_PLAN.md target weights:
- verification: 40%
- source_diversity: 25%
- content_depth: 20%
- synthesis_quality: 15%

Created: 2026-01-02
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfidenceSignal(str, Enum):
    """Types of confidence signals."""
    VERIFICATION = "verification"
    SOURCE_DIVERSITY = "source_diversity"
    CONTENT_DEPTH = "content_depth"
    SYNTHESIS_QUALITY = "synthesis_quality"
    # Additional signals that can be added
    CITATION_ACCURACY = "citation_accuracy"
    TEMPORAL_FRESHNESS = "temporal_freshness"
    DOMAIN_RELEVANCE = "domain_relevance"


class ConfidenceLevel(str, Enum):
    """Confidence level classifications."""
    VERY_HIGH = "very_high"      # >= 0.9
    HIGH = "high"                # >= 0.8
    MODERATE = "moderate"        # >= 0.6
    LOW = "low"                  # >= 0.4
    VERY_LOW = "very_low"        # < 0.4


# Default weights from AGENTIC_IMPROVEMENT_PLAN.md
DEFAULT_WEIGHTS = {
    ConfidenceSignal.VERIFICATION: 0.40,
    ConfidenceSignal.SOURCE_DIVERSITY: 0.25,
    ConfidenceSignal.CONTENT_DEPTH: 0.20,
    ConfidenceSignal.SYNTHESIS_QUALITY: 0.15,
}


@dataclass
class SignalScore:
    """Individual signal score with metadata."""
    signal: str
    raw_score: float  # 0-1 score
    weight: float  # Weight applied
    weighted_score: float  # raw_score * weight
    # Optional metadata
    source_count: int = 0
    threshold: Optional[float] = None
    above_threshold: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "raw": round(self.raw_score, 3),
            "weight": round(self.weight, 2),
            "weighted": round(self.weighted_score, 3),
            "source_count": self.source_count,
            "threshold": self.threshold,
            "above_threshold": self.above_threshold,
            "notes": self.notes
        }


@dataclass
class ConfidenceBreakdown:
    """Complete confidence calculation breakdown."""
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Individual signals
    signals: List[SignalScore] = field(default_factory=list)

    # Final scores
    final_confidence: float = 0.0
    confidence_level: str = ConfidenceLevel.VERY_LOW.value

    # Calibration info
    target_confidence: Optional[float] = None  # Expected confidence if known
    calibration_error: Optional[float] = None  # Difference from target

    # Thresholds used
    thresholds_applied: Dict[str, float] = field(default_factory=dict)

    # Flags
    low_confidence_warning: bool = False
    signals_disagree: bool = False  # If signals have high variance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "final_confidence": round(self.final_confidence, 3),
            "confidence_level": self.confidence_level,
            "target_confidence": self.target_confidence,
            "calibration_error": round(self.calibration_error, 3) if self.calibration_error else None,
            "thresholds": self.thresholds_applied,
            "warnings": {
                "low_confidence": self.low_confidence_warning,
                "signals_disagree": self.signals_disagree
            }
        }

    def to_log_string(self) -> str:
        """Generate concise log string."""
        parts = []
        for s in self.signals:
            parts.append(f"{s.signal}={s.raw_score:.2f}×{s.weight:.2f}={s.weighted_score:.2f}")
        return f"{' | '.join(parts)} | FINAL={self.final_confidence:.2f}"


class ConfidenceLogger:
    """
    Log confidence calculations with detailed breakdown.

    Usage:
        conf_logger = ConfidenceLogger(request_id="req-123")

        breakdown = conf_logger.calculate_and_log(
            verification_score=0.85,
            source_diversity_score=0.70,
            content_depth_score=0.60,
            synthesis_quality_score=0.75
        )

        # Or with custom signals
        conf_logger.add_signal("verification", 0.85, weight=0.40)
        conf_logger.add_signal("source_diversity", 0.70, weight=0.25)
        breakdown = conf_logger.finalize()
    """

    def __init__(
        self,
        request_id: str,
        emitter: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.request_id = request_id
        self.emitter = emitter
        self.weights = weights or {k.value: v for k, v in DEFAULT_WEIGHTS.items()}
        self.signals: List[SignalScore] = []
        self._thresholds: Dict[str, float] = {}

    def set_threshold(self, signal: str, threshold: float):
        """Set a threshold for a signal."""
        self._thresholds[signal] = threshold

    def add_signal(
        self,
        signal: str,
        score: float,
        weight: Optional[float] = None,
        source_count: int = 0,
        notes: str = ""
    ) -> SignalScore:
        """
        Add a signal score.

        Args:
            signal: Signal name
            score: Raw score (0-1)
            weight: Weight override (uses default if not provided)
            source_count: Number of sources contributing to this signal
            notes: Additional notes

        Returns:
            SignalScore instance
        """
        # Get weight
        if weight is None:
            weight = self.weights.get(signal, 0.10)

        # Calculate weighted score
        weighted_score = score * weight

        # Check threshold if set
        threshold = self._thresholds.get(signal)
        above_threshold = score >= threshold if threshold is not None else None

        signal_score = SignalScore(
            signal=signal,
            raw_score=score,
            weight=weight,
            weighted_score=weighted_score,
            source_count=source_count,
            threshold=threshold,
            above_threshold=above_threshold,
            notes=notes
        )

        self.signals.append(signal_score)
        return signal_score

    def finalize(self, target_confidence: Optional[float] = None) -> ConfidenceBreakdown:
        """
        Finalize and log the confidence calculation.

        Args:
            target_confidence: Expected confidence for calibration tracking

        Returns:
            ConfidenceBreakdown instance
        """
        # Calculate final confidence
        final_confidence = sum(s.weighted_score for s in self.signals)

        # Clamp to 0-1
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Determine level
        if final_confidence >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif final_confidence >= 0.8:
            level = ConfidenceLevel.HIGH
        elif final_confidence >= 0.6:
            level = ConfidenceLevel.MODERATE
        elif final_confidence >= 0.4:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        # Check for calibration error
        calibration_error = None
        if target_confidence is not None:
            calibration_error = final_confidence - target_confidence

        # Check for signal disagreement (high variance)
        raw_scores = [s.raw_score for s in self.signals]
        if raw_scores:
            variance = sum((s - sum(raw_scores)/len(raw_scores))**2 for s in raw_scores) / len(raw_scores)
            signals_disagree = variance > 0.05  # Threshold for "disagreement"
        else:
            signals_disagree = False

        breakdown = ConfidenceBreakdown(
            request_id=self.request_id,
            signals=self.signals,
            final_confidence=final_confidence,
            confidence_level=level.value,
            target_confidence=target_confidence,
            calibration_error=calibration_error,
            thresholds_applied=self._thresholds.copy(),
            low_confidence_warning=final_confidence < 0.6,
            signals_disagree=signals_disagree
        )

        # Log the breakdown
        self._log_breakdown(breakdown)

        return breakdown

    def _log_breakdown(self, breakdown: ConfidenceBreakdown):
        """Log the confidence breakdown."""
        # Structured log
        logger.info(
            f"[{self.request_id}] Confidence: {breakdown.to_log_string()}"
        )

        # Warning logs
        if breakdown.low_confidence_warning:
            logger.warning(
                f"[{self.request_id}] LOW CONFIDENCE: {breakdown.final_confidence:.2f} "
                f"(level={breakdown.confidence_level})"
            )

        if breakdown.signals_disagree:
            logger.warning(
                f"[{self.request_id}] SIGNAL DISAGREEMENT: High variance in confidence signals"
            )

        if breakdown.calibration_error is not None and abs(breakdown.calibration_error) > 0.2:
            logger.warning(
                f"[{self.request_id}] CALIBRATION DRIFT: "
                f"error={breakdown.calibration_error:.2f} "
                f"(expected={breakdown.target_confidence:.2f}, got={breakdown.final_confidence:.2f})"
            )

    async def emit_confidence_event(self, breakdown: ConfidenceBreakdown):
        """Emit SSE event for confidence calculation."""
        if self.emitter:
            try:
                from agentic.events import SearchEvent, EventType
                await self.emitter.emit(SearchEvent(
                    event_type=EventType.CONFIDENCE_CALCULATED,
                    request_id=self.request_id,
                    data={
                        "signals": {s.signal: s.raw_score for s in breakdown.signals},
                        "weights": {s.signal: s.weight for s in breakdown.signals},
                        "weighted_scores": {s.signal: s.weighted_score for s in breakdown.signals},
                        "final_confidence": breakdown.final_confidence,
                        "level": breakdown.confidence_level,
                        "warnings": {
                            "low_confidence": breakdown.low_confidence_warning,
                            "signals_disagree": breakdown.signals_disagree
                        }
                    }
                ))
            except Exception as e:
                logger.debug(f"Failed to emit confidence event: {e}")

    async def calculate_and_log(
        self,
        verification_score: float = 0.0,
        source_diversity_score: float = 0.0,
        content_depth_score: float = 0.0,
        synthesis_quality_score: float = 0.0,
        target_confidence: Optional[float] = None,
        emit_event: bool = True
    ) -> ConfidenceBreakdown:
        """
        Calculate confidence from standard signals and log breakdown.

        Args:
            verification_score: Verification signal (0-1)
            source_diversity_score: Source diversity signal (0-1)
            content_depth_score: Content depth signal (0-1)
            synthesis_quality_score: Synthesis quality signal (0-1)
            target_confidence: Expected confidence for calibration
            emit_event: Whether to emit SSE event

        Returns:
            ConfidenceBreakdown instance
        """
        # Add standard signals
        self.add_signal(
            ConfidenceSignal.VERIFICATION.value,
            verification_score,
            notes="Claim verification score"
        )
        self.add_signal(
            ConfidenceSignal.SOURCE_DIVERSITY.value,
            source_diversity_score,
            notes="Source diversity score"
        )
        self.add_signal(
            ConfidenceSignal.CONTENT_DEPTH.value,
            content_depth_score,
            notes="Content depth score"
        )
        self.add_signal(
            ConfidenceSignal.SYNTHESIS_QUALITY.value,
            synthesis_quality_score,
            notes="Synthesis quality score"
        )

        # Finalize
        breakdown = self.finalize(target_confidence)

        # Emit SSE event
        if emit_event:
            await self.emit_confidence_event(breakdown)

        return breakdown

    def calculate_and_log_sync(
        self,
        verification_score: float = 0.0,
        source_diversity_score: float = 0.0,
        content_depth_score: float = 0.0,
        synthesis_quality_score: float = 0.0,
        target_confidence: Optional[float] = None
    ) -> ConfidenceBreakdown:
        """
        Synchronous version of calculate_and_log (no SSE emission).
        """
        self.add_signal(ConfidenceSignal.VERIFICATION.value, verification_score)
        self.add_signal(ConfidenceSignal.SOURCE_DIVERSITY.value, source_diversity_score)
        self.add_signal(ConfidenceSignal.CONTENT_DEPTH.value, content_depth_score)
        self.add_signal(ConfidenceSignal.SYNTHESIS_QUALITY.value, synthesis_quality_score)
        return self.finalize(target_confidence)


# Calibration tracking
class ConfidenceCalibrationTracker:
    """
    Track confidence calibration over time.

    Detects systematic over/under-confidence.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: List[Tuple[float, float]] = []  # (predicted, actual)

    def record(self, predicted: float, actual: Optional[float] = None):
        """
        Record a confidence prediction.

        Args:
            predicted: Predicted confidence
            actual: Actual outcome (1.0 for correct, 0.0 for incorrect)
        """
        if actual is not None:
            self._history.append((predicted, actual))
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        if len(self._history) < 10:
            return {"status": "insufficient_data", "count": len(self._history)}

        predictions, actuals = zip(*self._history)

        # Mean predicted vs mean actual
        mean_predicted = sum(predictions) / len(predictions)
        mean_actual = sum(actuals) / len(actuals)

        # Calibration error (Brier score component)
        calibration_error = mean_predicted - mean_actual

        # Binned calibration (reliability diagram data)
        bins = {}
        for pred, actual in self._history:
            bin_idx = int(pred * 10) / 10  # Round to 0.1
            if bin_idx not in bins:
                bins[bin_idx] = {"count": 0, "actual_sum": 0}
            bins[bin_idx]["count"] += 1
            bins[bin_idx]["actual_sum"] += actual

        reliability = {
            k: v["actual_sum"] / v["count"]
            for k, v in bins.items()
            if v["count"] > 0
        }

        return {
            "status": "ok",
            "count": len(self._history),
            "mean_predicted": round(mean_predicted, 3),
            "mean_actual": round(mean_actual, 3),
            "calibration_error": round(calibration_error, 3),
            "over_confident": calibration_error > 0.1,
            "under_confident": calibration_error < -0.1,
            "reliability_by_bin": reliability
        }


# Global calibration tracker
_calibration_tracker = ConfidenceCalibrationTracker()


def get_confidence_logger(
    request_id: str,
    emitter: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None
) -> ConfidenceLogger:
    """
    Factory function to get a ConfidenceLogger instance.

    Args:
        request_id: Unique request identifier
        emitter: Optional SSE event emitter
        weights: Optional custom weights

    Returns:
        ConfidenceLogger instance
    """
    return ConfidenceLogger(request_id, emitter, weights)


def get_calibration_tracker() -> ConfidenceCalibrationTracker:
    """Get the global calibration tracker."""
    return _calibration_tracker
