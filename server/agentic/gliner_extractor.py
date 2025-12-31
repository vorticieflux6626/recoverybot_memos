"""
GLiNER + Regex Hybrid Entity Extraction.

Part of G.3.3: GLiNER + regex hybrid entity extraction for improved entity coverage.

Combines pattern-based extraction (100% precision for structured entities) with
GLiNER zero-shot NER (for variable entities like symptoms, causes, solutions).

Research Basis:
- GLiNER (NAACL 2024): Generalist and Lightweight Model for Named Entity Recognition
- "GLiNER can identify any entity type using a bidirectional transformer encoder"
- Zero-shot capability means no fine-tuning needed for new entity types

Architecture:
    Text Input
        |
        +---> [Regex Patterns] ---> Structured entities (error codes, parameters)
        |                                      |
        +---> [GLiNER Model] ---> Variable entities (symptoms, causes, solutions)
        |                                      |
        v                                      v
    [Merge with Priority] ---> regex wins for overlapping spans
        |
        v
    Unified Entity List

Key Features:
- Regex first for structured patterns (100% precision)
- GLiNER for variable entities (zero-shot NER)
- Merge with priority to regex on overlapping spans
- Multiple domain support (FANUC, Raspberry Pi, generic industrial)
- Caching for efficient model loading
- Async-compatible extraction

Usage:
    from agentic.gliner_extractor import GLiNERHybridExtractor, get_gliner_extractor

    extractor = get_gliner_extractor()
    entities = await extractor.extract("SRVO-063 caused by encoder failure on J1")
    # Returns: [
    #   Entity(text="SRVO-063", type="error_code", source="regex", confidence=1.0),
    #   Entity(text="encoder failure", type="cause", source="gliner", confidence=0.85),
    #   Entity(text="J1", type="axis", source="regex", confidence=1.0)
    # ]
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger("agentic.gliner_extractor")

# Try to import GLiNER
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
    logger.info("GLiNER available")
except ImportError as e:
    GLINER_AVAILABLE = False
    GLiNER = None
    logger.warning(f"GLiNER not available: {e}. Install with: pip install gliner")


class EntitySource(str, Enum):
    """Source of entity extraction."""
    REGEX = "regex"
    GLINER = "gliner"
    MERGED = "merged"


class EntityCategory(str, Enum):
    """Categories of entities for organization."""
    ERROR = "error"
    COMPONENT = "component"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"
    CAUSE = "cause"
    SOLUTION = "solution"
    MEASUREMENT = "measurement"
    IDENTIFIER = "identifier"
    SAFETY = "safety"
    OTHER = "other"


@dataclass
class ExtractedEntity:
    """
    Extracted entity with span information.

    Attributes:
        text: The extracted text
        entity_type: Specific type (e.g., "error_code", "symptom")
        category: General category (e.g., "error", "component")
        start: Start character offset
        end: End character offset
        source: Extraction source (regex or gliner)
        confidence: Confidence score (1.0 for regex, model confidence for gliner)
        context: Surrounding text for context
        metadata: Additional extraction metadata
    """
    text: str
    entity_type: str
    category: EntityCategory
    start: int
    end: int
    source: EntitySource
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "category": self.category.value,
            "start": self.start,
            "end": self.end,
            "source": self.source.value,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[ExtractedEntity]
    text: str
    regex_count: int = 0
    gliner_count: int = 0
    merged_count: int = 0
    processing_time_ms: float = 0.0

    def get_by_type(self, entity_type: str) -> List[ExtractedEntity]:
        """Get entities by type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_by_category(self, category: EntityCategory) -> List[ExtractedEntity]:
        """Get entities by category."""
        return [e for e in self.entities if e.category == category]

    def get_by_source(self, source: EntitySource) -> List[ExtractedEntity]:
        """Get entities by source."""
        return [e for e in self.entities if e.source == source]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "regex_count": self.regex_count,
            "gliner_count": self.gliner_count,
            "merged_count": self.merged_count,
            "processing_time_ms": self.processing_time_ms
        }


# ============================================
# DOMAIN-SPECIFIC REGEX PATTERNS
# ============================================

# Import FANUC patterns if available
try:
    from .schemas.fanuc_schema import (
        FANUC_ERROR_PATTERNS,
        FANUC_MODEL_PATTERNS,
        FANUC_CONTROLLER_PATTERNS,
        FANUC_COMPONENT_PATTERNS,
        FANUC_PARAMETER_PATTERNS,
        FANUC_IO_PATTERNS,
        FANUC_REGISTER_PATTERNS,
        FANUC_PROCEDURE_PATTERNS,
        FANUC_MEASUREMENT_PATTERNS,
        FANUC_PART_PATTERNS,
        FANUC_SAFETY_PATTERNS,
    )
    FANUC_PATTERNS_AVAILABLE = True
except ImportError:
    FANUC_PATTERNS_AVAILABLE = False
    FANUC_ERROR_PATTERNS = []
    FANUC_MODEL_PATTERNS = []
    FANUC_CONTROLLER_PATTERNS = []
    FANUC_COMPONENT_PATTERNS = []
    FANUC_PARAMETER_PATTERNS = []
    FANUC_IO_PATTERNS = []
    FANUC_REGISTER_PATTERNS = []
    FANUC_PROCEDURE_PATTERNS = []
    FANUC_MEASUREMENT_PATTERNS = []
    FANUC_PART_PATTERNS = []
    FANUC_SAFETY_PATTERNS = []

# Generic industrial patterns (cross-domain)
GENERIC_ERROR_PATTERNS = [
    r"\b[A-Z]{2,5}-\d{3,4}\b",           # Generic error code format
    r"\b(?:ERR|ERROR|FAULT|ALARM)\s*#?\d{1,5}\b",  # ERR-001, FAULT 123
    r"\bE\d{4,5}\b",                      # E0001, E12345
]

GENERIC_COMPONENT_PATTERNS = [
    (r"\b(?:servo|motor|encoder|sensor|actuator|valve|pump|drive)\b", "component"),
    (r"\b(?:PLC|HMI|VFD|I/O|relay|contactor|breaker)\b", "controller"),
    (r"\b(?:cable|connector|wire|terminal|fuse)\b", "wiring"),
]

GENERIC_MEASUREMENT_PATTERNS = [
    r"\d+\.?\d*\s*(?:mm|cm|m|in|ft)\b",   # Length
    r"\d+\.?\d*\s*(?:deg|°|rad)\b",        # Angle
    r"\d+\.?\d*\s*(?:V|A|mA|kW|W|Ω)\b",    # Electrical
    r"\d+\.?\d*\s*(?:Hz|kHz|MHz)\b",       # Frequency
    r"\d+\.?\d*\s*(?:psi|bar|Pa|kPa)\b",   # Pressure
    r"\d+\.?\d*\s*(?:°C|°F|K)\b",          # Temperature
    r"\d+\.?\d*\s*(?:rpm|RPS)\b",          # Speed
    r"\d+\.?\d*\s*%\b",                    # Percentage
]


# ============================================
# GLINER LABEL DEFINITIONS
# ============================================

# Labels for GLiNER zero-shot extraction
# These are entities that are harder to capture with regex (variable phrasing)

GLINER_LABELS_INDUSTRIAL = [
    # Symptoms (how problems manifest)
    "symptom",
    "malfunction",
    "abnormal_behavior",

    # Causes (root cause of issues)
    "cause",
    "failure_mode",
    "defect",

    # Solutions (how to fix)
    "solution",
    "repair_action",
    "replacement",

    # Components (when phrased variably)
    "component_name",
    "part_name",

    # Technical actions
    "procedure_step",
    "calibration_action",
    "adjustment",

    # Safety-related
    "hazard",
    "safety_requirement",
    "protective_action",
]

GLINER_LABELS_FANUC = [
    # FANUC-specific variable entities
    "FANUC_symptom",
    "FANUC_cause",
    "FANUC_solution",
    "axis_movement",
    "robot_behavior",
    "alarm_condition",
    "position_error",
    "velocity_error",
    "torque_condition",
    "communication_issue",
]


# ============================================
# PATTERN REGISTRY
# ============================================

@dataclass
class PatternDefinition:
    """Definition of a regex pattern for entity extraction."""
    pattern: str
    entity_type: str
    category: EntityCategory
    priority: int = 0  # Higher = more specific, wins in conflicts


class PatternRegistry:
    """Registry of domain-specific patterns."""

    def __init__(self):
        self.patterns: List[PatternDefinition] = []
        self._compiled: Optional[Dict[str, Pattern]] = None

    def register(
        self,
        pattern: str,
        entity_type: str,
        category: EntityCategory,
        priority: int = 0
    ):
        """Register a pattern."""
        self.patterns.append(PatternDefinition(
            pattern=pattern,
            entity_type=entity_type,
            category=category,
            priority=priority
        ))
        self._compiled = None  # Invalidate cache

    def register_all(
        self,
        patterns: List[str],
        entity_type: str,
        category: EntityCategory,
        priority: int = 0
    ):
        """Register multiple patterns for same entity type."""
        for pattern in patterns:
            self.register(pattern, entity_type, category, priority)

    def get_compiled(self) -> Dict[str, Tuple[Pattern, PatternDefinition]]:
        """Get compiled patterns."""
        if self._compiled is None:
            self._compiled = {}
            for pdef in self.patterns:
                try:
                    compiled = re.compile(pdef.pattern, re.IGNORECASE)
                    key = f"{pdef.entity_type}_{len(self._compiled)}"
                    self._compiled[key] = (compiled, pdef)
                except re.error as e:
                    logger.warning(f"Invalid pattern '{pdef.pattern}': {e}")
        return self._compiled


def create_fanuc_registry() -> PatternRegistry:
    """Create pattern registry for FANUC domain."""
    registry = PatternRegistry()

    if FANUC_PATTERNS_AVAILABLE:
        # Error codes (highest priority)
        registry.register_all(
            FANUC_ERROR_PATTERNS,
            "error_code",
            EntityCategory.ERROR,
            priority=100
        )

        # Model patterns
        registry.register_all(
            FANUC_MODEL_PATTERNS,
            "robot_model",
            EntityCategory.IDENTIFIER,
            priority=90
        )

        # Controller patterns
        registry.register_all(
            FANUC_CONTROLLER_PATTERNS,
            "controller",
            EntityCategory.COMPONENT,
            priority=90
        )

        # Component patterns
        for pattern, subtype in FANUC_COMPONENT_PATTERNS:
            registry.register(
                pattern,
                f"component_{subtype}",
                EntityCategory.COMPONENT,
                priority=70
            )

        # Parameter patterns
        registry.register_all(
            FANUC_PARAMETER_PATTERNS,
            "parameter",
            EntityCategory.IDENTIFIER,
            priority=80
        )

        # I/O patterns
        registry.register_all(
            FANUC_IO_PATTERNS,
            "io_signal",
            EntityCategory.IDENTIFIER,
            priority=80
        )

        # Register patterns
        registry.register_all(
            FANUC_REGISTER_PATTERNS,
            "register",
            EntityCategory.IDENTIFIER,
            priority=80
        )

        # Procedure patterns
        for pattern, subtype in FANUC_PROCEDURE_PATTERNS:
            registry.register(
                pattern,
                f"procedure_{subtype}",
                EntityCategory.PROCEDURE,
                priority=60
            )

        # Measurement patterns
        registry.register_all(
            FANUC_MEASUREMENT_PATTERNS,
            "measurement",
            EntityCategory.MEASUREMENT,
            priority=50
        )

        # Part number patterns
        registry.register_all(
            FANUC_PART_PATTERNS,
            "part_number",
            EntityCategory.IDENTIFIER,
            priority=85
        )

        # Safety patterns
        registry.register_all(
            FANUC_SAFETY_PATTERNS,
            "safety_feature",
            EntityCategory.SAFETY,
            priority=75
        )

    return registry


def create_generic_registry() -> PatternRegistry:
    """Create pattern registry for generic industrial domain."""
    registry = PatternRegistry()

    # Generic error patterns
    registry.register_all(
        GENERIC_ERROR_PATTERNS,
        "error_code",
        EntityCategory.ERROR,
        priority=90
    )

    # Generic component patterns
    for pattern, subtype in GENERIC_COMPONENT_PATTERNS:
        registry.register(
            pattern,
            f"component_{subtype}",
            EntityCategory.COMPONENT,
            priority=60
        )

    # Generic measurement patterns
    registry.register_all(
        GENERIC_MEASUREMENT_PATTERNS,
        "measurement",
        EntityCategory.MEASUREMENT,
        priority=50
    )

    return registry


# ============================================
# GLINER HYBRID EXTRACTOR
# ============================================

class GLiNERHybridExtractor:
    """
    Hybrid entity extractor combining regex patterns and GLiNER.

    Strategy:
    1. Extract entities using regex patterns (high precision)
    2. Extract entities using GLiNER (high recall for variable entities)
    3. Merge with priority to regex on overlapping spans
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        gliner_threshold: float = 0.5,
        enable_gliner: bool = True,
        context_window: int = 50,
        domains: Optional[List[str]] = None
    ):
        """
        Initialize hybrid extractor.

        Args:
            model_name: GLiNER model to use
            gliner_threshold: Minimum confidence for GLiNER entities
            enable_gliner: Whether to use GLiNER (can be disabled for speed)
            context_window: Characters of context to capture around entities
            domains: List of domains to enable ("fanuc", "generic", etc.)
        """
        self.model_name = model_name
        self.gliner_threshold = gliner_threshold
        self.enable_gliner = enable_gliner and GLINER_AVAILABLE
        self.context_window = context_window

        # Initialize registries based on domains
        domains = domains or ["fanuc", "generic"]
        self.registries: Dict[str, PatternRegistry] = {}

        if "fanuc" in domains:
            self.registries["fanuc"] = create_fanuc_registry()
        if "generic" in domains:
            self.registries["generic"] = create_generic_registry()

        # Combined registry for extraction
        self._combined_registry = PatternRegistry()
        for registry in self.registries.values():
            self._combined_registry.patterns.extend(registry.patterns)

        # GLiNER model (lazy loaded)
        self._gliner_model: Optional[GLiNER] = None

        # GLiNER labels based on domains
        self.gliner_labels: List[str] = GLINER_LABELS_INDUSTRIAL.copy()
        if "fanuc" in domains:
            self.gliner_labels.extend(GLINER_LABELS_FANUC)

        # Statistics
        self._total_extractions = 0
        self._regex_entity_count = 0
        self._gliner_entity_count = 0

        logger.info(
            f"GLiNERHybridExtractor initialized: "
            f"gliner_enabled={self.enable_gliner}, "
            f"domains={domains}, "
            f"patterns={len(self._combined_registry.patterns)}"
        )

    def _get_gliner_model(self) -> Optional[GLiNER]:
        """Lazy load GLiNER model."""
        if not self.enable_gliner:
            return None

        if self._gliner_model is None:
            try:
                logger.info(f"Loading GLiNER model: {self.model_name}")
                self._gliner_model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GLiNER model: {e}")
                self.enable_gliner = False
                return None

        return self._gliner_model

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around entity."""
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        return text[ctx_start:ctx_end]

    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities: List[ExtractedEntity] = []

        for key, (pattern, pdef) in self._combined_registry.get_compiled().items():
            for match in pattern.finditer(text):
                entity = ExtractedEntity(
                    text=match.group(),
                    entity_type=pdef.entity_type,
                    category=pdef.category,
                    start=match.start(),
                    end=match.end(),
                    source=EntitySource.REGEX,
                    confidence=1.0,
                    context=self._extract_context(text, match.start(), match.end()),
                    metadata={"priority": pdef.priority}
                )
                entities.append(entity)

        return entities

    def _extract_with_gliner(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using GLiNER."""
        if not self.enable_gliner:
            return []

        model = self._get_gliner_model()
        if model is None:
            return []

        entities: List[ExtractedEntity] = []

        try:
            # GLiNER prediction
            predictions = model.predict_entities(
                text,
                self.gliner_labels,
                threshold=self.gliner_threshold
            )

            for pred in predictions:
                # Map GLiNER labels to categories
                label = pred["label"]
                category = self._map_label_to_category(label)

                entity = ExtractedEntity(
                    text=pred["text"],
                    entity_type=label,
                    category=category,
                    start=pred["start"],
                    end=pred["end"],
                    source=EntitySource.GLINER,
                    confidence=pred["score"],
                    context=self._extract_context(text, pred["start"], pred["end"]),
                    metadata={"gliner_label": label}
                )
                entities.append(entity)

        except Exception as e:
            logger.error(f"GLiNER extraction failed: {e}")

        return entities

    def _map_label_to_category(self, label: str) -> EntityCategory:
        """Map GLiNER label to entity category."""
        label_lower = label.lower()

        if any(x in label_lower for x in ["symptom", "malfunction", "abnormal", "behavior"]):
            return EntityCategory.SYMPTOM
        if any(x in label_lower for x in ["cause", "failure", "defect"]):
            return EntityCategory.CAUSE
        if any(x in label_lower for x in ["solution", "repair", "replacement", "fix"]):
            return EntityCategory.SOLUTION
        if any(x in label_lower for x in ["component", "part"]):
            return EntityCategory.COMPONENT
        if any(x in label_lower for x in ["procedure", "calibration", "adjustment"]):
            return EntityCategory.PROCEDURE
        if any(x in label_lower for x in ["hazard", "safety", "protective"]):
            return EntityCategory.SAFETY
        if any(x in label_lower for x in ["error", "alarm"]):
            return EntityCategory.ERROR

        return EntityCategory.OTHER

    def _spans_overlap(self, e1: ExtractedEntity, e2: ExtractedEntity) -> bool:
        """Check if two entity spans overlap."""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    def _merge_entities(
        self,
        regex_entities: List[ExtractedEntity],
        gliner_entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Merge entities with priority to regex.

        Strategy:
        - Regex entities always kept (100% precision)
        - GLiNER entities only kept if they don't overlap with regex
        - For overlapping GLiNER entities, prefer higher confidence
        """
        merged: List[ExtractedEntity] = []

        # Add all regex entities
        for entity in regex_entities:
            merged.append(entity)

        # Add non-overlapping GLiNER entities
        for gliner_entity in gliner_entities:
            overlaps_with_regex = any(
                self._spans_overlap(gliner_entity, regex_entity)
                for regex_entity in regex_entities
            )

            if not overlaps_with_regex:
                # Check if it overlaps with already-added GLiNER entities
                overlaps_with_merged = any(
                    self._spans_overlap(gliner_entity, merged_entity)
                    for merged_entity in merged
                    if merged_entity.source == EntitySource.GLINER
                )

                if not overlaps_with_merged:
                    merged.append(gliner_entity)
                else:
                    # Compare with overlapping GLiNER entity, keep higher confidence
                    for i, merged_entity in enumerate(merged):
                        if (merged_entity.source == EntitySource.GLINER and
                            self._spans_overlap(gliner_entity, merged_entity)):
                            if gliner_entity.confidence > merged_entity.confidence:
                                merged[i] = gliner_entity
                            break

        # Sort by position
        merged.sort(key=lambda e: (e.start, -e.confidence))

        return merged

    async def extract(
        self,
        text: str,
        use_gliner: Optional[bool] = None
    ) -> ExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Text to extract from
            use_gliner: Override global enable_gliner setting

        Returns:
            ExtractionResult with all entities
        """
        import time
        start_time = time.time()

        self._total_extractions += 1

        # Extract with regex
        regex_entities = self._extract_with_regex(text)
        self._regex_entity_count += len(regex_entities)

        # Extract with GLiNER (if enabled)
        should_use_gliner = use_gliner if use_gliner is not None else self.enable_gliner
        gliner_entities = []

        if should_use_gliner:
            # Run GLiNER in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            gliner_entities = await loop.run_in_executor(
                None,
                self._extract_with_gliner,
                text
            )
            self._gliner_entity_count += len(gliner_entities)

        # Merge entities
        merged = self._merge_entities(regex_entities, gliner_entities)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            entities=merged,
            text=text,
            regex_count=len(regex_entities),
            gliner_count=len(gliner_entities),
            merged_count=len(merged),
            processing_time_ms=processing_time
        )

    def extract_sync(self, text: str, use_gliner: Optional[bool] = None) -> ExtractionResult:
        """Synchronous extraction (for non-async contexts)."""
        import time
        start_time = time.time()

        self._total_extractions += 1

        # Extract with regex
        regex_entities = self._extract_with_regex(text)
        self._regex_entity_count += len(regex_entities)

        # Extract with GLiNER
        should_use_gliner = use_gliner if use_gliner is not None else self.enable_gliner
        gliner_entities = []

        if should_use_gliner:
            gliner_entities = self._extract_with_gliner(text)
            self._gliner_entity_count += len(gliner_entities)

        # Merge entities
        merged = self._merge_entities(regex_entities, gliner_entities)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            entities=merged,
            text=text,
            regex_count=len(regex_entities),
            gliner_count=len(gliner_entities),
            merged_count=len(merged),
            processing_time_ms=processing_time
        )

    async def extract_batch(
        self,
        texts: List[str],
        use_gliner: Optional[bool] = None
    ) -> List[ExtractionResult]:
        """Extract from multiple texts."""
        results = []
        for text in texts:
            result = await self.extract(text, use_gliner)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "total_extractions": self._total_extractions,
            "regex_entity_count": self._regex_entity_count,
            "gliner_entity_count": self._gliner_entity_count,
            "gliner_available": GLINER_AVAILABLE,
            "gliner_enabled": self.enable_gliner,
            "model_name": self.model_name,
            "gliner_threshold": self.gliner_threshold,
            "pattern_count": len(self._combined_registry.patterns),
            "gliner_labels": len(self.gliner_labels),
            "domains": list(self.registries.keys())
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self._total_extractions = 0
        self._regex_entity_count = 0
        self._gliner_entity_count = 0


# ============================================
# SPECIALIZED EXTRACTORS
# ============================================

class TroubleshootingExtractor(GLiNERHybridExtractor):
    """
    Specialized extractor for troubleshooting documents.

    Focuses on extracting:
    - Error codes (regex)
    - Symptoms (GLiNER)
    - Causes (GLiNER)
    - Solutions (GLiNER)
    - Components (hybrid)
    """

    def __init__(self, **kwargs):
        # Enhanced GLiNER labels for troubleshooting
        kwargs.setdefault("domains", ["fanuc", "generic"])
        super().__init__(**kwargs)

        # Add troubleshooting-specific labels
        self.gliner_labels = [
            "symptom",
            "cause",
            "solution",
            "error_condition",
            "repair_action",
            "replacement_part",
            "diagnostic_step",
            "warning_sign",
            "preventive_action",
        ]

    async def extract_troubleshooting_chain(
        self,
        text: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Extract and organize entities into troubleshooting chain.

        Returns:
            Dict with keys: error_codes, symptoms, causes, solutions, components
        """
        result = await self.extract(text)

        chain = {
            "error_codes": [],
            "symptoms": [],
            "causes": [],
            "solutions": [],
            "components": [],
            "procedures": [],
            "other": []
        }

        for entity in result.entities:
            if entity.category == EntityCategory.ERROR:
                chain["error_codes"].append(entity)
            elif entity.category == EntityCategory.SYMPTOM:
                chain["symptoms"].append(entity)
            elif entity.category == EntityCategory.CAUSE:
                chain["causes"].append(entity)
            elif entity.category == EntityCategory.SOLUTION:
                chain["solutions"].append(entity)
            elif entity.category == EntityCategory.COMPONENT:
                chain["components"].append(entity)
            elif entity.category == EntityCategory.PROCEDURE:
                chain["procedures"].append(entity)
            else:
                chain["other"].append(entity)

        return chain


# ============================================
# GLOBAL INSTANCES
# ============================================

_gliner_extractor: Optional[GLiNERHybridExtractor] = None
_troubleshooting_extractor: Optional[TroubleshootingExtractor] = None


def get_gliner_extractor(**kwargs) -> GLiNERHybridExtractor:
    """Get or create global GLiNER extractor."""
    global _gliner_extractor
    if _gliner_extractor is None:
        _gliner_extractor = GLiNERHybridExtractor(**kwargs)
    return _gliner_extractor


def get_troubleshooting_extractor(**kwargs) -> TroubleshootingExtractor:
    """Get or create global troubleshooting extractor."""
    global _troubleshooting_extractor
    if _troubleshooting_extractor is None:
        _troubleshooting_extractor = TroubleshootingExtractor(**kwargs)
    return _troubleshooting_extractor


def is_gliner_available() -> bool:
    """Check if GLiNER is available."""
    return GLINER_AVAILABLE
