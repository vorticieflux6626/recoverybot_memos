"""
HSEA Controller for memOS

Orchestrates existing memOS embedding systems to implement Hierarchical
Stratified Embedding Architecture (HSEA) semantics.

Mathematical Foundation (HSEA White Paper):
    S = (R^n, Pi, Phi) - Stratified embedding space

    pi_1 (Systemic):    Binary index + Anchors -> Categories, patterns
    pi_2 (Structural):  Int8 index + Memory graph -> Relationships, clusters
    pi_3 (Substantive): FP16 + Hybrid retrieval -> Content instances

Cross-Stratum Traversal Phi_ij implemented via:
    - Anchor-guided navigation (systemic -> structural)
    - Auto-connection graph (structural -> substantive)
    - MRL dimension truncation (substantive -> systemic)

Research References:
    - Kusupati et al., 2022: Matryoshka Representation Learning
    - Nickel & Kiela, 2017: Poincare Embeddings for Hierarchical Representations
    - Jacob et al., 2018: Quantization and Training of Neural Networks
    - Gao et al., 2022: HyDE - Hypothetical Document Embeddings
    - Xiao et al., 2023: BGE M3-Embedding
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class StratumType(str, Enum):
    """
    HSEA strata mapped to memOS precision levels.

    Mathematical correspondence:
        SYSTEMIC    -> pi_1: Binary (32x), MRL 64d
        STRUCTURAL  -> pi_2: Int8 (4x), MRL 256d
        SUBSTANTIVE -> pi_3: FP16 (1x), MRL 1024-4096d
    """
    SYSTEMIC = "systemic"         # Categories, patterns (Binary + Anchors)
    STRUCTURAL = "structural"     # Relationships, clusters (Int8 + Memory)
    SUBSTANTIVE = "substantive"   # Content instances (FP16 + Hybrid)


class HSEASearchMode(str, Enum):
    """Search modes with different stratum emphasis."""
    SYSTEMIC_ONLY = "systemic"           # Fast coarse (Binary index)
    STRUCTURAL_ONLY = "structural"       # Balanced (Int8 + graph)
    SUBSTANTIVE_ONLY = "substantive"     # Full precision (FP16 + Hybrid)
    CONTEXTUAL = "contextual"            # All strata + cross-traversal (recommended)
    MRL_HIERARCHICAL = "mrl"             # Progressive 64->256->1024->4096


@dataclass
class HSEAConfig:
    """
    HSEA configuration mapping to memOS systems.

    Dimension allocation follows HSEA White Paper Appendix B:
        pi_1: 170d (17%) - Systemic patterns
        pi_2: 171d (17%) - Structural relationships
        pi_3: 683d (66%) - Substantive content
        Total: 1024d

    Memory efficiency (vs uniform FP16):
        Mixed precision: 597 bytes/vector
        Uniform FP16:    2048 bytes/vector
        Compression:     3.4x
    """
    # Database paths
    embeddings_db: str = "data/hsea_embeddings.db"
    corpus_db: str = "data/hsea_corpus.db"
    hybrid_db: str = "data/hsea_hybrid.db"

    # MRL dimensions (Matryoshka Representation Learning)
    mrl_dimensions: List[int] = field(default_factory=lambda: [64, 256, 1024, 4096])

    # HSEA stratum dimensions (explicit partitioning)
    systemic_dims: int = 170      # pi_1: ~17%
    structural_dims: int = 171    # pi_2: ~17%
    substantive_dims: int = 683   # pi_3: ~66%
    total_dims: int = 1024

    # Search parameters (three-stage retrieval)
    binary_top_k: int = 500       # Stage 1: Coarse (Hamming distance)
    int8_top_k: int = 50          # Stage 2: Refined (Cosine similarity)
    fp16_top_k: int = 10          # Stage 3: Final (Full precision)

    # SemanticMemory auto-connection threshold
    connection_threshold: float = 0.7

    # Embedding model
    embedding_model: str = "qwen3-embedding:latest"
    ollama_url: str = "http://localhost:11434"

    # HyDE configuration
    enable_hyde: bool = True
    hyde_document_type: str = "TECHNICAL"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CategoryAnchor:
    """
    Layer 1 (Systemic) anchor embedding for category-guided search.

    Mathematical role: Reference frame in systemic stratum pi_1
    """
    category: str                    # e.g., "SRVO"
    name: str                        # e.g., "Servo Alarms"
    description: str
    embedding: Optional[np.ndarray] = None
    error_count: int = 0
    typical_causes: List[str] = field(default_factory=list)
    typical_remedies: List[str] = field(default_factory=list)


@dataclass
class TroubleshootingPattern:
    """
    Layer 1 (Systemic) troubleshooting strategy pattern.

    Mathematical role: Abstract procedure template in pi_1
    """
    pattern_id: str                  # e.g., "pattern:encoder_replacement"
    name: str
    description: str
    steps: List[str] = field(default_factory=list)
    applicable_categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class ErrorCodeEntity:
    """
    Layer 3 (Substantive) error code entity.

    Maps directly to existing EntityNode in graph.pkl.
    """
    entity_id: str                   # e.g., "error_code_srvo_063"
    canonical_form: str              # e.g., "SRVO-063"
    title: str
    category: str
    code_number: int
    cause: str
    remedy: str
    severity: str
    related_codes: List[str] = field(default_factory=list)
    page_number: Optional[int] = None

    # HSEA enhancements
    cluster_id: Optional[str] = None
    matched_patterns: List[str] = field(default_factory=list)
    memory_node_id: Optional[str] = None


@dataclass
class CrossStratumContext:
    """
    Complete context from all three HSEA strata.
    """
    entity: ErrorCodeEntity
    score: float

    # Layer 1 (Systemic/pi_1) context
    category_anchor: Optional[CategoryAnchor] = None
    troubleshooting_patterns: List[TroubleshootingPattern] = field(default_factory=list)
    mrl_64d_score: float = 0.0

    # Layer 2 (Structural/pi_2) context
    related_codes: List[ErrorCodeEntity] = field(default_factory=list)
    cluster_members: List[ErrorCodeEntity] = field(default_factory=list)
    auto_connections: List[Dict] = field(default_factory=list)
    mrl_256d_score: float = 0.0

    # Layer 3 (Substantive/pi_3) context
    similar_by_cause: List[ErrorCodeEntity] = field(default_factory=list)
    similar_by_remedy: List[ErrorCodeEntity] = field(default_factory=list)
    bm25_matched_terms: List[str] = field(default_factory=list)
    dense_score: float = 0.0
    mrl_1024d_score: float = 0.0


@dataclass
class HSEASearchResult:
    """Complete HSEA search result with multi-stratum analysis."""
    query: str
    search_time_ms: float
    mode: HSEASearchMode

    results: List[CrossStratumContext]
    dominant_categories: List[str] = field(default_factory=list)
    suggested_patterns: List[TroubleshootingPattern] = field(default_factory=list)

    # MRL progression statistics
    mrl_progression: Dict[int, int] = field(default_factory=dict)

    # Three-stage retrieval statistics
    binary_candidates: int = 0
    int8_candidates: int = 0
    fp16_results: int = 0


# =============================================================================
# TROUBLESHOOTING PATTERNS (LAYER 1 KNOWLEDGE)
# =============================================================================

TROUBLESHOOTING_PATTERNS: Dict[str, Dict] = {
    "pattern:encoder_replacement": {
        "name": "Encoder Replacement",
        "description": "Pulsecoder/encoder replacement for servo alarms",
        "steps": [
            "Power off robot controller",
            "Replace encoder/pulsecoder unit",
            "Perform mastering procedure",
            "Execute RCAL calibration"
        ],
        "categories": ["SRVO"],
        "keywords": ["encoder", "pulsecoder", "RCAL", "mastering", "pulse", "rotation counter"]
    },
    "pattern:calibration": {
        "name": "Robot Calibration",
        "description": "Calibration and mastering for position errors",
        "steps": [
            "Perform zero return to reference position",
            "Set mechanical origin point",
            "Execute mastering procedure",
            "Verify all axis positions"
        ],
        "categories": ["SRVO", "MOTN"],
        "keywords": ["calibration", "mastering", "zero", "origin", "position", "reference"]
    },
    "pattern:communication_reset": {
        "name": "Communication Reset",
        "description": "Network/communication reset for HOST/COMM errors",
        "steps": [
            "Check physical cable connections",
            "Verify IP settings and configuration",
            "Cycle controller power",
            "Re-establish communication link"
        ],
        "categories": ["HOST", "COMM"],
        "keywords": ["communication", "network", "timeout", "connection", "ethernet", "IP"]
    },
    "pattern:parameter_adjustment": {
        "name": "Parameter Adjustment",
        "description": "System parameter modification for tuning issues",
        "steps": [
            "Backup current parameters",
            "Identify target parameter from documentation",
            "Adjust parameter value",
            "Test and verify operation"
        ],
        "categories": ["SYST", "SVGN", "MOTN"],
        "keywords": ["parameter", "setting", "adjustment", "value", "$PARAM", "variable"]
    },
    "pattern:safety_interlock": {
        "name": "Safety Interlock Check",
        "description": "Safety circuit verification for critical alarms",
        "steps": [
            "Check E-stop button status",
            "Verify safety fence sensor connections",
            "Check DCS configuration",
            "Reset safety circuit and test"
        ],
        "categories": ["SYST", "PRIO"],
        "keywords": ["safety", "interlock", "e-stop", "fence", "DCS", "emergency"]
    },
    "pattern:servo_power_cycle": {
        "name": "Servo Power Reset",
        "description": "Servo amplifier power cycle for transient errors",
        "steps": [
            "Turn off servo power via teach pendant",
            "Wait 30 seconds for capacitor discharge",
            "Turn on servo power",
            "Check alarm history for persistence"
        ],
        "categories": ["SRVO", "SVGN"],
        "keywords": ["servo", "power", "amplifier", "drive", "motor", "cycle"]
    },
    "pattern:vision_calibration": {
        "name": "Vision System Calibration",
        "description": "Camera calibration for iRVision errors",
        "steps": [
            "Clean camera lens",
            "Check lighting conditions",
            "Run camera calibration routine",
            "Update vision frame offsets"
        ],
        "categories": ["CVIS"],
        "keywords": ["vision", "camera", "calibration", "iRVision", "lens", "lighting"]
    },
}

# Category descriptions for anchor creation
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "SRVO": "Servo motor and drive system alarms including encoder, amplifier, and motor issues",
    "MOTN": "Motion control and trajectory alarms including path, speed, and position errors",
    "SYST": "System-level and controller alarms including hardware and software faults",
    "INTP": "Program interpreter and execution alarms including syntax and runtime errors",
    "HOST": "Host communication and external device alarms including Ethernet and fieldbus",
    "PRIO": "Priority and scheduling alarms including task management errors",
    "COMM": "Communication network alarms including protocol and timeout errors",
    "CVIS": "iRVision and camera system alarms including calibration and detection errors",
    "SVGN": "Servo gain and tuning alarms including parameter and optimization errors",
}


# =============================================================================
# HSEA CONTROLLER
# =============================================================================

class HSEAController:
    """
    Orchestrates memOS embedding systems to implement HSEA semantics.

    Maps HSEA three-layer architecture to:
        pi_1 (Systemic)    -> MixedPrecisionService.search() + anchors
        pi_2 (Structural)  -> SemanticMemoryNetwork + auto-connections
        pi_3 (Substantive) -> DomainCorpus + BGEM3Hybrid + HyDE

    Cross-stratum traversal Phi_ij via:
        Phi_12: Anchor -> related entities in memory graph
        Phi_23: Auto-connections -> substantive content
        Phi_31: MRL truncation (4096d -> 64d)
    """

    def __init__(self, config: Optional[HSEAConfig] = None):
        self.config = config or HSEAConfig()

        # Lazy-loaded memOS systems
        self._mp_service = None
        self._memory_network = None
        self._corpus = None
        self._corpus_retriever = None
        self._hybrid_retriever = None
        self._hyde_expander = None
        self._aggregator = None

        # HSEA-specific storage
        self.category_anchors: Dict[str, CategoryAnchor] = {}
        self.troubleshooting_patterns: Dict[str, TroubleshootingPattern] = {}
        self.entities: Dict[str, ErrorCodeEntity] = {}

        # Initialize patterns
        self._init_troubleshooting_patterns()

        # Load entities from database on startup
        self._load_entities_from_db()

        logger.info("HSEAController initialized")

    def _load_entities_from_db(self):
        """
        Load entities from hybrid database on startup.

        Parses stored content back into ErrorCodeEntity objects.
        This ensures entities persist across server restarts.
        """
        import sqlite3
        import re
        import os

        # Resolve relative path from module location
        db_path = self.config.hybrid_db
        if not os.path.isabs(db_path):
            # Resolve relative to the server directory (parent of agentic/)
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(module_dir, db_path)

        logger.debug(f"Loading HSEA entities from: {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if documents table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            if not cursor.fetchone():
                logger.info("No documents table found in HSEA database - starting fresh")
                conn.close()
                return

            # Load all documents
            cursor.execute("SELECT doc_id, content FROM documents")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                logger.info("No documents in HSEA database - starting fresh")
                return

            loaded = 0
            for doc_id, content in rows:
                try:
                    entity = self._parse_document_to_entity(doc_id, content)
                    if entity:
                        self.entities[doc_id] = entity

                        # Update category anchor
                        category = entity.category
                        if category not in self.category_anchors:
                            description = CATEGORY_DESCRIPTIONS.get(
                                category,
                                f"{category} category error codes and alarms"
                            )
                            self.category_anchors[category] = CategoryAnchor(
                                category=category,
                                name=f"{category} Alarms",
                                description=description,
                                error_count=1
                            )
                        else:
                            self.category_anchors[category].error_count += 1

                        # Match patterns
                        entity.matched_patterns = self._match_patterns(entity)
                        loaded += 1
                except Exception as e:
                    logger.debug(f"Failed to parse entity {doc_id}: {e}")

            logger.info(f"Loaded {loaded} entities from HSEA database ({len(self.category_anchors)} categories)")

        except Exception as e:
            logger.warning(f"Failed to load entities from database: {e}")

    def _parse_document_to_entity(self, doc_id: str, content: str) -> Optional['ErrorCodeEntity']:
        """
        Parse stored document content back into ErrorCodeEntity.

        Expected format:
            Error Code: SRVO-063
            Title: SRVO-063 RCAL alarm
            Category: SRVO
            Cause: <cause text>
            Remedy: <remedy text>
        """
        import re

        lines = content.strip().split('\n')
        if len(lines) < 2:
            return None

        entity_data = {
            'entity_id': doc_id,
            'canonical_form': '',
            'title': '',
            'category': '',
            'cause': '',
            'remedy': ''
        }

        current_field = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Error Code:'):
                entity_data['canonical_form'] = line.replace('Error Code:', '').strip()
            elif line.startswith('Title:'):
                entity_data['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Category:'):
                entity_data['category'] = line.replace('Category:', '').strip()
            elif line.startswith('Cause:'):
                entity_data['cause'] = line.replace('Cause:', '').strip()
                current_field = 'cause'
            elif line.startswith('Remedy:'):
                entity_data['remedy'] = line.replace('Remedy:', '').strip()
                current_field = 'remedy'
            elif current_field:
                # Continuation of previous field
                entity_data[current_field] += ' ' + line

        if not entity_data['canonical_form'] or not entity_data['category']:
            return None

        # Extract code number from canonical form (e.g., "SRVO-063" -> 63)
        code_number = 0
        parts = entity_data['canonical_form'].split('-')
        if len(parts) >= 2:
            try:
                code_number = int(parts[-1])
            except ValueError:
                pass

        return ErrorCodeEntity(
            entity_id=entity_data['entity_id'],
            canonical_form=entity_data['canonical_form'],
            title=entity_data['title'] or entity_data['canonical_form'],
            category=entity_data['category'],
            code_number=code_number,
            cause=entity_data['cause'],
            remedy=entity_data['remedy'],
            severity='warn'  # Default severity
        )

    def _init_troubleshooting_patterns(self):
        """Initialize Layer 1 troubleshooting patterns."""
        for pattern_id, data in TROUBLESHOOTING_PATTERNS.items():
            self.troubleshooting_patterns[pattern_id] = TroubleshootingPattern(
                pattern_id=pattern_id,
                name=data["name"],
                description=data["description"],
                steps=data["steps"],
                applicable_categories=data["categories"],
                keywords=data["keywords"]
            )

    # =========================================================================
    # LAZY LOADING OF memOS SYSTEMS
    # =========================================================================

    @property
    def mp_service(self):
        """
        Layer 1+2: Mixed-Precision Embedding Service

        Provides:
            - Binary index (32x compression) -> pi_1 systemic search
            - Int8 index (4x compression) -> pi_2 structural search
            - FP16 store (full precision) -> pi_3 ranking
            - Anchor embeddings -> Category navigation
            - MRL search -> Hierarchical dimension progression
        """
        if self._mp_service is None:
            from .mixed_precision_embeddings import get_mixed_precision_service
            self._mp_service = get_mixed_precision_service()
        return self._mp_service

    @property
    def memory_network(self):
        """
        Layer 2: Semantic Memory Network

        Provides:
            - Auto-connection via embedding similarity (threshold 0.7)
            - 7 connection types for rich relationship semantics
            - Bidirectional traversal
            - Strength-weighted paths
        """
        if self._memory_network is None:
            from .semantic_memory import SemanticMemoryNetwork
            self._memory_network = SemanticMemoryNetwork()
        return self._memory_network

    @property
    def hybrid_retriever(self):
        """
        Layer 3: BGE-M3 Hybrid Retriever

        Provides:
            - Dense embeddings (BGE-M3 1024d) -> Semantic similarity
            - Sparse BM25 index (k1=1.5, b=0.75) -> Lexical matching
            - RRF fusion: score(d) = Sum 1/(60 + rank_i(d))
        """
        if self._hybrid_retriever is None:
            from .bge_m3_hybrid import BGEM3HybridRetriever
            self._hybrid_retriever = BGEM3HybridRetriever(
                db_path=self.config.hybrid_db
            )
        return self._hybrid_retriever

    @property
    def hyde_expander(self):
        """
        Layer 3 Enhancement: HyDE Query Expansion

        Provides:
            - Hypothetical document generation via LLM
            - +15-25% recall improvement
        """
        if self._hyde_expander is None:
            from .hyde import HyDEExpander
            self._hyde_expander = HyDEExpander()
        return self._hyde_expander

    # =========================================================================
    # INDEXING METHODS
    # =========================================================================

    async def index_error_code(self, entity: ErrorCodeEntity) -> Dict[str, Any]:
        """
        Index error code across all three HSEA strata.

        Mathematical operation:
            E -> [pi_1(E), pi_2(E), pi_3(E)] with cross-references Phi_ij
        """
        stats = {
            "entity_id": entity.entity_id,
            "layer_1_indexed": False,
            "layer_2_indexed": False,
            "layer_3_indexed": False,
            "patterns_matched": [],
            "connections_created": 0
        }

        # Build rich content for embedding
        content = f"""
Error Code: {entity.canonical_form}
Title: {entity.title}
Category: {entity.category}
Cause: {entity.cause}
Remedy: {entity.remedy}
"""

        # =====================================================================
        # Layer 3 (pi_3 Substantive): Full content indexing
        # =====================================================================

        try:
            # Mixed-precision storage (Binary + Int8 + FP16)
            await self.mp_service.index_document(
                doc_id=entity.entity_id,
                text=content,
                content=content,
                metadata={
                    "canonical_form": entity.canonical_form,
                    "category": entity.category,
                    "severity": entity.severity,
                    "code_number": entity.code_number
                }
            )

            # Hybrid retrieval (Dense + Sparse/BM25)
            await self.hybrid_retriever.add_document(
                doc_id=entity.entity_id,
                content=content,
                metadata={
                    "category": entity.category,
                    "page": entity.page_number
                }
            )

            stats["layer_3_indexed"] = True
        except Exception as e:
            logger.warning(f"Layer 3 indexing failed for {entity.entity_id}: {e}")

        # =====================================================================
        # Layer 2 (pi_2 Structural): Semantic memory with auto-connections
        # =====================================================================

        try:
            from .semantic_memory import MemoryType

            memory_result = await self.memory_network.add_memory(
                content=content,
                memory_type=MemoryType.ENTITY,
                attributes={
                    "error_code": entity.canonical_form,
                    "category": entity.category,
                    "related_codes": entity.related_codes
                }
            )

            if memory_result:
                entity.memory_node_id = memory_result.id
                stats["connections_created"] = len(memory_result.connections)
            stats["layer_2_indexed"] = True
        except Exception as e:
            logger.warning(f"Layer 2 indexing failed for {entity.entity_id}: {e}")

        # =====================================================================
        # Layer 1 (pi_1 Systemic): Pattern matching and category anchoring
        # =====================================================================

        try:
            # Match troubleshooting patterns
            matched_patterns = self._match_patterns(entity)
            entity.matched_patterns = matched_patterns
            stats["patterns_matched"] = matched_patterns

            # Update category anchor
            await self._update_category_anchor(entity)

            stats["layer_1_indexed"] = True
        except Exception as e:
            logger.warning(f"Layer 1 indexing failed for {entity.entity_id}: {e}")

        # Store entity reference
        self.entities[entity.entity_id] = entity

        logger.debug(f"Indexed {entity.canonical_form}: L1={stats['layer_1_indexed']}, "
                    f"L2={stats['layer_2_indexed']}, L3={stats['layer_3_indexed']}")

        return stats

    async def index_batch(self, entities: List[ErrorCodeEntity]) -> Dict[str, Any]:
        """Batch index error codes."""
        stats = {"indexed": 0, "failed": 0, "patterns_matched": 0}

        for entity in entities:
            try:
                result = await self.index_error_code(entity)
                stats["indexed"] += 1
                stats["patterns_matched"] += len(result["patterns_matched"])
            except Exception as e:
                logger.error(f"Failed to index {entity.entity_id}: {e}")
                stats["failed"] += 1

        logger.info(f"Batch indexed {stats['indexed']}/{len(entities)} entities")
        return stats

    async def index_loaded_entities(self, limit: int = 100, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Index entities that were loaded from database but not yet in embedding indices.

        Args:
            limit: Maximum number of entities to index (default 100)
            category_filter: Only index entities from this category (e.g., "SRVO")

        Returns:
            Statistics about indexing results
        """
        stats = {"indexed": 0, "failed": 0, "skipped": 0, "total_available": len(self.entities)}

        entities_to_index = []
        for entity_id, entity in self.entities.items():
            if category_filter and entity.category != category_filter:
                continue
            entities_to_index.append(entity)
            if len(entities_to_index) >= limit:
                break

        logger.info(f"Indexing {len(entities_to_index)} loaded entities (limit={limit}, filter={category_filter})")

        for entity in entities_to_index:
            try:
                await self.index_error_code(entity)
                stats["indexed"] += 1
            except Exception as e:
                logger.error(f"Failed to index loaded entity {entity.entity_id}: {e}")
                stats["failed"] += 1

        logger.info(f"Indexed {stats['indexed']}/{len(entities_to_index)} loaded entities")
        return stats

    def _match_patterns(self, entity: ErrorCodeEntity) -> List[str]:
        """Match troubleshooting patterns based on category and keywords."""
        matched = []
        text = f"{entity.cause} {entity.remedy}".lower()

        for pattern_id, pattern in self.troubleshooting_patterns.items():
            # Category match
            if entity.category in pattern.applicable_categories:
                if pattern_id not in matched:
                    matched.append(pattern_id)
                continue

            # Keyword match
            for keyword in pattern.keywords:
                if keyword.lower() in text:
                    if pattern_id not in matched:
                        matched.append(pattern_id)
                    break

        return matched

    async def _update_category_anchor(self, entity: ErrorCodeEntity):
        """Update or create category anchor in mixed-precision service."""
        category = entity.category

        if category not in self.category_anchors:
            description = CATEGORY_DESCRIPTIONS.get(
                category,
                f"{category} category error codes and alarms"
            )

            # Create anchor in MixedPrecisionService
            try:
                self.mp_service.register_anchor(
                    category=category.lower(),
                    embedding=None  # Will be created from examples
                )
            except Exception as e:
                logger.debug(f"Anchor registration: {e}")

            self.category_anchors[category] = CategoryAnchor(
                category=category,
                name=f"{category} Alarms",
                description=description,
                error_count=1
            )
        else:
            self.category_anchors[category].error_count += 1

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    async def search(
        self,
        query: str,
        mode: HSEASearchMode = HSEASearchMode.CONTEXTUAL,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        enable_hyde: Optional[bool] = None
    ) -> HSEASearchResult:
        """
        Multi-stratum semantic search.

        Modes:
            SYSTEMIC:     Binary index + anchors (fast, 32x compression)
            STRUCTURAL:   Int8 index + memory graph (balanced, 4x compression)
            SUBSTANTIVE:  FP16 + hybrid (full precision)
            CONTEXTUAL:   All strata with cross-traversal (recommended)
            MRL:          Progressive 64->256->1024->4096 refinement
        """
        start_time = time.time()

        enable_hyde = enable_hyde if enable_hyde is not None else self.config.enable_hyde

        # HyDE query expansion if enabled
        hyde_embedding = None
        if enable_hyde and mode in [HSEASearchMode.CONTEXTUAL, HSEASearchMode.SUBSTANTIVE_ONLY]:
            try:
                from .hyde import HyDEMode, DocumentType
                hyde_result = await self.hyde_expander.expand(
                    query=query,
                    document_type=DocumentType.TECHNICAL,
                    mode=HyDEMode.SINGLE
                )
                hyde_embedding = hyde_result.fused_embedding
            except Exception as e:
                logger.debug(f"HyDE expansion skipped: {e}")

        # Dispatch to appropriate search method
        mrl_stats = {}
        if mode == HSEASearchMode.MRL_HIERARCHICAL:
            results, mrl_stats = await self._search_mrl(query, top_k, category_filter)
        elif mode == HSEASearchMode.SYSTEMIC_ONLY:
            results = await self._search_systemic(query, top_k, category_filter)
        elif mode == HSEASearchMode.STRUCTURAL_ONLY:
            results = await self._search_structural(query, top_k, category_filter)
        elif mode == HSEASearchMode.SUBSTANTIVE_ONLY:
            results = await self._search_substantive(query, top_k, category_filter, hyde_embedding)
        else:  # CONTEXTUAL
            results, mrl_stats = await self._search_contextual(query, top_k, category_filter, hyde_embedding)

        # Aggregate statistics
        search_time = (time.time() - start_time) * 1000

        category_counts: Dict[str, int] = defaultdict(int)
        for ctx in results:
            category_counts[ctx.entity.category] += 1

        dominant_categories = sorted(
            category_counts.keys(),
            key=lambda c: category_counts[c],
            reverse=True
        )[:3]

        # Suggest patterns based on dominant categories
        suggested_patterns = []
        for pattern in self.troubleshooting_patterns.values():
            for cat in dominant_categories:
                if cat in pattern.applicable_categories:
                    if pattern not in suggested_patterns:
                        suggested_patterns.append(pattern)
                    break

        return HSEASearchResult(
            query=query,
            search_time_ms=search_time,
            mode=mode,
            results=results[:top_k],
            dominant_categories=dominant_categories,
            suggested_patterns=suggested_patterns[:3],
            mrl_progression=mrl_stats,
            binary_candidates=mrl_stats.get(64, 0),
            int8_candidates=mrl_stats.get(256, 0),
            fp16_results=len(results)
        )

    async def _search_mrl(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None
    ) -> Tuple[List[CrossStratumContext], Dict[int, int]]:
        """
        MRL hierarchical search: 64 -> 256 -> 1024 -> 4096 dimensions.
        """
        mrl_stats = {}

        try:
            results = await self.mp_service.mrl_hierarchical_search(
                query=query,
                top_k=top_k,
                instruction="Find error codes related to:",
                stages=self.config.mrl_dimensions
            )

            # Track candidates at each level
            for i, dim in enumerate(self.config.mrl_dimensions):
                mrl_stats[dim] = results[1].get(f"stage_{i}_candidates", 0) if len(results) > 1 else 0

            search_results = results[0] if isinstance(results, tuple) else results
        except Exception as e:
            logger.warning(f"MRL search failed: {e}")
            search_results = []

        # Build contexts
        contexts = []
        for result in search_results[:top_k]:
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id")
            score = result.relevance_score if hasattr(result, 'relevance_score') else result.get("score", 0.0)

            entity = self.entities.get(doc_id)
            if not entity:
                continue

            if category_filter and entity.category != category_filter.upper():
                continue

            ctx = await self._build_context(entity, score)
            contexts.append(ctx)

        return contexts, mrl_stats

    async def _search_systemic(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None
    ) -> List[CrossStratumContext]:
        """
        Layer 1 (pi_1) search: Fast coarse retrieval.
        """
        try:
            results, _ = await self.mp_service.search(
                query=query,
                top_k=top_k * 3,
                instruction="Find error codes:"
            )
        except Exception as e:
            logger.warning(f"Systemic search failed: {e}")
            results = []

        contexts = []
        for result in results:
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id")
            score = result.relevance_score if hasattr(result, 'relevance_score') else result.get("score", 0.0)

            entity = self.entities.get(doc_id)
            if not entity:
                continue

            if category_filter and entity.category != category_filter.upper():
                continue

            ctx = await self._build_context(entity, score)
            contexts.append(ctx)

        return contexts[:top_k]

    async def _search_structural(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None
    ) -> List[CrossStratumContext]:
        """
        Layer 2 (pi_2) search: Memory graph with auto-connections.
        """
        try:
            similar_memories = await self.memory_network.find_similar(
                query=query,
                top_k=top_k * 2
            )
        except Exception as e:
            logger.warning(f"Structural search failed: {e}")
            similar_memories = []

        contexts = []
        for memory, similarity in similar_memories:
            # Find entity by memory node
            entity = None
            for e in self.entities.values():
                if e.memory_node_id == memory.id:
                    entity = e
                    break

            if not entity:
                continue

            if category_filter and entity.category != category_filter.upper():
                continue

            ctx = await self._build_context(entity, similarity)
            ctx.auto_connections = [c.to_dict() for c in memory.connections[:5]]
            contexts.append(ctx)

        return contexts[:top_k]

    async def _search_substantive(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None,
        hyde_embedding: Optional[np.ndarray] = None
    ) -> List[CrossStratumContext]:
        """
        Layer 3 (pi_3) search: FP16 + BGE-M3 hybrid.
        """
        from .bge_m3_hybrid import RetrievalMode

        try:
            hybrid_results = await self.hybrid_retriever.search(
                query=query,
                top_k=top_k * 2,
                mode=RetrievalMode.HYBRID
            )
        except Exception as e:
            logger.warning(f"Substantive search failed: {e}")
            hybrid_results = []

        contexts = []
        for result in hybrid_results:
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id")
            score = result.combined_score if hasattr(result, 'combined_score') else result.get("combined_score", 0.0)

            entity = self.entities.get(doc_id)
            if not entity:
                continue

            if category_filter and entity.category != category_filter.upper():
                continue

            ctx = await self._build_context(entity, score)
            ctx.dense_score = result.dense_score if hasattr(result, 'dense_score') else 0.0
            contexts.append(ctx)

        return contexts[:top_k]

    async def _search_contextual(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None,
        hyde_embedding: Optional[np.ndarray] = None
    ) -> Tuple[List[CrossStratumContext], Dict[int, int]]:
        """
        Full contextual search across all HSEA strata.

        Three-stage retrieval pipeline:
            Stage 1: MixedPrecision coarse search
            Stage 2: SemanticMemory enrichment
            Stage 3: Hybrid fine ranking
        """
        mrl_stats = {}

        # Stage 1: Mixed-precision search
        try:
            mp_results, mp_stats = await self.mp_service.search(
                query=query,
                top_k=self.config.binary_top_k,
                instruction="Find error codes related to:"
            )
            mrl_stats[64] = len(mp_results)
        except Exception as e:
            logger.warning(f"Stage 1 (MP) failed: {e}")
            mp_results = []
            mrl_stats[64] = 0

        # Stage 2: Hybrid refinement
        try:
            from .bge_m3_hybrid import RetrievalMode
            hybrid_results = await self.hybrid_retriever.search(
                query=query,
                top_k=self.config.int8_top_k,
                mode=RetrievalMode.HYBRID
            )
            mrl_stats[256] = len(hybrid_results)
        except Exception as e:
            logger.warning(f"Stage 2 (Hybrid) failed: {e}")
            hybrid_results = []
            mrl_stats[256] = 0

        # Merge results with RRF
        merged = self._rrf_merge(mp_results, hybrid_results)
        mrl_stats[1024] = len(merged)

        # Build contexts with cross-stratum enrichment
        contexts = []
        for doc_id, rrf_score in merged[:top_k * 2]:
            entity = self.entities.get(doc_id)
            if not entity:
                continue

            if category_filter and entity.category != category_filter.upper():
                continue

            ctx = await self._build_context(entity, rrf_score)
            contexts.append(ctx)

        return contexts[:top_k], mrl_stats

    async def _build_context(
        self,
        entity: ErrorCodeEntity,
        score: float
    ) -> CrossStratumContext:
        """Build complete cross-stratum context for entity."""
        ctx = CrossStratumContext(entity=entity, score=score)

        # Layer 1: Category anchor and patterns
        if entity.category in self.category_anchors:
            ctx.category_anchor = self.category_anchors[entity.category]

        for pattern_id in entity.matched_patterns:
            if pattern_id in self.troubleshooting_patterns:
                ctx.troubleshooting_patterns.append(self.troubleshooting_patterns[pattern_id])

        # Layer 2: Related codes
        for related_code in entity.related_codes[:5]:
            for e in self.entities.values():
                if e.canonical_form == related_code:
                    ctx.related_codes.append(e)
                    break

        # Layer 2: Cluster members (same category)
        if not ctx.cluster_members:
            for e in self.entities.values():
                if e.category == entity.category and e.entity_id != entity.entity_id:
                    ctx.cluster_members.append(e)
                    if len(ctx.cluster_members) >= 5:
                        break

        return ctx

    def _rrf_merge(
        self,
        results1: List,
        results2: List
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        Formula: score(d) = Sum 1/(k + rank_i(d))
        where k=60 (standard constant)
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        k = 60

        for rank, result in enumerate(results1):
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id", str(rank))
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        for rank, result in enumerate(results2):
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id", str(rank))
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # =========================================================================
    # TRAVERSAL METHODS
    # =========================================================================

    async def get_troubleshooting_context(
        self,
        error_code: str
    ) -> Optional[CrossStratumContext]:
        """
        Get complete troubleshooting context for error code.

        Cross-stratum traversal: pi_3 -> pi_2 -> pi_1
        """
        entity = None
        for e in self.entities.values():
            if e.canonical_form == error_code.upper():
                entity = e
                break

        if not entity:
            return None

        ctx = await self._build_context(entity, 1.0)

        # Enrich with semantic memory traversal
        if entity.memory_node_id:
            try:
                from .semantic_memory import ConnectionType
                traversal = await self.memory_network.traverse(
                    start_id=entity.memory_node_id,
                    max_hops=2,
                    connection_types=[
                        ConnectionType.SEMANTIC,
                        ConnectionType.RELATED
                    ]
                )
                if traversal:
                    ctx.auto_connections = [
                        {"path": t.path, "strength": t.total_strength}
                        for t in traversal[:5]
                    ]
            except Exception as e:
                logger.debug(f"Traversal enrichment skipped: {e}")

        return ctx

    async def find_similar(
        self,
        error_code: str,
        top_k: int = 5
    ) -> List[CrossStratumContext]:
        """
        Find similar error codes using substantive similarity.
        """
        entity = None
        for e in self.entities.values():
            if e.canonical_form == error_code.upper():
                entity = e
                break

        if not entity:
            return []

        # Search by entity content
        content = f"{entity.cause} {entity.remedy}"

        try:
            results, _ = await self.mp_service.search(
                query=content,
                top_k=top_k + 1,
                instruction="Find similar error codes:"
            )
        except Exception as e:
            logger.warning(f"Similar search failed: {e}")
            return []

        contexts = []
        for result in results:
            doc_id = result.doc_id if hasattr(result, 'doc_id') else result.get("doc_id")
            if doc_id == entity.entity_id:
                continue

            score = result.relevance_score if hasattr(result, 'relevance_score') else result.get("score", 0.0)
            similar_entity = self.entities.get(doc_id)
            if similar_entity:
                ctx = await self._build_context(similar_entity, score)
                contexts.append(ctx)

        return contexts[:top_k]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get HSEA system statistics."""
        return {
            "total_entities": len(self.entities),
            "total_categories": len(self.category_anchors),
            "total_patterns": len(self.troubleshooting_patterns),
            "category_counts": {
                cat: anchor.error_count
                for cat, anchor in self.category_anchors.items()
            },
            "pattern_matches": sum(
                len(e.matched_patterns) for e in self.entities.values()
            ),
            "config": {
                "mrl_dimensions": self.config.mrl_dimensions,
                "binary_top_k": self.config.binary_top_k,
                "int8_top_k": self.config.int8_top_k,
                "fp16_top_k": self.config.fp16_top_k,
                "connection_threshold": self.config.connection_threshold,
                "enable_hyde": self.config.enable_hyde
            }
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_hsea_controller: Optional[HSEAController] = None


def get_hsea_controller(config: Optional[HSEAConfig] = None) -> HSEAController:
    """Get or create singleton HSEA controller."""
    global _hsea_controller
    if _hsea_controller is None:
        _hsea_controller = HSEAController(config)
    return _hsea_controller
