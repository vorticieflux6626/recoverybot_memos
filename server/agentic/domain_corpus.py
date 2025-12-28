"""
Domain-Specific Persistent Scratchpad System

A general-purpose framework for building domain-specific knowledge corpuses
that persist across sessions. Designed for technical troubleshooting domains
like FANUC robotics, Raspberry Pi, industrial equipment, etc.

Key Features:
- Domain-agnostic schema definition (customize for any domain)
- Incremental corpus building with content hashing
- Hybrid retrieval (Vector DB + Knowledge Graph)
- Integration with existing memOS infrastructure:
  - ScratchpadCache for persistent storage
  - EntityTracker for GSW-style extraction
  - KVCacheService for cache warming
  - ExperienceDistiller for learning patterns

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DOMAIN CORPUS SYSTEM                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  DomainSchema ─────▶ DomainCorpus ─────▶ CorpusRetriever        │
    │       ↓                    ↓                    ↓                │
    │  Entity Types         Knowledge        Hybrid Search            │
    │  Relationships        Storage          Contextual Synthesis     │
    │  Extraction Hints     Persistence                               │
    └─────────────────────────────────────────────────────────────────┘

Research Basis:
- HybridRAG (2025): Entity-focused retrieval with knowledge graphs
- GSW (2025): Actor-centric memory for 51% token reduction
- Industrial KGs: 97.5% accuracy in technical troubleshooting
- Incremental Learning: Delta indexing without full rebuild

Usage:
    # Define domain schema
    fanuc_schema = DomainSchema(
        domain_id="fanuc_robotics",
        entity_types=FANUC_ENTITY_TYPES,
        relationships=FANUC_RELATIONSHIPS,
        extraction_hints={"error_pattern": r"SRVO-\\d+"}
    )

    # Create domain corpus
    corpus = DomainCorpus(schema=fanuc_schema, db_path="fanuc_corpus.db")

    # Build incrementally
    builder = CorpusBuilder(corpus)
    await builder.add_document("Manual excerpt...", source="fanuc_manual_v8.pdf")

    # Retrieve with context
    retriever = CorpusRetriever(corpus)
    context = await retriever.query("SRVO-001 overcurrent error")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
import uuid
import numpy as np
import httpx

logger = logging.getLogger("agentic.domain_corpus")


# ============================================
# DOMAIN-AGNOSTIC ENTITY TYPE DEFINITIONS
# ============================================

class TroubleshootingEntityType(str, Enum):
    """Universal entity types for technical troubleshooting domains"""
    ERROR_CODE = "error_code"          # SRVO-001, GPIO Error, etc.
    COMPONENT = "component"            # J1 motor, GPIO pin, servo amplifier
    SYMPTOM = "symptom"                # Overcurrent, position deviation, overheating
    CAUSE = "cause"                    # Worn gearbox, loose cable, voltage spike
    SOLUTION = "solution"              # Replace component, recalibrate, adjust settings
    PROCEDURE = "procedure"            # Step-by-step troubleshooting process
    PARAMETER = "parameter"            # Configuration settings, thresholds
    PART_NUMBER = "part_number"        # A06B-6079-H101, BCM2837, etc.
    TOOL = "tool"                      # Multimeter, oscilloscope, diagnostic software
    SYSTEM = "system"                  # Controller, drive unit, subsystem
    MEASUREMENT = "measurement"        # Voltage, current, temperature reading
    CONCEPT = "concept"                # Theoretical knowledge, best practices


class TroubleshootingRelationType(str, Enum):
    """Universal relationships for troubleshooting knowledge graphs"""
    HAS_SYMPTOM = "has_symptom"        # error_code → symptom
    CAUSED_BY = "caused_by"            # symptom → cause
    RESOLVED_BY = "resolved_by"        # cause → solution
    REQUIRES_PART = "requires_part"    # solution → part_number
    FOLLOWS_PROCEDURE = "follows_procedure"  # solution → procedure
    AFFECTS = "affects"                # cause → component
    PART_OF = "part_of"                # component → system
    INDICATES = "indicates"            # symptom → error_code
    MEASURED_BY = "measured_by"        # parameter → measurement
    ALTERNATIVE_TO = "alternative_to"  # solution → solution
    PREREQUISITE_OF = "prerequisite"   # procedure → procedure


@dataclass
class DomainEntityDef:
    """Definition of an entity type for a specific domain"""
    entity_type: str
    description: str
    extraction_patterns: List[str] = field(default_factory=list)  # Regex patterns
    examples: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)  # Expected attributes


@dataclass
class DomainRelationDef:
    """Definition of a relationship type for a specific domain"""
    relation_type: str
    source_types: List[str]  # Entity types that can be source
    target_types: List[str]  # Entity types that can be target
    description: str
    bidirectional: bool = False


@dataclass
class DomainSchema:
    """
    Schema definition for a specific domain.

    This defines what entities and relationships to look for
    when building the domain corpus.
    """
    domain_id: str
    domain_name: str
    description: str
    entity_types: List[DomainEntityDef]
    relationships: List[DomainRelationDef]
    extraction_hints: Dict[str, Any] = field(default_factory=dict)
    priority_patterns: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_entity_type_names(self) -> List[str]:
        """Get list of entity type names"""
        return [e.entity_type for e in self.entity_types]

    def get_extraction_prompt_fragment(self) -> str:
        """Generate extraction prompt fragment for this domain"""
        entity_types_str = ", ".join(
            f"{e.entity_type} (e.g., {', '.join(e.examples[:2])})"
            for e in self.entity_types if e.examples
        )
        return f"""Domain: {self.domain_name}
Entity types to extract: {entity_types_str}
Look for: {', '.join(self.priority_patterns[:5]) if self.priority_patterns else 'technical terms and relationships'}"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary"""
        return {
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "description": self.description,
            "entity_types": [
                {
                    "type": e.entity_type,
                    "description": e.description,
                    "patterns": e.extraction_patterns,
                    "examples": e.examples,
                    "attributes": e.attributes
                }
                for e in self.entity_types
            ],
            "relationships": [
                {
                    "type": r.relation_type,
                    "source_types": r.source_types,
                    "target_types": r.target_types,
                    "description": r.description,
                    "bidirectional": r.bidirectional
                }
                for r in self.relationships
            ],
            "extraction_hints": self.extraction_hints,
            "priority_patterns": self.priority_patterns
        }


# ============================================
# DOMAIN ENTITY AND KNOWLEDGE STORAGE
# ============================================

@dataclass
class DomainEntity:
    """An entity within a domain corpus"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    entity_type: str = ""
    name: str = ""
    canonical_name: str = ""  # Normalized form for deduplication
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source_documents: Set[str] = field(default_factory=set)  # Content hashes
    embedding: Optional[List[float]] = None
    confidence: float = 0.8
    mention_count: int = 1
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self._normalize_name(self.name)

    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication"""
        return name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")

    def merge_with(self, other: "DomainEntity") -> None:
        """Merge another entity into this one"""
        self.aliases.update(other.aliases)
        self.aliases.add(other.name)
        self.source_documents.update(other.source_documents)
        self.attributes.update(other.attributes)
        self.mention_count += other.mention_count
        self.last_updated = datetime.now(timezone.utc)
        # Keep higher confidence
        self.confidence = max(self.confidence, other.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "attributes": self.attributes,
            "description": self.description,
            "source_count": len(self.source_documents),
            "confidence": self.confidence,
            "mention_count": self.mention_count
        }

    def generate_context(self, max_length: int = 200) -> str:
        """Generate concise context for LLM consumption"""
        parts = [f"**{self.name}** ({self.entity_type})"]
        if self.description:
            parts.append(self.description[:100])
        if self.attributes:
            attr_str = ", ".join(f"{k}: {v}" for k, v in list(self.attributes.items())[:3])
            parts.append(f"[{attr_str}]")
        context = " | ".join(parts)
        return context[:max_length] if len(context) > max_length else context


@dataclass
class DomainRelation:
    """A relationship between entities in the corpus"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = ""
    description: str = ""
    confidence: float = 0.8
    source_document: str = ""  # Content hash of source
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_entity_id,
            "target": self.target_entity_id,
            "type": self.relation_type,
            "description": self.description,
            "confidence": self.confidence
        }


@dataclass
class CorpusDocument:
    """A source document in the corpus"""
    content_hash: str = ""
    content: str = ""
    source_url: str = ""
    source_type: str = ""  # manual, forum, log, etc.
    title: str = ""
    indexed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    chunk_hashes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        normalized = " ".join(self.content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ============================================
# DOMAIN CORPUS - PERSISTENT KNOWLEDGE STORE
# ============================================

class DomainCorpus:
    """
    Persistent knowledge store for a specific domain.

    Features:
    - SQLite-backed persistence
    - Incremental updates with content hashing
    - Entity deduplication via canonical names
    - Relationship graph storage
    - Embedding storage for semantic search
    """

    def __init__(
        self,
        schema: DomainSchema,
        db_path: str = "data/domain_corpus.db",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text"
    ):
        self.schema = schema
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model

        # In-memory caches
        self.entities: Dict[str, DomainEntity] = {}
        self.relations: List[DomainRelation] = []
        self._canonical_to_id: Dict[str, str] = {}  # canonical_name → entity_id
        self._content_hashes: Set[str] = set()  # For deduplication

        # Statistics
        self.stats = {
            "documents_indexed": 0,
            "entities_extracted": 0,
            "entities_merged": 0,
            "relations_found": 0,
            "embedding_calls": 0,
            "cache_hits": 0
        }

        # Initialize database
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        """Initialize SQLite schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.executescript(f"""
            -- Schema metadata
            CREATE TABLE IF NOT EXISTS corpus_metadata (
                domain_id TEXT PRIMARY KEY,
                schema_json TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );

            -- Entities
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases TEXT,  -- JSON array
                attributes TEXT,  -- JSON object
                description TEXT,
                source_documents TEXT,  -- JSON array
                embedding BLOB,
                confidence REAL DEFAULT 0.8,
                mention_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_updated TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

            -- Relations
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                description TEXT,
                confidence REAL DEFAULT 0.8,
                source_document TEXT,
                created_at TEXT,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

            -- Documents
            CREATE TABLE IF NOT EXISTS documents (
                content_hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT,
                source_type TEXT,
                title TEXT,
                indexed_at TEXT,
                embedding BLOB,
                chunk_hashes TEXT  -- JSON array
            );

            -- Chunks (for long documents)
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_hash TEXT PRIMARY KEY,
                document_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_hash) REFERENCES documents(content_hash)
            );
        """)

        # Store schema
        cursor.execute("""
            INSERT OR REPLACE INTO corpus_metadata (domain_id, schema_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            self.schema.domain_id,
            json.dumps(self.schema.to_dict()),
            self.schema.created_at.isoformat(),
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()
        logger.info(f"Initialized corpus database for domain: {self.schema.domain_id}")

    def _load_from_db(self):
        """Load entities and relations from database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Load entities
        cursor.execute("SELECT * FROM entities")
        for row in cursor.fetchall():
            entity = DomainEntity(
                id=row[0],
                entity_type=row[1],
                name=row[2],
                canonical_name=row[3],
                aliases=set(json.loads(row[4])) if row[4] else set(),
                attributes=json.loads(row[5]) if row[5] else {},
                description=row[6] or "",
                source_documents=set(json.loads(row[7])) if row[7] else set(),
                embedding=list(np.frombuffer(row[8], dtype=np.float32)) if row[8] else None,
                confidence=row[9] or 0.8,
                mention_count=row[10] or 1,
                first_seen=datetime.fromisoformat(row[11]) if row[11] else datetime.now(timezone.utc),
                last_updated=datetime.fromisoformat(row[12]) if row[12] else datetime.now(timezone.utc)
            )
            self.entities[entity.id] = entity
            self._canonical_to_id[entity.canonical_name] = entity.id

        # Load relations
        cursor.execute("SELECT * FROM relations")
        for row in cursor.fetchall():
            relation = DomainRelation(
                id=row[0],
                source_entity_id=row[1],
                target_entity_id=row[2],
                relation_type=row[3],
                description=row[4] or "",
                confidence=row[5] or 0.8,
                source_document=row[6] or "",
                created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(timezone.utc)
            )
            self.relations.append(relation)

        # Load content hashes
        cursor.execute("SELECT content_hash FROM documents")
        for row in cursor.fetchall():
            self._content_hashes.add(row[0])

        conn.close()
        logger.info(f"Loaded {len(self.entities)} entities, {len(self.relations)} relations from corpus")

    def has_content(self, content: str) -> bool:
        """Check if content has already been indexed"""
        normalized = " ".join(content.lower().split())
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        if content_hash in self._content_hashes:
            self.stats["cache_hits"] += 1
            return True
        return False

    def add_entity(self, entity: DomainEntity) -> str:
        """
        Add or merge entity into corpus.

        Returns the entity ID (may be existing if merged).
        """
        # Check for existing entity with same canonical name
        if entity.canonical_name in self._canonical_to_id:
            existing_id = self._canonical_to_id[entity.canonical_name]
            existing = self.entities[existing_id]
            existing.merge_with(entity)
            self._persist_entity(existing)
            self.stats["entities_merged"] += 1
            return existing_id

        # Add new entity
        self.entities[entity.id] = entity
        self._canonical_to_id[entity.canonical_name] = entity.id
        self._persist_entity(entity)
        self.stats["entities_extracted"] += 1
        return entity.id

    def add_relation(self, relation: DomainRelation) -> str:
        """Add relation to corpus"""
        # Check for duplicate
        for existing in self.relations:
            if (existing.source_entity_id == relation.source_entity_id and
                existing.target_entity_id == relation.target_entity_id and
                existing.relation_type == relation.relation_type):
                # Update confidence if higher
                if relation.confidence > existing.confidence:
                    existing.confidence = relation.confidence
                return existing.id

        self.relations.append(relation)
        self._persist_relation(relation)
        self.stats["relations_found"] += 1
        return relation.id

    def add_document(self, document: CorpusDocument) -> bool:
        """
        Add document to corpus if not already present.

        Returns True if added, False if duplicate.
        """
        if document.content_hash in self._content_hashes:
            self.stats["cache_hits"] += 1
            return False

        self._content_hashes.add(document.content_hash)
        self._persist_document(document)
        self.stats["documents_indexed"] += 1
        return True

    def _persist_entity(self, entity: DomainEntity):
        """Persist entity to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        embedding_bytes = None
        if entity.embedding:
            embedding_bytes = np.array(entity.embedding, dtype=np.float32).tobytes()

        cursor.execute("""
            INSERT OR REPLACE INTO entities
            (id, entity_type, name, canonical_name, aliases, attributes, description,
             source_documents, embedding, confidence, mention_count, first_seen, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            entity.entity_type,
            entity.name,
            entity.canonical_name,
            json.dumps(list(entity.aliases)),
            json.dumps(entity.attributes),
            entity.description,
            json.dumps(list(entity.source_documents)),
            embedding_bytes,
            entity.confidence,
            entity.mention_count,
            entity.first_seen.isoformat(),
            entity.last_updated.isoformat()
        ))

        conn.commit()
        conn.close()

    def _persist_relation(self, relation: DomainRelation):
        """Persist relation to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO relations
            (id, source_entity_id, target_entity_id, relation_type, description,
             confidence, source_document, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            relation.id,
            relation.source_entity_id,
            relation.target_entity_id,
            relation.relation_type,
            relation.description,
            relation.confidence,
            relation.source_document,
            relation.created_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def _persist_document(self, document: CorpusDocument):
        """Persist document to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        embedding_bytes = None
        if document.embedding:
            embedding_bytes = np.array(document.embedding, dtype=np.float32).tobytes()

        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (content_hash, content, source_url, source_type, title, indexed_at, embedding, chunk_hashes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.content_hash,
            document.content,
            document.source_url,
            document.source_type,
            document.title,
            document.indexed_at.isoformat(),
            embedding_bytes,
            json.dumps(document.chunk_hashes)
        ))

        conn.commit()
        conn.close()

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text[:2000]  # Truncate if too long
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    self.stats["embedding_calls"] += 1
                    return result.get("embedding")
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
        return None

    def get_entity(self, entity_id: str) -> Optional[DomainEntity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Optional[DomainEntity]:
        """Find entity by name or alias"""
        canonical = name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
        entity_id = self._canonical_to_id.get(canonical)
        if entity_id:
            return self.entities.get(entity_id)

        # Check aliases
        for entity in self.entities.values():
            for alias in entity.aliases:
                if alias.lower() == name.lower():
                    return entity
        return None

    def get_entities_by_type(self, entity_type: str) -> List[DomainEntity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_relations_for_entity(self, entity_id: str) -> List[Tuple[DomainEntity, DomainRelation]]:
        """Get all relations involving an entity"""
        results = []
        for relation in self.relations:
            if relation.source_entity_id == entity_id:
                target = self.entities.get(relation.target_entity_id)
                if target:
                    results.append((target, relation))
            elif relation.target_entity_id == entity_id:
                source = self.entities.get(relation.source_entity_id)
                if source:
                    results.append((source, relation))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        return {
            "domain_id": self.schema.domain_id,
            "domain_name": self.schema.domain_name,
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_documents": len(self._content_hashes),
            "entity_types": {
                etype: len(self.get_entities_by_type(etype))
                for etype in self.schema.get_entity_type_names()
            },
            **self.stats
        }

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export corpus as knowledge graph for visualization"""
        return {
            "nodes": [e.to_dict() for e in self.entities.values()],
            "edges": [r.to_dict() for r in self.relations],
            "schema": self.schema.to_dict()
        }


# ============================================
# CORPUS BUILDER - INCREMENTAL UPDATES
# ============================================

class CorpusBuilder:
    """
    Builds domain corpus incrementally with entity extraction.

    Features:
    - Content hashing for deduplication
    - Semantic chunking for long documents
    - LLM-based entity extraction using domain schema
    - Automatic relationship inference
    """

    # Extraction prompt template
    EXTRACTION_PROMPT = """You are extracting structured information for a {domain_name} knowledge base.

{schema_context}

Content to analyze:
{content}

Extract all relevant entities and relationships. Return ONLY valid JSON:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "{entity_types_hint}",
      "description": "Brief description",
      "attributes": {{"key": "value"}}
    }}
  ],
  "relations": [
    {{
      "source": "Entity1 Name",
      "target": "Entity2 Name",
      "type": "{relation_types_hint}",
      "description": "Relationship description"
    }}
  ]
}}

/no_think"""

    def __init__(
        self,
        corpus: DomainCorpus,
        extraction_model: str = "qwen3:8b",
        chunk_size: int = 1500,
        chunk_overlap: int = 200
    ):
        self.corpus = corpus
        self.extraction_model = extraction_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def add_document(
        self,
        content: str,
        source_url: str = "",
        source_type: str = "unknown",
        title: str = "",
        extract_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Add document to corpus with optional entity extraction.

        Returns extraction results.
        """
        # Check for duplicate
        if self.corpus.has_content(content):
            return {"status": "duplicate", "entities": 0, "relations": 0}

        # Create document
        document = CorpusDocument(
            content=content,
            source_url=source_url,
            source_type=source_type,
            title=title
        )

        # Generate embedding for document
        document.embedding = await self.corpus.generate_embedding(content[:2000])

        # Chunk if necessary
        chunks = self._chunk_content(content) if len(content) > self.chunk_size else [content]
        document.chunk_hashes = [
            hashlib.sha256(c.encode()).hexdigest()[:16] for c in chunks
        ]

        # Add document
        self.corpus.add_document(document)

        # Extract entities if requested
        entities_count = 0
        relations_count = 0

        if extract_entities:
            for chunk in chunks:
                result = await self._extract_from_chunk(chunk, document.content_hash)
                entities_count += result["entities"]
                relations_count += result["relations"]

        return {
            "status": "indexed",
            "content_hash": document.content_hash,
            "chunks": len(chunks),
            "entities": entities_count,
            "relations": relations_count
        }

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(content):
                # Look for period followed by space
                boundary = content.rfind(". ", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary + 2

            chunks.append(content[start:end])
            start = end - self.chunk_overlap

        return chunks

    async def _extract_from_chunk(self, chunk: str, document_hash: str) -> Dict[str, int]:
        """Extract entities and relations from a chunk"""
        schema = self.corpus.schema

        # Build prompt
        prompt = self.EXTRACTION_PROMPT.format(
            domain_name=schema.domain_name,
            schema_context=schema.get_extraction_prompt_fragment(),
            content=chunk[:3000],
            entity_types_hint=" | ".join(schema.get_entity_type_names()[:5]),
            relation_types_hint=" | ".join([r.relation_type for r in schema.relationships[:5]])
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.corpus.ollama_url}/api/generate",
                    json={
                        "model": self.extraction_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 2048
                        }
                    }
                )

                if response.status_code != 200:
                    logger.warning(f"Extraction failed: {response.status_code}")
                    return {"entities": 0, "relations": 0}

                result = response.json()
                response_text = result.get("response", "")

                # Parse JSON
                extracted = self._parse_json(response_text)
                if not extracted:
                    return {"entities": 0, "relations": 0}

                # Process entities
                entity_id_map = {}  # name -> id for relation resolution
                entities_added = 0

                for entity_data in extracted.get("entities", []):
                    entity = DomainEntity(
                        entity_type=entity_data.get("type", "concept"),
                        name=entity_data.get("name", ""),
                        description=entity_data.get("description", ""),
                        attributes=entity_data.get("attributes", {}),
                        source_documents={document_hash}
                    )

                    if entity.name:
                        # Generate embedding
                        entity.embedding = await self.corpus.generate_embedding(
                            f"{entity.name} {entity.description}"
                        )
                        entity_id = self.corpus.add_entity(entity)
                        entity_id_map[entity.name.lower()] = entity_id
                        entities_added += 1

                # Process relations
                relations_added = 0
                for rel_data in extracted.get("relations", []):
                    source_name = rel_data.get("source", "").lower()
                    target_name = rel_data.get("target", "").lower()

                    source_id = entity_id_map.get(source_name)
                    target_id = entity_id_map.get(target_name)

                    if source_id and target_id:
                        relation = DomainRelation(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relation_type=rel_data.get("type", "related_to"),
                            description=rel_data.get("description", ""),
                            source_document=document_hash
                        )
                        self.corpus.add_relation(relation)
                        relations_added += 1

                return {"entities": entities_added, "relations": relations_added}

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"entities": 0, "relations": 0}

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None


# ============================================
# CORPUS RETRIEVER - HYBRID SEARCH
# ============================================

class CorpusRetriever:
    """
    Retrieves relevant context from domain corpus.

    Features:
    - Semantic search via embeddings
    - Graph traversal for related entities
    - Contextual synthesis for LLM consumption
    """

    def __init__(self, corpus: DomainCorpus, max_results: int = 10):
        self.corpus = corpus
        self.max_results = max_results

    async def query(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        include_relations: bool = True,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Query corpus with hybrid search.

        Returns entities, relations, and synthesized context.
        """
        # Generate query embedding
        query_embedding = await self.corpus.generate_embedding(query)

        # Find relevant entities
        scored_entities = []
        query_terms = set(query.lower().split())

        for entity in self.corpus.entities.values():
            # Filter by type if specified
            if entity_types and entity.entity_type not in entity_types:
                continue

            score = 0.0

            # Semantic similarity (if embeddings available)
            if query_embedding and entity.embedding:
                similarity = self._cosine_similarity(
                    np.array(query_embedding),
                    np.array(entity.embedding)
                )
                score += similarity * 0.6  # 60% weight for semantic

            # Keyword match (40% weight)
            name_lower = entity.name.lower()
            if any(term in name_lower for term in query_terms):
                score += 0.3

            for alias in entity.aliases:
                if any(term in alias.lower() for term in query_terms):
                    score += 0.1

            # Mention count bonus
            score += min(entity.mention_count / 100, 0.1)

            if score > 0.1:
                scored_entities.append((entity, score))

        # Sort by score
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        top_entities = scored_entities[:self.max_results]

        # Get related entities via graph traversal
        related_entities = []
        if include_relations and top_entities:
            seen_ids = {e.id for e, _ in top_entities}
            for entity, _ in top_entities:
                relations = self.corpus.get_relations_for_entity(entity.id)
                for related, relation in relations[:3]:  # Limit depth
                    if related.id not in seen_ids:
                        related_entities.append((related, relation, 0.5))  # Lower score for related
                        seen_ids.add(related.id)

        # Generate synthesized context
        context = self._generate_context(top_entities, related_entities, query)

        return {
            "entities": [
                {"entity": e.to_dict(), "score": s}
                for e, s in top_entities
            ],
            "related": [
                {"entity": e.to_dict(), "relation": r.to_dict(), "score": s}
                for e, r, s in related_entities
            ],
            "context": context,
            "query": query
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _generate_context(
        self,
        entities: List[Tuple[DomainEntity, float]],
        related: List[Tuple[DomainEntity, DomainRelation, float]],
        query: str
    ) -> str:
        """Generate LLM-ready context from search results"""
        parts = [f"## Domain Knowledge for: {query}\n"]

        # Primary entities
        if entities:
            parts.append("### Key Entities")
            for entity, score in entities[:5]:
                parts.append(f"- {entity.generate_context()}")

        # Related entities with relationships
        if related:
            parts.append("\n### Related Information")
            for entity, relation, _ in related[:5]:
                parts.append(f"- {entity.name} ({relation.relation_type}): {entity.description[:100]}")

        return "\n".join(parts)

    async def get_troubleshooting_path(
        self,
        error_code: str
    ) -> Dict[str, Any]:
        """
        Get complete troubleshooting path for an error code.

        Traverses: error_code → symptoms → causes → solutions
        """
        # Find error code entity
        error_entity = self.corpus.find_entity_by_name(error_code)
        if not error_entity:
            return {"error": f"Error code '{error_code}' not found in corpus"}

        path = {
            "error_code": error_entity.to_dict(),
            "symptoms": [],
            "causes": [],
            "solutions": []
        }

        # Get symptoms (error_code → HAS_SYMPTOM → symptom)
        for entity, relation in self.corpus.get_relations_for_entity(error_entity.id):
            if relation.relation_type == TroubleshootingRelationType.HAS_SYMPTOM.value:
                path["symptoms"].append({
                    "entity": entity.to_dict(),
                    "relation": relation.to_dict()
                })

                # Get causes for each symptom
                for cause_entity, cause_rel in self.corpus.get_relations_for_entity(entity.id):
                    if cause_rel.relation_type == TroubleshootingRelationType.CAUSED_BY.value:
                        path["causes"].append({
                            "entity": cause_entity.to_dict(),
                            "relation": cause_rel.to_dict(),
                            "symptom": entity.name
                        })

                        # Get solutions for each cause
                        for sol_entity, sol_rel in self.corpus.get_relations_for_entity(cause_entity.id):
                            if sol_rel.relation_type == TroubleshootingRelationType.RESOLVED_BY.value:
                                path["solutions"].append({
                                    "entity": sol_entity.to_dict(),
                                    "relation": sol_rel.to_dict(),
                                    "cause": cause_entity.name
                                })

        return path


# ============================================
# PRE-DEFINED DOMAIN SCHEMAS
# ============================================

def create_fanuc_schema() -> DomainSchema:
    """Create schema for FANUC robotics troubleshooting"""
    return DomainSchema(
        domain_id="fanuc_robotics",
        domain_name="FANUC Robotics",
        description="Knowledge base for FANUC robot troubleshooting and maintenance",
        entity_types=[
            DomainEntityDef(
                entity_type="error_code",
                description="FANUC alarm/error codes",
                extraction_patterns=[r"SRVO-\d+", r"SYST-\d+", r"HOST-\d+"],
                examples=["SRVO-001", "SRVO-050", "SYST-001"],
                attributes=["severity", "category", "message"]
            ),
            DomainEntityDef(
                entity_type="component",
                description="Robot components and hardware",
                extraction_patterns=[r"J[1-6]", r"axis \d", r"servo"],
                examples=["J1 motor", "teach pendant", "servo amplifier", "encoder"],
                attributes=["location", "part_number", "specifications"]
            ),
            DomainEntityDef(
                entity_type="symptom",
                description="Observable symptoms and behaviors",
                examples=["overcurrent", "position deviation", "communication error"],
                attributes=["frequency", "severity"]
            ),
            DomainEntityDef(
                entity_type="cause",
                description="Root causes of issues",
                examples=["worn gearbox", "loose cable", "encoder fault"],
                attributes=["likelihood", "difficulty_to_diagnose"]
            ),
            DomainEntityDef(
                entity_type="solution",
                description="Fix or workaround",
                examples=["replace servo amp", "recalibrate", "mastering"],
                attributes=["difficulty", "time_required", "parts_needed"]
            ),
            DomainEntityDef(
                entity_type="procedure",
                description="Step-by-step process",
                examples=["backup procedure", "mastering procedure", "zero-point calibration"],
                attributes=["steps", "safety_requirements"]
            ),
            DomainEntityDef(
                entity_type="parameter",
                description="Configuration parameters",
                extraction_patterns=[r"\$[A-Z_]+", r"PARAM\["],
                examples=["$PARAM_GROUP", "$MOR_GRP", "$SERVO"],
                attributes=["default_value", "range"]
            ),
            DomainEntityDef(
                entity_type="part_number",
                description="FANUC part numbers",
                extraction_patterns=[r"A\d{2}B-\d{4}-\w+"],
                examples=["A06B-6079-H101", "A05B-1215-B201"],
                attributes=["description", "price_range"]
            )
        ],
        relationships=[
            DomainRelationDef(
                relation_type="has_symptom",
                source_types=["error_code"],
                target_types=["symptom"],
                description="Error code manifests as symptom"
            ),
            DomainRelationDef(
                relation_type="caused_by",
                source_types=["symptom", "error_code"],
                target_types=["cause"],
                description="Symptom is caused by root cause"
            ),
            DomainRelationDef(
                relation_type="resolved_by",
                source_types=["cause", "error_code"],
                target_types=["solution"],
                description="Issue is resolved by solution"
            ),
            DomainRelationDef(
                relation_type="requires_part",
                source_types=["solution"],
                target_types=["part_number"],
                description="Solution requires specific part"
            ),
            DomainRelationDef(
                relation_type="follows_procedure",
                source_types=["solution"],
                target_types=["procedure"],
                description="Solution follows procedure"
            ),
            DomainRelationDef(
                relation_type="affects",
                source_types=["error_code", "cause"],
                target_types=["component"],
                description="Error/cause affects component"
            )
        ],
        extraction_hints={
            "error_pattern": r"SRVO-\d+|SYST-\d+|HOST-\d+|MOTN-\d+",
            "part_pattern": r"A\d{2}B-\d{4}-\w+",
            "axis_pattern": r"J[1-6]|axis [1-6]"
        },
        priority_patterns=[
            "SRVO-", "alarm", "error", "fault", "overcurrent",
            "servo amplifier", "encoder", "motor", "mastering"
        ]
    )


def create_raspberry_pi_schema() -> DomainSchema:
    """Create schema for Raspberry Pi troubleshooting"""
    return DomainSchema(
        domain_id="raspberry_pi",
        domain_name="Raspberry Pi",
        description="Knowledge base for Raspberry Pi projects and troubleshooting",
        entity_types=[
            DomainEntityDef(
                entity_type="error_code",
                description="System errors and kernel messages",
                examples=["kernel panic", "I/O error", "GPIO error"],
                attributes=["dmesg_output", "log_level"]
            ),
            DomainEntityDef(
                entity_type="component",
                description="Hardware components and peripherals",
                examples=["GPIO pin", "BCM2837", "HDMI port", "USB controller"],
                attributes=["pin_number", "specifications"]
            ),
            DomainEntityDef(
                entity_type="symptom",
                description="Observable issues",
                examples=["no boot", "overheating", "SD card corruption"],
                attributes=["frequency", "conditions"]
            ),
            DomainEntityDef(
                entity_type="cause",
                description="Root causes",
                examples=["insufficient power", "bad SD card", "kernel mismatch"],
                attributes=["likelihood"]
            ),
            DomainEntityDef(
                entity_type="solution",
                description="Fixes and workarounds",
                examples=["increase swap", "update firmware", "add heatsink"],
                attributes=["difficulty", "commands"]
            ),
            DomainEntityDef(
                entity_type="command",
                description="Linux commands and scripts",
                extraction_patterns=[r"sudo \w+", r"apt-get \w+", r"raspi-config"],
                examples=["raspi-config", "vcgencmd", "gpio readall"],
                attributes=["options", "example_usage"]
            ),
            DomainEntityDef(
                entity_type="configuration",
                description="Config files and settings",
                extraction_patterns=[r"/boot/config\.txt", r"/etc/\w+"],
                examples=["config.txt", "cmdline.txt", "rc.local"],
                attributes=["location", "parameters"]
            ),
            DomainEntityDef(
                entity_type="model",
                description="Raspberry Pi models",
                examples=["Pi 4B", "Pi Zero W", "Pi 5"],
                attributes=["ram", "cpu", "release_date"]
            )
        ],
        relationships=[
            DomainRelationDef(
                relation_type="has_symptom",
                source_types=["error_code"],
                target_types=["symptom"],
                description="Error manifests as symptom"
            ),
            DomainRelationDef(
                relation_type="caused_by",
                source_types=["symptom"],
                target_types=["cause"],
                description="Symptom caused by issue"
            ),
            DomainRelationDef(
                relation_type="resolved_by",
                source_types=["cause", "symptom"],
                target_types=["solution"],
                description="Issue resolved by solution"
            ),
            DomainRelationDef(
                relation_type="requires_command",
                source_types=["solution"],
                target_types=["command"],
                description="Solution requires command"
            ),
            DomainRelationDef(
                relation_type="modifies",
                source_types=["solution", "command"],
                target_types=["configuration"],
                description="Action modifies configuration"
            ),
            DomainRelationDef(
                relation_type="affects",
                source_types=["cause"],
                target_types=["component"],
                description="Cause affects component"
            ),
            DomainRelationDef(
                relation_type="compatible_with",
                source_types=["solution", "component"],
                target_types=["model"],
                description="Compatible with specific model"
            )
        ],
        extraction_hints={
            "gpio_pattern": r"GPIO\s*\d+|BCM\s*\d+|pin\s*\d+",
            "command_pattern": r"sudo\s+\w+|apt-get\s+\w+|pip\s+\w+",
            "config_pattern": r"/boot/\w+|/etc/\w+"
        },
        priority_patterns=[
            "GPIO", "boot", "kernel", "power supply", "SD card",
            "config.txt", "apt-get", "sudo", "temperature"
        ]
    )


# ============================================
# CORPUS MANAGER - HIGH-LEVEL API
# ============================================

class DomainCorpusManager:
    """
    High-level manager for multiple domain corpuses.

    Provides:
    - Multi-domain support
    - Unified query interface
    - Integration with memOS infrastructure
    """

    def __init__(self, base_path: str = "data/corpuses"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.corpuses: Dict[str, DomainCorpus] = {}
        self.builders: Dict[str, CorpusBuilder] = {}
        self.retrievers: Dict[str, CorpusRetriever] = {}

    def register_corpus(
        self,
        schema: DomainSchema,
        ollama_url: str = "http://localhost:11434"
    ) -> DomainCorpus:
        """Register and initialize a domain corpus"""
        db_path = self.base_path / f"{schema.domain_id}.db"

        corpus = DomainCorpus(
            schema=schema,
            db_path=str(db_path),
            ollama_url=ollama_url
        )

        self.corpuses[schema.domain_id] = corpus
        self.builders[schema.domain_id] = CorpusBuilder(corpus)
        self.retrievers[schema.domain_id] = CorpusRetriever(corpus)

        logger.info(f"Registered corpus: {schema.domain_id} ({schema.domain_name})")
        return corpus

    def get_corpus(self, domain_id: str) -> Optional[DomainCorpus]:
        """Get corpus by domain ID"""
        return self.corpuses.get(domain_id)

    def get_builder(self, domain_id: str) -> Optional[CorpusBuilder]:
        """Get builder for domain"""
        return self.builders.get(domain_id)

    def get_retriever(self, domain_id: str) -> Optional[CorpusRetriever]:
        """Get retriever for domain"""
        return self.retrievers.get(domain_id)

    async def add_document(
        self,
        domain_id: str,
        content: str,
        source_url: str = "",
        source_type: str = "unknown",
        title: str = ""
    ) -> Dict[str, Any]:
        """Add document to specified domain corpus"""
        builder = self.builders.get(domain_id)
        if not builder:
            return {"error": f"Domain '{domain_id}' not registered"}

        return await builder.add_document(
            content=content,
            source_url=source_url,
            source_type=source_type,
            title=title
        )

    async def query(
        self,
        domain_id: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Query specified domain corpus"""
        retriever = self.retrievers.get(domain_id)
        if not retriever:
            return {"error": f"Domain '{domain_id}' not registered"}

        return await retriever.query(query, **kwargs)

    async def cross_domain_query(
        self,
        query: str,
        domain_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query across multiple domains"""
        target_domains = domain_ids or list(self.corpuses.keys())
        results = {}

        for domain_id in target_domains:
            if domain_id in self.retrievers:
                results[domain_id] = await self.retrievers[domain_id].query(query)

        return {
            "query": query,
            "domains_queried": target_domains,
            "results": results
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all corpuses"""
        return {
            domain_id: corpus.get_stats()
            for domain_id, corpus in self.corpuses.items()
        }

    def list_domains(self) -> List[Dict[str, Any]]:
        """List all registered domains"""
        return [
            {
                "domain_id": corpus.schema.domain_id,
                "domain_name": corpus.schema.domain_name,
                "description": corpus.schema.description,
                "entities": len(corpus.entities),
                "relations": len(corpus.relations)
            }
            for corpus in self.corpuses.values()
        ]


# ============================================
# SINGLETON INSTANCE
# ============================================

_corpus_manager: Optional[DomainCorpusManager] = None


def get_corpus_manager(base_path: str = "data/corpuses") -> DomainCorpusManager:
    """Get or create singleton corpus manager"""
    global _corpus_manager
    if _corpus_manager is None:
        _corpus_manager = DomainCorpusManager(base_path)
    return _corpus_manager


def initialize_default_corpuses(
    ollama_url: str = "http://localhost:11434"
) -> DomainCorpusManager:
    """Initialize manager with default domain schemas"""
    manager = get_corpus_manager()

    # Register FANUC robotics corpus
    manager.register_corpus(create_fanuc_schema(), ollama_url)

    # Register Raspberry Pi corpus
    manager.register_corpus(create_raspberry_pi_schema(), ollama_url)

    logger.info(f"Initialized {len(manager.corpuses)} domain corpuses")
    return manager
