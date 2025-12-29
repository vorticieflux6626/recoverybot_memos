"""
Phase 5: Cross-Session Meta-Buffer

Persistent storage of successful reasoning templates that transfers across sessions.
Based on Buffer of Thoughts (NeurIPS 2024) - 12% cost of Tree/Graph of Thoughts.

Key Features:
- SQLite persistence for cross-session learning
- Template distillation from successful searches
- Query pattern matching for template reuse
- Success rate tracking and template evolution
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of reasoning templates"""
    QUERY_DECOMPOSITION = "query_decomposition"
    SEARCH_STRATEGY = "search_strategy"
    SYNTHESIS_STRUCTURE = "synthesis_structure"
    VERIFICATION_PATTERN = "verification_pattern"
    REFINEMENT_APPROACH = "refinement_approach"


@dataclass
class DistilledTemplate:
    """
    A template distilled from successful search execution.

    Contains:
    - Query pattern (regex or semantic)
    - Successful decomposition
    - Effective search strategies
    - Synthesis structure
    - Performance metrics
    """
    id: str
    template_type: TemplateType
    query_pattern: str  # Regex pattern or semantic description
    structure: str  # The actual template content (required)

    # Optional content
    query_embedding: Optional[List[float]] = None
    example_queries: List[str] = field(default_factory=list)
    example_outcomes: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    domain_hints: List[str] = field(default_factory=list)
    complexity_range: Tuple[str, str] = ("simple", "complex")

    # Performance
    usage_count: int = 0
    success_count: int = 0
    avg_confidence: float = 0.0
    avg_execution_time_ms: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "template_type": self.template_type.value,
            "query_pattern": self.query_pattern,
            "structure": self.structure,
            "example_queries": self.example_queries,
            "example_outcomes": self.example_outcomes,
            "domain_hints": self.domain_hints,
            "complexity_range": list(self.complexity_range),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "avg_confidence": self.avg_confidence,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistilledTemplate":
        return cls(
            id=data["id"],
            template_type=TemplateType(data["template_type"]),
            query_pattern=data["query_pattern"],
            structure=data["structure"],
            example_queries=data.get("example_queries", []),
            example_outcomes=data.get("example_outcomes", []),
            domain_hints=data.get("domain_hints", []),
            complexity_range=tuple(data.get("complexity_range", ["simple", "complex"])),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            avg_confidence=data.get("avg_confidence", 0.0),
            avg_execution_time_ms=data.get("avg_execution_time_ms", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None
        )


@dataclass
class InstantiatedTemplate:
    """A template instantiated for a specific query"""
    template_id: str
    template_type: TemplateType
    query: str
    instantiated_content: str
    skipped_phases: List[str] = field(default_factory=list)
    confidence_boost: float = 0.0


class MetaBuffer:
    """
    Cross-Session Meta-Buffer for reasoning template persistence.

    Benefits (from Buffer of Thoughts research):
    - 12% cost of Tree/Graph of Thoughts
    - 11-51% accuracy improvement
    - Templates transfer across task types
    - Continuous learning from successful searches

    Architecture:
    - SQLite for persistent storage
    - Embedding-based semantic retrieval
    - Pattern-based fast matching
    - Automatic template distillation
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        similarity_threshold: float = 0.75
    ):
        self.db_path = db_path or str(Path.home() / ".memos" / "meta_buffer.db")
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Cache for hot templates
        self._hot_cache: Dict[str, DistilledTemplate] = {}

        # Statistics
        self._stats = {
            "templates_stored": 0,
            "templates_retrieved": 0,
            "cache_hits": 0,
            "distillations": 0,
            "successful_reuses": 0
        }

        logger.info(f"MetaBuffer initialized with DB at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Templates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                template_type TEXT NOT NULL,
                query_pattern TEXT NOT NULL,
                query_embedding BLOB,
                structure TEXT NOT NULL,
                example_queries TEXT,
                example_outcomes TEXT,
                domain_hints TEXT,
                complexity_min TEXT,
                complexity_max TEXT,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                avg_execution_time_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            )
        """)

        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_template_type
            ON templates(template_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success_rate
            ON templates(success_count, usage_count)
        """)

        conn.commit()
        conn.close()

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": self.embedding_model, "input": text}
                )
                response.raise_for_status()
                result = response.json()
                embeddings = result.get("embeddings", [])
                return embeddings[0] if embeddings else None
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if not a or not b or len(a) != len(b):
            return 0.0

        import numpy as np
        a_arr, b_arr = np.array(a), np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    async def store_template(self, template: DistilledTemplate) -> str:
        """Store a template persistently"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Compute embedding if not present
        if template.query_embedding is None:
            template.query_embedding = await self._get_embedding(
                f"{template.query_pattern} {' '.join(template.example_queries[:3])}"
            )

        # Serialize embedding
        embedding_blob = json.dumps(template.query_embedding).encode() if template.query_embedding else None

        cursor.execute("""
            INSERT OR REPLACE INTO templates (
                id, template_type, query_pattern, query_embedding, structure,
                example_queries, example_outcomes, domain_hints,
                complexity_min, complexity_max,
                usage_count, success_count, avg_confidence, avg_execution_time_ms,
                created_at, updated_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            template.id,
            template.template_type.value,
            template.query_pattern,
            embedding_blob,
            template.structure,
            json.dumps(template.example_queries),
            json.dumps(template.example_outcomes),
            json.dumps(template.domain_hints),
            template.complexity_range[0],
            template.complexity_range[1],
            template.usage_count,
            template.success_count,
            template.avg_confidence,
            template.avg_execution_time_ms,
            template.created_at.isoformat(),
            template.updated_at.isoformat(),
            template.last_used_at.isoformat() if template.last_used_at else None
        ))

        conn.commit()
        conn.close()

        # Update hot cache
        self._hot_cache[template.id] = template
        self._stats["templates_stored"] += 1

        logger.info(f"Stored template: {template.id} ({template.template_type.value})")
        return template.id

    def _load_template_from_row(self, row: tuple) -> DistilledTemplate:
        """Load template from database row"""
        embedding = json.loads(row[3].decode()) if row[3] else None

        return DistilledTemplate(
            id=row[0],
            template_type=TemplateType(row[1]),
            query_pattern=row[2],
            query_embedding=embedding,
            structure=row[4],
            example_queries=json.loads(row[5]) if row[5] else [],
            example_outcomes=json.loads(row[6]) if row[6] else [],
            domain_hints=json.loads(row[7]) if row[7] else [],
            complexity_range=(row[8], row[9]),
            usage_count=row[10],
            success_count=row[11],
            avg_confidence=row[12],
            avg_execution_time_ms=row[13],
            created_at=datetime.fromisoformat(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
            last_used_at=datetime.fromisoformat(row[16]) if row[16] else None
        )

    async def retrieve_template(
        self,
        query: str,
        template_type: Optional[TemplateType] = None,
        min_success_rate: float = 0.5
    ) -> Optional[Tuple[DistilledTemplate, float]]:
        """
        Retrieve best matching template for a query.

        Returns:
            Tuple of (template, confidence_score) or None
        """
        self._stats["templates_retrieved"] += 1

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        sql = """
            SELECT * FROM templates
            WHERE usage_count >= 3
            AND CAST(success_count AS REAL) / CASE WHEN usage_count = 0 THEN 1 ELSE usage_count END >= ?
        """
        params = [min_success_rate]

        if template_type:
            sql += " AND template_type = ?"
            params.append(template_type.value)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Find best match
        best_match: Optional[Tuple[DistilledTemplate, float]] = None

        for row in rows:
            template = self._load_template_from_row(row)

            # Check cache first
            if template.id in self._hot_cache:
                template = self._hot_cache[template.id]
                self._stats["cache_hits"] += 1

            # Calculate similarity
            if query_embedding and template.query_embedding:
                similarity = self._cosine_similarity(query_embedding, template.query_embedding)
            else:
                # Fallback to pattern matching
                similarity = self._pattern_match_score(query, template)

            if similarity >= self.similarity_threshold:
                # Boost by success rate
                score = similarity * (0.7 + 0.3 * template.success_rate)

                if best_match is None or score > best_match[1]:
                    best_match = (template, score)

        return best_match

    def _pattern_match_score(self, query: str, template: DistilledTemplate) -> float:
        """Fallback pattern matching score"""
        query_lower = query.lower()

        # Check regex pattern
        try:
            if re.search(template.query_pattern, query, re.IGNORECASE):
                return 0.9
        except re.error:
            pass

        # Check domain hints
        hint_matches = sum(1 for hint in template.domain_hints if hint.lower() in query_lower)
        if hint_matches > 0:
            return 0.6 + (0.1 * min(hint_matches, 3))

        # Check example queries
        for example in template.example_queries:
            example_words = set(example.lower().split())
            query_words = set(query_lower.split())
            overlap = len(example_words & query_words) / max(len(example_words | query_words), 1)
            if overlap > 0.5:
                return 0.5 + (0.3 * overlap)

        return 0.0

    async def instantiate(
        self,
        template: DistilledTemplate,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> InstantiatedTemplate:
        """
        Instantiate a template for a specific query.

        This can skip low-confidence initial phases entirely.
        """
        context = context or {}

        # Replace placeholders
        content = template.structure
        content = content.replace("{query}", query)

        for key, value in context.items():
            content = content.replace(f"{{{key}}}", str(value))

        # Determine which phases can be skipped
        skipped_phases = []
        confidence_boost = 0.0

        if template.success_rate >= 0.8:
            if template.template_type == TemplateType.QUERY_DECOMPOSITION:
                skipped_phases.append("query_analysis")
                confidence_boost += 0.1
            elif template.template_type == TemplateType.SEARCH_STRATEGY:
                skipped_phases.append("search_planning")
                confidence_boost += 0.05

        # Update template usage
        template.usage_count += 1
        template.last_used_at = datetime.now(timezone.utc)
        template.updated_at = datetime.now(timezone.utc)

        # Persist update
        await self.store_template(template)

        return InstantiatedTemplate(
            template_id=template.id,
            template_type=template.template_type,
            query=query,
            instantiated_content=content,
            skipped_phases=skipped_phases,
            confidence_boost=confidence_boost
        )

    async def distill_from_search(
        self,
        query: str,
        decomposed_questions: List[str],
        search_queries: List[str],
        synthesis: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        execution_time_ms: int
    ) -> Optional[DistilledTemplate]:
        """
        Distill a reusable template from a successful search.

        Only creates template if confidence >= 0.75 (proven successful).
        """
        if confidence < 0.75:
            logger.debug(f"Skipping distillation: confidence {confidence:.2f} < 0.75")
            return None

        self._stats["distillations"] += 1

        # Generate template ID
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        template_id = f"distilled_{query_hash}_{int(datetime.now().timestamp())}"

        # Extract query pattern
        query_pattern = self._extract_pattern(query)

        # Extract domain hints
        domain_hints = self._extract_domain_hints(query, sources)

        # Create decomposition template
        decomp_structure = self._create_decomposition_structure(
            query, decomposed_questions, search_queries
        )

        template = DistilledTemplate(
            id=template_id,
            template_type=TemplateType.QUERY_DECOMPOSITION,
            query_pattern=query_pattern,
            structure=decomp_structure,
            example_queries=[query],
            example_outcomes=[{
                "confidence": confidence,
                "sources_count": len(sources),
                "execution_time_ms": execution_time_ms
            }],
            domain_hints=domain_hints,
            usage_count=1,
            success_count=1,
            avg_confidence=confidence,
            avg_execution_time_ms=execution_time_ms
        )

        # Store template
        await self.store_template(template)

        logger.info(f"Distilled template {template_id} from successful search (conf={confidence:.2f})")
        return template

    def _extract_pattern(self, query: str) -> str:
        """Extract a generalized regex pattern from query"""
        # Replace specific terms with generic placeholders
        pattern = query.lower()

        # Replace error codes
        pattern = re.sub(r'[A-Z]{2,4}[-_]?\d{3,4}', r'[A-Z]{2,4}[-_]?\\d{3,4}', pattern)

        # Replace model numbers
        pattern = re.sub(r'\b[A-Z]\d{3,}[A-Z]?\b', r'[A-Z]\\d{3,}[A-Z]?', pattern, flags=re.IGNORECASE)

        # Replace dates
        pattern = re.sub(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', r'\\d{4}[-/]\\d{2}[-/]\\d{2}', pattern)

        return pattern

    def _extract_domain_hints(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract domain hints from query and sources"""
        hints = []
        query_lower = query.lower()

        # Technical domain keywords
        domain_keywords = {
            "fanuc": ["robot", "servo", "alarm", "error", "axis"],
            "programming": ["code", "function", "error", "debug", "api"],
            "networking": ["network", "connection", "tcp", "ip", "protocol"],
            "database": ["sql", "query", "table", "database", "schema"],
            "raspberry_pi": ["gpio", "pin", "raspberry", "rpi", "board"]
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                hints.append(domain)

        # Extract from source domains
        for source in sources[:5]:
            url = source.get("url", "")
            if "stackoverflow" in url:
                hints.append("technical_qa")
            elif "github" in url:
                hints.append("code_repository")
            elif "docs" in url or "documentation" in url:
                hints.append("documentation")

        return list(set(hints))

    def _create_decomposition_structure(
        self,
        query: str,
        decomposed_questions: List[str],
        search_queries: List[str]
    ) -> str:
        """Create a reusable decomposition structure"""
        structure = f"""Query Decomposition Template

For queries similar to: {query}

## Decomposition Pattern
{{query}} should be broken into:
"""

        for i, question in enumerate(decomposed_questions, 1):
            # Generalize the question
            generalized = self._generalize_question(question)
            structure += f"\n{i}. {generalized}"

        structure += "\n\n## Search Strategy\nRecommended queries:\n"

        for sq in search_queries[:4]:
            generalized_sq = self._generalize_question(sq)
            structure += f"- {generalized_sq}\n"

        return structure

    def _generalize_question(self, question: str) -> str:
        """Generalize a specific question into a template"""
        # Replace specific terms with placeholders
        generalized = question

        # Replace error codes
        generalized = re.sub(r'[A-Z]{2,4}[-_]?\d{3,4}', '{error_code}', generalized)

        # Replace model numbers
        generalized = re.sub(r'\b[A-Z]\d{3,}[A-Z]?\b', '{model}', generalized, flags=re.IGNORECASE)

        return generalized

    def record_outcome(
        self,
        template_id: str,
        success: bool,
        confidence: float,
        execution_time_ms: int
    ) -> None:
        """Record outcome of template usage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current values
        cursor.execute(
            "SELECT usage_count, success_count, avg_confidence, avg_execution_time_ms FROM templates WHERE id = ?",
            (template_id,)
        )
        row = cursor.fetchone()

        if row:
            usage_count = row[0] + 1
            success_count = row[1] + (1 if success else 0)

            # Running average for confidence
            old_avg_conf = row[2]
            new_avg_conf = old_avg_conf + (confidence - old_avg_conf) / usage_count

            # Running average for execution time
            old_avg_time = row[3]
            new_avg_time = int(old_avg_time + (execution_time_ms - old_avg_time) / usage_count)

            cursor.execute("""
                UPDATE templates
                SET usage_count = ?, success_count = ?,
                    avg_confidence = ?, avg_execution_time_ms = ?,
                    updated_at = ?, last_used_at = ?
                WHERE id = ?
            """, (
                usage_count, success_count, new_avg_conf, new_avg_time,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                template_id
            ))

            conn.commit()

            if success:
                self._stats["successful_reuses"] += 1

        conn.close()

    def get_top_templates(
        self,
        limit: int = 10,
        template_type: Optional[TemplateType] = None
    ) -> List[DistilledTemplate]:
        """Get top performing templates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = """
            SELECT * FROM templates
            WHERE usage_count >= 3
        """
        params = []

        if template_type:
            sql += " AND template_type = ?"
            params.append(template_type.value)

        sql += """
            ORDER BY CAST(success_count AS REAL) / CASE WHEN usage_count = 0 THEN 1 ELSE usage_count END DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._load_template_from_row(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-buffer statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM templates")
        total_templates = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(usage_count), SUM(success_count) FROM templates")
        row = cursor.fetchone()
        total_usages = row[0] or 0
        total_successes = row[1] or 0

        cursor.execute("SELECT AVG(avg_confidence) FROM templates WHERE usage_count >= 3")
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            **self._stats,
            "total_templates": total_templates,
            "total_usages": total_usages,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / max(total_usages, 1),
            "avg_template_confidence": avg_confidence,
            "hot_cache_size": len(self._hot_cache)
        }


# Singleton instance
_meta_buffer: Optional[MetaBuffer] = None


def get_meta_buffer(
    db_path: Optional[str] = None,
    ollama_url: str = "http://localhost:11434"
) -> MetaBuffer:
    """Get or create singleton MetaBuffer instance"""
    global _meta_buffer
    if _meta_buffer is None:
        _meta_buffer = MetaBuffer(db_path=db_path, ollama_url=ollama_url)
    return _meta_buffer
