# memOS + PDF Extraction Tools Integration Report

**Generated:** 2025-12-29
**Status:** Phase 1 Complete - Ready for Integration
**PDF Extraction Tools API:** Running on port 8002

---

## Executive Summary

This report outlines the integration requirements between the memOS server (Recovery_Bot) and the PDF Extraction Tools API for enabling RAG-powered navigation of technical documentation (FANUC manuals).

### Key Decision: Hybrid Microservice Architecture

After analyzing both codebases, the recommended approach is to keep PDF Extraction Tools as a **separate microservice** that integrates with memOS via REST API, rather than merging the codebases.

**Rationale:**
- Separation of concerns (document processing vs. retrieval orchestration)
- Scalability (CPU-intensive PDF processing isolated from real-time queries)
- Code reuse (~70% infrastructure already exists)
- Clean integration surface via well-defined API

---

## Current State

### memOS Server (Port 8001)
| Component | Status | Description |
|-----------|--------|-------------|
| FastAPI Server | ✅ Ready | Production-grade API with JWT auth |
| UniversalOrchestrator | ✅ Ready | 5 presets, 42+ configurable features |
| EmbeddingService | ✅ Ready | Ollama + mxbai-embed-large (1024d) |
| DomainCorpus | ✅ Ready | Schema-driven entity definitions |
| EntityTracker | ✅ Ready | GSW-style extraction + coreference |
| EmbeddingAggregator | ✅ Ready | MoE routing + RRF fusion |
| PostgreSQL + pgvector | ✅ Ready | Vector storage with HIPAA compliance |
| Redis Cache | ✅ Ready | Session and embedding caching |

### PDF Extraction Tools API (Port 8002)
| Component | Status | Description |
|-----------|--------|-------------|
| FastAPI Server | ✅ Ready | New API layer (Phase 1 complete) |
| Document Ingestion | ✅ Ready | PDF → Graph pipeline |
| UnifiedDocumentGraph | ✅ Ready | Hypergraph with cross-doc edges |
| GraphAlgorithms | ✅ Ready | A* pathfinding, clustering |
| PathRAG Traversal | ✅ Ready | Flow-based pruning endpoints |
| Search Engine | ✅ Ready | BM25F + hybrid ranking |
| Neo4j Persistence | ✅ Ready | Optional graph database |

---

## Integration Requirements

### 1. New memOS Service: DocumentGraphService

Create a new service in memOS that acts as a bridge to the PDF Extraction Tools API.

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/core/document_graph_service.py`

```python
"""
Document Graph Service - Bridge to PDF Extraction Tools API

Provides memOS with access to document graph operations,
PathRAG traversal, and technical documentation search.
"""

import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class TroubleshootingStep:
    """A step in a troubleshooting path"""
    node_id: str
    title: str
    content: str
    step_type: str  # 'error', 'diagnosis', 'solution', 'procedure'
    relevance_score: float

@dataclass
class DocumentSearchResult:
    """Result from document graph search"""
    node_id: str
    title: str
    content_preview: str
    score: float
    document_path: List[str]
    matched_terms: List[str]

class DocumentGraphService:
    """
    Bridge between memOS retrieval and PDF Extraction Tools graphs

    Provides:
    - PathRAG traversal for troubleshooting chains
    - Semantic search across technical documentation
    - Graph-based context retrieval for RAG
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.pdf_api_url  # Add to settings
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        self._cache = {}  # Redis integration recommended

    async def query_troubleshooting_path(
        self,
        error_code: str,
        robot_model: Optional[str] = None,
        max_hops: int = 5
    ) -> List[TroubleshootingStep]:
        """
        PathRAG traversal for error resolution

        Args:
            error_code: FANUC error code (e.g., 'SRVO-023')
            robot_model: Optional robot model filter
            max_hops: Maximum traversal depth

        Returns:
            Ordered list of troubleshooting steps
        """
        try:
            response = await self.client.get(
                f"/traverse/troubleshoot/{error_code}",
                params={"max_hops": max_hops}
            )
            response.raise_for_status()
            data = response.json()

            steps = []
            for path in data.get('paths', []):
                for step in path.get('steps', []):
                    steps.append(TroubleshootingStep(
                        node_id=step['node_id'],
                        title=step['title'],
                        content=step.get('content', ''),
                        step_type=self._classify_step_type(step),
                        relevance_score=step.get('score', 0.0)
                    ))

            return steps

        except Exception as e:
            logger.error(f"Troubleshooting query failed: {e}")
            return []

    async def search_documentation(
        self,
        query: str,
        search_type: str = "hybrid",
        max_results: int = 10
    ) -> List[DocumentSearchResult]:
        """
        Search technical documentation

        Args:
            query: Search query
            search_type: 'keyword', 'semantic', or 'hybrid'
            max_results: Maximum results to return

        Returns:
            List of search results with relevance scores
        """
        try:
            response = await self.client.post(
                "/search",
                json={
                    "query": query,
                    "search_type": search_type,
                    "max_results": max_results
                }
            )
            response.raise_for_status()
            data = response.json()

            return [
                DocumentSearchResult(
                    node_id=r['node_id'],
                    title=r['title'],
                    content_preview=r['content_preview'],
                    score=r['score'],
                    document_path=r.get('path', []),
                    matched_terms=r.get('matched_terms', [])
                )
                for r in data.get('results', [])
            ]

        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return []

    async def get_related_sections(
        self,
        node_id: str,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get sections related to a specific node

        Args:
            node_id: Starting node ID
            depth: Traversal depth

        Returns:
            List of related sections with metadata
        """
        try:
            response = await self.client.get(
                f"/graph/nodes/{node_id}/neighbors",
                params={"max_depth": depth}
            )
            response.raise_for_status()
            return response.json().get('neighbors', {})

        except Exception as e:
            logger.error(f"Related sections query failed: {e}")
            return []

    async def get_context_for_rag(
        self,
        query: str,
        context_type: str = "troubleshooting"
    ) -> str:
        """
        Get formatted context for RAG prompts

        Args:
            query: User query
            context_type: Type of context needed

        Returns:
            Formatted context string for LLM prompt
        """
        context_parts = []

        # Search for relevant documentation
        results = await self.search_documentation(query, max_results=5)

        if results:
            context_parts.append("## Relevant Documentation\n")
            for r in results:
                context_parts.append(f"### {r.title}")
                context_parts.append(f"{r.content_preview}\n")

        # If query contains error code, get troubleshooting path
        import re
        error_pattern = r'[A-Z]{4}-\d{3,4}'
        error_codes = re.findall(error_pattern, query.upper())

        if error_codes:
            for code in error_codes[:2]:  # Limit to 2 codes
                steps = await self.query_troubleshooting_path(code)
                if steps:
                    context_parts.append(f"\n## Troubleshooting: {code}\n")
                    for i, step in enumerate(steps, 1):
                        context_parts.append(f"{i}. **{step.title}**")
                        if step.content:
                            context_parts.append(f"   {step.content[:200]}...")

        return "\n".join(context_parts)

    def _classify_step_type(self, step: Dict) -> str:
        """Classify the type of troubleshooting step"""
        title = step.get('title', '').lower()

        if any(kw in title for kw in ['error', 'alarm', 'fault']):
            return 'error'
        elif any(kw in title for kw in ['cause', 'diagnos', 'check']):
            return 'diagnosis'
        elif any(kw in title for kw in ['solution', 'remedy', 'fix', 'action']):
            return 'solution'
        elif any(kw in title for kw in ['procedure', 'step', 'instruction']):
            return 'procedure'
        else:
            return 'info'

    async def health_check(self) -> bool:
        """Check if PDF Extraction Tools API is available"""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except:
            return False

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance
document_graph_service = DocumentGraphService()
```

### 2. Settings Configuration Update

Add PDF API configuration to memOS settings.

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/config/settings.py`

Add these fields to the Settings class:

```python
# PDF Extraction Tools API
pdf_api_url: str = "http://localhost:8002"
pdf_api_timeout: int = 30
pdf_api_enabled: bool = True
```

### 3. FANUC Domain Schema

Create a domain schema for FANUC entity extraction.

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/schemas/fanuc_schema.py`

```python
"""
FANUC Domain Schema for Entity Extraction

Defines entity types and patterns specific to FANUC robot documentation.
"""

from ..domain_corpus import DomainSchema, DomainEntityDef

FANUC_SCHEMA = DomainSchema(
    name="fanuc_robotics",
    description="FANUC robot technical documentation",
    entities=[
        # Error Codes
        DomainEntityDef(
            name="error_code",
            description="FANUC alarm/error codes",
            pattern=r"(SRVO|MOTN|SYST|PROG|INTP|HOST|SSPC|MACR|MCTL|PRIO|DICT|MEMO|FILE|COND|PALT|SPOT|ACAL|APSH|TOOL|PICK|WEAV|TAST|ARC|WELD)-\d{3,4}",
            examples=["SRVO-001", "MOTN-023", "SYST-100"]
        ),

        # Robot Models
        DomainEntityDef(
            name="robot_model",
            description="FANUC robot model identifiers",
            pattern=r"(Arc Mate|M-\d+i[A-Z]?(/\d+[A-Z]?)?|LR Mate|CR-\d+i[A-Z]?|R-\d+i[A-Z]?)",
            examples=["Arc Mate 50iD", "M-20iA", "LR Mate 200iD"]
        ),

        # Parameters
        DomainEntityDef(
            name="parameter",
            description="System parameters and variables",
            pattern=r"\$[A-Z_]+(\[\d+\])?(\.[A-Z_]+)?",
            examples=["$SCR_GRP[1]", "$PARAM_GROUP", "$MOTYPE"]
        ),

        # I/O Signals
        DomainEntityDef(
            name="io_signal",
            description="Input/Output signal references",
            pattern=r"(DI|DO|RI|RO|UI|UO|SI|SO|WI|WO|GI|GO|AI|AO)\[\d+\]",
            examples=["DI[1]", "DO[101]", "RI[5]"]
        ),

        # Registers
        DomainEntityDef(
            name="register",
            description="Register references",
            pattern=r"(R|PR|SR|AR|VR)\[\d+\]",
            examples=["R[1]", "PR[10]", "SR[5]"]
        ),

        # Procedures
        DomainEntityDef(
            name="procedure",
            description="Technical procedures and methods",
            keywords=["procedure", "steps", "method", "process", "calibration", "mastering"],
            context_window=100
        ),

        # Components
        DomainEntityDef(
            name="component",
            description="Robot components and parts",
            keywords=[
                "servo motor", "encoder", "brake", "reducer", "cable",
                "teach pendant", "controller", "amplifier", "axis", "joint",
                "wrist", "arm", "base", "tool flange", "end effector"
            ],
            context_window=50
        ),

        # Safety Terms
        DomainEntityDef(
            name="safety_term",
            description="Safety-related terms and concepts",
            keywords=[
                "emergency stop", "e-stop", "deadman", "safety fence",
                "DCS", "dual check safety", "safe speed", "safe position"
            ],
            context_window=50
        ),

        # Measurement Values
        DomainEntityDef(
            name="measurement",
            description="Numeric measurements with units",
            pattern=r"\d+\.?\d*\s*(mm|cm|m|deg|°|rpm|A|V|kW|N·m|Nm|kg|lb|ms|sec|s|Hz|%)",
            examples=["100mm", "45deg", "1500rpm", "24V"]
        )
    ],

    # Relationships between entities
    relationships=[
        ("error_code", "causes", "component"),
        ("error_code", "requires", "procedure"),
        ("procedure", "uses", "parameter"),
        ("procedure", "uses", "io_signal"),
        ("robot_model", "has", "component"),
        ("component", "measured_by", "measurement")
    ]
)
```

### 4. UniversalOrchestrator Integration

Add PDF documentation search as a new search provider in the orchestrator.

**Modifications to:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/universal_orchestrator.py`

```python
# Add to imports
from core.document_graph_service import document_graph_service

# Add new search provider method
async def _search_technical_documentation(
    self,
    query: str,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Search FANUC technical documentation via PDF Extraction Tools API
    """
    if not config.get('search_technical_docs', False):
        return []

    try:
        results = await document_graph_service.search_documentation(
            query=query,
            search_type="hybrid",
            max_results=config.get('max_doc_results', 5)
        )

        return [
            {
                'source': 'technical_documentation',
                'node_id': r.node_id,
                'title': r.title,
                'content': r.content_preview,
                'score': r.score,
                'path': r.document_path
            }
            for r in results
        ]

    except Exception as e:
        logger.warning(f"Technical documentation search failed: {e}")
        return []

# Add to orchestrator presets
ORCHESTRATOR_PRESETS['TECHNICAL'] = {
    'search_technical_docs': True,
    'max_doc_results': 10,
    'enable_error_code_extraction': True,
    'enable_pathrag_traversal': True,
    # ... other settings
}
```

### 5. API Route for Technical Queries

Add a new API endpoint for technical documentation queries.

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/api/technical.py`

```python
"""
Technical Documentation API Routes

Provides endpoints for FANUC manual queries and troubleshooting.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel

from core.document_graph_service import document_graph_service

router = APIRouter(prefix="/api/technical", tags=["Technical Documentation"])


class TroubleshootRequest(BaseModel):
    error_code: str
    robot_model: Optional[str] = None
    max_steps: int = 10


class TroubleshootResponse(BaseModel):
    error_code: str
    found: bool
    steps: List[dict]
    context: str


@router.post("/troubleshoot", response_model=TroubleshootResponse)
async def troubleshoot_error(request: TroubleshootRequest):
    """
    Get troubleshooting steps for a FANUC error code
    """
    steps = await document_graph_service.query_troubleshooting_path(
        error_code=request.error_code,
        robot_model=request.robot_model,
        max_hops=request.max_steps
    )

    context = await document_graph_service.get_context_for_rag(
        query=f"Error {request.error_code}",
        context_type="troubleshooting"
    )

    return TroubleshootResponse(
        error_code=request.error_code,
        found=len(steps) > 0,
        steps=[
            {
                'title': s.title,
                'content': s.content,
                'type': s.step_type,
                'score': s.relevance_score
            }
            for s in steps
        ],
        context=context
    )


@router.get("/search")
async def search_documentation(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search technical documentation
    """
    results = await document_graph_service.search_documentation(
        query=q,
        max_results=limit
    )

    return {
        "query": q,
        "total": len(results),
        "results": [
            {
                "node_id": r.node_id,
                "title": r.title,
                "preview": r.content_preview,
                "score": r.score,
                "path": r.document_path
            }
            for r in results
        ]
    }


@router.get("/health")
async def check_pdf_api_health():
    """
    Check PDF Extraction Tools API health
    """
    is_healthy = await document_graph_service.health_check()
    return {
        "pdf_api_available": is_healthy,
        "pdf_api_url": document_graph_service.base_url
    }
```

---

## Database Schema Updates

### New PostgreSQL Tables

```sql
-- Document references table
CREATE TABLE document_references (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) NOT NULL,
    document_title VARCHAR(500),
    document_path VARCHAR(1000),
    page_number INTEGER,
    content_hash VARCHAR(64),
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_doc_ref_node_id ON document_references(node_id);
CREATE INDEX idx_doc_ref_embedding ON document_references USING ivfflat (embedding vector_cosine_ops);

-- Error code mappings
CREATE TABLE error_code_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    error_code VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    severity VARCHAR(20),
    robot_models TEXT[],
    related_nodes TEXT[],
    troubleshooting_path JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_error_code ON error_code_mappings(error_code);

-- Search cache for frequent queries
CREATE TABLE search_cache (
    query_hash VARCHAR(64) PRIMARY KEY,
    query_text TEXT,
    results JSONB,
    hit_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);
```

---

## Implementation Checklist

### Phase 2: memOS Integration (Estimated: 1-2 days)

- [ ] Create `DocumentGraphService` in memOS
- [ ] Add PDF API configuration to settings
- [ ] Create FANUC domain schema
- [ ] Add technical documentation search provider to orchestrator
- [ ] Create `/api/technical` routes
- [ ] Run database migrations
- [ ] Update Docker Compose for multi-service deployment
- [ ] Write integration tests

### Phase 3: PathRAG Enhancement (Estimated: 2-3 days)

- [ ] Implement flow-based pruning improvements
- [ ] Add troubleshooting path caching in Redis
- [ ] Create error code → solution knowledge graph
- [ ] Implement cross-document path finding
- [ ] Add path quality scoring

### Phase 4: FANUC Corpus Ingestion (Estimated: 1-2 days)

- [ ] Batch ingest 137 FANUC manuals
- [ ] Extract and index all error codes
- [ ] Build error → diagnosis → solution graph
- [ ] Generate embeddings for all chunks
- [ ] Verify search quality

---

## Environment Configuration

### Docker Compose Addition

```yaml
# Add to docker-compose.yml
services:
  pdf-api:
    build:
      context: ../PDF_Extraction_Tools
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - DATA_DIR=/data
      - OUTPUT_DIR=/output
    volumes:
      - pdf_data:/data
      - pdf_output:/output
      - /home/sparkone/Documents/FANUC_manuals:/manuals:ro
    depends_on:
      - ollama
    networks:
      - memos-network

volumes:
  pdf_data:
  pdf_output:
```

### Environment Variables

```bash
# Add to .env
PDF_API_URL=http://localhost:8002
PDF_API_TIMEOUT=30
PDF_API_ENABLED=true

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## Testing Plan

### Unit Tests
1. `DocumentGraphService` connection and methods
2. FANUC entity extraction patterns
3. Error code parsing and validation

### Integration Tests
1. memOS → PDF API → Graph traversal flow
2. Search results ranking accuracy
3. PathRAG troubleshooting path quality

### End-to-End Tests
1. "What causes SRVO-023?" → Troubleshooting steps
2. "How do I calibrate the wrist?" → Procedure retrieval
3. "Arc Mate 50iD specifications" → Documentation search

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search latency | < 500ms | P95 response time |
| PathRAG traversal | < 1s | P95 for 5-hop paths |
| Error code coverage | > 95% | Codes found vs. total |
| Search relevance | > 0.8 | MRR@10 score |
| API availability | > 99.5% | Uptime monitoring |

---

## Next Steps

1. **Immediate:** Review this report and approve integration approach
2. **Week 1:** Implement DocumentGraphService and API routes
3. **Week 2:** Integrate with UniversalOrchestrator
4. **Week 3:** Batch ingest FANUC manuals
5. **Week 4:** Testing and optimization

---

## References

- PDF Extraction Tools API: `http://localhost:8002/docs`
- memOS API: `http://localhost:8001/docs`
- LightRAG Research: `/home/sparkone/sdd/PDF_Extraction_Tools/docs/LIGHTRAG_RESEARCH.md`
- PathRAG Research: `/home/sparkone/sdd/PDF_Extraction_Tools/docs/PATHRAG_RESEARCH.md`
