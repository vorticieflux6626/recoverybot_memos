# HSEA Corpus Ingestion Plan

## Executive Summary

This document outlines a comprehensive plan to ingest all documents from `PDF_Extraction_Tools/` into the HSEA (Hierarchical Stratified Embedding Architecture) system, creating searchable entity graphs with three-layer contextual search for FANUC robotics troubleshooting.

**Target Corpus:**
- 8,449 error code entities across 105 categories
- 8,626 total graph nodes with 17 edge types
- Source: `PDF_Extraction_Tools/data/graph.pkl`

**Expected Outcomes:**
- Sub-50ms contextual search latency
- 3-layer troubleshooting context (systemic → structural → substantive)
- +15-25% recall improvement via MRL + HyDE + RRF

---

## Phase 1: Pre-Ingestion Analysis & Optimization

### 1.1 Graph Structure Assessment

```
Current PDF Extraction Graph:
┌─────────────────────────────────────────────────────────────────┐
│  Node Types (9)           │  Edge Types (17)                   │
├───────────────────────────┼────────────────────────────────────┤
│  document, section        │  contains, follows, references     │
│  chunk, entity            │  cross_references, cites           │
│  error_code, category     │  similar_to, caused_by             │
│  concept, metadata        │  remedied_by, relates_to           │
│  relationship             │  parent_of, child_of, etc.         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Scalability Bottleneck Resolution

**Critical Issue Identified:** `SemanticMemoryNetwork.auto_connect()` has O(n²) complexity.

For 8,449 entities: `8,449 × 8,449 = 71,385,601 similarity comparisons`

**Solution: HNSW Spatial Indexing**

```python
# Before: O(n²) brute force
for i, entity_i in enumerate(entities):
    for j, entity_j in enumerate(entities[i+1:]):
        similarity = cosine_similarity(embed_i, embed_j)
        if similarity > threshold:
            create_edge(entity_i, entity_j)

# After: O(n log n) with HNSW
import hnswlib
index = hnswlib.Index(space='cosine', dim=256)
index.init_index(max_elements=10000, ef_construction=200, M=16)
index.add_items(embeddings, ids)
# Query k-nearest neighbors in O(log n)
labels, distances = index.knn_query(query_embedding, k=10)
```

**Performance Impact:**
| Approach | Time for 8,449 entities | Memory |
|----------|-------------------------|--------|
| Brute force O(n²) | ~2 hours | ~500MB |
| HNSW O(n log n) | ~5 minutes | ~200MB |

### 1.3 Memory Budget Calculation

```
Per-Entity Memory Requirements:
┌────────────────────────────────────────────────────────────────┐
│  π₁ Systemic (MRL 64d)   │  64 × 4 bytes = 256 bytes         │
│  π₂ Structural (MRL 256d)│  256 × 4 bytes = 1,024 bytes      │
│  π₃ Substantive (1024d)  │  1024 × 4 bytes = 4,096 bytes     │
│  Metadata overhead       │  ~500 bytes avg                    │
├────────────────────────────────────────────────────────────────┤
│  Total per entity        │  ~5.9 KB                           │
│  8,449 entities          │  ~49.8 MB embeddings               │
│  HNSW index overhead     │  ~150 MB                           │
│  Working memory buffer   │  ~200 MB                           │
├────────────────────────────────────────────────────────────────┤
│  Total RAM requirement   │  ~400-500 MB                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Data Extraction Pipeline

### 2.1 Entity Extraction from graph.pkl

Use the existing `export_to_hsea.py` bridge script:

```bash
cd /home/sparkone/sdd/PDF_Extraction_Tools

# Dry-run to verify extraction
python scripts/export_to_hsea.py --dry-run --sample 10

# Full extraction with statistics
python scripts/export_to_hsea.py --stats-only
```

**ErrorCodeExport Schema:**
```python
@dataclass
class ErrorCodeExport:
    entity_id: str           # "error_code_srvo_063"
    canonical_form: str      # "SRVO-063"
    title: str               # "SRVO-063 RCAL alarm(Group:%d Axis:%d)"
    category: str            # "SRVO"
    code_number: int         # 63
    cause: str               # Root cause explanation
    remedy: str              # Resolution steps
    severity: str            # "alarm", "warning", "error"
    related_codes: List[str] # Cross-references
    page_number: Optional[int]
```

### 2.2 Category Distribution (Top 15)

| Category | Count | Description |
|----------|-------|-------------|
| CVIS | 726 | Vision system errors |
| SVGN | 471 | Servo gain/tuning |
| INTP | 460 | Interpreter errors |
| MOTN | 455 | Motion control |
| SRVO | 429 | Servo motor/drive |
| PRIO | 367 | Priority/scheduling |
| IBSS | 301 | Internal bus |
| SEAL | 294 | Sealing operations |
| SYST | 293 | System-level |
| FORC | 244 | Force control |
| HOST | 203 | Host communication |
| COMM | 189 | Communication |
| TPIF | 175 | Teach pendant interface |
| FILE | 162 | File system |
| VARS | 155 | Variable errors |

---

## Phase 3: Three-Stratum Embedding Generation

### 3.1 Parallel Embedding Pipeline

```python
async def generate_embeddings_parallel(
    entities: List[ErrorCodeExport],
    batch_size: int = 32,
    workers: int = 4
) -> Dict[str, EmbeddingSet]:
    """
    Generate MRL embeddings for all three strata in parallel.

    Strategy: Process in batches across 4 worker tasks
    Expected time: 10-15 minutes for 8,449 entities
    """
    async with asyncio.TaskGroup() as tg:
        # Worker pool for embedding generation
        queue = asyncio.Queue()
        results = {}

        # Producer: batch entities
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            await queue.put(batch)

        # Signal workers to stop
        for _ in range(workers):
            await queue.put(None)

        # Workers: generate embeddings
        for worker_id in range(workers):
            tg.create_task(
                embedding_worker(queue, results, worker_id)
            )

    return results
```

### 3.2 MRL Dimension Truncation

Based on **Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022):

```python
def truncate_mrl(full_embedding: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Truncate full embedding to MRL dimensions.
    Full embedding: 4096d from BGE-M3
    """
    return {
        "systemic_64d": full_embedding[:64],      # π₁: Coarse semantics
        "structural_256d": full_embedding[:256],  # π₂: Balanced precision
        "substantive_1024d": full_embedding[:1024], # π₃: Fine-grained
        "full_4096d": full_embedding              # Backup for edge cases
    }
```

### 3.3 Index Structure Creation

```python
# π₁ Systemic: Binary index for fast filtering
systemic_index = hnswlib.Index(space='cosine', dim=64)
systemic_index.init_index(max_elements=10000, ef_construction=100, M=16)

# π₂ Structural: Int8 quantized index
structural_index = hnswlib.Index(space='cosine', dim=256)
structural_index.init_index(max_elements=10000, ef_construction=150, M=24)

# π₃ Substantive: Full precision store
substantive_store = {}  # entity_id -> full_embedding
```

---

## Phase 4: Semantic Memory Network Construction

### 4.1 Auto-Connection with HNSW Acceleration

```python
async def build_semantic_network_optimized(
    entities: List[ErrorCodeExport],
    embeddings: Dict[str, np.ndarray],
    similarity_threshold: float = 0.7,
    k_neighbors: int = 50
) -> SemanticMemoryNetwork:
    """
    Build semantic memory network with HNSW acceleration.

    O(n log n) instead of O(n²)
    Expected time: 2-5 minutes for 8,449 entities
    """
    network = SemanticMemoryNetwork()

    # Build HNSW index
    hnsw = hnswlib.Index(space='cosine', dim=256)
    hnsw.init_index(max_elements=len(entities) + 1000)

    # Add all embeddings
    entity_ids = list(embeddings.keys())
    embed_matrix = np.array([embeddings[eid] for eid in entity_ids])
    hnsw.add_items(embed_matrix, list(range(len(entity_ids))))

    # Query k-nearest neighbors for each entity
    for idx, entity_id in enumerate(entity_ids):
        labels, distances = hnsw.knn_query(embed_matrix[idx], k=k_neighbors)

        for neighbor_idx, distance in zip(labels[0], distances[0]):
            if neighbor_idx != idx:
                similarity = 1 - distance  # cosine distance to similarity
                if similarity >= similarity_threshold:
                    neighbor_id = entity_ids[neighbor_idx]
                    network.add_connection(
                        entity_id, neighbor_id,
                        weight=similarity,
                        edge_type="semantically_similar"
                    )

    return network
```

### 4.2 Category Anchor Generation

```python
def generate_category_anchors(
    entities: List[ErrorCodeExport],
    embeddings: Dict[str, np.ndarray]
) -> Dict[str, CategoryAnchor]:
    """
    Create π₁ systemic layer category anchors.

    Each anchor is the centroid of all entities in that category.
    """
    anchors = {}

    # Group entities by category
    by_category = defaultdict(list)
    for entity in entities:
        by_category[entity.category].append(entity)

    for category, cat_entities in by_category.items():
        # Compute centroid embedding
        cat_embeddings = [embeddings[e.entity_id] for e in cat_entities]
        centroid = np.mean(cat_embeddings, axis=0)

        # Determine common troubleshooting patterns
        patterns = infer_troubleshooting_patterns(cat_entities)

        anchors[category] = CategoryAnchor(
            category_id=category,
            centroid_embedding=centroid,
            entity_count=len(cat_entities),
            common_patterns=patterns,
            severity_distribution=compute_severity_dist(cat_entities)
        )

    return anchors
```

---

## Phase 5: Ingestion Execution Workflow

### 5.1 Complete Ingestion Command

```bash
# Step 1: Start memOS server if not running
cd /home/sparkone/sdd/Recovery_Bot/memOS
./memos_server.sh start

# Step 2: Verify HSEA endpoints
curl http://localhost:8001/api/v1/search/hsea/stats

# Step 3: Extract and index all entities
cd /home/sparkone/sdd/PDF_Extraction_Tools
python scripts/export_to_hsea.py \
    --graph data/graph.pkl \
    --memos-url http://localhost:8001 \
    --batch-size 50

# Step 4: Verify indexing
curl http://localhost:8001/api/v1/search/hsea/stats
```

### 5.2 Batch Ingestion API

```bash
# Batch indexing endpoint
POST /api/v1/search/hsea/index/batch
Content-Type: application/json

{
    "entities": [
        {
            "entity_id": "error_code_srvo_063",
            "canonical_form": "SRVO-063",
            "title": "SRVO-063 RCAL alarm(Group:%d Axis:%d)",
            "category": "SRVO",
            "code_number": 63,
            "cause": "The robot control detected an error in the pulsecoder...",
            "remedy": "Replace the pulsecoder or encoder cable...",
            "severity": "alarm",
            "related_codes": ["SRVO-068", "SRVO-069"],
            "page_number": 142
        }
        // ... more entities
    ]
}
```

### 5.3 Expected Ingestion Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Extraction | 30 seconds | Load graph.pkl, convert to entities |
| 2. Embedding Gen | 10-12 minutes | BGE-M3 embeddings (GPU accelerated) |
| 3. MRL Truncation | 15 seconds | Create 64d/256d/1024d views |
| 4. Index Building | 1-2 minutes | HNSW index construction |
| 5. Network Build | 2-3 minutes | Semantic auto-connections |
| 6. Anchor Gen | 30 seconds | Category centroids |
| 7. Verification | 30 seconds | Stats and health checks |
| **Total** | **15-18 minutes** | Full corpus ingestion |

---

## Phase 6: Post-Ingestion Verification

### 6.1 Health Check Queries

```bash
# Verify system statistics
curl http://localhost:8001/api/v1/search/hsea/stats

# Expected response:
{
    "success": true,
    "data": {
        "total_entities": 8449,
        "total_categories": 105,
        "total_patterns": 7,
        "index_sizes": {
            "systemic_64d": 8449,
            "structural_256d": 8449,
            "substantive_1024d": 8449
        },
        "auto_connections": 42000,  # ~5 per entity avg
        "memory_usage_mb": 450
    }
}
```

### 6.2 Test Queries

```bash
# Test 1: Contextual search
curl -X POST http://localhost:8001/api/v1/search/hsea/search \
    -H "Content-Type: application/json" \
    -d '{"query": "encoder replacement RCAL alarm", "mode": "contextual", "top_k": 5}'

# Test 2: Troubleshooting context
curl http://localhost:8001/api/v1/search/hsea/troubleshoot/SRVO-063

# Test 3: Similar codes
curl http://localhost:8001/api/v1/search/hsea/similar/SRVO-063?top_k=10
```

### 6.3 Performance Benchmarks

| Query Type | Target Latency | Pass Criteria |
|------------|---------------|---------------|
| Contextual search | < 50ms | 95th percentile |
| Troubleshoot lookup | < 20ms | 95th percentile |
| Similar codes | < 30ms | 95th percentile |
| Batch index (50) | < 5s | Per batch |

---

## Phase 7: Integration with Agentic Search

### 7.1 UniversalOrchestrator Integration

```python
# In orchestrator_universal.py

class UniversalOrchestrator:
    async def _gather_domain_context(
        self,
        query: str,
        classified_query: QueryClassification
    ) -> Optional[DomainContext]:
        """
        Gather HSEA domain context for technical queries.

        Called when query classification detects:
        - domain == "technical" or "engineering"
        - contains error code patterns (SRVO-XXX, MOTN-XXX, etc.)
        """
        if not self.hsea_controller:
            return None

        # Detect error codes in query
        error_pattern = r'[A-Z]{3,4}-\d{3}'
        codes = re.findall(error_pattern, query)

        if codes:
            # Direct troubleshooting lookup
            contexts = []
            for code in codes:
                ctx = await self.hsea_controller.get_troubleshooting_context(code)
                if ctx:
                    contexts.append(ctx)
            return DomainContext(error_codes=codes, contexts=contexts)

        # Semantic search for technical terms
        hsea_results = await self.hsea_controller.search(
            query=query,
            mode=HSEASearchMode.CONTEXTUAL,
            top_k=3
        )

        return DomainContext(
            semantic_matches=hsea_results,
            confidence=hsea_results[0].score if hsea_results else 0.0
        )
```

### 7.2 Context Injection into Synthesis

```python
async def _synthesize_with_domain_context(
    self,
    query: str,
    web_results: List[SearchResult],
    domain_context: Optional[DomainContext]
) -> str:
    """
    Synthesize answer with three-layer HSEA context.
    """
    context_parts = []

    if domain_context and domain_context.contexts:
        for ctx in domain_context.contexts:
            context_parts.append(f"""
## {ctx.error_code}: {ctx.title}

**Category Context (π₁):**
- Category: {ctx.layer_1_context.category_anchor}
- Troubleshooting Patterns: {', '.join(ctx.layer_1_context.patterns)}

**Relationship Context (π₂):**
- Related Codes: {', '.join(ctx.layer_2_context.related_codes[:5])}
- Cluster Members: {', '.join(ctx.layer_2_context.cluster_members[:5])}

**Detailed Context (π₃):**
- Cause: {ctx.layer_3_context.cause[:500]}
- Remedy: {ctx.layer_3_context.remedy[:500]}
""")

    # Inject into synthesis prompt
    synthesis_prompt = f"""
You are a FANUC robotics troubleshooting expert.

## Domain Knowledge (from HSEA):
{chr(10).join(context_parts)}

## Web Search Results:
{format_web_results(web_results)}

## User Query:
{query}

Provide a comprehensive troubleshooting answer combining the domain knowledge
and web search results. Cite specific error codes and their remedies.
"""

    return await self._llm_synthesize(synthesis_prompt)
```

---

## Phase 8: Maintenance & Updates

### 8.1 Incremental Updates

```python
async def incremental_index_update(
    new_entities: List[ErrorCodeExport],
    hsea_controller: HSEAController
) -> UpdateResult:
    """
    Add new entities without full reindex.

    Use when new PDFs are processed.
    """
    # Generate embeddings for new entities only
    new_embeddings = await generate_embeddings(new_entities)

    # Add to HNSW indices (O(n log n) per entity)
    for entity in new_entities:
        hsea_controller.add_entity(entity, new_embeddings[entity.entity_id])

    # Update auto-connections (neighbors of new entities only)
    await hsea_controller.update_connections(new_entities)

    # Update category anchors if needed
    affected_categories = set(e.category for e in new_entities)
    await hsea_controller.recompute_anchors(affected_categories)

    return UpdateResult(
        added=len(new_entities),
        connections_updated=True,
        anchors_recomputed=list(affected_categories)
    )
```

### 8.2 Scheduled Maintenance

```bash
# Cron job: nightly index optimization
0 3 * * * python /home/sparkone/sdd/Recovery_Bot/memOS/scripts/hsea_maintenance.py --optimize

# Weekly: full reindex (if graph.pkl updated)
0 4 * * 0 python /home/sparkone/sdd/PDF_Extraction_Tools/scripts/export_to_hsea.py --memos-url http://localhost:8001 --batch-size 100
```

---

## Research Basis & References

This plan incorporates techniques from:

1. **Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022)
   - Progressive dimension truncation for multi-resolution search

2. **GraphRAG** (Microsoft Research, 2024)
   - Community-based summarization for global queries

3. **HybridRAG** (Sarmah et al., ACM ICAIF 2024)
   - Combining knowledge graphs with vector retrieval

4. **BGE-M3** (BAAI, 2024)
   - Multi-granularity embeddings (dense + sparse in single model)

5. **HNSW** (Malkov & Yashunin, 2018)
   - Hierarchical Navigable Small World graphs for ANN search

6. **RAGFlow** (InfiniFlow, 2024)
   - Production RAG with semantic chunking and deduplication

7. **Reciprocal Rank Fusion** (Cormack et al., SIGIR 2009)
   - Score merging formula: `score(d) = Σ 1/(60 + rank_i(d))`

---

## Appendix A: File Locations

| File | Purpose |
|------|---------|
| `PDF_Extraction_Tools/scripts/export_to_hsea.py` | Bridge script |
| `PDF_Extraction_Tools/data/graph.pkl` | Source graph |
| `memOS/server/agentic/hsea_controller.py` | HSEA controller |
| `memOS/server/agentic/mixed_precision_embeddings.py` | MRL indexing |
| `memOS/server/agentic/semantic_memory.py` | Auto-connection network |
| `memOS/server/agentic/bge_m3_hybrid.py` | Dense + BM25 retrieval |

## Appendix B: Quick Start

```bash
# Complete ingestion in 3 commands:

# 1. Ensure memOS is running
curl http://localhost:8001/health

# 2. Run ingestion
cd /home/sparkone/sdd/PDF_Extraction_Tools
python scripts/export_to_hsea.py --memos-url http://localhost:8001 --batch-size 50

# 3. Verify
curl http://localhost:8001/api/v1/search/hsea/stats
```

---

*Document Version: 1.0*
*Created: 2025-12-29*
*Author: Claude Code (memOS Development)*
