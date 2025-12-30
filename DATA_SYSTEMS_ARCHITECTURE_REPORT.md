# memOS Data Systems Architecture Report

**Date:** 2025-12-29
**Project:** memOS Server - Recovery Bot
**Scope:** Comprehensive audit of data storage architecture and alignment with project capabilities
**Method:** 12 parallel research agents (3 codebase review + 3 best practices + 6 audit)

---

## Executive Summary

This report examines whether memOS is storing data in ways that make use of the project's architecture. The audit reveals **significant misalignment** between configured infrastructure capabilities and actual implementation.

### Key Finding: Infrastructure Underutilization

| Infrastructure | Configured | Actually Used | Verdict |
|----------------|------------|---------------|---------|
| PostgreSQL + pgvector | Yes | Partial (Mem0 only) | **UNDERUTILIZED** |
| Redis | Yes (2GB) | **Not at all** | **UNUSED** |
| Row-Level Security | No | No | **MISSING** |
| Full-Text Search | No | No | **MISSING** |
| Table Partitioning | No | No | **MISSING** |

### Storage Fragmentation

| Storage System | Size | Purpose |
|----------------|------|---------|
| PostgreSQL | ? | User memories (via Mem0) |
| SQLite: mixed_precision_embeddings.db | **195 MB** | Embeddings (should be pgvector) |
| SQLite: hsea_hybrid.db | 44 MB | HSEA index |
| SQLite: content_cache.db | 8.7 MB | Web content cache |
| SQLite: scratchpad_cache.db | 36 KB | Working memory cache |
| In-memory dicts | Variable | Entity tracker, semantic memory |

**Recommendation:** Consolidate 6+ SQLite databases into PostgreSQL with proper indexing.

---

## Part 1: Database Layer Audit

### 1.1 Connection Pooling - ADEQUATE

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/config/database.py:33-48`

```python
async_engine = create_async_engine(
    get_database_url(async_mode=True),
    pool_pre_ping=True,     # Good: health checks
    pool_size=10,           # Good: reasonable default
    max_overflow=20,        # Good: burst capacity
    # MISSING: pool_recycle=1800
    # MISSING: pool_timeout=30
)
```

| Configuration | Current | Recommended | Status |
|---------------|---------|-------------|--------|
| pool_pre_ping | True | True | OK |
| pool_size | 10 | 10 | OK |
| max_overflow | 20 | 20 | OK |
| pool_recycle | Missing | 1800 | **ADD** |
| pool_timeout | Missing | 30 | **ADD** |

### 1.2 Session Management - CRITICAL ISSUES

**Issue 1: Dual Session Patterns**

Two conflicting patterns exist:
- `get_db_dependency()` - No auto-commit (FastAPI endpoints)
- `get_async_db()` - Auto-commit (context manager)

**Issue 2: Services Create Own Sessions**

**File:** `memory_service.py:109-181`
```python
async def store_memory(self, ...):
    async with AsyncSessionLocal() as session:  # Creates OWN session!
        await session.commit()  # Internal commit
```

Services should accept session as parameter, not create their own.

**Issue 3: Nested Session Context**

**File:** `quest.py:118`
```python
async with db as session:  # WRONG: db IS already the session!
```

### 1.3 Migration System - CRITICAL: NONE

**Finding:** No Alembic migrations found.

| File | Purpose | Status |
|------|---------|--------|
| alembic/ | Migration directory | **MISSING** |
| alembic.ini | Configuration | **MISSING** |
| migrations/ | Alternative location | **MISSING** |

**Current approach:** `init_database.py` uses `Base.metadata.create_all()` - no versioning, no rollback capability.

**Impact:**
- Cannot track schema changes
- No zero-downtime migrations
- Cannot rollback failed deployments
- HIPAA audit trail incomplete

### 1.4 Eager Loading - MOSTLY CORRECT

Quest service correctly uses `selectinload`:
```python
query = select(UserQuest).options(
    selectinload(UserQuest.quest).selectinload(Quest.tasks),
    selectinload(UserQuest.tasks)
)
```

**Minor N+1 risk:** Memory stats queries could use aggregation.

---

## Part 2: Caching Architecture Audit

### 2.1 Multi-Tier Cache - INCOMPLETE

**Configured Architecture:**
```
L1 (In-Memory)  →  L2 (Redis)  →  L3 (SQLite)
     ↓                 ↓              ↓
  Python dict      NOT USED      content_cache.db
```

**Critical Finding: Redis Configured But Unused**

**File:** `settings.py:36-40`
```python
redis_host: str = "localhost"
redis_port: int = 6379
redis_db: int = 2
```

No actual Redis client usage found in codebase. All caching uses SQLite or in-memory dicts.

### 2.2 Cache Coherency - GAPS

| Issue | Severity | Description |
|-------|----------|-------------|
| L1/L3 sync race | MEDIUM | Write to memory, then SQLite - failure leaves inconsistent state |
| No cross-worker invalidation | HIGH | Each worker has independent L1 cache |
| No cache invalidation API | HIGH | Cannot force refresh before TTL |

### 2.3 TTL Strategies - MOSTLY CORRECT

| Cache | TTL | Status |
|-------|-----|--------|
| Content cache | 3600s | OK |
| Query cache | 900s | OK |
| KV cache pins | 1-60s | OK |

**Missing:** TTL jitter to prevent cache stampede.

### 2.4 Semantic Cache - UNDERUTILIZED

**File:** `base_pipeline.py:165-166`
```python
# Note: Semantic similarity search via find_similar_query requires
# pre-computed embeddings, so we skip it here for simplicity
```

The semantic cache implementation exists but is bypassed, missing 30-60% additional cache hits.

---

## Part 3: Memory Systems Audit

### 3.1 A-MEM (Semantic Memory) - NOT PERSISTED

**File:** `semantic_memory.py:168`
```python
self.memories: Dict[str, Memory] = {}  # In-memory only!
```

**Critical Issue:** The entire semantic memory network is lost on server restart.

- `to_dict()` / `from_dict()` serialization exists
- But no integration with PostgreSQL
- All A-MEM memories are ephemeral

### 3.2 RAISE Scratchpad - CORRECT BUT ISOLATED

The scratchpad correctly implements the four-component blackboard pattern:
- Observations
- Reasoning
- Examples
- Trajectory

**Gap:** No promotion from RAISE scratchpad to A-MEM long-term storage.

### 3.3 Entity Tracker (GSW) - FUNCTIONAL

Entity extraction with 51% token reduction claim is implemented, but:
- Claim is not instrumented/measured
- Coreference resolution is basic (exact string match only)
- No embedding-based entity matching

### 3.4 Memory Tiers - IMPLEMENTED BUT DISCONNECTED

Three-tier architecture exists:
- COLD: PlaintextStorage (in-memory dict, not PostgreSQL!)
- WARM: KV cache service
- HOT: LoRA weights (future, not implemented)

**Issue:** Cold tier claims to be PostgreSQL but uses in-memory dict.

---

## Part 4: Vector Storage Audit

### 4.1 pgvector Configuration - CRITICAL GAP

**Finding:** No HNSW or IVFFlat index on `embedding_vector` column.

**File:** `init_database.py:65-88`
- Creates B-tree indexes on scalar columns
- **NO vector index created**

**Impact:** Vector similarity queries are O(n) sequential scans.

**Fix Required:**
```sql
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
ON memories USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 4.2 Embedding Model - ADEQUATE

| Configuration | Value | Assessment |
|---------------|-------|------------|
| Model | mxbai-embed-large | Good local choice |
| Dimensions | 1024 | Matches model output |
| Alternative | bge-m3 | Better for hybrid search |

### 4.3 Hybrid Search - IMPLEMENTED

BGE-M3 hybrid search with RRF fusion is correctly implemented:
```python
rrf_score = dense_weight / (k + dense_rank) +
            sparse_weight / (k + sparse_rank)
```

But disabled by default (only ENHANCED/RESEARCH/FULL presets).

### 4.4 Embedding Caching - SUBOPTIMAL

**File:** `embedding_service.py:34-36`
```python
self._embedding_cache = {}
self._cache_max_size = 1000
```

Uses FIFO eviction instead of LRU. Hash function not stable across runs.

---

## Part 5: Data Flow Audit

### 5.1 Query Flow - CORRECT

```
Query → In-memory cache → Semantic cache → Search → Synthesis → Cache store
```

All layers checked in correct order.

### 5.2 Memory Creation Flow - BROKEN

Expected:
```
Search results → Scratchpad → Entity extraction → A-MEM promotion
```

Actual:
```
Search results → Scratchpad → Entity extraction → DEAD END
                                                  ↓
                                           (No promotion to A-MEM)
```

### 5.3 Blackboard Pattern - IMPLEMENTED BUT UNUSED

The scratchpad has full blackboard implementation:
- `write_public()`, `read_public()`
- `write_private()`, `read_private()`
- `register_kv_cache()`

**But no agent phase actually uses these methods.**

### 5.4 Data Duplication

| Data Type | Location 1 | Location 2 | Issue |
|-----------|------------|------------|-------|
| Embeddings | PostgreSQL/pgvector | SQLite (195 MB) | Dual storage |
| Entities | EntityTracker | Scratchpad | Per-session duplication |
| Query cache | content_cache.py | scratchpad_cache.py | Overlapping |

---

## Part 6: Architecture Alignment Audit

### 6.1 PostgreSQL - Underutilized

| Feature | Configured | Used | Recommendation |
|---------|------------|------|----------------|
| pgvector | Yes | Partial | Add HNSW indexes |
| RLS | No | No | **Add for HIPAA** |
| tsvector | No | No | Add for hybrid search |
| Partitioning | No | No | Add for 7-year retention |

### 6.2 Redis - Completely Unused

**Configured:** 2GB RAM allocation in docker-compose.yml
**Used:** Not at all

**Recommendation:** Either:
- Remove Redis (save 2GB RAM)
- Migrate SQLite caches to Redis

### 6.3 SQLite Fragmentation

Six separate SQLite databases should be consolidated:
- `content_cache.db` → PostgreSQL or Redis
- `scratchpad_cache.db` → Redis (ephemeral)
- `mixed_precision_embeddings.db` → **pgvector** (195 MB!)
- `hsea_hybrid.db` → pgvector
- `embedding_aggregator.db` → pgvector
- `fanuc_robotics.db` → PostgreSQL with tsvector

---

## Part 7: Issue Summary by Severity

### CRITICAL (7 issues)

| # | Issue | Location |
|---|-------|----------|
| 1 | No Alembic migrations | init_database.py |
| 2 | No HNSW index on vectors | init_database.py |
| 3 | A-MEM not persisted | semantic_memory.py:168 |
| 4 | Redis configured but unused | settings.py, docker-compose.yml |
| 5 | No cache invalidation mechanism | All cache files |
| 6 | Mixed session patterns | database.py, services |
| 7 | No RAISE→A-MEM promotion | orchestrator_universal.py |

### HIGH (8 issues)

| # | Issue | Location |
|---|-------|----------|
| 8 | Services create own sessions | memory_service.py:109 |
| 9 | No TTL jitter (cache stampede) | content_cache.py, scratchpad_cache.py |
| 10 | Blackboard pattern unused | orchestrator_universal.py |
| 11 | No cross-worker cache sync | All cache files |
| 12 | BM25 index not persisted | bge_m3_hybrid.py |
| 13 | FIFO cache instead of LRU | embedding_service.py:280 |
| 14 | No Row-Level Security | database.py |
| 15 | Memory decay not implemented | semantic_memory.py |

### MEDIUM (12 issues)

| # | Issue | Location |
|---|-------|----------|
| 16 | L1/L3 sync race condition | scratchpad_cache.py:233-267 |
| 17 | Semantic cache bypassed | base_pipeline.py:165-166 |
| 18 | ContentCache MAX_ENTRIES unused | content_cache.py:44 |
| 19 | Query cache embeddings not stored | orchestrator_universal.py:1523 |
| 20 | Cold tier uses memory not PostgreSQL | memory_tiers.py:110-113 |
| 21 | Generic entity extraction patterns | entity_tracker.py:296-320 |
| 22 | Basic coreference resolution | entity_tracker.py:568-581 |
| 23 | HOT tier not implemented | memory_tiers.py:48 |
| 24 | ColBERT disabled | bge_m3_hybrid.py:707 |
| 25 | MRL not in core memory | embedding_service.py |
| 26 | Linear semantic scan | content_cache.py:360-396 |
| 27 | Missing pool_recycle | database.py |

### LOW (5 issues)

| # | Issue | Location |
|---|-------|----------|
| 28 | Orphaned SQLite entries | scratchpad_cache.py:475-497 |
| 29 | No table partitioning | memory.py |
| 30 | Fixed quality weights | raise_scratchpad.py:569-573 |
| 31 | 51% token reduction unverified | entity_tracker.py |
| 32 | Hardcoded paths | search.py:134 |

---

## Part 8: Recommendations

### Phase 1: Critical Infrastructure (Week 1)

| Task | Files | Effort |
|------|-------|--------|
| Add HNSW index to embedding_vector | init_database.py | 1h |
| Implement Alembic migrations | New alembic/ directory | 4h |
| Fix session pattern (DI everywhere) | memory_service.py, all services | 8h |
| Add pool_recycle=1800 | database.py | 0.5h |

### Phase 2: Consolidate Storage (Week 2)

| Task | Files | Effort |
|------|-------|--------|
| Migrate embeddings to pgvector | mixed_precision_embeddings.py → database | 8h |
| Decide Redis: use or remove | docker-compose.yml, settings.py | 4h |
| Add A-MEM PostgreSQL persistence | semantic_memory.py | 8h |
| Implement cache invalidation API | content_cache.py, scratchpad_cache.py | 4h |

### Phase 3: Memory Integration (Week 3)

| Task | Files | Effort |
|------|-------|--------|
| Add RAISE→A-MEM promotion | orchestrator_universal.py | 4h |
| Enable blackboard in agents | orchestrator_universal.py (all phases) | 8h |
| Add memory decay/consolidation | semantic_memory.py | 4h |
| Fix cold tier to use PostgreSQL | memory_tiers.py | 4h |

### Phase 4: Security & Compliance (Week 4)

| Task | Files | Effort |
|------|-------|--------|
| Add Row-Level Security | database.py, init_database.py | 4h |
| Add table partitioning | memory.py, init_database.py | 4h |
| Add FOR UPDATE locking | quest_service_fixed.py | 2h |
| Add connection pool monitoring | database.py | 4h |

---

## Part 9: Architecture Vision

### Current State

```
┌─────────────────────────────────────────────────────────────┐
│                     Current Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PostgreSQL                 Redis (UNUSED)                 │
│   ┌─────────┐               ┌─────────┐                    │
│   │ Mem0    │               │ 2GB RAM │                    │
│   │ vectors │               │ wasted  │                    │
│   └─────────┘               └─────────┘                    │
│        ↕                                                    │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          SQLite Fragmentation (244 MB+)              │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │  │
│   │  │ content  │ │ embeddings│ │  hsea   │ │ cache  │  │  │
│   │  │  cache   │ │  195 MB  │ │  44 MB  │ │ scratch│  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │  │
│   └─────────────────────────────────────────────────────┘  │
│        ↕                                                    │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              In-Memory (Lost on Restart)             │  │
│   │  EntityTracker | SemanticMemory | EmbeddingCache    │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Target State

```
┌─────────────────────────────────────────────────────────────┐
│                     Target Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PostgreSQL (Unified)           Redis (Optional L2)        │
│   ┌────────────────────────┐    ┌─────────────────────┐    │
│   │ User memories (RLS)    │    │ Session cache       │    │
│   │ Embeddings (HNSW idx)  │←──→│ Query results       │    │
│   │ Entities (tsvector)    │    │ TTL-based eviction  │    │
│   │ Semantic memory (A-MEM)│    │ Pub/sub invalidation│    │
│   │ Partitioned by time    │    └─────────────────────┘    │
│   └────────────────────────┘                               │
│        ↕                                                    │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              In-Memory (Hot Cache Only)              │  │
│   │  LRU embedding cache | Active scratchpads           │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   Benefits:                                                 │
│   • O(log n) vector search via HNSW                        │
│   • HIPAA compliance via RLS                               │
│   • 7-year retention via partitioning                      │
│   • Cross-worker consistency via Redis pub/sub             │
│   • Single source of truth for embeddings                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 10: Metrics to Track

After implementing recommendations, monitor:

| Metric | Current | Target | Tool |
|--------|---------|--------|------|
| Vector query latency | O(n) | O(log n) | pg_stat_statements |
| Cache hit rate | Unknown | >60% | Custom metrics |
| Memory persistence | 0% | 100% | Integration tests |
| Cross-worker invalidation | 0% | 100% | Redis pub/sub metrics |
| Session creation per request | 2+ | 1 | APM tracing |

---

## Conclusion

The memOS data systems demonstrate sophisticated architectural thinking with A-MEM, RAISE scratchpad, entity tracking, and multi-tier memory. However, **implementation lags behind design**:

1. **PostgreSQL pgvector is configured but lacks indexes** - vectors are scanned sequentially
2. **Redis is allocated 2GB but completely unused** - resources wasted
3. **Six SQLite databases fragment data** - 244 MB should be in PostgreSQL
4. **Semantic memory is ephemeral** - lost on restart despite persistence code
5. **Blackboard pattern exists but agents don't use it** - coordination missed

The recommended 4-week plan will:
- Consolidate storage into PostgreSQL with proper indexing
- Either use Redis for distributed caching or remove it
- Persist semantic memory for cross-session learning
- Enable full multi-agent coordination via blackboard

**Estimated Impact:**
- 10-100x faster vector search (HNSW vs sequential scan)
- 2GB RAM recovered (if Redis removed) or proper caching (if Redis used)
- Cross-session learning (A-MEM persistence)
- HIPAA compliance (RLS + partitioning)

---

**Report Generated:** 2025-12-29
**Audit Method:** 12 parallel research agents (3 codebase + 3 best practices + 6 audit)
**Total Issues Identified:** 32 (7 CRITICAL, 8 HIGH, 12 MEDIUM, 5 LOW)
