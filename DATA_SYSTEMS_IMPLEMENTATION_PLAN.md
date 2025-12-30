# memOS Data Systems Implementation Plan

## Based on: DATA_SYSTEMS_ARCHITECTURE_REPORT.md + 12 Agent Audit/Research Results
## Date: 2025-12-29
## Total Issues: 32 (7 CRITICAL, 8 HIGH, 12 MEDIUM, 5 LOW)

---

## Executive Summary

This plan addresses data infrastructure issues identified by comprehensive auditing. The codebase implements cutting-edge research patterns (Self-RAG, CRAG, GoT, BoT, AIME, GSW, A-MEM, RAISE) but has significant infrastructure gaps between design and implementation.

### Critical Findings from 12 Agent Audits

| Agent | Key Finding | Impact |
|-------|-------------|--------|
| Database/Session | 15+ direct AsyncSessionLocal() calls | Transaction inconsistency |
| Caching | Redis configured but unused | 2GB RAM wasted |
| Memory Systems | 2,554 lines of memory code - ALL EPHEMERAL | Lost on restart |
| pgvector Indexes | NO HNSW index on embedding_vector | 4,166x slower at scale |
| SQLite Consolidation | 11 databases totaling 256.6 MB | Fragmentation |
| Session DI | API endpoints ignore injected sessions | Nested transaction failures |
| Memory Persistence | to_dict()/from_dict() exist but NEVER CALLED | Design disconnected |
| Cache Coherency | No TTL jitter, semantic cache BYPASSED | 98.5% speedup lost |
| Blackboard Pattern | 78% of AgenticScratchpad interface UNUSED | Coordination missed |

---

## Implementation Phases

### Phase 1: Critical Infrastructure (2-4 hours)
**Goal:** Fix show-stopping issues that affect correctness and performance

#### 1.1 Add HNSW Index to pgvector (5 minutes)

**File:** Create new migration file

```sql
-- Add HNSW index for O(log n) vector search
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
ON memories USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Add GIN indexes for JSON columns
CREATE INDEX IF NOT EXISTS idx_memories_tags_gin
ON memories USING gin (tags);

CREATE INDEX IF NOT EXISTS idx_memories_entities_gin
ON memories USING gin (entities);
```

**Impact:** 4,166x faster vector queries at 100K scale

#### 1.2 Add pool_recycle to Database Config (5 minutes)

**File:** `config/database.py:42-48`

```python
# BEFORE
async_engine = create_async_engine(
    get_database_url(async_mode=True),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug
)

# AFTER
async_engine = create_async_engine(
    get_database_url(async_mode=True),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_timeout=30,    # Timeout waiting for connection
    echo=settings.debug
)
```

**Impact:** Prevents stale connection errors in long-running processes

#### 1.3 Enable Semantic Cache in Base Pipeline (10 minutes)

**File:** `agentic/base_pipeline.py:165-166`

The semantic cache is implemented but commented out. Enable it:

```python
# BEFORE (bypassed)
# Note: Semantic similarity search via find_similar_query requires
# pre-computed embeddings, so we skip it here for simplicity

# AFTER (enabled)
similar_result = await self.content_cache.find_similar_query(
    query=query,
    similarity_threshold=0.88  # High threshold for precision
)
if similar_result:
    return similar_result
```

**Impact:** 98.5% speedup on semantically similar queries

#### 1.4 Add TTL Jitter to Content Cache (15 minutes)

**File:** `agentic/content_cache.py`

Add jitter function and apply to all TTL calculations:

```python
import random

def _jitter_ttl(base_ttl: int, jitter_pct: float = 0.1) -> int:
    """Add random jitter to TTL to prevent cache stampede"""
    jitter = int(base_ttl * jitter_pct)
    return base_ttl + random.randint(-jitter, jitter)

# Apply to all cache operations
ttl = self._jitter_ttl(self.config.query_cache_ttl)
```

**Impact:** Prevents cache stampede when multiple workers refresh simultaneously

---

### Phase 2: Session Management Fix (1-2 hours)
**Goal:** Fix 15+ DI violations causing transaction inconsistency

#### 2.1 Refactor MemoryService to Accept Session

**File:** `core/memory_service.py`

Current pattern (WRONG):
```python
async def store_memory(self, user_id: str, content: str, ...):
    async with AsyncSessionLocal() as session:  # Creates own session!
        # ... operations ...
        await session.commit()
```

New pattern (CORRECT):
```python
async def store_memory(self, session: AsyncSession, user_id: str, content: str, ...):
    # Use injected session - let caller handle commit
    # ... operations ...
```

**Methods to refactor (15 total):**
- `store_memory()` (line 109)
- `get_memory()` (line 210)
- `search_memories()` (line 286)
- `update_memory()` (line 322)
- `delete_memory()` (line 355)
- `get_user_memories()` (line 677)
- `get_memory_stats()` (line 718)
- `_store_embedding()` (line 746)
- And 7 more internal methods

#### 2.2 Update API Endpoints to Pass Session

**File:** `api/memory.py`

```python
# BEFORE
@router.post("/memories")
async def create_memory(
    memory: MemoryCreate,
    db: AsyncSession = Depends(get_db_dependency),  # Injected but...
    memory_service: MemoryService = Depends(get_memory_service)
):
    return await memory_service.store_memory(...)  # ...not passed!

# AFTER
@router.post("/memories")
async def create_memory(
    memory: MemoryCreate,
    db: AsyncSession = Depends(get_db_dependency),
    memory_service: MemoryService = Depends(get_memory_service)
):
    return await memory_service.store_memory(db, ...)  # Pass session!
```

---

### Phase 3: Memory Persistence (2-3 hours)
**Goal:** Connect 2,554 lines of memory code to PostgreSQL

#### 3.1 A-MEM PostgreSQL Integration

**File:** `agentic/semantic_memory.py:168`

Current (ephemeral):
```python
self.memories: Dict[str, Memory] = {}  # Lost on restart!
```

Add persistence layer:
```python
class SemanticMemoryNetwork:
    def __init__(self, session_factory=None):
        self.memories: Dict[str, Memory] = {}
        self._session_factory = session_factory or AsyncSessionLocal

    async def persist(self, session: AsyncSession):
        """Persist in-memory network to PostgreSQL"""
        for memory_id, memory in self.memories.items():
            await self._upsert_memory(session, memory)

    async def load(self, session: AsyncSession, user_id: str):
        """Load persisted memories for user"""
        stmt = select(MemoryModel).where(
            MemoryModel.user_id == user_id,
            MemoryModel.memory_type == 'semantic'
        )
        result = await session.execute(stmt)
        for row in result.scalars():
            self.memories[str(row.id)] = Memory.from_dict(row.to_dict())
```

#### 3.2 Call Orchestrator Memory Methods

**File:** `agentic/orchestrator_universal.py`

The following methods exist but are NEVER CALLED:
- `_add_to_semantic_memory()`
- `_record_observation()`
- `_record_reasoning()`
- `_get_quality_signal()`

Add calls in search pipeline:

```python
async def _phase_synthesis(self, state, ...):
    # ... existing synthesis code ...

    # ADD: Record to semantic memory
    if self.config.enable_semantic_memory:
        await self._add_to_semantic_memory(
            content=synthesis,
            memory_type=MemoryType.FINDING,
            source_ids=[s.url for s in state.raw_results[:5]]
        )
```

---

### Phase 4: SQLite Consolidation (1-2 hours)
**Goal:** Migrate 229 MB of SQLite to PostgreSQL

#### 4.1 Migration Priority

| Database | Size | Action |
|----------|------|--------|
| mixed_precision_embeddings.db | 186 MB | Migrate to pgvector |
| hsea_hybrid.db | 43 MB | Migrate to pgvector |
| content_cache.db | 8.7 MB | Keep (local cache) |
| scratchpad_cache.db | 36 KB | Keep (ephemeral) |
| fanuc_robotics.db | 27 MB | Migrate to PostgreSQL |
| Test/legacy files | Various | DELETE |

#### 4.2 Mixed Precision Migration Script

```python
async def migrate_embeddings_to_postgres():
    """Migrate SQLite embeddings to PostgreSQL pgvector"""
    import sqlite3

    # Connect to SQLite
    sqlite_conn = sqlite3.connect("data/mixed_precision_embeddings.db")
    cursor = sqlite_conn.cursor()

    # Read all embeddings
    cursor.execute("SELECT id, content, embedding FROM embeddings")

    async with AsyncSessionLocal() as session:
        for row in cursor.fetchall():
            # Insert into PostgreSQL
            stmt = insert(EmbeddingModel).values(
                id=row[0],
                content=row[1],
                embedding=row[2]  # pgvector handles conversion
            )
            await session.execute(stmt)
        await session.commit()

    sqlite_conn.close()
    # Archive: mv data/mixed_precision_embeddings.db archive/
```

---

### Phase 5: Blackboard Pattern Activation (1 hour)
**Goal:** Use the 78% of AgenticScratchpad that's currently dormant

#### 5.1 Enable Finding Repository

**File:** `agentic/orchestrator_universal.py`

Currently unused methods:
- `scratchpad.add_finding()`
- `scratchpad.add_contradiction()`
- `scratchpad.write_public()`

Add to search phases:

```python
async def _phase_search(self, state, ...):
    for result in search_results:
        # ADD: Record each finding
        state.scratchpad.add_finding(
            question=state.query,
            finding=result.snippet,
            source=result.url,
            confidence=result.score
        )

async def _detect_contradictions(self, state, ...):
    # ADD: Record contradictions
    for contradiction in detected:
        state.scratchpad.add_contradiction(
            claim_a=contradiction.claim_a,
            claim_b=contradiction.claim_b,
            source_a=contradiction.source_a,
            source_b=contradiction.source_b
        )
```

---

### Phase 6: Redis Decision (30 minutes)
**Goal:** Either use Redis or remove it

#### Option A: Enable Redis for Distributed Caching

```python
# config/settings.py - already configured
redis_host: str = "localhost"
redis_port: int = 6379
redis_db: int = 2

# Add redis client to content_cache.py
import redis.asyncio as redis

class ContentCache:
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db
        )
```

#### Option B: Remove Redis to Save 2GB RAM

```yaml
# docker-compose.yml
# Comment out or remove redis service
# services:
#   redis:
#     image: redis:alpine
#     ...
```

**Recommendation:** Option B (remove) unless multi-worker deployment is planned.

---

## Verification Checklist

### After Phase 1
- [ ] Vector queries use HNSW index: `EXPLAIN ANALYZE SELECT ... ORDER BY embedding <-> ...`
- [ ] Semantic cache hit rate > 0%: `GET /api/v1/search/cache/stats`
- [ ] No duplicate cache refresh (check logs for jitter)

### After Phase 2
- [ ] No `AsyncSessionLocal()` in service methods
- [ ] Quest-memory integration works (test quest completion with memory storage)
- [ ] No greenlet errors in logs

### After Phase 3
- [ ] A-MEM persists across server restarts
- [ ] `_add_to_semantic_memory()` called in logs
- [ ] Memory count increases after searches

### After Phase 4
- [ ] SQLite files archived: `ls archive/*.db`
- [ ] PostgreSQL has migrated data: `SELECT COUNT(*) FROM embeddings`
- [ ] No SQLite I/O in production logs

### After Phase 5
- [ ] `add_finding()` logged during search
- [ ] Contradictions surfaced in synthesis
- [ ] Scratchpad stats show usage: `GET /api/v1/search/scratchpad/stats`

---

## Estimated Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Vector query latency | O(n) | O(log n) | 4,166x at 100K |
| Cache hit rate | ~0% | >60% | Memory efficient |
| Memory persistence | 0% | 100% | Cross-session learning |
| Session consistency | Broken | Correct | No transaction errors |
| SQLite fragmentation | 256 MB / 11 DBs | 0 | Unified storage |

---

## Risk Mitigation

1. **Git branch:** `fix/data-systems-remediation`
2. **Incremental commits:** One commit per phase
3. **Test after each phase:** Run `./test_system.sh` before proceeding
4. **Database backup:** `pg_dump memos > backup_$(date +%Y%m%d).sql`
5. **Archive before delete:** `mv file archive/` not `rm file`

---

## Implementation Order

1. **Phase 1.1** - HNSW Index (highest impact, lowest risk)
2. **Phase 1.2** - pool_recycle (prevents stale connections)
3. **Phase 1.3** - Semantic cache (immediate performance gain)
4. **Phase 1.4** - TTL jitter (prevents stampede)
5. **Phase 2** - Session DI (correctness fix)
6. **Phase 5** - Blackboard activation (feature enablement)
7. **Phase 3** - Memory persistence (major refactor)
8. **Phase 4** - SQLite migration (data consolidation)
9. **Phase 6** - Redis decision (infrastructure cleanup)

---

**Plan Created:** 2025-12-29
**Based On:** DATA_SYSTEMS_ARCHITECTURE_REPORT.md + 12 Agent Audits
**Priority:** Phase 1 is critical for production readiness
