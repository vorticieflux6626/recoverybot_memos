# Metasearch Integration Architecture for Agentic LLM Systems

> **Created**: 2025-12-31 | **Version**: 1.0.0 | **Author**: Claude Opus 4.5

## Executive Summary

This document provides architectural recommendations for integrating self-hosted metasearch (SearXNG) into LLM-powered agentic search systems. Based on analysis of Perplexity AI, Tavily, RouteLLM, and academic RAG research, combined with the existing Recovery Bot architecture.

---

## 1. High-Level Architecture

### 1.1 Reference Architecture (Perplexity-Style)

```
                                    +------------------+
                                    |   ANDROID        |
                                    |   CLIENT         |
                                    +--------+---------+
                                             |
                                             v
+---------------------------------------------------------------------------------+
|                              memOS SERVER (Port 8001)                           |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  +----------------+     +------------------+     +---------------------+         |
|  | Query Analyzer |---->| Query Classifier |---->| Pipeline Router     |         |
|  | (Intent Parse) |     | (RouteLLM-style) |     | (Preset-based)      |         |
|  +----------------+     +------------------+     +----------+----------+         |
|                                                            |                    |
|          +----------------+----------------+----------------+                   |
|          |                |                |                |                   |
|          v                v                v                v                   |
|  +-------------+  +-------------+  +-------------+  +-------------+             |
|  |   MINIMAL   |  |  BALANCED   |  |  ENHANCED   |  |  RESEARCH   |             |
|  | (8 features)|  |(18 features)|  |(28 features)|  |(38 features)|             |
|  +------+------+  +------+------+  +------+------+  +------+------+             |
|         |                |                |                |                    |
|         +----------------+----------------+----------------+                    |
|                                    |                                            |
|                                    v                                            |
|  +------------------------------------------------------------------------+    |
|  |                        SEARCH ORCHESTRATION                             |    |
|  +------------------------------------------------------------------------+    |
|  |                                                                         |    |
|  |  +-----------------------+    +------------------------+                |    |
|  |  | CRAG Pre-Evaluation   |--->| Corrective Actions     |                |    |
|  |  | (Before Synthesis)    |    | REFINE | DECOMPOSE     |                |    |
|  |  +-----------------------+    +------------------------+                |    |
|  |                                                                         |    |
|  |  +-------------------------------------------------------------------+ |    |
|  |  |                  MULTI-PROVIDER SEARCH LAYER                       | |    |
|  |  +-------------------------------------------------------------------+ |    |
|  |  |                                                                    | |    |
|  |  |  +-------------+   +-------------+   +-------------+              | |    |
|  |  |  |  SEARXNG    |   |  DUCKDUCKGO |   |    BRAVE    |              | |    |
|  |  |  |  (Primary)  |   |  (Fallback) |   |  (Tertiary) |              | |    |
|  |  |  | :8888       |   | (no API)    |   |  (API key)  |              | |    |
|  |  |  +------+------+   +------+------+   +------+------+              | |    |
|  |  |         |                 |                 |                     | |    |
|  |  |         v                 v                 v                     | |    |
|  |  |  +----------------------------------------------------+          | |    |
|  |  |  |           RESULT FUSION & DEDUPLICATION             |          | |    |
|  |  |  | - Reciprocal Rank Fusion (RRF)                      |          | |    |
|  |  |  | - Semantic Deduplication (0.88 threshold)           |          | |    |
|  |  |  | - Source Quality Scoring                            |          | |    |
|  |  |  +----------------------------------------------------+          | |    |
|  |  +-------------------------------------------------------------------+ |    |
|  |                                                                         |    |
|  |  +-----------------------+    +------------------------+                |    |
|  |  | Content Scraping      |--->| LLM Synthesis          |                |    |
|  |  | (VL + Deep Reader)    |    | (DeepSeek R1 / qwen3)  |                |    |
|  |  +-----------------------+    +------------------------+                |    |
|  |                                                                         |    |
|  |  +-----------------------+    +------------------------+                |    |
|  |  | Self-RAG Evaluation   |--->| Experience Distillation|                |    |
|  |  | (Post-Synthesis QC)   |    | (Template Learning)    |                |    |
|  |  +-----------------------+    +------------------------+                |    |
|  +------------------------------------------------------------------------+    |
|                                                                                 |
+---------------------------------------------------------------------------------+
                                             |
                                             v
                      +------------------------------------------+
                      |            SUPPORTING SERVICES           |
                      +------------------------------------------+
                      |  Redis     | Qdrant      | PostgreSQL   |
                      |  (Cache)   | (Vectors)   | (Memories)   |
                      +------------------------------------------+
```

### 1.2 Data Flow Sequence

```
User Query
    |
    v
+-------------------+
| 1. Query Analysis |  Extract intent, entities, domain
+-------------------+
    |
    v
+-------------------+
| 2. Classification |  academic | technical | general | code
+-------------------+
    |
    v
+-------------------+
| 3. Engine Routing |  Select SearXNG engine group
+-------------------+
    |
    +-----> [academic] arxiv, semantic_scholar, pubmed, crossref
    |
    +-----> [technical] github, stackoverflow, pypi, brave, bing
    |
    +-----> [robotics] reddit, brave, bing, electronics_se, robotics_se
    |
    +-----> [general] brave, bing, duckduckgo, wikipedia
    |
    v
+-------------------+
| 4. Parallel Search|  Execute across selected engines
+-------------------+
    |
    v
+-------------------+
| 5. Result Fusion  |  RRF + dedup + quality scoring
+-------------------+
    |
    v
+-------------------+
| 6. CRAG Evaluate  |  CORRECT | AMBIGUOUS | INCORRECT
+-------------------+
    |
    +-----> [CORRECT] Proceed to synthesis
    |
    +-----> [AMBIGUOUS] Refine queries, re-search
    |
    +-----> [INCORRECT] Fallback to different engines
    |
    v
+-------------------+
| 7. Content Scrape |  Deep read top-N URLs
+-------------------+
    |
    v
+-------------------+
| 8. LLM Synthesis  |  Generate answer with citations
+-------------------+
    |
    v
+-------------------+
| 9. Self-RAG Check |  Validate ISREL/ISSUP/ISUSE
+-------------------+
    |
    v
+-------------------+
| 10. Response      |  Return to client with sources
+-------------------+
```

---

## 2. Multi-Provider Search with Fallback

### 2.1 Provider Priority Chain

Based on research and your current setup, the recommended provider priority:

```python
PROVIDER_PRIORITY = [
    {
        "provider": "searxng",
        "base_url": "http://localhost:8888",
        "engines": ["brave", "bing", "startpage", "reddit"],
        "timeout": 8.0,
        "retry_count": 2
    },
    {
        "provider": "duckduckgo",
        "method": "ddgs",  # duckduckgo_search library
        "timeout": 10.0,
        "retry_count": 1
    },
    {
        "provider": "brave",
        "api_key": "$BRAVE_API_KEY",
        "timeout": 8.0,
        "retry_count": 1
    }
]
```

### 2.2 Circuit Breaker Pattern

Implement circuit breaker for graceful degradation:

```python
class SearchCircuitBreaker:
    """
    Circuit breaker for search providers.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service recovered

    Based on: https://www.geeksforgeeks.org/system-design/graceful-degradation-in-distributed-systems/
    """

    def __init__(
        self,
        failure_threshold: int = 5,     # Failures before opening
        recovery_timeout: float = 60.0,  # Seconds before half-open
        half_open_max_calls: int = 3    # Calls to test recovery
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpen()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
        self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 2.3 Multi-Provider Search Implementation

```python
class MultiProviderSearch:
    """
    Search across multiple providers with fallback.

    Based on Tavily architecture: https://www.tavily.com/
    """

    def __init__(self):
        self.providers = {
            "searxng": SearXNGSearcher(),
            "duckduckgo": DuckDuckGoSearcher(),
            "brave": BraveSearcher()
        }
        self.circuit_breakers = {
            name: SearchCircuitBreaker()
            for name in self.providers
        }

    async def search(
        self,
        queries: List[str],
        engine_group: str = "general",
        max_results: int = 20
    ) -> List[SearchResultItem]:
        """
        Execute search with automatic fallback.

        Priority: SearXNG -> DuckDuckGo -> Brave
        """
        for provider_name in ["searxng", "duckduckgo", "brave"]:
            try:
                cb = self.circuit_breakers[provider_name]
                provider = self.providers[provider_name]

                results = await cb.call(
                    provider.search,
                    queries,
                    engine_group=engine_group,
                    max_results=max_results
                )

                if results:
                    logger.info(f"Search succeeded with {provider_name}: {len(results)} results")
                    return results

            except CircuitBreakerOpen:
                logger.warning(f"{provider_name} circuit breaker is OPEN, skipping")
                continue
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                continue

        logger.error("All search providers failed")
        return []
```

---

## 3. Result Fusion and Deduplication

### 3.1 Reciprocal Rank Fusion (RRF)

Based on OpenSearch's implementation: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/

```python
def reciprocal_rank_fusion(
    ranked_lists: List[List[SearchResult]],
    k: int = 60  # Constant to prevent high-ranked docs dominating
) -> List[SearchResult]:
    """
    Merge multiple ranked lists using RRF.

    RRF Score = sum(1 / (k + rank_i)) for each list where doc appears

    Advantages over score normalization:
    - Doesn't require comparable scores across sources
    - More stable for hybrid search (semantic + keyword)
    - Prioritizes docs that rank well across multiple sources
    """

    # Collect scores per URL
    url_scores: Dict[str, float] = {}
    url_docs: Dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            url = doc.url
            rrf_score = 1.0 / (k + rank)

            url_scores[url] = url_scores.get(url, 0) + rrf_score

            # Keep highest-quality version of doc
            if url not in url_docs or doc.score > url_docs[url].score:
                url_docs[url] = doc

    # Sort by RRF score
    sorted_urls = sorted(url_scores.keys(), key=lambda u: url_scores[u], reverse=True)

    # Return merged results with RRF scores
    results = []
    for url in sorted_urls:
        doc = url_docs[url]
        doc.rrf_score = url_scores[url]
        results.append(doc)

    return results
```

### 3.2 Semantic Deduplication

Based on SemHash: https://github.com/MinishLab/semhash

```python
class SemanticDeduplicator:
    """
    Remove semantically duplicate search results.

    Uses embedding similarity to find near-duplicates.
    More effective than URL-only dedup for results that
    cover the same content from different sources.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,  # Validated in your system
        embedding_model: str = "nomic-embed-text"
    ):
        self.threshold = similarity_threshold
        self.embedding_model = embedding_model

    async def deduplicate(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Remove semantic duplicates while preserving diversity.
        """
        if len(results) <= 1:
            return results

        # Generate embeddings for all snippets
        texts = [f"{r.title}. {r.snippet}" for r in results]
        embeddings = await self._get_embeddings(texts)

        # Greedy selection: keep results that are dissimilar to kept ones
        kept = [0]  # Always keep first result
        kept_embeddings = [embeddings[0]]

        for i in range(1, len(results)):
            max_similarity = max(
                self._cosine_similarity(embeddings[i], kept_emb)
                for kept_emb in kept_embeddings
            )

            if max_similarity < self.threshold:
                kept.append(i)
                kept_embeddings.append(embeddings[i])

        return [results[i] for i in kept]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 4. Query Routing and Classification

### 4.1 Engine Group Routing

Based on your domain and query types:

```python
class QueryRouter:
    """
    Route queries to appropriate SearXNG engine groups.

    Based on RouteLLM patterns: https://lmsys.org/blog/2024-07-01-routellm/
    """

    ENGINE_GROUPS = {
        "academic": ["arxiv", "semantic_scholar", "pubmed", "crossref"],
        "technical": ["github", "stackoverflow", "pypi", "npm", "dockerhub", "brave", "bing"],
        "robotics": ["reddit", "brave", "bing", "electronics_stackexchange", "robotics_stackexchange"],
        "fanuc": ["reddit", "brave", "bing", "arxiv", "electronics_stackexchange"],
        "linux": ["askubuntu", "unix_stackexchange", "archlinux", "serverfault", "reddit"],
        "general": ["brave", "bing", "duckduckgo", "wikipedia", "startpage"],
        "news": ["bing_news", "reddit"]
    }

    DOMAIN_PATTERNS = {
        "academic": [
            r"research\s+on",
            r"paper\s+about",
            r"study\s+of",
            r"literature\s+review",
            r"arxiv",
            r"scientific"
        ],
        "technical": [
            r"how\s+to\s+implement",
            r"code\s+for",
            r"python|javascript|rust|golang",
            r"api|sdk|library",
            r"docker|kubernetes"
        ],
        "robotics": [
            r"fanuc|kuka|abb|yaskawa",
            r"servo\s+error|srvo-",
            r"robot\s+programming",
            r"industrial\s+automation",
            r"plc|hmi|scada"
        ],
        "fanuc": [
            r"fanuc",
            r"srvo-\d+",
            r"motn-\d+",
            r"syst-\d+",
            r"tp\s+program",
            r"karel"
        ]
    }

    async def route(self, query: str) -> str:
        """
        Determine best engine group for query.

        Returns engine group name.
        """
        query_lower = query.lower()

        # Check domain patterns
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Routed to {domain} based on pattern: {pattern}")
                    return domain

        # Fallback: use lightweight LLM classification
        classification = await self._classify_with_llm(query)
        return classification

    async def _classify_with_llm(self, query: str) -> str:
        """Use fast LLM to classify query domain"""
        prompt = f"""Classify this search query into ONE category:
- academic: Research papers, scientific studies
- technical: Programming, code, software development
- robotics: Industrial robots, automation, PLCs
- general: Everything else

Query: {query}

Category:"""

        # Use fast model (gemma3:4b or qwen3:4b)
        response = await self._call_llm(prompt, max_tokens=10)
        category = response.strip().lower()

        if category in self.ENGINE_GROUPS:
            return category
        return "general"
```

---

## 5. Semantic Caching Strategy

### 5.1 Multi-Level Cache Architecture

Based on GPTCache and your existing implementation:

```
+------------------------------------------------------------------+
|                      SEMANTIC CACHE LAYERS                        |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+    +-------------------+    +------------+ |
|  |   EXACT MATCH     |    |  SEMANTIC MATCH   |    |   REDIS    | |
|  |   (Hash-based)    |    |  (Embedding-based)|    |   (TTL)    | |
|  +-------------------+    +-------------------+    +------------+ |
|                                                                   |
|  Query Hash → Cache     Query Embedding        Query → Results   |
|  O(1) lookup           → Vector Search          5-60 min TTL     |
|                        → Threshold 0.88                          |
|                                                                   |
+------------------------------------------------------------------+

                          Cache Hit Path
                               |
    +------------+       +------------+       +------------+
    | Exact Hash |  NO   |  Semantic  |  NO   |   Fresh    |
    |   Match?   | ----> |   Match?   | ----> |   Search   |
    +------------+       +------------+       +------------+
          |                    |
         YES                  YES
          |                    |
          v                    v
    +------------+       +------------+
    |   Return   |       |   Return   |
    |   Cached   |       |   Cached   |
    |   Result   |       |   Result   |
    +------------+       +------------+
```

### 5.2 Semantic Cache Implementation

```python
class SemanticSearchCache:
    """
    Multi-level semantic cache for search results.

    Based on GPTCache: https://github.com/zilliztech/GPTCache
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedding_model: str = "nomic-embed-text",
        semantic_threshold: float = 0.88,
        cache_ttl: int = 1800  # 30 minutes
    ):
        self.redis = Redis.from_url(redis_url)
        self.embedding_model = embedding_model
        self.threshold = semantic_threshold
        self.ttl = cache_ttl

        # Vector store for semantic matching
        self.vector_store = Qdrant(collection="search_cache")

    async def get(self, query: str) -> Optional[CachedSearchResult]:
        """
        Try to get cached result for query.

        1. Check exact hash match
        2. Check semantic match
        3. Return None if no match
        """
        # Level 1: Exact match
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cached = self.redis.get(f"search:exact:{query_hash}")
        if cached:
            logger.debug(f"Cache HIT (exact): {query[:30]}...")
            return CachedSearchResult.from_json(cached)

        # Level 2: Semantic match
        query_embedding = await self._get_embedding(query)
        similar = await self.vector_store.search(
            vector=query_embedding,
            limit=1,
            score_threshold=self.threshold
        )

        if similar:
            match = similar[0]
            cached = self.redis.get(f"search:semantic:{match.id}")
            if cached:
                logger.debug(f"Cache HIT (semantic, score={match.score:.3f}): {query[:30]}...")
                return CachedSearchResult.from_json(cached)

        logger.debug(f"Cache MISS: {query[:30]}...")
        return None

    async def set(
        self,
        query: str,
        results: List[SearchResult],
        metadata: Dict[str, Any] = None
    ):
        """Cache search results with both exact and semantic keys"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        query_embedding = await self._get_embedding(query)

        cached = CachedSearchResult(
            query=query,
            results=results,
            metadata=metadata,
            cached_at=datetime.utcnow()
        )
        cached_json = cached.to_json()

        # Store with exact hash
        self.redis.setex(
            f"search:exact:{query_hash}",
            self.ttl,
            cached_json
        )

        # Store with semantic key
        cache_id = str(uuid.uuid4())
        self.redis.setex(
            f"search:semantic:{cache_id}",
            self.ttl,
            cached_json
        )

        # Index embedding
        await self.vector_store.upsert(
            id=cache_id,
            vector=query_embedding,
            payload={"query": query, "hash": query_hash}
        )
```

### 5.3 TTL-Based Cache Pinning

Prevent cache eviction during long-running operations:

```python
class TTLCacheManager:
    """
    Manage cache TTL with tool-aware pinning.

    Based on Continuum-inspired TTL pinning from your design decisions.
    """

    TOOL_LATENCIES_P95 = {
        "web_search": 8.0,      # SearXNG search
        "content_scrape": 15.0,  # Deep reader
        "llm_synthesis": 30.0,   # DeepSeek R1
        "pdf_extraction": 20.0   # PDF Tools API
    }

    def __init__(self, redis: Redis):
        self.redis = redis
        self.buffer_multiplier = 1.3  # 30% safety buffer

    def calculate_pin_ttl(self, tool_type: str) -> int:
        """Calculate TTL for cache pinning during tool execution"""
        base_latency = self.TOOL_LATENCIES_P95.get(tool_type, 10.0)
        return int(base_latency * self.buffer_multiplier)

    async def pin_cache_entry(
        self,
        key: str,
        tool_type: str
    ) -> int:
        """
        Extend TTL to prevent eviction during tool execution.

        Returns the new TTL.
        """
        pin_ttl = self.calculate_pin_ttl(tool_type)
        current_ttl = self.redis.ttl(key)

        if current_ttl < pin_ttl:
            self.redis.expire(key, pin_ttl)
            return pin_ttl
        return current_ttl
```

---

## 6. Search Quality Metrics

### 6.1 Core Metrics

Based on search evaluation research:

```python
class SearchQualityMetrics:
    """
    Track search quality using standard IR metrics.

    References:
    - https://weaviate.io/blog/retrieval-evaluation-metrics
    - https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    """

    def __init__(self):
        self.metrics = {
            "mrr": [],           # Mean Reciprocal Rank
            "ndcg@5": [],        # NDCG at 5
            "precision@5": [],   # Precision at 5
            "coverage": [],      # Query coverage
            "latency_ms": [],    # Search latency
            "cache_hit_rate": 0.0
        }

    def record_search(
        self,
        query: str,
        results: List[SearchResult],
        relevance_judgments: Optional[Dict[str, float]] = None,
        latency_ms: float = 0,
        from_cache: bool = False
    ):
        """Record metrics for a search"""

        # Update cache hit rate
        self._update_cache_rate(from_cache)

        # Latency
        self.metrics["latency_ms"].append(latency_ms)

        if relevance_judgments:
            # Calculate MRR
            mrr = self._calculate_mrr(results, relevance_judgments)
            self.metrics["mrr"].append(mrr)

            # Calculate NDCG@5
            ndcg = self._calculate_ndcg(results[:5], relevance_judgments)
            self.metrics["ndcg@5"].append(ndcg)

            # Calculate Precision@5
            precision = self._calculate_precision(results[:5], relevance_judgments)
            self.metrics["precision@5"].append(precision)

    def _calculate_mrr(
        self,
        results: List[SearchResult],
        relevance: Dict[str, float]
    ) -> float:
        """Mean Reciprocal Rank - position of first relevant result"""
        for i, result in enumerate(results, start=1):
            if relevance.get(result.url, 0) > 0.5:
                return 1.0 / i
        return 0.0

    def _calculate_ndcg(
        self,
        results: List[SearchResult],
        relevance: Dict[str, float],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.

        Measures ranking quality with graded relevance.
        """
        # DCG
        dcg = sum(
            relevance.get(r.url, 0) / np.log2(i + 2)
            for i, r in enumerate(results[:k])
        )

        # IDCG (ideal ordering)
        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
        )

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _calculate_precision(
        self,
        results: List[SearchResult],
        relevance: Dict[str, float],
        threshold: float = 0.5
    ) -> float:
        """Precision at K"""
        relevant_count = sum(
            1 for r in results
            if relevance.get(r.url, 0) >= threshold
        )
        return relevant_count / len(results) if results else 0.0
```

### 6.2 Feedback Loop for Continuous Improvement

```python
class SearchFeedbackLoop:
    """
    Learn from search outcomes to improve routing and caching.

    Based on Adaptive-RAG classifier feedback patterns.
    """

    def __init__(self):
        self.outcome_history = []  # (query_hash, engine_group, success)
        self.overkill_patterns = []  # Queries that used too much processing
        self.underkill_patterns = []  # Queries that needed more processing

    async def record_outcome(
        self,
        query: str,
        engine_group: str,
        preset: str,
        confidence: float,
        user_feedback: Optional[bool] = None
    ):
        """Record search outcome for learning"""

        success = confidence >= 0.7 or user_feedback is True

        self.outcome_history.append({
            "query": query,
            "engine_group": engine_group,
            "preset": preset,
            "confidence": confidence,
            "success": success
        })

        # Detect overkill/underkill
        if preset == "research" and success and confidence > 0.9:
            # Could have used simpler preset
            self.overkill_patterns.append(query)
        elif preset == "minimal" and not success:
            # Needed more processing
            self.underkill_patterns.append(query)

    def get_recommended_preset(self, query: str) -> str:
        """Get recommended preset based on similar past queries"""
        # Use embedding similarity to find similar past queries
        # Return the preset that worked best for them
        pass
```

---

## 7. Implementation Recommendations

### 7.1 Priority 1: Immediate Improvements

1. **Add RRF to Result Fusion**
   - Current: URL-based deduplication only
   - Recommended: Implement RRF for multi-engine results
   - Impact: Better ranking when results come from multiple SearXNG engines

2. **Enhance Circuit Breaker**
   - Current: Basic availability check
   - Recommended: Full circuit breaker with half-open state
   - Impact: Faster recovery, less wasted requests

3. **Add Semantic Cache Layer**
   - Current: Not implemented
   - Recommended: Redis + Qdrant semantic cache
   - Impact: 60-80% faster for similar queries

### 7.2 Priority 2: Medium-Term Enhancements

4. **Implement Query Router**
   - Current: Default engine selection
   - Recommended: Pattern + LLM-based routing
   - Impact: Better results for domain-specific queries

5. **Add Search Quality Metrics**
   - Current: Basic stats tracking
   - Recommended: MRR, NDCG, latency tracking
   - Impact: Data-driven optimization decisions

6. **Implement Feedback Loop**
   - Current: Experience distillation exists
   - Recommended: Add preset recommendation learning
   - Impact: Automatic preset optimization over time

### 7.3 Priority 3: Advanced Features

7. **Cross-Encoder Reranking for Search Results**
   - Already exists for RAG retrieval
   - Extend to web search results
   - Impact: Better top-5 result quality

8. **Async Speculative Search**
   - Start secondary engine queries speculatively
   - Cancel if primary succeeds quickly
   - Impact: Lower P95 latency

---

## 8. References and Sources

### Research Papers
- [Retrieval-Augmented Generation: A Comprehensive Survey (2025)](https://arxiv.org/abs/2506.00054)
- [Enhancing RAG: A Study of Best Practices (2025)](https://arxiv.org/abs/2501.07391)
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/html/2406.18665v1)
- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)

### Industry Implementations
- [Perplexity AI Architecture](https://www.perplexity.ai/api-platform/resources/architecting-and-evaluating-an-ai-first-search-api)
- [Tavily Search API](https://www.tavily.com/)
- [OpenSearch Reciprocal Rank Fusion](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)

### Caching and Optimization
- [GPTCache: Semantic Caching for LLMs](https://github.com/zilliztech/GPTCache)
- [SemHash: Semantic Deduplication](https://github.com/MinishLab/semhash)
- [Redis Semantic Caching](https://redis.io/blog/what-is-semantic-caching/)

### Graceful Degradation
- [Graceful Degradation in Distributed Systems](https://www.geeksforgeeks.org/system-design/graceful-degradation-in-distributed-systems/)
- [AWS Well-Architected: Graceful Degradation](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_mitigate_interaction_failure_graceful_degradation.html)

---

## Appendix A: ASCII Architecture Diagrams

### A.1 SearXNG Engine Routing

```
                    +------------------+
                    |  User Query      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Query Classifier |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
     +-----------+    +-----------+    +-----------+
     | ACADEMIC  |    | TECHNICAL |    |  GENERAL  |
     +-----------+    +-----------+    +-----------+
            |                |                |
            v                v                v
     +-----------+    +-----------+    +-----------+
     | arxiv     |    | github    |    | brave     |
     | semantic  |    | stack     |    | bing      |
     | scholar   |    | overflow  |    | duckduck  |
     | pubmed    |    | pypi      |    | wikipedia |
     +-----------+    +-----------+    +-----------+
            |                |                |
            +----------------+----------------+
                             |
                             v
                    +------------------+
                    | Result Fusion    |
                    | (RRF + Dedup)    |
                    +------------------+
```

### A.2 Fallback Chain

```
    +------------+     +------------+     +------------+
    |  SearXNG   | --> | DuckDuckGo | --> |   Brave    |
    |  (Primary) |     | (Secondary)|     | (Tertiary) |
    +-----+------+     +-----+------+     +-----+------+
          |                  |                  |
          v                  v                  v
    [ AVAILABLE ]      [ AVAILABLE ]      [ AVAILABLE ]
          |                  |                  |
          +---> SUCCESS      +---> SUCCESS      +---> SUCCESS
          |                  |                  |
          +---> FAIL --------+---> FAIL --------+---> ERROR
```

### A.3 Cache Hierarchy

```
                    +------------------+
                    |  Search Query    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | L1: Exact Hash   | -----> HIT: Return cached
                    +--------+---------+
                             |
                            MISS
                             |
                             v
                    +------------------+
                    | L2: Semantic     | -----> HIT: Return cached
                    | (Embedding sim)  |
                    +--------+---------+
                             |
                            MISS
                             |
                             v
                    +------------------+
                    | Fresh Search     |
                    | Store in L1 + L2 |
                    +------------------+
```

---

*Document generated by Claude Opus 4.5 for Recovery Bot project*
