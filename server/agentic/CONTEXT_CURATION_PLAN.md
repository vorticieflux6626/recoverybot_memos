# Context Curation & Adaptive Refinement Optimization Plan

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Complete (Phase 17)

## Research Report Analysis - Gap Assessment

Based on the multi-agent systems research synthesis (50+ papers, NeurIPS/ICLR/EMNLP/ACL 2024-2025), this plan identifies optimization opportunities for our agentic search system.

### Current Implementation Status

| Component | Research Framework | Status | Gap |
|-----------|-------------------|--------|-----|
| Scratchpad | GSW/bMAS Blackboard | ✅ Partial | Missing semantic connections, confidence metadata |
| CRAG Evaluator | CRAG 3-tier | ✅ Complete | Missing query rewriting on Incorrect |
| Self-RAG | ISREL/ISSUP/ISUSE | ✅ Complete | No token-level triggering |
| Adaptive Refinement | AT-RAG/FAIR-RAG | ✅ Complete | Missing tree decoding for queries |
| ThoughtLibrary | Buffer of Thoughts | ✅ Partial | Missing cross-session meta-buffer |
| HyDE | Query Expansion | ✅ Complete | - |
| BGE-M3 Hybrid | Dense+Sparse | ✅ Complete | - |
| **InfoGain Scoring** | InfoGain-RAG | ❌ Missing | **HIGH PRIORITY** |
| **Context-Picker** | Two-stage RL | ❌ Missing | **HIGH PRIORITY** |
| **Entropy Halting** | UALA/REFRAIN | ❌ Missing | Medium priority |
| **Self-Consistency** | CISC | ❌ Missing | Medium priority |
| **FLARE Retrieval** | Forward-looking | ❌ Missing | Medium priority |
| **A-MEM Connections** | Zettelkasten | ❌ Missing | Lower priority |

---

## Phase 1: Context Curation Pipeline (HIGH PRIORITY)

**Goal**: Build a corpus of high-quality, non-redundant information before synthesis.

### 1.1 Document Information Gain (DIG) Scoring

**Research Basis**: InfoGain-RAG (Kuaishou Technology) - +17.9% over naive RAG

**Implementation**: `context_curator.py`

```python
class DocumentInformationGain:
    """
    Calculate Document Information Gain (DIG) - the change in LLM generation
    confidence with vs without each document.

    DIG(d) = P(correct | context + d) - P(correct | context)

    Key insight: Document utility measured by impact on generation confidence,
    not just semantic similarity.
    """

    async def calculate_dig(
        self,
        query: str,
        document: str,
        existing_context: List[str]
    ) -> float:
        """
        Calculate information gain of adding this document.

        Returns: DIG score (-1.0 to 1.0)
        - Positive: Document increases confidence
        - Zero: Document has no effect
        - Negative: Document decreases confidence (noise/contradiction)
        """

    async def batch_dig_ranking(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rank documents by cumulative information gain.
        Uses sliding window to mitigate length bias.
        """
```

**Integration Point**: After URL scraping, before synthesis

### 1.2 Two-Stage Context Filtering (Context-Picker Pattern)

**Research Basis**: Context-Picker (arXiv 2512.14465) - Two-stage RL

**Stage 1: Recall-Oriented** - Maximize coverage of reasoning chains
**Stage 2: Precision-Oriented** - Aggressively prune redundancy

```python
class ContextPicker:
    """
    Two-stage context filtering inspired by Context-Picker.

    Stage 1 (Recall): Keep documents that cover ANY part of the query
    Stage 2 (Precision): Remove redundant documents via Leave-One-Out
    """

    async def stage1_recall_filter(
        self,
        query: str,
        decomposed_questions: List[str],
        documents: List[ScrapedContent]
    ) -> List[ScrapedContent]:
        """
        Keep documents covering at least one decomposed question.
        High recall, may include redundancy.
        """

    async def stage2_precision_prune(
        self,
        query: str,
        documents: List[ScrapedContent]
    ) -> List[ScrapedContent]:
        """
        Leave-One-Out pruning: Remove doc if answer quality unchanged without it.
        Finds "minimal sufficient set" - smallest set preserving answer quality.
        """

    async def mine_minimal_sufficient_set(
        self,
        query: str,
        documents: List[ScrapedContent]
    ) -> List[ScrapedContent]:
        """
        Iteratively remove documents until quality degrades.
        Returns the minimal set that still answers the query.
        """
```

### 1.3 Redundancy Detection & Clustering

**Goal**: Identify semantically similar documents, keep only the best representative

```python
class RedundancyDetector:
    """
    Cluster documents by semantic similarity, keep best representative.
    """

    async def detect_clusters(
        self,
        documents: List[ScrapedContent],
        similarity_threshold: float = 0.85
    ) -> List[DocumentCluster]:
        """
        Cluster documents with >85% semantic similarity.
        """

    async def select_representatives(
        self,
        clusters: List[DocumentCluster],
        scoring_method: str = "information_gain"
    ) -> List[ScrapedContent]:
        """
        Select best document from each cluster.
        Methods: information_gain, source_authority, recency, length
        """
```

### 1.4 Multi-Stage Curation Pipeline

```python
class ContextCurationPipeline:
    """
    Full pipeline: Scrape → Filter → Score → Cluster → Select → Curate

    Research-validated thresholds:
    - Initial retrieval: k=40-150 (high recall)
    - Similarity dedup: >0.85 threshold
    - Relevance filter: >0.76 threshold
    - Final selection: n=5-20 documents
    """

    async def curate(
        self,
        query: str,
        scraped_content: List[ScrapedContent],
        target_docs: int = 10
    ) -> CuratedContext:
        """
        Full curation pipeline:
        1. Deduplication (semantic similarity >0.85)
        2. Relevance scoring (filter <0.76)
        3. DIG scoring (document information gain)
        4. Redundancy clustering
        5. Representative selection
        6. Quality threshold check

        Returns curated context with:
        - Selected documents
        - Coverage analysis (which questions answered)
        - Confidence estimate
        - Curation trace for debugging
        """
```

---

## Phase 2: Confidence-Calibrated Halting

**Goal**: Know when to stop iterating vs when more retrieval helps.

### 2.1 Entropy-Based Confidence Signals

**Research Basis**: UALA (ACL 2024) - >50% reduction in tool calls

```python
class EntropyMonitor:
    """
    Track generation entropy to detect confident vs uncertain outputs.

    Thresholds (from research):
    - High confidence: entropy < 0.2 → stop iterating
    - Uncertain: entropy > 0.5 → continue iterating
    """

    def calculate_token_entropy(
        self,
        logprobs: List[float]
    ) -> float:
        """Calculate entropy from token log probabilities."""

    def should_halt(
        self,
        entropy: float,
        iteration: int,
        max_iterations: int
    ) -> HaltDecision:
        """
        Decide whether to halt based on entropy trajectory.

        Returns: CONTINUE, HALT_CONFIDENT, HALT_MAX_ITERATIONS
        """
```

### 2.2 Self-Consistency Convergence

**Research Basis**: CISC (Google, arXiv 2502.20233) - >40% sample reduction

```python
class SelfConsistencyChecker:
    """
    Weighted majority voting across multiple reasoning paths.

    Threshold: >60% agreement = sufficient confidence
    """

    async def check_convergence(
        self,
        query: str,
        synthesis_attempts: List[str],
        min_agreement: float = 0.6
    ) -> ConvergenceResult:
        """
        Check if multiple synthesis attempts converge to same answer.

        Uses semantic similarity + key fact extraction for comparison.
        Returns agreement ratio and confidence-weighted answer.
        """
```

### 2.3 UCB Bandit for Iteration Decisions

**Research Basis**: REFRAIN (arXiv 2510.10103) - 20-55% token reduction

```python
class IterationBandit:
    """
    Multi-armed bandit for exploration/exploitation tradeoff.

    Arms: [SEARCH_MORE, REFINE_QUERY, SYNTHESIZE_NOW, DECOMPOSE]
    Reward: Confidence improvement or successful synthesis
    """

    def select_action(
        self,
        state: RefinementState,
        history: List[ActionOutcome]
    ) -> RefinementDecision:
        """
        UCB action selection with sliding window.

        UCB(a) = Q(a) + c * sqrt(log(t) / N(a))

        Where:
        - Q(a) = average reward for action a
        - N(a) = times action a selected
        - c = exploration constant (typically 2.0)
        """
```

---

## Phase 3: Enhanced Query Generation

**Goal**: Generate better follow-up queries when initial retrieval is insufficient.

### 3.1 FLARE Forward-Looking Retrieval

**Research Basis**: FLARE (EMNLP 2023) - Retrieve based on predicted needs

```python
class FLARERetriever:
    """
    Forward-Looking Active REtrieval.

    During generation, if any token probability < threshold,
    use tentative content as retrieval query.
    """

    async def forward_looking_retrieve(
        self,
        query: str,
        partial_synthesis: str,
        confidence_threshold: float = 0.6
    ) -> List[str]:
        """
        Generate what the model predicts it needs to say next,
        then retrieve documents for that predicted content.
        """
```

### 3.2 RQ-RAG Tree Decoding

**Research Basis**: RQ-RAG (arXiv 2404.00610) - +33.5% on QA benchmarks

```python
class QueryTreeDecoder:
    """
    Explore multiple query variations in parallel via tree decoding.

    Operations:
    - REWRITE: Rephrase query differently
    - DECOMPOSE: Break into sub-questions
    - DISAMBIGUATE: Clarify ambiguous terms
    """

    async def tree_decode(
        self,
        query: str,
        max_branches: int = 4,
        max_depth: int = 2
    ) -> QueryTree:
        """
        Generate tree of query variations.
        Each branch retrieves distinct context.
        Aggregate results weighted by branch confidence.
        """
```

---

## Phase 4: Scratchpad Enhancement

**Goal**: Richer working memory with semantic connections.

### 4.1 A-MEM Semantic Connections

**Research Basis**: A-MEM (arXiv 2502.12110) - 35% F1 improvement

```python
class SemanticMemoryNetwork:
    """
    Zettelkasten-inspired interconnected memory.

    Each memory carries:
    - Structured text attributes
    - Embedding vector
    - Dynamic connections based on similarity
    """

    def establish_connections(
        self,
        new_memory: Memory,
        existing_memories: List[Memory],
        similarity_threshold: float = 0.7
    ) -> List[MemoryConnection]:
        """
        Dynamically connect semantically similar memories.
        Enables traversal: finding → related findings → sources
        """
```

### 4.2 RAISE Four-Component Structure

**Research Basis**: RAISE (arXiv 2401.02777)

```python
@dataclass
class RAISEScratchpad:
    """
    Four-component working memory structure.
    """
    observations: List[Observation]      # Tool outputs, retrieved docs
    reasoning: List[ReasoningStep]       # Intermediate conclusions, confidence
    examples: List[Example]              # Successful patterns from session
    trajectory: List[TrajectoryStep]     # Execution history with timestamps

    def get_quality_signal(self) -> QualitySignal:
        """
        Extract quality signal from scratchpad contents.

        High uncertainty indicators:
        - Conflicting evidence in observations
        - Low confidence in reasoning steps
        - Missing information for trajectory goals
        """
```

---

## Phase 5: Template Reuse Optimization

**Goal**: Learn from successful searches for future efficiency.

### 5.1 Cross-Session Meta-Buffer

**Research Basis**: Buffer of Thoughts (NeurIPS 2024) - 12% cost of ToT

```python
class MetaBuffer:
    """
    Persistent storage of successful reasoning templates.

    Benefits:
    - 12% cost of Tree/Graph of Thoughts
    - 11-51% accuracy improvement
    - Templates transfer across task types
    """

    async def distill_template(
        self,
        successful_search: SearchResult,
        query_pattern: str
    ) -> ThoughtTemplate:
        """
        Extract reusable template from successful search.

        Template contains:
        - Query pattern (regex or semantic)
        - Successful query decomposition
        - Effective search strategies
        - Synthesis structure
        """

    async def retrieve_and_instantiate(
        self,
        new_query: str
    ) -> Optional[InstantiatedTemplate]:
        """
        Find matching template and instantiate for new query.
        Can skip low-confidence initial phases entirely.
        """
```

### 5.2 Self-Discover Reasoning Composition

**Research Basis**: Self-Discover (NeurIPS 2024, Google) - 87.5% correct structures

```python
class ReasoningComposer:
    """
    Compose task-specific reasoning from atomic modules.

    Meta-actions:
    - SELECT: Choose relevant reasoning modules
    - ADAPT: Make them task-specific
    - IMPLEMENT: Structure as executable plan
    """

    async def compose_strategy(
        self,
        query: str,
        available_modules: List[ReasoningModule]
    ) -> ComposedStrategy:
        """
        LLM composes task-specific strategy from atomic modules.

        Modules: critical_thinking, step_by_step, compare_contrast,
                 root_cause_analysis, evidence_synthesis, etc.
        """
```

---

## Implementation Priority & Timeline

### Immediate (Phase 1 - Context Curation) - HIGH IMPACT

| Component | Effort | Impact | Dependencies |
|-----------|--------|--------|--------------|
| DIG Scoring | 2 days | +17.9% accuracy | None |
| Two-Stage Filter | 2 days | Reduced noise | DIG Scoring |
| Redundancy Clustering | 1 day | Token efficiency | Embeddings |
| Curation Pipeline | 1 day | Integration | All above |

**Total: ~6 days**

### Short-term (Phase 2 - Confidence Calibration)

| Component | Effort | Impact | Dependencies |
|-----------|--------|--------|--------------|
| Entropy Monitor | 1 day | >50% tool call reduction | Ollama logprobs |
| Self-Consistency | 2 days | >40% sample reduction | Multiple synth |
| UCB Bandit | 2 days | 20-55% token reduction | Phase 1 |

**Total: ~5 days**

### Medium-term (Phase 3 - Query Generation)

| Component | Effort | Impact | Dependencies |
|-----------|--------|--------|--------------|
| FLARE Retrieval | 2 days | Better targeting | Entropy Monitor |
| Query Tree Decoder | 3 days | +33.5% on complex Q | Scratchpad |

**Total: ~5 days**

### Longer-term (Phase 4-5 - Infrastructure)

| Component | Effort | Impact | Dependencies |
|-----------|--------|--------|--------------|
| A-MEM Connections | 2 days | 35% F1 improvement | Embeddings |
| RAISE Structure | 2 days | Better signals | Scratchpad refactor |
| Meta-Buffer | 3 days | 12% cost reduction | ThoughtLibrary |
| Self-Discover | 3 days | 87.5% correct plans | Actor Factory |

**Total: ~10 days**

---

## Integration with Existing Pipeline

### Current Flow
```
Query → Analyze → Search → Scrape → Verify → Synthesize → Reflect
```

### Enhanced Flow with Context Curation
```
Query → Analyze → Search → Scrape
                              ↓
                    ┌─────────────────────┐
                    │  CONTEXT CURATION   │
                    │  1. Deduplication   │
                    │  2. DIG Scoring     │
                    │  3. Two-Stage Filter│
                    │  4. Clustering      │
                    │  5. Representative  │
                    └─────────────────────┘
                              ↓
          ┌─────────────────────────────────────┐
          │  CONFIDENCE-CALIBRATED ITERATION    │
          │  - Entropy monitoring               │
          │  - UCB action selection             │
          │  - Self-consistency check           │
          │  Loop until: confident OR max iter  │
          └─────────────────────────────────────┘
                              ↓
                    Verify → Synthesize → Reflect
```

---

## Success Metrics

| Metric | Current | Target | Research Basis |
|--------|---------|--------|----------------|
| Confidence (avg) | 65-73% | 80%+ | CRAG +7% PopQA |
| Token efficiency | Baseline | -30% | BoT 12% cost |
| Redundancy | Unknown | <10% | Context-Picker |
| Iteration count | 1-3 | 1-2 avg | UALA >50% reduction |
| Answer accuracy | Good | +15-20% | InfoGain +17.9% |

---

## Files to Create

```
memOS/server/agentic/
├── context_curator.py          # Phase 1: Main curation pipeline
├── information_gain.py         # Phase 1: DIG scoring
├── redundancy_detector.py      # Phase 1: Clustering & dedup
├── entropy_monitor.py          # Phase 2: Confidence signals
├── self_consistency.py         # Phase 2: Convergence check
├── iteration_bandit.py         # Phase 2: UCB action selection
├── flare_retriever.py          # Phase 3: Forward-looking retrieval
├── query_tree.py               # Phase 3: RQ-RAG tree decoding
├── semantic_memory.py          # Phase 4: A-MEM connections
├── meta_buffer.py              # Phase 5: Cross-session templates
└── reasoning_composer.py       # Phase 5: Self-Discover
```

---

## References

- InfoGain-RAG: Kuaishou Technology
- Context-Picker: arXiv 2512.14465
- UALA: ACL Findings 2024
- CISC: arXiv 2502.20233
- REFRAIN: arXiv 2510.10103
- CRAG: arXiv 2401.15884
- Self-RAG: ICLR 2024 Oral
- FLARE: EMNLP 2023
- RQ-RAG: arXiv 2404.00610
- GSW: arXiv 2406.04555 (AAAI 2026)
- bMAS: arXiv 2507.01701
- A-MEM: arXiv 2502.12110
- RAISE: arXiv 2401.02777
- Buffer of Thoughts: arXiv 2406.04271 (NeurIPS 2024)
- Self-Discover: NeurIPS 2024, Google DeepMind

---

*Plan created: 2025-12-29*
*Based on: Multi-agent systems research synthesis (50+ papers, 2024-2025)*
