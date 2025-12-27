# memOS Agentic Workflow Enhancement Plan

## Research Summary

Based on comprehensive research into cutting-edge agentic AI frameworks from 2024-2025, this document outlines high-impact enhancements for the memOS agentic search system.

### Research Sources

| Framework | Key Innovation | Performance |
|-----------|----------------|-------------|
| **AIME** (ByteDance) | Dynamic Planner, Actor Factory, Progress Management | 77.6% GAIA, 92.3% WebVoyager |
| **GSW** (Hippocampal Memory) | Actor-centric episodic memory, Operator/Reconciler | 51% token reduction, 20% retrieval improvement |
| **DAG-Math / DoT** | Graph-structured reasoning with shared derivations | Perfect reasoning rate metric |
| **Graph of Thoughts** | Multi-path exploration with convergence | 200-300% improvement over ToT |
| **Buffer of Thoughts** | Reusable thought-template library | Llama3-8B surpasses Llama3-70B |

---

## Gap Analysis: Current State vs. Research Findings

### Current memOS Capabilities
| Feature | Implementation | Location |
|---------|---------------|----------|
| Agent Step Graph | KVFlow-inspired workflow DAG | `agent_step_graph.py` |
| Enhanced Reasoning | Pre-Act, self-reflection, stuck detection | `enhanced_reasoning.py` |
| Scratchpad/Blackboard | Question tracking, findings, agent notes | `scratchpad.py` |
| Query Classification | DeepSeek-R1 categorization | `query_classifier.py` |
| Content Cache | Semantic query + content caching | `content_cache.py` |
| Memory Tiers | Three-tier cold/warm/hot | `memory_tiers.py` |

### Identified Gaps

| Gap | Research Source | Impact | Priority |
|-----|-----------------|--------|----------|
| **Dual-Output Planning** | AIME | Continuous strategic + tactical planning | HIGH |
| **Actor Factory Pattern** | AIME | Dynamic agent specialization with tool bundles | HIGH |
| **Entity-Centric Memory** | GSW | Actor tracking reduces token usage by 51% | HIGH |
| **Multi-Path Reasoning** | GoT/BoT | Graph branching/aggregation for complex queries | MEDIUM |
| **Thought Template Library** | BoT | Reusable reasoning patterns | MEDIUM |
| **Verification DAG** | GoV | Topologically-ordered claim verification | MEDIUM |
| **Progress Update Tool** | AIME | Proactive status reporting from agents | LOW |

---

## Phase 1: AIME-Style Progress Management (Week 1-2)

### 1.1 Dual-Output Dynamic Planner

**Location**: `agentic/dynamic_planner.py` (NEW FILE)

Enhance the existing Pre-Act pattern to produce simultaneous strategic and tactical outputs:

```python
@dataclass
class PlannerOutput:
    """Dual output from Dynamic Planner"""
    strategic: TaskHierarchy  # Updated global task list
    tactical: NextAction      # Immediate executable action
    reasoning: str            # Planner's thinking

class DynamicPlanner:
    """
    AIME-style Dynamic Planner with dual outputs.

    Unlike one-shot planners, continuously operates in a loop:
    1. Receives feedback from executed actions
    2. Updates strategic understanding (task hierarchy)
    3. Dispatches next tactical action
    """

    async def plan_iteration(
        self,
        goal: str,
        current_tasks: List[TaskNode],
        history: List[ExecutionResult],
        scratchpad: AgenticScratchpad
    ) -> PlannerOutput:
        """
        Produce dual outputs: updated task hierarchy + next action.

        Formula: (L_{t+1}, g_{t+1}) = LLM(goal, L_t, H_t)
        """
        pass
```

**Integration Points**:
- Wraps existing `PreActPlan` from `enhanced_reasoning.py`
- Feeds into `AgenticScratchpad` for task tracking
- Consumes `ExecutionResult` from ReAct loops

### 1.2 Hierarchical Progress List

**Location**: `agentic/scratchpad.py` (ENHANCE)

Add AIME-style markdown progress tracking:

```python
class TaskNode(BaseModel):
    """Hierarchical task with AIME-style attributes"""
    id: str
    description: str
    status: TaskStatus  # pending, in_progress, completed, failed, blocked
    completion_criteria: str
    subtasks: List['TaskNode'] = []
    artifacts: List[str] = []  # Pointers to files, URLs, IDs
    dependencies: List[str] = []  # Task IDs this depends on

class ProgressManager:
    """AIME-style hierarchical progress management"""

    def render_markdown(self) -> str:
        """Render task tree as markdown for LLM context"""
        pass

    def update_progress(self, task_id: str, status: TaskStatus, message: str):
        """Called by agents to report progress"""
        pass

    def get_executable_tasks(self) -> List[TaskNode]:
        """Return tasks with satisfied dependencies"""
        pass
```

### 1.3 Progress Update Tool

**Location**: `agentic/orchestrator.py` (ENHANCE)

Add system-provided tool for proactive progress reporting:

```python
PROGRESS_UPDATE_TOOL = {
    "name": "update_progress",
    "description": "Report significant progress or obstacles to the planner",
    "parameters": {
        "status": "in_progress|milestone_reached|obstacle_encountered",
        "message": "Description of what happened"
    }
}
```

---

## Phase 2: GSW-Style Entity Memory (Week 2-3)

### 2.1 Entity Tracker

**Location**: `agentic/entity_tracker.py` (NEW FILE)

Implement actor-centric working memory inspired by GSW:

```python
@dataclass
class EntityState:
    """GSW-style actor state with temporal tracking"""
    id: str
    name: str
    roles: List[RoleAssignment]  # {role, context, timestamp, confidence}
    states: List[str]  # Current state indicators
    interactions: List[EntityRelation]
    timeline: List[Event]

class EntityTracker:
    """
    GSW Operator-inspired entity extraction and tracking.

    Reduces token usage by 51% via entity-centric summaries
    instead of full document retrieval.
    """

    def __init__(self, llm_client):
        self.entities: Dict[str, EntityState] = {}
        self.relationships: List[Relationship] = []

    async def extract_entities(self, content: str) -> List[EntityState]:
        """
        GSW Operator: Extract actors, roles, states, verbs from content.

        Uses semantic role labeling to identify:
        - ACTORS: Named entities with unique IDs
        - ROLES: What role each actor plays
        - STATES: Current state modulating behavior
        - VERBS: Actions with (subject, predicate, object, time, place)
        """
        pass

    async def reconcile(
        self,
        new_entities: List[EntityState],
        existing_memory: Dict[str, EntityState]
    ) -> Dict[str, EntityState]:
        """
        GSW Reconciler: Integrate new entities into coherent timeline.

        - Resolve entity references (same entity, different mentions)
        - Maintain temporal ordering
        - Track state transitions
        """
        pass

    def generate_entity_summary(self, entity_id: str, query: str) -> str:
        """
        Generate query-relevant summary for an entity.

        This is the key to 51% token reduction:
        Instead of retrieving full documents, generate focused summaries.
        """
        pass
```

### 2.2 Scratchpad Integration

**Location**: `agentic/scratchpad.py` (ENHANCE)

Add entity tracking to the blackboard:

```python
class AgenticScratchpad(BaseModel):
    # ... existing fields ...

    # GSW-style entity tracking
    entity_tracker: EntityTracker = Field(default_factory=EntityTracker)

    # Forward-falling questions (GSW concept)
    pending_questions: List[str] = Field(default_factory=list)

    def add_finding_with_entities(
        self,
        finding: str,
        source_url: str,
        entities: List[EntityState]
    ):
        """Add finding and update entity tracker simultaneously"""
        pass

    def get_entity_context(self, query: str) -> str:
        """
        Generate entity-centric context for synthesis.

        Instead of dumping all findings, generate focused summaries
        for entities relevant to the query.
        """
        pass
```

---

## Phase 3: DAG-Based Reasoning (Week 3-4)

### 3.1 Reasoning DAG Structure

**Location**: `agentic/reasoning_dag.py` (NEW FILE)

Implement GoT-style graph reasoning:

```python
class NodeType(Enum):
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    CONCLUSION = "conclusion"
    CRITIQUE = "critique"
    REFINEMENT = "refinement"

@dataclass
class ReasoningNode:
    """Node in reasoning DAG"""
    id: str
    content: str
    node_type: NodeType
    parent_ids: List[str]  # DAG edges
    status: NodeStatus  # pending, validated, invalidated
    confidence: float
    source: Optional[str]

class ReasoningDAG:
    """
    Graph of Thoughts implementation for multi-path reasoning.

    Unlike linear CoT:
    - Multiple paths can explore simultaneously
    - Shared derivations avoid redundant reasoning
    - Aggregation combines insights from multiple paths
    - Backtracking enabled via graph structure
    """

    def __init__(self):
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_ids: List[str] = []
        self.sink_ids: List[str] = []

    def branch(self, parent_id: str, num_branches: int = 3) -> List[str]:
        """
        GoT Branching: Generate multiple reasoning paths from a node.
        Returns IDs of new child nodes.
        """
        pass

    def aggregate(self, node_ids: List[str]) -> str:
        """
        GoT Aggregation: Combine insights from multiple nodes.
        Creates a new node synthesizing the inputs.
        """
        pass

    def refine(self, node_id: str, critique: str) -> str:
        """
        GoT Refinement: Improve a node based on critique.
        """
        pass

    def verify_topologically(self) -> List[VerificationResult]:
        """
        GoV-style verification: Verify nodes in topological order.
        Premises verified before conclusions.
        """
        pass

    def get_convergent_answer(self) -> str:
        """
        Extract final answer from sink nodes.
        Multiple paths converge to conclusion.
        """
        pass
```

### 3.2 Integration with Orchestrator

**Location**: `agentic/orchestrator_enhanced.py` (ENHANCE)

Add DAG reasoning to the search workflow:

```python
async def enhanced_search_with_dag(
    self,
    request: SearchRequest,
    scratchpad: AgenticScratchpad
) -> SearchResponse:
    """
    Enhanced search using DAG reasoning for complex queries.

    Flow:
    1. Classify query complexity
    2. For complex queries, use GoT branching
    3. Explore multiple reasoning paths in parallel
    4. Aggregate findings from successful paths
    5. Verify conclusions topologically
    """

    # Create reasoning DAG
    dag = ReasoningDAG()

    # Initial hypothesis generation (branching)
    hypotheses = await self._generate_hypotheses(request.query, num_branches=3)
    for h in hypotheses:
        dag.add_node(h, node_type=NodeType.HYPOTHESIS)

    # Parallel exploration of each path
    async with asyncio.TaskGroup() as tg:
        for h in hypotheses:
            tg.create_task(self._explore_path(h, dag, scratchpad))

    # Aggregate successful paths
    successful_nodes = [n for n in dag.sink_ids if dag.nodes[n].status == NodeStatus.VALIDATED]
    final_answer = dag.aggregate(successful_nodes)

    # Verify topologically
    verification_results = dag.verify_topologically()

    return SearchResponse(
        synthesized_context=final_answer,
        reasoning_trace=dag.to_trace()
    )
```

---

## Phase 4: Thought Template Library (Week 4-5)

### 4.1 Buffer of Thoughts Implementation

**Location**: `agentic/thought_library.py` (NEW FILE)

Implement BoT-style reusable reasoning templates:

```python
@dataclass
class ThoughtTemplate:
    """Reusable high-level reasoning template"""
    id: str
    name: str
    description: str
    applicability: List[str]  # Query types this applies to
    structure: str  # Template with placeholders
    examples: List[str]
    embedding: List[float]  # For similarity retrieval
    usage_count: int = 0
    success_rate: float = 0.0

class ThoughtLibrary:
    """
    Buffer of Thoughts: Meta-buffer of reusable reasoning patterns.

    Benefits:
    - Llama3-8B + BoT can surpass Llama3-70B
    - Reduces token usage by reusing proven patterns
    - Continuous improvement via buffer-manager
    """

    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.templates: Dict[str, ThoughtTemplate] = {}
        self.embedder = EmbeddingModel(embedding_model)
        self._load_default_templates()

    def _load_default_templates(self):
        """Load pre-defined templates for common reasoning patterns"""
        self.add_template(ThoughtTemplate(
            id="source_credibility",
            name="Evaluate Source Credibility",
            description="Assess reliability of information source",
            applicability=["verification", "fact_checking"],
            structure="""
            To evaluate credibility of {source}:
            1. Check domain authority ({domain})
            2. Look for author credentials
            3. Verify date of publication
            4. Cross-reference with known reliable sources
            5. Assign confidence score: {confidence}
            """,
            examples=[]
        ))
        # Add more default templates...

    async def retrieve_template(self, query: str, top_k: int = 3) -> List[ThoughtTemplate]:
        """Retrieve relevant templates via embedding similarity"""
        query_embedding = await self.embedder.embed(query)
        similarities = [
            (t, cosine_similarity(query_embedding, t.embedding))
            for t in self.templates.values()
        ]
        return sorted(similarities, key=lambda x: -x[1])[:top_k]

    async def instantiate_template(
        self,
        template: ThoughtTemplate,
        context: Dict[str, Any]
    ) -> str:
        """Customize template with task-specific reasoning"""
        return template.structure.format(**context)

    def update_from_success(self, template_id: str, outcome: dict):
        """Buffer-Manager: Learn from successful reasoning"""
        template = self.templates[template_id]
        template.usage_count += 1
        if outcome['success']:
            template.success_rate = (
                template.success_rate * (template.usage_count - 1) + 1
            ) / template.usage_count
```

### 4.2 Integration with Enhanced Reasoning

**Location**: `agentic/enhanced_reasoning.py` (ENHANCE)

Add thought library to Pre-Act planning:

```python
async def create_pre_act_plan_with_templates(
    self,
    query: str,
    thought_library: ThoughtLibrary
) -> PreActPlan:
    """
    Pre-Act planning enhanced with Buffer of Thoughts.

    1. Retrieve relevant thought templates
    2. Instantiate templates for current query
    3. Build plan using proven reasoning structures
    """
    # Retrieve relevant templates
    templates = await thought_library.retrieve_template(query, top_k=3)

    # Build plan using templates as scaffolding
    template_context = self._build_template_context(query, templates)

    # Generate plan with template guidance
    plan = await self._generate_plan_with_templates(query, template_context)

    return plan
```

---

## Phase 5: Tool Bundle Selection (Week 5-6)

### 5.1 Actor Factory

**Location**: `agentic/actor_factory.py` (NEW FILE)

Implement AIME-style on-demand agent specialization:

```python
@dataclass
class ToolBundle:
    """Pre-packaged tool collection for functional completeness"""
    name: str
    description: str
    tools: List[str]  # Tool names
    model_requirements: List[str]  # Required model capabilities

# Predefined bundles
TOOL_BUNDLES = {
    "web_research": ToolBundle(
        name="WebResearch",
        description="Web search, URL fetching, content extraction",
        tools=["brave_search", "scrape_url", "extract_content"],
        model_requirements=["text_generation"]
    ),
    "vision_extraction": ToolBundle(
        name="VisionExtraction",
        description="Screenshot capture, VL model extraction",
        tools=["capture_screenshot", "vl_extract"],
        model_requirements=["vision"]
    ),
    "verification": ToolBundle(
        name="Verification",
        description="Fact checking, cross-referencing",
        tools=["search_for_verification", "compare_sources"],
        model_requirements=["reasoning"]
    ),
    "synthesis": ToolBundle(
        name="Synthesis",
        description="Content synthesis with citations",
        tools=["synthesize_content", "add_citations"],
        model_requirements=["long_context"]
    )
}

class ActorFactory:
    """
    AIME Actor Factory: Create purpose-built agents on demand.

    Components assembled per subtask:
    - LLM: Cognitive engine (model selection based on requirements)
    - Toolkit: Selected tool bundles (not individual tools)
    - Persona: Customized system prompt
    - Memory: Relevant context from scratchpad
    """

    def __init__(self, model_selector, tool_registry):
        self.model_selector = model_selector
        self.tools = tool_registry

    async def create_actor(self, subtask: TaskNode) -> DynamicActor:
        """
        Instantiate specialized actor for subtask.

        Formula: A_t = {LLM_t, T_t, P_t, M_t}
        """
        # Select tool bundles based on subtask requirements
        bundles = self._select_bundles(subtask)

        # Select model based on bundle requirements
        model = await self._select_model(bundles)

        # Generate persona
        persona = await self._generate_persona(subtask)

        # Compose system prompt
        prompt = self._compose_prompt(persona, bundles, subtask)

        return DynamicActor(
            model=model,
            bundles=bundles,
            persona=persona,
            system_prompt=prompt
        )

    def _select_bundles(self, subtask: TaskNode) -> List[ToolBundle]:
        """Select appropriate tool bundles for subtask"""
        required = self._analyze_requirements(subtask)
        return [TOOL_BUNDLES[b] for b in required if b in TOOL_BUNDLES]
```

---

## Implementation Priority

### Immediate (Week 1-2) - High Impact, Low Complexity
1. **Progress Management** - Enhance scratchpad with AIME-style task hierarchy
2. **Progress Update Tool** - Add proactive reporting to agents

### Short-term (Week 2-4) - High Impact, Medium Complexity
3. **Entity Tracker** - GSW-style actor-centric memory
4. **Reasoning DAG** - Replace linear reasoning with graph structure

### Medium-term (Week 4-6) - Medium Impact, Higher Complexity
5. **Thought Library** - Buffer of Thoughts pattern
6. **Actor Factory** - AIME-style tool bundle selection

### Long-term (Future)
7. **Full Verification DAG** - GoV-style topological verification
8. **Memory Consolidation** - GSW-style time-decay and recall

---

## Expected Outcomes

| Enhancement | Expected Improvement | Metric |
|-------------|---------------------|--------|
| Dual-Output Planner | Better error recovery | Task completion rate |
| Entity Tracker | 51% token reduction | Context tokens used |
| Reasoning DAG | Handle complex queries | Multi-step accuracy |
| Thought Library | Faster convergence | Iterations to answer |
| Tool Bundles | Reduced cognitive load | Tool selection accuracy |

---

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `agentic/dynamic_planner.py` | AIME-style dual-output planner |
| `agentic/entity_tracker.py` | GSW-style actor-centric memory |
| `agentic/reasoning_dag.py` | GoT-style graph reasoning |
| `agentic/thought_library.py` | BoT-style template library |
| `agentic/actor_factory.py` | AIME-style agent specialization |

### Enhanced Files
| File | Enhancement |
|------|-------------|
| `agentic/scratchpad.py` | Add entity tracking, task hierarchy |
| `agentic/orchestrator_enhanced.py` | Integrate DAG reasoning |
| `agentic/enhanced_reasoning.py` | Add thought library integration |
| `agentic/prompts.py` | Add tool bundle descriptions |

---

## Research References

### AIME (ByteDance)
- Paper: https://arxiv.org/abs/2507.11988
- Related: [Trae Agent](https://github.com/bytedance/trae-agent), [DeerFlow](https://github.com/bytedance/deer-flow)

### GSW (Generative Semantic Workspace)
- Paper: https://arxiv.org/abs/2511.07587
- Related: [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

### DAG Reasoning
- [Diagram of Thought](https://arxiv.org/abs/2409.10038) | [GitHub](https://github.com/diagram-of-thought/diagram-of-thought)
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) | [GitHub](https://github.com/spcl/graph-of-thoughts)
- [Buffer of Thoughts](https://arxiv.org/abs/2406.04271) | [GitHub](https://github.com/YangLing0818/buffer-of-thought-llm)
- [Graph of Verification](https://arxiv.org/abs/2506.12509)
- [PathRAG](https://arxiv.org/abs/2502.14902) | [GitHub](https://github.com/BUPT-GAMMA/PathRAG)
