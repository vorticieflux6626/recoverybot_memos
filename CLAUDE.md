# memOS Server

> **Updated**: 2026-01-19 | **Parent**: [Root CLAUDE.md](../CLAUDE.md) | **Version**: 0.91.0

## Quick Reference

| Action | Command | Notes |
|--------|---------|-------|
| **Activate venv** | `cd server && source venv/bin/activate` | **REQUIRED first!** |
| Start Server | `uvicorn main:app --host 0.0.0.0 --port 8001` | After venv activation |
| Run Tests | `pytest tests/ -v` | After venv activation |
| **Test Search** | `./test_search.sh "query" [preset]` | Pipeline testing with auto-timeout |
| DB Migration | `alembic upgrade head` | After venv activation |
| Format Code | `ruff format .` | After venv activation |
| Lint Code | `ruff check .` | After venv activation |
| Ollama Config | `source setup_ollama_optimization.sh && systemctl restart ollama` | Apply optimizations |
| **LLM Config** | `curl localhost:8001/api/v1/config/llm-models` | View model assignments |
| Apply Preset | `curl -X POST localhost:8001/api/v1/config/llm-models/presets/speed` | speed/quality/balanced |
| **Prompt Config** | `from agentic.prompt_config import get_prompt_config` | Central prompt access |

## Configuration System

### Central Configuration Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `config/llm_models.yaml` | LLM model assignments | `get_llm_config()` |
| `config/prompts.yaml` | All agent prompts | `get_prompt_config()` |
| `config/settings.py` | Server settings | `get_settings()` |

### Prompt Configuration (`config/prompts.yaml`)

All LLM prompts are centralized in `config/prompts.yaml`. Access via:

```python
from agentic.prompt_config import get_prompt_config

config = get_prompt_config()

# Access specific prompts
synth_prompt = config.agent_prompts.synthesizer.main
cod_instruction = config.instructions.chain_of_draft
template = config.templates.search_query_generation

# Format with variables
prompt = synth_prompt.format(query=query, results_text=results)
```

**Structure:**
- `instructions`: Shared snippets (chain-of-draft, citation requirements)
- `system_prompts`: Core system prefixes
- `agent_prompts`: Per-agent prompts (analyzer, synthesizer, verifier, etc.)
- `templates`: Reusable templates with `{placeholders}`

**Environment Override:** Set `MEMOS_PROMPTS_CONFIG=/path/to/prompts.yaml`

### Semantic Query Parser (`agentic/semantic_query_parser.py`)

The semantic query parser extracts the core semantic content from natural language queries before searching. This prevents verbose questions from polluting search results with dictionary definitions.

**Problem Solved:**
```
Input: "What is the FANUC SRVO-062 alarm code and its meaning?"
Without parser: SearXNG matches on "what", "is", "meaning" → grammar sites
With parser: Searches "SRVO-062 FANUC alarm code" → FANUC documentation
```

**Key Features:**
- **Technical entity extraction**: Regex patterns for error codes (SRVO-062), part numbers (A06B-*), model names (R-30iB)
- **Question word removal**: Filters "what", "how", "where", "why" that pollute search results
- **Intent detection**: Classifies as troubleshooting, definition, procedure, etc.
- **Focus term extraction**: Identifies what the query is "about" using noun phrases

**Usage:**
```python
from agentic.semantic_query_parser import parse_query, generate_search_queries

parsed = parse_query("What causes FANUC SRVO-062 alarm?")
print(parsed.technical_entities)  # [('SRVO-062', 'FANUC_ERROR')]
print(parsed.intent)              # "causal"
print(parsed.optimized_query)     # "SRVO-062 FANUC alarm causes"

# Generate search queries (best first)
queries = generate_search_queries(parsed)
# ['SRVO-062', 'SRVO-062 FANUC alarm causes', 'SRVO-062 cause']
```

**Integration Points:**
- `analyzer.py`: `analyze()` and `create_search_plan()` optimize queries before search
- Automatic - no configuration needed

## Critical Rules

1. **NEVER** push from wrong directory - verify with `pwd && git remote -v` first
2. **NEVER** use sync operations inside async functions - always use `await` with aiohttp/asyncpg
3. **NEVER** hardcode API keys - use environment variables via `settings.py`
4. **ALWAYS** use `StateFlow.update{}` pattern when updating shared state (no Mutex)
5. **ALWAYS** include `request_id` in error responses for debugging
6. **ALWAYS** return unified response format: `{success, data, meta, errors}`
7. **ALWAYS** activate the project's local venv before running Python commands:
   ```bash
   cd /home/sparkone/sdd/Recovery_Bot/memOS/server
   source venv/bin/activate  # REQUIRED before pytest, python, pip
   ```

## Feature Registry

**Before adding new features, check the registry:** `grep -i "feature" ../../.memOS/FEATURE_REGISTRY.yaml`

memOS owns the majority of features in the registry (50+ agentic search flags). Key categories:

| Category | Feature Count | Examples |
|----------|---------------|----------|
| `agentic_search` | 20+ | query_analysis, crag_evaluation, self_reflection |
| `retrieval` | 7 | hyde, hybrid_reranking, cross_encoder, flare |
| `caching` | 8 | semantic_cache, ttl_pinning, graph_cache |
| `reasoning` | 7 | meta_buffer, reasoning_dag, entity_tracking |
| `tts` | 4 (server-side) | melotts, openvoice, emotivoice |

**When adding new features:**
1. Add to `../.memOS/FEATURE_REGISTRY.yaml` with `status: "planned"`
2. Add flag to `FeatureConfig` in `agentic/orchestrator_universal.py`
3. Update preset configurations in `PRESET_CONFIGS`
4. Validate: `python ../../scripts/validate_feature_registry.py`

## SSOT Responsibilities

memOS is the authoritative source for:

| Data Domain | Description |
|-------------|-------------|
| **User Authentication** | JWT tokens, sessions, user identity |
| **User Profiles** | Preferences, settings, consent |
| **Quests & Progress** | All quest logic, achievements, gamification |
| **User Memories** | Semantic memory storage with privacy controls |
| **Chat Context** | Conversation history, context injection |
| **Agentic Search** | Orchestration, caching, synthesis |

## Dependencies

| Service | Port | Required | Purpose |
|---------|------|----------|---------|
| Ollama | 11434 | Yes | LLM inference |
| PostgreSQL | 5432 | Yes | Primary database with pgvector |
| SearXNG | 8888 | No | Metasearch (fallback to DDG) |
| Redis | 6379 | No | Session caching |
| PDF Tools API | 8002 | No | FANUC document retrieval |
| Docling | 8003 | No | Document processing (97.9% table accuracy) |

## Testing

```bash
# Activate venv first
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=agentic --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_orchestrator_presets.py -v

# Run integration tests only
pytest tests/integration/ -v
```

**Test Directories:**
- `tests/unit/` - Isolated unit tests
- `tests/integration/` - Cross-component tests
- `tests/contracts/` - API contract validation (via root tests/)

### Search Pipeline Testing

Use `test_search.sh` for reliable search pipeline testing with intelligent timeouts:

```bash
# Basic usage (BALANCED preset, auto-timeout)
./test_search.sh "What is injection molding?"

# With specific preset
./test_search.sh "SRVO-063 encoder error" ENHANCED

# Verbose mode (shows progress)
./test_search.sh "short shots PA66 mold temperature" RESEARCH --verbose

# JSON output for scripting
./test_search.sh "query" MINIMAL --json
```

**Timeout Calculation by Preset:**

| Preset | Base Timeout | Per 10 Words | Typical Query |
|--------|-------------|--------------|---------------|
| MINIMAL | 60s | +10s | 70-90s |
| BALANCED | 120s | +15s | 135-165s |
| ENHANCED | 180s | +20s | 200-240s |
| RESEARCH | 300s | +30s | 330-390s |
| FULL | 420s | +40s | 460-540s |

The script:
- Validates server availability before running
- Shows progress from server logs (with `--verbose`)
- Parses and formats results with confidence scores
- Highlights if `technical_docs` feature is active
- Checks for IMM-specific terms in synthesis

## MCP Code Intelligence

This project is indexed by the ecosystem-wide MCP code intelligence layer.

**Infrastructure**: `/home/sparkone/sdd/mcp_infrastructure/`

| Tool | Command | Purpose |
|------|---------|---------|
| **Search code** | `kit search /home/sparkone/sdd/Recovery_Bot "query"` | Find code patterns |
| **Find usages** | `kit usages /home/sparkone/sdd/Recovery_Bot "ClassName"` | Find symbol references |
| **Dependencies** | `kit dependencies /path/to/file.py -l python` | Analyze imports |
| **File tree** | `kit file-tree /home/sparkone/sdd/Recovery_Bot/memOS` | View structure |
| **Re-index** | `/home/sparkone/sdd/mcp_infrastructure/scripts/index_ecosystem.sh` | Refresh after major changes |
| **Health check** | `/home/sparkone/sdd/mcp_infrastructure/scripts/check_mcp_health.sh` | Verify services |

**Index location**: Root project `.kit/index.json` (51.4 MB, includes memOS)

**Cross-project search**: This project is indexed alongside PDF_Extraction_Tools, MCP_Servers, and MCP_Node_Editor for unified semantic search.

**Documentation**: See `/home/sparkone/sdd/mcp_infrastructure/CLAUDE.md` for full architecture details.

---

## Overview

memOS is a memory management, quest/gamification, and **intelligent data injection** system for the Recovery Bot Android application. It provides REST APIs for storing user memories, tracking progress through quests and achievements, and **orchestrating agentic AI workflows for enhanced research and troubleshooting assistance**.

## Strategic Vision: Intelligent Research Hub

memOS is the **central intelligence layer** for the Recovery Bot ecosystem, responsible for:

1. **Memory Management** (Current) - Secure storage and semantic search
2. **Quest Gamification** (Current) - Progress tracking and achievements
3. **Agentic Search Orchestration** (Active) - Multi-agent web search and context enhancement
4. **Context Injection** (Active) - Intelligent data augmentation for LLM conversations

### Core Architecture Principle
memOS serves as the **Single Source of Truth (SSOT)** for user context, memory, and intelligent data retrieval. All context augmentation flows through memOS before reaching the primary LLM.

---

## Current Status (2026-01-02)

### Implementation Status Summary

All major implementation phases are **COMPLETE**. See [server/docs/IMPLEMENTATION_PHASES.md](server/docs/IMPLEMENTATION_PHASES.md) for detailed documentation.

| Phase Range | Description | Status |
|-------------|-------------|--------|
| **Phases 1-10** | Core agentic pipeline (AIME planning, GSW entity tracking, GoT reasoning, Self-RAG, CRAG) | ✅ Complete |
| **Phases 11-20** | SSE integration, context curation, confidence halting, FLARE/RQ-RAG, A-MEM/RAISE scratchpad | ✅ Complete |
| **Phases 21-27** | Template optimization, PDF integration, HSEA indexing, error standardization, feature synergy | ✅ Complete |
| **Part F** | Benchmark test suite, technical accuracy scorer | ✅ Complete |
| **Part G.1-G.7** | RAG foundation, hierarchical retrieval, agent coordination, hyperbolic embeddings, OT fusion, TSDAE | ✅ Complete |
| **Part K.2-K.3** | Docling processor, table complexity routing | ✅ Complete |
| **Phase 50** | Extended diagram integration (circuit, harness, pinout, flowchart) with intent detection | ✅ Complete |

### Test Coverage
- Unit tests: 359 passing
- Integration tests: 42 passing
- Contract tests: 11 passing
- **Total: 412 tests**

### Current Improvement Focus

See [server/agentic/AGENTIC_IMPROVEMENT_PLAN.md](server/agentic/AGENTIC_IMPROVEMENT_PLAN.md) for the active improvement roadmap.

| Metric | Current | Target |
|--------|---------|--------|
| Confidence Score | 57-69% | 80%+ |
| Term Coverage | 50% | 80%+ |
| Citation Accuracy | ~50% | 100% |
| Response Time | 120-180s | <60s |

### Observability Enhancement

See [server/agentic/OBSERVABILITY_IMPROVEMENT_PLAN.md](server/agentic/OBSERVABILITY_IMPROVEMENT_PLAN.md) for the observability roadmap.

**Priority P0-P3 (Complete):**
- Decision Logger - Track agent decisions with reasoning (`decision_logger.py`)
- Context Tracker - Monitor context flow between pipeline stages (`context_tracker.py`)
- LLM Call Logger - Comprehensive LLM invocation tracking (`llm_logger.py`)
- Scratchpad Observer - Track scratchpad state changes (`scratchpad_observer.py`)
- Technician Log - Human-readable diagnostic summaries (`technician_log.py`)
- Confidence Logger - Multi-signal confidence breakdown (`confidence_logger.py`)
- GenAI Semantic Conventions - OpenTelemetry GenAI attributes (`tracing.py`)
- Dashboard Endpoint - Aggregate observability views (`observability_dashboard.py`)

### Prompt & Context Engineering (2026-01-02)

See [server/agentic/PROMPT_CONTEXT_IMPROVEMENT_PLAN.md](server/agentic/PROMPT_CONTEXT_IMPROVEMENT_PLAN.md) for the improvement roadmap.

**Bugs Fixed:**
- LLM URL bug - Components were passing `None` instead of resolved URL
- float16 serialization - numpy float types in reranking couldn't be JSON serialized

**Pipeline Audit Results:**
- Duration: 82.7s (target: <60s)
- Confidence: 56% (target: 80%+)
- Found relevance drift issue (synthesis answering off-topic)

**Planned Improvements:**
- P0: Chain-of-Draft for fast agents (68-92% token reduction)
- P1: Role-based agent personas for better topic focus
- P1: Lost-in-middle mitigation for document reordering
- P2: `/no_think` suffix for qwen3:8b, temperature tuning

### Diagnostic Path Traversal Enhancement (2026-01-04)

Enhanced integration with PDF Extraction Tools for industrial troubleshooting:

**New Features:**
- **Symptom-Based Entry**: Query without knowing error codes using `INDICATES` edge traversal
  - Pattern: `symptom → INDICATES → error_code → RESOLVED_BY → remedy`
  - Example: "servo motor overheating" → SRVO-063 → Check encoder cable
- **Structured Causal Chain**: XML-like output for better LLM synthesis comprehension
  - Groups steps by type (error, diagnosis, solution, procedure)
  - Includes edge type markers (CAUSED_BY, RESOLVED_BY)
  - Preserves severity classification from error category
- **Configurable Traversal Modes**:
  - `semantic_astar` (default): A* with semantic embeddings
  - `flow_based`: PathRAG with alpha/theta pruning (RESEARCH preset)
  - `multi_hop`: Cross-document reasoning (FULL preset)

**FeatureConfig Parameters:**
```python
enable_symptom_entry: bool = False          # INDICATES edge traversal
enable_structured_causal_chain: bool = False  # XML-like output
technical_traversal_mode: str = "semantic_astar"
technical_max_hops: int = 4                 # Depth (2-6)
technical_beam_width: int = 10              # Width (5-50)
```

**Preset Integration:**
| Preset | Mode | Max Hops | Beam Width |
|--------|------|----------|------------|
| ENHANCED | semantic_astar | 4 | 10 |
| RESEARCH | flow_based | 5 | 20 |
| FULL | multi_hop | 6 | 50 |

**Files Modified:**
- `core/document_graph_service.py`: `query_by_symptom()`, `format_causal_chain_for_synthesis()`, `get_structured_troubleshooting_context()`
- `agentic/orchestrator_universal.py`: PHASE 4.6, updated `_search_technical_docs()`, new FeatureConfig params

### Cross-Domain Hallucination Mitigation (Phase 48, 2026-01-04)

Multi-layered validation system to catch spurious cross-domain claims in industrial automation queries:

**Problem Solved:**
- False causal chains: "SRVO-068 robot alarm causes IMM hydraulic fluctuations" (impossible)
- Fabricated entities: Invalid error codes, wrong part number formats
- Spurious cross-domain links: Robot servo → Hydraulic → eDart (no physical connection)

**Architecture:**
| Layer | Component | Location | Purpose |
|-------|-----------|----------|---------|
| Schema | Cross-System Connection Registry | PDF Tools | Prevent invalid edges at write-time |
| Schema | Part Number Pattern Registry | PDF Tools | Validate format before storage |
| Agent | CrossDomainValidator | memOS | Validate claims at query-time |
| Agent | EntityGroundingAgent | memOS | Verify entities via PDF API lookup |
| Prompt | CROSS_DOMAIN_CONSTRAINTS | memOS | LLM-level guardrails in synthesis |

**FeatureConfig Parameters:**
```python
enable_cross_domain_validation: bool = False  # ENHANCED/RESEARCH/FULL
enable_entity_grounding: bool = False         # ENHANCED/RESEARCH/FULL
cross_domain_severity_threshold: str = "warning"  # critical/warning/info
```

**Preset Integration:**
| Preset | cross_domain_validation | entity_grounding | severity |
|--------|------------------------|------------------|----------|
| MINIMAL | false | false | - |
| BALANCED | false | false | - |
| ENHANCED | true | true | warning |
| RESEARCH | true | true | critical |
| FULL | true | true | critical |

**Files Created:**
- `agentic/cross_domain_validator.py`: CrossDomainValidator agent
- `agentic/entity_grounding.py`: EntityGroundingAgent with pattern matching
- `agentic/data/industrial_relationships.yaml`: 488-line validation corpus

**Files Modified:**
- `agentic/orchestrator_universal.py`: Feature flags, PHASE 9.5, pipeline integration
- `agentic/synthesizer.py`: CROSS_DOMAIN_CONSTRAINTS prompt block
- `agentic/self_reflection.py`: `_check_cross_domain_claims()` method

**Test Results (2026-01-04):**
- ENHANCED preset: ✅ Correctly rejected false cross-domain causation
- BALANCED preset: ✅ Synthesis prompt constraints effective
- Technical accuracy: ✅ SRVO-062/068 troubleshooting verified against FANUC docs

### Document Citations with Deduplication (2026-01-13)

Enhanced RAG context generation with proper document name citations from the PDF Tools knowledge graph.

**Problem Solved:**
- Previously showed raw paths: `["Manual", "Chapter", "Section"]`
- Now shows clean citations: `Fanuc R-30iA and R-30iB Manual > 12.10 - TROUBLESHOOTING (p. 276)`

**Key Components:**

| Component | Location | Purpose |
|-----------|----------|---------|
| `SourceDocument` | `core/document_graph_service.py` | Dataclass with `citation_name` property |
| `_get_cited_documents()` | `core/document_graph_service.py` | Deduplication by document_id |
| `get_context_for_rag()` | `core/document_graph_service.py` | Formatted context with citations |

**Example Output:**
```markdown
## Relevant Technical Documentation

### [1] Node 317f126b
**Source:** Fanuc R-30iA and R-30iB Manual > 12.10 - TROUBLESHOOTING (p. 276)
**Relevance:** 0.30
12. DATA TRANSFER BETWEEN ROBOTS OVER ETHERNET...

## Source Documents Referenced
- **Fanuc R-30iA and R-30iB Manual** (manual) - 5 citations
```

**Features:**
- **Clean document names**: Underscores replaced with spaces, "Manual" suffix added
- **Section info**: Shows section title from document hierarchy
- **Page numbers**: Displays page number when available from PDF metadata
- **Deduplication**: Groups multiple results from same document with citation count
- **Unified API format**: Handles `{success, data: {results: [...]}}` response structure

**Usage:**
```python
from core.document_graph_service import document_graph_service

# Get RAG context with document citations
context = await document_graph_service.get_context_for_rag(
    query="SRVO-063 encoder error",
    context_type="troubleshooting",
    max_tokens=2000
)

# Get search results with source documents
results = await document_graph_service.search_documentation("SRVO-063")
for r in results:
    if r.source_document:
        print(f"{r.source_document.citation_name} (p. {r.source_document.page_number})")
```

### Phase 50: Extended Diagram Integration (2026-01-19)

User-requested and auto-generated technical diagrams for industrial troubleshooting.

**Diagram Types Supported:**

| Type | Subtype Examples | Source | Status |
|------|------------------|--------|--------|
| **Circuit** | SERVO_DRIVE, POWER_DISTRIBUTION, ENCODER_INTERFACE | PDF Tools API | ✅ Working |
| **Harness** | ENCODER_17PIN, MOTOR_POWER, SAFETY_ESTOP | PDF Tools API | ✅ Working |
| **Flowchart** | SRVO-062, SRVO-063, MOTN-017 (33+ error codes) | PDF Tools API | ✅ Working |
| **Pinout** | COP1, COP2, TBOP13, ENCODER_17PIN | PDF Tools API | 🔲 Needs endpoint |

**Key Components:**

| Component | Location | Purpose |
|-----------|----------|---------|
| `DiagramIntent` | `agentic/models.py` | Intent detection result model |
| `DIAGRAM_INTENT_PATTERNS` | `agentic/analyzer.py` | 40+ regex patterns for diagram detection |
| `detect_diagram_intent()` | `agentic/analyzer.py` | Pattern-based intent extraction |
| `get_circuit_diagram()` | `core/document_graph_service.py` | Fetch SVG circuit from PDF Tools |
| `get_harness_diagram()` | `core/document_graph_service.py` | Fetch SVG harness from PDF Tools |
| `_detect_diagram_opportunity()` | `agentic/orchestrator_universal.py` | Auto-suggest diagrams from synthesis |

**Feature Flags (FeatureConfig):**
```python
enable_circuit_diagrams: bool = False    # ENHANCED+
enable_harness_diagrams: bool = False    # ENHANCED+
enable_pinout_diagrams: bool = False     # ENHANCED+
auto_generate_diagrams: bool = False     # RESEARCH+
```

**Usage:**
```python
from agentic.analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()
intent = analyzer.detect_diagram_intent("show me the servo drive circuit diagram")
# DiagramIntent(requested=True, diagram_type=CIRCUIT, diagram_subtype='SERVO_DRIVE', confidence=0.9)

# Fetch diagram
from core.document_graph_service import document_graph_service
diagram = await document_graph_service.get_circuit_diagram("SERVO_DRIVE", theme="dark")
# Returns: {"type": "circuit", "format": "svg", "content": "<svg...>", ...}
```

---

## Reference Documentation

Detailed documentation has been extracted to separate reference files to reduce context clutter:

| Document | Description | Lines |
|----------|-------------|-------|
| [server/docs/IMPLEMENTATION_PHASES.md](server/docs/IMPLEMENTATION_PHASES.md) | Detailed phase-by-phase implementation documentation | ~2200 |
| [server/docs/EMBEDDING_MODELS.md](server/docs/EMBEDDING_MODELS.md) | Embedding model research, mixed-precision systems, BGE-M3, HyDE, RAGAS | ~400 |
| [server/docs/AGENTIC_ARCHITECTURE.md](server/docs/AGENTIC_ARCHITECTURE.md) | Pipeline architecture, ReAct implementation, performance optimizations | ~400 |
| [server/agentic/AGENTIC_IMPROVEMENT_PLAN.md](server/agentic/AGENTIC_IMPROVEMENT_PLAN.md) | Current improvement roadmap based on 3 audits + 2 research studies | ~320 |
| [server/agentic/OBSERVABILITY_IMPROVEMENT_PLAN.md](server/agentic/OBSERVABILITY_IMPROVEMENT_PLAN.md) | Three-tier observability architecture plan | ~830 |
| [server/agentic/CONTEXT_FLOW_AUDIT.md](server/agentic/CONTEXT_FLOW_AUDIT.md) | Pipeline context flow audit findings | ~340 |

---

## Development Commands

```bash
# ============================================
# CRITICAL: Always activate venv first!
# ============================================
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate  # REQUIRED - uses local Python environment

# ============================================
# IMPORTANT: Apply Ollama optimizations first!
# ============================================
source /home/sparkone/sdd/Recovery_Bot/memOS/server/setup_ollama_optimization.sh
# Then restart Ollama if running: systemctl restart ollama

# Start the server (venv must be active)
python -m uvicorn main:app --reload --port 8001

# Or use convenience scripts
./start_server.sh                # Start server in background
./stop_server.sh                 # Stop server
./restart_server.sh              # Restart server
./status_server.sh               # Check server status
./logs_server.sh                 # Tail server logs

# Run tests
./test_system.sh                 # Full system test suite
./test_system.sh quick           # Quick tests (no LLM calls)
./test_system.sh hybrid          # Test BGE-M3 hybrid retrieval
./test_system.sh hyde            # Test HyDE query expansion
./test_system.sh ragas           # Test RAGAS evaluation
./test_system.sh api             # Test API endpoints
python test_quest_simple.py      # Test read operations (working)
python test_quest_assignment.py  # Test full workflow (has issues)
python test_vl_scraper.py        # Test VL screenshot scraper

# Model management
curl -X POST "http://localhost:8001/api/v1/models/refresh?force=true"  # Refresh model specs
curl -X POST "http://localhost:8001/api/v1/models/refresh?resynthesize_all=true"  # Re-synthesize descriptions
curl "http://localhost:8001/api/v1/models/specs?capability=vision"  # Get vision models
curl "http://localhost:8001/api/v1/models/gpu/status"  # Check GPU/VRAM status

# Agentic search testing
curl -X POST "http://localhost:8001/api/v1/search/agentic" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "max_iterations": 3}'

# Database management
python init_database.py          # Initialize schema
python create_sample_quests.py   # Populate sample quests

# Verify Ollama optimization settings
env | grep OLLAMA
```

## Technical Stack

- **Framework**: FastAPI with async support
- **Database**: PostgreSQL 15 with pgvector
- **ORM**: SQLAlchemy 2.0 with async sessions
- **Validation**: Pydantic v2
- **AI/Embeddings**: Ollama with qwen3:8b (synthesis), qwen3-embedding (embeddings)
- **Authentication**: JWT tokens with refresh
- **Logging**: Python logging with audit trail

## Architecture Notes

- Service-oriented architecture with clear separation of concerns
- Repository pattern for data access
- Dependency injection for database sessions
- HIPAA-compliant data handling throughout
- Event-driven updates for real-time features

For detailed architecture documentation, see:
- [server/docs/AGENTIC_ARCHITECTURE.md](server/docs/AGENTIC_ARCHITECTURE.md) - Pipeline design, ReAct implementation
- [server/docs/IMPLEMENTATION_PHASES.md](server/docs/IMPLEMENTATION_PHASES.md) - Phase-by-phase implementation details

---

## Unified Architecture Integration

memOS follows the Recovery Bot **Unified Architecture Recommendations** (see `/UNIFIED_ARCHITECTURE_RECOMMENDATIONS.md`):

### Response Format Compliance

All memOS endpoints return the unified response envelope:
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "timestamp": "2025-12-24T00:00:00Z",
    "request_id": "uuid",
    "version": "1.0.0"
  },
  "errors": []
}
```

### SSOT Data Ownership

| Domain | Owner |
|--------|-------|
| User Memories | memOS PostgreSQL |
| Quest Progress | memOS PostgreSQL |
| Search Context | memOS PostgreSQL |
| User Settings | memOS PostgreSQL |

### Cross-Service Communication

memOS publishes events for other services:
```python
# Event types
"memory.stored"      # New memory added
"quest.completed"    # Quest milestone reached
"search.completed"   # Agentic search finished
"context.injected"   # Context added to session
```

---

## Key API Endpoints

### Health & System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Basic health check |
| `/api/v1/system/health/aggregate` | GET | Aggregate all subsystem health |
| `/api/v1/models/specs` | GET | List available LLM models |
| `/api/v1/models/gpu/status` | GET | GPU/VRAM status |

### Agentic Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/chat-gateway` | POST | Main chat gateway with SSE streaming |
| `/api/v1/search/universal` | POST | Universal orchestrator (non-streaming) |
| `/api/v1/search/metrics` | GET | Performance metrics |
| `/api/v1/search/cache/stats` | GET | Cache statistics |

### Memory & Quest

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/search` | POST | Semantic memory search |
| `/api/v1/quests/available` | GET | Available quests |
| `/api/v1/quests/{id}/assign` | POST | Assign quest to user |

For complete API documentation, see the FastAPI auto-generated docs at `http://localhost:8001/docs`.

---

## Presets Summary

5 preset levels control feature activation:

| Preset | Features | Use Case |
|--------|----------|----------|
| **MINIMAL** | 8 | Fast, simple queries |
| **BALANCED** | 18 | Default for most queries |
| **ENHANCED** | 28 | Complex research |
| **RESEARCH** | 35 | Academic/thorough |
| **FULL** | 38+ | Maximum capability |

See `agentic/orchestrator_universal.py` → `PRESET_CONFIGS` for detailed feature flags.

---

## Mem0 Integration (2026-01-12)

memOS uses **Mem0** as a memory layer for automated fact extraction and cross-session continuity.

### Architecture

```
Conversation → Mem0 Extraction → memOS Storage (HIPAA-compliant)
                    ↓
Query → Mem0 Search → Context Augmentation → Orchestrator
```

### Key Files

| File | Purpose |
|------|---------|
| `agentic/mem0_config.py` | Gateway-routed configuration |
| `agentic/mem0_adapter.py` | Agentic pipeline integration |
| `core/memory_service.py` | Core memory (existing, uses Mem0) |
| `docs/MEM0_INTEGRATION_REPORT.md` | Full integration analysis |

### Usage

```python
from agentic.mem0_adapter import get_adapter_for_user

# Get adapter for user
adapter = get_adapter_for_user("user123")

# Process interaction (auto-extract memories)
await adapter.process_interaction(
    query="How do I fix SRVO-063?",
    response="SRVO-063 indicates servo disconnect...",
    context={"preset": "RESEARCH", "domain": "FANUC"}
)

# Get context for query augmentation
context = await adapter.get_context_for_query("What about SRVO-062?")
```

### Configuration

Mem0 routes through LLM Gateway for VRAM management:

```python
from agentic.mem0_config import get_mem0_config

config = get_mem0_config(
    use_gateway=True,  # Route through :8100
    collection_name="user_memories",
    llm_model="qwen3:8b",
    embedding_model="nomic-embed-text"
)
```

### What Mem0 Provides (vs memOS native)

| Feature | Mem0 | memOS Native |
|---------|------|--------------|
| Automated fact extraction | **Yes** | No |
| Cross-session continuity | **Yes** | Partial |
| Memory consolidation | **Yes** | No |
| Domain expert routing | No | **Yes** |
| Three-tier memory | No | **Yes** |
| VRAM management | No | **Yes** |

---

*Last Updated: 2026-01-13 | Added document citations with deduplication*
