# MCP Node Editor - memOS Integration Reference

**System Location**: `/home/sparkone/sdd/MCP_Node_Editor`
**Status**: Production-ready reactive execution engine
**Version**: As of 2025-01-05

---

## Overview

The MCP Node Editor is a **production-grade, event-driven AI workflow orchestration system** that enables visual design and programmatic execution of complex pipelines. It supports 27 node types, cyclic workflows, real-time re-execution, and enterprise-level deadlock prevention.

### Primary Capabilities

1. **AI Pipeline Orchestration**: Chain LLM calls with data transformations
2. **Code Generation & Execution**: Generate, extract, and run code in sandboxed environments
3. **Iterative Debugging**: Cyclic workflows for refinement until convergence
4. **Multi-Agent Coordination**: Orchestrate multiple AI agents with state management
5. **Data Processing**: Transform, chunk, compress, and route data
6. **Database Integration**: MariaDB CRUD operations with NL-to-SQL
7. **RAG Pipelines**: Retrieval-augmented generation with vector memory

---

## API Integration

### Base URL
```
http://localhost:7777
```

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/execute` | POST | Execute pipeline (async, returns pipeline_id) |
| `/api/execution/{id}` | GET | Real-time execution status |
| `/api/result/{id}` | GET | Fetch completed results |
| `/api/cancel/{id}` | POST | Cancel running pipeline |
| `/api/status` | GET | System health check |
| `/api/models` | GET | Available LLM models |
| `/api/logs` | GET | Execution logs |
| `/api/logs/stream` | GET | Real-time log stream |

### Pipeline Execution Example

```python
import aiohttp
import asyncio

async def execute_pipeline(pipeline_json: dict) -> dict:
    """Execute a pipeline and return results."""
    async with aiohttp.ClientSession() as session:
        # Submit pipeline
        async with session.post(
            'http://localhost:7777/api/execute',
            json=pipeline_json
        ) as resp:
            data = await resp.json()
            pipeline_id = data['pipeline_id']

        # Poll for completion
        while True:
            async with session.get(
                f'http://localhost:7777/api/result/{pipeline_id}'
            ) as resp:
                result = await resp.json()
                if result.get('status') in ['completed', 'failed']:
                    return result
            await asyncio.sleep(0.5)
```

---

## Node Types Reference (27 Total)

### Basic Nodes (8)
| Node | Purpose | Key Properties |
|------|---------|----------------|
| `input` | Data ingestion | source_type: text/file/api/previous |
| `output` | Result capture | format: text/json/markdown |
| `prompt` | Template processing | template with {{variable}} substitution |
| `model` | LLM integration | model_name, temperature, max_tokens |
| `refiner` | Iterative improvement | refinement_rounds, criteria |
| `combiner` | Multi-input merge | operation: join/conditional/template |
| `processor` | Text manipulation | operation: extract_code/format/transform |
| `image` | Image handling | compression, format conversion |

### Advanced Nodes (15)
| Node | Purpose | Key Properties |
|------|---------|----------------|
| `memory` | Persistent state | memory_type: vector/kv/session |
| `state_manager` | State machines | states, transitions, conditions |
| `agent_orchestrator` | Multi-agent coordination | agents, routing_strategy |
| `cycle_handler` | Iterative refinement | max_iterations, convergence_threshold |
| `tree_explorer` | Tree-of-thought | branching_factor, evaluation_fn |
| `chunker` | Document splitting | chunk_size, overlap |
| `context_compressor` | Token optimization | target_tokens, preservation_priority |
| `graph_of_thoughts` | Advanced reasoning | thought_graph structure |
| `mariadb` | Database CRUD | operation, query/data |
| `query_processor` | NL-to-SQL | schema_context |
| `rag_pipeline` | Retrieval-augmented gen | retriever_config, k_documents |
| `web_search` | Web integration | search_engine, max_results |
| `code_executor` | Sandboxed code run | language, timeout, sandbox_mode |

### Control Flow Nodes (4)
| Node | Purpose | Key Properties |
|------|---------|----------------|
| `conditional` | Dynamic branching | condition_type, branches |
| `loop` | Iterative execution | loop_type, iterations/condition |
| `gate` | Flow synchronization | gate_type: wait_all/wait_any |
| `merge` | Path convergence | merge_strategy: first/all/custom |

---

## Workflow Patterns for Agentic Use

### Pattern 1: Code Generation + Validation
```
Use Case: Generate code, extract it, execute, validate output
Flow: Input → Prompt → Model → Processor(extract_code) → Code Executor → Output
```

### Pattern 2: Iterative Debugging
```
Use Case: Generate code, test it, fix errors until working
Flow: Input → Prompt → Model ↔ Cycle Handler ↔ Code Executor → Output
```

### Pattern 3: Multi-Agent Reasoning
```
Use Case: Coordinate multiple specialized agents
Flow: Input → Agent Orchestrator → [Agents...] → Merge → Output
```

### Pattern 4: RAG-Enhanced Response
```
Use Case: Retrieve context before generating
Flow: Query → RAG Pipeline → Memory → Model → Output
```

### Pattern 5: Data Processing Pipeline
```
Use Case: Transform data through multiple stages
Flow: Input → Chunker → Processor → Combiner → Output
```

---

## Pipeline JSON Structure

```json
{
  "nodes": [
    {
      "id": 0,
      "type": "input",
      "title": "User Query",
      "x": 100, "y": 100,
      "properties": {
        "source_type": "text",
        "text": "Initial input"
      },
      "inputs": [],
      "outputs": [{"name": "output", "type": "text"}]
    },
    {
      "id": 1,
      "type": "model",
      "title": "LLM",
      "x": 300, "y": 100,
      "properties": {
        "model_name": "llama3.2",
        "temperature": 0.7,
        "max_tokens": 2048
      },
      "inputs": [{"name": "prompt", "type": "text"}],
      "outputs": [{"name": "response", "type": "text"}]
    }
  ],
  "connections": [
    {
      "id": 0,
      "source": {"node_id": 0, "port_index": 0, "port_type": "output"},
      "target": {"node_id": 1, "port_index": 0}
    }
  ],
  "settings": {
    "max_iterations_per_node": 100,
    "enable_re_execution": true,
    "execution_timeout": 300
  }
}
```

---

## Execution Response Format

### Status Response
```json
{
  "pipeline_id": "abc123",
  "status": "running|completed|failed|cancelled",
  "progress": {
    "total_nodes": 5,
    "completed_nodes": 3,
    "current_node": "model_1"
  },
  "execution_time": 2.5,
  "node_iterations": {
    "input_0": 1,
    "model_1": 2
  }
}
```

### Result Response
```json
{
  "pipeline_id": "abc123",
  "status": "completed",
  "outputs": {
    "output_4": {
      "value": "Final result text",
      "type": "text",
      "iteration": 1
    }
  },
  "execution_history": [...],
  "total_execution_time": 3.2
}
```

---

## System Management

### Start System
```bash
cd /home/sparkone/sdd/MCP_Node_Editor
./start-app.sh
```

### Stop System
```bash
./stop-app.sh
```

### Check Status
```bash
./status-app.sh
```

### View Logs
```bash
tail -f logs/pipeline-execution.log
# or via API
curl http://localhost:7777/api/logs
```

---

## Configuration

**Config File**: `mcp-servers/pipeline-config.json`

### Key Settings
```json
{
  "reactive_execution": {
    "max_iterations_per_node": 100,
    "enable_re_execution": true,
    "execution_timeout": 300
  },
  "circuit_breakers": {
    "enabled": true,
    "error_threshold": 3,
    "cooldown_period": 30.0
  },
  "rate_limiting": {
    "enabled": true,
    "per_node_overrides": {
      "model": {"capacity": 5, "refill_rate": 1.0}
    }
  },
  "node_timeouts": {
    "default_timeout": 30,
    "model_api_timeout": 300
  }
}
```

---

## Current System Status

### Fully Operational
- Linear workflows (sub-second execution)
- All 27 node types
- Deadlock prevention (multi-layered)
- Async execution (non-blocking)
- Code executor with sandboxing
- Database integration (MariaDB)
- Event-driven architecture (1000+ events/sec)

### In Development
- Cyclic workflow bootstrap injection
- Quasi-active node state transitions
- Full cycle convergence detection

---

## memOS Integration Recommendations

### As Workflow Executor
```python
# memOS agent can execute pre-designed pipelines
result = await execute_pipeline(pipeline_json)
```

### As Code Validator
```python
# Generate code, then use code_executor node to validate
code_test_pipeline = create_code_test_pipeline(generated_code)
result = await execute_pipeline(code_test_pipeline)
```

### As Data Processor
```python
# Transform data through connected nodes
transform_pipeline = create_transform_pipeline(data, operations)
result = await execute_pipeline(transform_pipeline)
```

### As Multi-Step Reasoning Engine
```python
# Chain multiple AI calls with intermediate processing
reasoning_pipeline = create_reasoning_pipeline(query)
result = await execute_pipeline(reasoning_pipeline)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `complete_reactive_executor.py` | Main execution engine (3,017 lines) |
| `reactive_node_implementations.py` | All 27 node handlers (1,434 lines) |
| `pipeline_launcher.py` | HTTP API server (1,114 lines) |
| `node_execution_scheduler.py` | Scheduling system (1,336 lines) |
| `execution_event_bus.py` | Event-driven communication (608 lines) |
| `pipeline-config.json` | System configuration |

---

## Performance Characteristics

- **Event throughput**: 1000+ events/second
- **Node concurrency**: 10-50 simultaneous
- **Event routing latency**: <1ms
- **Linear pipeline completion**: ~1 second
- **Memory per node buffer**: 10MB default
- **Request size limit**: 10MB

---

## Security Features

- **Code Executor Sandboxing**: Process isolation with configurable permissions
- **API Validation**: Schema validation before processing
- **Request Size Limits**: Enforced before parsing
- **Rate Limiting**: Prevents resource exhaustion
- **Circuit Breakers**: Automatic error loop prevention

---

## SSOT Integration Guidelines

**Reference Document**: See `SSOT_CROSS_PROJECT_ARCHITECTURE.md` for complete cross-project architecture.

### MCP Node Editor as Workflow SSOT

The MCP Node Editor is the **Single Source of Truth** for:
- Pipeline definition and execution
- Workflow orchestration logic
- Node type implementations
- Execution state and history

### Data Flow Principles

```
memOS (Intelligence SSOT) ←→ MCP Node Editor (Workflow SSOT)
         ↓                              ↓
    User Context                 Pipeline Execution
    Memory Storage               Node Orchestration
    Quest State                  Event Processing
```

### Unified Response Format

MCP Node Editor responses should be wrapped in the unified envelope when consumed by memOS:

```python
def wrap_mcp_response(mcp_result: dict) -> dict:
    """Wrap MCP Node Editor response in unified format."""
    return {
        "success": mcp_result.get("status") == "completed",
        "data": mcp_result.get("outputs", {}),
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": mcp_result.get("pipeline_id"),
            "version": "1.0.0",
            "source_service": "mcp_node_editor",
            "execution_time": mcp_result.get("total_execution_time")
        },
        "errors": [] if mcp_result.get("status") == "completed" else [
            {
                "code": "ERR_4001",
                "message": mcp_result.get("error", "Pipeline execution failed"),
                "details": mcp_result.get("execution_history", [])
            }
        ]
    }
```

### Event Integration

Subscribe to MCP events for real-time updates:

```python
# memOS can listen for MCP events
async def subscribe_to_mcp_events():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:7777/api/logs/stream') as resp:
            async for line in resp.content:
                event = json.loads(line)
                await handle_mcp_event(event)

async def handle_mcp_event(event: dict):
    if event.get("type") == "pipeline.completed":
        await cache_search_results(event)
    elif event.get("type") == "node.failed":
        await log_node_failure(event)
```

### Error Code Mapping

| MCP Error | Unified Code | Description |
|-----------|-------------|-------------|
| Pipeline timeout | ERR_4002 | Node timeout exceeded |
| Circuit breaker open | ERR_4003 | Too many failures |
| Validation failed | ERR_1001 | Input validation failed |
| Rate limited | ERR_1004 | Too many requests |

---

*Generated: 2025-12-24*
*For memOS agentic workflow integration*
*See also: SSOT_CROSS_PROJECT_ARCHITECTURE.md*
