# memOS - Agentic Search Server

Intelligent AI research platform with advanced agentic search capabilities, HIPAA-compliant memory management, and multi-phase orchestration.

## Features

- **Agentic Search**: Multi-agent pipeline with analyzer, planner, searcher, verifier, and synthesizer
- **Memory Tiers**: Three-tier memory architecture (Cold/Warm/Hot) for 80-94% TTFT reduction
- **KV Cache Management**: Intelligent caching with TTL pinning during tool operations
- **Circuit Breakers**: Production-hardened reliability patterns
- **LLM Configuration**: Dynamic model assignment with presets (speed/quality/balanced)
- **Observability**: Real-time SSE events, decision logs, and confidence tracking

## Quick Start

```bash
cd server

# Activate virtual environment (required)
source venv/bin/activate

# Start the server
./start_server.sh

# Or manually:
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Testing

```bash
# Quick search test with auto-timeout
./test_search.sh "What is FANUC SRVO-062?" balanced

# Available presets: minimal, balanced, enhanced, research, full
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/agentic` | POST | Multi-agent search |
| `/api/v1/search/stream` | POST | Streaming search with SSE |
| `/api/v1/config/llm-models` | GET | Current LLM configuration |
| `/api/v1/config/llm-models/presets/:name` | POST | Apply preset (speed/quality/balanced) |
| `/api/v1/system/health/aggregate` | GET | All service health statuses |
| `/api/v1/observability/recent` | GET | Recent search requests |

## Configuration

### LLM Models (`config/llm_models.yaml`)

Configure which models handle each pipeline stage:

```yaml
pipeline:
  analyzer: gemma3:4b      # Fast query analysis
  synthesizer: qwen3:8b    # Response synthesis
  thinking: deepseek-r1:14b # Extended reasoning
```

### Presets

| Preset | Typical Time | Best For |
|--------|-------------|----------|
| `speed` | 60-90s | Quick lookups |
| `balanced` | 135-165s | Production default |
| `quality` | 200-240s | Complex research |

Apply via API:
```bash
curl -X POST http://localhost:8001/api/v1/config/llm-models/presets/speed
```

## Architecture

```
Search Request
      │
      ▼
┌─────────────────┐
│   Orchestrator  │ (Universal preset-based)
├─────────────────┤
│  ┌───────────┐  │
│  │ Analyzer  │──┼──▶ Query classification
│  └─────┬─────┘  │
│        ▼        │
│  ┌───────────┐  │
│  │ Searcher  │──┼──▶ SearXNG + RAG
│  └─────┬─────┘  │
│        ▼        │
│  ┌───────────┐  │
│  │ Verifier  │──┼──▶ Fact checking
│  └─────┬─────┘  │
│        ▼        │
│  ┌───────────┐  │
│  │Synthesizer│──┼──▶ Response generation
│  └───────────┘  │
└─────────────────┘
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Configuration system details
- Prompt configuration
- Semantic query parser
- Pipeline stages
- Development guidelines

## Port

**8001** - memOS API Server

## Dependencies

- PostgreSQL (5432) - User data and memories
- Redis (6379) - Session cache
- Ollama (11434) - LLM inference
- SearXNG (8888) - Metasearch
- Gateway (8100) - VRAM-managed embeddings
