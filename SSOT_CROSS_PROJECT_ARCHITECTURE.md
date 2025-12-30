# Single Source of Truth (SSOT) Cross-Project Architecture

> **Updated**: 2025-12-30 | **Parent**: [Root CLAUDE.md](../CLAUDE.md) | **Version**: 1.0.0

**Scope**: MCP_Node_Editor + Recovery_Bot + memOS Integration

---

## Executive Summary

This document establishes the **Single Source of Truth (SSOT)** architecture governing data ownership, API contracts, and cross-service communication between the following projects:

| Project | Location | Port | Role |
|---------|----------|------|------|
| **MCP Node Editor** | `/home/sparkone/sdd/MCP_Node_Editor` | 7777 | Workflow Orchestration Engine |
| **Recovery_Bot** | `/home/sparkone/sdd/Recovery_Bot` | 8443 (proxy) | Community Services API + Admin |
| **memOS Server** | `/home/sparkone/sdd/Recovery_Bot/memOS` | 8001 | Memory/Quest/Agentic Intelligence Hub |

---

## SSOT Data Ownership Matrix

Each data domain has **ONE authoritative source**. All other systems consume via APIs.

### Primary Data Ownership

| Data Domain | Authoritative Source | Storage | Consumers | Sync Pattern |
|-------------|---------------------|---------|-----------|--------------|
| **User Authentication** | memOS | PostgreSQL | Android, PHP, MCP | JWT tokens |
| **User Profiles** | memOS | PostgreSQL | Android, PHP Admin | REST API |
| **User Memories** | memOS | PostgreSQL + pgvector | Android, RAG | Semantic search |
| **Quest Progress** | memOS | PostgreSQL | Android | Event-driven |
| **Service Directory** | PHP admin_api | SQLite | Android, RAG, memOS | Polling/Webhook |
| **Scraped Raw Data** | PHP scrapers | SQLite | Admin, memOS | Pipeline |
| **Pipeline Workflows** | MCP Node Editor | JSON files | memOS agentic search | File-based |
| **Pipeline Execution** | MCP Node Editor | In-memory + logs | memOS | REST API |
| **Chat Context** | Ollama + memOS | Ephemeral + PostgreSQL | Android | Session-based |

### SSOT Decision Rules

1. **Never duplicate business logic** - If logic exists in one service, others call its API
2. **Data flows downstream** - SSOT pushes; consumers pull or subscribe
3. **Schema changes require coordination** - All consumers must be notified
4. **Caching is allowed** - But cache invalidation follows SSOT updates

---

## Unified API Response Format

**ALL APIs across ALL services MUST return this envelope:**

### Success Response
```json
{
  "success": true,
  "data": {
    // Response payload (varies by endpoint)
  },
  "meta": {
    "timestamp": "2025-12-24T00:00:00Z",
    "request_id": "uuid-v4",
    "version": "1.0.0",
    "cache_ttl": 300,
    "source_service": "memOS|mcp_node_editor|recovery_bot"
  },
  "errors": []
}
```

### Error Response
```json
{
  "success": false,
  "data": null,
  "meta": {
    "timestamp": "2025-12-24T00:00:00Z",
    "request_id": "uuid-v4"
  },
  "errors": [
    {
      "code": "ERR_1001",
      "message": "Human-readable error message",
      "field": "optional_field_name",
      "details": {}
    }
  ]
}
```

---

## Unified Error Code Registry

Shared error codes ensure consistent error handling across all services:

| Code | Category | Description | Service Owner |
|------|----------|-------------|---------------|
| ERR_1001 | Validation | Input validation failed | All |
| ERR_1002 | Auth | Authentication failed | memOS |
| ERR_1003 | Auth | Invalid or expired token | memOS |
| ERR_1004 | Rate Limit | Too many requests | All |
| ERR_1005 | Not Found | Resource not found | All |
| ERR_2001 | Quest | Quest not available | memOS |
| ERR_2002 | Quest | Prerequisites not met | memOS |
| ERR_3001 | Privacy | Consent required | memOS |
| ERR_3002 | Privacy | HIPAA violation prevented | memOS |
| ERR_4001 | Pipeline | Pipeline execution failed | MCP Node Editor |
| ERR_4002 | Pipeline | Node timeout exceeded | MCP Node Editor |
| ERR_4003 | Pipeline | Circuit breaker open | MCP Node Editor |
| ERR_5001 | Scraper | Source unreachable | Recovery_Bot |
| ERR_5002 | Scraper | Rate limit by source | Recovery_Bot |

---

## Cross-Service Communication Patterns

### 1. Synchronous REST (Request-Response)

**Use for**: User-facing operations, immediate feedback needed

```
Android → memOS/api/v1/quests → PostgreSQL
Android → memOS/api/v1/search/agentic → MCP Node Editor → Results
```

### 2. Asynchronous Webhooks (Event-Driven)

**Use for**: Long-running operations, loose coupling

```python
# memOS publishes events
event_bus.publish("quest.completed", {
    "user_id": user_id,
    "quest_id": quest_id,
    "timestamp": datetime.now().isoformat()
})

# MCP Node Editor can trigger workflows on events
event_bus.subscribe("service.updated", trigger_rag_reindex)
```

### 3. Pipeline Execution (Workflow Orchestration)

**Use for**: Multi-step AI reasoning, agentic search

```python
# memOS calls MCP Node Editor for workflow execution
async def execute_agentic_search(query: str) -> dict:
    pipeline = load_pipeline("agentic_search.json")
    pipeline["nodes"][0]["properties"]["text"] = query

    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            "http://localhost:7777/api/execute",
            json=pipeline
        )
        data = await resp.json()
        return await poll_for_result(data["pipeline_id"])
```

---

## MCP Node Editor as Workflow SSOT

The MCP Node Editor is the **Single Source of Truth** for workflow orchestration.

### Capabilities Exposed to memOS

| Capability | Endpoint | Use Case |
|------------|----------|----------|
| Pipeline Execution | `POST /api/execute` | Run agentic search workflows |
| Status Polling | `GET /api/execution/{id}` | Track long-running pipelines |
| Result Retrieval | `GET /api/result/{id}` | Get completed outputs |
| Model Listing | `GET /api/models` | Discover available LLMs |
| Health Check | `GET /api/status` | Monitor availability |

### Node Types for Agentic Workflows

| Node | Purpose in memOS Context |
|------|--------------------------|
| `input` | Inject user query + context |
| `prompt` | Template system prompts |
| `model` | LLM inference (llama3.2) |
| `web_search` | Execute web searches |
| `processor` | Extract/transform text |
| `memory` | Store/retrieve from memOS |
| `cycle_handler` | Iterative refinement |
| `agent_orchestrator` | Multi-agent coordination |
| `conditional` | Route based on content |
| `output` | Capture final results |

### Example Agentic Search Pipeline

```json
{
  "nodes": [
    {"id": 0, "type": "input", "title": "User Query"},
    {"id": 1, "type": "prompt", "title": "Planner", "properties": {
      "template": "Decompose this query into search steps: {{input}}"
    }},
    {"id": 2, "type": "model", "title": "LLM Planner"},
    {"id": 3, "type": "processor", "title": "Extract Queries"},
    {"id": 4, "type": "web_search", "title": "Execute Searches"},
    {"id": 5, "type": "prompt", "title": "Synthesizer"},
    {"id": 6, "type": "model", "title": "LLM Synthesizer"},
    {"id": 7, "type": "output", "title": "Final Result"}
  ],
  "connections": [
    {"source": {"node_id": 0}, "target": {"node_id": 1}},
    {"source": {"node_id": 1}, "target": {"node_id": 2}},
    {"source": {"node_id": 2}, "target": {"node_id": 3}},
    {"source": {"node_id": 3}, "target": {"node_id": 4}},
    {"source": {"node_id": 4}, "target": {"node_id": 5}},
    {"source": {"node_id": 5}, "target": {"node_id": 6}},
    {"source": {"node_id": 6}, "target": {"node_id": 7}}
  ]
}
```

---

## memOS as Intelligence SSOT

memOS is the **Single Source of Truth** for:
- User context and memory
- Quest/gamification state
- Search result caching
- Privacy/consent management

### API Surface for Integration

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/memory/search` | POST | Semantic memory search |
| `/api/v1/memory/store` | POST | Store new memory |
| `/api/v1/quests/available` | GET | Get user's available quests |
| `/api/v1/quests/{id}/assign` | POST | Assign quest to user |
| `/api/v1/search/agentic` | POST | Execute agentic search |
| `/api/v1/context/inject` | POST | Inject context for session |

---

## Recovery_Bot as Service Directory SSOT

Recovery_Bot is the **Single Source of Truth** for:
- Community service listings
- Scraping schedules and sources
- Admin interface state

### Data Flow to Other Services

```
Scrapers → SQLite → admin_api.php → memOS (for RAG indexing)
                                   → Android (for display)
```

---

## Contract-First Development

### OpenAPI Specification Files (Recommended Structure)

```
/schemas/
├── openapi/
│   ├── mcp-node-editor-api.yaml    # Pipeline orchestration
│   ├── memos-api.yaml              # Memory/Quest/Search
│   ├── recovery-bot-admin-api.yaml # Service directory
│   └── unified-error-codes.yaml    # Shared error definitions
└── shared/
    ├── response-envelope.schema.json
    ├── user.schema.json
    ├── quest.schema.json
    └── pipeline.schema.json
```

### Benefits
- Generate client code for Android (Kotlin)
- Validate responses in all services
- Document APIs automatically
- Detect breaking changes early

---

## Backward Compatibility Rules

Based on [Microsoft API Design Guidelines](https://learn.microsoft.com/en-us/azure/architecture/microservices/design/api-design):

### ALLOWED (Non-Breaking)
- Adding new optional fields to responses
- Adding new endpoints
- Adding new optional query parameters
- Relaxing validation (accepting more inputs)

### FORBIDDEN (Breaking)
- Removing or renaming fields
- Changing field types
- Making optional fields required
- Changing endpoint paths
- Changing error code meanings

### Migration Strategy: Strangler Fig Pattern

For migrating functionality between services:

1. **Identify** bounded context to migrate
2. **Create facade** in target service that calls source
3. **Implement** new logic behind facade
4. **Redirect** traffic incrementally (feature flags)
5. **Remove** legacy code when fully migrated

---

## Event-Driven Architecture

### Event Types (Cross-Service)

```python
# memOS Events
"memory.stored"           # New memory added
"quest.completed"         # Quest milestone reached
"search.completed"        # Agentic search finished
"context.injected"        # Context added to session

# MCP Node Editor Events
"pipeline.started"        # Pipeline execution began
"pipeline.completed"      # Pipeline finished successfully
"pipeline.failed"         # Pipeline execution failed
"node.executed"           # Individual node completed

# Recovery_Bot Events
"service.created"         # New service added
"service.updated"         # Service info changed
"scrape.completed"        # Scraping job finished
```

### Event Bus Integration

```python
# memOS subscribes to MCP events
async def on_pipeline_completed(event: dict):
    if event["pipeline_type"] == "agentic_search":
        await cache_search_results(event["results"])

# MCP subscribes to memOS events
async def on_memory_stored(event: dict):
    if event["memory_type"] == "procedural":
        await update_rag_index(event["content"])
```

---

## Caching Strategy

| Data Type | TTL | Invalidation Trigger |
|-----------|-----|---------------------|
| Quest list | 5 min | quest.updated event |
| User stats | 1 min | quest.completed event |
| Service directory | 1 hour | service.updated event |
| Search results | 15 min | Manual or memory.stored |
| Pipeline results | Session | Pipeline re-execution |

---

## Security & Compliance

### Authentication Flow
```
Android → memOS (JWT) → MCP Node Editor (Internal, trusted)
                     → Recovery_Bot (JWT forwarded)
```

### HIPAA Considerations
- **memOS**: Stores PHI (user memories) - encrypted at rest
- **MCP Node Editor**: No PHI storage - transient processing only
- **Recovery_Bot**: Service directory only - no PHI

### Audit Logging
All services log to their respective `logs/` directories with:
- Request ID correlation
- User ID (if authenticated)
- Timestamp
- Action performed
- Success/failure status

---

## Monitoring & Health Checks

### Health Endpoints

| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| MCP Node Editor | `GET /api/status` | `{"launcher": "running", "mode": "REACTIVE"}` |
| memOS | `GET /api/v1/memory/health` | `{"status": "healthy"}` |
| Recovery_Bot | `GET /admin_api.php?action=health` | `{"status": "ok"}` |

### Startup Order
1. **PostgreSQL** (memOS database)
2. **Ollama** (LLM inference)
3. **MCP Node Editor** (port 7777)
4. **memOS Server** (port 8001)
5. **nginx proxy** (routes external traffic)

---

## Development Workflow

### Adding a New Cross-Service Feature

1. **Define schema** in `/schemas/shared/`
2. **Update OpenAPI** specs in all affected services
3. **Implement SSOT** in authoritative service first
4. **Add consumer code** in dependent services
5. **Write integration tests** covering cross-service flow
6. **Update this document** with new data ownership

### Code Review Checklist
- [ ] Response format matches unified envelope
- [ ] Error codes use registry
- [ ] No business logic duplication
- [ ] Backward compatible changes only
- [ ] Event published for state changes
- [ ] Audit logging included

---

## References

- [Red Hat: Single Source of Truth Architecture](https://www.redhat.com/en/blog/single-source-truth-architecture)
- [Microsoft: API Design for Microservices](https://learn.microsoft.com/en-us/azure/architecture/microservices/design/api-design)
- [MuleSoft: What is SSOT](https://www.mulesoft.com/resources/esb/what-is-single-source-of-truth-ssot)
- [Medium: SSOT Challenges in Microservices](https://medium.com/@v4sooraj/navigating-the-challenges-of-single-source-of-truth-ssot-in-microservices-architecture-8538afe931a3)
- [Atlassian: Building a True SSOT](https://www.atlassian.com/work-management/knowledge-sharing/documentation/building-a-single-source-of-truth-ssot-for-your-team)
- [InfoQ: Contract-Driven Development](https://www.infoq.com/articles/contract-driven-development/)

---

*This document is the SSOT for cross-project architecture decisions.*
