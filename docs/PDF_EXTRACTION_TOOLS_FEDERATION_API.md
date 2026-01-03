# PDF Extraction Tools - Multi-Domain Federation API

**Integration Report for memOS**
**Date:** 2026-01-03
**Version:** 1.0.0
**API Base:** `http://localhost:8002/api/v1`

---

## Executive Summary

PDF Extraction Tools now provides a **Multi-Domain Federation API** enabling memOS to query across 4 industrial knowledge domains through a unified interface. The API supports cross-domain search with RRF fusion, MCP-compatible tool definitions for LLM consumption, and domain-specific entity type filtering.

**Total Knowledge Base:**
- **286,400+ nodes** across all domains
- **4 specialized domains**: FANUC, IMM, Industrial Automation, OEM IMM
- **8 MCP-compatible tools** for AI agent consumption

---

## API Endpoints

### Domain Registry

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/domains/registry` | GET | List all registered domains with statistics |
| `/api/v1/domains/{domain}` | GET | Get specific domain information |
| `/api/v1/domains/search` | POST | Cross-domain federated search |
| `/api/v1/domains/{domain}/entities` | GET | List entities by type with pagination |

### MCP Tool Definitions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tools/registry` | GET | All 8 MCP-compatible tool definitions |
| `/api/v1/tools/tool/{name}` | GET | Specific tool definition |
| `/api/v1/tools/by-category/{cat}` | GET | Tools by category |
| `/api/v1/tools/mcp-manifest` | GET | MCP server manifest |
| `/api/v1/tools/openapi-tools` | GET | OpenAI function calling format |
| `/api/v1/tools/anthropic-tools` | GET | Anthropic Claude format |

---

## Available Domains

### 1. FANUC Robotics (`fanuc`)
- **Nodes:** 268,886
- **Description:** Robot error codes, troubleshooting, KAREL programming, servo motors
- **Entity Types:** `error_code`, `component`, `procedure`, `symptom`, `remedy`, `root_cause`, `part_number`, `alarm_type`
- **Graph File:** `data/graph_full_corpus.pkl`
- **Node Format:** Dataclass objects (EntityNode, DocumentNode)

### 2. Injection Molding (`imm`)
- **Nodes:** 2,993
- **Description:** Injection molding defects, processes, troubleshooting
- **Entity Types:** `defect`, `process_param`, `material`, `machine_component`, `procedure`
- **Graph File:** `data/imm_graph_v2.pkl`
- **Node Format:** Dictionary

### 3. Industrial Automation (`industrial_automation`)
- **Nodes:** 4,965
- **Description:** PLCs, sensors, control systems, industrial protocols
- **Entity Types:** `plc`, `sensor`, `protocol`, `error_code`, `procedure`, `component`
- **Graph File:** `data/industrial_automation_graph.pkl`
- **Node Format:** Dictionary

### 4. OEM IMM (`oem_imm`)
- **Nodes:** 9,556
- **Description:** Polymers, press OEMs, Allen Bradley, RJG sensors
- **Entity Types:** `polymer`, `press_oem`, `plc_controllogix`, `sensor_rjg`, `process_setting`
- **Graph File:** `data/oem_imm_graph.pkl`
- **Node Format:** Dictionary

---

## Cross-Domain Search

### Request Format

```json
POST /api/v1/domains/search
{
    "query": "servo motor error",
    "domains": ["fanuc", "industrial_automation"],
    "top_k": 10,
    "search_mode": "hybrid",
    "entity_types": ["error_code", "component"],
    "include_paths": false
}
```

### Response Format

```json
{
    "success": true,
    "data": {
        "query": "servo motor error",
        "total_results": 10,
        "results": [
            {
                "node_id": "symptom_4932a787",
                "domain": "fanuc",
                "node_type": "symptom",
                "label": "An error is found in the motor current detection data",
                "score": 0.016,
                "content_preview": "...",
                "metadata": {...}
            }
        ],
        "domains_searched": ["fanuc", "industrial_automation"],
        "search_mode": "hybrid",
        "execution_time_ms": 45.2
    },
    "meta": {
        "timestamp": "2026-01-03T12:00:00Z",
        "request_id": "uuid",
        "version": "1.0.0"
    },
    "errors": []
}
```

### Search Modes

| Mode | Description |
|------|-------------|
| `keyword` | BM25-based keyword matching |
| `semantic` | Embedding similarity search |
| `hybrid` | RRF fusion of keyword + semantic (default) |

---

## MCP Tool Definitions

### 1. `search_error_codes`
Search for FANUC robot error codes by keyword or semantic query.

**Parameters:**
- `query` (required): Natural language description of error/symptom
- `category_filter`: Optional category (SRVO, MOTN, SYST, etc.)
- `top_k`: Number of results (1-50, default 10)

### 2. `get_troubleshooting_path`
Find diagnostic paths from symptom to remedy using HSEA guidance.

**Parameters:**
- `symptom` (required): Description of observed problem
- `error_code`: Optional specific error code
- `max_hops`: Path depth (1-6, default 4)
- `beam_width`: Candidate paths (default 10)

### 3. `find_related_entities`
Explore relationships in the knowledge graph.

**Parameters:**
- `node_id` (required): Starting node ID
- `relationship_types`: Filter by edge types
- `depth`: Traversal depth (1-3)
- `direction`: outgoing, incoming, or both

### 4. `cross_domain_search`
Search across multiple industrial knowledge domains.

**Parameters:**
- `query` (required): Search query
- `domains`: List of domains to search
- `entity_types`: Filter by entity types
- `top_k`: Number of results

### 5. `get_entity_details`
Get full details about a specific entity.

**Parameters:**
- `domain` (required): Domain containing entity
- `entity_id` (required): Entity ID or canonical form
- `include_relationships`: Include connected entities

### 6. `list_entity_types`
Discover available entity types per domain.

**Parameters:**
- `domain`: Specific domain (optional)

### 7. `get_injection_molding_defect`
Get IMM defect information with causes and remedies.

**Parameters:**
- `defect_name` (required): Name of defect
- `include_process_params`: Include process parameters

### 8. `get_polymer_properties`
Get polymer material properties and processing guidelines.

**Parameters:**
- `polymer_name` (required): Polymer name or grade

---

## memOS Integration Patterns

### Pattern 1: Tool Discovery

```python
import httpx

async def discover_tools():
    async with httpx.AsyncClient() as client:
        # Get MCP manifest for Claude Code integration
        resp = await client.get("http://localhost:8002/api/v1/tools/mcp-manifest")
        manifest = resp.json()
        return manifest["tools"]
```

### Pattern 2: Federated Troubleshooting

```python
async def troubleshoot_cross_domain(symptom: str):
    async with httpx.AsyncClient() as client:
        # Search across all relevant domains
        resp = await client.post(
            "http://localhost:8002/api/v1/domains/search",
            json={
                "query": symptom,
                "domains": ["fanuc", "industrial_automation"],
                "entity_types": ["error_code", "remedy"],
                "top_k": 10
            }
        )
        results = resp.json()["data"]["results"]

        # Get detailed paths for top result
        if results:
            path_resp = await client.post(
                "http://localhost:8002/api/v1/traverse",
                json={
                    "query": symptom,
                    "target_types": ["remedy"],
                    "max_hops": 4
                }
            )
            return path_resp.json()["data"]["paths"]
```

### Pattern 3: Domain-Aware Routing

```python
def route_query_to_domain(query: str) -> list:
    """Route queries to appropriate domains based on keywords."""
    domains = []
    query_lower = query.lower()

    if any(kw in query_lower for kw in ["servo", "robot", "srvo", "motn", "fanuc"]):
        domains.append("fanuc")
    if any(kw in query_lower for kw in ["injection", "mold", "plastic", "defect"]):
        domains.append("imm")
    if any(kw in query_lower for kw in ["plc", "allen bradley", "controllogix", "sensor"]):
        domains.append("industrial_automation")
    if any(kw in query_lower for kw in ["polymer", "ultramid", "lexan", "rjg"]):
        domains.append("oem_imm")

    return domains or ["fanuc", "imm", "industrial_automation", "oem_imm"]
```

---

## Rate Limiting

The API implements tiered rate limiting:

| Tier | Requests/Minute | Description |
|------|-----------------|-------------|
| Internal | Unlimited | memOS (localhost) |
| Enterprise | 1000 | API key required |
| Standard | 100 | Authenticated |
| Free | 10 | Unauthenticated |

memOS requests from localhost are treated as internal and bypass rate limiting.

---

## Response Format

All responses follow the Recovery_Bot unified envelope format:

```json
{
    "success": true|false,
    "data": { ... },
    "meta": {
        "timestamp": "ISO8601",
        "request_id": "uuid",
        "version": "1.0.0",
        "path": "/api/v1/..."
    },
    "errors": [
        {
            "code": "ERR_XXXX",
            "message": "...",
            "field": "...",
            "details": {}
        }
    ]
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| ERR_1001 | Input validation failed |
| ERR_2001 | Entity not found |
| ERR_2002 | Graph not loaded |
| ERR_3001 | Processing failed |
| ERR_4001 | Ollama connection failed |
| ERR_5001 | Internal server error |

---

## Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "version": "1.0.0",
        "graph_loaded": true,
        "num_documents": 268886
    }
}
```

---

## Quick Start

1. **Start PDF Extraction Tools API:**
   ```bash
   cd /home/sparkone/sdd/PDF_Extraction_Tools
   source venv/bin/activate
   python -m uvicorn pdf_extractor.api.main:app --host 0.0.0.0 --port 8002
   ```

2. **Verify domains are loaded:**
   ```bash
   curl "http://localhost:8002/api/v1/domains/registry?load_all=true"
   ```

3. **Test cross-domain search:**
   ```bash
   curl -X POST "http://localhost:8002/api/v1/domains/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "servo motor error", "top_k": 5}'
   ```

4. **Get MCP tools for Claude:**
   ```bash
   curl "http://localhost:8002/api/v1/tools/mcp-manifest"
   ```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     memOS (Port 8001)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              UniversalOrchestrator                    │  │
│  │  ├── Query Router (domain detection)                 │  │
│  │  ├── MCP Tool Registry (from PDF Tools)              │  │
│  │  └── RRF Result Fusion                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            PDF Extraction Tools (Port 8002)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Domain Federation Layer                  │  │
│  │  ├── DomainGraphCache (lazy loading)                 │  │
│  │  ├── Cross-domain search with RRF                    │  │
│  │  └── Entity type normalization                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  FANUC   │  │   IMM    │  │ Ind.Auto │  │ OEM_IMM  │   │
│  │ 268,886  │  │  2,993   │  │  4,965   │  │  9,556   │   │
│  │  nodes   │  │  nodes   │  │  nodes   │  │  nodes   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Changelog

### v1.0.0 (2026-01-03)
- Initial multi-domain federation API
- Cross-domain search with RRF fusion
- 8 MCP-compatible tool definitions
- Domain-specific entity type filtering
- Rate limiting middleware
- Unified response format compliance

---

*Report generated for memOS integration*
*PDF Extraction Tools - Recovery_Bot Ecosystem*
