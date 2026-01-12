# Machine Entity Graph for FANUC Robot Architecture

**Date**: 2026-01-11
**Phase**: 49
**Status**: Implemented
**API Endpoint**: `http://localhost:8002/api/v1/machine/*`

## Overview

The Machine Entity Graph provides hierarchical modeling of FANUC robot physical architecture, enabling path-based traversal from error codes to affected physical components. This feature enhances troubleshooting workflows by mapping servo errors (SRVO-0xx) to specific motors, encoders, brakes, and other components.

## Architecture

### Physical Hierarchy

```
Robot (e.g., M-16iB/20)
├── Controller (R-30iB)
└── Frame
    └── Axis (J1-J6, E1-E6)
        ├── Motor (A06B-0xxx-Bxxx)
        ├── Encoder/Pulsecoder (A860-xxxx-Txxx)
        ├── Brake (holding brake)
        ├── Motor Cable (A660-xxxx-Txxx)
        └── Encoder Cable
```

### Node Types

| Type | Description | Example |
|------|-------------|---------|
| `robot` | Complete robot unit | M-16iB/20 |
| `controller` | Robot controller | R-30iB |
| `frame` | Mechanical structure | M-16iB frame |
| `axis` | Individual axis (J1-J6) | J1, J2, ... J6 |
| `motor` | Servo motor | A06B-0128-B076 |
| `encoder` | Pulsecoder | A860-0360-T001 |
| `brake` | Holding brake | J1 brake |
| `cable` | Motor/encoder cable | Motor cable J1 |
| `amplifier` | Servo amplifier | A06B-6114-H105 |

### Edge Types

| Type | Description |
|------|-------------|
| `has_axis` | Robot contains axis |
| `has_motor` | Axis contains motor |
| `has_encoder` | Axis contains encoder |
| `has_brake` | Motor has brake |
| `drives` | Motor drives axis |
| `monitors` | Encoder monitors axis |
| `error_affects` | Error code affects component |
| `component_causes` | Component can cause error |

## Error-to-Component Mapping

The system maps 20+ SRVO-0xx error codes to specific component types:

| Error Code | Component | Severity | Description |
|------------|-----------|----------|-------------|
| SRVO-001 | motor | warning | Servo motor warning |
| SRVO-002 | motor | stop | Servo motor fault |
| SRVO-004 | encoder | stop | Encoder communication error |
| SRVO-006 | encoder | stop | Encoder data error |
| SRVO-007 | brake | warning | Brake abnormality |
| SRVO-023 | amplifier | stop | Servo amplifier fault |
| SRVO-036 | encoder | stop | Encoder position error |
| SRVO-045 | encoder | stop | Encoder battery low |
| SRVO-063 | motor | warning | Servo motor overheat |
| SRVO-064 | encoder | stop | Encoder disconnect |
| SRVO-068 | motor | stop | Servo motor overload |

## API Endpoints

### List Robot Models
```http
GET /api/v1/machine/robots
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "model": "M-16iB/20",
      "series": "M-16iB",
      "num_axes": 6,
      "controller_model": "R-30iB"
    }
  ]
}
```

### Get Robot Hierarchy
```http
GET /api/v1/machine/robots/{model}
```

**Example:** `GET /api/v1/machine/robots/M-16iB%2F20`

**Response:**
```json
{
  "success": true,
  "data": {
    "robot": {
      "model": "M-16iB/20",
      "series": "M-16iB",
      "num_axes": 6
    },
    "axes": {
      "J1": {
        "axis": {"axis_number": 1, "motion_range_deg": [-185, 185]},
        "motor": {"motor_model": "A06B-0128-B076"},
        "encoder": {"encoder_model": "A860-0360-T001"},
        "brake": {"holding_torque_nm": 35.0},
        "cables": [...]
      },
      "J2": {...},
      ...
    }
  }
}
```

### Get Axis Components
```http
GET /api/v1/machine/axes/{model}/{axis_number}
```

**Example:** `GET /api/v1/machine/axes/M-16iB%2F20/1`

**Response:**
```json
{
  "success": true,
  "data": {
    "axis_number": 1,
    "robot_model": "M-16iB/20",
    "motor": {
      "motor_model": "A06B-0128-B076",
      "rated_torque_nm": 35.0
    },
    "encoder": {
      "encoder_model": "A860-0360-T001",
      "resolution_ppr": 131072
    },
    "brake": {...},
    "cables": [...]
  }
}
```

### Find Error-to-Component Path
```http
POST /api/v1/machine/error-path
Content-Type: application/json

{
  "error_code": "SRVO-063",
  "robot_model": "M-16iB/20",
  "axis_number": 1
}
```

**Response:**
```json
{
  "error_code": "SRVO-063",
  "robot_model": "M-16iB/20",
  "axis_number": 1,
  "path": {
    "nodes": ["error:SRVO-063", "motor_abc123", "axis_def456", "robot_ghi789"],
    "edges": ["affects:motor", "has_motor", "has_axis"],
    "path_type": "error_to_component"
  },
  "affected_components": [
    {
      "id": "motor_abc123",
      "type": "motor",
      "name": "M-16iB/20_J1_motor",
      "severity": "warning",
      "siblings": ["encoder_xyz", "brake_123"]
    }
  ]
}
```

### Get Troubleshooting Context
```http
GET /api/v1/machine/troubleshoot?error_code=SRVO-063&robot_model=M-16iB/20&axis_number=1
```

**Response:**
```json
{
  "success": true,
  "data": {
    "error_code": "SRVO-063",
    "robot_model": "M-16iB/20",
    "axis_number": 1,
    "error_info": {
      "axis_specific": true,
      "component": "motor",
      "severity": "warning"
    },
    "affected_components": [
      {"id": "motor_...", "type": "motor", "name": "M-16iB/20_J1_motor"}
    ],
    "component_path": {
      "nodes": ["error:SRVO-063", "motor_...", "axis_...", "robot_..."],
      "path_type": "error_to_component"
    },
    "related_errors": [
      {"error_code": "SRVO-068", "confidence": 0.95},
      {"error_code": "SRVO-069", "confidence": 0.8}
    ],
    "sibling_components": ["encoder_...", "brake_..."]
  }
}
```

### List Error Patterns
```http
GET /api/v1/machine/error-patterns
```

### Get Graph Statistics
```http
GET /api/v1/machine/stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "robots": 5,
    "frames": 5,
    "axes": 30,
    "motors": 30,
    "encoders": 30,
    "brakes": 20,
    "cables": 60,
    "edges": 180
  }
}
```

## Integration with memOS

### Use Case: Technician Troubleshooting

When a technician reports "SRVO-063 on J1", memOS can:

1. **Query error-to-component path:**
   ```python
   response = requests.post(
       "http://localhost:8002/api/v1/machine/error-path",
       json={
           "error_code": "SRVO-063",
           "robot_model": "M-16iB/20",
           "axis_number": 1
       }
   )
   affected = response.json()["affected_components"]
   # -> Motor A06B-0128-B076 on J1
   ```

2. **Get troubleshooting context:**
   ```python
   response = requests.get(
       "http://localhost:8002/api/v1/machine/troubleshoot",
       params={
           "error_code": "SRVO-063",
           "robot_model": "M-16iB/20",
           "axis_number": 1
       }
   )
   context = response.json()["data"]
   # -> Related errors, sibling components, severity
   ```

3. **Combine with document search:**
   ```python
   # Get remedies from knowledge graph
   search_response = requests.post(
       "http://localhost:8002/api/v1/search",
       json={
           "query": f"SRVO-063 J1 motor {context['affected_components'][0]['name']}",
           "corpus_filter": {"categories": ["SRVO"]}
       }
   )
   ```

### MCP Tool Integration

The machine graph can be exposed as MCP tools:

```json
{
  "name": "fanuc_error_to_component",
  "description": "Map FANUC error code to affected physical component",
  "parameters": {
    "error_code": "string (e.g., SRVO-063)",
    "robot_model": "string (e.g., M-16iB/20)",
    "axis_number": "integer (1-6, optional)"
  }
}
```

```json
{
  "name": "fanuc_get_axis_components",
  "description": "Get all components for a robot axis",
  "parameters": {
    "robot_model": "string",
    "axis_number": "integer (1-6)"
  }
}
```

## Supported Robot Models

| Series | Models | Axes |
|--------|--------|------|
| M-16iB | M-16iB/20, M-16iB/10L | 6 |
| LR Mate 200iD | 200iD/7L, 200iD/4S | 6 |
| M-710iC | M-710iC/50, M-710iC/70 | 6 |
| M-20iA/iB/iD | M-20iA/35M, M-20iB/25 | 6 |
| R-2000iC | R-2000iC/165F, R-2000iC/210F | 6 |
| M-410iB/iC | M-410iB/700, M-410iC/500 | 4 |
| CRX | CRX-10iA, CRX-25iA | 6 |
| Arc Mate | 100iC, 120iC | 6 |

## File Structure

```
pdf_extractor/graph/machine_architecture/
├── __init__.py          # Module exports
├── models.py            # Node/Edge types, dataclasses
├── builder.py           # Graph construction
└── traversal.py         # Path finding algorithms

pdf_extractor/api/routes/
└── machine.py           # REST API endpoints

tests/graph/machine_architecture/
└── test_machine_graph.py  # 31 tests
```

## Test Coverage

```
31 passed in 9.51s

- TestMachineNodeTypes (3 tests)
- TestRobotNode (2 tests)
- TestAxisNode (2 tests)
- TestMotorNode (1 test)
- TestEncoderNode (1 test)
- TestMachineEdge (1 test)
- TestMachineGraphBuilder (9 tests)
- TestMachineTraversal (8 tests)
- TestMachineGraphIntegration (2 tests)
- TestMachineConstants (2 tests)
```

## Future Enhancements

1. **Part Number Validation**: Cross-reference with FANUC parts catalog
2. **Wiring Diagram Integration**: Map cable routes between components
3. **Maintenance Scheduling**: Track component wear based on usage
4. **Cross-Robot Compatibility**: Identify shared components across models
5. **3D Position Data**: Add spatial coordinates for visualization

## Dependencies

- Existing `UnifiedDocumentGraph` for integration
- Existing `EdgeType` and `NodeType` from `pdf_extractor.graph.models`
- FastAPI for REST endpoints

## Related Documentation

- [DIAGNOSTIC_PATHS.md](../../PDF_Extraction_Tools/docs/DIAGNOSTIC_PATHS.md) - Path traversal for troubleshooting
- [FEDERATION_API.md](../../PDF_Extraction_Tools/docs/FEDERATION_API.md) - Multi-domain API
- [HSEA_INTEGRATION.md](../../PDF_Extraction_Tools/docs/HSEA_INTEGRATION.md) - Embedding architecture

---

*Implementation by Claude Code - Phase 49*
