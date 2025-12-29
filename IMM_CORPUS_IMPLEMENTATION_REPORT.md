# IMM Corpus Ingestion Implementation Report

**Date:** 2025-12-29
**Project:** Recovery Bot - Agentic Search
**Domain:** Injection Molding Machine (IMM) & FANUC Robot Integration

---

## Executive Summary

This report documents the implementation of a specialized knowledge corpus ingestion system for Injection Molding Machines (IMM) and FANUC robot integration. The system enables AI-powered troubleshooting by extracting structured entities from technical documentation, manufacturer resources, and industry forums.

---

## 1. Scope & Objectives

### 1.1 Target Knowledge Domains

| Domain | Description |
|--------|-------------|
| **Euromap Protocols** | Robot-IMM interface standards (67, 67.1, 73, 77) |
| **Machine Manufacturers** | KraussMaffei, Cincinnati Milacron, Van Dorn/Sumitomo Demag |
| **Control Systems** | MC5/MC6 (KraussMaffei), MOSAIC+ (Milacron), PathFinder (Van Dorn) |
| **FANUC Integration** | Roboshot, R-30iB controllers, robot-IMM communication |
| **Defect Analysis** | Short shot, flash, sink marks, warpage, weld lines, etc. |
| **Scientific Molding** | RJG DECOUPLED MOLDING, process optimization |

### 1.2 Objectives

1. Build a structured URL corpus from authoritative technical sources
2. Define entity extraction patterns for IMM-specific terminology
3. Create relationship mappings for troubleshooting workflows
4. Integrate IMM query detection into the agentic search pipeline
5. Enable domain-boosted search results for IMM-related queries

---

## 2. Implementation Details

### 2.1 Files Created

| File | Location | Lines | Purpose |
|------|----------|-------|---------|
| `imm_schema.py` | `agentic/schemas/` | 961 | Domain schema, patterns, URL sources |
| `imm_corpus_builder.py` | `agentic/` | 911 | Corpus builder with entity extraction |
| `ingest_imm_corpus.py` | `agentic/scripts/` | 230 | CLI ingestion script |

### 2.2 Files Modified

| File | Changes |
|------|---------|
| `schemas/__init__.py` | Added IMM schema exports |
| `searcher.py` | Added IMM query detection, engine groups, trusted domains |

### 2.3 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMM Corpus Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  URL Sources │───▶│   Ingestion  │───▶│   Corpus     │       │
│  │  (81 URLs)   │    │   Pipeline   │    │   Storage    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Schema     │    │   Pattern    │    │   Entity     │       │
│  │  Definition  │    │  Extraction  │    │  Relations   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Entity Type Definitions

### 3.1 IMMEntityType Enumeration

```python
class IMMEntityType(str, Enum):
    MACHINE_MODEL = "machine_model"        # GX, CX, MC5, MOSAIC+
    CONTROL_SYSTEM = "control_system"      # MC6, PathFinder
    COMPONENT = "component"                # Screw, barrel, nozzle
    ERROR_CODE = "error_code"              # Machine alarms
    EUROMAP_SIGNAL = "euromap_signal"      # X1-X32, Y1-Y32
    EUROMAP_PROTOCOL = "euromap_protocol"  # EM67, EM73, EM77
    PROCESS_VARIABLE = "process_variable"  # Pressure, temp, speed
    DEFECT = "defect"                      # Short shot, flash
    MATERIAL = "material"                  # ABS, PP, PC, Nylon
    MOLD_COMPONENT = "mold_component"      # Sprue, runner, gate
    TROUBLESHOOTING_STEP = "troubleshooting_step"
    PARAMETER = "parameter"                # Machine settings
    PROCEDURE = "procedure"                # Setup, maintenance
    SPECIFICATION = "specification"        # Technical specs
```

### 3.2 IMMRelationType Enumeration

```python
class IMMRelationType(str, Enum):
    CAUSES = "causes"                      # Defect causation
    INDICATES = "indicates"                # Symptom indication
    RESOLVED_BY = "resolved_by"            # Solution mapping
    USES_CONTROL = "uses_control"          # Machine-controller
    FOLLOWS_PROTOCOL = "follows_protocol"  # Euromap compliance
    REQUIRES_SIGNAL = "requires_signal"    # Signal dependencies
    AFFECTS = "affects"                    # Process impact
    COMPATIBLE_WITH = "compatible_with"    # Material compatibility
    REPLACES = "replaces"                  # Component substitution
    MANUFACTURED_BY = "manufactured_by"    # OEM relationships
    DOCUMENTED_IN = "documented_in"        # Source references
```

---

## 4. Pattern Definitions

### 4.1 Machine Model Patterns

| Manufacturer | Pattern | Examples |
|--------------|---------|----------|
| KraussMaffei | `GX\s*\d+`, `CX\s*\d+`, `MX\s*\d+` | GX 550, CX 160 |
| Milacron | `NT\s*\d+`, `Magna\s*T`, `Roboshot` | NT 440, Magna T |
| Van Dorn | `HT\s*\d+`, `Demag\s*\d+` | HT 500, Demag 1000 |

### 4.2 Control System Patterns

| System | Pattern | Manufacturer |
|--------|---------|--------------|
| MC5/MC6 | `MC[3-6]` | KraussMaffei |
| MOSAIC+ | `MOSAIC\+?` | Milacron |
| PathFinder | `PathFinder` | Van Dorn |
| FANUC | `R-30i[AB]`, `CNC\s*\d+i` | FANUC |

### 4.3 Euromap Protocol Patterns

```python
EUROMAP_PROTOCOL_PATTERNS = {
    "EM67": r"(?:euromap|em)\s*67(?:\.1)?",   # Robot interface
    "EM73": r"(?:euromap|em)\s*73",            # Data exchange
    "EM77": r"(?:euromap|em)\s*77",            # OPC-UA
    "EM79": r"(?:euromap|em)\s*79",            # Energy monitoring
    "EM82": r"(?:euromap|em)\s*82",            # MES interface
}
```

### 4.4 Euromap Signal Patterns

```python
EUROMAP_SIGNAL_PATTERNS = {
    # Robot → IMM signals
    "X1": "Mold area free",
    "X2": "Start signal",
    "X3": "Mold close request",
    # ... (32 input signals)

    # IMM → Robot signals
    "Y1": "Mold closed",
    "Y2": "Ejector forward",
    "Y3": "Safety gate closed",
    # ... (32 output signals)
}
```

### 4.5 Defect Patterns

| Defect | Pattern | Root Causes |
|--------|---------|-------------|
| Short Shot | `short\s+shot` | Low pressure, cold melt |
| Flash | `flash(?:ing)?` | High pressure, worn mold |
| Sink Marks | `sink\s+mark` | Insufficient packing |
| Warpage | `warp(?:age\|ing)?` | Uneven cooling |
| Weld Lines | `weld\s+line` | Flow front collision |
| Burn Marks | `burn\s+mark` | Trapped air, degradation |
| Jetting | `jetting` | High injection speed |
| Delamination | `delamination` | Contamination |

---

## 5. URL Corpus Summary

### 5.1 Source Distribution

| Source Category | Count | Priority |
|-----------------|-------|----------|
| **Euromap Standards** | 5 | Critical |
| **Euromap Guides** | 6 | High |
| **KraussMaffei Official** | 4 | High |
| **KraussMaffei Manuals** | 5 | High |
| **Milacron Official** | 4 | High |
| **Milacron Manuals** | 5 | High |
| **Van Dorn Official** | 4 | High |
| **Van Dorn Manuals** | 5 | High |
| **FANUC Roboshot** | 4 | High |
| **Robot Forum (FANUC)** | 5 | High |
| **Robot Integration Guides** | 4 | High |
| **IM Online Forum** | 5 | High |
| **PLCtalk Forums** | 4 | High |
| **PTOnline Troubleshooting** | 6 | High |
| **Plastics Publications** | 5 | Medium |
| **RJG Training** | 4 | Medium |
| **Mold Components** | 3 | Medium |
| **Control Repair** | 3 | Medium |
| **TOTAL** | **81** | |

### 5.2 Priority Breakdown

| Priority Level | URL Count | Percentage |
|----------------|-----------|------------|
| Critical | 5 | 6% |
| High | 62 | 77% |
| Medium | 14 | 17% |
| **Total** | **81** | 100% |

### 5.3 Content Types

| Content Type | Description | Sources |
|--------------|-------------|---------|
| `standard` | Official specifications | Euromap |
| `manufacturer` | OEM documentation | KraussMaffei, Milacron, Van Dorn |
| `manual` | Technical manuals | All manufacturers |
| `forum` | Community discussions | IM Online, PLCtalk, Robot Forum |
| `article` | Technical articles | PTOnline, Plastics Technology |
| `training` | Educational content | RJG |

---

## 6. Search Integration

### 6.1 Query Detection Patterns

The following patterns trigger IMM-specific search behavior:

```python
IMM_PATTERNS = [
    r"\beuromap\s*(6[7-9]|7\d|8\d)",  # Euromap protocols
    r"\bem\s*6[7-9]",                  # EM67, EM68, etc.
    r"\bkraussmaffei\b",               # Manufacturer
    r"\bmilacron\b",                   # Manufacturer
    r"\bvan\s*dorn\b",                 # Manufacturer
    r"\bmc[3456]\b",                   # Control systems
    r"\bmosaic\+?\b",                  # Control system
    r"\bpathfinder\b",                 # Control system
    r"\bshort\s+shot\b",               # Defects
    r"\bsink\s+mark\b",
    r"\bweld\s+line\b",
    r"\binjection\s+mold",             # Process
    r"\bscientific\s+molding\b",
    r"\bdecoupled\s+molding\b",        # RJG methodology
    r"\broboshot\b",                   # FANUC
]
```

### 6.2 Engine Groups

```python
ENGINE_GROUPS["imm"] = "brave,bing,duckduckgo,reddit,startpage"
ENGINE_GROUPS["imm_technical"] = "github,stackoverflow,arxiv"
ENGINE_GROUPS["imm_academic"] = "arxiv,semantic_scholar,google_scholar"
```

### 6.3 Trusted Domains (30+)

**Premium Domains (0.25 boost):**
- euromap.org
- kraussmaffei.com
- milacron.com
- ptonline.com
- rfriedrich.com

**Trusted Domains (0.15 boost):**
- plasticstechnology.com
- injectionmoldingonline.com
- inmold.com
- moldmakingtechnology.com
- scientificmolding.com
- plctalk.net
- robot-forum.com

---

## 7. Usage Instructions

### 7.1 CLI Commands

```bash
# Navigate to agentic directory
cd /home/sparkone/sdd/Recovery_Bot/memOS/server/agentic

# Dry run - preview what will be ingested
python -m agentic.scripts.ingest_imm_corpus --dry-run

# Ingest priority URLs only (critical + high = 67 URLs)
python -m agentic.scripts.ingest_imm_corpus --priority

# Ingest specific source
python -m agentic.scripts.ingest_imm_corpus --source euromap_standards

# Ingest all sources (81 URLs)
python -m agentic.scripts.ingest_imm_corpus --all

# Show corpus statistics
python -m agentic.scripts.ingest_imm_corpus --stats

# Custom delay between requests (default 1.0s)
python -m agentic.scripts.ingest_imm_corpus --all --delay 2.0

# Custom Ollama URL
python -m agentic.scripts.ingest_imm_corpus --all --ollama-url http://localhost:11434
```

### 7.2 Programmatic Usage

```python
from agentic.imm_corpus_builder import get_imm_builder

# Initialize builder
builder = get_imm_builder(ollama_url="http://localhost:11434")

# Ingest a single URL
result = await builder.ingest_url(
    url="https://www.euromap.org/euromap67",
    source_type="standard"
)

# Get troubleshooting for a defect
troubleshooting = await builder.get_defect_troubleshooting("short shot")

# Extract patterns from text
extraction = builder.extract_imm_patterns(technical_document)
print(extraction.euromap_signals)
print(extraction.defects)
print(extraction.process_variables)

# Get corpus statistics
stats = builder.get_stats()
```

### 7.3 Query Detection

```python
from agentic.schemas import is_imm_query, detect_manufacturer

# Check if query is IMM-related
query = "euromap 67 signal timing for robot interface"
if is_imm_query(query):
    # Route to IMM-specific search
    manufacturer = detect_manufacturer(query)
    # manufacturer = None (no specific manufacturer)

query2 = "KraussMaffei MC6 alarm codes"
if is_imm_query(query2):
    manufacturer = detect_manufacturer(query2)
    # manufacturer = "kraussmaffei"
```

---

## 8. Quality Assurance

### 8.1 Dry Run Validation

The dry run was successfully executed showing:
- 81 total URLs properly categorized
- 18 source groups correctly organized
- Priority levels assigned appropriately
- No duplicate URLs detected

### 8.2 Syntax Validation

All Python files pass syntax validation:
- `imm_schema.py` - Fixed regex quote escaping issue
- `imm_corpus_builder.py` - Clean
- `ingest_imm_corpus.py` - Clean

### 8.3 Import Validation

```bash
python -c "from agentic.schemas import IMM_SCHEMA, is_imm_query; print('OK')"
# Output: OK
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

1. **PDF Extraction Integration**
   - Connect to DocumentGraphService for PDF manual processing
   - Extract structured data from manufacturer PDFs

2. **Entity Relationship Graphs**
   - Build Neo4j-style relationships between entities
   - Enable graph-based troubleshooting traversal

3. **Defect Image Classification**
   - Integrate VL models for defect photo analysis
   - Map visual defects to root causes

4. **Real-time Forum Monitoring**
   - Watch IM Online and PLCtalk for new threads
   - Auto-ingest relevant discussions

### 9.2 Maintenance Schedule

| Task | Frequency |
|------|-----------|
| Re-ingest forum sources | Weekly |
| Check for new Euromap specs | Monthly |
| Validate URL accessibility | Monthly |
| Update manufacturer URLs | Quarterly |

---

## 10. Appendix

### A. Complete URL List

See `agentic/schemas/imm_schema.py` → `IMM_URL_SOURCES` dictionary for the full list of 81 URLs with metadata.

### B. Pattern Test Cases

```python
# Test Euromap detection
assert is_imm_query("euromap 67.1 robot interface")
assert is_imm_query("EM73 data exchange protocol")

# Test manufacturer detection
assert detect_manufacturer("KraussMaffei GX 550") == "kraussmaffei"
assert detect_manufacturer("Milacron MOSAIC+ controller") == "milacron"
assert detect_manufacturer("Van Dorn HT 500") == "vandorn"

# Test defect extraction
from agentic.schemas import extract_defect_types
defects = extract_defect_types("Part has short shot and sink marks")
assert "short_shot" in defects
assert "sink_marks" in defects
```

### C. Related Documentation

- [CLAUDE.md](/home/sparkone/sdd/Recovery_Bot/CLAUDE.md) - Project overview
- [memOS CLAUDE.md](/home/sparkone/sdd/Recovery_Bot/memOS/CLAUDE.md) - memOS documentation
- [FANUC Schema](/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/schemas/fanuc_schema.py) - Related FANUC patterns

---

**Report Generated:** 2025-12-29
**Implementation Status:** Complete
**Ready for Ingestion:** Yes
