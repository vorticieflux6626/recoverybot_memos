# RESEARCH/FULL Preset Critical Bugs

**Date**: 2026-01-03
**Severity**: P0 - Critical
**Affected Presets**: RESEARCH, FULL

## Bug Summary

| Bug ID | Description | Impact | Location |
|--------|-------------|--------|----------|
| BUG-001 | 500-char synthesis truncation | Syntheses cut to 538 bytes | `orchestrator_universal.py:5730` |
| BUG-002 | DAG singleton without clear | Cross-query contamination | `orchestrator_universal.py:987-991` |

## BUG-001: 500-Character Synthesis Truncation

### Symptom
RESEARCH and FULL presets consistently produce 538-byte outputs instead of 2000-6000 bytes.

### Root Cause
In `_phase_reasoning_dag_conclusion()`:
```python
# Line 5730 - TRUNCATION BUG
dag.add_node(synthesis[:500], NodeType.CONCLUSION)
```

The full synthesis is truncated to 500 characters before being added to the reasoning DAG.

Then in `reasoning_dag.py:778`:
```python
# Returns truncated content + 38-char marker
return best.content + convergence_info  # convergence_info = "[Confidence: X.XX, Type: conclusion]"
```

### Evidence
All RESEARCH/FULL syntheses are exactly 538 bytes:
- Query 1 RESEARCH: 538 bytes
- Query 1 FULL: 538 bytes
- Query 2 RESEARCH: 538 bytes
- Query 2 FULL: 538 bytes

### Fix
```python
# orchestrator_universal.py:5730
# BEFORE (BUG):
dag.add_node(synthesis[:500], NodeType.CONCLUSION)

# AFTER (FIX):
dag.add_node(synthesis, NodeType.CONCLUSION)  # Don't truncate!
```

Or if truncation is intentional for DAG reasoning, preserve full synthesis:
```python
# Add summary node for DAG reasoning
dag.add_node(synthesis[:500], NodeType.CONCLUSION)

# But return full synthesis, not DAG output
result = {
    "paths": len(dag.nodes),
    "enhanced_synthesis": synthesis  # Return FULL synthesis, not DAG conclusion
}
```

---

## BUG-002: DAG Singleton Without Clear

### Symptom
Query 2 (Allen-Bradley) returns FANUC content from Query 1.

### Root Cause
The reasoning DAG is a singleton that persists across requests:
```python
# Lines 987-991 - SINGLETON BUG
def _get_reasoning_dag(self) -> ReasoningDAG:
    if self._reasoning_dag is None:
        self._reasoning_dag = create_reasoning_dag(self.ollama_url)
    return self._reasoning_dag  # Same instance reused!
```

The DAG has a `clear()` method (line 909) but it's never called:
```python
def clear(self) -> None:
    """Clear all nodes from the DAG."""
    self.nodes.clear()
    self.root_ids.clear()
    self.sink_ids.clear()
```

When `get_convergent_answer()` runs, it may return high-confidence nodes from previous queries.

### Evidence
Query 2 FULL synthesis file contains:
```
"To address the intermittent SRVO-023 (Stop error excess) and MOTN-023 (In singularity) errors
on the FANUC R-30iB Plus controller..."
```
This is FANUC content, but Query 2 was about Allen-Bradley ControlLogix!

### Fix
Clear the DAG at the start of each request:
```python
# orchestrator_universal.py - In _phase_init_reasoning_dag()
async def _phase_init_reasoning_dag(self, request: SearchRequest, request_id: str):
    start = time.time()
    try:
        dag = self._get_reasoning_dag()
        dag.clear()  # ADD THIS LINE - Clear previous query's nodes!
        # ... rest of initialization
```

Or create a new DAG instance per request:
```python
# Alternative: Don't use singleton
async def _phase_init_reasoning_dag(self, request: SearchRequest, request_id: str):
    dag = create_reasoning_dag(self.ollama_url)  # Fresh instance
    # ...
```

---

## Immediate Fixes Required

### Fix 1: Remove truncation (5 seconds)
```python
# File: orchestrator_universal.py
# Line: 5730
# Change:
dag.add_node(synthesis[:500], NodeType.CONCLUSION)
# To:
dag.add_node(synthesis, NodeType.CONCLUSION)
```

### Fix 2: Clear DAG between requests (5 seconds)
```python
# File: orchestrator_universal.py
# In _phase_init_reasoning_dag(), after line 5406:
dag = self._get_reasoning_dag()
dag.clear()  # Add this line
```

---

## Testing After Fix

```bash
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate

# Test Query 1 with RESEARCH preset
timeout 600 python tests/test_industrial_preset_comparison.py --query 1 --preset research

# Verify:
# 1. Synthesis length > 1000 chars (not 538)
# 2. Content matches query (FANUC for Q1, Allen-Bradley for Q2)
# 3. No [Confidence: X.XX, Type: conclusion] marker in output
```

---

## Additional Observations

### Why ENHANCED Works
ENHANCED preset has `enable_reasoning_dag=False` (line 513 doesn't set it), so it:
1. Doesn't call `_phase_reasoning_dag_conclusion()`
2. Returns the full synthesis directly
3. No truncation or cache contamination

### Preset Comparison

| Preset | enable_reasoning_dag | Result |
|--------|---------------------|--------|
| BALANCED | False | Full synthesis |
| ENHANCED | False | Full synthesis |
| RESEARCH | True | Truncated + contaminated |
| FULL | True | Truncated + contaminated |

---

*Report generated from industrial automation preset comparison tests*
