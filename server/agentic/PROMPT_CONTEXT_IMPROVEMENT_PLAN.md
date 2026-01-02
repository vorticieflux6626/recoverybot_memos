# Prompt & Context Engineering Improvement Plan

> **Created**: 2026-01-02 | **Status**: Active | **Version**: 1.1 | **Updated**: 2026-01-02

## Executive Summary

Based on comprehensive research into prompt engineering and context engineering best practices, combined with an audit of our current agentic pipeline, this plan outlines high-impact improvements for the Recovery Bot memOS industrial troubleshooting system.

### Implementation Status (2026-01-02)

| Improvement | Status | Notes |
|-------------|--------|-------|
| ✅ Fix LLM URL issue | **COMPLETE** | Components now use `self.ollama_url` |
| ✅ Fix float16 serialization | **COMPLETE** | Added `float()` conversion |
| ✅ Role-based agent personas | **COMPLETE** | All 5 agents updated |
| ✅ Thorough reasoning instruction | **COMPLETE** | Replaced CoD with full reasoning |
| ✅ STAY ON TOPIC requirement | **COMPLETE** | Synthesizer now rejects off-topic |
| 🔄 Lost-in-middle mitigation | Pending | Document reordering |
| 🔄 Temperature tuning | Pending | Per-agent optimization |

### Design Decision: Full Reasoning Over Token Reduction

**Original Plan**: Chain-of-Draft (CoD) to reduce tokens by 68-92%

**Changed To**: Thorough step-by-step reasoning

**Rationale**: For industrial troubleshooting queries, accuracy is more important than token efficiency. The relevance drift issue (answering about "Tool Life Management" instead of "SRVO-023 collision detection") showed that abbreviated reasoning can lose focus. Full reasoning provides:
- Better topic adherence
- More thorough evaluation of source relevance
- Higher confidence in technical accuracy

### Current State Assessment

| Component | Current Implementation | Status |
|-----------|----------------------|--------|
| **Thorough Reasoning** | Applied to all agents | ✅ Complete |
| **Role-Based Prompts** | All 5 agents have personas | ✅ Complete |
| **STAY ON TOPIC** | Synthesizer has explicit requirement | ✅ Complete |
| **Output Structure** | JSON for most agents, XML tags for roles | ✅ Complete |
| **Lost-in-Middle** | Not addressed | 🔄 Pending |
| **Temperature Tuning** | Default values | 🔄 Pending |

### Remaining Priority Matrix

| Priority | Improvement | Impact | Effort | Target |
|----------|-------------|--------|--------|--------|
| **P1** | Implement lost-in-middle mitigation | High | Low | 2h |
| **P2** | Temperature tuning per agent | Medium | Low | 1h |
| **P3** | Self-consistency for STRICT verification | Medium | Medium | 4h |
| **P3** | Dynamic few-shot from experience store | Medium | High | 8h |

---

## Completed Improvements

### 1. Fix Pipeline Issues ✅

**Issue 1**: LLM URL missing protocol - **FIXED**
- Root cause: `orchestrator_universal.py` was passing `ollama_url` parameter instead of `self.ollama_url`
- Fix: Changed all component initializations to use `self.ollama_url`

**Issue 2**: float16 serialization error - **FIXED**
- Location: `orchestrator_universal.py` lines ~4648, ~4699
- Fix: Added `float()` conversion for all numpy scores

### 2. Role-Based Agent Personas ✅

All 5 agents now have explicit role personas with industrial automation expertise:

| Agent | Role | Expertise |
|-------|------|-----------|
| `analyzer.py` | INDUSTRIAL QUERY ANALYZER | FANUC, Allen-Bradley, Siemens, servo systems |
| `verifier.py` | TECHNICAL FACT VERIFIER | Cross-reference, source credibility |
| `query_classifier.py` | QUERY ROUTING SPECIALIST | Pipeline routing, complexity assessment |
| `retrieval_evaluator.py` | RETRIEVAL QUALITY EVALUATOR | Document relevance, coverage scoring |
| `synthesizer.py` | EXPERT RESEARCH SYNTHESIZER | Multi-source synthesis, citation accuracy |

### 3. STAY ON TOPIC Requirement ✅

Added explicit focus requirements to synthesizer prompts:

```
**FOCUS REQUIREMENT**: Your answer MUST directly address the specific topic
in the Original Question below. Do NOT answer about tangentially related
topics that appear in search results. Stay focused on exactly what was asked.
```

And in CRITICAL REQUIREMENTS:
```
1. **STAY ON TOPIC**: Answer ONLY about the specific error code, component,
   or procedure mentioned in the Original Question. Ignore unrelated content
   in search results.
```

Warning updated:
```
WARNING: Responses that answer about a DIFFERENT topic than asked will be
rejected. Responses without [Source N] citations will be considered incomplete.
```

---

## P1: High-Impact Improvements

### 3. Add Role-Based Agent Personas

**Purpose**: Improve agent specialization and reduce ambiguity.

**Pattern for Each Agent**:

```python
# In analyzer.py
ANALYZER_PERSONA = """<role>INDUSTRIAL QUERY ANALYZER</role>
<expertise>
- FANUC robotics (R-30iB, R-30iA controllers, SRVO/MOTN/SYST alarms)
- Allen-Bradley PLCs (ControlLogix, CompactLogix, fault codes)
- Siemens automation (S7-1500, SINAMICS drives, fault codes)
- Servo systems, motor drives, encoder feedback, collision detection
</expertise>
<task>Analyze the query to determine search strategy and complexity.</task>
"""

# In verifier.py
VERIFIER_PERSONA = """<role>TECHNICAL FACT VERIFIER</role>
<expertise>
Cross-reference claims against manufacturer documentation.
Identify conflicts between sources. Assess source credibility.
</expertise>
"""

# In synthesizer.py
SYNTHESIZER_PERSONA = """<role>EXPERT RESEARCH SYNTHESIZER</role>
<expertise>
Combine information from multiple sources into accurate, actionable answers.
Every claim MUST be cited. Use technical terminology correctly.
</expertise>
"""
```

### 4. Implement Lost-in-Middle Mitigation

**Purpose**: LLMs attend less to middle of context. Reorder documents to place important content at start and end.

**Implementation in `context_curator.py` or `orchestrator_universal.py`**:

```python
def reorder_for_attention(sources: List[Dict], relevance_key: str = "relevance") -> List[Dict]:
    """Reorder sources to mitigate lost-in-the-middle problem.

    Pattern: Most relevant at start, second-most at end, alternating.
    """
    sorted_sources = sorted(sources, key=lambda s: s.get(relevance_key, 0), reverse=True)

    if len(sorted_sources) <= 2:
        return sorted_sources

    reordered = []
    for i, source in enumerate(sorted_sources):
        if i % 2 == 0:
            reordered.insert(0, source)  # Add to beginning
        else:
            reordered.append(source)  # Add to end

    return reordered
```

**Apply before synthesis prompt construction.**

---

## P2: Medium-Impact Improvements

### 5. Add `/no_think` Suffix for qwen3:8b Fast Operations

**Purpose**: Qwen3 supports thinking mode toggle. Disable for fast agents.

**Current**: Coverage evaluation already uses `/no_think`:
```python
# In analyzer.py line ~874
prompt = f"""...Return ONLY a JSON object. /no_think"""
```

**Change**: Apply to all qwen3:8b prompts:
- `_build_analysis_prompt()` - add `/no_think`
- Verifier prompt - add `/no_think`
- Query classifier - add `/no_think`

### 6. Temperature Tuning Per Agent

**Current**: Temperature defaults vary across agents.

**Optimal Settings**:
| Agent | Temperature | Reasoning |
|-------|-------------|-----------|
| Analyzer | 0.3 | Deterministic classification |
| Query Classifier | 0.2 | Consistent routing |
| Verifier | 0.1 | High precision required |
| Synthesizer (qwen3:8b) | 0.5 | Balanced creativity/accuracy |
| Synthesizer (DeepSeek R1) | 0.6 | Per DeepSeek docs |

**Implementation**: Add to agent constructors or config.

---

## P3: Long-Term Improvements

### 7. Self-Consistency for STRICT Verification

**Purpose**: Generate multiple reasoning paths and vote on verification result.

**Implementation**:
```python
async def _verify_strict_self_consistent(
    self,
    claim: str,
    sources: List[WebSearchResult],
    n_paths: int = 3
) -> VerificationResult:
    # Generate n_paths independent verifications
    tasks = [self._verify_strict(claim, sources) for _ in range(n_paths)]
    results = await asyncio.gather(*tasks)

    # Vote on verified status
    verified_count = sum(1 for r in results if r.verified)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    return VerificationResult(
        claim=claim,
        verified=verified_count > n_paths / 2,
        confidence=avg_confidence,
        sources=list(set(s for r in results for s in r.sources)),
        conflicts=list(set(c for r in results for c in r.conflicts))
    )
```

**Only apply for STRICT verification level (3x cost).**

### 8. Dynamic Few-Shot from Experience Store

**Purpose**: Use successful past resolutions as examples.

**Integration with `experience_distiller.py`**:
```python
async def get_dynamic_examples(query: str, k: int = 2) -> List[Dict]:
    """Retrieve relevant successful resolutions as few-shot examples."""
    # Search experience store for similar queries
    similar = await experience_store.search(
        query=query,
        filter={"confidence": {"$gte": 0.8}, "user_satisfied": True},
        limit=k
    )

    return [
        {
            "query": e.original_query,
            "answer": e.synthesis[:500],  # Truncate for context budget
            "sources_used": len(e.sources)
        }
        for e in similar
    ]
```

**Inject into synthesis prompt when examples found.**

---

## Prompt Templates (Updated)

### Analyzer Prompt (with CoD and Persona)

```python
def _build_analysis_prompt(self, query: str, context: Optional[Dict]) -> str:
    return f"""<role>INDUSTRIAL QUERY ANALYZER for manufacturing automation</role>
<expertise>FANUC, Allen-Bradley, Siemens, servo systems, PLCs, industrial robots</expertise>

Think step by step, but only keep ~5 words per step.

Analyze this query: "{query}"

Return JSON with:
- requires_search: bool
- query_type: factual|troubleshooting|comparative|procedural
- key_topics: list
- estimated_complexity: low|medium|high|expert
- requires_thinking_model: bool

/no_think
JSON:"""
```

### Verifier Prompt (with Role and CoD)

```python
VERIFIER_PROMPT = """<role>TECHNICAL FACT VERIFIER for industrial automation</role>

Think minimally, then verify.

Claim: {claim}

Sources:
{sources_text}

Verify if claim is supported. Return JSON:
{{"supported": bool, "confidence": 0-1, "conflicts": [], "reasoning": "brief"}}

/no_think
JSON:"""
```

### Synthesizer Prompt (with XML Sections)

```python
SYNTHESIS_PROMPT = """<role>EXPERT RESEARCH SYNTHESIZER</role>

<domain_knowledge>
{domain_context}
</domain_knowledge>

<sources>
{formatted_sources}
</sources>

<question>{query}</question>

<requirements>
1. EVERY claim MUST have [Source N] citation
2. Use key technical terms from question
3. Note source conflicts if any
4. Be specific with parameters/values
</requirements>

<answer>"""
```

---

## Metrics to Track

After implementing improvements, monitor:

| Metric | Current Baseline | Target |
|--------|-----------------|--------|
| Avg Response Time | 120-180s | <60s |
| Confidence Score | 57-69% | 80%+ |
| Term Coverage | 50% | 80%+ |
| Citation Rate | ~50% | 100% |
| Token Efficiency | - | 50% reduction |

Use the observability dashboard to track these metrics per-request.

---

## Implementation Order

1. **Day 1 (P0)**:
   - Fix LLM URL issue
   - Fix float16 serialization
   - Add CoD to analyzer, verifier, query_classifier

2. **Day 2 (P1)**:
   - Add role-based personas to all agents
   - Implement lost-in-middle reordering

3. **Day 3 (P2)**:
   - Add `/no_think` to all qwen3:8b prompts
   - Tune temperatures per agent

4. **Week 2 (P3)**:
   - Self-consistency for STRICT verification
   - Dynamic few-shot integration

---

## Audit Results (2026-01-02)

### Pipeline Test Results

Tested industrial query: "FANUC R-30iB SRVO-023 collision detection alarm troubleshooting"

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Duration | 82.7s | <60s | ❌ Slow |
| Confidence | 56% | 80%+ | ❌ Low |
| Has Citations | True | True | ✅ OK |
| Has Tech Terms | True | True | ✅ OK |
| Synthesis Length | 2567 chars | 1000+ | ✅ OK |

### Critical Issues Found

1. **P0 Issues Fixed**:
   - ✅ LLM URL issue fixed (was passing `None` instead of resolved URL)
   - ✅ float16 serialization fixed (added `float()` conversion)

2. **Relevance Drift Problem**:
   - Synthesis answered about "Tool Life Management" instead of "SRVO-023 collision detection"
   - Likely caused by search results including tangentially related FANUC content
   - Need better query focusing and document filtering

3. **Search Engine Rate Limiting**:
   - Brave, Reddit, Wikipedia all rate-limited during test
   - Fallback to DuckDuckGo only

### Recommended Next Steps

1. **Immediate (P0)**: Apply Chain-of-Draft to analyzer for faster query analysis
2. **Short-term (P1)**: Add role personas to improve topic focus
3. **Medium-term (P2)**: Implement lost-in-middle reordering for better relevance

---

*Generated from research audits a5cb50b (prompt engineering) and a2c0edf (context engineering)*
*Updated with pipeline audit results 2026-01-02*
