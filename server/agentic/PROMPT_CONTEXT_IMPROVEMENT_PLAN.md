# Prompt & Context Engineering Improvement Plan

> **Created**: 2026-01-02 | **Status**: Active | **Version**: 1.0

## Executive Summary

Based on comprehensive research into prompt engineering and context engineering best practices, combined with an audit of our current agentic pipeline, this plan outlines high-impact improvements for the Recovery Bot memOS industrial troubleshooting system.

### Current State Assessment

| Component | Current Implementation | Gap |
|-----------|----------------------|-----|
| **Chain-of-Draft** | Applied to synthesizer only | Not applied to fast agents |
| **Role-Based Prompts** | Partial (analyzer has domain knowledge) | Missing explicit agent personas |
| **Output Structure** | JSON for most agents | Could use XML sections for synthesis |
| **KV Cache Optimization** | Good prefix structure exists | Not consistently applied |
| **Lost-in-Middle** | Not addressed | Need document reordering |
| **Context Limits** | Well implemented | Could add source prioritization |

### Priority Matrix

| Priority | Improvement | Impact | Effort | Target |
|----------|-------------|--------|--------|--------|
| **P0** | Apply Chain-of-Draft to all fast agents | High | Low | 2h |
| **P0** | Fix pipeline issues (LLM URL, float16) | Critical | Low | 1h |
| **P1** | Add role-based agent personas | High | Medium | 4h |
| **P1** | Implement lost-in-middle mitigation | High | Low | 2h |
| **P2** | Add `/no_think` suffix for qwen3:8b | Medium | Low | 1h |
| **P2** | Temperature tuning per agent | Medium | Low | 1h |
| **P3** | Self-consistency for STRICT verification | Medium | Medium | 4h |
| **P3** | Dynamic few-shot from experience store | Medium | High | 8h |

---

## P0: Critical Fixes

### 1. Apply Chain-of-Draft to All Fast Agents

**Current State**: Only `synthesizer.py` uses CoD for thinking models.

**Change**: Add CoD instruction to all qwen3:8b agent prompts.

**Files to Modify**:
- `analyzer.py` → `_build_analysis_prompt()`
- `verifier.py` → prompt in `_verify_strict()`
- `query_classifier.py` → classification prompt
- `retrieval_evaluator.py` → CRAG evaluation prompt

**Pattern**:
```python
COD_INSTRUCTION = "Think step by step, but only keep ~5 words per step. Then provide your answer."

def _build_analysis_prompt(self, query, context):
    return f"""{COD_INSTRUCTION}

Analyze this query...
"""
```

**Expected Impact**: 68-92% fewer reasoning tokens, 40-76% faster response.

### 2. Fix Pipeline Issues

**Issue 1**: LLM URL missing protocol
- Check `gateway_client.py` for missing `http://` prefix
- Ensure fallback to direct Ollama uses full URL

**Issue 2**: float16 serialization error in hybrid reranking
- Location: `orchestrator_universal.py` line ~700
- Fix: Convert numpy float16 to Python float before JSON serialization

```python
# Before
scores = reranker.compute_score(...)
# After
scores = [float(s) for s in reranker.compute_score(...)]
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
