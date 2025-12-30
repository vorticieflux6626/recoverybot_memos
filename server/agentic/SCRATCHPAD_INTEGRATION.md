# Scratchpad Integration for Agentic Search

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Complete

## Overview

The Scratchpad provides a **shared working memory** that all agents in the pipeline can read and write to. This enables intelligent coordination without each agent needing to understand the full pipeline.

## Current vs. Proposed Architecture

### Current (Linear Pipeline)
```
Query → Analyzer → Planner → Searcher → Verifier → Synthesizer → Response
              ↓         ↓          ↓          ↓
           (state)   (state)    (state)    (state)

Problem: Each agent only sees its input; no awareness of overall progress
```

### Proposed (Scratchpad-Coordinated)
```
Query → Analyzer ←→ SCRATCHPAD ←→ Synthesizer → Response
             ↑↓         ↑↓
          Planner ←→ Searcher ←→ Verifier

Benefit: All agents read/write shared context, enabling:
- Gap awareness
- Contradiction detection
- Task completion tracking
- Agent-to-agent communication
```

## Integration Points

### 1. Orchestrator Changes

```python
# In orchestrator.py search() method:

async def search(self, request: SearchRequest) -> SearchResponse:
    # Create scratchpad at start
    scratchpad = self.scratchpad_manager.create(
        query=request.query,
        request_id=request_id,
        user_id=request.user_id
    )

    # ANALYZE: Set mission with completion criteria
    analysis = await self.analyzer.analyze(request.query, request.context)
    scratchpad.set_mission(
        decomposed_questions=analysis.decomposed_questions,
        completion_criteria=analysis.completion_criteria  # NEW
    )

    # SEARCH LOOP: Check scratchpad for completion
    while scratchpad.should_continue(request.max_sources):
        # Get context for searcher (includes gaps, contradictions, notes)
        searcher_context = scratchpad.to_context_for_agent("searcher")

        # Searcher uses scratchpad context to prioritize
        results = await self.searcher.search_with_context(
            queries=scratchpad.peek_next_actions(),
            context=searcher_context
        )

        # Record findings in scratchpad
        for result in results:
            question_id = self._match_to_question(result, scratchpad)
            scratchpad.add_finding(
                question_id=question_id,
                content=result.snippet,
                source_url=result.url,
                confidence=result.relevance_score
            )

        # Check for contradictions
        await self._check_contradictions(scratchpad)

        # Update completion status
        status = scratchpad.get_completion_status()
        if status["is_complete"]:
            break

    # SYNTHESIZE: Use scratchpad for comprehensive context
    synthesis = await self.synthesizer.synthesize_with_scratchpad(
        query=request.query,
        scratchpad=scratchpad
    )

    return response
```

### 2. Analyzer Changes

```python
# In analyzer.py:

async def analyze(self, query: str, context: dict) -> QueryAnalysis:
    # Enhanced analysis that generates completion criteria

    prompt = f"""Analyze this query and decompose it into answerable sub-questions.

Query: {query}

For each sub-question, specify:
1. The question itself
2. What information would constitute a complete answer (completion criteria)
3. Priority (1-5, with 1 being most important)

Output JSON:
{{
    "decomposed_questions": ["Q1", "Q2", ...],
    "completion_criteria": {{
        "q1": "Criteria for Q1 to be considered answered",
        "q2": "Criteria for Q2 to be considered answered"
    }},
    "priority_order": [1, 2, 3]
}}
"""

    # ... LLM call and parsing
```

### 3. Searcher Changes

```python
# In searcher.py:

async def search_with_context(
    self,
    queries: List[str],
    scratchpad_context: str
) -> List[WebSearchResult]:
    """Search with awareness of what's already been found"""

    # Parse scratchpad context to understand:
    # - What questions are still unanswered
    # - What gaps exist
    # - What contradictions need resolution

    # Prioritize queries that fill gaps
    prioritized_queries = self._prioritize_for_gaps(
        queries,
        scratchpad_context
    )

    return await self.search(prioritized_queries)
```

### 4. Synthesizer Changes

```python
# In synthesizer.py:

async def synthesize_with_scratchpad(
    self,
    query: str,
    scratchpad: AgenticScratchpad
) -> str:
    """Synthesize using full scratchpad context"""

    # Build context from scratchpad findings
    findings_by_question = {}
    for f_id, finding in scratchpad.findings.items():
        q_id = finding.question_id
        if q_id not in findings_by_question:
            findings_by_question[q_id] = []
        findings_by_question[q_id].append(finding)

    # Build synthesis prompt
    prompt = f"""Synthesize an answer to: {query}

QUESTIONS AND FINDINGS:
"""

    for q_id, q in scratchpad.questions.items():
        prompt += f"\n{q_id}: {q.question_text}\n"
        prompt += f"Status: {q.status.value} | Confidence: {q.confidence:.0%}\n"

        findings = findings_by_question.get(q_id, [])
        for f in findings:
            prompt += f"  - {f.content} [Source: {f.source_title}] (conf: {f.confidence:.0%})\n"

        if q.gaps:
            prompt += f"  GAPS: {', '.join(q.gaps)}\n"

    # Handle contradictions
    if scratchpad.contradictions:
        prompt += "\nCONTRADICTIONS TO ADDRESS:\n"
        for c in scratchpad.contradictions:
            if not c.get("resolved"):
                prompt += f"  - {c['description']}\n"

    prompt += """
Provide a comprehensive answer that:
1. Addresses each question
2. Acknowledges any gaps in information
3. Notes any contradictions and how you resolved them
4. Cites sources for key claims
"""

    return await self._generate(prompt)
```

## Benefits

### 1. Intelligent Gap Detection
```
Before: Synthesizer doesn't know what's missing
After:  Scratchpad tracks gaps per question
        → Synthesizer can say "Cost information was not found"
```

### 2. Contradiction Resolution
```
Before: Conflicting info silently averaged
After:  Scratchpad flags contradictions
        → Additional searches targeted at resolution
        → Synthesizer explicitly addresses conflicts
```

### 3. Task Completion Awareness
```
Before: Fixed iteration count or heuristics
After:  Explicit completion criteria per question
        → Stop when all criteria met (or clearly unachievable)
```

### 4. Agent Coordination
```
Before: Agents work in isolation
After:  Agents leave notes for each other
        → Searcher notes: "Found conflicting dates, verifier should check"
        → Verifier notes: "Source A more reliable than B"
```

### 5. Debugging & Transparency
```
Before: Black box pipeline
After:  Full audit trail in scratchpad
        → See exactly what was found for each question
        → Understand why certain info is missing
```

## Example Scratchpad State

After searching for "FDA-approved OUD medications":

```json
{
  "request_id": "abc12345",
  "original_query": "What are FDA-approved OUD medications?",
  "questions": {
    "q1": {
      "text": "Which medications are FDA-approved for OUD?",
      "status": "answered",
      "confidence": 0.95,
      "findings": ["f1", "f2", "f3", "f4"],
      "gaps": []
    },
    "q2": {
      "text": "What are their mechanisms of action?",
      "status": "answered",
      "confidence": 0.90,
      "findings": ["f5", "f6", "f7", "f8"],
      "gaps": []
    },
    "q3": {
      "text": "Which have extended-release formulations?",
      "status": "partial",
      "confidence": 0.70,
      "findings": ["f9", "f10"],
      "gaps": ["Specific ER formulation for naltrexone unclear"]
    }
  },
  "findings": {
    "f1": {"content": "Methadone is FDA-approved", "confidence": 0.95, ...},
    "f2": {"content": "Buprenorphine is FDA-approved", "confidence": 0.95, ...},
    ...
  },
  "contradictions": [
    {
      "finding_1": "f9",
      "finding_2": "f11",
      "description": "Conflicting info about methadone ER availability",
      "resolved": true
    }
  ],
  "agent_notes": [
    {"agent": "verifier", "observation": "FDA.gov most reliable source"},
    {"agent": "searcher", "observation": "Limited 2024 data available"}
  ],
  "overall_confidence": 0.85,
  "is_complete": true
}
```

## Implementation Priority

1. **Phase 1**: Basic scratchpad with question tracking (1-2 days)
   - Create scratchpad at search start
   - Set mission from analyzer
   - Track findings per question
   - Basic completion checking

2. **Phase 2**: Agent context injection (1 day)
   - Generate context strings for each agent
   - Pass scratchpad state to synthesizer

3. **Phase 3**: Contradiction detection (1 day)
   - Track conflicting findings
   - Add to next actions queue

4. **Phase 4**: Full integration (1-2 days)
   - Agent notes
   - memOS persistence
   - Cleanup and optimization
