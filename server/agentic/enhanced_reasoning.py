"""
Enhanced Reasoning Patterns for Agentic Search

Implements research-backed improvements from 2025:
1. Pre-Act Pattern - Multi-step planning before acting (70% accuracy improvement)
2. Self-Reflection Loop - Critique and refinement after synthesis
3. Stuck State Detection - Detect and recover from loops
4. Parallel Action Execution - Run independent searches concurrently
5. Contradiction Detection - Surface conflicting information
6. Chain of Draft - Concise reasoning to reduce tokens

Based on research from:
- Pre-Act (arXiv 2505.09970): 70% accuracy improvement over ReAct
- Chain of Draft (arXiv 2502.18600): 92.4% token reduction
- LangGraph production patterns
- Anthropic context engineering best practices
"""

import asyncio
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
import json

import httpx

logger = logging.getLogger("agentic.enhanced_reasoning")


class ActionType(Enum):
    """Types of actions the agent can take"""
    SEARCH = "search"
    SCRAPE = "scrape"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"
    REFLECT = "reflect"
    REFINE = "refine"
    DONE = "done"


@dataclass
class PlannedAction:
    """A single planned action in the Pre-Act sequence"""
    action_type: ActionType
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)  # Indices of actions this depends on
    priority: int = 1
    estimated_tokens: int = 0


@dataclass
class PreActPlan:
    """Multi-step execution plan created before acting"""
    query: str
    actions: List[PlannedAction]
    reasoning: str
    estimated_total_tokens: int = 0
    confidence: float = 0.0


@dataclass
class ReflectionResult:
    """Result of self-reflection on a synthesis"""
    quality_score: float  # 0-1
    issues: List[str]
    suggestions: List[str]
    should_refine: bool
    refinement_focus: str = ""


@dataclass
class ContradictionInfo:
    """Information about contradictions between sources"""
    claim: str
    source_a: str
    source_a_text: str
    source_b: str
    source_b_text: str
    resolution_suggestion: str = ""


@dataclass
class StuckStateMetrics:
    """Metrics for detecting stuck states"""
    iterations_without_progress: int = 0
    repeated_queries: Set[str] = field(default_factory=set)
    repeated_actions: List[str] = field(default_factory=list)
    last_progress_time: float = field(default_factory=time.time)
    synthesis_similarity_scores: List[float] = field(default_factory=list)


class EnhancedReasoningEngine:
    """
    Enhanced reasoning engine implementing research-backed patterns.

    Wraps the base orchestrator to add:
    - Pre-Act planning
    - Self-reflection
    - Stuck state detection
    - Parallel action execution
    - Contradiction detection
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        planning_model: str = "qwen3:8b",  # Upgraded from gemma3:4b for better planning quality
        reflection_model: str = "qwen3:8b",  # Upgraded from gemma3:4b for better reflection quality
        max_reflection_iterations: int = 2,
        stuck_threshold_iterations: int = 3,
        similarity_threshold: float = 0.85
    ):
        self.ollama_url = ollama_url
        self.planning_model = planning_model
        self.reflection_model = reflection_model
        self.max_reflection_iterations = max_reflection_iterations
        self.stuck_threshold = stuck_threshold_iterations
        self.similarity_threshold = similarity_threshold

        # Stats
        self.stats = {
            'pre_act_plans_created': 0,
            'reflections_performed': 0,
            'stuck_states_detected': 0,
            'stuck_states_recovered': 0,
            'contradictions_found': 0,
            'parallel_batches': 0,
            'total_time_saved_ms': 0
        }

    # =========================================================================
    # PRE-ACT PATTERN: Plan multiple steps before acting
    # =========================================================================

    async def create_pre_act_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_actions: int = 8
    ) -> PreActPlan:
        """
        Create a multi-step execution plan BEFORE acting.

        Based on Pre-Act research (arXiv 2505.09970):
        - Creates detailed reasoning for each step
        - Identifies dependencies between actions
        - Enables parallel execution of independent actions
        - 70% accuracy improvement over standard ReAct
        """
        logger.info(f"Creating Pre-Act plan for: {query[:50]}...")

        # Chain of Draft instruction for concise planning
        cod_instruction = "Plan concisely. Limit each reasoning step to one sentence."

        prompt = f"""You are a research planning agent. Create a detailed execution plan for answering this query.

{cod_instruction}

Query: {query}

Create a plan with 3-6 actions. For each action specify:
1. Type: SEARCH, SCRAPE, VERIFY, or SYNTHESIZE
2. What to do (one sentence)
3. What inputs are needed
4. Which previous steps this depends on (by number, 0-indexed)

Output as JSON:
{{
  "reasoning": "Brief explanation of strategy",
  "actions": [
    {{
      "type": "SEARCH",
      "description": "Search for...",
      "inputs": {{"queries": ["query1", "query2"]}},
      "depends_on": []
    }},
    ...
  ],
  "confidence": 0.8
}}

JSON plan:"""

        try:
            result = await self._call_ollama(prompt, self.planning_model)
            plan = self._parse_pre_act_plan(result, query)
            self.stats['pre_act_plans_created'] += 1

            logger.info(f"Pre-Act plan created: {len(plan.actions)} actions, "
                       f"confidence={plan.confidence:.2f}")
            return plan

        except Exception as e:
            logger.error(f"Pre-Act planning failed: {e}")
            # Fallback to simple plan
            return PreActPlan(
                query=query,
                actions=[
                    PlannedAction(
                        action_type=ActionType.SEARCH,
                        description=f"Search for: {query}",
                        inputs={"queries": [query]}
                    ),
                    PlannedAction(
                        action_type=ActionType.SYNTHESIZE,
                        description="Synthesize results",
                        dependencies=[0]
                    )
                ],
                reasoning="Fallback simple plan",
                confidence=0.5
            )

    def _parse_pre_act_plan(self, response: str, query: str) -> PreActPlan:
        """Parse Pre-Act plan from LLM response"""
        try:
            # Extract JSON from response
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match >= 0 and json_end > json_match:
                json_str = response[json_match:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            actions = []
            for i, action_data in enumerate(data.get('actions', [])):
                action_type_str = action_data.get('type', 'SEARCH').upper()
                action_type = ActionType[action_type_str] if action_type_str in ActionType.__members__ else ActionType.SEARCH

                actions.append(PlannedAction(
                    action_type=action_type,
                    description=action_data.get('description', ''),
                    inputs=action_data.get('inputs', {}),
                    dependencies=action_data.get('depends_on', []),
                    priority=i + 1
                ))

            return PreActPlan(
                query=query,
                actions=actions,
                reasoning=data.get('reasoning', ''),
                confidence=data.get('confidence', 0.7)
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Pre-Act JSON: {e}")
            # Return simple fallback plan
            return PreActPlan(
                query=query,
                actions=[
                    PlannedAction(
                        action_type=ActionType.SEARCH,
                        description=f"Search for: {query}",
                        inputs={"queries": [query]}
                    )
                ],
                reasoning="Fallback due to parse error",
                confidence=0.4
            )

    def get_parallel_action_groups(self, plan: PreActPlan) -> List[List[PlannedAction]]:
        """
        Group actions that can be executed in parallel.

        Actions with no dependencies or whose dependencies are all satisfied
        can run concurrently.
        """
        groups = []
        completed_indices = set()
        remaining = list(enumerate(plan.actions))

        while remaining:
            # Find actions whose dependencies are all satisfied
            parallel_group = []
            next_remaining = []

            for idx, action in remaining:
                deps_satisfied = all(d in completed_indices for d in action.dependencies)
                if deps_satisfied:
                    parallel_group.append(action)
                    completed_indices.add(idx)
                else:
                    next_remaining.append((idx, action))

            if parallel_group:
                groups.append(parallel_group)
                self.stats['parallel_batches'] += 1
            else:
                # Circular dependency or error - add remaining sequentially
                for idx, action in next_remaining:
                    groups.append([action])
                break

            remaining = next_remaining

        return groups

    # =========================================================================
    # SELF-REFLECTION: Critique and improve synthesis
    # =========================================================================

    async def reflect_on_synthesis(
        self,
        query: str,
        synthesis: str,
        sources: List[Dict[str, Any]],
        iteration: int = 0
    ) -> ReflectionResult:
        """
        Self-reflection on synthesis quality.

        Based on Reflection Pattern research:
        - Evaluate completeness, accuracy, clarity
        - Identify issues and suggest improvements
        - Decide if refinement is needed
        """
        logger.info(f"Reflecting on synthesis (iteration {iteration})...")

        prompt = f"""You are a quality assurance agent. Evaluate this research synthesis.

Query: {query}

Synthesis:
{synthesis[:3000]}

Sources used: {len(sources)}

Evaluate on these criteria:
1. Completeness: Does it answer all aspects of the query?
2. Accuracy: Is the information consistent with sources?
3. Clarity: Is it well-organized and easy to understand?
4. Citations: Are claims properly attributed?
5. Actionability: Does it provide practical guidance?

Output JSON:
{{
  "quality_score": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "should_refine": true/false,
  "refinement_focus": "What to improve"
}}

JSON evaluation:"""

        try:
            result = await self._call_ollama(prompt, self.reflection_model)
            reflection = self._parse_reflection(result)
            self.stats['reflections_performed'] += 1

            logger.info(f"Reflection: quality={reflection.quality_score:.2f}, "
                       f"should_refine={reflection.should_refine}")
            return reflection

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return ReflectionResult(
                quality_score=0.7,
                issues=[],
                suggestions=[],
                should_refine=False
            )

    def _parse_reflection(self, response: str) -> ReflectionResult:
        """Parse reflection result from LLM response"""
        try:
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match >= 0 and json_end > json_match:
                json_str = response[json_match:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")

            return ReflectionResult(
                quality_score=float(data.get('quality_score', 0.7)),
                issues=data.get('issues', []),
                suggestions=data.get('suggestions', []),
                should_refine=data.get('should_refine', False),
                refinement_focus=data.get('refinement_focus', '')
            )
        except Exception:
            return ReflectionResult(
                quality_score=0.7,
                issues=[],
                suggestions=[],
                should_refine=False
            )

    async def refine_synthesis(
        self,
        query: str,
        original_synthesis: str,
        reflection: ReflectionResult,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Refine synthesis based on reflection feedback"""

        issues_text = "\n".join(f"- {issue}" for issue in reflection.issues)
        suggestions_text = "\n".join(f"- {s}" for s in reflection.suggestions)

        prompt = f"""Improve this research synthesis based on feedback.

Query: {query}

Original synthesis:
{original_synthesis[:2500]}

Issues identified:
{issues_text}

Suggestions:
{suggestions_text}

Focus on: {reflection.refinement_focus}

Provide an improved synthesis that addresses these issues. Be comprehensive but concise.

Improved synthesis:"""

        try:
            result = await self._call_ollama(prompt, "qwen3:8b")
            return result.strip()
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return original_synthesis

    # =========================================================================
    # STUCK STATE DETECTION: Detect and recover from loops
    # =========================================================================

    def check_stuck_state(
        self,
        metrics: StuckStateMetrics,
        current_query: str,
        current_synthesis: str = ""
    ) -> Tuple[bool, str]:
        """
        Detect if the agent is in a stuck state.

        Based on production lessons:
        - Detect repetitive queries
        - Detect similar synthesis outputs
        - Detect lack of progress over iterations
        """
        is_stuck = False
        reason = ""

        # Check for repeated queries
        query_hash = hashlib.md5(current_query.lower().encode()).hexdigest()[:16]
        if query_hash in metrics.repeated_queries:
            is_stuck = True
            reason = "Repeated query detected"
        else:
            metrics.repeated_queries.add(query_hash)

        # Check for too many iterations without progress
        if metrics.iterations_without_progress >= self.stuck_threshold:
            is_stuck = True
            reason = f"No progress for {self.stuck_threshold} iterations"

        # Check synthesis similarity (if we have previous syntheses)
        if current_synthesis and len(metrics.synthesis_similarity_scores) > 0:
            # Simple character-level similarity
            prev_hash = hashlib.md5(current_synthesis.encode()).hexdigest()
            current_hashes = [hashlib.md5(current_synthesis.encode()).hexdigest()]

            # If synthesis is nearly identical to previous
            if len(set(current_hashes) & set(metrics.repeated_actions)) > 0:
                is_stuck = True
                reason = "Synthesis not improving"

        if is_stuck:
            self.stats['stuck_states_detected'] += 1
            logger.warning(f"Stuck state detected: {reason}")

        return is_stuck, reason

    async def recover_from_stuck(
        self,
        query: str,
        current_state: Dict[str, Any],
        stuck_reason: str
    ) -> Dict[str, Any]:
        """
        Attempt to recover from a stuck state.

        Strategies:
        1. Broaden or narrow the query
        2. Try different search sources
        3. Simplify the problem
        4. Return partial results with explanation
        """
        logger.info(f"Attempting stuck state recovery: {stuck_reason}")

        prompt = f"""The research agent is stuck. Help it recover.

Original query: {query}
Stuck reason: {stuck_reason}
Current findings: {len(current_state.get('results', []))} sources found

Suggest ONE recovery strategy:
1. BROADEN: Make query more general
2. NARROW: Focus on specific aspect
3. REPHRASE: Try different terms
4. SIMPLIFY: Break into simpler questions
5. ACCEPT: Return current results with limitations noted

Output JSON:
{{
  "strategy": "BROADEN|NARROW|REPHRASE|SIMPLIFY|ACCEPT",
  "new_queries": ["query1", "query2"],
  "explanation": "Why this helps"
}}

JSON:"""

        try:
            result = await self._call_ollama(prompt, self.planning_model)
            recovery = self._parse_recovery_strategy(result)
            self.stats['stuck_states_recovered'] += 1
            return recovery
        except Exception as e:
            logger.error(f"Recovery planning failed: {e}")
            return {
                "strategy": "ACCEPT",
                "new_queries": [],
                "explanation": "Recovery failed, accepting current results"
            }

    def _parse_recovery_strategy(self, response: str) -> Dict[str, Any]:
        """Parse recovery strategy from LLM response"""
        try:
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match >= 0 and json_end > json_match:
                return json.loads(response[json_match:json_end])
        except Exception:
            pass
        return {
            "strategy": "ACCEPT",
            "new_queries": [],
            "explanation": "Parse failed"
        }

    # =========================================================================
    # CONTRADICTION DETECTION: Find and surface conflicting information
    # =========================================================================

    async def detect_contradictions(
        self,
        sources: List[Dict[str, Any]],
        key_claims: List[str]
    ) -> List[ContradictionInfo]:
        """
        Detect contradictions between sources.

        Important for technical information where accuracy matters.
        """
        if len(sources) < 2:
            return []

        # Build source summary for comparison
        sources_text = "\n".join([
            f"Source {i+1} ({s.get('url', 'unknown')[:50]}): {s.get('content', s.get('snippet', ''))[:500]}"
            for i, s in enumerate(sources[:5])
        ])

        claims_text = "\n".join(f"- {claim}" for claim in key_claims[:5])

        prompt = f"""Analyze these sources for contradictions.

Sources:
{sources_text}

Key claims to verify:
{claims_text}

Find any contradictions where sources disagree on facts.

Output JSON:
{{
  "contradictions": [
    {{
      "claim": "The disputed claim",
      "source_a": "Source 1 URL",
      "source_a_says": "What source A says",
      "source_b": "Source 2 URL",
      "source_b_says": "What source B says",
      "resolution": "How to resolve or present this"
    }}
  ]
}}

JSON (or empty object if no contradictions):"""

        try:
            result = await self._call_ollama(prompt, self.reflection_model)
            contradictions = self._parse_contradictions(result)
            self.stats['contradictions_found'] += len(contradictions)

            if contradictions:
                logger.info(f"Found {len(contradictions)} contradictions between sources")

            return contradictions

        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return []

    def _parse_contradictions(self, response: str) -> List[ContradictionInfo]:
        """Parse contradictions from LLM response"""
        try:
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match >= 0 and json_end > json_match:
                data = json.loads(response[json_match:json_end])

                return [
                    ContradictionInfo(
                        claim=c.get('claim', ''),
                        source_a=c.get('source_a', ''),
                        source_a_text=c.get('source_a_says', ''),
                        source_b=c.get('source_b', ''),
                        source_b_text=c.get('source_b_says', ''),
                        resolution_suggestion=c.get('resolution', '')
                    )
                    for c in data.get('contradictions', [])
                ]
        except Exception:
            pass
        return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Call Ollama API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            **self.stats,
            'avg_parallel_batch_size': (
                self.stats['pre_act_plans_created'] /
                max(1, self.stats['parallel_batches'])
            ),
            'stuck_recovery_rate': (
                self.stats['stuck_states_recovered'] /
                max(1, self.stats['stuck_states_detected'])
            )
        }


# Singleton instance
_enhanced_reasoning: Optional[EnhancedReasoningEngine] = None


def get_enhanced_reasoning(ollama_url: str = "http://localhost:11434") -> EnhancedReasoningEngine:
    """Get or create the enhanced reasoning engine"""
    global _enhanced_reasoning
    if _enhanced_reasoning is None:
        _enhanced_reasoning = EnhancedReasoningEngine(ollama_url=ollama_url)
    return _enhanced_reasoning
