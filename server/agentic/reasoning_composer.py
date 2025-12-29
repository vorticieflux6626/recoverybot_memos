"""
Phase 5: Self-Discover Reasoning Composer

Compose task-specific reasoning strategies from atomic modules.
Based on Self-Discover (NeurIPS 2024, Google DeepMind) - 87.5% correct structures.

Key Features:
- SELECT: Choose relevant reasoning modules for task
- ADAPT: Make modules task-specific
- IMPLEMENT: Structure as executable plan

Meta-actions allow LLM to compose custom reasoning strategies
from a library of atomic reasoning modules.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)


class ReasoningModule(Enum):
    """Atomic reasoning modules (from Self-Discover paper)"""

    # Critical Analysis
    CRITICAL_THINKING = "critical_thinking"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    BIAS_DETECTION = "bias_detection"

    # Problem Decomposition
    STEP_BY_STEP = "step_by_step"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    ABSTRACTION = "abstraction"

    # Comparison & Contrast
    COMPARE_CONTRAST = "compare_contrast"
    PROS_CONS = "pros_cons"
    TRADEOFF_ANALYSIS = "tradeoff_analysis"

    # Root Cause Analysis
    ROOT_CAUSE = "root_cause"
    FIVE_WHYS = "five_whys"
    FAULT_TREE = "fault_tree"

    # Synthesis
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    MULTI_SOURCE_INTEGRATION = "multi_source_integration"
    CONFIDENCE_AGGREGATION = "confidence_aggregation"

    # Verification
    CROSS_REFERENCE = "cross_reference"
    FACT_CHECKING = "fact_checking"
    CONSISTENCY_CHECK = "consistency_check"

    # Planning
    GOAL_DECOMPOSITION = "goal_decomposition"
    RESOURCE_ALLOCATION = "resource_allocation"
    CONTINGENCY_PLANNING = "contingency_planning"

    # Domain-Specific
    TECHNICAL_DEBUGGING = "technical_debugging"
    SCIENTIFIC_METHOD = "scientific_method"
    LEGAL_REASONING = "legal_reasoning"


@dataclass
class ModuleDefinition:
    """Definition of a reasoning module"""
    module: ReasoningModule
    name: str
    description: str
    prompt_template: str
    applicable_to: List[str]  # Task types this applies to
    requires: List[ReasoningModule] = field(default_factory=list)  # Dependencies
    incompatible_with: List[ReasoningModule] = field(default_factory=list)


@dataclass
class ComposedStrategy:
    """A composed reasoning strategy for a specific task"""
    task_description: str
    selected_modules: List[ReasoningModule]
    adapted_prompts: Dict[ReasoningModule, str]
    execution_order: List[ReasoningModule]
    expected_outputs: Dict[ReasoningModule, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "selected_modules": [m.value for m in self.selected_modules],
            "adapted_prompts": {m.value: p for m, p in self.adapted_prompts.items()},
            "execution_order": [m.value for m in self.execution_order],
            "expected_outputs": {m.value: o for m, o in self.expected_outputs.items()},
            "created_at": self.created_at.isoformat()
        }


class ReasoningComposer:
    """
    Self-Discover style reasoning composition.

    Composes task-specific reasoning strategies from atomic modules:
    1. SELECT: Choose relevant modules for the task
    2. ADAPT: Make modules task-specific
    3. IMPLEMENT: Structure as executable plan

    Research basis (Google DeepMind, NeurIPS 2024):
    - 87.5% correct structure composition
    - Outperforms CoT, ToT on complex reasoning
    - Transferable across task types
    """

    # Definitions for all reasoning modules
    MODULE_DEFINITIONS: Dict[ReasoningModule, ModuleDefinition] = {
        ReasoningModule.CRITICAL_THINKING: ModuleDefinition(
            module=ReasoningModule.CRITICAL_THINKING,
            name="Critical Thinking",
            description="Analyze arguments, identify assumptions, evaluate evidence strength",
            prompt_template="""Apply critical thinking to analyze:

1. **Identify Claims**: What specific claims are being made?
2. **Find Assumptions**: What unstated assumptions underlie these claims?
3. **Evaluate Evidence**: How strong is the supporting evidence?
4. **Check Logic**: Are the logical connections valid?
5. **Consider Alternatives**: What alternative interpretations exist?

For: {task_context}""",
            applicable_to=["analysis", "verification", "evaluation"]
        ),

        ReasoningModule.EVIDENCE_EVALUATION: ModuleDefinition(
            module=ReasoningModule.EVIDENCE_EVALUATION,
            name="Evidence Evaluation",
            description="Assess quality, relevance, and reliability of evidence",
            prompt_template="""Evaluate the evidence:

For each piece of evidence:
- **Source Quality**: Is the source authoritative and reliable?
- **Relevance**: How directly does this address the question?
- **Recency**: Is this information current?
- **Corroboration**: Is it supported by other sources?

Evidence to evaluate: {evidence}
Question: {task_context}""",
            applicable_to=["verification", "research", "fact_checking"]
        ),

        ReasoningModule.STEP_BY_STEP: ModuleDefinition(
            module=ReasoningModule.STEP_BY_STEP,
            name="Step-by-Step Reasoning",
            description="Break down complex problems into sequential steps",
            prompt_template="""Break this down step by step:

Problem: {task_context}

Step 1: [First action or consideration]
Step 2: [Next logical step]
...
Final Step: [Conclusion or solution]

For each step, explain:
- What are we doing?
- Why is this necessary?
- What do we learn?""",
            applicable_to=["problem_solving", "how_to", "debugging", "planning"]
        ),

        ReasoningModule.DIVIDE_AND_CONQUER: ModuleDefinition(
            module=ReasoningModule.DIVIDE_AND_CONQUER,
            name="Divide and Conquer",
            description="Split complex problems into manageable sub-problems",
            prompt_template="""Divide this problem into sub-problems:

Main Problem: {task_context}

Sub-problems:
1. [First sub-problem]
   - Scope: ...
   - Dependencies: ...

2. [Second sub-problem]
   - Scope: ...
   - Dependencies: ...

...

Integration Strategy: How do sub-solutions combine?""",
            applicable_to=["complex_problems", "research", "engineering"]
        ),

        ReasoningModule.COMPARE_CONTRAST: ModuleDefinition(
            module=ReasoningModule.COMPARE_CONTRAST,
            name="Compare and Contrast",
            description="Systematic comparison of options or viewpoints",
            prompt_template="""Compare and contrast:

Items: {items_to_compare}
Context: {task_context}

| Criterion | Option A | Option B | ... |
|-----------|----------|----------|-----|
| [Criterion 1] | ... | ... | ... |

Key Similarities:
- ...

Key Differences:
- ...

Implications of differences:
- ...""",
            applicable_to=["comparison", "decision_making", "evaluation"]
        ),

        ReasoningModule.ROOT_CAUSE: ModuleDefinition(
            module=ReasoningModule.ROOT_CAUSE,
            name="Root Cause Analysis",
            description="Identify underlying causes of problems",
            prompt_template="""Perform root cause analysis:

Symptom/Problem: {task_context}

1. **Immediate Cause**: What directly caused this?
2. **Contributing Factors**: What made it possible/likely?
3. **Root Cause**: What fundamental issue enabled this chain?

Causal Chain:
Root Cause → Contributing Factor → Immediate Cause → Symptom

Verification: How do we confirm this is the true root cause?""",
            applicable_to=["debugging", "troubleshooting", "problem_solving"]
        ),

        ReasoningModule.FIVE_WHYS: ModuleDefinition(
            module=ReasoningModule.FIVE_WHYS,
            name="Five Whys",
            description="Iterative questioning to reach root cause",
            prompt_template="""Apply the Five Whys technique:

Problem: {task_context}

Why 1: Why did this happen?
→ [Answer]

Why 2: Why did [Answer 1] happen?
→ [Answer]

Why 3: Why did [Answer 2] happen?
→ [Answer]

Why 4: Why did [Answer 3] happen?
→ [Answer]

Why 5: Why did [Answer 4] happen?
→ [Root Cause]""",
            applicable_to=["debugging", "troubleshooting", "quality_improvement"]
        ),

        ReasoningModule.EVIDENCE_SYNTHESIS: ModuleDefinition(
            module=ReasoningModule.EVIDENCE_SYNTHESIS,
            name="Evidence Synthesis",
            description="Combine multiple pieces of evidence into coherent conclusions",
            prompt_template="""Synthesize the available evidence:

Evidence pieces:
{evidence_list}

Question: {task_context}

Synthesis:
1. **Converging Evidence**: What do multiple sources agree on?
2. **Unique Contributions**: What does each source uniquely add?
3. **Gaps**: What remains unknown?
4. **Conflicts**: Where do sources disagree?
5. **Integrated Conclusion**: Based on the full evidence...""",
            applicable_to=["research", "synthesis", "analysis"]
        ),

        ReasoningModule.CROSS_REFERENCE: ModuleDefinition(
            module=ReasoningModule.CROSS_REFERENCE,
            name="Cross-Reference Verification",
            description="Verify claims against multiple independent sources",
            prompt_template="""Cross-reference and verify:

Claim to verify: {claim}
Context: {task_context}

Source 1: [What does it say?]
Source 2: [What does it say?]
Source 3: [What does it say?]

Agreement Matrix:
| Claim Aspect | S1 | S2 | S3 | Consensus |
|--------------|----|----|----|----|

Verification Conclusion:
- Confirmed: ...
- Unconfirmed: ...
- Contradicted: ...""",
            applicable_to=["verification", "fact_checking", "research"]
        ),

        ReasoningModule.TECHNICAL_DEBUGGING: ModuleDefinition(
            module=ReasoningModule.TECHNICAL_DEBUGGING,
            name="Technical Debugging",
            description="Systematic approach to diagnosing technical issues",
            prompt_template="""Debug this technical issue:

Problem: {task_context}
Error/Symptom: {error_info}

1. **Reproduce**: Can we consistently reproduce this?
2. **Isolate**: What components are involved?
3. **Hypothesize**: What could cause this?
   - Hypothesis A: ...
   - Hypothesis B: ...
4. **Test**: How do we test each hypothesis?
5. **Fix**: What's the solution?
6. **Verify**: How do we confirm it's fixed?""",
            applicable_to=["debugging", "troubleshooting", "technical"]
        ),

        ReasoningModule.SCIENTIFIC_METHOD: ModuleDefinition(
            module=ReasoningModule.SCIENTIFIC_METHOD,
            name="Scientific Method",
            description="Hypothesis-driven inquiry and testing",
            prompt_template="""Apply scientific method:

Question: {task_context}

1. **Observation**: What have we observed?
2. **Hypothesis**: What explanation might account for this?
3. **Prediction**: If the hypothesis is true, what should we see?
4. **Experiment/Test**: How can we test this?
5. **Analysis**: What do the results tell us?
6. **Conclusion**: What can we conclude?""",
            applicable_to=["research", "analysis", "investigation"]
        ),

        ReasoningModule.GOAL_DECOMPOSITION: ModuleDefinition(
            module=ReasoningModule.GOAL_DECOMPOSITION,
            name="Goal Decomposition",
            description="Break down goals into actionable sub-goals",
            prompt_template="""Decompose this goal:

Main Goal: {task_context}

Sub-goals (in priority order):
1. [Sub-goal 1]
   - Success criteria: ...
   - Dependencies: ...
   - Resources needed: ...

2. [Sub-goal 2]
   - Success criteria: ...
   - Dependencies: ...
   - Resources needed: ...

...

Milestone checkpoints:
- Checkpoint 1: When [sub-goal 1] complete
- ...""",
            applicable_to=["planning", "project_management", "complex_tasks"]
        ),
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:8b"
    ):
        self.ollama_url = ollama_url
        self.model = model
        self._stats = {
            "compositions": 0,
            "modules_selected": 0,
            "adaptations": 0
        }

    async def _llm_call(self, prompt: str) -> str:
        """Make LLM call"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_ctx": 8192
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def select_modules(
        self,
        task: str,
        max_modules: int = 4
    ) -> List[ReasoningModule]:
        """
        SELECT: Choose relevant reasoning modules for the task.
        """
        # Get module descriptions
        module_list = "\n".join([
            f"- {m.value}: {d.description}"
            for m, d in self.MODULE_DEFINITIONS.items()
        ])

        prompt = f"""Given this task, select the most relevant reasoning modules.

Task: {task}

Available modules:
{module_list}

Select up to {max_modules} modules that would be most helpful for this task.
Consider:
1. Task type (analysis, problem-solving, research, etc.)
2. Complexity level
3. Whether verification is needed
4. Whether comparison is involved

Return ONLY a JSON array of module names, e.g.: ["step_by_step", "root_cause", "evidence_synthesis"]
No explanation, just the JSON array."""

        response = await self._llm_call(prompt)

        # Parse response
        try:
            # Find JSON array in response
            import re
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                module_names = json.loads(match.group())
                selected = []
                for name in module_names[:max_modules]:
                    try:
                        selected.append(ReasoningModule(name))
                    except ValueError:
                        continue
                self._stats["modules_selected"] += len(selected)
                return selected
        except json.JSONDecodeError:
            pass

        # Fallback: return default modules based on keywords
        return self._keyword_select(task, max_modules)

    def _keyword_select(self, task: str, max_modules: int) -> List[ReasoningModule]:
        """Fallback keyword-based module selection"""
        task_lower = task.lower()
        selected = []

        keyword_map = {
            "debug": [ReasoningModule.TECHNICAL_DEBUGGING, ReasoningModule.ROOT_CAUSE],
            "troubleshoot": [ReasoningModule.FIVE_WHYS, ReasoningModule.ROOT_CAUSE],
            "compare": [ReasoningModule.COMPARE_CONTRAST],
            "verify": [ReasoningModule.CROSS_REFERENCE, ReasoningModule.EVIDENCE_EVALUATION],
            "research": [ReasoningModule.EVIDENCE_SYNTHESIS, ReasoningModule.CRITICAL_THINKING],
            "plan": [ReasoningModule.GOAL_DECOMPOSITION, ReasoningModule.STEP_BY_STEP],
            "how to": [ReasoningModule.STEP_BY_STEP],
            "error": [ReasoningModule.TECHNICAL_DEBUGGING, ReasoningModule.ROOT_CAUSE],
            "alarm": [ReasoningModule.TECHNICAL_DEBUGGING, ReasoningModule.FIVE_WHYS]
        }

        for keyword, modules in keyword_map.items():
            if keyword in task_lower:
                for m in modules:
                    if m not in selected:
                        selected.append(m)

        # Always include step-by-step as fallback
        if not selected:
            selected = [ReasoningModule.STEP_BY_STEP]

        return selected[:max_modules]

    async def adapt_modules(
        self,
        task: str,
        modules: List[ReasoningModule]
    ) -> Dict[ReasoningModule, str]:
        """
        ADAPT: Make modules task-specific.
        """
        adapted = {}
        self._stats["adaptations"] += len(modules)

        for module in modules:
            definition = self.MODULE_DEFINITIONS.get(module)
            if not definition:
                continue

            # Simple adaptation: fill in task context
            adapted_prompt = definition.prompt_template.replace("{task_context}", task)

            # For technical tasks, add error info placeholder
            if "{error_info}" in adapted_prompt:
                adapted_prompt = adapted_prompt.replace("{error_info}", "[See problem description above]")

            # For evidence tasks, add placeholder
            if "{evidence}" in adapted_prompt or "{evidence_list}" in adapted_prompt:
                adapted_prompt = adapted_prompt.replace("{evidence}", "[Retrieved evidence]")
                adapted_prompt = adapted_prompt.replace("{evidence_list}", "[Retrieved evidence list]")

            adapted[module] = adapted_prompt

        return adapted

    async def implement_strategy(
        self,
        task: str,
        modules: List[ReasoningModule],
        adapted_prompts: Dict[ReasoningModule, str]
    ) -> ComposedStrategy:
        """
        IMPLEMENT: Structure as executable plan.
        """
        # Determine execution order based on dependencies
        execution_order = self._determine_order(modules)

        # Define expected outputs for each module
        expected_outputs = {}
        for module in modules:
            definition = self.MODULE_DEFINITIONS.get(module)
            if definition:
                expected_outputs[module] = f"Output from {definition.name}: structured analysis/results"

        return ComposedStrategy(
            task_description=task,
            selected_modules=modules,
            adapted_prompts=adapted_prompts,
            execution_order=execution_order,
            expected_outputs=expected_outputs
        )

    def _determine_order(self, modules: List[ReasoningModule]) -> List[ReasoningModule]:
        """Determine optimal execution order"""
        # Priority order (lower = execute first)
        priority = {
            # Analysis first
            ReasoningModule.CRITICAL_THINKING: 1,
            ReasoningModule.DIVIDE_AND_CONQUER: 1,
            ReasoningModule.GOAL_DECOMPOSITION: 1,

            # Investigation second
            ReasoningModule.ROOT_CAUSE: 2,
            ReasoningModule.FIVE_WHYS: 2,
            ReasoningModule.TECHNICAL_DEBUGGING: 2,
            ReasoningModule.SCIENTIFIC_METHOD: 2,

            # Evaluation third
            ReasoningModule.EVIDENCE_EVALUATION: 3,
            ReasoningModule.COMPARE_CONTRAST: 3,

            # Synthesis fourth
            ReasoningModule.EVIDENCE_SYNTHESIS: 4,
            ReasoningModule.STEP_BY_STEP: 4,

            # Verification last
            ReasoningModule.CROSS_REFERENCE: 5,
        }

        return sorted(modules, key=lambda m: priority.get(m, 3))

    async def compose_strategy(
        self,
        task: str,
        max_modules: int = 4
    ) -> ComposedStrategy:
        """
        Full Self-Discover composition: SELECT → ADAPT → IMPLEMENT

        Returns a complete reasoning strategy for the task.
        """
        self._stats["compositions"] += 1

        # SELECT
        modules = await self.select_modules(task, max_modules)
        logger.info(f"Selected modules: {[m.value for m in modules]}")

        # ADAPT
        adapted = await self.adapt_modules(task, modules)

        # IMPLEMENT
        strategy = await self.implement_strategy(task, modules, adapted)

        logger.info(f"Composed strategy with {len(modules)} modules for: {task[:50]}...")
        return strategy

    def get_execution_prompt(self, strategy: ComposedStrategy) -> str:
        """
        Generate a unified prompt for executing the strategy.
        """
        prompt = f"""Execute this reasoning strategy for the task:

TASK: {strategy.task_description}

REASONING MODULES (execute in order):

"""
        for i, module in enumerate(strategy.execution_order, 1):
            adapted_prompt = strategy.adapted_prompts.get(module, "")
            prompt += f"\n{'='*50}\n"
            prompt += f"STEP {i}: {module.value.replace('_', ' ').title()}\n"
            prompt += f"{'='*50}\n"
            prompt += f"{adapted_prompt}\n"

        prompt += f"""
{'='*50}
FINAL SYNTHESIS
{'='*50}
Combine insights from all reasoning steps into a coherent conclusion.
"""

        return prompt

    def get_stats(self) -> Dict[str, Any]:
        """Get composer statistics"""
        return {
            **self._stats,
            "available_modules": len(self.MODULE_DEFINITIONS)
        }


# Singleton instance
_reasoning_composer: Optional[ReasoningComposer] = None


def get_reasoning_composer(
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3:8b"
) -> ReasoningComposer:
    """Get or create singleton ReasoningComposer instance"""
    global _reasoning_composer
    if _reasoning_composer is None:
        _reasoning_composer = ReasoningComposer(ollama_url=ollama_url, model=model)
    return _reasoning_composer
