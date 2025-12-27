"""
AIME-Style Dynamic Planner with Dual Strategic/Tactical Outputs

Based on ByteDance AIME framework (arXiv 2507.11988):
- Continuous loop operation (not one-shot planning)
- Dual outputs: strategic (task hierarchy) + tactical (next action)
- Real-time adaptation based on execution feedback

Key formula: (L_{t+1}, g_{t+1}) = LLM(goal, L_t, H_t)
Where:
- L_{t+1}: Updated global task hierarchy
- g_{t+1}: Specific executable action for immediate dispatch
- H_t: History of past outcomes
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone
import aiohttp

logger = logging.getLogger("agentic.dynamic_planner")


class TaskStatus(str, Enum):
    """Status of a task in the hierarchy"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ActionType(str, Enum):
    """Types of tactical actions"""
    SEARCH = "search"
    SCRAPE = "scrape"
    ANALYZE = "analyze"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"
    REFINE = "refine"
    COMPLETE = "complete"


@dataclass
class TaskNode:
    """
    Hierarchical task node with AIME-style attributes.

    Supports:
    - Nested subtasks (tree structure)
    - Explicit completion criteria
    - Dependency tracking
    - Artifact references
    """
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    completion_criteria: str = ""
    subtasks: List['TaskNode'] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "completion_criteria": self.completion_criteria,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "artifacts": self.artifacts,
            "dependencies": self.dependencies,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskNode':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            completion_criteria=data.get("completion_criteria", ""),
            subtasks=[cls.from_dict(st) for st in data.get("subtasks", [])],
            artifacts=data.get("artifacts", []),
            dependencies=data.get("dependencies", []),
            notes=data.get("notes", "")
        )


@dataclass
class TacticalAction:
    """
    Immediate executable action (g_{t+1} in AIME formula).

    Dispatched to an actor for execution.
    """
    action_type: ActionType
    task_id: str  # Which task this action addresses
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "task_id": self.task_id,
            "description": self.description,
            "inputs": self.inputs,
            "required_tools": self.required_tools,
            "priority": self.priority
        }


@dataclass
class ExecutionResult:
    """Result from executing a tactical action"""
    task_id: str
    action_type: ActionType
    success: bool
    output: Any
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action_type": self.action_type.value,
            "success": self.success,
            "output": str(self.output)[:500] if self.output else None,
            "artifacts": self.artifacts,
            "error": self.error,
            "duration_ms": self.duration_ms
        }


@dataclass
class PlannerOutput:
    """
    Dual output from Dynamic Planner.

    AIME's key innovation: simultaneous strategic + tactical outputs.
    """
    strategic: List[TaskNode]  # Updated global task hierarchy
    tactical: Optional[TacticalAction]  # Next action to dispatch
    reasoning: str  # Planner's thinking process
    is_complete: bool = False  # Whether the goal is achieved
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategic": [t.to_dict() for t in self.strategic],
            "tactical": self.tactical.to_dict() if self.tactical else None,
            "reasoning": self.reasoning,
            "is_complete": self.is_complete,
            "confidence": self.confidence
        }


class DynamicPlanner:
    """
    AIME-Style Dynamic Planner.

    Unlike one-shot planners, operates in a continuous loop:
    1. Receives feedback from executed actions
    2. Updates strategic understanding (task hierarchy)
    3. Dispatches next tactical action

    Key features:
    - Dual outputs (strategic + tactical)
    - Real-time adaptation
    - Hierarchical task decomposition
    - Dependency tracking
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        planning_model: str = "qwen3:8b",
        max_depth: int = 3,
        max_tasks_per_level: int = 5
    ):
        self.ollama_url = ollama_url
        self.planning_model = planning_model
        self.max_depth = max_depth
        self.max_tasks_per_level = max_tasks_per_level

        # State
        self.task_hierarchy: List[TaskNode] = []
        self.execution_history: List[ExecutionResult] = []
        self.iteration_count: int = 0

        # Stats
        self.stats = {
            "planning_iterations": 0,
            "tasks_created": 0,
            "tasks_completed": 0,
            "actions_dispatched": 0,
            "replanning_events": 0
        }

    async def initial_decomposition(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PlannerOutput:
        """
        Initial task decomposition for a new goal.

        Creates the first strategic plan and dispatches first action.
        """
        logger.info(f"Initial decomposition for goal: {goal[:100]}...")

        prompt = self._build_initial_prompt(goal, context)

        try:
            response = await self._call_ollama(prompt)
            output = self._parse_planner_output(response, goal)

            self.task_hierarchy = output.strategic
            self.stats["planning_iterations"] += 1
            self.stats["tasks_created"] += self._count_tasks(output.strategic)

            if output.tactical:
                self.stats["actions_dispatched"] += 1

            logger.info(f"Created {len(output.strategic)} top-level tasks, "
                       f"dispatching action: {output.tactical.action_type.value if output.tactical else 'none'}")

            return output

        except Exception as e:
            logger.error(f"Initial decomposition failed: {e}")
            # Fallback to simple decomposition
            return self._fallback_decomposition(goal)

    async def plan_iteration(
        self,
        goal: str,
        execution_result: ExecutionResult
    ) -> PlannerOutput:
        """
        Continuous planning iteration after action execution.

        Updates strategic plan based on result and dispatches next action.

        Args:
            goal: Original goal
            execution_result: Result from last executed action

        Returns:
            PlannerOutput with updated hierarchy and next action
        """
        self.iteration_count += 1
        self.execution_history.append(execution_result)

        # Update task status based on result
        self._update_task_status(execution_result)

        logger.info(f"Planning iteration {self.iteration_count}: "
                   f"task={execution_result.task_id}, success={execution_result.success}")

        # Check if goal is achieved
        if self._is_goal_achieved():
            return PlannerOutput(
                strategic=self.task_hierarchy,
                tactical=None,
                reasoning="Goal achieved - all tasks completed",
                is_complete=True,
                confidence=self._calculate_confidence()
            )

        # Build planning prompt with history
        prompt = self._build_iteration_prompt(goal, execution_result)

        try:
            response = await self._call_ollama(prompt)
            output = self._parse_planner_output(response, goal)

            # Merge strategic updates
            self._merge_strategic_updates(output.strategic)

            self.stats["planning_iterations"] += 1
            if output.tactical:
                self.stats["actions_dispatched"] += 1

            return PlannerOutput(
                strategic=self.task_hierarchy,
                tactical=output.tactical,
                reasoning=output.reasoning,
                is_complete=output.is_complete,
                confidence=output.confidence
            )

        except Exception as e:
            logger.error(f"Planning iteration failed: {e}")
            return self._fallback_next_action(goal)

    def update_progress(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        artifacts: Optional[List[str]] = None
    ):
        """
        Update progress for a task (called by agents via Progress Update tool).

        This is the AIME proactive progress reporting mechanism.
        """
        task = self._find_task(task_id)
        if task:
            task.status = status
            task.notes = message
            if artifacts:
                task.artifacts.extend(artifacts)
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now(timezone.utc)
                self.stats["tasks_completed"] += 1

            logger.debug(f"Progress update: {task_id} -> {status.value}: {message[:50]}")

    def get_executable_tasks(self) -> List[TaskNode]:
        """
        Get tasks that are ready for execution.

        A task is executable if:
        - Status is PENDING
        - All dependencies are COMPLETED
        - No blocking subtasks
        """
        executable = []
        self._collect_executable(self.task_hierarchy, executable)
        return executable

    def render_markdown(self) -> str:
        """
        Render task hierarchy as markdown for LLM context.

        AIME uses human-readable markdown format for progress lists.
        """
        lines = ["## Task Progress\n"]
        self._render_tasks_markdown(self.task_hierarchy, lines, indent=0)
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            **self.stats,
            "current_tasks": self._count_tasks(self.task_hierarchy),
            "completed_tasks": self._count_completed(self.task_hierarchy),
            "pending_tasks": self._count_pending(self.task_hierarchy),
            "execution_history_length": len(self.execution_history)
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_initial_prompt(self, goal: str, context: Optional[Dict]) -> str:
        """Build prompt for initial decomposition"""
        context_str = json.dumps(context) if context else "None"

        return f"""You are a Dynamic Planner for an AI research assistant.

GOAL: {goal}

CONTEXT: {context_str}

Decompose this goal into a hierarchical task plan. For each task:
1. Provide a clear description
2. Define completion criteria (how to know it's done)
3. Identify dependencies on other tasks
4. Break complex tasks into subtasks (max 3 levels deep)

Also determine the FIRST action to execute immediately.

Output as JSON:
{{
  "reasoning": "Your analysis of how to approach this goal",
  "tasks": [
    {{
      "id": "t1",
      "description": "First main task",
      "completion_criteria": "Specific criteria for completion",
      "dependencies": [],
      "subtasks": [
        {{
          "id": "t1.1",
          "description": "Subtask",
          "completion_criteria": "...",
          "dependencies": []
        }}
      ]
    }}
  ],
  "next_action": {{
    "action_type": "search|scrape|analyze|verify|synthesize",
    "task_id": "t1",
    "description": "What to do",
    "inputs": {{"queries": ["search query"]}}
  }},
  "confidence": 0.8
}}

JSON response:"""

    def _build_iteration_prompt(self, goal: str, result: ExecutionResult) -> str:
        """Build prompt for planning iteration"""
        current_progress = self.render_markdown()
        recent_history = self.execution_history[-5:]
        history_str = json.dumps([r.to_dict() for r in recent_history], indent=2)

        return f"""You are a Dynamic Planner. Update the plan based on new information.

GOAL: {goal}

CURRENT PROGRESS:
{current_progress}

LAST ACTION RESULT:
- Task: {result.task_id}
- Action: {result.action_type.value}
- Success: {result.success}
- Output: {str(result.output)[:500] if result.output else "None"}
- Error: {result.error or "None"}

RECENT HISTORY:
{history_str}

Based on this result:
1. Should any tasks be updated, added, or removed?
2. What is the next action to take?
3. Is the goal complete?

Output as JSON:
{{
  "reasoning": "Your analysis of the result and next steps",
  "task_updates": [
    {{"id": "t1", "status": "completed|in_progress|failed", "notes": "..."}}
  ],
  "new_tasks": [],
  "next_action": {{
    "action_type": "search|scrape|analyze|verify|synthesize|complete",
    "task_id": "...",
    "description": "...",
    "inputs": {{}}
  }},
  "is_complete": false,
  "confidence": 0.7
}}

JSON response:"""

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.planning_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048
                    }
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama API error: {resp.status}")
                data = await resp.json()
                return data.get("response", "")

    def _parse_planner_output(self, response: str, goal: str) -> PlannerOutput:
        """Parse LLM response into PlannerOutput"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Parse tasks
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(self._parse_task_node(t))

            # Parse next action
            tactical = None
            action_data = data.get("next_action")
            if action_data and action_data.get("action_type") != "complete":
                tactical = TacticalAction(
                    action_type=ActionType(action_data.get("action_type", "search")),
                    task_id=action_data.get("task_id", "t1"),
                    description=action_data.get("description", ""),
                    inputs=action_data.get("inputs", {}),
                    required_tools=action_data.get("required_tools", [])
                )

            return PlannerOutput(
                strategic=tasks if tasks else self.task_hierarchy,
                tactical=tactical,
                reasoning=data.get("reasoning", ""),
                is_complete=data.get("is_complete", False),
                confidence=data.get("confidence", 0.5)
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse planner output: {e}")
            return self._fallback_decomposition(goal)

    def _parse_task_node(self, data: Dict) -> TaskNode:
        """Parse a task node from JSON data"""
        subtasks = [self._parse_task_node(st) for st in data.get("subtasks", [])]

        return TaskNode(
            id=data.get("id", f"t{self.stats['tasks_created'] + 1}"),
            description=data.get("description", ""),
            completion_criteria=data.get("completion_criteria", ""),
            subtasks=subtasks,
            dependencies=data.get("dependencies", [])
        )

    def _fallback_decomposition(self, goal: str) -> PlannerOutput:
        """Fallback when parsing fails"""
        task = TaskNode(
            id="t1",
            description=f"Research: {goal}",
            completion_criteria="Find relevant information from multiple sources"
        )

        action = TacticalAction(
            action_type=ActionType.SEARCH,
            task_id="t1",
            description=f"Search for: {goal}",
            inputs={"queries": [goal]}
        )

        return PlannerOutput(
            strategic=[task],
            tactical=action,
            reasoning="Fallback to simple search due to parsing error",
            confidence=0.3
        )

    def _fallback_next_action(self, goal: str) -> PlannerOutput:
        """Fallback when iteration planning fails"""
        # Find first pending task
        pending = self.get_executable_tasks()
        if pending:
            task = pending[0]
            action = TacticalAction(
                action_type=ActionType.SEARCH,
                task_id=task.id,
                description=f"Continue: {task.description}",
                inputs={"queries": [task.description]}
            )
        else:
            action = TacticalAction(
                action_type=ActionType.SYNTHESIZE,
                task_id="final",
                description="Synthesize available findings",
                inputs={}
            )

        return PlannerOutput(
            strategic=self.task_hierarchy,
            tactical=action,
            reasoning="Fallback action due to planning error",
            confidence=0.3
        )

    def _update_task_status(self, result: ExecutionResult):
        """Update task status based on execution result"""
        task = self._find_task(result.task_id)
        if task:
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                task.artifacts.extend(result.artifacts)
                self.stats["tasks_completed"] += 1
            else:
                task.status = TaskStatus.FAILED
                task.notes = result.error or "Execution failed"

    def _merge_strategic_updates(self, new_tasks: List[TaskNode]):
        """Merge strategic updates into existing hierarchy"""
        # For now, just add new tasks
        # A more sophisticated version would diff and merge
        for new_task in new_tasks:
            existing = self._find_task(new_task.id)
            if not existing:
                self.task_hierarchy.append(new_task)
                self.stats["tasks_created"] += 1
                self.stats["replanning_events"] += 1

    def _find_task(self, task_id: str) -> Optional[TaskNode]:
        """Find a task by ID in the hierarchy"""
        def search(tasks: List[TaskNode]) -> Optional[TaskNode]:
            for task in tasks:
                if task.id == task_id:
                    return task
                found = search(task.subtasks)
                if found:
                    return found
            return None

        return search(self.task_hierarchy)

    def _collect_executable(self, tasks: List[TaskNode], result: List[TaskNode]):
        """Collect executable tasks recursively"""
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self._find_task(dep) and
                    self._find_task(dep).status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_met:
                    result.append(task)

            # Check subtasks
            self._collect_executable(task.subtasks, result)

    def _is_goal_achieved(self) -> bool:
        """Check if the goal is achieved (all tasks completed)"""
        def all_completed(tasks: List[TaskNode]) -> bool:
            for task in tasks:
                if task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                    return False
                if not all_completed(task.subtasks):
                    return False
            return True

        return len(self.task_hierarchy) > 0 and all_completed(self.task_hierarchy)

    def _calculate_confidence(self) -> float:
        """Calculate overall confidence based on completed tasks"""
        completed = self._count_completed(self.task_hierarchy)
        total = self._count_tasks(self.task_hierarchy)
        return completed / max(total, 1)

    def _count_tasks(self, tasks: List[TaskNode]) -> int:
        """Count total tasks recursively"""
        count = len(tasks)
        for task in tasks:
            count += self._count_tasks(task.subtasks)
        return count

    def _count_completed(self, tasks: List[TaskNode]) -> int:
        """Count completed tasks recursively"""
        count = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        for task in tasks:
            count += self._count_completed(task.subtasks)
        return count

    def _count_pending(self, tasks: List[TaskNode]) -> int:
        """Count pending tasks recursively"""
        count = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        for task in tasks:
            count += self._count_pending(task.subtasks)
        return count

    def _render_tasks_markdown(
        self,
        tasks: List[TaskNode],
        lines: List[str],
        indent: int = 0
    ):
        """Render tasks as markdown list"""
        for task in tasks:
            marker = "[x]" if task.status == TaskStatus.COMPLETED else \
                     "[~]" if task.status == TaskStatus.IN_PROGRESS else \
                     "[!]" if task.status == TaskStatus.FAILED else "[ ]"

            prefix = "  " * indent
            status_note = f" ({task.notes})" if task.notes else ""
            lines.append(f"{prefix}- {marker} **{task.id}**: {task.description}{status_note}")

            if task.completion_criteria:
                lines.append(f"{prefix}  *Criteria: {task.completion_criteria}*")

            if task.subtasks:
                self._render_tasks_markdown(task.subtasks, lines, indent + 1)


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

_planner_instance: Optional[DynamicPlanner] = None


def get_dynamic_planner(
    ollama_url: str = "http://localhost:11434",
    planning_model: str = "qwen3:8b"
) -> DynamicPlanner:
    """Get or create the singleton DynamicPlanner instance"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = DynamicPlanner(
            ollama_url=ollama_url,
            planning_model=planning_model
        )
    return _planner_instance


def reset_planner():
    """Reset the planner for a new goal"""
    global _planner_instance
    _planner_instance = None
