"""
Dynamic Orchestrator with AIME-Style Task Planning

Integrates the DynamicPlanner for dual strategic/tactical outputs,
enabling hierarchical task management and adaptive replanning.

Key Features:
- Dual Output Planning: Strategic task hierarchy + Tactical next action
- Progress-Driven Replanning: Agents report progress, planner adapts
- Task Dependency Awareness: Respects subtask relationships
- Execution History Feedback: Uses past results to inform future plans

Based on AIME (ByteDance) framework research:
- Dynamic Planner creates and updates hierarchical task tree
- Progress Manager tracks completion as SSOT
- Agents use Progress Update tool to report status

Usage:
    orchestrator = DynamicOrchestrator(ollama_url="http://localhost:11434")
    await orchestrator.initialize()

    result = await orchestrator.execute(
        goal="Compare FastAPI vs Django for REST API development",
        max_iterations=10
    )
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import uuid

from .dynamic_planner import (
    DynamicPlanner,
    TaskNode,
    TaskStatus,
    ActionType,
    TacticalAction,
    ExecutionResult,
    PlannerOutput
)
from .progress_tools import (
    ProgressReporter,
    ProgressAggregator,
    ProgressStatus,
    ProgressUpdate
)
from .scratchpad import AgenticScratchpad, ScratchpadManager
from .searcher import SearcherAgent
from .synthesizer import SynthesizerAgent
from .analyzer import QueryAnalyzer
from .verifier import VerifierAgent
from .scraper import ContentScraper
from .content_cache import get_content_cache
from .events import EventEmitter, EventType, SearchEvent
from .models import SearchRequest, SearchResponse, SearchResultData, SearchMeta, ConfidenceLevel

logger = logging.getLogger("agentic.orchestrator_dynamic")


class DynamicOrchestrator:
    """
    Orchestrator using AIME-style DynamicPlanner for hierarchical task execution.

    This orchestrator differs from the standard one by:
    1. Using DynamicPlanner for task decomposition and replanning
    2. Tracking progress via task hierarchy (not just iteration count)
    3. Dispatching tactical actions one at a time with feedback
    4. Adapting plan based on execution results
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None
    ):
        self.ollama_url = ollama_url

        # Core agents
        self.analyzer = QueryAnalyzer(ollama_url=ollama_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=ollama_url)
        self.scraper = ContentScraper()

        # AIME-style components
        self.planner = DynamicPlanner(ollama_url=ollama_url)
        self.progress_aggregator = ProgressAggregator()

        # Scratchpad for shared working memory
        self.scratchpad_manager = ScratchpadManager(memory_service=memory_service)

        # Event emitter for SSE streaming
        self.event_emitter: Optional[EventEmitter] = None

        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_tasks_completed": 0,
            "total_replans": 0,
            "average_task_count": 0.0
        }

    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("DynamicOrchestrator initialized with AIME-style planning")

    def set_event_emitter(self, emitter: EventEmitter):
        """Set event emitter for SSE streaming"""
        self.event_emitter = emitter

    async def _emit_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        request_id: str = ""
    ):
        """Emit an event if emitter is set"""
        if self.event_emitter:
            await self.event_emitter.emit(SearchEvent(
                event_type=event_type,
                request_id=request_id or str(uuid.uuid4())[:8],
                data=data
            ))

    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute goal-directed search with AIME-style planning.

        This is the main entry point for dynamic orchestration.

        Args:
            goal: High-level goal/query to accomplish
            context: Optional context (conversation history, user info, etc.)
            max_iterations: Maximum planning iterations
            request_id: Optional request ID for tracking

        Returns:
            Dict with results, task hierarchy, and execution trace
        """
        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())

        logger.info(f"[{request_id}] Starting dynamic execution: {goal[:50]}...")

        # Create scratchpad for this execution
        scratchpad = self.scratchpad_manager.create(goal, request_id)

        # Create progress reporter connected to scratchpad
        async def on_progress(update: ProgressUpdate):
            await self.progress_aggregator.record_update(update)
            await self._emit_event(EventType.PROGRESS, update.to_dict(), request_id)

        progress_reporter = ProgressReporter(
            scratchpad=scratchpad,
            planner=self.planner,
            on_progress=on_progress
        )

        # Execution trace
        trace = []

        try:
            # PHASE 1: Initial Decomposition
            await self._emit_event(EventType.PLANNING, {"status": "decomposing", "goal": goal}, request_id)

            output = await self.planner.initial_decomposition(goal, context)

            # Sync task hierarchy to scratchpad
            scratchpad.set_task_hierarchy([t.to_dict() for t in output.strategic])

            trace.append({
                "phase": "decomposition",
                "tasks_created": len(self.planner.task_hierarchy),
                "first_action": output.tactical.action_type.value if output.tactical else None,
                "reasoning": output.reasoning[:200] if output.reasoning else None
            })

            logger.info(f"[{request_id}] Decomposed into {len(self.planner.task_hierarchy)} tasks")

            # PHASE 2: Iterative Execution Loop
            iteration = 0
            all_results = []

            while iteration < max_iterations and not output.is_complete:
                iteration += 1

                if not output.tactical:
                    logger.info(f"[{request_id}] No tactical action - checking completion")
                    break

                action = output.tactical
                logger.info(f"[{request_id}] Iteration {iteration}: {action.action_type.value} - {action.description[:50]}")

                await self._emit_event(EventType.AGENT_START, {
                    "agent": action.action_type.value,
                    "task_id": action.task_id,
                    "description": action.description,
                    "iteration": iteration
                }, request_id)

                # Report task started
                scratchpad.set_current_action(action.to_dict())
                await progress_reporter.report_started(action.task_id, action.description)

                # Execute the action
                action_start = time.time()
                try:
                    result = await self._execute_action(action, scratchpad, request_id)
                    duration_ms = int((time.time() - action_start) * 1000)

                    # Report completion
                    await progress_reporter.report_completed(
                        task_id=action.task_id,
                        output=result.output,
                        duration_ms=duration_ms,
                        artifacts=result.artifacts
                    )

                    all_results.append(result)

                    await self._emit_event(EventType.AGENT_COMPLETE, {
                        "agent": action.action_type.value,
                        "task_id": action.task_id,
                        "success": result.success,
                        "duration_ms": duration_ms
                    }, request_id)

                except Exception as e:
                    duration_ms = int((time.time() - action_start) * 1000)
                    logger.error(f"[{request_id}] Action failed: {e}")

                    result = ExecutionResult(
                        task_id=action.task_id,
                        action_type=action.action_type,
                        success=False,
                        output={"error": str(e)},
                        duration_ms=duration_ms
                    )

                    await progress_reporter.report_failed(
                        task_id=action.task_id,
                        reason=str(e),
                        error=e,
                        should_retry=True,
                        duration_ms=duration_ms
                    )

                trace.append({
                    "phase": "execution",
                    "iteration": iteration,
                    "action_type": action.action_type.value,
                    "task_id": action.task_id,
                    "success": result.success,
                    "duration_ms": result.duration_ms
                })

                # PHASE 3: Replan based on result
                output = await self.planner.plan_iteration(goal, result)

                # Update scratchpad with new hierarchy
                scratchpad.set_task_hierarchy([t.to_dict() for t in output.strategic])

                if output.reasoning:
                    trace.append({
                        "phase": "replan",
                        "iteration": iteration,
                        "is_complete": output.is_complete,
                        "confidence": output.confidence,
                        "reasoning": output.reasoning[:200]
                    })

                self._stats["total_replans"] += 1

            # PHASE 4: Final Synthesis
            await self._emit_event(EventType.SYNTHESIZING, {"status": "synthesizing"}, request_id)

            # Gather all findings for synthesis
            findings = list(scratchpad.findings.values())
            synthesis_content = await self._synthesize_results(goal, findings, all_results, scratchpad)

            # Calculate confidence
            task_progress = scratchpad.get_task_progress()
            confidence = output.confidence if output else task_progress["progress"]

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Update stats
            self._stats["total_executions"] += 1
            self._stats["successful_executions"] += 1
            self._stats["total_tasks_completed"] += task_progress.get("completed", 0)

            result = {
                "success": True,
                "goal": goal,
                "synthesized_result": synthesis_content,
                "confidence": confidence,
                "task_hierarchy": scratchpad.get_task_markdown(),
                "task_progress": task_progress,
                "sources": [f.source_url for f in findings],
                "execution_trace": trace,
                "meta": {
                    "request_id": request_id,
                    "iterations": iteration,
                    "tasks_total": task_progress.get("total", 0),
                    "tasks_completed": task_progress.get("completed", 0),
                    "execution_time_ms": execution_time_ms,
                    "planner_stats": self.planner.get_stats()
                }
            }

            await self._emit_event(EventType.COMPLETE, {
                "success": True,
                "confidence": confidence,
                "execution_time_ms": execution_time_ms
            }, request_id)

            logger.info(f"[{request_id}] Dynamic execution complete: "
                       f"{task_progress.get('completed', 0)}/{task_progress.get('total', 0)} tasks, "
                       f"{confidence:.0%} confidence, {execution_time_ms}ms")

            return result

        except Exception as e:
            logger.error(f"[{request_id}] Dynamic execution failed: {e}", exc_info=True)

            await self._emit_event(EventType.ERROR, {"error": str(e)}, request_id)

            return {
                "success": False,
                "goal": goal,
                "error": str(e),
                "execution_trace": trace,
                "meta": {
                    "request_id": request_id,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }
            }

    async def _execute_action(
        self,
        action: TacticalAction,
        scratchpad: AgenticScratchpad,
        request_id: str
    ) -> ExecutionResult:
        """
        Execute a tactical action based on its type.

        Maps action types to appropriate agent calls.
        """
        action_type = action.action_type
        inputs = action.inputs or {}
        start_time = time.time()

        try:
            if action_type == ActionType.SEARCH:
                # Web search
                query = inputs.get("query", action.description)
                queries = inputs.get("queries", [query])

                results = await self.searcher.search(queries)

                # Record in scratchpad
                for q in queries:
                    scratchpad.record_search(q)

                output = {
                    "results_count": len(results),
                    "results": [
                        {"title": r.title, "url": r.url, "snippet": r.snippet[:200]}
                        for r in results[:5]
                    ]
                }

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=len(results) > 0,
                    output=output,
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.SCRAPE:
                # Content scraping
                url = inputs.get("url", "")
                urls = inputs.get("urls", [url] if url else [])

                all_content = []
                for u in urls[:3]:  # Limit to 3 URLs
                    try:
                        content = await self.scraper.scrape_url(u)
                        if content:
                            all_content.append({
                                "url": u,
                                "content": content[:2000],
                                "length": len(content)
                            })
                            scratchpad.record_scrape(u)
                    except Exception as e:
                        logger.warning(f"Failed to scrape {u}: {e}")

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=len(all_content) > 0,
                    output={"scraped_urls": len(all_content), "content": all_content},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.ANALYZE:
                # Content analysis with LLM
                content = inputs.get("content", "")
                analysis_query = inputs.get("query", action.description)

                if not content:
                    # Use recent findings from scratchpad
                    recent_findings = list(scratchpad.findings.values())[-5:]
                    content = "\n".join([f.content for f in recent_findings])

                analysis = await self.analyzer.analyze_content(content, analysis_query)

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=bool(analysis),
                    output={"analysis": analysis},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.VERIFY:
                # Fact verification
                claims = inputs.get("claims", [])
                if not claims:
                    # Extract from recent findings
                    recent_findings = list(scratchpad.findings.values())[-10:]
                    claims = [f.content for f in recent_findings]

                verified = await self.verifier.verify_claims(claims)

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=True,
                    output={"verified_claims": verified},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.SYNTHESIZE:
                # Content synthesis
                findings = list(scratchpad.findings.values())
                goal = inputs.get("goal", scratchpad.original_query)

                synthesis = await self.synthesizer.synthesize(
                    goal,
                    [],  # Raw results handled via findings
                    scratchpad,
                    {}
                )

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=bool(synthesis),
                    output={"synthesis": synthesis[:1000] if synthesis else ""},
                    artifacts=["synthesis_result"],
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.DELEGATE:
                # Delegate to specialized agent (for future ActorFactory)
                agent_type = inputs.get("agent_type", "general")
                logger.info(f"Delegation requested to: {agent_type} (not yet implemented)")

                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=True,
                    output={"delegation": agent_type, "status": "placeholder"},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            elif action_type == ActionType.REPORT:
                # Progress reporting (handled by reporter, just acknowledge)
                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=True,
                    output={"report": action.description},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            else:
                logger.warning(f"Unknown action type: {action_type}")
                return ExecutionResult(
                    task_id=action.task_id,
                    action_type=action_type,
                    success=False,
                    output={"error": f"Unknown action type: {action_type}"},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return ExecutionResult(
                task_id=action.task_id,
                action_type=action_type,
                success=False,
                output={"error": str(e)},
                duration_ms=int((time.time() - start_time) * 1000)
            )

    async def _synthesize_results(
        self,
        goal: str,
        findings: List[Any],
        execution_results: List[ExecutionResult],
        scratchpad: AgenticScratchpad
    ) -> str:
        """Synthesize all results into a final answer"""
        # Collect successful outputs
        successful_outputs = []
        for result in execution_results:
            if result.success and result.output:
                if isinstance(result.output, dict):
                    # Extract meaningful content
                    if "synthesis" in result.output:
                        successful_outputs.append(result.output["synthesis"])
                    elif "analysis" in result.output:
                        successful_outputs.append(result.output["analysis"])
                    elif "results" in result.output:
                        for r in result.output.get("results", []):
                            if isinstance(r, dict):
                                successful_outputs.append(f"{r.get('title', '')}: {r.get('snippet', '')}")

        # Add findings content
        for finding in findings:
            successful_outputs.append(f"[{finding.source_title or finding.source_url}]: {finding.content}")

        if not successful_outputs:
            return "No results were gathered during execution."

        # Use synthesizer for final combination
        combined = "\n\n".join(successful_outputs[:20])  # Limit to prevent context overflow

        try:
            synthesis = await self.synthesizer.synthesize(
                goal,
                [],
                scratchpad,
                {"raw_content": combined}
            )
            return synthesis
        except Exception as e:
            logger.warning(f"Synthesis failed, returning combined results: {e}")
            return combined[:5000]

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Standard search interface compatible with main orchestrator.

        Wraps execute() with SearchRequest/SearchResponse types.
        """
        result = await self.execute(
            goal=request.query,
            context=request.context,
            max_iterations=request.max_iterations
        )

        return SearchResponse(
            success=result.get("success", False),
            data=SearchResultData(
                synthesized_context=result.get("synthesized_result", ""),
                sources=[{"url": s} for s in result.get("sources", [])],
                search_queries=[],
                search_trace=result.get("execution_trace", []),
                confidence_score=result.get("confidence", 0.0),
                confidence_level=ConfidenceLevel.MEDIUM if result.get("confidence", 0) >= 0.5 else ConfidenceLevel.LOW
            ),
            meta=SearchMeta(
                request_id=result.get("meta", {}).get("request_id", ""),
                iterations=result.get("meta", {}).get("iterations", 0),
                queries_executed=result.get("meta", {}).get("tasks_completed", 0),
                sources_consulted=len(result.get("sources", [])),
                execution_time_ms=result.get("meta", {}).get("execution_time_ms", 0),
                cache_hit=False
            ),
            errors=[result.get("error")] if result.get("error") else []
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self._stats,
            "planner_stats": self.planner.get_stats(),
            "progress_summary": self.progress_aggregator.get_summary()
        }


# Factory function for easy creation
def create_dynamic_orchestrator(
    ollama_url: str = "http://localhost:11434",
    brave_api_key: Optional[str] = None
) -> DynamicOrchestrator:
    """Create a DynamicOrchestrator instance"""
    return DynamicOrchestrator(
        ollama_url=ollama_url,
        brave_api_key=brave_api_key
    )
