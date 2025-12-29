"""
Multi-Agent Orchestration System

Implements a sophisticated agentic workflow with:
- Smart model selection based on VRAM, capabilities, and task requirements
- Sub-agent spawning for parallel task execution
- VRAM-aware parallel/sequential scheduling
- Iterative refinement until answer found or leads exhausted

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    SMART PLANNER (reasoning model)              │
│  Analyzes query → Decomposes into sub-tasks → Creates plan      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SUB-AGENT POOL                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │            │
│  │ (fast)  │  │ (fast)  │  │ (fast)  │  │ (fast)  │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│       ↓ parallel if VRAM allows, else sequential ↓              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SMART EVALUATOR (reasoning model)               │
│  Reviews results → Determines if sufficient → Plans next steps   │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [SUFFICIENT]                    [NEEDS MORE WORK]           │
│     → Synthesize                    → Refine queries            │
│     → Return answer                 → Spawn more agents         │
│                                     → Loop back                  │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import time
import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Tuple
import httpx

# Import static fallback specs
from .model_specs import OLLAMA_MODEL_SPECS, SPEED_TIERS

# Database specs fetching (optional, for database-backed model selection)
_DB_SPECS_CACHE: Dict[str, Dict[str, Any]] = {}
_DB_SPECS_CACHE_TIME: float = 0
_DB_SPECS_CACHE_TTL: int = 300  # 5 minutes


async def fetch_model_specs_from_db() -> Dict[str, Dict[str, Any]]:
    """
    Fetch model specs from the memOS database API.
    Falls back to static specs if database is unavailable.
    """
    global _DB_SPECS_CACHE, _DB_SPECS_CACHE_TIME

    now = time.time()
    if now - _DB_SPECS_CACHE_TIME < _DB_SPECS_CACHE_TTL and _DB_SPECS_CACHE:
        return _DB_SPECS_CACHE

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to fetch from local memOS API
            response = await client.get("http://localhost:8001/api/v1/models/specs")
            if response.status_code == 200:
                specs_list = response.json()
                # Convert list to dict keyed by model_name
                specs_dict = {}
                for spec in specs_list:
                    model_name = spec.get("model_name")
                    if model_name:
                        specs_dict[model_name] = {
                            "vram_min_gb": spec.get("vram_min_gb", 4.0),
                            "context_window": spec.get("context_window", 4096),
                            "capabilities": spec.get("capabilities", ["chat"]),
                            "specialization": spec.get("specialization", "general_purpose"),
                            "speed_tier": spec.get("speed_tier", "medium"),
                            "parameter_count": spec.get("parameter_count"),
                            "file_size_gb": spec.get("file_size_gb"),
                            "multimodal": spec.get("multimodal", False),
                            "vision": spec.get("vision", False),
                        }
                _DB_SPECS_CACHE = specs_dict
                _DB_SPECS_CACHE_TIME = now
                logging.getLogger("agentic.multi_agent").info(
                    f"Loaded {len(specs_dict)} model specs from database"
                )
                return specs_dict
    except Exception as e:
        logging.getLogger("agentic.multi_agent").warning(
            f"Failed to fetch model specs from database: {e}, using static fallback"
        )

    # Return static fallback
    return OLLAMA_MODEL_SPECS

logger = logging.getLogger("agentic.multi_agent")


class TaskType(Enum):
    """Types of tasks that require different model capabilities"""
    PLANNING = "planning"          # Needs reasoning, high capability
    EVALUATION = "evaluation"      # Needs reasoning, high capability
    SYNTHESIS = "synthesis"        # Needs reasoning, moderate capability
    SEARCH = "search"              # No LLM needed
    SCRAPE = "scrape"              # No LLM needed
    ANALYZE = "analyze"            # Needs reasoning or general chat
    VERIFY = "verify"              # Needs reasoning
    CODE = "code"                  # Needs code capability
    VISION = "vision"              # Needs vision capability


@dataclass
class GPUInfo:
    """Information about a GPU"""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float


@dataclass
class ModelCandidate:
    """A model that could be selected for a task"""
    name: str
    specs: Dict[str, Any]
    vram_required: float
    context_window: int
    speed_tier: str
    capabilities: List[str]
    specialization: str
    score: float = 0.0  # Computed suitability score


@dataclass
class SubTask:
    """A sub-task to be executed by an agent"""
    id: str
    description: str
    task_type: str  # "search", "scrape", "analyze", "verify"
    query: str
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Result from a sub-agent execution"""
    task_id: str
    success: bool
    data: Dict[str, Any]
    execution_time_ms: int
    model_used: str
    tokens_generated: int = 0


class ModelSelector:
    """
    Smart model selector that considers:
    - Total GPU VRAM available across all devices
    - Model capabilities (reasoning, code, vision, etc.)
    - Context window requirements
    - Speed tier (fast/medium/slow)
    - Task-specific specializations

    Uses the OLLAMA_MODEL_SPECS database for accurate model information.
    """

    # Task type to required capabilities mapping
    TASK_REQUIREMENTS = {
        TaskType.PLANNING: {
            "required_caps": ["reasoning"],
            "preferred_caps": ["chat"],
            "preferred_specialization": ["reasoning", "hybrid_reasoning"],
            "min_context": 8000,
            "prefer_speed": "medium",  # Balance speed and capability
            "priority_score_boost": 1.5,  # Boost reasoning models
        },
        TaskType.EVALUATION: {
            "required_caps": ["reasoning"],
            "preferred_caps": ["chat"],
            "preferred_specialization": ["reasoning", "hybrid_reasoning"],
            "min_context": 16000,
            "prefer_speed": "medium",
            "priority_score_boost": 1.5,
        },
        TaskType.SYNTHESIS: {
            "required_caps": ["chat"],
            "preferred_caps": ["reasoning"],
            "preferred_specialization": ["general_purpose", "reasoning"],
            "min_context": 16000,
            "prefer_speed": "medium",
            "priority_score_boost": 1.0,
        },
        TaskType.ANALYZE: {
            "required_caps": ["chat"],
            "preferred_caps": ["reasoning"],
            "preferred_specialization": ["general_purpose"],
            "min_context": 8000,
            "prefer_speed": "fast",
            "priority_score_boost": 1.0,
        },
        TaskType.VERIFY: {
            "required_caps": ["reasoning"],
            "preferred_caps": ["logic"],
            "preferred_specialization": ["reasoning"],
            "min_context": 4000,
            "prefer_speed": "fast",
            "priority_score_boost": 1.2,
        },
        TaskType.CODE: {
            "required_caps": ["code"],
            "preferred_caps": ["agentic"],
            "preferred_specialization": ["code"],
            "min_context": 8000,
            "prefer_speed": "fast",
            "priority_score_boost": 1.3,
        },
        TaskType.VISION: {
            "required_caps": ["vision"],
            "preferred_caps": ["chat"],
            "preferred_specialization": ["vision", "vision_agentic", "multimodal"],
            "min_context": 4000,
            "prefer_speed": "fast",
            "priority_score_boost": 1.0,
        },
    }

    # Practical max VRAM for planning/evaluation (don't use 120B for planning)
    MAX_PRACTICAL_VRAM_FOR_PLANNING = 24  # GB

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self._available_models: List[str] = []
        self._model_specs: Dict[str, Dict[str, Any]] = {}  # Cached specs from DB
        self._gpu_info: List[GPUInfo] = []
        self._last_refresh = 0
        self._last_gpu_refresh = 0
        self._last_specs_refresh = 0
        self._cache_ttl = 60  # Refresh model list every 60 seconds
        self._gpu_cache_ttl = 5  # Refresh GPU info every 5 seconds
        self._specs_cache_ttl = 300  # Refresh specs every 5 minutes

    async def refresh_gpu_info(self) -> List[GPUInfo]:
        """Get current GPU memory status from nvidia-smi"""
        now = time.time()
        if now - self._last_gpu_refresh < self._gpu_cache_ttl and self._gpu_info:
            return self._gpu_info

        gpus = []
        try:
            # Query nvidia-smi for detailed GPU info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpus.append(GPUInfo(
                                index=int(parts[0]),
                                name=parts[1],
                                total_memory_gb=float(parts[2]) / 1024,
                                free_memory_gb=float(parts[3]) / 1024,
                                used_memory_gb=float(parts[4]) / 1024,
                                utilization_percent=float(parts[5]) if parts[5] != '[N/A]' else 0.0
                            ))
                self._gpu_info = gpus
                self._last_gpu_refresh = now
                logger.debug(f"GPU info: {len(gpus)} GPUs, total free: {sum(g.free_memory_gb for g in gpus):.1f}GB")
        except Exception as e:
            logger.warning(f"Failed to query GPU info: {e}")
            # Default to assuming 24GB available
            gpus = [GPUInfo(index=0, name="Unknown", total_memory_gb=24, free_memory_gb=20,
                           used_memory_gb=4, utilization_percent=0)]

        return gpus

    def get_total_free_vram(self) -> float:
        """Get total free VRAM across all GPUs"""
        return sum(g.free_memory_gb for g in self._gpu_info) if self._gpu_info else 20.0

    def get_max_single_gpu_vram(self) -> float:
        """Get max free VRAM on any single GPU"""
        return max((g.free_memory_gb for g in self._gpu_info), default=20.0)

    async def refresh_models(self) -> List[str]:
        """Fetch available models from Ollama and specs from database"""
        now = time.time()
        if now - self._last_refresh < self._cache_ttl and self._available_models:
            return self._available_models

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m.get("name", "") for m in data.get("models", [])]
                    self._last_refresh = now
                    logger.info(f"Found {len(self._available_models)} Ollama models")
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")

        # Also refresh model specs from database
        if now - self._last_specs_refresh > self._specs_cache_ttl or not self._model_specs:
            self._model_specs = await fetch_model_specs_from_db()
            self._last_specs_refresh = now
            logger.info(f"Loaded {len(self._model_specs)} model specs from database")

        return self._available_models

    def get_model_specs(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get specs for a model from database (with static fallback)"""
        # First try database specs (exact match)
        if model_name in self._model_specs:
            return self._model_specs[model_name]

        # Try base name match in database specs
        base_name = model_name.split(':')[0] if ':' in model_name else model_name
        for spec_name, specs in self._model_specs.items():
            if spec_name.startswith(base_name + ':') or spec_name == base_name:
                return specs

        # Fallback to static specs (exact match)
        if model_name in OLLAMA_MODEL_SPECS:
            return OLLAMA_MODEL_SPECS[model_name]

        # Fallback to static specs (base name match)
        for spec_name, specs in OLLAMA_MODEL_SPECS.items():
            if spec_name.startswith(base_name):
                return specs

        return None

    def _score_model_for_task(
        self,
        model_name: str,
        specs: Dict[str, Any],
        task_type: TaskType,
        available_vram: float
    ) -> float:
        """
        Score a model's suitability for a task (0.0 - 1.0).

        Considers:
        - VRAM fit (must fit, preferably with buffer)
        - Capability match
        - Specialization match
        - Speed tier preference
        - Context window adequacy
        """
        requirements = self.TASK_REQUIREMENTS.get(task_type, {})
        score = 0.0

        # CRITICAL: Reject embedding models for all non-embedding tasks
        # Embedding models cannot generate text, only produce embeddings
        embedding_patterns = [
            "embedding", "embed",           # General embedding models
            "bge-", "bge_", "bge:",         # BGE embedding models
            "minilm",                       # All-minilm models
            "arctic-embed", "snowflake",    # Snowflake embedding
            "nomic-embed",                  # Nomic embedding
            "functiongemma", "embeddinggemma",  # Gemma embedding variants
            "mxbai-embed",                  # MixedBread embedding
        ]
        model_lower = model_name.lower()
        if any(pattern in model_lower for pattern in embedding_patterns):
            return 0.0  # Embedding models cannot be used for text generation

        # VRAM check - must fit with some buffer
        vram_needed = specs.get("vram_min_gb", 4)
        if vram_needed > available_vram - 2.0:  # 2GB buffer
            return 0.0  # Can't fit

        # For planning/eval, cap at practical size
        if task_type in [TaskType.PLANNING, TaskType.EVALUATION]:
            if vram_needed > self.MAX_PRACTICAL_VRAM_FOR_PLANNING:
                return 0.0

        # VRAM efficiency score (prefer models that use available VRAM well)
        vram_utilization = vram_needed / available_vram
        if vram_utilization < 0.3:
            score += 0.1  # Small model penalty (underutilizing resources)
        elif vram_utilization < 0.7:
            score += 0.3  # Good utilization
        else:
            score += 0.2  # Close to limit

        # Capability match
        model_caps = set(specs.get("capabilities", []))
        required_caps = set(requirements.get("required_caps", []))
        preferred_caps = set(requirements.get("preferred_caps", []))

        if not required_caps.issubset(model_caps):
            return 0.0  # Missing required capabilities

        # Score for preferred capabilities
        preferred_match = len(preferred_caps & model_caps) / max(len(preferred_caps), 1)
        score += 0.25 * preferred_match

        # Specialization match
        model_spec = specs.get("specialization", "general_purpose")
        preferred_specs = requirements.get("preferred_specialization", [])
        if model_spec in preferred_specs:
            score += 0.2

        # Speed tier preference
        model_speed = specs.get("speed_tier", "medium")
        preferred_speed = requirements.get("prefer_speed", "medium")
        if model_speed == preferred_speed:
            score += 0.15
        elif model_speed == "fast" and preferred_speed != "slow":
            score += 0.1
        elif model_speed == "slow" and preferred_speed == "medium":
            score += 0.05

        # Context window adequacy
        min_context = requirements.get("min_context", 4000)
        model_context = specs.get("context_window", 4096)
        if model_context >= min_context:
            # Bonus for extra context capacity
            context_ratio = min(model_context / min_context, 4.0)
            score += 0.1 * (context_ratio - 1) / 3

        # Apply task-specific priority boost
        priority_boost = requirements.get("priority_score_boost", 1.0)
        score *= priority_boost

        return min(score, 1.0)

    async def select_model_for_task(
        self,
        task_type: TaskType,
        context_needed: int = 4000
    ) -> Optional[str]:
        """
        Select the best model for a specific task type.

        Args:
            task_type: The type of task to perform
            context_needed: Minimum context window required

        Returns:
            Model name or None if no suitable model found
        """
        await self.refresh_gpu_info()
        await self.refresh_models()

        available_vram = self.get_max_single_gpu_vram()
        logger.info(f"Selecting model for {task_type.value}, available VRAM: {available_vram:.1f}GB")

        candidates: List[ModelCandidate] = []

        for model_name in self._available_models:
            specs = self.get_model_specs(model_name)
            if not specs:
                continue

            score = self._score_model_for_task(model_name, specs, task_type, available_vram)
            if score > 0:
                candidates.append(ModelCandidate(
                    name=model_name,
                    specs=specs,
                    vram_required=specs.get("vram_min_gb", 4),
                    context_window=specs.get("context_window", 4096),
                    speed_tier=specs.get("speed_tier", "medium"),
                    capabilities=specs.get("capabilities", []),
                    specialization=specs.get("specialization", "general_purpose"),
                    score=score
                ))

        if not candidates:
            logger.warning(f"No suitable model found for {task_type.value}")
            return None

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        best = candidates[0]
        logger.info(f"Selected {best.name} for {task_type.value} (score: {best.score:.2f}, "
                   f"VRAM: {best.vram_required}GB, speed: {best.speed_tier})")

        return best.name

    async def get_model_for_task(self, task_type: str) -> str:
        """
        Legacy interface for backwards compatibility.
        Maps string task types to TaskType enum.
        """
        type_map = {
            "plan": TaskType.PLANNING,
            "planning": TaskType.PLANNING,
            "evaluate": TaskType.EVALUATION,
            "evaluation": TaskType.EVALUATION,
            "synthesize": TaskType.SYNTHESIS,
            "synthesis": TaskType.SYNTHESIS,
            "analyze": TaskType.ANALYZE,
            "analysis": TaskType.ANALYZE,
            "verify": TaskType.VERIFY,
            "verification": TaskType.VERIFY,
            "code": TaskType.CODE,
            "vision": TaskType.VISION,
            "search": TaskType.ANALYZE,  # Map to analyze as fallback
            "scrape": TaskType.ANALYZE,
        }

        task_enum = type_map.get(task_type.lower(), TaskType.ANALYZE)
        model = await self.select_model_for_task(task_enum)

        # Fallback to a known small model if nothing found
        return model or "gemma3:4b"

    async def can_run_parallel(self, models: List[str]) -> bool:
        """Check if multiple models can run in parallel based on total VRAM"""
        await self.refresh_gpu_info()
        total_free = self.get_total_free_vram()

        total_needed = 0
        for model_name in models:
            specs = self.get_model_specs(model_name)
            if specs:
                total_needed += specs.get("vram_min_gb", 4)
            else:
                total_needed += 4  # Default estimate

        # Need 4GB buffer for system
        can_parallel = total_needed < (total_free - 4.0)
        logger.info(f"Parallel check: {total_needed:.1f}GB needed, {total_free:.1f}GB free, can_parallel={can_parallel}")
        return can_parallel

    async def estimate_vram_usage(self, model_name: str) -> float:
        """Estimate VRAM usage for a model"""
        specs = self.get_model_specs(model_name)
        if specs:
            return specs.get("vram_min_gb", 4)
        return 4.0

    async def get_available_vram(self) -> float:
        """Get available GPU VRAM (legacy interface)"""
        await self.refresh_gpu_info()
        return self.get_max_single_gpu_vram()


class SubAgent:
    """
    A sub-agent that executes a specific task with its own context.

    Each sub-agent has:
    - Its own context window
    - A specific task to complete
    - Access to tools (search, scrape, analyze)
    """

    def __init__(
        self,
        task: SubTask,
        ollama_url: str,
        model: str,
        scraper: Any,  # ContentScraper instance
        searcher: Any  # SearcherAgent instance
    ):
        self.task = task
        self.ollama_url = ollama_url
        self.model = model
        self.scraper = scraper
        self.searcher = searcher
        self.context: List[Dict[str, str]] = []

    async def execute(self) -> AgentResult:
        """Execute the assigned task"""
        start_time = time.time()
        self.task.status = "running"

        try:
            if self.task.task_type == "search":
                result = await self._execute_search()
            elif self.task.task_type == "scrape":
                result = await self._execute_scrape()
            elif self.task.task_type == "analyze":
                result = await self._execute_analyze()
            elif self.task.task_type == "verify":
                result = await self._execute_verify()
            else:
                result = {"error": f"Unknown task type: {self.task.task_type}"}

            self.task.status = "completed"
            self.task.result = result

            return AgentResult(
                task_id=self.task.id,
                success=True,
                data=result,
                execution_time_ms=int((time.time() - start_time) * 1000),
                model_used=self.model
            )

        except Exception as e:
            self.task.status = "failed"
            self.task.error = str(e)
            logger.error(f"Sub-agent {self.task.id} failed: {e}")

            return AgentResult(
                task_id=self.task.id,
                success=False,
                data={"error": str(e)},
                execution_time_ms=int((time.time() - start_time) * 1000),
                model_used=self.model
            )

    async def _execute_search(self) -> Dict[str, Any]:
        """Execute a search task"""
        results = await self.searcher.search([self.task.query], max_results_per_query=5)
        return {
            "type": "search",
            "query": self.task.query,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "domain": r.source_domain}
                for r in results
            ],
            "count": len(results)
        }

    async def _execute_scrape(self) -> Dict[str, Any]:
        """Execute a scrape task"""
        # Task query contains URL(s) to scrape
        urls = self.task.query.split("|")
        scraped = await self.scraper.scrape_urls(urls, max_concurrent=2)
        return {
            "type": "scrape",
            "urls": urls,
            "content": [
                {
                    "url": s.get("url"),
                    "title": s.get("title"),
                    "content": s.get("content", "")[:20000],  # Limit content size
                    "success": s.get("success", False)
                }
                for s in scraped
            ],
            "successful": sum(1 for s in scraped if s.get("success"))
        }

    async def _execute_analyze(self) -> Dict[str, Any]:
        """Execute an analysis task using LLM"""
        prompt = f"""Analyze the following information to answer this question:

QUESTION: {self.task.description}

INFORMATION:
{self.task.query}

Provide:
1. A direct answer based on the information
2. Key facts found
3. Confidence level (HIGH/MEDIUM/LOW)
4. What's still unclear or missing

Format as JSON:
{{"answer": "...", "key_facts": [...], "confidence": "...", "missing": "..."}}
"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 1000}
                }
            )
            response.raise_for_status()
            result = response.json().get("response", "")

        # Try to parse as JSON
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"answer": result, "confidence": "low", "key_facts": [], "missing": ""}

    async def _execute_verify(self) -> Dict[str, Any]:
        """Execute a verification task"""
        prompt = f"""Verify the following claim using the provided sources:

CLAIM: {self.task.description}

SOURCES:
{self.task.query}

Respond with JSON:
{{"verified": true/false, "confidence": 0.0-1.0, "supporting_evidence": [...], "contradicting_evidence": [...]}}
"""

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                }
            )
            response.raise_for_status()
            result = response.json().get("response", "")

        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"verified": False, "confidence": 0.0}


class SmartPlanner:
    """
    Uses a large model to analyze queries and create execution plans.
    """

    def __init__(self, ollama_url: str, model_selector: ModelSelector):
        self.ollama_url = ollama_url
        self.model_selector = model_selector

    async def create_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[SubTask], str]:
        """
        Analyze query and create a plan of sub-tasks.

        Returns:
            Tuple of (list of SubTasks, reasoning string)
        """
        model = await self.model_selector.get_model_for_task("plan")
        logger.info(f"Planning with model: {model}")

        prompt = f"""You are an expert research planner. Analyze this query and create a search plan.

QUERY: "{query}"

Create a plan with specific sub-tasks. Each task should be one of:
- search: Web search for specific information
- scrape: Fetch full content from specific URLs (if known)
- analyze: Analyze gathered information
- verify: Cross-check a specific claim

Consider:
1. What specific information is needed?
2. What search queries would find this?
3. What types of sources are most likely to have this info?
4. Can tasks run in parallel or do they depend on each other?

Respond with JSON:
{{
    "reasoning": "Your analysis of what's needed...",
    "tasks": [
        {{
            "id": "task_1",
            "type": "search",
            "description": "What this task finds",
            "query": "specific search query",
            "priority": 1,
            "depends_on": []
        }},
        {{
            "id": "task_2",
            "type": "search",
            "description": "Another aspect to search",
            "query": "another search query",
            "priority": 1,
            "depends_on": []
        }},
        {{
            "id": "task_3",
            "type": "analyze",
            "description": "Combine and analyze results",
            "query": "",
            "priority": 2,
            "depends_on": ["task_1", "task_2"]
        }}
    ]
}}

For the query "{query}", create 3-6 focused search tasks that can run in parallel, then an analysis task."""

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 1500}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            # Parse the plan
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                tasks = []

                for t in plan_data.get("tasks", []):
                    tasks.append(SubTask(
                        id=t.get("id", f"task_{len(tasks)}"),
                        description=t.get("description", ""),
                        task_type=t.get("type", "search"),
                        query=t.get("query", ""),
                        dependencies=t.get("depends_on", []),
                        priority=t.get("priority", 1)
                    ))

                reasoning = plan_data.get("reasoning", "")
                logger.info(f"Created plan with {len(tasks)} tasks")
                return tasks, reasoning

        except Exception as e:
            logger.error(f"Planning failed: {e}")

        # Fallback: simple plan
        return [
            SubTask(id="task_1", description="Search for information",
                   task_type="search", query=query, priority=1)
        ], "Fallback plan"


class SmartEvaluator:
    """
    Uses a large model to evaluate results and decide next steps.
    """

    def __init__(self, ollama_url: str, model_selector: ModelSelector):
        self.ollama_url = ollama_url
        self.model_selector = model_selector

    async def evaluate_results(
        self,
        original_query: str,
        completed_tasks: List[SubTask],
        iteration: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Evaluate current results and decide next steps.

        Returns:
            {
                "sufficient": bool,
                "confidence": float,
                "answer": str (if sufficient),
                "missing_info": str,
                "next_tasks": List[SubTask] (if not sufficient),
                "reasoning": str
            }
        """
        model = await self.model_selector.get_model_for_task("evaluate")
        logger.info(f"Evaluating with model: {model} (iteration {iteration}/{max_iterations})")

        # Compile results
        results_summary = []
        for task in completed_tasks:
            if task.result:
                results_summary.append(f"[{task.task_type}] {task.description}:\n{json.dumps(task.result, indent=2)[:2000]}")

        results_text = "\n\n".join(results_summary)

        prompt = f"""You are evaluating research results to answer a question.

ORIGINAL QUESTION: "{original_query}"

ITERATION: {iteration} of {max_iterations}

GATHERED INFORMATION:
{results_text}

Evaluate:
1. Do we have enough information to answer the question confidently?
2. What specific information is still missing?
3. What additional searches might help?

Respond with JSON:
{{
    "sufficient": true/false,
    "confidence": 0.0-1.0,
    "answer": "Complete answer if sufficient, otherwise partial answer so far",
    "key_findings": ["finding 1", "finding 2"],
    "missing_info": "What's still unclear or missing",
    "reasoning": "Why you made this determination",
    "next_queries": ["refined search 1", "refined search 2"] (if not sufficient)
}}"""

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 1500}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())

                # Create next tasks if needed
                next_tasks = []
                if not eval_data.get("sufficient", False) and iteration < max_iterations:
                    for i, query in enumerate(eval_data.get("next_queries", [])[:3]):
                        next_tasks.append(SubTask(
                            id=f"refined_{iteration}_{i}",
                            description=f"Refined search: {query}",
                            task_type="search",
                            query=query,
                            priority=1
                        ))

                return {
                    "sufficient": eval_data.get("sufficient", False),
                    "confidence": eval_data.get("confidence", 0.5),
                    "answer": eval_data.get("answer", ""),
                    "key_findings": eval_data.get("key_findings", []),
                    "missing_info": eval_data.get("missing_info", ""),
                    "reasoning": eval_data.get("reasoning", ""),
                    "next_tasks": next_tasks
                }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

        return {
            "sufficient": iteration >= max_iterations,
            "confidence": 0.3,
            "answer": "Evaluation failed",
            "key_findings": [],
            "missing_info": str(e) if 'e' in dir() else "Unknown error",
            "reasoning": "Evaluation error",
            "next_tasks": []
        }


class AgentPool:
    """
    Manages parallel or sequential execution of sub-agents based on resources.
    """

    def __init__(
        self,
        ollama_url: str,
        model_selector: ModelSelector,
        scraper: Any,
        searcher: Any
    ):
        self.ollama_url = ollama_url
        self.model_selector = model_selector
        self.scraper = scraper
        self.searcher = searcher

    async def execute_tasks(
        self,
        tasks: List[SubTask],
        force_sequential: bool = False
    ) -> List[AgentResult]:
        """
        Execute a batch of tasks, in parallel if possible.
        """
        if not tasks:
            return []

        # Group tasks by dependency level
        levels = self._topological_sort(tasks)
        all_results = []

        for level_tasks in levels:
            # Determine model for each task
            task_models = {}
            for task in level_tasks:
                model = await self.model_selector.get_model_for_task(task.task_type)
                task_models[task.id] = model

            # Check if we can run in parallel
            can_parallel = not force_sequential and await self.model_selector.can_run_parallel(
                list(set(task_models.values()))
            )

            if can_parallel and len(level_tasks) > 1:
                logger.info(f"Executing {len(level_tasks)} tasks in PARALLEL")
                results = await self._execute_parallel(level_tasks, task_models)
            else:
                logger.info(f"Executing {len(level_tasks)} tasks SEQUENTIALLY")
                results = await self._execute_sequential(level_tasks, task_models)

            all_results.extend(results)

        return all_results

    def _topological_sort(self, tasks: List[SubTask]) -> List[List[SubTask]]:
        """Sort tasks by dependency levels"""
        task_map = {t.id: t for t in tasks}
        levels = []
        remaining = set(t.id for t in tasks)
        completed = set()

        while remaining:
            # Find tasks with no unmet dependencies
            ready = []
            for task_id in remaining:
                task = task_map[task_id]
                if all(dep in completed for dep in task.dependencies):
                    ready.append(task)

            if not ready:
                # Circular dependency or missing dependency - take all remaining
                ready = [task_map[tid] for tid in remaining]

            levels.append(ready)
            for task in ready:
                remaining.remove(task.id)
                completed.add(task.id)

        return levels

    async def _execute_parallel(
        self,
        tasks: List[SubTask],
        task_models: Dict[str, str]
    ) -> List[AgentResult]:
        """Execute tasks in parallel"""
        agents = [
            SubAgent(
                task=task,
                ollama_url=self.ollama_url,
                model=task_models[task.id],
                scraper=self.scraper,
                searcher=self.searcher
            )
            for task in tasks
        ]

        results = await asyncio.gather(
            *[agent.execute() for agent in agents],
            return_exceptions=True
        )

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(AgentResult(
                    task_id=tasks[i].id,
                    success=False,
                    data={"error": str(result)},
                    execution_time_ms=0,
                    model_used=task_models[tasks[i].id]
                ))
            else:
                processed.append(result)

        return processed

    async def _execute_sequential(
        self,
        tasks: List[SubTask],
        task_models: Dict[str, str]
    ) -> List[AgentResult]:
        """Execute tasks one at a time"""
        results = []

        for task in tasks:
            agent = SubAgent(
                task=task,
                ollama_url=self.ollama_url,
                model=task_models[task.id],
                scraper=self.scraper,
                searcher=self.searcher
            )
            result = await agent.execute()
            results.append(result)

        return results


class MultiAgentOrchestrator:
    """
    Main orchestrator for multi-agent search with iterative refinement.

    Workflow:
    1. Smart Planner creates initial task plan
    2. Agent Pool executes tasks (parallel if possible)
    3. Smart Evaluator reviews results
    4. If not sufficient, create refined tasks and loop
    5. Final synthesis with smart model
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        scraper: Any = None,
        searcher: Any = None,
        max_iterations: int = 5
    ):
        self.ollama_url = ollama_url
        self.model_selector = ModelSelector(ollama_url)
        self.planner = SmartPlanner(ollama_url, self.model_selector)
        self.evaluator = SmartEvaluator(ollama_url, self.model_selector)
        self.scraper = scraper
        self.searcher = searcher
        self.agent_pool = AgentPool(ollama_url, self.model_selector, scraper, searcher)
        self.max_iterations = max_iterations

    async def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-agent search with iterative refinement.
        """
        start_time = time.time()
        all_tasks: List[SubTask] = []
        all_results: List[AgentResult] = []
        trace = []

        logger.info(f"=== MULTI-AGENT SEARCH START ===")
        logger.info(f"Query: {query}")

        # Refresh available models and GPU info
        await self.model_selector.refresh_gpu_info()
        models = await self.model_selector.refresh_models()
        # Models is now a list of strings (model names)
        smartest = models[0] if models else "unknown"
        available_vram = self.model_selector.get_max_single_gpu_vram()
        logger.info(f"Available models: {len(models)}, VRAM: {available_vram:.1f}GB, largest: {smartest}")

        trace.append({
            "step": "init",
            "models_available": models[:5],
            "available_vram_gb": available_vram,
            "largest_model": smartest
        })

        # PHASE 1: Planning
        logger.info("PHASE 1: Creating execution plan...")
        tasks, reasoning = await self.planner.create_plan(query, context)
        all_tasks.extend(tasks)

        trace.append({
            "step": "plan",
            "task_count": len(tasks),
            "reasoning": reasoning,
            "tasks": [{"id": t.id, "type": t.task_type, "query": t.query} for t in tasks]
        })

        # PHASE 2: Iterative execution and evaluation
        iteration = 0
        final_evaluation = None

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n=== ITERATION {iteration}/{self.max_iterations} ===")

            # Get pending tasks
            pending = [t for t in all_tasks if t.status == "pending"]
            if not pending:
                logger.info("No pending tasks")
                break

            logger.info(f"Executing {len(pending)} pending tasks...")

            # Execute tasks
            results = await self.agent_pool.execute_tasks(pending)
            all_results.extend(results)

            trace.append({
                "step": "execute",
                "iteration": iteration,
                "tasks_executed": len(results),
                "parallel": len(pending) > 1,
                "results_summary": [
                    {"task_id": r.task_id, "success": r.success, "time_ms": r.execution_time_ms}
                    for r in results
                ]
            })

            # Scrape top results if we have search results
            search_results = [r for r in results if r.success and r.data.get("type") == "search"]
            if search_results:
                urls_to_scrape = []
                for sr in search_results:
                    for result in sr.data.get("results", [])[:2]:  # Top 2 from each search
                        url = result.get("url", "")
                        if url and url not in urls_to_scrape:
                            urls_to_scrape.append(url)

                if urls_to_scrape[:5]:  # Limit to 5 URLs
                    logger.info(f"Scraping {len(urls_to_scrape[:5])} URLs...")
                    scrape_task = SubTask(
                        id=f"scrape_{iteration}",
                        description="Scrape search results",
                        task_type="scrape",
                        query="|".join(urls_to_scrape[:5])
                    )
                    scrape_results = await self.agent_pool.execute_tasks([scrape_task])
                    all_results.extend(scrape_results)
                    all_tasks.append(scrape_task)

            # Evaluate results
            logger.info("Evaluating results...")
            completed = [t for t in all_tasks if t.status == "completed"]
            evaluation = await self.evaluator.evaluate_results(
                query, completed, iteration, self.max_iterations
            )
            final_evaluation = evaluation

            trace.append({
                "step": "evaluate",
                "iteration": iteration,
                "sufficient": evaluation.get("sufficient"),
                "confidence": evaluation.get("confidence"),
                "reasoning": evaluation.get("reasoning"),
                "next_tasks": len(evaluation.get("next_tasks", []))
            })

            if evaluation.get("sufficient"):
                logger.info(f"Information sufficient at iteration {iteration}")
                break

            # Add refined tasks for next iteration
            next_tasks = evaluation.get("next_tasks", [])
            if next_tasks:
                logger.info(f"Adding {len(next_tasks)} refined tasks")
                all_tasks.extend(next_tasks)
            else:
                logger.info("No more refined queries, stopping")
                break

        # PHASE 3: Final synthesis
        logger.info("\nPHASE 3: Final synthesis...")

        execution_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "success": True,
            "answer": final_evaluation.get("answer", "") if final_evaluation else "",
            "key_findings": final_evaluation.get("key_findings", []) if final_evaluation else [],
            "confidence": final_evaluation.get("confidence", 0.0) if final_evaluation else 0.0,
            "missing_info": final_evaluation.get("missing_info", "") if final_evaluation else "",
            "sources": self._extract_sources(all_results),
            "iterations": iteration,
            "tasks_executed": len([t for t in all_tasks if t.status == "completed"]),
            "execution_time_ms": execution_time_ms,
            "models_used": list(set(r.model_used for r in all_results)),
            "trace": trace
        }

        logger.info(f"=== MULTI-AGENT SEARCH COMPLETE ===")
        logger.info(f"Iterations: {iteration}, Tasks: {result['tasks_executed']}, Time: {execution_time_ms}ms")
        logger.info(f"Confidence: {result['confidence']:.0%}")

        return result

    def _extract_sources(self, results: List[AgentResult]) -> List[Dict[str, str]]:
        """Extract unique sources from all results"""
        sources = []
        seen_urls = set()

        for r in results:
            if not r.success:
                continue

            # From search results
            for item in r.data.get("results", []):
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "url": url,
                        "title": item.get("title", ""),
                        "domain": item.get("domain", "")
                    })

            # From scrape results
            for item in r.data.get("content", []):
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "url": url,
                        "title": item.get("title", ""),
                        "scraped": item.get("success", False)
                    })

        return sources[:20]  # Limit to 20 sources
