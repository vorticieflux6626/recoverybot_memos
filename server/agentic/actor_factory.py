"""
Phase 5: AIME-Style Actor Factory

Implements dynamic agent specialization with tool bundles:
- Actors are created on-demand for specific subtasks
- Each actor has: LLM, Toolkit (bundles), Persona, Memory
- Tool bundles provide functional completeness for task types

Reference: AIME (ByteDance) - https://arxiv.org/abs/2507.11988
Formula: A_t = {LLM_t, T_t, P_t, M_t}
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable
import httpx

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools"""
    SEARCH = "search"
    SCRAPE = "scrape"
    EXTRACT = "extract"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"
    ANALYZE = "analyze"
    CODE = "code"
    VISION = "vision"
    MEMORY = "memory"


class ModelCapability(Enum):
    """Required model capabilities"""
    TEXT_GENERATION = "text_generation"
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning"
    VISION = "vision"
    CODE = "code"
    EMBEDDING = "embedding"
    FAST = "fast"  # Quick responses, smaller models


@dataclass
class Tool:
    """Individual tool definition"""
    name: str
    description: str
    category: ToolCategory
    function: Optional[Callable[..., Awaitable[Any]]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation
        }


@dataclass
class ToolBundle:
    """
    Pre-packaged tool collection for functional completeness.

    AIME insight: Group tools by functional purpose rather than
    exposing individual tools. Reduces cognitive load on agents.
    """
    name: str
    description: str
    tools: List[str]  # Tool names
    model_requirements: List[ModelCapability]
    category: ToolCategory
    priority: int = 1  # Higher = more important

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": self.tools,
            "model_requirements": [m.value for m in self.model_requirements],
            "category": self.category.value,
            "priority": self.priority
        }


# Predefined tool bundles based on AIME research
DEFAULT_TOOL_BUNDLES = {
    "web_research": ToolBundle(
        name="WebResearch",
        description="Web search, URL fetching, content extraction for information gathering",
        tools=["brave_search", "duckduckgo_search", "scrape_url", "extract_content"],
        model_requirements=[ModelCapability.TEXT_GENERATION],
        category=ToolCategory.SEARCH,
        priority=1
    ),
    "vision_extraction": ToolBundle(
        name="VisionExtraction",
        description="Screenshot capture and vision-language model extraction for JS-rendered pages",
        tools=["capture_screenshot", "vl_extract", "ocr_text"],
        model_requirements=[ModelCapability.VISION],
        category=ToolCategory.VISION,
        priority=2
    ),
    "verification": ToolBundle(
        name="Verification",
        description="Fact checking, cross-referencing, source validation",
        tools=["search_for_verification", "compare_sources", "check_credibility"],
        model_requirements=[ModelCapability.REASONING],
        category=ToolCategory.VERIFY,
        priority=2
    ),
    "synthesis": ToolBundle(
        name="Synthesis",
        description="Content synthesis with citations, summarization, aggregation",
        tools=["synthesize_content", "add_citations", "summarize"],
        model_requirements=[ModelCapability.LONG_CONTEXT],
        category=ToolCategory.SYNTHESIZE,
        priority=1
    ),
    "analysis": ToolBundle(
        name="Analysis",
        description="Deep content analysis, pattern detection, insight extraction",
        tools=["analyze_content", "extract_insights", "identify_patterns"],
        model_requirements=[ModelCapability.REASONING],
        category=ToolCategory.ANALYZE,
        priority=2
    ),
    "code_analysis": ToolBundle(
        name="CodeAnalysis",
        description="Code understanding, bug detection, documentation extraction",
        tools=["parse_code", "analyze_dependencies", "extract_docs"],
        model_requirements=[ModelCapability.CODE],
        category=ToolCategory.CODE,
        priority=3
    ),
    "memory_ops": ToolBundle(
        name="MemoryOps",
        description="Store and retrieve from working memory, context management",
        tools=["store_finding", "retrieve_context", "update_scratchpad"],
        model_requirements=[ModelCapability.TEXT_GENERATION],
        category=ToolCategory.MEMORY,
        priority=1
    ),
    "quick_response": ToolBundle(
        name="QuickResponse",
        description="Fast, simple responses without external tools",
        tools=["direct_response"],
        model_requirements=[ModelCapability.FAST],
        category=ToolCategory.SYNTHESIZE,
        priority=1
    )
}


@dataclass
class ActorPersona:
    """
    Customized persona for an actor.

    AIME insight: Each actor should have a clear role and expertise
    that guides its behavior and tool usage.
    """
    role: str
    expertise: List[str]
    communication_style: str
    constraints: List[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Generate system prompt from persona"""
        expertise_str = ", ".join(self.expertise)
        constraints_str = "\n".join(f"- {c}" for c in self.constraints) if self.constraints else "None"

        return f"""You are a specialized AI agent with the following role:

## Role
{self.role}

## Expertise
{expertise_str}

## Communication Style
{self.communication_style}

## Constraints
{constraints_str}

Always stay within your area of expertise. If a task falls outside your capabilities, clearly indicate this."""


@dataclass
class DynamicActor:
    """
    Purpose-built agent instantiated for a specific subtask.

    AIME Formula: A_t = {LLM_t, T_t, P_t, M_t}
    - LLM_t: Cognitive engine (model selection)
    - T_t: Toolkit (selected bundles)
    - P_t: Persona (customized prompt)
    - M_t: Memory (relevant context)
    """
    id: str
    model: str
    bundles: List[ToolBundle]
    persona: ActorPersona
    system_prompt: str
    memory_context: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executions: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions

    def get_available_tools(self) -> List[str]:
        """Get all tools from all bundles"""
        tools = []
        for bundle in self.bundles:
            tools.extend(bundle.tools)
        return list(set(tools))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "bundles": [b.name for b in self.bundles],
            "persona_role": self.persona.role,
            "tools": self.get_available_tools(),
            "executions": self.executions,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat()
        }


class ActorFactory:
    """
    AIME Actor Factory: Create purpose-built agents on demand.

    Key Innovation:
    - Agents are NOT pre-defined but dynamically assembled
    - Tool bundles provide functional completeness
    - Model selection based on task requirements
    - Personas generated to match subtask needs
    """

    # Task type to bundle mapping
    TASK_BUNDLE_MAP = {
        "search": ["web_research"],
        "research": ["web_research", "synthesis"],
        "verify": ["verification", "web_research"],
        "synthesize": ["synthesis"],
        "analyze": ["analysis"],
        "extract": ["web_research", "analysis"],
        "compare": ["web_research", "analysis", "synthesis"],
        "screenshot": ["vision_extraction"],
        "code": ["code_analysis"],
        "quick": ["quick_response"],
        "memory": ["memory_ops"]
    }

    # Model selection based on capabilities
    MODEL_CAPABILITY_MAP = {
        ModelCapability.TEXT_GENERATION: ["qwen3:8b", "llama3.2:3b", "gemma3:4b"],
        ModelCapability.LONG_CONTEXT: ["qwen3:8b", "deepseek-r1:14b"],
        ModelCapability.REASONING: ["deepseek-r1:14b", "qwen3:8b"],
        ModelCapability.VISION: ["qwen3-vl", "llama3.2-vision", "gemma3:4b-it-vision"],
        ModelCapability.CODE: ["qwen3:8b", "deepseek-r1:14b", "codellama:7b"],
        ModelCapability.EMBEDDING: ["mxbai-embed-large", "nomic-embed-text"],
        ModelCapability.FAST: ["gemma3:4b", "llama3.2:3b", "qwen3:4b"]
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        default_model: str = "qwen3:8b"
    ):
        """
        Initialize ActorFactory.

        Args:
            ollama_url: Ollama API URL
            default_model: Fallback model if capability match fails
        """
        self.ollama_url = ollama_url
        self.default_model = default_model
        self.bundles = dict(DEFAULT_TOOL_BUNDLES)
        self.tools: Dict[str, Tool] = {}
        self.actors: Dict[str, DynamicActor] = {}
        self._stats = {
            "actors_created": 0,
            "actors_reused": 0,
            "total_executions": 0
        }

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for use in bundles"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_bundle(self, bundle_id: str, bundle: ToolBundle) -> None:
        """Register a custom tool bundle"""
        self.bundles[bundle_id] = bundle
        logger.info(f"Registered bundle: {bundle_id}")

    def _analyze_task_requirements(self, task_description: str) -> Set[str]:
        """
        Analyze task to determine required bundle types.

        Uses keyword matching for simplicity; could be enhanced with
        LLM-based classification.
        """
        task_lower = task_description.lower()
        required = set()

        # Keyword to bundle type mapping
        keywords = {
            "search": ["search", "find", "look up", "query", "what is", "who is"],
            "research": ["research", "investigate", "explore", "learn about"],
            "verify": ["verify", "check", "validate", "confirm", "fact-check"],
            "synthesize": ["synthesize", "combine", "summarize", "aggregate"],
            "analyze": ["analyze", "examine", "evaluate", "assess"],
            "extract": ["extract", "get", "retrieve", "pull out"],
            "compare": ["compare", "contrast", "difference", "versus", "vs"],
            "screenshot": ["screenshot", "visual", "render", "js-rendered"],
            "code": ["code", "programming", "function", "bug", "debug"],
            "quick": ["quick", "simple", "fast", "direct"],
            "memory": ["remember", "store", "recall", "context"]
        }

        for task_type, keyword_list in keywords.items():
            for keyword in keyword_list:
                if keyword in task_lower:
                    required.add(task_type)
                    break

        # Default to research if nothing matched
        if not required:
            required.add("research")

        return required

    def _select_bundles(self, task_types: Set[str]) -> List[ToolBundle]:
        """Select tool bundles based on task types"""
        bundle_names = set()
        for task_type in task_types:
            if task_type in self.TASK_BUNDLE_MAP:
                bundle_names.update(self.TASK_BUNDLE_MAP[task_type])

        bundles = [
            self.bundles[name]
            for name in bundle_names
            if name in self.bundles
        ]

        # Sort by priority
        bundles.sort(key=lambda b: b.priority)

        return bundles

    def _get_required_capabilities(self, bundles: List[ToolBundle]) -> Set[ModelCapability]:
        """Get all model capabilities required by bundles"""
        capabilities = set()
        for bundle in bundles:
            capabilities.update(bundle.model_requirements)
        return capabilities

    async def _select_model(self, capabilities: Set[ModelCapability]) -> str:
        """
        Select best model for required capabilities.

        Prioritizes models that satisfy the most capabilities.
        """
        if not capabilities:
            return self.default_model

        # Score models by capability coverage
        model_scores: Dict[str, int] = {}

        for capability in capabilities:
            if capability in self.MODEL_CAPABILITY_MAP:
                for model in self.MODEL_CAPABILITY_MAP[capability]:
                    model_scores[model] = model_scores.get(model, 0) + 1

        if not model_scores:
            return self.default_model

        # Return model with highest score
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]

        # Verify model is available
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    available = {m["name"] for m in response.json().get("models", [])}
                    if best_model in available or any(best_model in m for m in available):
                        return best_model
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")

        return self.default_model

    def _generate_persona(
        self,
        task_description: str,
        bundles: List[ToolBundle]
    ) -> ActorPersona:
        """Generate persona based on task and bundles"""

        # Determine expertise from bundles
        expertise = []
        for bundle in bundles:
            expertise.append(bundle.description.split(" for ")[0])

        # Determine role based on primary bundle
        if bundles:
            primary = bundles[0]
            role_map = {
                ToolCategory.SEARCH: "Research Specialist",
                ToolCategory.VERIFY: "Fact Checker",
                ToolCategory.SYNTHESIZE: "Content Synthesizer",
                ToolCategory.ANALYZE: "Data Analyst",
                ToolCategory.VISION: "Visual Content Analyst",
                ToolCategory.CODE: "Code Analyst",
                ToolCategory.MEMORY: "Context Manager"
            }
            role = role_map.get(primary.category, "General Assistant")
        else:
            role = "General Assistant"

        # Communication style based on task type
        if "verify" in task_description.lower():
            style = "Precise and evidence-based. Always cite sources."
        elif "synthesize" in task_description.lower():
            style = "Clear and comprehensive. Organize information logically."
        elif "analyze" in task_description.lower():
            style = "Analytical and thorough. Highlight key insights."
        else:
            style = "Helpful and informative. Focus on actionable information."

        return ActorPersona(
            role=role,
            expertise=expertise,
            communication_style=style,
            constraints=[
                "Only use available tools",
                "Report when information is unavailable",
                "Cite sources for factual claims"
            ]
        )

    def _compose_system_prompt(
        self,
        persona: ActorPersona,
        bundles: List[ToolBundle],
        memory_context: str
    ) -> str:
        """Compose complete system prompt for actor"""

        persona_prompt = persona.to_system_prompt()

        # Add available tools section
        tools_section = "## Available Tool Bundles\n"
        for bundle in bundles:
            tools_section += f"\n### {bundle.name}\n"
            tools_section += f"{bundle.description}\n"
            tools_section += f"Tools: {', '.join(bundle.tools)}\n"

        # Add memory context if available
        memory_section = ""
        if memory_context:
            memory_section = f"\n## Relevant Context\n{memory_context}\n"

        return f"{persona_prompt}\n\n{tools_section}{memory_section}"

    async def create_actor(
        self,
        task_description: str,
        memory_context: str = "",
        force_bundles: Optional[List[str]] = None,
        force_model: Optional[str] = None
    ) -> DynamicActor:
        """
        Create a specialized actor for a subtask.

        AIME Formula: A_t = {LLM_t, T_t, P_t, M_t}

        Args:
            task_description: Description of the subtask
            memory_context: Relevant context from scratchpad
            force_bundles: Override automatic bundle selection
            force_model: Override automatic model selection

        Returns:
            DynamicActor configured for the task
        """
        import hashlib

        # Analyze task requirements
        task_types = self._analyze_task_requirements(task_description)

        # Select bundles
        if force_bundles:
            bundles = [self.bundles[b] for b in force_bundles if b in self.bundles]
        else:
            bundles = self._select_bundles(task_types)

        # Get required capabilities
        capabilities = self._get_required_capabilities(bundles)

        # Select model
        if force_model:
            model = force_model
        else:
            model = await self._select_model(capabilities)

        # Generate persona
        persona = self._generate_persona(task_description, bundles)

        # Compose system prompt
        system_prompt = self._compose_system_prompt(persona, bundles, memory_context)

        # Generate actor ID
        actor_id = hashlib.md5(
            f"{task_description}:{model}:{[b.name for b in bundles]}".encode()
        ).hexdigest()[:8]

        # Create actor
        actor = DynamicActor(
            id=actor_id,
            model=model,
            bundles=bundles,
            persona=persona,
            system_prompt=system_prompt,
            memory_context=memory_context
        )

        self.actors[actor_id] = actor
        self._stats["actors_created"] += 1

        logger.info(
            f"Created actor {actor_id}: {persona.role} with "
            f"{len(bundles)} bundles, model={model}"
        )

        return actor

    def get_actor(self, actor_id: str) -> Optional[DynamicActor]:
        """Get an existing actor by ID"""
        return self.actors.get(actor_id)

    def record_execution(self, actor_id: str, success: bool) -> None:
        """Record execution result for an actor"""
        if actor_id in self.actors:
            actor = self.actors[actor_id]
            actor.executions += 1
            if success:
                actor.successes += 1
            self._stats["total_executions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            **self._stats,
            "total_actors": len(self.actors),
            "registered_bundles": len(self.bundles),
            "registered_tools": len(self.tools),
            "actor_success_rates": {
                aid: actor.success_rate
                for aid, actor in self.actors.items()
                if actor.executions > 0
            }
        }

    def get_bundle_info(self) -> List[Dict[str, Any]]:
        """Get information about all available bundles"""
        return [bundle.to_dict() for bundle in self.bundles.values()]

    def clear_actors(self) -> int:
        """Clear all actors and return count cleared"""
        count = len(self.actors)
        self.actors.clear()
        return count


# Factory function
def create_actor_factory(
    ollama_url: str = "http://localhost:11434",
    default_model: str = "qwen3:8b"
) -> ActorFactory:
    """Create a new ActorFactory instance"""
    return ActorFactory(ollama_url=ollama_url, default_model=default_model)


# Singleton instance
_actor_factory: Optional[ActorFactory] = None


def get_actor_factory(
    ollama_url: str = "http://localhost:11434",
    default_model: str = "qwen3:8b"
) -> ActorFactory:
    """Get or create the singleton ActorFactory instance"""
    global _actor_factory
    if _actor_factory is None:
        _actor_factory = create_actor_factory(ollama_url, default_model)
    return _actor_factory
