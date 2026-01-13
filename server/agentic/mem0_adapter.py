"""
Mem0 Adapter for Agentic Pipeline Integration

This adapter bridges the Mem0 memory layer with memOS's agentic search pipeline,
providing automated fact extraction from conversations, cross-session entity
tracking, and memory consolidation.

Architecture:
    Conversation → Mem0 Extraction → memOS Storage (HIPAA-compliant)
                        ↓
    Query → Mem0 Search → Context Augmentation → Orchestrator

Key Features:
- Routes LLM calls through Gateway for VRAM management
- Automated fact extraction from search interactions
- Cross-turn entity resolution for multi-turn queries
- Memory consolidation with smart UPDATE/NOOP decisions

Usage:
    from agentic.mem0_adapter import AgenticMemoryAdapter

    adapter = AgenticMemoryAdapter(user_id="user123")

    # Extract and store facts from conversation
    await adapter.process_interaction(
        query="How do I fix SRVO-063?",
        response="SRVO-063 indicates servo disconnect...",
        context={"preset": "RESEARCH", "domain": "FANUC"}
    )

    # Get relevant context for new query
    context = await adapter.get_context_for_query(
        query="What about SRVO-062?",
        limit=5
    )

See: docs/MEM0_INTEGRATION_REPORT.md for design decisions.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories for extracted memories."""
    PREFERENCE = "preference"
    DOMAIN = "domain"
    EXPERTISE = "expertise"
    EQUIPMENT = "equipment"
    PROBLEM_TYPE = "problem_type"
    APPROACH = "approach"
    ENTITY = "entity"


@dataclass
class ExtractedMemory:
    """A memory extracted from conversation."""
    content: str
    category: MemoryCategory
    confidence: float
    source_query: str
    source_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationContext:
    """Context retrieved from memory for query augmentation."""
    memories: List[Dict[str, Any]]
    entities: List[str]
    user_preferences: Dict[str, Any]
    relevant_domains: List[str]
    total_score: float


class AgenticMemoryAdapter:
    """
    Adapter bridging Mem0 with the agentic search pipeline.

    Provides:
    1. Automated memory extraction from search interactions
    2. Cross-turn entity tracking for conversation continuity
    3. User preference learning for personalization
    4. Context augmentation for improved search quality
    """

    def __init__(
        self,
        user_id: str,
        use_gateway: bool = True,
        collection_name: str = "agentic_memories",
        auto_extract: bool = True,
        min_confidence: float = 0.7,
    ):
        """
        Initialize the agentic memory adapter.

        Args:
            user_id: User identifier for memory isolation
            use_gateway: Route LLM calls through Gateway (recommended)
            collection_name: Qdrant collection for memories
            auto_extract: Automatically extract facts from interactions
            min_confidence: Minimum confidence for storing extracted memories
        """
        self.user_id = user_id
        self.use_gateway = use_gateway
        self.collection_name = collection_name
        self.auto_extract = auto_extract
        self.min_confidence = min_confidence

        # Lazy initialization
        self._mem0_client = None
        self._initialized = False

        # Entity tracking for current conversation
        self._current_entities: List[str] = []
        self._conversation_id: Optional[str] = None

    async def initialize(self) -> bool:
        """
        Initialize the Mem0 client lazily.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            from .mem0_config import get_mem0_instance

            self._mem0_client = get_mem0_instance(
                use_gateway=self.use_gateway,
                collection_name=self.collection_name,
            )
            self._initialized = True
            logger.info(f"AgenticMemoryAdapter initialized for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}")
            return False

    async def process_interaction(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> List[ExtractedMemory]:
        """
        Process a search interaction, extracting and storing relevant memories.

        This is the main entry point for the agentic pipeline to report
        successful searches for learning.

        Args:
            query: The user's query
            response: The synthesized response
            context: Additional context (preset, domain, confidence, etc.)
            conversation_id: Conversation ID for cross-turn tracking

        Returns:
            List of extracted memories
        """
        if not self.auto_extract:
            return []

        if not await self.initialize():
            logger.warning("Mem0 not available, skipping memory extraction")
            return []

        context = context or {}
        self._conversation_id = conversation_id

        extracted = []

        # 1. Extract domain information
        domain_memory = await self._extract_domain_info(query, context)
        if domain_memory:
            extracted.append(domain_memory)

        # 2. Extract entities for cross-turn tracking
        entities = await self._extract_entities(query, response)
        self._current_entities.extend(entities)

        # 3. Extract user preferences from interaction pattern
        if "preset" in context:
            pref_memory = await self._extract_preference(
                f"Used {context['preset']} preset for: {query[:50]}",
                context
            )
            if pref_memory:
                extracted.append(pref_memory)

        # 4. Store high-confidence memories
        for memory in extracted:
            if memory.confidence >= self.min_confidence:
                await self._store_memory(memory)

        return extracted

    async def get_context_for_query(
        self,
        query: str,
        limit: int = 5,
        include_entities: bool = True,
    ) -> ConversationContext:
        """
        Retrieve relevant context from memory for a query.

        Used by the orchestrator to augment queries with:
        - Relevant past interactions
        - User preferences
        - Cross-turn entity resolution

        Args:
            query: The current query
            limit: Maximum memories to retrieve
            include_entities: Include entity resolution

        Returns:
            ConversationContext with relevant memories and metadata
        """
        if not await self.initialize():
            return ConversationContext(
                memories=[],
                entities=[],
                user_preferences={},
                relevant_domains=[],
                total_score=0.0,
            )

        try:
            # Search for relevant memories
            results = self._mem0_client.search(
                query=query,
                user_id=self.user_id,
                limit=limit,
            )

            memories = results.get("results", []) if isinstance(results, dict) else results

            # Extract entities from current conversation tracking
            entities = self._current_entities.copy()

            # Extract domains and preferences from memories
            domains = set()
            preferences = {}

            for mem in memories:
                metadata = mem.get("metadata", {})
                if metadata.get("category") == "domain":
                    domains.add(metadata.get("domain", "general"))
                elif metadata.get("category") == "preference":
                    pref_key = metadata.get("preference_type", "general")
                    preferences[pref_key] = mem.get("memory", "")

            # Calculate aggregate score
            total_score = sum(
                mem.get("score", 0.5) for mem in memories
            ) / max(len(memories), 1)

            return ConversationContext(
                memories=memories,
                entities=entities,
                user_preferences=preferences,
                relevant_domains=list(domains),
                total_score=total_score,
            )

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return ConversationContext(
                memories=[],
                entities=[],
                user_preferences={},
                relevant_domains=[],
                total_score=0.0,
            )

    async def resolve_entity(self, reference: str) -> Optional[str]:
        """
        Resolve a pronoun or reference to a concrete entity.

        Used for cross-turn entity resolution:
        - "it" → "SRVO-063"
        - "that error" → "MOTN-023"
        - "the robot" → "FANUC R-2000iC"

        Args:
            reference: The reference to resolve (e.g., "it", "that")

        Returns:
            Resolved entity name or None if unresolvable
        """
        if not self._current_entities:
            return None

        # Simple heuristic: return most recently mentioned entity
        # In production, this should use LLM for smarter resolution
        reference_lower = reference.lower()

        if reference_lower in ["it", "this", "that"]:
            return self._current_entities[-1] if self._current_entities else None

        if "error" in reference_lower:
            # Find most recent error code
            for entity in reversed(self._current_entities):
                if any(code in entity.upper() for code in ["SRVO", "MOTN", "SYST", "INTP"]):
                    return entity

        return None

    async def add_explicit_memory(
        self,
        content: str,
        category: MemoryCategory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Explicitly add a memory (for manual additions).

        Args:
            content: Memory content
            category: Memory category
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        if not await self.initialize():
            return False

        memory = ExtractedMemory(
            content=content,
            category=category,
            confidence=1.0,  # Explicit memories are fully confident
            source_query="explicit",
            metadata=metadata or {},
        )

        return await self._store_memory(memory)

    async def clear_conversation_context(self):
        """Clear the current conversation entity tracking."""
        self._current_entities = []
        self._conversation_id = None

    async def get_user_profile(self) -> Dict[str, Any]:
        """
        Get aggregated user profile from memories.

        Returns:
            Dictionary with user preferences, domains, expertise level, etc.
        """
        if not await self.initialize():
            return {}

        try:
            # Get all user memories
            all_memories = self._mem0_client.get_all(user_id=self.user_id)
            memories = all_memories.get("results", []) if isinstance(all_memories, dict) else all_memories

            profile = {
                "domains": [],
                "preferences": {},
                "expertise_indicators": [],
                "equipment": [],
                "common_problems": [],
                "memory_count": len(memories),
            }

            for mem in memories:
                metadata = mem.get("metadata", {})
                category = metadata.get("category")
                content = mem.get("memory", "")

                if category == "domain":
                    profile["domains"].append(metadata.get("domain", content))
                elif category == "preference":
                    pref_type = metadata.get("preference_type", "general")
                    profile["preferences"][pref_type] = content
                elif category == "expertise":
                    profile["expertise_indicators"].append(content)
                elif category == "equipment":
                    profile["equipment"].append(content)
                elif category == "problem_type":
                    profile["common_problems"].append(content)

            # Deduplicate
            profile["domains"] = list(set(profile["domains"]))
            profile["equipment"] = list(set(profile["equipment"]))

            return profile

        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _store_memory(self, memory: ExtractedMemory) -> bool:
        """Store an extracted memory in Mem0."""
        try:
            metadata = {
                "category": memory.category.value,
                "confidence": memory.confidence,
                "source_query": memory.source_query[:200],
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata,
            }

            self._mem0_client.add(
                memory.content,
                user_id=self.user_id,
                metadata=metadata,
            )

            logger.debug(f"Stored memory: {memory.content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def _extract_domain_info(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[ExtractedMemory]:
        """Extract domain information from query."""
        # Domain detection patterns
        domain_patterns = {
            "fanuc": ["fanuc", "srvo", "motn", "syst", "intp", "r-2000", "m-20"],
            "allen_bradley": ["allen-bradley", "rockwell", "plc5", "controllogix", "1756"],
            "siemens": ["siemens", "s7-", "step7", "tia portal", "profinet"],
            "imm": ["injection", "molding", "barrel", "screw", "clamp", "platen"],
            "raspberry_pi": ["raspberry", "rpi", "gpio", "python", "linux embedded"],
        }

        query_lower = query.lower()
        detected_domain = None
        confidence = 0.0

        for domain, patterns in domain_patterns.items():
            matches = sum(1 for p in patterns if p in query_lower)
            if matches > 0:
                domain_confidence = min(0.5 + matches * 0.15, 0.95)
                if domain_confidence > confidence:
                    detected_domain = domain
                    confidence = domain_confidence

        if detected_domain and confidence >= self.min_confidence:
            return ExtractedMemory(
                content=f"User queries about {detected_domain.upper()} systems",
                category=MemoryCategory.DOMAIN,
                confidence=confidence,
                source_query=query,
                metadata={"domain": detected_domain},
            )

        return None

    async def _extract_preference(
        self,
        observation: str,
        context: Dict[str, Any]
    ) -> Optional[ExtractedMemory]:
        """Extract user preference from interaction pattern."""
        preset = context.get("preset", "").upper()

        if preset in ["RESEARCH", "FULL"]:
            return ExtractedMemory(
                content=f"User prefers detailed/thorough responses",
                category=MemoryCategory.PREFERENCE,
                confidence=0.75,
                source_query=observation,
                metadata={"preference_type": "detail_level", "preset": preset},
            )
        elif preset == "MINIMAL":
            return ExtractedMemory(
                content=f"User prefers quick, concise responses",
                category=MemoryCategory.PREFERENCE,
                confidence=0.75,
                source_query=observation,
                metadata={"preference_type": "detail_level", "preset": preset},
            )

        return None

    async def _extract_entities(
        self,
        query: str,
        response: str
    ) -> List[str]:
        """Extract entities from query and response for cross-turn tracking."""
        import re

        entities = []

        # Error code patterns
        error_patterns = [
            r'\b(SRVO-\d+)\b',
            r'\b(MOTN-\d+)\b',
            r'\b(SYST-\d+)\b',
            r'\b(INTP-\d+)\b',
            r'\b(SPOT-\d+)\b',
            r'\b(WELD-\d+)\b',
        ]

        # Part number patterns
        part_patterns = [
            r'\b(A06B-\d{4}-\w+)\b',  # FANUC
            r'\b(A860-\d{4}-\w+)\b',  # FANUC encoders
            r'\b(1756-\w+)\b',         # Allen-Bradley
            r'\b(6ES7\d+-\w+)\b',      # Siemens
        ]

        # Equipment patterns
        equipment_patterns = [
            r'\b(R-2000i[A-Z/]+)\b',   # FANUC robots
            r'\b(M-20i[A-Z/]+)\b',
            r'\b(LR Mate \d+i[A-Z/]+)\b',
        ]

        all_patterns = error_patterns + part_patterns + equipment_patterns
        combined_text = f"{query} {response}"

        for pattern in all_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            entities.extend(matches)

        return list(set(entities))


# Singleton instance for shared use
_adapter_instances: Dict[str, AgenticMemoryAdapter] = {}


def get_adapter_for_user(
    user_id: str,
    use_gateway: bool = True,
) -> AgenticMemoryAdapter:
    """
    Get or create an AgenticMemoryAdapter for a user.

    Maintains singleton instances per user for conversation continuity.

    Args:
        user_id: User identifier
        use_gateway: Route through LLM Gateway

    Returns:
        AgenticMemoryAdapter instance
    """
    if user_id not in _adapter_instances:
        _adapter_instances[user_id] = AgenticMemoryAdapter(
            user_id=user_id,
            use_gateway=use_gateway,
        )
    return _adapter_instances[user_id]


def clear_adapter_for_user(user_id: str):
    """Clear adapter instance for a user (e.g., on logout)."""
    if user_id in _adapter_instances:
        del _adapter_instances[user_id]
