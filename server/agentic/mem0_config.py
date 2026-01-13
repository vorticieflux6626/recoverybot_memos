"""
Mem0 Configuration for memOS Integration

This module provides configuration for the Mem0 memory layer, routing all LLM calls
through the LLM Gateway for VRAM management, priority scheduling, and fallback chains.

Usage:
    from agentic.mem0_config import get_mem0_config, get_mem0_instance

    # Get configured Memory instance
    memory = get_mem0_instance()

    # Add memories
    memory.add("User prefers FANUC troubleshooting", user_id="user123")

    # Search memories
    results = memory.search("What does user prefer?", user_id="user123")

See: docs/MEM0_INTEGRATION_REPORT.md for architecture decisions.
"""

import os
from typing import Optional
from functools import lru_cache

# Default configuration values
GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8100")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Route through Gateway for VRAM management and priority scheduling
# Gateway now has /v1/models endpoint (added 2026-01-13) for Mem0 compatibility
USE_GATEWAY = os.getenv("MEM0_USE_GATEWAY", "true").lower() == "true"


def get_mem0_config(
    use_gateway: bool = USE_GATEWAY,
    collection_name: str = "user_memories",
    llm_model: str = "qwen3:8b",
    embedding_model: str = "nomic-embed-text",
    embedding_dims: int = 768,
    enable_graph: bool = False,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> dict:
    """
    Get Mem0 configuration dictionary.

    Args:
        use_gateway: Route LLM calls through Gateway (recommended for VRAM management)
        collection_name: Qdrant collection name for memories
        llm_model: Model for memory extraction (qwen3:8b recommended for speed)
        embedding_model: Model for embeddings (nomic-embed-text for 768d compatibility)
        embedding_dims: Embedding dimensions (must match model output)
        enable_graph: Enable Mem0g graph-based memory (requires Neo4j)
        neo4j_url: Neo4j connection URL (if enable_graph=True)
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Configuration dictionary for Memory.from_config()
    """
    # Determine LLM endpoint
    llm_base_url = GATEWAY_URL if use_gateway else OLLAMA_URL

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.1,  # Low temp for consistent fact extraction
                "max_tokens": 1000,
                "ollama_base_url": llm_base_url,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embedding_model,
                "ollama_base_url": llm_base_url,
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "embedding_model_dims": embedding_dims,
            }
        },
        # Custom prompts for better extraction quality
        "custom_prompt": get_extraction_prompt(),
    }

    # Add graph store if enabled (Mem0g for entity relationships)
    if enable_graph and neo4j_url:
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_user or "neo4j",
                "password": neo4j_password or "password",
            }
        }

    return config


def get_extraction_prompt() -> str:
    """
    Custom extraction prompt optimized for technical/industrial domain.

    Returns higher quality memories for:
    - User preferences for troubleshooting approaches
    - Equipment and system expertise levels
    - Common query patterns and domains
    """
    return """You are a memory extraction assistant for a technical research platform.
Extract factual information about the user from the conversation.

Focus on extracting:
1. User preferences (response length, detail level, preset preferences)
2. Technical domains of interest (FANUC, Allen-Bradley, Siemens, IMM, etc.)
3. Expertise level indicators (beginner, intermediate, expert)
4. Equipment/systems the user works with
5. Common problem types they encounter
6. Preferred troubleshooting approaches

Do NOT extract:
- Temporary/session-specific information
- Generic conversational pleasantries
- Information about the AI assistant itself

Output format: Return a JSON list of memory objects, each with:
- "memory": The factual statement to remember
- "category": One of [preference, domain, expertise, equipment, problem_type, approach]

Example:
[
  {"memory": "User frequently troubleshoots FANUC servo errors", "category": "domain"},
  {"memory": "User prefers detailed step-by-step explanations", "category": "preference"}
]
"""


def get_mem0_instance(
    use_gateway: bool = USE_GATEWAY,
    collection_name: str = "user_memories",
    **kwargs
):
    """
    Get a configured Mem0 Memory instance.

    This is the primary entry point for using Mem0 in memOS.

    Args:
        use_gateway: Route through LLM Gateway (default: True)
        collection_name: Qdrant collection for memories
        **kwargs: Additional config options passed to get_mem0_config()

    Returns:
        Configured mem0.Memory instance

    Example:
        memory = get_mem0_instance()
        memory.add("User prefers FANUC docs", user_id="user123")
        results = memory.search("FANUC", user_id="user123")
    """
    try:
        from mem0 import Memory
    except ImportError:
        raise ImportError(
            "mem0ai not installed. Run: pip install mem0ai"
        )

    config = get_mem0_config(
        use_gateway=use_gateway,
        collection_name=collection_name,
        **kwargs
    )

    return Memory.from_config(config)


@lru_cache(maxsize=1)
def get_shared_mem0_instance():
    """
    Get a shared/singleton Mem0 instance for the application.

    Use this when you need a consistent Memory instance across
    multiple calls without creating new connections.

    Returns:
        Cached mem0.Memory instance
    """
    return get_mem0_instance()


# Configuration presets for different use cases
PRESETS = {
    "user_preferences": {
        "collection_name": "user_preferences",
        "llm_model": "qwen3:8b",
        "embedding_model": "nomic-embed-text",
        "embedding_dims": 768,
    },
    "conversation_context": {
        "collection_name": "conversation_context",
        "llm_model": "qwen3:8b",
        "embedding_model": "nomic-embed-text",
        "embedding_dims": 768,
    },
    "experience_templates": {
        "collection_name": "experience_templates",
        "llm_model": "qwen3:8b",
        "embedding_model": "mxbai-embed-large",
        "embedding_dims": 1024,
    },
}


def get_preset_config(preset_name: str) -> dict:
    """
    Get a predefined configuration preset.

    Available presets:
    - user_preferences: Store user preference facts
    - conversation_context: Cross-turn entity tracking
    - experience_templates: Successful search pattern storage

    Args:
        preset_name: Name of the preset

    Returns:
        Configuration dictionary
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    return get_mem0_config(**PRESETS[preset_name])
