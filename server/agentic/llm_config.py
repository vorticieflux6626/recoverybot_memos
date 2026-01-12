"""
Centralized LLM Model Configuration Loader

Loads model assignments from config/llm_models.yaml and provides
a unified interface for all pipeline components.

Usage:
    from agentic.llm_config import get_llm_config, get_model_for_task

    # Get full config
    config = get_llm_config()
    model = config.pipeline.analyzer.model

    # Get model for specific task
    model = get_model_for_task("url_evaluator")
    model = get_model_for_task("pipeline.synthesizer")

    # Apply a preset
    config.apply_preset("speed")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

# Default config path
CONFIG_PATH = Path(__file__).parent.parent / "config" / "llm_models.yaml"


@dataclass
class ModelConfig:
    """Configuration for a single model assignment."""
    model: str
    context_window: int = 32000
    temperature: float = 0.3
    max_tokens: int = 1024
    description: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_window": self.context_window,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "description": self.description,
            "notes": self.notes,
        }


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model: str
    dimensions: int = 768
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "description": self.description,
        }


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    url: str = "http://localhost:11434"
    default_timeout: int = 120
    keep_alive: str = "30m"


@dataclass
class PipelineModels:
    """Model assignments for pipeline stages (benchmark-optimized 2026-01-12)."""
    analyzer: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))  # 0.91 acc, 5091ms
    url_evaluator: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))  # 0.66 acc, 5712ms
    coverage_evaluator: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    planner: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    synthesizer: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    thinking: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    retrieval_evaluator: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:4b-instruct-2507-q8_0"))  # 0.83 acc, 4513ms
    self_reflection: ModelConfig = field(default_factory=lambda: ModelConfig(model="cogito:8b"))  # 1.0 acc, 5018ms
    verifier: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))  # 0.75 acc, 5411ms


@dataclass
class UtilityModels:
    """Model assignments for utility tasks."""
    entity_extractor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    query_decomposer: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    relevance_scorer: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    uncertainty_detector: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    experience_distiller: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    prompt_compressor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen2.5:0.5b"))
    raptor_summarizer: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    graph_extractor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    graph_summarizer: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    cross_domain_validator: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    entity_grounder: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    # Reasoning components
    reasoning_composer: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    reasoning_dag: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    enhanced_planner: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    enhanced_reflector: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    # Retrieval components
    cross_encoder: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    hyde_generator: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    flare_detector: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    information_bottleneck: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    sufficient_context: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    self_consistency: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    speculative_verifier: ModelConfig = field(default_factory=lambda: ModelConfig(model="deepseek-r1:14b-qwen-distill-q8_0"))
    ragas_judge: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    entropy_monitor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    scraper_analyzer: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:14b"))
    actor_factory: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    adaptive_refinement: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    information_gain: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))


@dataclass
class EmbeddingModels:
    """Embedding model assignments."""
    primary: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(model="qwen3-embedding:latest"))
    cache: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(model="nomic-embed-text"))
    shadow: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(model="qwen3-embedding:0.6b"))


@dataclass
class CorpusModels:
    """Model assignments for corpus building."""
    plc_extractor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))
    imm_extractor: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:4b"))
    domain_extractor: ModelConfig = field(default_factory=lambda: ModelConfig(model="qwen3:8b"))


class LLMConfig:
    """
    Central LLM configuration manager.

    Loads model assignments from YAML and provides unified access
    for all pipeline components.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.version = "1.0.0"
        self.last_updated = ""

        # Initialize with defaults
        self.ollama = OllamaConfig()
        self.pipeline = PipelineModels()
        self.utility = UtilityModels()
        self.embeddings = EmbeddingModels()
        self.corpus = CorpusModels()
        self.presets: Dict[str, Dict[str, str]] = {}
        self.benchmarks: Dict[str, list] = {}

        # Load from YAML
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Empty config file, using defaults")
                return

            self.version = data.get("version", "1.0.0")
            self.last_updated = data.get("last_updated", "")

            # Load Ollama config
            if "ollama" in data:
                ollama_data = data["ollama"]
                self.ollama = OllamaConfig(
                    url=ollama_data.get("url", "http://localhost:11434"),
                    default_timeout=ollama_data.get("default_timeout", 120),
                    keep_alive=ollama_data.get("keep_alive", "30m"),
                )

            # Load pipeline models
            if "pipeline" in data:
                self._load_pipeline_models(data["pipeline"])

            # Load utility models
            if "utility" in data:
                self._load_utility_models(data["utility"])

            # Load embedding models
            if "embeddings" in data:
                self._load_embedding_models(data["embeddings"])

            # Load corpus models
            if "corpus" in data:
                self._load_corpus_models(data["corpus"])

            # Load presets
            self.presets = data.get("presets", {})

            # Load benchmarks reference
            self.benchmarks = data.get("benchmarks", {})

            logger.info(f"Loaded LLM config v{self.version} from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading LLM config: {e}, using defaults")

    def _load_pipeline_models(self, pipeline_data: Dict[str, Any]):
        """Load pipeline model configurations."""
        for key, value in pipeline_data.items():
            if hasattr(self.pipeline, key) and isinstance(value, dict):
                setattr(self.pipeline, key, ModelConfig(
                    model=value.get("model", "qwen3:8b"),
                    context_window=value.get("context_window", 32000),
                    temperature=value.get("temperature", 0.3),
                    max_tokens=value.get("max_tokens", 1024),
                    description=value.get("description", ""),
                    notes=value.get("notes", ""),
                ))

    def _load_utility_models(self, utility_data: Dict[str, Any]):
        """Load utility model configurations."""
        for key, value in utility_data.items():
            if hasattr(self.utility, key) and isinstance(value, dict):
                setattr(self.utility, key, ModelConfig(
                    model=value.get("model", "qwen3:8b"),
                    context_window=value.get("context_window", 32000),
                    temperature=value.get("temperature", 0.3),
                    max_tokens=value.get("max_tokens", 1024),
                    description=value.get("description", ""),
                    notes=value.get("notes", ""),
                ))

    def _load_embedding_models(self, embedding_data: Dict[str, Any]):
        """Load embedding model configurations."""
        for key, value in embedding_data.items():
            if hasattr(self.embeddings, key) and isinstance(value, dict):
                setattr(self.embeddings, key, EmbeddingConfig(
                    model=value.get("model", "nomic-embed-text"),
                    dimensions=value.get("dimensions", 768),
                    description=value.get("description", ""),
                ))

    def _load_corpus_models(self, corpus_data: Dict[str, Any]):
        """Load corpus model configurations."""
        for key, value in corpus_data.items():
            if hasattr(self.corpus, key) and isinstance(value, dict):
                setattr(self.corpus, key, ModelConfig(
                    model=value.get("model", "qwen3:8b"),
                    context_window=value.get("context_window", 32000),
                    temperature=value.get("temperature", 0.3),
                    max_tokens=value.get("max_tokens", 2048),
                    description=value.get("description", ""),
                    notes=value.get("notes", ""),
                ))

    def get_model(self, task: str) -> str:
        """
        Get the model name for a specific task.

        Args:
            task: Task identifier (e.g., "url_evaluator", "pipeline.synthesizer")

        Returns:
            Model name string
        """
        # Handle dotted notation (e.g., "pipeline.analyzer")
        if "." in task:
            category, task_name = task.split(".", 1)
        else:
            # Default to pipeline category
            category = "pipeline"
            task_name = task

        # Look up in appropriate category
        if category == "pipeline" and hasattr(self.pipeline, task_name):
            return getattr(self.pipeline, task_name).model
        elif category == "utility" and hasattr(self.utility, task_name):
            return getattr(self.utility, task_name).model
        elif category == "embeddings" and hasattr(self.embeddings, task_name):
            return getattr(self.embeddings, task_name).model
        elif category == "corpus" and hasattr(self.corpus, task_name):
            return getattr(self.corpus, task_name).model

        # Fallback
        logger.warning(f"Unknown task '{task}', using default model")
        return "qwen3:8b"

    def get_config(self, task: str) -> ModelConfig:
        """
        Get the full config for a specific task.

        Args:
            task: Task identifier

        Returns:
            ModelConfig object
        """
        if "." in task:
            category, task_name = task.split(".", 1)
        else:
            category = "pipeline"
            task_name = task

        if category == "pipeline" and hasattr(self.pipeline, task_name):
            return getattr(self.pipeline, task_name)
        elif category == "utility" and hasattr(self.utility, task_name):
            return getattr(self.utility, task_name)
        elif category == "corpus" and hasattr(self.corpus, task_name):
            return getattr(self.corpus, task_name)

        # Return default
        return ModelConfig(model="qwen3:8b")

    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply a preset configuration.

        Args:
            preset_name: Name of preset (e.g., "speed", "quality", "balanced")

        Returns:
            True if preset was applied successfully
        """
        if preset_name not in self.presets:
            logger.warning(f"Unknown preset: {preset_name}")
            return False

        preset = self.presets[preset_name]
        for task, model in preset.items():
            if "." in task:
                category, task_name = task.split(".", 1)
            else:
                continue

            if category == "pipeline" and hasattr(self.pipeline, task_name):
                config = getattr(self.pipeline, task_name)
                config.model = model
            elif category == "utility" and hasattr(self.utility, task_name):
                config = getattr(self.utility, task_name)
                config.model = model

        logger.info(f"Applied preset: {preset_name}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary (for API responses)."""
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "config_path": str(self.config_path),
            "ollama": {
                "url": self.ollama.url,
                "default_timeout": self.ollama.default_timeout,
                "keep_alive": self.ollama.keep_alive,
            },
            "pipeline": {
                name: getattr(self.pipeline, name).to_dict()
                for name in ["analyzer", "url_evaluator", "coverage_evaluator",
                            "planner", "synthesizer", "thinking",
                            "retrieval_evaluator", "self_reflection", "verifier"]
            },
            "utility": {
                name: getattr(self.utility, name).to_dict()
                for name in ["entity_extractor", "query_decomposer", "relevance_scorer",
                            "uncertainty_detector", "experience_distiller", "prompt_compressor",
                            "raptor_summarizer", "graph_extractor", "graph_summarizer",
                            "cross_domain_validator", "entity_grounder",
                            # Reasoning components
                            "reasoning_composer", "reasoning_dag", "enhanced_planner", "enhanced_reflector",
                            # Retrieval components
                            "cross_encoder", "hyde_generator", "flare_detector", "information_bottleneck",
                            "sufficient_context", "self_consistency", "speculative_verifier", "ragas_judge",
                            "entropy_monitor", "scraper_analyzer", "actor_factory", "adaptive_refinement",
                            "information_gain"]
            },
            "embeddings": {
                name: getattr(self.embeddings, name).to_dict()
                for name in ["primary", "cache", "shadow"]
            },
            "corpus": {
                name: getattr(self.corpus, name).to_dict()
                for name in ["plc_extractor", "imm_extractor", "domain_extractor"]
            },
            "presets": list(self.presets.keys()),
            "benchmarks": self.benchmarks,
        }

    def reload(self):
        """Reload configuration from file."""
        self._load_config()
        logger.info("LLM config reloaded")


# Global singleton instance
_llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM config instance."""
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config


def get_model_for_task(task: str) -> str:
    """
    Convenience function to get model name for a task.

    Args:
        task: Task identifier (e.g., "url_evaluator", "pipeline.synthesizer")

    Returns:
        Model name string
    """
    return get_llm_config().get_model(task)


def get_config_for_task(task: str) -> ModelConfig:
    """
    Convenience function to get full config for a task.

    Args:
        task: Task identifier

    Returns:
        ModelConfig object with model, temperature, max_tokens, etc.
    """
    return get_llm_config().get_config(task)


def reload_llm_config():
    """Reload the global LLM config from file."""
    global _llm_config
    if _llm_config is not None:
        _llm_config.reload()
    else:
        _llm_config = LLMConfig()
