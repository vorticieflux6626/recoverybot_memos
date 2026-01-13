"""
memOS Server Settings Configuration
HIPAA-compliant configuration management for Recovery Bot memory system

Port defaults are loaded from the central ecosystem configuration:
    /home/sparkone/sdd/ecosystem_config/ports.yaml
"""

import os
import sys
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path

# Add ecosystem_config to path for imports
sys.path.insert(0, '/home/sparkone/sdd')

try:
    from ecosystem_config import (
        MEMOS_PORT,
        GATEWAY_PORT,
        PDF_TOOLS_PORT,
        SEARXNG_PORT,
        OLLAMA_PORT,
        MCP_NODE_EDITOR_PORT,
        POSTGRES_PORT,
        REDIS_PORT,
        PROMETHEUS_PORT,
        GRAFANA_PORT,
    )
except ImportError:
    # Fallback defaults if ecosystem_config not available
    MEMOS_PORT = 8001
    GATEWAY_PORT = 8100
    PDF_TOOLS_PORT = 8002
    SEARXNG_PORT = 8888
    OLLAMA_PORT = 11434
    MCP_NODE_EDITOR_PORT = 7777
    POSTGRES_PORT = 5432
    REDIS_PORT = 6379
    PROMETHEUS_PORT = 9090
    GRAFANA_PORT = 3000


class MemOSSettings(BaseSettings):
    """Configuration settings for memOS server"""
    
    # Server Configuration (ports from ecosystem_config)
    host: str = "localhost"
    port: int = MEMOS_PORT
    debug: bool = False
    environment: str = "development"

    # Database Configuration (ports from ecosystem_config)
    postgres_host: str = "localhost"
    postgres_port: int = POSTGRES_PORT
    postgres_db: str = "memos_recovery_bot"
    postgres_user: str = "memos_user"
    postgres_password: str = "change_this_password"

    # Vector Database
    vector_store: str = "pgvector"  # pgvector, chroma, or qdrant
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "mxbai-embed-large"
    embedding_dimensions: int = 1024

    # Redis Configuration (ports from ecosystem_config)
    redis_host: str = "localhost"
    redis_port: int = REDIS_PORT
    redis_db: int = 2
    redis_password: Optional[str] = None

    # Ollama Integration (ports from ecosystem_config)
    ollama_host: str = "localhost"
    ollama_port: int = OLLAMA_PORT
    ollama_model: str = "llama3.3:70b"
    ollama_embedding_model: str = "mxbai-embed-large"

    # Gateway Integration (ports from ecosystem_config)
    gateway_host: str = "localhost"
    gateway_port: int = GATEWAY_PORT

    # Agentic Search Configuration (ports from ecosystem_config)
    mcp_url: str = f"http://localhost:{MCP_NODE_EDITOR_PORT}"
    searxng_url: str = f"http://localhost:{SEARXNG_PORT}"
    classifier_model: str = "deepseek-r1:14b-qwen-distill-q8_0"  # Query classifier
    synthesizer_model: str = "qwen3:8b"  # Synthesis model
    thinking_model: str = "deepseek-r1:14b-qwen-distill-q8_0"  # Reasoning model
    brave_api_key: Optional[str] = None  # Brave Search API key (optional fallback)
    data_dir: str = "/home/sparkone/sdd/Recovery_Bot/memOS/data"  # Data directory

    # Model Fallback Configuration
    # Each key maps to a list of fallback models in priority order
    model_fallbacks: dict = {
        # Main LLM fallbacks (largest to smallest)
        "llama3.3:70b": ["qwen3:30b-a3b", "qwen3:8b", "llama3.2:3b"],
        # Thinking/reasoning model fallbacks
        "deepseek-r1:14b-qwen-distill-q8_0": ["deepseek-r1:8b-0528-qwen3-fp16", "qwen3:8b"],
        # Synthesis model fallbacks
        "qwen3:8b": ["llama3.2:3b", "llama3.2:1b-instruct-fp16"],
        # Embedding model fallbacks
        "mxbai-embed-large": ["bge-m3", "nomic-embed-text"],
    }

    # Model validation settings
    validate_models_on_startup: bool = True
    model_validation_timeout: float = 5.0  # seconds

    # PDF Extraction Tools API (ports from ecosystem_config)
    pdf_api_url: str = f"http://localhost:{PDF_TOOLS_PORT}"
    pdf_api_timeout: int = 30
    pdf_api_enabled: bool = True
    pdf_api_max_results: int = 10
    pdf_api_cache_ttl: int = 300  # 5 minutes

    # HTTP Timeout Configuration (seconds)
    default_http_timeout: float = 30.0  # General HTTP requests
    llm_request_timeout: float = 90.0   # LLM inference requests (longer)
    health_check_timeout: float = 5.0   # Quick health checks
    embedding_timeout: float = 60.0     # Embedding generation

    # Cache Configuration
    query_cache_ttl: int = 3600         # 1 hour for query result cache
    content_cache_ttl: int = 3600       # 1 hour for scraped content
    semantic_cache_threshold: float = 0.88  # Similarity threshold for cache hits
    max_cache_entries: int = 10000      # Max entries in content cache

    # Security Configuration
    jwt_secret_key: str = "recovery-bot-memOS-jwt-secret-key-for-authentication-2025"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    encryption_key: str = "RecoveryBot2025MemOSEncryptionKey"
    
    # Memory Configuration
    max_memories_per_user: int = 10000
    memory_retention_days: int = 2555  # 7 years HIPAA compliance
    default_memory_type: str = "conversational"
    auto_summarization_threshold: int = 1000
    
    # Privacy and HIPAA
    enable_audit_logging: bool = True
    audit_log_retention_years: int = 7
    enable_encryption_at_rest: bool = True
    enable_memory_anonymization: bool = True
    default_privacy_level: str = "balanced"
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # Storage Paths
    storage_path: str = "./storage"
    log_path: str = "./logs"
    backup_path: str = "./backups"
    
    # CORS Configuration
    enable_cors: bool = True
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000", 
        "https://deals.sparkonelabs.com"
    ]
    
    # Recovery Bot Integration
    recovery_bot_api_url: str = "https://deals.sparkonelabs.com/Recovery_Bot"
    recovery_bot_api_key: Optional[str] = None
    sync_with_recovery_bot: bool = True
    
    # Monitoring (ports from ecosystem_config)
    enable_prometheus_metrics: bool = True
    metrics_port: int = PROMETHEUS_PORT
    log_level: str = "INFO"
    structured_logging: bool = True
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def ollama_base_url(self) -> str:
        """Construct Ollama base URL"""
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def gateway_base_url(self) -> str:
        """Construct Gateway base URL"""
        return f"http://{self.gateway_host}:{self.gateway_port}"
    
    @field_validator("storage_path", "log_path", "backup_path")
    @classmethod
    def ensure_paths_exist(cls, v):
        """Ensure storage paths exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v):
        """Ensure JWT secret is secure enough"""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        if v == "your-super-secret-jwt-key-change-this":
            raise ValueError("JWT secret key must be changed from default value")
        return v
    
    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v):
        """Ensure encryption key is proper length"""
        if len(v) != 32:
            raise ValueError("Encryption key must be exactly 32 characters long")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "MEMOS_",
        "case_sensitive": False
    }


# Global settings instance
settings = MemOSSettings()


def get_settings() -> MemOSSettings:
    """Get settings instance (for dependency injection)"""
    return settings


# Model Validation Utilities
import httpx
import logging

logger = logging.getLogger(__name__)

_available_models_cache: Optional[set] = None


def get_available_ollama_models(force_refresh: bool = False) -> set:
    """
    Fetch available models from Ollama API.
    Caches result to avoid repeated API calls.

    Returns:
        Set of available model names (includes both with and without :latest suffix)
    """
    global _available_models_cache

    if _available_models_cache is not None and not force_refresh:
        return _available_models_cache

    try:
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=settings.model_validation_timeout
        )
        if response.status_code == 200:
            data = response.json()
            raw_names = {m["name"] for m in data.get("models", [])}

            # Also add versions without :latest suffix for easier matching
            # e.g., "mxbai-embed-large:latest" -> also include "mxbai-embed-large"
            normalized_names = set()
            for name in raw_names:
                normalized_names.add(name)
                if name.endswith(":latest"):
                    base_name = name.rsplit(":latest", 1)[0]
                    normalized_names.add(base_name)

            _available_models_cache = normalized_names
            logger.info(f"Ollama models available: {len(raw_names)} (normalized: {len(normalized_names)})")
            return _available_models_cache
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")

    return set()


def resolve_model(requested_model: str, fallback_key: Optional[str] = None) -> str:
    """
    Resolve a model name to an available model, using fallbacks if needed.

    Args:
        requested_model: The preferred model name
        fallback_key: Optional key in model_fallbacks dict (defaults to requested_model)

    Returns:
        The first available model from [requested_model, ...fallbacks]
    """
    available = get_available_ollama_models()

    # If validation is disabled or no models found, return requested
    if not settings.validate_models_on_startup or not available:
        return requested_model

    # Check if requested model is available
    if requested_model in available:
        return requested_model

    # Check fallbacks
    key = fallback_key or requested_model
    fallbacks = settings.model_fallbacks.get(key, [])

    for fallback in fallbacks:
        if fallback in available:
            logger.warning(
                f"Model '{requested_model}' not available, using fallback: '{fallback}'"
            )
            return fallback

    # No fallback available - log error and return requested (will fail at runtime)
    logger.error(
        f"Model '{requested_model}' not available and no fallbacks found. "
        f"Available models: {sorted(available)[:10]}..."
    )
    return requested_model


def validate_configured_models() -> dict:
    """
    Validate all configured models and return a status report.

    Returns:
        Dict with model availability status and resolved models
    """
    available = get_available_ollama_models(force_refresh=True)

    models_to_check = {
        "ollama_model": settings.ollama_model,
        "classifier_model": settings.classifier_model,
        "synthesizer_model": settings.synthesizer_model,
        "thinking_model": settings.thinking_model,
        "embedding_model": settings.ollama_embedding_model,
    }

    report = {
        "available_count": len(available),
        "models": {}
    }

    for name, model in models_to_check.items():
        is_available = model in available
        resolved = resolve_model(model) if not is_available else model

        report["models"][name] = {
            "configured": model,
            "available": is_available,
            "resolved": resolved,
            "using_fallback": resolved != model,
        }

    return report


def get_resolved_model_config() -> dict:
    """
    Get all model configurations with fallback resolution applied.

    Returns:
        Dict mapping config key to resolved model name
    """
    return {
        "ollama_model": resolve_model(settings.ollama_model),
        "classifier_model": resolve_model(settings.classifier_model),
        "synthesizer_model": resolve_model(settings.synthesizer_model),
        "thinking_model": resolve_model(settings.thinking_model),
        "embedding_model": resolve_model(settings.ollama_embedding_model),
    }


# Validation for production environment
def validate_production_config():
    """Validate configuration is ready for production deployment"""
    issues = []
    
    if settings.debug:
        issues.append("Debug mode should be disabled in production")
    
    if settings.jwt_secret_key == "your-super-secret-jwt-key-change-this":
        issues.append("JWT secret key must be changed from default")
    
    if settings.postgres_password == "change_this_password":
        issues.append("Database password must be changed from default")
    
    if not settings.enable_encryption_at_rest:
        issues.append("Encryption at rest should be enabled for HIPAA compliance")
    
    if not settings.enable_audit_logging:
        issues.append("Audit logging must be enabled for HIPAA compliance")
    
    if issues:
        raise ValueError(f"Production configuration issues: {'; '.join(issues)}")
    
    return True


if __name__ == "__main__":
    print("memOS Configuration:")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"PDF API URL: {settings.pdf_api_url} (enabled: {settings.pdf_api_enabled})")
    print(f"Storage Path: {settings.storage_path}")
    print(f"HIPAA Compliance: {settings.enable_audit_logging and settings.enable_encryption_at_rest}")