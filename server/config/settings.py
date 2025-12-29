"""
memOS Server Settings Configuration
HIPAA-compliant configuration management for Recovery Bot memory system
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path


class MemOSSettings(BaseSettings):
    """Configuration settings for memOS server"""
    
    # Server Configuration
    host: str = "localhost"
    port: int = 8001
    debug: bool = False
    environment: str = "development"
    
    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "memos_recovery_bot"
    postgres_user: str = "memos_user"
    postgres_password: str = "change_this_password"
    
    # Vector Database
    vector_store: str = "pgvector"  # pgvector, chroma, or qdrant
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "mxbai-embed-large"
    embedding_dimensions: int = 1024
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 2
    redis_password: Optional[str] = None
    
    # Ollama Integration
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "llama3.3:70b"
    ollama_embedding_model: str = "mxbai-embed-large"

    # PDF Extraction Tools API (FANUC Technical Documentation)
    pdf_api_url: str = "http://localhost:8002"
    pdf_api_timeout: int = 30
    pdf_api_enabled: bool = True
    pdf_api_max_results: int = 10
    pdf_api_cache_ttl: int = 300  # 5 minutes

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
    
    # Monitoring
    enable_prometheus_metrics: bool = True
    metrics_port: int = 8002
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