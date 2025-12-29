"""
Logging Configuration for memOS Server
HIPAA-compliant structured logging with audit trails
"""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime, timezone

from .settings import get_settings

settings = get_settings()


def setup_logging():
    """
    Set up comprehensive logging for memOS server
    Includes HIPAA-compliant audit logging
    """
    
    # Ensure log directory exists
    log_path = Path(settings.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Define log file paths
    main_log_file = log_path / "memos_server.log"
    audit_log_file = log_path / "audit.log"
    error_log_file = log_path / "errors.log"
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "audit": {
                "format": "%(asctime)s [AUDIT] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(main_log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "audit",
                "filename": str(audit_log_file),
                "maxBytes": 52428800,  # 50MB
                "backupCount": 20,  # Keep 20 files (HIPAA retention)
                "encoding": "utf8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(error_log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": settings.log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "memos_server": {
                "level": settings.log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "audit": {
                "level": "INFO",
                "handlers": ["audit_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING" if not settings.debug else "INFO",
                "handlers": ["file"],
                "propagate": False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log startup information
    logger = logging.getLogger("memos_server")
    logger.info("Logging system initialized")
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Log directory: {settings.log_path}")
    logger.info(f"Audit logging: {'enabled' if settings.enable_audit_logging else 'disabled'}")


class HIPAAAuditLogger:
    """
    HIPAA-compliant audit logger for memory operations
    Logs all access to protected health information
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        self.settings = get_settings()
    
    def log_memory_access(
        self,
        user_id: str,
        memory_id: str,
        operation: str,
        requester_id: str,
        requester_role: str,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        details: dict = None
    ):
        """
        Log memory access for HIPAA audit trail
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "MEMORY_ACCESS",
            "user_id": user_id,
            "memory_id": memory_id,
            "operation": operation,  # CREATE, READ, UPDATE, DELETE, SEARCH
            "requester_id": requester_id,
            "requester_role": requester_role,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "details": details or {}
        }
        
        self.logger.info(f"MEMORY_ACCESS: {audit_entry}")
    
    def log_consent_event(
        self,
        user_id: str,
        consent_type: str,
        consent_given: bool,
        requester_id: str,
        consent_version: str,
        ip_address: str = None
    ):
        """
        Log consent events for HIPAA compliance
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "CONSENT_EVENT",
            "user_id": user_id,
            "consent_type": consent_type,
            "consent_given": consent_given,
            "requester_id": requester_id,
            "consent_version": consent_version,
            "ip_address": ip_address
        }
        
        self.logger.info(f"CONSENT_EVENT: {audit_entry}")
    
    def log_data_export(
        self,
        user_id: str,
        export_type: str,
        requester_id: str,
        data_scope: str,
        ip_address: str = None
    ):
        """
        Log data export events
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "DATA_EXPORT",
            "user_id": user_id,
            "export_type": export_type,
            "requester_id": requester_id,
            "data_scope": data_scope,
            "ip_address": ip_address
        }
        
        self.logger.info(f"DATA_EXPORT: {audit_entry}")
    
    def log_quest_event(
        self,
        user_id: str,
        quest_id: str,
        event_type: str,
        **kwargs
    ):
        """
        Log quest-related events for gamification tracking
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "quest_id": quest_id,
            **kwargs
        }
        
        self.logger.info(f"QUEST_EVENT: {audit_entry}")
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: str = None,
        ip_address: str = None,
        additional_data: dict = None
    ):
        """
        Log security-related events
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "SECURITY_EVENT",
            "security_event_type": event_type,
            "severity": severity,  # LOW, MEDIUM, HIGH, CRITICAL
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "additional_data": additional_data or {}
        }
        
        self.logger.info(f"SECURITY_EVENT: {audit_entry}")
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        status: str,
        message: str,
        details: dict = None
    ):
        """
        Log system events for operational monitoring
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "SYSTEM_EVENT",
            "system_event_type": event_type,
            "component": component,
            "status": status,  # SUCCESS, FAILURE, WARNING
            "message": message,
            "details": details or {}
        }
        
        self.logger.info(f"SYSTEM_EVENT: {audit_entry}")
    
    def info(self, message: str, extra: dict = None):
        """
        Generic info logging method for compatibility
        Routes to appropriate specialized method based on action
        """
        if extra:
            action = extra.get('action', '')
            
            # Route to specialized methods
            if action == 'memory_access':
                self.log_memory_access(
                    user_id=extra.get('user_id', ''),
                    memory_id=extra.get('memory_id', ''),
                    operation=extra.get('operation', 'UNKNOWN'),
                    requester_id=extra.get('requester_id', ''),
                    requester_role=extra.get('requester_role', 'user'),
                    ip_address=extra.get('ip_address'),
                    user_agent=extra.get('user_agent'),
                    details=extra.get('details')
                )
            elif action == 'memory_created':
                self.log_memory_created(
                    user_id=extra.get('user_id', ''),
                    memory_id=extra.get('memory_id', ''),
                    memory_type=extra.get('memory_type', 'general'),
                    privacy_level=extra.get('privacy_level', 'standard'),
                    content_length=extra.get('content_length', 0)
                )
            elif action == 'memory_retrieved':
                self.log_memory_retrieved(
                    user_id=extra.get('user_id', ''),
                    memory_ids=extra.get('memory_ids', []),
                    retrieval_type=extra.get('retrieval_type', 'search'),
                    query=extra.get('query', '')
                )
            elif action in ['quest_assigned', 'quest_completed', 'achievement_unlocked']:
                self.log_quest_event(
                    user_id=extra.get('user_id', ''),
                    event_type=action.upper(),
                    quest_id=extra.get('quest_id', ''),
                    details=extra
                )
            elif action == 'settings_updated':
                self.log_system_event(
                    event_type='SETTINGS_UPDATE',
                    component='user_settings',
                    status='SUCCESS',
                    message=f"User settings updated by {extra.get('user_id', 'unknown')}",
                    details=extra
                )
            else:
                # Generic system event
                self.log_system_event(
                    event_type=action.upper() if action else 'INFO',
                    component='system',
                    status='INFO',
                    message=message,
                    details=extra
                )
        else:
            # Simple message logging
            self.logger.info(message)


class MemoryOperationLogger:
    """
    Specialized logger for memory operations with therapeutic context
    """
    
    def __init__(self):
        self.logger = logging.getLogger("memos_server.memory")
        self.audit_logger = HIPAAAuditLogger()
    
    def log_memory_created(
        self,
        user_id: str,
        memory_id: str,
        memory_type: str,
        privacy_level: str,
        content_length: int,
        therapeutic_relevance: float
    ):
        """Log memory creation with therapeutic context"""
        self.logger.info(
            f"Memory created - User: {user_id}, ID: {memory_id}, "
            f"Type: {memory_type}, Privacy: {privacy_level}, "
            f"Length: {content_length}, Relevance: {therapeutic_relevance:.2f}"
        )
    
    def log_memory_retrieved(
        self,
        user_id: str,
        query: str,
        results_count: int,
        search_time_ms: float,
        avg_relevance: float
    ):
        """Log memory retrieval operations"""
        self.logger.info(
            f"Memory search - User: {user_id}, Query length: {len(query)}, "
            f"Results: {results_count}, Time: {search_time_ms:.1f}ms, "
            f"Avg relevance: {avg_relevance:.2f}"
        )
    
    def log_privacy_violation(
        self,
        user_id: str,
        violation_type: str,
        content_sample: str,
        privacy_level: str
    ):
        """Log privacy violations for investigation"""
        self.logger.warning(
            f"Privacy violation detected - User: {user_id}, "
            f"Type: {violation_type}, Privacy level: {privacy_level}, "
            f"Content sample: {content_sample[:50]}..."
        )
        
        # Also log to audit trail
        self.audit_logger.log_security_event(
            event_type="PRIVACY_VIOLATION",
            severity="HIGH",
            description=f"Privacy violation: {violation_type}",
            user_id=user_id,
            additional_data={
                "violation_type": violation_type,
                "privacy_level": privacy_level,
                "content_length": len(content_sample)
            }
        )


# Global logger instances
audit_logger = HIPAAAuditLogger()
memory_logger = MemoryOperationLogger()


def get_audit_logger() -> HIPAAAuditLogger:
    """Get audit logger instance"""
    return audit_logger


def get_memory_logger() -> MemoryOperationLogger:
    """Get memory operation logger instance"""
    return memory_logger