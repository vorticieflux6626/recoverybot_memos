"""
Unified exception handling for memOS server.

All endpoints should use AppException for consistent error responses.
This implements Phase 7 of the engineering remediation plan.

Usage:
    from core.exceptions import AppException, ErrorCode, ValidationError

    # Raise a validation error
    raise ValidationError("Query must be at least 3 characters", field="query")

    # Raise a search error
    raise SearchError("Search failed", query=request.query)

    # Raise a custom error
    raise AppException(
        code=ErrorCode.INTERNAL_ERROR,
        message="Something went wrong",
        status_code=500,
        details={"component": "synthesizer"}
    )
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(str, Enum):
    """
    Standardized error codes across all endpoints.

    Code ranges:
    - 1xxx: Validation errors
    - 2xxx: Authentication/authorization errors
    - 3xxx: Resource errors (not found, already exists, etc.)
    - 4xxx: Search/Agentic errors
    - 5xxx: External service errors (Ollama, SearXNG, etc.)
    - 9xxx: System errors (internal, unavailable, rate limited)
    """

    # Validation errors (1xxx)
    VALIDATION_ERROR = "ERR_1001"
    INVALID_REQUEST = "ERR_1002"
    MISSING_FIELD = "ERR_1003"
    INVALID_FORMAT = "ERR_1004"
    QUERY_TOO_SHORT = "ERR_1005"
    QUERY_TOO_LONG = "ERR_1006"

    # Authentication errors (2xxx)
    AUTH_REQUIRED = "ERR_2001"
    AUTH_FAILED = "ERR_2002"
    TOKEN_EXPIRED = "ERR_2003"
    TOKEN_INVALID = "ERR_2004"
    INSUFFICIENT_PERMISSIONS = "ERR_2005"

    # Resource errors (3xxx)
    NOT_FOUND = "ERR_3001"
    ALREADY_EXISTS = "ERR_3002"
    RESOURCE_LOCKED = "ERR_3003"
    RESOURCE_DELETED = "ERR_3004"

    # Search/Agentic errors (4xxx)
    SEARCH_FAILED = "ERR_4001"
    SEARCH_TIMEOUT = "ERR_4002"
    NO_RESULTS = "ERR_4003"
    SYNTHESIS_FAILED = "ERR_4004"
    VERIFICATION_FAILED = "ERR_4005"
    SCRAPING_FAILED = "ERR_4006"
    CLASSIFICATION_FAILED = "ERR_4007"
    CRAG_EVALUATION_FAILED = "ERR_4008"
    SELF_RAG_FAILED = "ERR_4009"

    # External service errors (5xxx)
    OLLAMA_ERROR = "ERR_5001"
    OLLAMA_UNAVAILABLE = "ERR_5002"
    SEARXNG_ERROR = "ERR_5003"
    SEARXNG_UNAVAILABLE = "ERR_5004"
    MCP_ERROR = "ERR_5005"
    PDF_API_ERROR = "ERR_5006"
    BRAVE_API_ERROR = "ERR_5007"
    DUCKDUCKGO_ERROR = "ERR_5008"

    # Memory/Quest errors (6xxx)
    MEMORY_STORE_FAILED = "ERR_6001"
    MEMORY_RETRIEVE_FAILED = "ERR_6002"
    QUEST_NOT_FOUND = "ERR_6003"
    QUEST_ALREADY_ASSIGNED = "ERR_6004"
    QUEST_NOT_ASSIGNED = "ERR_6005"

    # System errors (9xxx)
    INTERNAL_ERROR = "ERR_9001"
    SERVICE_UNAVAILABLE = "ERR_9002"
    RATE_LIMITED = "ERR_9003"
    DATABASE_ERROR = "ERR_9004"
    CONFIGURATION_ERROR = "ERR_9005"


class AppException(Exception):
    """
    Base exception for all application errors.

    Provides unified error response format:
    {
        "success": false,
        "data": null,
        "meta": {...},
        "errors": [{"code": "ERR_xxxx", "message": "...", "details": {...}}]
    }

    Args:
        code: ErrorCode enum value
        message: Human-readable error message
        status_code: HTTP status code (default 400)
        details: Additional error context (optional)
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to error response format."""
        result = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# =============================================================================
# Convenience subclasses for common error types
# =============================================================================

class ValidationError(AppException):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        **details
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=400,
            details={"field": field, **details} if field else details
        )


class AuthenticationError(AppException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication required",
        code: ErrorCode = ErrorCode.AUTH_REQUIRED,
        **details
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=401,
            details=details
        )


class AuthorizationError(AppException):
    """Raised when user lacks permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        resource: Optional[str] = None,
        **details
    ):
        super().__init__(
            code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            message=message,
            status_code=403,
            details={"resource": resource, **details} if resource else details
        )


class NotFoundError(AppException):
    """Raised when a resource is not found."""

    def __init__(
        self,
        resource: str,
        identifier: Optional[str] = None,
        **details
    ):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} not found: {identifier}"
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=message,
            status_code=404,
            details={"resource": resource, "identifier": identifier, **details}
        )


class SearchError(AppException):
    """Raised when search operations fail."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SEARCH_FAILED,
        **details
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=500,
            details=details
        )


class SearchTimeoutError(SearchError):
    """Raised when search operations timeout."""

    def __init__(
        self,
        message: str = "Search operation timed out",
        timeout_seconds: Optional[float] = None,
        **details
    ):
        super().__init__(
            message=message,
            code=ErrorCode.SEARCH_TIMEOUT,
            timeout_seconds=timeout_seconds,
            **details
        )


class ExternalServiceError(AppException):
    """Raised when external services (Ollama, SearXNG, etc.) fail."""

    def __init__(
        self,
        service: str,
        message: str,
        code: Optional[ErrorCode] = None,
        **details
    ):
        # Auto-detect error code based on service name
        if code is None:
            code_map = {
                "ollama": ErrorCode.OLLAMA_ERROR,
                "searxng": ErrorCode.SEARXNG_ERROR,
                "mcp": ErrorCode.MCP_ERROR,
                "pdf_api": ErrorCode.PDF_API_ERROR,
                "brave": ErrorCode.BRAVE_API_ERROR,
                "duckduckgo": ErrorCode.DUCKDUCKGO_ERROR,
            }
            code = code_map.get(service.lower(), ErrorCode.INTERNAL_ERROR)

        super().__init__(
            code=code,
            message=f"{service} error: {message}",
            status_code=502,
            details={"service": service, **details}
        )


class ServiceUnavailableError(AppException):
    """Raised when a required service is unavailable."""

    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        **details
    ):
        super().__init__(
            code=ErrorCode.SERVICE_UNAVAILABLE,
            message=message or f"{service} is currently unavailable",
            status_code=503,
            details={"service": service, **details}
        )


class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **details
    ):
        super().__init__(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            status_code=429,
            details={"retry_after": retry_after, **details} if retry_after else details
        )


class DatabaseError(AppException):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **details
    ):
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            status_code=500,
            details={"operation": operation, **details} if operation else details
        )


class MemoryError(AppException):
    """Raised when memory service operations fail."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.MEMORY_STORE_FAILED,
        **details
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=500,
            details=details
        )


class QuestError(AppException):
    """Raised when quest operations fail."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.QUEST_NOT_FOUND,
        **details
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=400 if code != ErrorCode.QUEST_NOT_FOUND else 404,
            details=details
        )
