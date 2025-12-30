"""
Suppress third-party deprecation warnings.

This module MUST be imported before any other modules to effectively suppress warnings.
Import this at the very top of any entry point (main.py, test scripts, etc.)

Pydantic v2 uses a custom warning class (PydanticDeprecatedSince20) that requires
special handling since standard warnings.filterwarnings doesn't catch it reliably.
"""

import warnings

# Store original showwarning
_original_showwarning = warnings.showwarning

# Patterns to suppress
_SUPPRESS_PATTERNS = [
    "class-based `config` is deprecated",
    "json_encoders` is deprecated",
    "declarative_base",
    "pythonjsonlogger.jsonlogger has been moved",
]

def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that suppresses known third-party deprecation warnings."""
    msg_str = str(message)

    # Suppress known deprecation patterns
    for pattern in _SUPPRESS_PATTERNS:
        if pattern in msg_str:
            return  # Suppress this warning

    # Show all other warnings normally
    _original_showwarning(message, category, filename, lineno, file, line)

# Install custom handler
warnings.showwarning = _custom_showwarning

# Also set standard filters as backup (for warnings raised before handler installed)
warnings.filterwarnings("ignore", message=".*class-based.*config.*deprecated.*")
warnings.filterwarnings("ignore", message=".*json_encoders.*deprecated.*")
warnings.filterwarnings("ignore", message=".*declarative_base.*deprecated.*")
warnings.filterwarnings("ignore", message=".*pythonjsonlogger.jsonlogger has been moved.*")
