"""
Machine Entity Graph Service

Bridge between memOS retrieval and PDF Extraction Tools Machine Entity Graph.
Provides error-to-component mapping for FANUC robot troubleshooting.

Phase 49 Integration - 2026-01-11
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.settings import get_settings

logger = logging.getLogger("core.machine_entity_service")


# ============================================
# SRVO ERROR PATTERNS
# ============================================

# Pattern to detect SRVO error codes in queries
SRVO_ERROR_PATTERN = re.compile(
    r'\bSRVO[-\s]?0?(\d{2,3})\b',
    re.IGNORECASE
)

# Pattern to detect axis references (J1-J6, axis 1-6)
AXIS_PATTERN = re.compile(
    r'\b(?:J|axis\s*)([1-6])\b',
    re.IGNORECASE
)

# Pattern to detect robot model references
ROBOT_MODEL_PATTERN = re.compile(
    r'\b(M-\d+i[A-Z](?:/\d+[A-Z]*)?|LR\s*Mate\s*\d+i[A-Z](?:/\d+[A-Z]*)?|R-\d+i[A-Z](?:/\d+[A-Z]*)?|CRX-\d+i[A-Z]|Arc\s*Mate\s*\d+i[A-Z])\b',
    re.IGNORECASE
)


# ============================================
# MACHINE ENTITY SERVICE
# ============================================

class MachineEntityService:
    """
    Bridge between memOS and PDF Extraction Tools Machine Entity Graph.

    Provides:
    - Error-to-component path mapping
    - Troubleshooting context retrieval
    - Robot hierarchy queries
    - Component information lookup

    Implements:
    - Connection pooling for efficiency
    - Response caching with TTL
    - Graceful degradation when API unavailable
    - Circuit breaker pattern
    """

    # Cache TTL in seconds
    CACHE_TTL = 300  # 5 minutes
    HEALTH_CHECK_INTERVAL = 60  # 1 minute

    def __init__(self):
        self.settings = get_settings()
        self.base_url = getattr(self.settings, 'pdf_api_url', 'http://localhost:8002')
        self.timeout = getattr(self.settings, 'pdf_api_timeout', 30)
        self.enabled = getattr(self.settings, 'pdf_api_enabled', True)

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Simple in-memory cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._health_status: Optional[bool] = None
        self._health_checked_at: float = 0

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open_until: float = 0

        logger.info(f"MachineEntityService initialized: {self.base_url}")

    # ============================================
    # HEALTH & AVAILABILITY
    # ============================================

    async def health_check(self) -> bool:
        """Check if Machine Entity Graph API is available."""
        now = time.time()

        # Check circuit breaker
        if now < self._circuit_open_until:
            logger.debug("Circuit breaker open, skipping health check")
            return False

        # Use cached health status if recent
        if self._health_status is not None and (now - self._health_checked_at) < self.HEALTH_CHECK_INTERVAL:
            return self._health_status

        try:
            response = await self.client.get("/health", timeout=5.0)
            self._health_status = response.status_code == 200
            self._health_checked_at = now

            if self._health_status:
                self._failure_count = 0
                logger.debug("Machine Entity API health check: OK")
            else:
                self._record_failure()
                logger.warning(f"Machine Entity API health check failed: {response.status_code}")

            return self._health_status

        except Exception as e:
            self._record_failure()
            self._health_status = False
            self._health_checked_at = now
            logger.warning(f"Machine Entity API health check error: {e}")
            return False

    def _record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self._failure_count += 1
        if self._failure_count >= 3:
            # Open circuit for 60 seconds after 3 failures
            self._circuit_open_until = time.time() + 60
            logger.warning("Circuit breaker opened for Machine Entity API (60s)")

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.CACHE_TTL:
                return value
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Set value in cache with current timestamp"""
        self._cache[key] = (value, time.time())

    # ============================================
    # QUERY PARSING
    # ============================================

    def extract_error_info(self, query: str) -> Dict[str, Any]:
        """
        Extract SRVO error code, axis, and robot model from query.

        Args:
            query: User query string

        Returns:
            Dict with error_code, axis_number, robot_model (all optional)
        """
        result = {
            "error_code": None,
            "axis_number": None,
            "robot_model": None,
            "has_srvo_error": False
        }

        # Extract SRVO error code
        srvo_match = SRVO_ERROR_PATTERN.search(query)
        if srvo_match:
            error_num = srvo_match.group(1).zfill(3)
            result["error_code"] = f"SRVO-{error_num}"
            result["has_srvo_error"] = True

        # Extract axis number
        axis_match = AXIS_PATTERN.search(query)
        if axis_match:
            result["axis_number"] = int(axis_match.group(1))

        # Extract robot model
        model_match = ROBOT_MODEL_PATTERN.search(query)
        if model_match:
            result["robot_model"] = model_match.group(1)

        return result

    # ============================================
    # API METHODS
    # ============================================

    async def get_error_path(
        self,
        error_code: str,
        robot_model: Optional[str] = None,
        axis_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get error-to-component path for an SRVO error.

        Args:
            error_code: SRVO error code (e.g., "SRVO-063")
            robot_model: Robot model (e.g., "M-16iB/20")
            axis_number: Axis number (1-6)

        Returns:
            Dict with path information and affected components
        """
        if not await self.health_check():
            logger.debug("Machine Entity API unavailable, skipping error path lookup")
            return None

        # Check cache
        cache_key = f"error_path:{error_code}:{robot_model}:{axis_number}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            payload = {"error_code": error_code}
            if robot_model:
                payload["robot_model"] = robot_model
            if axis_number:
                payload["axis_number"] = axis_number

            response = await self.client.post(
                "/api/v1/machine/error-path",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                logger.info(f"Error path retrieved: {error_code} -> {len(data.get('affected_components', []))} components")
                return data
            else:
                logger.warning(f"Error path lookup failed: {response.status_code}")
                return None

        except Exception as e:
            self._record_failure()
            logger.error(f"Error path lookup error: {e}")
            return None

    async def get_troubleshooting_context(
        self,
        error_code: str,
        robot_model: Optional[str] = None,
        axis_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive troubleshooting context for an error.

        Uses the PDF Extraction Tools API endpoints:
        - /api/v1/troubleshoot/{error_code} - Structured troubleshooting steps
        - /api/v1/search/hsea/troubleshoot/{error_code} - HSEA-enhanced context

        Args:
            error_code: SRVO error code (e.g., "SRVO-063")
            robot_model: Robot model (optional, for component-specific context)
            axis_number: Axis number (optional, for axis-specific context)

        Returns:
            Dict with error info, troubleshooting steps, related errors, etc.
        """
        if not await self.health_check():
            logger.debug("Machine Entity API unavailable, skipping troubleshooting context")
            return None

        # Check cache
        cache_key = f"troubleshoot:{error_code}:{robot_model}:{axis_number}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Use the correct endpoint: /api/v1/troubleshoot/{error_code}
            response = await self.client.get(
                f"/api/v1/troubleshoot/{error_code}"
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", result)

                # Transform to expected format for machine context
                context = {
                    "error_info": {
                        "error_code": data.get("error_code"),
                        "title": data.get("title"),
                        "category": data.get("category"),
                        "component": "servo_drive",  # SRVO errors relate to servo drive
                        "severity": "alarm",
                        "axis_specific": axis_number is not None
                    },
                    "troubleshooting_steps": data.get("steps", []),
                    "related_errors": [
                        {"error_code": code, "confidence": 0.9}
                        for code in data.get("related_codes", [])
                    ],
                    "required_tools": data.get("required_tools", []),
                    "affected_components": [
                        {"type": "pulsecoder", "name": f"J{axis_number or 1}_encoder"}
                    ] if "Pulsecoder" in str(data) else [],
                    "sibling_components": []
                }

                self._set_cached(cache_key, context)
                logger.info(f"Troubleshooting context retrieved: {error_code} -> {len(context.get('troubleshooting_steps', []))} steps")
                return context
            else:
                logger.warning(f"Troubleshooting context lookup failed: {response.status_code}")
                return None

        except Exception as e:
            self._record_failure()
            logger.error(f"Troubleshooting context error: {e}")
            return None

    async def get_axis_components(
        self,
        robot_model: str,
        axis_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get all components for a specific robot axis.

        Args:
            robot_model: Robot model (e.g., "M-16iB/20")
            axis_number: Axis number (1-6)

        Returns:
            Dict with motor, encoder, brake, cable information
        """
        if not await self.health_check():
            return None

        # Check cache
        cache_key = f"axis:{robot_model}:{axis_number}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # URL encode the model (/ becomes %2F)
            import urllib.parse
            encoded_model = urllib.parse.quote(robot_model, safe='')

            response = await self.client.get(
                f"/api/v1/machine/axes/{encoded_model}/{axis_number}"
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", result)
                self._set_cached(cache_key, data)
                return data
            else:
                logger.warning(f"Axis components lookup failed: {response.status_code}")
                return None

        except Exception as e:
            self._record_failure()
            logger.error(f"Axis components error: {e}")
            return None

    async def get_robot_hierarchy(self, robot_model: str) -> Optional[Dict[str, Any]]:
        """
        Get full robot hierarchy with all axes and components.

        Args:
            robot_model: Robot model

        Returns:
            Dict with robot info and all axes
        """
        if not await self.health_check():
            return None

        # Check cache
        cache_key = f"robot:{robot_model}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import urllib.parse
            encoded_model = urllib.parse.quote(robot_model, safe='')

            response = await self.client.get(
                f"/api/v1/machine/robots/{encoded_model}"
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", result)
                self._set_cached(cache_key, data)
                return data
            else:
                return None

        except Exception as e:
            self._record_failure()
            logger.error(f"Robot hierarchy error: {e}")
            return None

    async def list_robots(self) -> List[Dict[str, Any]]:
        """List all available robot models."""
        if not await self.health_check():
            return []

        cached = self._get_cached("robots_list")
        if cached:
            return cached

        try:
            response = await self.client.get("/api/v1/machine/robots")
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", [])
                self._set_cached("robots_list", data)
                return data
            return []

        except Exception as e:
            self._record_failure()
            logger.error(f"List robots error: {e}")
            return []

    # ============================================
    # CONTEXT GENERATION
    # ============================================

    async def get_machine_context_for_query(
        self,
        query: str,
        default_robot_model: str = "M-16iB/20"
    ) -> Optional[Dict[str, Any]]:
        """
        Get machine entity context for a query.

        Automatically extracts error code, axis, and robot model from query,
        then retrieves relevant machine architecture context.

        Args:
            query: User query
            default_robot_model: Default robot model if not specified

        Returns:
            Dict with:
            - error_info: Parsed error information
            - troubleshooting_context: Full troubleshooting context
            - affected_components: List of affected components
            - formatted_context: Text formatted for LLM synthesis
        """
        # Extract error info from query
        error_info = self.extract_error_info(query)

        if not error_info["has_srvo_error"]:
            logger.debug("No SRVO error detected in query")
            return None

        error_code = error_info["error_code"]
        robot_model = error_info["robot_model"] or default_robot_model
        axis_number = error_info["axis_number"]

        logger.info(f"Machine context lookup: {error_code}, model={robot_model}, axis={axis_number}")

        # Get troubleshooting context
        context = await self.get_troubleshooting_context(
            error_code=error_code,
            robot_model=robot_model,
            axis_number=axis_number
        )

        if not context:
            return None

        # Format for LLM synthesis
        formatted = self._format_machine_context(context, error_info)

        return {
            "error_info": error_info,
            "troubleshooting_context": context,
            "affected_components": context.get("affected_components", []),
            "related_errors": context.get("related_errors", []),
            "formatted_context": formatted
        }

    def _format_machine_context(
        self,
        context: Dict[str, Any],
        error_info: Dict[str, Any]
    ) -> str:
        """
        Format machine context as text for LLM synthesis.

        Creates a structured text block that can be included in the
        synthesis prompt to provide physical component awareness.
        """
        lines = []
        lines.append("<machine_architecture_context>")

        # Error information
        error_code = error_info.get("error_code", "Unknown")
        lines.append(f"  <error code=\"{error_code}\">")

        if context.get("error_info"):
            ei = context["error_info"]
            lines.append(f"    <component>{ei.get('component', 'unknown')}</component>")
            lines.append(f"    <severity>{ei.get('severity', 'unknown')}</severity>")
            lines.append(f"    <axis_specific>{ei.get('axis_specific', False)}</axis_specific>")
        lines.append("  </error>")

        # Affected components
        affected = context.get("affected_components", [])
        if affected:
            lines.append("  <affected_components>")
            for comp in affected:
                comp_type = comp.get("type", "unknown")
                comp_name = comp.get("name", "unknown")
                lines.append(f"    <component type=\"{comp_type}\" name=\"{comp_name}\" />")
            lines.append("  </affected_components>")

        # Related errors
        related = context.get("related_errors", [])
        if related:
            lines.append("  <related_errors>")
            for rel in related[:5]:  # Limit to 5
                rel_code = rel.get("error_code", "unknown")
                confidence = rel.get("confidence", 0)
                lines.append(f"    <error code=\"{rel_code}\" confidence=\"{confidence:.2f}\" />")
            lines.append("  </related_errors>")

        # Sibling components (other components on same axis)
        siblings = context.get("sibling_components", [])
        if siblings:
            lines.append("  <sibling_components>")
            lines.append(f"    <!-- Other components to check: {', '.join(siblings[:5])} -->")
            lines.append("  </sibling_components>")

        lines.append("</machine_architecture_context>")

        return "\n".join(lines)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ============================================
# SINGLETON INSTANCE
# ============================================

_machine_entity_service: Optional[MachineEntityService] = None


def get_machine_entity_service() -> MachineEntityService:
    """Get or create the singleton MachineEntityService instance."""
    global _machine_entity_service
    if _machine_entity_service is None:
        _machine_entity_service = MachineEntityService()
    return _machine_entity_service
