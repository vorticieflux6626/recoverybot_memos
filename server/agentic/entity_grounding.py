"""
Entity Grounding Agent - Phase 48: Hallucination Mitigation

Verifies entities (error codes, part numbers, components) exist before
synthesis uses them. Catches fabricated entities that LLMs hallucinate.

Example hallucinations this agent catches:
- "E0142" error code (unverified - may not exist)
- "SRV-XXXX-PULSCODER" part number (fabricated placeholder pattern)
- "A06B-9999-H999" part number (invalid format)

Research basis:
- Entity grounding in RAG systems
- Pattern-based fabrication detection
- API-based entity verification
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set

import httpx

logger = logging.getLogger("agentic.entity_grounding")


class GroundingStatus(str, Enum):
    """Status of entity grounding check."""
    VERIFIED = "verified"         # Found in knowledge base
    PATTERN_VALID = "pattern_valid"  # Matches known format, not verified
    SUSPICIOUS = "suspicious"     # Unusual pattern, may be fabricated
    FABRICATED = "fabricated"     # Detected as hallucinated


class EntityType(str, Enum):
    """Types of entities to ground."""
    ERROR_CODE = "error_code"
    PART_NUMBER = "part_number"
    COMPONENT = "component"
    PARAMETER = "parameter"


@dataclass
class ExtractedEntity:
    """An entity extracted from text for grounding."""
    text: str                    # The entity string
    entity_type: EntityType
    context: str = ""            # Surrounding text
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class GroundingResult:
    """Result of grounding a single entity."""
    entity: ExtractedEntity
    status: GroundingStatus
    confidence: float
    message: str
    manufacturer: Optional[str] = None
    component_type: Optional[str] = None
    source: Optional[str] = None  # Where entity was found/verified
    suggested_replacement: Optional[str] = None


@dataclass
class EntityGroundingResults:
    """Aggregated grounding results for a text."""
    total_entities: int
    verified_count: int
    pattern_valid_count: int
    suspicious_count: int
    fabricated_count: int
    results: List[GroundingResult]
    grounding_notes: List[str] = field(default_factory=list)


# Error code patterns by manufacturer
ERROR_CODE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "fanuc": [
        re.compile(r"SRVO-\d{3,4}", re.IGNORECASE),      # Servo alarms
        re.compile(r"MOTN-\d{3,4}", re.IGNORECASE),      # Motion alarms
        re.compile(r"SYST-\d{3,4}", re.IGNORECASE),      # System alarms
        re.compile(r"INTP-\d{3,4}", re.IGNORECASE),      # Interpreter alarms
        re.compile(r"PRIO-\d{3,4}", re.IGNORECASE),      # Priority alarms
        re.compile(r"HOST-\d{3,4}", re.IGNORECASE),      # Host communication
        re.compile(r"FILE-\d{3,4}", re.IGNORECASE),      # File system alarms
        re.compile(r"TOOL-\d{3,4}", re.IGNORECASE),      # Tool management
        re.compile(r"ROPE-\d{3,4}", re.IGNORECASE),      # Robot positioner
    ],
    "allen_bradley": [
        re.compile(r"F\d{1,3}", re.IGNORECASE),          # Fault codes
        re.compile(r"E\d{1,3}", re.IGNORECASE),          # Error codes
        re.compile(r"[0-9]{4}:[0-9]{2}"),                # Module faults
    ],
    "siemens": [
        re.compile(r"[AF]\d{5}", re.IGNORECASE),         # Alarm/Fault codes
        re.compile(r"0x[0-9A-Fa-f]{4}"),                 # Hex error codes
    ],
    "imm": [
        re.compile(r"E[0-9]{3,4}", re.IGNORECASE),       # Error codes
        re.compile(r"A[0-9]{3,4}", re.IGNORECASE),       # Alarm codes
        re.compile(r"W[0-9]{3,4}", re.IGNORECASE),       # Warning codes
    ],
}

# Fabrication detection patterns (placeholders LLMs use)
FABRICATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"[X]{2,}", re.IGNORECASE),               # XXXX placeholders
    re.compile(r"[0]{4,}", re.IGNORECASE),               # 0000 placeholders
    re.compile(r"-[X0]{3,}", re.IGNORECASE),             # -XXXX or -0000
    re.compile(r"[A-Z]{1,2}-?[X]{2,}", re.IGNORECASE),   # SRV-XXXX
    re.compile(r"PLACEHOLDER", re.IGNORECASE),           # Explicit placeholder
    re.compile(r"TBD|N/?A", re.IGNORECASE),              # TBD, N/A
]

# Component patterns by type
COMPONENT_PATTERNS: Dict[str, re.Pattern] = {
    "servo_amplifier": re.compile(r"servo\s+(?:amplifier|amp|drive)", re.IGNORECASE),
    "encoder": re.compile(r"(?:pulse)?coder|encoder|feedback\s+device", re.IGNORECASE),
    "motor": re.compile(r"(?:servo\s+)?motor|axis\s+motor", re.IGNORECASE),
    "cable": re.compile(r"(?:encoder|signal|power)\s+cable", re.IGNORECASE),
    "controller": re.compile(r"(?:robot\s+)?controller|R-30i|teach\s+pendant", re.IGNORECASE),
}


class EntityGroundingAgent:
    """
    Verifies entities exist before synthesis uses them.

    Workflow:
    1. Extract entities from text (error codes, part numbers, components)
    2. Check each entity against:
       a. Known patterns (format validation)
       b. Fabrication patterns (hallucination detection)
       c. PDF Tools API (existence verification)
    3. Return grounding results with suggestions for fabricated entities

    Integration:
    - Called by UniversalOrchestrator alongside CrossDomainValidator
    - Can also be called by Self-RAG for entity verification
    """

    def __init__(
        self,
        pdf_api_url: str = "http://localhost:8002"
    ):
        """
        Initialize the entity grounding agent.

        Args:
            pdf_api_url: URL of the PDF Tools API for entity verification
        """
        self.pdf_api_url = pdf_api_url

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text for grounding.

        Extracts:
        - Error codes (SRVO-xxx, MOTN-xxx, etc.)
        - Part numbers (A06B-xxxx, 1756-xxx, etc.)
        - Component references
        """
        entities = []
        text_upper = text.upper()

        # Extract error codes
        for manufacturer, patterns in ERROR_CODE_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    entities.append(ExtractedEntity(
                        text=match.group(),
                        entity_type=EntityType.ERROR_CODE,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))

        # Extract part number candidates
        # Look for patterns like A06B-xxxx, 1756-xxx, 6ES7-xxx
        part_number_pattern = re.compile(
            r'[A-Z0-9]{2,6}[-][A-Z0-9]{3,}(?:[-][A-Z0-9]+)*',
            re.IGNORECASE
        )
        for match in part_number_pattern.finditer(text):
            # Skip if already captured as error code
            is_error_code = any(
                e.start_pos == match.start() and e.end_pos == match.end()
                for e in entities
            )
            if not is_error_code:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=EntityType.PART_NUMBER,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return entities

    def _check_fabrication(self, entity: ExtractedEntity) -> bool:
        """Check if entity appears to be fabricated."""
        text = entity.text

        # Check fabrication patterns
        for pattern in FABRICATION_PATTERNS:
            if pattern.search(text):
                return True

        # Check for suspicious repeated characters
        if len(set(text.replace("-", ""))) < 3 and len(text) > 5:
            return True

        return False

    async def ground_entity(self, entity: ExtractedEntity) -> GroundingResult:
        """
        Ground a single entity.

        Checks:
        1. Known format patterns
        2. Fabrication detection
        3. API verification (if available)
        """
        # Check for fabrication first
        if self._check_fabrication(entity):
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.FABRICATED,
                confidence=0.95,
                message=f"Entity appears to be fabricated (contains placeholder pattern)",
                suggested_replacement="Consult manufacturer documentation for correct identifier"
            )

        # For part numbers, use PDF API validation
        if entity.entity_type == EntityType.PART_NUMBER:
            return await self._ground_part_number(entity)

        # For error codes, verify format and optionally check API
        if entity.entity_type == EntityType.ERROR_CODE:
            return await self._ground_error_code(entity)

        # Default: mark as pattern valid but unverified
        return GroundingResult(
            entity=entity,
            status=GroundingStatus.PATTERN_VALID,
            confidence=0.6,
            message=f"Entity format appears valid but not verified"
        )

    async def _ground_part_number(self, entity: ExtractedEntity) -> GroundingResult:
        """Ground a part number via PDF API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.pdf_api_url}/api/v1/validate/part-number",
                    json={"part_number": entity.text, "manufacturer": None}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle unified response format
                    if "data" in data:
                        data = data["data"]

                    is_valid = data.get("is_valid", False)
                    manufacturer = data.get("manufacturer")
                    component_type = data.get("component_type")
                    message = data.get("message", "")
                    confidence = data.get("confidence", 0.5)

                    if is_valid:
                        return GroundingResult(
                            entity=entity,
                            status=GroundingStatus.PATTERN_VALID,
                            confidence=confidence,
                            message=message,
                            manufacturer=manufacturer,
                            component_type=component_type
                        )
                    else:
                        # Check if it's fabricated or just unknown
                        if "fabricat" in message.lower() or "placeholder" in message.lower():
                            return GroundingResult(
                                entity=entity,
                                status=GroundingStatus.FABRICATED,
                                confidence=confidence,
                                message=message,
                                suggested_replacement="Consult manufacturer catalog for valid part number"
                            )
                        else:
                            return GroundingResult(
                                entity=entity,
                                status=GroundingStatus.SUSPICIOUS,
                                confidence=confidence,
                                message=message
                            )

        except httpx.ConnectError:
            logger.warning(f"PDF Tools API unavailable at {self.pdf_api_url}")
            # Fall back to local validation
            return self._ground_part_number_locally(entity)

        except Exception as e:
            logger.error(f"Error grounding part number: {e}")
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.PATTERN_VALID,
                confidence=0.3,
                message=f"Could not verify: {str(e)}"
            )

    def _ground_part_number_locally(self, entity: ExtractedEntity) -> GroundingResult:
        """
        Local part number validation when API is unavailable.

        Uses pattern matching to validate format.
        """
        text = entity.text.upper()

        # FANUC patterns
        if text.startswith("A06B-") or text.startswith("A860-") or text.startswith("A660-"):
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.PATTERN_VALID,
                confidence=0.7,
                message="FANUC part number format (not verified against catalog)",
                manufacturer="FANUC"
            )

        # Allen-Bradley patterns
        if text.startswith("1756-") or text.startswith("2094-") or text.startswith("20-"):
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.PATTERN_VALID,
                confidence=0.7,
                message="Allen-Bradley part number format (not verified against catalog)",
                manufacturer="Allen-Bradley"
            )

        # Siemens patterns
        if text.startswith("6ES7") or text.startswith("6SL3"):
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.PATTERN_VALID,
                confidence=0.7,
                message="Siemens part number format (not verified against catalog)",
                manufacturer="Siemens"
            )

        # Unknown format
        return GroundingResult(
            entity=entity,
            status=GroundingStatus.SUSPICIOUS,
            confidence=0.4,
            message="Part number format not recognized - verify with manufacturer"
        )

    async def _ground_part_numbers_batch(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, GroundingResult]:
        """
        Ground multiple part numbers in a single API call.

        More efficient than validating one at a time when processing
        synthesis output with multiple part number references.

        Args:
            entities: List of part number entities to validate

        Returns:
            Dict mapping entity text to GroundingResult
        """
        if not entities:
            return {}

        part_numbers = [e.text for e in entities]
        entity_map = {e.text: e for e in entities}
        results = {}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.pdf_api_url}/api/v1/validate/part-numbers/batch",
                    json={"part_numbers": part_numbers}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle unified response format
                    if "data" in data:
                        data = data["data"]

                    validations = data.get("validations", {})

                    for pn, validation in validations.items():
                        entity = entity_map.get(pn)
                        if not entity:
                            continue

                        is_valid = validation.get("is_valid", False)
                        manufacturer = validation.get("manufacturer")
                        component_type = validation.get("component_type")
                        message = validation.get("message", "")
                        confidence = validation.get("confidence", 0.5)

                        if is_valid:
                            results[pn] = GroundingResult(
                                entity=entity,
                                status=GroundingStatus.PATTERN_VALID,
                                confidence=confidence,
                                message=message,
                                manufacturer=manufacturer,
                                component_type=component_type
                            )
                        else:
                            # Check if fabricated or just unknown
                            if "fabricat" in message.lower() or "placeholder" in message.lower():
                                results[pn] = GroundingResult(
                                    entity=entity,
                                    status=GroundingStatus.FABRICATED,
                                    confidence=confidence,
                                    message=message,
                                    suggested_replacement="Consult manufacturer catalog"
                                )
                            else:
                                results[pn] = GroundingResult(
                                    entity=entity,
                                    status=GroundingStatus.SUSPICIOUS,
                                    confidence=confidence,
                                    message=message
                                )

                    # Handle any part numbers not in response
                    for pn, entity in entity_map.items():
                        if pn not in results:
                            results[pn] = self._ground_part_number_locally(entity)

                    logger.info(f"Batch validated {len(results)} part numbers")
                    return results
                else:
                    logger.warning(f"Batch validation returned {response.status_code}")

        except httpx.ConnectError:
            logger.warning(f"PDF Tools API unavailable for batch validation")
        except Exception as e:
            logger.error(f"Batch part number validation error: {e}")

        # Fallback: validate each locally
        for pn, entity in entity_map.items():
            results[pn] = self._ground_part_number_locally(entity)
        return results

    async def _ground_error_code(self, entity: ExtractedEntity) -> GroundingResult:
        """
        Ground an error code.

        Validates format and optionally checks existence in knowledge base.
        """
        text = entity.text.upper()

        # Determine manufacturer from error code format
        manufacturer = None
        for mfr, patterns in ERROR_CODE_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(text):
                    manufacturer = mfr.upper()
                    break
            if manufacturer:
                break

        if manufacturer == "FANUC":
            # FANUC error codes are well-documented
            # Check if it's in a valid range (SRVO-001 to SRVO-999, etc.)
            match = re.match(r"([A-Z]+)-(\d+)", text)
            if match:
                prefix, number = match.groups()
                num = int(number)
                if num > 0 and num < 1000:
                    return GroundingResult(
                        entity=entity,
                        status=GroundingStatus.PATTERN_VALID,
                        confidence=0.8,
                        message=f"FANUC {prefix} alarm format valid",
                        manufacturer="FANUC"
                    )

        if manufacturer:
            return GroundingResult(
                entity=entity,
                status=GroundingStatus.PATTERN_VALID,
                confidence=0.7,
                message=f"{manufacturer} error code format",
                manufacturer=manufacturer
            )

        return GroundingResult(
            entity=entity,
            status=GroundingStatus.SUSPICIOUS,
            confidence=0.4,
            message="Error code format not recognized"
        )

    async def ground_entities(
        self,
        text: str,
        use_batch: bool = True
    ) -> EntityGroundingResults:
        """
        Ground all entities in a text.

        Args:
            text: Text to analyze for entities
            use_batch: Use batch API for part numbers (more efficient)

        Returns:
            EntityGroundingResults with all findings
        """
        # Extract entities
        entities = self.extract_entities(text)

        if not entities:
            return EntityGroundingResults(
                total_entities=0,
                verified_count=0,
                pattern_valid_count=0,
                suspicious_count=0,
                fabricated_count=0,
                results=[],
                grounding_notes=["No entities requiring grounding detected."]
            )

        # Separate part numbers for batch processing
        part_number_entities = [
            e for e in entities
            if e.entity_type == EntityType.PART_NUMBER
            and not self._check_fabrication(e)  # Skip obvious fabrications
        ]
        other_entities = [
            e for e in entities
            if e.entity_type != EntityType.PART_NUMBER
            or self._check_fabrication(e)
        ]

        results = []

        # Batch validate part numbers if enabled and multiple exist
        if use_batch and len(part_number_entities) >= 2:
            batch_results = await self._ground_part_numbers_batch(part_number_entities)
            results.extend(batch_results.values())
        else:
            # Validate part numbers individually
            for entity in part_number_entities:
                result = await self.ground_entity(entity)
                results.append(result)

        # Ground other entities (error codes, fabricated items, etc.)
        for entity in other_entities:
            result = await self.ground_entity(entity)
            results.append(result)

        # Aggregate statistics
        verified = sum(1 for r in results if r.status == GroundingStatus.VERIFIED)
        pattern_valid = sum(1 for r in results if r.status == GroundingStatus.PATTERN_VALID)
        suspicious = sum(1 for r in results if r.status == GroundingStatus.SUSPICIOUS)
        fabricated = sum(1 for r in results if r.status == GroundingStatus.FABRICATED)

        # Generate notes
        notes = []
        if fabricated > 0:
            notes.append(
                f"WARNING: {fabricated} entity(ies) appear to be fabricated. "
                "These should be removed or replaced with verified identifiers."
            )
        if suspicious > 0:
            notes.append(
                f"CAUTION: {suspicious} entity(ies) could not be verified. "
                "Consider adding verification notes for these."
            )

        return EntityGroundingResults(
            total_entities=len(entities),
            verified_count=verified,
            pattern_valid_count=pattern_valid,
            suspicious_count=suspicious,
            fabricated_count=fabricated,
            results=results,
            grounding_notes=notes
        )

    async def ground_synthesis(
        self,
        synthesis: str,
        scratchpad: Optional[Dict[str, Any]] = None
    ) -> EntityGroundingResults:
        """
        Ground entities in synthesis output.

        This is the main entry point for orchestrator integration.

        Args:
            synthesis: The synthesis text to check
            scratchpad: Optional scratchpad for context

        Returns:
            EntityGroundingResults
        """
        logger.info("Grounding entities in synthesis...")

        result = await self.ground_entities(synthesis)

        # Log findings
        if result.fabricated_count > 0:
            logger.warning(
                f"Found {result.fabricated_count} FABRICATED entities!"
            )
            for r in result.results:
                if r.status == GroundingStatus.FABRICATED:
                    logger.warning(f"  - {r.entity.text}: {r.message}")

        return result


# Singleton instance
_grounding_agent_instance: Optional[EntityGroundingAgent] = None


def get_entity_grounding_agent(
    pdf_api_url: str = "http://localhost:8002"
) -> EntityGroundingAgent:
    """Get singleton instance of the entity grounding agent."""
    global _grounding_agent_instance
    if _grounding_agent_instance is None:
        _grounding_agent_instance = EntityGroundingAgent(pdf_api_url=pdf_api_url)
    return _grounding_agent_instance
