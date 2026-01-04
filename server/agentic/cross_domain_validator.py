"""
Cross-Domain Validator Agent - Phase 48: Hallucination Mitigation

Validates cross-domain relationship claims in synthesis output.
Catches spurious causal chains between physically isolated systems.

Example hallucinations this agent catches:
- "Servo encoder alarm causes IMM hydraulic fault" (INVALID - no connection)
- "Robot alarm propagates to eDart pressure monitoring" (INVALID - isolated systems)

Research basis:
- ISA-95 equipment hierarchy
- OPC UA connection semantics
- Multi-agent RAG verification patterns
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set

import httpx

logger = logging.getLogger("agentic.cross_domain_validator")


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    CRITICAL = "critical"   # Must be removed/corrected
    WARNING = "warning"     # Should be hedged
    INFO = "info"           # Minor concern, can keep


class ClaimType(str, Enum):
    """Type of cross-domain claim."""
    CAUSAL = "causal"           # "X causes Y"
    AFFECTS = "affects"         # "X affects Y"
    PROPAGATES = "propagates"   # "X propagates to Y"
    TRIGGERS = "triggers"       # "X triggers Y"
    LEADS_TO = "leads_to"       # "X leads to Y"


@dataclass
class ExtractedClaim:
    """A cross-domain relationship claim extracted from text."""
    text: str                           # Original claim text
    claim_type: ClaimType
    source_system: str                  # System A
    target_system: str                  # System B
    start_pos: int
    end_pos: int
    context: str = ""                   # Surrounding text


@dataclass
class ClaimValidationResult:
    """Result of validating a single claim."""
    claim: ExtractedClaim
    is_valid: bool
    severity: ValidationSeverity
    message: str
    allowed_connections: List[str] = field(default_factory=list)
    suggested_revision: Optional[str] = None
    confidence: float = 1.0


@dataclass
class CrossDomainValidationResult:
    """Overall validation result for a text/synthesis."""
    total_claims: int
    valid_claims: int
    invalid_claims: int
    critical_issues: int
    warnings: int
    claim_results: List[ClaimValidationResult]
    revised_text: Optional[str] = None
    validation_notes: List[str] = field(default_factory=list)


# System detection patterns - map common terms to system types
SYSTEM_PATTERNS: Dict[str, List[str]] = {
    "robot_controller": [
        r"robot(?:\s+controller)?",
        r"fanuc",
        r"r-30i[a-z]*",
        r"controller",
    ],
    "servo_drive": [
        r"servo(?:\s+(?:drive|amplifier|amp))?",
        r"amplifier",
        r"axis\s+\d+",
    ],
    "encoder": [
        r"encoder",
        r"pulsecoder",
        r"pulse\s*coder",
        r"feedback\s+device",
    ],
    "imm_controller": [
        r"imm(?:\s+controller)?",
        r"injection\s+molding(?:\s+machine)?",
        r"molding\s+machine",
        r"krauss[\s-]*maffei",
        r"km\s+\d+",
        r"engel",
        r"arburg",
        r"press(?:\s+controller)?",
    ],
    "hydraulic_system": [
        r"hydraulic(?:\s+(?:system|pressure|pump|valve))?",
        r"hpu",
        r"power\s+unit",
    ],
    "monitoring_system": [
        r"edart",
        r"rjg",
        r"kistler",
        r"cavity\s+pressure",
        r"process\s+monitor(?:ing)?",
    ],
    "plc": [
        r"plc",
        r"allen[\s-]*bradley",
        r"controllogix",
        r"siemens\s+s7",
        r"programmable\s+logic",
    ],
}

# Causal relationship patterns
CAUSAL_PATTERNS: List[Tuple[str, ClaimType]] = [
    (r"(?:causes?|causing)\s+", ClaimType.CAUSAL),
    (r"(?:affects?|affecting)\s+", ClaimType.AFFECTS),
    (r"(?:propagates?\s+to|propagating\s+to)\s+", ClaimType.PROPAGATES),
    (r"(?:triggers?|triggering)\s+", ClaimType.TRIGGERS),
    (r"(?:leads?\s+to|leading\s+to)\s+", ClaimType.LEADS_TO),
    (r"(?:results?\s+in|resulting\s+in)\s+", ClaimType.CAUSAL),
    (r"(?:creates?|creating)\s+", ClaimType.CAUSAL),
    (r"(?:produces?|producing)\s+", ClaimType.CAUSAL),
]


class CrossDomainValidator:
    """
    Validates cross-domain relationship claims before/after synthesis.

    Workflow:
    1. Extract claims about system relationships from text
    2. Identify source and target systems
    3. Call PDF Tools validation API to check if relationship is valid
    4. Return validation results with suggested revisions

    Integration:
    - Called by UniversalOrchestrator after synthesis (PHASE 5.5)
    - Can also be called by Self-RAG for reflection
    """

    def __init__(
        self,
        pdf_api_url: str = "http://localhost:8002",
        severity_threshold: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize the validator.

        Args:
            pdf_api_url: URL of the PDF Tools API
            severity_threshold: Minimum severity to flag (WARNING = flag warnings+critical)
        """
        self.pdf_api_url = pdf_api_url
        self.severity_threshold = severity_threshold
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        # Compile system patterns
        self.compiled_system_patterns: Dict[str, List[re.Pattern]] = {}
        for system_type, patterns in SYSTEM_PATTERNS.items():
            self.compiled_system_patterns[system_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Compile causal patterns
        self.compiled_causal_patterns: List[Tuple[re.Pattern, ClaimType]] = [
            (re.compile(p, re.IGNORECASE), claim_type)
            for p, claim_type in CAUSAL_PATTERNS
        ]

    def _detect_system_type(self, text: str) -> Optional[str]:
        """
        Detect the system type mentioned in text.

        Returns the most specific match.
        """
        text_lower = text.lower()

        for system_type, patterns in self.compiled_system_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return system_type

        return None

    def _extract_systems_from_text(self, text: str) -> Set[str]:
        """Extract all system types mentioned in text."""
        systems = set()
        text_lower = text.lower()

        for system_type, patterns in self.compiled_system_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    systems.add(system_type)
                    break  # One match per system type is enough

        return systems

    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract cross-domain relationship claims from text.

        Looks for patterns like:
        - "Servo alarm causes hydraulic fault"
        - "Robot error propagates to IMM pressure"
        - "Encoder failure affects molding machine"
        """
        claims = []

        # Split into sentences for context
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check each causal pattern
            for pattern, claim_type in self.compiled_causal_patterns:
                match = pattern.search(sentence)
                if match:
                    # Split at the causal word
                    before = sentence[:match.start()].strip()
                    after = sentence[match.end():].strip()

                    # Detect systems in before/after
                    source_system = self._detect_system_type(before)
                    target_system = self._detect_system_type(after)

                    # Only record if we have two different systems
                    if source_system and target_system and source_system != target_system:
                        claims.append(ExtractedClaim(
                            text=sentence,
                            claim_type=claim_type,
                            source_system=source_system,
                            target_system=target_system,
                            start_pos=text.find(sentence),
                            end_pos=text.find(sentence) + len(sentence),
                            context=sentence
                        ))

        return claims

    async def validate_claim(self, claim: ExtractedClaim) -> ClaimValidationResult:
        """
        Validate a single cross-domain claim via PDF Tools API.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.pdf_api_url}/api/v1/validate/cross-system",
                    json={
                        "source_system": claim.source_system,
                        "target_system": claim.target_system,
                        "connection_type": None  # Let API determine
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle unified response format
                    if "data" in data:
                        data = data["data"]

                    status = data.get("status", "unknown")
                    message = data.get("message", "")
                    allowed = data.get("allowed_connections", [])
                    confidence = data.get("confidence", 0.5)

                    if status == "valid":
                        return ClaimValidationResult(
                            claim=claim,
                            is_valid=True,
                            severity=ValidationSeverity.INFO,
                            message=f"Valid relationship: {message}",
                            allowed_connections=allowed,
                            confidence=confidence
                        )
                    elif status == "invalid":
                        # Generate suggested revision
                        suggested = self._generate_revision(claim)
                        return ClaimValidationResult(
                            claim=claim,
                            is_valid=False,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"INVALID relationship: {message}",
                            allowed_connections=allowed,
                            suggested_revision=suggested,
                            confidence=confidence
                        )
                    else:  # unknown
                        return ClaimValidationResult(
                            claim=claim,
                            is_valid=True,  # Allow but warn
                            severity=ValidationSeverity.WARNING,
                            message=f"Unverified relationship: {message}",
                            allowed_connections=allowed,
                            suggested_revision=self._generate_hedged_revision(claim),
                            confidence=confidence
                        )
                else:
                    # Non-200 response - fall back to local validation
                    logger.warning(f"PDF Tools API returned {response.status_code}")
                    return self._validate_claim_locally(claim)

        except httpx.ConnectError:
            logger.warning(f"PDF Tools API unavailable at {self.pdf_api_url}")
            # Fall back to local heuristic validation
            return self._validate_claim_locally(claim)

        except Exception as e:
            logger.error(f"Error validating claim: {e}")
            return ClaimValidationResult(
                claim=claim,
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Could not validate: {str(e)}",
                confidence=0.3
            )

    def _validate_claim_locally(self, claim: ExtractedClaim) -> ClaimValidationResult:
        """
        Local heuristic validation when API is unavailable.

        Uses hardcoded rules for known invalid relationships.
        """
        # Known invalid relationships (physically impossible)
        KNOWN_INVALID = {
            ("servo_drive", "hydraulic_system"),
            ("encoder", "hydraulic_system"),
            ("encoder", "imm_controller"),
            ("encoder", "monitoring_system"),
            ("robot_controller", "hydraulic_system"),
            ("servo_drive", "monitoring_system"),
        }

        pair = (claim.source_system, claim.target_system)
        reverse_pair = (claim.target_system, claim.source_system)

        if pair in KNOWN_INVALID or reverse_pair in KNOWN_INVALID:
            return ClaimValidationResult(
                claim=claim,
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"No physical/electrical connection between {claim.source_system} "
                        f"and {claim.target_system}. These are isolated systems.",
                suggested_revision=self._generate_revision(claim),
                confidence=0.95
            )

        # Known valid relationships
        KNOWN_VALID = {
            ("robot_controller", "imm_controller"),  # Discrete I/O
            ("imm_controller", "robot_controller"),
            ("imm_controller", "monitoring_system"),  # Analog sensors
            ("plc", "robot_controller"),             # Fieldbus
            ("plc", "imm_controller"),
        }

        if pair in KNOWN_VALID or reverse_pair in KNOWN_VALID:
            return ClaimValidationResult(
                claim=claim,
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Valid cross-system relationship (discrete I/O or fieldbus)",
                confidence=0.9
            )

        # Unknown - flag for review
        return ClaimValidationResult(
            claim=claim,
            is_valid=True,
            severity=ValidationSeverity.WARNING,
            message=f"Unverified relationship between {claim.source_system} and "
                    f"{claim.target_system}. Manual verification recommended.",
            suggested_revision=self._generate_hedged_revision(claim),
            confidence=0.5
        )

    def _generate_revision(self, claim: ExtractedClaim) -> str:
        """Generate a corrected version of an invalid claim."""
        # Replace the causal claim with a statement about independence
        source_name = claim.source_system.replace("_", " ")
        target_name = claim.target_system.replace("_", " ")

        return (
            f"Note: The {source_name} and {target_name} are independent systems "
            f"with no direct physical or electrical connection. Issues in one "
            f"system do not directly cause problems in the other. Each system "
            f"should be troubleshot separately."
        )

    def _generate_hedged_revision(self, claim: ExtractedClaim) -> str:
        """Generate a hedged version of an unverified claim."""
        return (
            f"It is possible that issues in the {claim.source_system.replace('_', ' ')} "
            f"may be related to the {claim.target_system.replace('_', ' ')}, "
            f"but this relationship should be verified by checking the system "
            f"integration documentation."
        )

    async def validate_text(
        self,
        text: str,
        revise: bool = True
    ) -> CrossDomainValidationResult:
        """
        Validate all cross-domain claims in a text.

        Args:
            text: Text to validate (e.g., synthesis output)
            revise: Whether to generate revised text

        Returns:
            CrossDomainValidationResult with all findings
        """
        # Extract claims
        claims = self.extract_claims(text)

        if not claims:
            return CrossDomainValidationResult(
                total_claims=0,
                valid_claims=0,
                invalid_claims=0,
                critical_issues=0,
                warnings=0,
                claim_results=[],
                validation_notes=["No cross-domain relationship claims detected."]
            )

        # Validate each claim
        results = []
        for claim in claims:
            result = await self.validate_claim(claim)
            results.append(result)

        # Aggregate statistics
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        critical_count = sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)

        # Generate revised text if requested
        revised_text = None
        if revise and invalid_count > 0:
            revised_text = self._apply_revisions(text, results)

        # Generate validation notes
        notes = []
        if critical_count > 0:
            notes.append(
                f"CRITICAL: {critical_count} spurious cross-domain relationship(s) detected. "
                "These claims involve physically isolated systems."
            )
        if warning_count > 0:
            notes.append(
                f"WARNING: {warning_count} unverified cross-domain relationship(s). "
                "These should be hedged or verified."
            )

        return CrossDomainValidationResult(
            total_claims=len(claims),
            valid_claims=valid_count,
            invalid_claims=invalid_count,
            critical_issues=critical_count,
            warnings=warning_count,
            claim_results=results,
            revised_text=revised_text,
            validation_notes=notes
        )

    def _apply_revisions(
        self,
        text: str,
        results: List[ClaimValidationResult]
    ) -> str:
        """Apply suggested revisions to text."""
        revised = text

        # Sort by position (reverse) to preserve indices
        invalid_results = [
            r for r in results
            if not r.is_valid and r.suggested_revision
        ]
        invalid_results.sort(key=lambda r: r.claim.start_pos, reverse=True)

        for result in invalid_results:
            claim = result.claim
            if result.suggested_revision:
                # Replace the original claim with revision
                revised = (
                    revised[:claim.start_pos] +
                    result.suggested_revision +
                    revised[claim.end_pos:]
                )

        return revised

    async def validate_synthesis(
        self,
        synthesis: str,
        scratchpad: Optional[Dict[str, Any]] = None
    ) -> CrossDomainValidationResult:
        """
        Validate synthesis output for cross-domain hallucinations.

        This is the main entry point for orchestrator integration.

        Args:
            synthesis: The synthesis text to validate
            scratchpad: Optional scratchpad for context

        Returns:
            CrossDomainValidationResult
        """
        logger.info("Validating synthesis for cross-domain hallucinations...")

        result = await self.validate_text(synthesis, revise=True)

        # Log findings
        if result.critical_issues > 0:
            logger.warning(
                f"Found {result.critical_issues} CRITICAL cross-domain hallucinations!"
            )
            for claim_result in result.claim_results:
                if claim_result.severity == ValidationSeverity.CRITICAL:
                    logger.warning(f"  - {claim_result.message}")

        return result


# Singleton instance
_validator_instance: Optional[CrossDomainValidator] = None


def get_cross_domain_validator(
    pdf_api_url: str = "http://localhost:8002"
) -> CrossDomainValidator:
    """Get singleton instance of the cross-domain validator."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CrossDomainValidator(pdf_api_url=pdf_api_url)
    return _validator_instance
