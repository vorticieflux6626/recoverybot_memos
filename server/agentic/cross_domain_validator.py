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
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

import httpx
import yaml

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


# Effect detection patterns - map text phrases to causal_rules effect names
EFFECT_PATTERNS: Dict[str, List[str]] = {
    # IMM hydraulic effects
    "imm_hydraulic_pressure": [
        r"hydraulic\s*(?:pressure|fluctuation|issue|problem|fault)",
        r"hpu\s*(?:pressure|issue|fault)",
        r"pump\s*pressure",
    ],
    "imm_injection_pressure": [
        r"injection\s*pressure",
        r"pack(?:ing)?\s*pressure",
        r"hold\s*pressure",
    ],
    "cavity_pressure_readings": [
        r"cavity\s*pressure",
        r"mold\s*pressure",
        r"edart\s*(?:reading|sensor|pressure)",
        r"rjg\s*(?:reading|sensor)",
    ],
    "hot_runner_temperatures": [
        r"hot\s*runner\s*(?:temperature|temp|zone)",
        r"nozzle\s*(?:temperature|temp|heat)",
        r"manifold\s*(?:temperature|temp)",
    ],
    "mold_temperature": [
        r"mold\s*(?:temperature|temp)",
        r"tooling\s*(?:temperature|temp)",
    ],
    "process_monitoring_accuracy": [
        r"(?:process\s*)?monitor(?:ing)?\s*(?:accuracy|reading|error)",
        r"sensor\s*(?:accuracy|calibration)",
    ],
    # Robot effects
    "robot_servo_accuracy": [
        r"(?:robot\s*)?servo\s*(?:accuracy|precision|error)",
        r"axis\s*(?:accuracy|precision)",
    ],
    "robot_encoder_readings": [
        r"encoder\s*(?:reading|error|drift)",
        r"pulsecoder\s*(?:reading|error)",
        r"position\s*feedback",
    ],
    "robot_internal_faults": [
        r"robot\s*(?:internal\s*)?fault",
        r"controller\s*(?:internal\s*)?error",
    ],
}


def _load_industrial_relationships() -> Dict[str, Any]:
    """Load the industrial relationships YAML file."""
    yaml_path = Path(__file__).parent / "data" / "industrial_relationships.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load industrial_relationships.yaml: {e}")
    return {}


# Load causal rules on module import
_INDUSTRIAL_RELATIONSHIPS = _load_industrial_relationships()
_CAUSAL_RULES = _INDUSTRIAL_RELATIONSHIPS.get("causal_rules", {})


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

    Observability:
    - Emits SSE events for real-time progress tracking
    - Logs decisions via DecisionLogger
    - Structured logging for debugging
    """

    def __init__(
        self,
        pdf_api_url: str = "http://localhost:8002",
        severity_threshold: ValidationSeverity = ValidationSeverity.WARNING,
        request_id: Optional[str] = None,
        emitter: Optional[Any] = None,
        decision_logger: Optional[Any] = None,
    ):
        """
        Initialize the validator.

        Args:
            pdf_api_url: URL of the PDF Tools API
            severity_threshold: Minimum severity to flag (WARNING = flag warnings+critical)
            request_id: Request ID for logging/tracing
            emitter: EventEmitter for SSE events
            decision_logger: DecisionLogger for structured decision tracking
        """
        self.pdf_api_url = pdf_api_url
        self.severity_threshold = severity_threshold
        self.request_id = request_id or "unknown"
        self.emitter = emitter
        self.decision_logger = decision_logger
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

        # Compile effect patterns
        self.compiled_effect_patterns: Dict[str, List[re.Pattern]] = {}
        for effect_name, patterns in EFFECT_PATTERNS.items():
            self.compiled_effect_patterns[effect_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    async def _emit_event(
        self,
        event_type: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit SSE event for real-time progress tracking."""
        if not self.emitter:
            return

        try:
            from agentic.events import SearchEvent, EventType

            # Map string to EventType enum
            event_type_enum = getattr(EventType, event_type.upper(), EventType.DECISION_POINT)

            await self.emitter.emit(SearchEvent(
                event_type=event_type_enum,
                request_id=self.request_id,
                message=message,
                data=data or {}
            ))
        except Exception as e:
            logger.warning(f"Failed to emit SSE event: {e}")

    async def _log_decision(
        self,
        decision_type: str,
        decision_made: str,
        reasoning: str = "",
        alternatives: Optional[List[str]] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a validation decision via DecisionLogger."""
        if not self.decision_logger:
            return

        try:
            await self.decision_logger.log_decision(
                agent_name="cross_domain_validator",
                decision_type=decision_type,
                decision_made=decision_made,
                reasoning=reasoning,
                alternatives=alternatives,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")

    def _detect_effects_in_text(self, text: str) -> List[str]:
        """
        Detect specific effects mentioned in claim text.

        Returns list of effect names from causal_rules (e.g., "imm_hydraulic_pressure").
        """
        detected_effects = []
        text_lower = text.lower()

        for effect_name, patterns in self.compiled_effect_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    detected_effects.append(effect_name)
                    break  # One match per effect type is enough

        return detected_effects

    def _check_prohibited_effects(
        self,
        source_system: str,
        claimed_effects: List[str]
    ) -> Optional[Tuple[str, str]]:
        """
        Check if any claimed effects are prohibited for the source system.

        Uses causal_rules from industrial_relationships.yaml.

        Args:
            source_system: The source system type (e.g., "robot_controller")
            claimed_effects: List of detected effects in the claim

        Returns:
            Tuple of (prohibited_effect, rule_name) if found, None otherwise
        """
        if not claimed_effects:
            return None

        # Map source systems to their causal rule names
        SYSTEM_TO_RULE = {
            "robot_controller": "robot_alarm_effects",
            "servo_drive": "servo_fault_effects",
            "encoder": "encoder_fault_effects",
            "imm_controller": "imm_alarm_effects",
        }

        rule_name = SYSTEM_TO_RULE.get(source_system)
        if not rule_name:
            return None

        rule = _CAUSAL_RULES.get(rule_name, {})
        does_not_affect = rule.get("does_not_affect", [])

        for effect in claimed_effects:
            if effect in does_not_affect:
                return (effect, rule_name)

        return None

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

        Also checks for prohibited effects even when systems can communicate,
        since the API only validates system connectivity, not specific effects.
        """
        # First, check for prohibited effects locally (always applies)
        # This catches claims like "robot causes hydraulic pressure" even though
        # robot and IMM can communicate via I/O for other purposes
        claimed_effects = self._detect_effects_in_text(claim.text)
        prohibited = self._check_prohibited_effects(claim.source_system, claimed_effects)

        if prohibited:
            effect_name, rule_name = prohibited
            effect_readable = effect_name.replace("_", " ")
            source_readable = claim.source_system.replace("_", " ")

            logger.info(
                f"[{self.request_id}] Claim rejected: {source_readable} → {effect_readable} "
                f"is prohibited per {rule_name}"
            )

            return ClaimValidationResult(
                claim=claim,
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"{source_readable.title()} faults do NOT affect {effect_readable}. "
                        f"These are isolated parameters per {rule_name}.",
                suggested_revision=self._generate_effect_specific_revision(
                    claim, effect_name, rule_name
                ),
                confidence=0.95
            )

        # If no prohibited effects, proceed with API validation for system connectivity
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

        Uses hardcoded rules for known invalid relationships AND
        effect-specific rules from industrial_relationships.yaml.
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

        # Even for valid relationships, check if specific effect is prohibited
        # e.g., robot_controller→imm_controller is valid for I/O, but robot
        # alarms don't affect hydraulic pressure
        claimed_effects = self._detect_effects_in_text(claim.text)
        prohibited = self._check_prohibited_effects(claim.source_system, claimed_effects)

        if prohibited:
            effect_name, rule_name = prohibited
            effect_readable = effect_name.replace("_", " ")
            source_readable = claim.source_system.replace("_", " ")

            return ClaimValidationResult(
                claim=claim,
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"{source_readable.title()} faults do NOT affect {effect_readable}. "
                        f"These are isolated parameters per {rule_name}.",
                suggested_revision=self._generate_effect_specific_revision(
                    claim, effect_name, rule_name
                ),
                confidence=0.95
            )

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

    def _generate_effect_specific_revision(
        self,
        claim: ExtractedClaim,
        prohibited_effect: str,
        rule_name: str
    ) -> str:
        """Generate revision for claims with prohibited specific effects."""
        source_name = claim.source_system.replace("_", " ")
        effect_name = prohibited_effect.replace("_", " ")

        # Get the 'note' from the causal rule if available
        rule = _CAUSAL_RULES.get(rule_name, {})
        note = rule.get("note", "")

        base_revision = (
            f"Note: {source_name.title()} issues do not directly affect {effect_name}. "
            f"While the systems may communicate via discrete I/O for coordination, "
            f"the internal parameters (like {effect_name}) are independently controlled."
        )

        if note:
            base_revision += f" {note}"

        return base_revision

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
        # Emit start event
        await self._emit_event(
            "CROSS_DOMAIN_VALIDATING",
            "Extracting cross-domain relationship claims...",
            {"text_length": len(text)}
        )

        # Extract claims
        claims = self.extract_claims(text)

        if not claims:
            logger.info(f"[{self.request_id}] No cross-domain claims found in text")
            await self._log_decision(
                decision_type="skip",
                decision_made="no_claims_found",
                reasoning="No cross-domain relationship patterns detected in text",
                confidence=1.0
            )
            return CrossDomainValidationResult(
                total_claims=0,
                valid_claims=0,
                invalid_claims=0,
                critical_issues=0,
                warnings=0,
                claim_results=[],
                validation_notes=["No cross-domain relationship claims detected."]
            )

        logger.info(f"[{self.request_id}] Extracted {len(claims)} cross-domain claims for validation")

        # Validate each claim
        results = []
        for i, claim in enumerate(claims):
            # Emit claim extraction event
            await self._emit_event(
                "CROSS_DOMAIN_CLAIM_EXTRACTED",
                f"Validating claim {i+1}/{len(claims)}: {claim.source_system} → {claim.target_system}",
                {
                    "claim_index": i + 1,
                    "total_claims": len(claims),
                    "source_system": claim.source_system,
                    "target_system": claim.target_system,
                    "claim_type": claim.claim_type.value
                }
            )

            result = await self.validate_claim(claim)
            results.append(result)

            # Emit result event
            if result.is_valid:
                await self._emit_event(
                    "CROSS_DOMAIN_CLAIM_VALID",
                    f"Claim {i+1} valid: {result.message[:100]}",
                    {
                        "claim_index": i + 1,
                        "severity": result.severity.value,
                        "confidence": result.confidence
                    }
                )
            else:
                await self._emit_event(
                    "CROSS_DOMAIN_CLAIM_INVALID",
                    f"Claim {i+1} INVALID: {result.message[:100]}",
                    {
                        "claim_index": i + 1,
                        "severity": result.severity.value,
                        "confidence": result.confidence,
                        "suggested_revision": result.suggested_revision[:200] if result.suggested_revision else None
                    }
                )

            # Log decision for each claim
            await self._log_decision(
                decision_type="verification",
                decision_made="valid" if result.is_valid else "invalid",
                reasoning=result.message,
                confidence=result.confidence,
                metadata={
                    "source_system": claim.source_system,
                    "target_system": claim.target_system,
                    "claim_type": claim.claim_type.value,
                    "severity": result.severity.value
                }
            )

        # Aggregate statistics
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        critical_count = sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)

        # Generate revised text if requested
        revised_text = None
        if revise and invalid_count > 0:
            revised_text = self._apply_revisions(text, results)
            logger.info(f"[{self.request_id}] Generated revised text with {invalid_count} corrections")

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

        # Emit completion event
        await self._emit_event(
            "CROSS_DOMAIN_VALIDATION_COMPLETE",
            f"Validation complete: {valid_count} valid, {invalid_count} invalid ({critical_count} critical)",
            {
                "total_claims": len(claims),
                "valid_claims": valid_count,
                "invalid_claims": invalid_count,
                "critical_issues": critical_count,
                "warnings": warning_count
            }
        )

        # Log overall decision
        overall_decision = "pass" if invalid_count == 0 else ("revise" if revise else "flag")
        await self._log_decision(
            decision_type="evaluation",
            decision_made=overall_decision,
            reasoning=f"{len(claims)} claims checked: {valid_count} valid, {invalid_count} invalid",
            alternatives=["pass", "revise", "flag", "reject"],
            confidence=1.0 - (critical_count / max(len(claims), 1)),
            metadata={
                "total_claims": len(claims),
                "valid_claims": valid_count,
                "invalid_claims": invalid_count,
                "critical_issues": critical_count
            }
        )

        logger.info(
            f"[{self.request_id}] Cross-domain validation complete: "
            f"{len(claims)} claims, {valid_count} valid, {invalid_count} invalid, "
            f"{critical_count} critical"
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

    async def validate_causal_chain(
        self,
        chain: List[str],
        claim_text: str = ""
    ) -> ClaimValidationResult:
        """
        Validate an entire causal chain via PDF Tools API.

        More efficient than validating pair-by-pair when synthesis
        contains multi-hop reasoning like "A causes B causes C".

        Args:
            chain: List of system types in causal order
                   e.g., ["servo_drive", "robot_controller", "imm_controller"]
            claim_text: Original claim text for context

        Returns:
            ClaimValidationResult with chain-level validation
        """
        if len(chain) < 2:
            return ClaimValidationResult(
                claim=ExtractedClaim(
                    text=claim_text,
                    claim_type=ClaimType.CAUSAL,
                    source_system=chain[0] if chain else "",
                    target_system="",
                    start_pos=0,
                    end_pos=len(claim_text)
                ),
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Single system - no cross-domain validation needed",
                confidence=1.0
            )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.pdf_api_url}/api/v1/validate/causal-chain",
                    json={"chain": chain}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle unified response format
                    if "data" in data:
                        data = data["data"]

                    is_valid = data.get("is_valid", False)
                    message = data.get("message", "")
                    invalid_links = data.get("invalid_links", [])
                    confidence = data.get("confidence", 0.5)

                    # Create a synthetic claim for the full chain
                    claim = ExtractedClaim(
                        text=claim_text or f"{' → '.join(chain)}",
                        claim_type=ClaimType.CAUSAL,
                        source_system=chain[0],
                        target_system=chain[-1],
                        start_pos=0,
                        end_pos=len(claim_text) if claim_text else len(chain) * 20
                    )

                    if is_valid:
                        return ClaimValidationResult(
                            claim=claim,
                            is_valid=True,
                            severity=ValidationSeverity.INFO,
                            message=f"Valid causal chain: {message}",
                            confidence=confidence
                        )
                    else:
                        # Build message about invalid links
                        invalid_msg = "; ".join([
                            f"{link.get('from')} → {link.get('to')}: {link.get('reason', 'no connection')}"
                            for link in invalid_links
                        ])
                        return ClaimValidationResult(
                            claim=claim,
                            is_valid=False,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"INVALID causal chain: {invalid_msg}",
                            suggested_revision=self._generate_chain_revision(chain, invalid_links),
                            confidence=confidence
                        )
                else:
                    logger.warning(f"Causal chain validation returned {response.status_code}")

        except httpx.ConnectError:
            logger.warning(f"PDF Tools API unavailable for chain validation")
        except Exception as e:
            logger.error(f"Causal chain validation error: {e}")

        # Fallback: validate each link locally
        return await self._validate_chain_locally(chain, claim_text)

    async def _validate_chain_locally(
        self,
        chain: List[str],
        claim_text: str
    ) -> ClaimValidationResult:
        """Local fallback for chain validation when API unavailable."""
        invalid_links = []

        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]

            # Create synthetic claim for this link
            link_claim = ExtractedClaim(
                text=f"{source} → {target}",
                claim_type=ClaimType.CAUSAL,
                source_system=source,
                target_system=target,
                start_pos=0,
                end_pos=0
            )

            result = self._validate_claim_locally(link_claim)
            if not result.is_valid:
                invalid_links.append({
                    "from": source,
                    "to": target,
                    "reason": result.message
                })

        claim = ExtractedClaim(
            text=claim_text or f"{' → '.join(chain)}",
            claim_type=ClaimType.CAUSAL,
            source_system=chain[0],
            target_system=chain[-1],
            start_pos=0,
            end_pos=len(claim_text) if claim_text else len(chain) * 20
        )

        if invalid_links:
            invalid_msg = "; ".join([
                f"{link['from']} → {link['to']}: {link['reason']}"
                for link in invalid_links
            ])
            return ClaimValidationResult(
                claim=claim,
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"INVALID causal chain: {invalid_msg}",
                suggested_revision=self._generate_chain_revision(chain, invalid_links),
                confidence=0.8
            )

        return ClaimValidationResult(
            claim=claim,
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Causal chain validated locally",
            confidence=0.7
        )

    def _generate_chain_revision(
        self,
        chain: List[str],
        invalid_links: List[Dict[str, str]]
    ) -> str:
        """Generate revision suggestion for invalid causal chain."""
        invalid_systems = set()
        for link in invalid_links:
            invalid_systems.add(link.get("from", ""))
            invalid_systems.add(link.get("to", ""))

        return (
            f"The causal chain {' → '.join(chain)} contains invalid connections. "
            f"The following systems are not directly connected: {', '.join(invalid_systems)}. "
            "Each system should be troubleshot independently. Check integration documentation "
            "for valid communication paths between these systems."
        )

    def extract_causal_chains(self, text: str) -> List[List[str]]:
        """
        Extract multi-hop causal chains from text.

        Finds patterns like "A causes B which leads to C"
        and returns as chains: [["A", "B", "C"]]
        """
        chains = []

        # Pattern for chained causation (X causes Y which causes Z)
        chain_patterns = [
            r"(.+?)\s+(?:causes?|leading\s+to|triggering)\s+(.+?)\s+(?:which\s+)?(?:causes?|leads?\s+to|triggers?|results?\s+in)\s+(.+?)(?:\.|,|$)",
            r"(.+?)\s*→\s*(.+?)\s*→\s*(.+?)(?:\.|,|$)",
        ]

        for pattern in chain_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                chain = []
                for group in match.groups():
                    system = self._detect_system_type(group)
                    if system:
                        chain.append(system)

                if len(chain) >= 2:
                    chains.append(chain)

        return chains

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
        logger.info(f"[{self.request_id}] Validating synthesis for cross-domain hallucinations...")

        # Log decision to start validation
        await self._log_decision(
            decision_type="action",
            decision_made="validate_synthesis",
            reasoning="Checking synthesis for invalid cross-system causal claims",
            confidence=1.0,
            metadata={"synthesis_length": len(synthesis)}
        )

        result = await self.validate_text(synthesis, revise=True)

        # Log findings with structured output
        if result.critical_issues > 0:
            logger.warning(
                f"[{self.request_id}] Found {result.critical_issues} CRITICAL cross-domain hallucinations!"
            )
            for claim_result in result.claim_results:
                if claim_result.severity == ValidationSeverity.CRITICAL:
                    logger.warning(
                        f"[{self.request_id}]   - {claim_result.claim.source_system} → "
                        f"{claim_result.claim.target_system}: {claim_result.message}"
                    )

            # Log critical findings to decision logger
            await self._log_decision(
                decision_type="verification",
                decision_made="hallucinations_detected",
                reasoning=f"Found {result.critical_issues} critical cross-domain hallucinations that must be corrected",
                confidence=0.95,
                metadata={
                    "critical_issues": result.critical_issues,
                    "claims": [
                        {
                            "source": cr.claim.source_system,
                            "target": cr.claim.target_system,
                            "message": cr.message[:100]
                        }
                        for cr in result.claim_results
                        if cr.severity == ValidationSeverity.CRITICAL
                    ]
                }
            )
        else:
            logger.info(f"[{self.request_id}] Cross-domain validation passed: no hallucinations detected")
            await self._log_decision(
                decision_type="verification",
                decision_made="validation_passed",
                reasoning="No cross-domain hallucinations detected in synthesis",
                confidence=1.0
            )

        return result


# Singleton instance (for basic usage without observability)
_validator_instance: Optional[CrossDomainValidator] = None


def get_cross_domain_validator(
    pdf_api_url: str = "http://localhost:8002",
    request_id: Optional[str] = None,
    emitter: Optional[Any] = None,
    decision_logger: Optional[Any] = None,
) -> CrossDomainValidator:
    """
    Get a cross-domain validator instance.

    For observability integration, pass request_id, emitter, and decision_logger.
    Without these, returns a singleton instance for basic validation.

    Args:
        pdf_api_url: URL of the PDF Tools API
        request_id: Request ID for logging/tracing (creates new instance if provided)
        emitter: EventEmitter for SSE events
        decision_logger: DecisionLogger for structured decision tracking

    Returns:
        CrossDomainValidator instance
    """
    # If observability parameters provided, create a new instance
    if request_id or emitter or decision_logger:
        return CrossDomainValidator(
            pdf_api_url=pdf_api_url,
            request_id=request_id,
            emitter=emitter,
            decision_logger=decision_logger,
        )

    # Otherwise, return singleton for basic usage
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CrossDomainValidator(pdf_api_url=pdf_api_url)
    return _validator_instance
