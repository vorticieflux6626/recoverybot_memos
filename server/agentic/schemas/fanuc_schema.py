"""
FANUC Domain Schema for Entity Extraction

Defines entity types, patterns, and relationships specific to FANUC robot
technical documentation. Used by:
- DocumentGraphService for query enhancement
- FANUCCorpusBuilder for entity extraction
- UniversalOrchestrator for query routing

Patterns are designed to match content from FANUC manuals including:
- Alarm/error codes (SRVO-xxx, MOTN-xxx, etc.)
- Robot models (R-2000iC, M-710iC, Arc Mate, etc.)
- System parameters ($PARAM_GROUP, $MCR, etc.)
- I/O signals (DI[x], DO[x], etc.)
- Components (servo amplifier, pulsecoder, etc.)
- Procedures (mastering, RCAL, calibration, etc.)

Usage:
    from agentic.schemas.fanuc_schema import (
        FANUC_SCHEMA,
        FANUC_ERROR_PATTERNS,
        create_fanuc_domain_schema
    )

    # Check if query contains FANUC error code
    for pattern in FANUC_ERROR_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            # Route to technical documentation
            ...

    # Create domain schema for corpus
    schema = create_fanuc_domain_schema()
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple
from enum import Enum


# ============================================
# ENTITY TYPE DEFINITIONS
# ============================================

class FANUCEntityType(str, Enum):
    """Types of entities in FANUC technical documentation"""
    ERROR_CODE = "error_code"           # SRVO-001, MOTN-023, etc.
    ROBOT_MODEL = "robot_model"         # R-2000iC, M-710iC, Arc Mate
    CONTROLLER = "controller"           # R-30iA, R-30iB Plus
    COMPONENT = "component"             # Servo amplifier, encoder, motor
    PARAMETER = "parameter"             # $PARAM_GROUP, $MCR_GRP
    IO_SIGNAL = "io_signal"             # DI[1], DO[101], etc.
    REGISTER = "register"               # R[1], PR[10], SR[5]
    PROCEDURE = "procedure"             # Mastering, RCAL, calibration
    MEASUREMENT = "measurement"         # 100mm, 45deg, 24V
    PART_NUMBER = "part_number"         # A06B-xxxx-xxxx
    AXIS = "axis"                       # J1, J2, Axis 1
    SOFTWARE = "software"               # KAREL, TP program
    SAFETY = "safety"                   # DCS, SafeMove, e-stop


class FANUCRelationType(str, Enum):
    """Types of relationships between FANUC entities"""
    CAUSES = "causes"                   # Error → Symptom
    INDICATES = "indicates"             # Symptom → Cause
    RESOLVED_BY = "resolved_by"         # Cause → Solution
    REQUIRES_PART = "requires_part"     # Solution → Part
    FOLLOWS_PROCEDURE = "follows_procedure"  # Solution → Procedure
    AFFECTS_COMPONENT = "affects_component"  # Error → Component
    HAS_PARAMETER = "has_parameter"     # Component → Parameter
    TRIGGERS = "triggers"               # Error → Error (cascading)
    COMPATIBLE_WITH = "compatible_with" # Model → Controller
    DOCUMENTED_IN = "documented_in"     # Entity → Manual section


# ============================================
# REGEX PATTERNS
# ============================================

# Error/Alarm code patterns - comprehensive FANUC coverage
FANUC_ERROR_PATTERNS: List[str] = [
    # Servo alarms (most common)
    r"SRVO-\d{3,4}",
    # Motion alarms
    r"MOTN-\d{3,4}",
    # System alarms
    r"SYST-\d{3,4}",
    # Host communication
    r"HOST-\d{3,4}",
    # Interpreter
    r"INTP-\d{3,4}",
    # Priority
    r"PRIO-\d{3,4}",
    # Communication
    r"COMM-\d{3,4}",
    # Vision
    r"VISI-\d{3,4}",
    # Serial I/O
    r"SRIO-\d{3,4}",
    # File system
    r"FILE-\d{3,4}",
    # Macro
    r"MACR-\d{3,4}",
    # Palletizing
    r"PALL-\d{3,4}",
    # Spot welding
    r"SPOT-\d{3,4}",
    # Arc welding
    r"ARC-\d{3,4}",
    # Dispense
    r"DISP-\d{3,4}",
    # Dictionary
    r"DICT-\d{3,4}",
    # Memory
    r"MEMO-\d{3,4}",
    # Condition
    r"COND-\d{3,4}",
    # Tool
    r"TOOL-\d{3,4}",
    # Acceleration/collision
    r"ACAL-\d{3,4}",
    # Generic alarm pattern
    r"ALRM-\d{3,4}",
]

# Robot model patterns
FANUC_MODEL_PATTERNS: List[str] = [
    # R-series (large robots)
    r"R-\d{4}i[A-Z](/\d+[A-Z]?)?",       # R-2000iC/165F
    # M-series (medium robots)
    r"M-\d+i[A-Z](/\d+[A-Z]?)?",         # M-710iC/50, M-20iA
    # LR Mate (small robots)
    r"LR\s*Mate\s*\d+i[A-Z](/\d+)?",     # LR Mate 200iD/7L
    # Arc Mate (welding)
    r"Arc\s*Mate\s*\d+i[A-Z](/\d+)?",    # Arc Mate 100iC
    # Paint Mate
    r"Paint\s*Mate\s*\d+i[A-Z]?",
    # CR series (collaborative)
    r"CR-\d+i[A-Z]?(/\d+)?",             # CR-35iA
    # CRX series (newer collaborative)
    r"CRX-\d+i[A-Z]?(/\d+)?",            # CRX-10iA/L
    # SCARA
    r"SR-\d+i[A-Z]?",
]

# Controller patterns
FANUC_CONTROLLER_PATTERNS: List[str] = [
    r"R-30i[AB]\s*(Plus)?",              # R-30iA, R-30iB Plus
    r"R-30i[AB]\s*Mate",                 # R-30iA Mate
    r"R-J3i[BC]",                        # Older controllers
]

# Component patterns
FANUC_COMPONENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\bJ([1-9])\b", "axis"),                      # J1, J2, J3...
    (r"\baxis\s*([1-9])\b", "axis"),                # axis 1, axis 2
    (r"servo\s*(motor|amplifier|drive)", "servo"),  # servo components
    (r"teach\s*pendant|TP\b", "teach_pendant"),     # teach pendant
    (r"pulsecoder|pulse\s*coder", "encoder"),       # encoder/pulsecoder
    (r"encoder\b", "encoder"),                       # encoder
    (r"\bmotor\b", "motor"),                         # motor
    (r"brake\s*(release)?", "brake"),               # brake
    (r"gearbox|gear\s*box|reducer", "gearbox"),     # gearbox
    (r"robot\s*arm", "robot_arm"),                  # robot arm
    (r"wrist\s*(assembly)?", "wrist"),              # wrist
    (r"end\s*effector|EOAT", "end_effector"),       # end effector
    (r"gripper", "gripper"),                        # gripper
    (r"tool\s*flange", "tool_flange"),              # tool flange
    (r"cable\s*(assembly)?", "cable"),              # cable
    (r"connector", "connector"),                    # connector
    (r"PCB|circuit\s*board", "pcb"),                # circuit board
    (r"power\s*supply|PSU", "power_supply"),        # power supply
    (r"cooling\s*fan", "fan"),                      # cooling fan
    (r"battery|backup\s*battery", "battery"),       # battery
]

# Parameter patterns (system variables)
FANUC_PARAMETER_PATTERNS: List[str] = [
    r"\$[A-Z_]+(\[\d+\])?(\.[A-Z_]+)?",  # $PARAM_GROUP, $MCR[1].STATUS
    r"\$SCR_GRP\[\d+\]",                  # Screen group
    r"\$MOTYPE",                          # Motion type
    r"\$SPEED",                           # Speed setting
    r"\$TERMTYPE",                        # Termination type
    r"\$MASTER",                          # Master reference
    r"\$DCSS_",                           # DCS safety
]

# I/O signal patterns
FANUC_IO_PATTERNS: List[str] = [
    r"(D|R|U|S|W|G|A)(I|O)\[\d+\]",       # DI[1], DO[1], RI[1], RO[1], etc.
    r"(D|R|U|S|W|G|A)(I|O)\s*\d+",        # DI 1, DO 1 (without brackets)
    r"(F|M|UI|UO|SI|SO|WI|WO|GI|GO|AI|AO)\[\d+\]",  # Extended I/O
]

# Register patterns
FANUC_REGISTER_PATTERNS: List[str] = [
    r"R\[\d+\]",                          # Numeric register
    r"PR\[\d+\]",                         # Position register
    r"SR\[\d+\]",                         # String register
    r"AR\[\d+\]",                         # Argument register
    r"VR\[\d+\]",                         # Vision register
]

# Procedure/action patterns
FANUC_PROCEDURE_PATTERNS: List[Tuple[str, str]] = [
    (r"master(ing)?", "mastering"),
    (r"RCAL|recalibrat", "calibration"),
    (r"zero\s*point|zero\s*position", "zero_point"),
    (r"backup|back\s*up", "backup"),
    (r"restore", "restore"),
    (r"cold\s*start", "cold_start"),
    (r"controlled\s*start", "controlled_start"),
    (r"hot\s*start", "hot_start"),
    (r"power\s*cycle", "power_cycle"),
    (r"reset", "reset"),
    (r"clear\s*alarm", "clear_alarm"),
    (r"fault\s*reset", "fault_reset"),
    (r"teach(ing)?", "teaching"),
    (r"jog(ging)?", "jogging"),
    (r"payload\s*(identification|setting)", "payload_id"),
    (r"collision\s*detect", "collision_detect"),
    (r"soft\s*limit", "soft_limit"),
    (r"world\s*frame", "world_frame"),
    (r"user\s*frame", "user_frame"),
    (r"tool\s*(frame|center\s*point|TCP)", "tool_frame"),
]

# Measurement patterns
FANUC_MEASUREMENT_PATTERNS: List[str] = [
    r"\d+\.?\d*\s*mm",                    # millimeters
    r"\d+\.?\d*\s*cm",                    # centimeters
    r"\d+\.?\d*\s*m\b",                   # meters
    r"\d+\.?\d*\s*(deg|°)",               # degrees
    r"\d+\.?\d*\s*rpm",                   # rotations per minute
    r"\d+\.?\d*\s*[AV]",                  # amps/volts
    r"\d+\.?\d*\s*kW",                    # kilowatts
    r"\d+\.?\d*\s*(N[·m]|Nm)",            # newton-meters
    r"\d+\.?\d*\s*(kg|lb)",               # kilograms/pounds
    r"\d+\.?\d*\s*(ms|sec|s)\b",          # time
    r"\d+\.?\d*\s*Hz",                    # frequency
    r"\d+\.?\d*\s*%",                     # percentage
]

# Part number patterns
FANUC_PART_PATTERNS: List[str] = [
    r"A\d{2}B-\d{4}-[A-Z]\d{3}",          # Standard FANUC parts
    r"A\d{2}B-\d{4}-[A-Z]\d{2,4}",        # Variant
]

# Safety-related patterns
FANUC_SAFETY_PATTERNS: List[str] = [
    r"DCS\b|Dual\s*Check\s*Safety",       # DCS
    r"SafeMove",                          # SafeMove
    r"[Ee][-]?[Ss]top|emergency\s*stop",  # E-stop
    r"deadman|dead\s*man",                # Deadman switch
    r"safety\s*fence",                    # Safety fence
    r"light\s*curtain",                   # Light curtain
    r"safe\s*(speed|position|zone)",      # Safe limits
    r"collaborative|cobot",               # Collaborative operation
]


# ============================================
# COMPILED PATTERNS (for performance)
# ============================================

COMPILED_ERROR_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in FANUC_ERROR_PATTERNS
]

COMPILED_MODEL_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in FANUC_MODEL_PATTERNS
]


# ============================================
# DOMAIN SCHEMA DATACLASS
# ============================================

@dataclass
class FANUCEntityDef:
    """Definition of a FANUC entity type for extraction"""
    name: str
    description: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    context_window: int = 50  # Characters of context to capture


@dataclass
class FANUCDomainSchema:
    """Complete FANUC domain schema for entity extraction"""
    name: str = "fanuc_robotics"
    description: str = "FANUC robot technical documentation"
    entities: List[FANUCEntityDef] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)


# ============================================
# SCHEMA FACTORY
# ============================================

def create_fanuc_domain_schema() -> FANUCDomainSchema:
    """
    Create a complete FANUC domain schema for entity extraction.

    Returns:
        FANUCDomainSchema with all entity definitions and relationships
    """
    entities = [
        FANUCEntityDef(
            name="error_code",
            description="FANUC alarm and error codes",
            patterns=FANUC_ERROR_PATTERNS,
            examples=["SRVO-001", "MOTN-023", "SYST-100", "INTP-127"]
        ),
        FANUCEntityDef(
            name="robot_model",
            description="FANUC robot model identifiers",
            patterns=FANUC_MODEL_PATTERNS,
            examples=["R-2000iC/165F", "M-710iC/50", "LR Mate 200iD/7L"]
        ),
        FANUCEntityDef(
            name="controller",
            description="FANUC controller model identifiers",
            patterns=FANUC_CONTROLLER_PATTERNS,
            examples=["R-30iA", "R-30iB Plus", "R-30iA Mate"]
        ),
        FANUCEntityDef(
            name="parameter",
            description="System parameters and variables",
            patterns=FANUC_PARAMETER_PATTERNS,
            examples=["$PARAM_GROUP", "$MCR[1].STATUS", "$MOTYPE"]
        ),
        FANUCEntityDef(
            name="io_signal",
            description="Input/Output signal references",
            patterns=FANUC_IO_PATTERNS,
            examples=["DI[1]", "DO[101]", "RI[5]", "UI[3]"]
        ),
        FANUCEntityDef(
            name="register",
            description="Register references",
            patterns=FANUC_REGISTER_PATTERNS,
            examples=["R[1]", "PR[10]", "SR[5]", "VR[1]"]
        ),
        FANUCEntityDef(
            name="component",
            description="Robot components and parts",
            patterns=[p for p, _ in FANUC_COMPONENT_PATTERNS],
            keywords=[
                "servo motor", "encoder", "pulsecoder", "brake",
                "reducer", "gearbox", "cable", "teach pendant",
                "controller", "amplifier", "axis", "joint",
                "wrist", "arm", "base", "tool flange", "end effector"
            ],
            examples=["J1 motor", "servo amplifier", "pulsecoder"]
        ),
        FANUCEntityDef(
            name="procedure",
            description="Technical procedures and methods",
            patterns=[p for p, _ in FANUC_PROCEDURE_PATTERNS],
            keywords=[
                "mastering", "calibration", "RCAL", "backup",
                "restore", "cold start", "reset", "teaching",
                "jogging", "payload identification"
            ],
            examples=["mastering procedure", "RCAL calibration", "zero position return"]
        ),
        FANUCEntityDef(
            name="measurement",
            description="Numeric measurements with units",
            patterns=FANUC_MEASUREMENT_PATTERNS,
            examples=["100mm", "45deg", "1500rpm", "24V", "50Nm"]
        ),
        FANUCEntityDef(
            name="part_number",
            description="FANUC part numbers",
            patterns=FANUC_PART_PATTERNS,
            examples=["A06B-6079-H101", "A06B-0238-B605"]
        ),
        FANUCEntityDef(
            name="safety",
            description="Safety-related terms and systems",
            patterns=FANUC_SAFETY_PATTERNS,
            keywords=[
                "DCS", "dual check safety", "SafeMove", "e-stop",
                "deadman", "safety fence", "light curtain",
                "safe speed", "safe position", "collaborative"
            ],
            examples=["DCS Safe Position", "e-stop circuit", "SafeMove Pro"]
        ),
    ]

    relationships = [
        # Error relationships
        ("error_code", "causes", "component"),
        ("error_code", "indicates", "procedure"),
        ("error_code", "triggers", "error_code"),

        # Component relationships
        ("component", "part_of", "robot_model"),
        ("component", "has_parameter", "parameter"),
        ("component", "uses_signal", "io_signal"),

        # Procedure relationships
        ("procedure", "resolves", "error_code"),
        ("procedure", "affects", "component"),
        ("procedure", "uses", "parameter"),
        ("procedure", "requires", "measurement"),

        # Model relationships
        ("robot_model", "compatible_with", "controller"),
        ("robot_model", "has_component", "component"),

        # Safety relationships
        ("safety", "monitors", "component"),
        ("safety", "uses", "io_signal"),
    ]

    return FANUCDomainSchema(
        name="fanuc_robotics",
        description="FANUC robot technical documentation for troubleshooting and maintenance",
        entities=entities,
        relationships=relationships
    )


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def is_fanuc_query(query: str) -> bool:
    """
    Quick check if a query is FANUC-related.

    Args:
        query: User query string

    Returns:
        True if query contains FANUC-specific patterns
    """
    query_upper = query.upper()

    # Check error codes first (most definitive)
    for pattern in COMPILED_ERROR_PATTERNS:
        if pattern.search(query_upper):
            return True

    # Check model patterns
    for pattern in COMPILED_MODEL_PATTERNS:
        if pattern.search(query):
            return True

    # Check keywords
    fanuc_keywords = [
        "fanuc", "teach pendant", "pulsecoder", "mastering",
        "rcal", "karel", "tp program", "roboguide"
    ]

    query_lower = query.lower()
    return any(kw in query_lower for kw in fanuc_keywords)


def extract_error_codes(text: str) -> List[str]:
    """
    Extract all FANUC error codes from text.

    Args:
        text: Text to search

    Returns:
        List of unique error codes found
    """
    codes = set()
    for pattern in COMPILED_ERROR_PATTERNS:
        matches = pattern.findall(text.upper())
        codes.update(matches)
    return sorted(codes)


def get_error_category(error_code: str) -> str:
    """
    Get the category for an error code.

    Args:
        error_code: FANUC error code (e.g., "SRVO-001")

    Returns:
        Category name (e.g., "Servo Alarms")
    """
    prefix = error_code.split("-")[0].upper() if "-" in error_code else ""

    categories = {
        "SRVO": "Servo Alarms",
        "MOTN": "Motion Alarms",
        "SYST": "System Alarms",
        "HOST": "Host Communication Alarms",
        "INTP": "Interpreter Alarms",
        "PRIO": "Priority Alarms",
        "COMM": "Communication Alarms",
        "VISI": "Vision Alarms",
        "SRIO": "Serial I/O Alarms",
        "FILE": "File System Alarms",
        "MACR": "Macro Alarms",
        "PALL": "Palletizing Alarms",
        "SPOT": "Spot Welding Alarms",
        "ARC": "Arc Welding Alarms",
        "DISP": "Dispense Alarms",
    }

    return categories.get(prefix, "Unknown Category")


# ============================================
# SINGLETON SCHEMA INSTANCE
# ============================================

FANUC_SCHEMA = create_fanuc_domain_schema()
