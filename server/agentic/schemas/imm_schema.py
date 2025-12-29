"""
Injection Molding Machine (IMM) & Robot Integration Domain Schema

Defines entity types, patterns, and relationships for:
- KraussMaffei, Cincinnati Milacron, Van Dorn injection molding machines
- FANUC robot integration via Euromap 67/73/77 protocols
- Machine operation, troubleshooting, maintenance, and calibration

Used by:
- DomainCorpusManager for corpus building
- UniversalOrchestrator for query routing
- IMMCorpusBuilder for entity extraction

Sources:
- Euromap standards (67, 67.1, 73, 77)
- Manufacturer documentation (KraussMaffei MC5/MC6, Milacron MOSAIC)
- Technical forums (Robot-Forum, PLCTalk, InjectionMoldingOnline)
- Trade publications (Plastics Technology, RJG)

Usage:
    from agentic.schemas.imm_schema import (
        IMM_SCHEMA,
        create_imm_domain_schema,
        is_imm_query,
        EUROMAP_PATTERNS
    )

    # Check if query is IMM-related
    if is_imm_query(query):
        # Route to IMM corpus
        ...

    # Create domain schema for corpus
    schema = create_imm_domain_schema()
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple, Set
from enum import Enum


# ============================================
# ENTITY TYPE DEFINITIONS
# ============================================

class IMMEntityType(str, Enum):
    """Types of entities in IMM technical documentation"""
    # Machine-related
    MACHINE_MODEL = "machine_model"          # KM MX1600, CM Vista Toggle, Van Dorn HT
    CONTROL_SYSTEM = "control_system"        # MC6, MOSAIC+, PathFinder, Acramatic
    COMPONENT = "component"                  # Barrel, screw, clamp, hydraulic unit
    ERROR_CODE = "error_code"                # Machine alarm codes
    PARAMETER = "parameter"                  # Machine settings

    # Protocol-related
    EUROMAP_SIGNAL = "euromap_signal"        # Euromap 67 I/O signals
    EUROMAP_PROTOCOL = "euromap_protocol"    # Euromap 67, 73, 77
    INTERFACE = "interface"                  # Robot-IMM interface

    # Process-related
    PROCESS_VARIABLE = "process_variable"    # Pack pressure, fill time, etc.
    DEFECT = "defect"                        # Short shot, flash, sink marks
    MATERIAL = "material"                    # Nylon, PP, ABS, PC

    # Troubleshooting
    SYMPTOM = "symptom"                      # Observable issue
    CAUSE = "cause"                          # Root cause
    SOLUTION = "solution"                    # Fix or remedy
    PROCEDURE = "procedure"                  # Calibration, startup, maintenance

    # Parts and specifications
    PART_NUMBER = "part_number"              # OEM part numbers
    MEASUREMENT = "measurement"              # Tonnage, shot size, etc.
    TOOL = "tool"                            # Diagnostic tools


class IMMRelationType(str, Enum):
    """Types of relationships between IMM entities"""
    # Error/troubleshooting
    CAUSES = "causes"                        # Cause → Defect
    INDICATES = "indicates"                  # Symptom → Cause
    RESOLVED_BY = "resolved_by"              # Cause → Solution
    PREVENTS = "prevents"                    # Solution → Defect

    # Machine relationships
    USES_CONTROL = "uses_control"            # Machine → Control System
    HAS_COMPONENT = "has_component"          # Machine → Component
    COMPATIBLE_WITH = "compatible_with"      # Robot → Machine

    # Protocol relationships
    SIGNALS_WITH = "signals_with"            # Interface → Euromap Signal
    FOLLOWS_PROTOCOL = "follows_protocol"    # Interface → Euromap Protocol

    # Process relationships
    AFFECTS_VARIABLE = "affects_variable"    # Parameter → Process Variable
    PRODUCES_DEFECT = "produces_defect"      # Process deviation → Defect
    SUITABLE_FOR = "suitable_for"            # Material → Process

    # Documentation
    DOCUMENTED_IN = "documented_in"          # Entity → Source
    REQUIRES_PROCEDURE = "requires_procedure" # Solution → Procedure


# ============================================
# MANUFACTURER PATTERNS
# ============================================

# KraussMaffei patterns
KRAUSSMAFFEI_MODEL_PATTERNS: List[str] = [
    r"KM\s*\d+",                              # KM 80, KM 1600
    r"MX\s*\d+(-\d+)?",                       # MX1600, MX1600-8100
    r"CX\s*\d+(-\d+)?",                       # CX 35-180
    r"GX\s*\d+(-\d+)?",                       # GX 450
    r"PX\s*\d+(-\d+)?",                       # PX 25-55
    r"KraussMaffei\s+[A-Z]{2}\s*\d+",         # KraussMaffei MX 1600
]

KRAUSSMAFFEI_CONTROL_PATTERNS: List[str] = [
    r"MC6\s*(Touch)?",                        # MC6, MC6 Touch
    r"MC5",
    r"MC4",
    r"MC3F?",
    r"VARAN",                                 # VARAN bus
]

# Cincinnati Milacron patterns
MILACRON_MODEL_PATTERNS: List[str] = [
    r"Vista\s*Toggle\s*\d+",                  # Vista Toggle 85, Vista Toggle 440
    r"VT\s*\d+(-\d+)?",                       # VT 85, VT 85-440
    r"Arrow\s*[A-Z]?\d*",                     # Arrow, Arrow E, Arrow ERO
    r"Hawk\s*\d+",                            # Hawk 225
    r"Magna\s*[T]?\s*\d+",                    # Magna T 500
    r"M-Series\s*\d+",                        # M-Series 500
    r"Roboshot\s*[a-z]*\d*",                  # Roboshot (note: FANUC brand)
    r"Elektron\s*\d+",                        # Elektron 110
    r"Powerline\s*\d+",                       # Powerline NT 440
]

MILACRON_CONTROL_PATTERNS: List[str] = [
    r"MOSAIC\+?",                             # MOSAIC, MOSAIC+
    r"Acramatic\s*\d+[A-Z]*",                 # Acramatic 2100, Acramatic 2100E
    r"Camac\s*\d*",                           # Camac, Camac 64
    r"PathFinder",
    r"Siemens\s*S5",                          # Siemens S5 (Van Dorn)
    r"GE Fanuc\s*\d+-\d+",                    # GE Fanuc 90-30
]

# Van Dorn / Sumitomo Demag patterns
VANDORN_MODEL_PATTERNS: List[str] = [
    r"Van\s*Dorn\s*\d+\s*HT",                 # Van Dorn 320 HT
    r"HT\s*\d+(-\d+)?",                       # HT 270, HT 300A
    r"H\d{3}[A]?",                            # H270, H300A, H400
    r"Demag\s*\d+[A-Z]*",                     # Demag 500
    r"El-Exis\s*[A-Z]?\s*\d+",                # El-Exis SP 200
    r"IntElect\s*\d+",                        # IntElect 100
    r"Systec\s*\d+",                          # Systec 130
    r"Newbury\s*\d+",                         # Newbury (legacy)
]

VANDORN_CONTROL_PATTERNS: List[str] = [
    r"PathFinder\s*\d*",                      # PathFinder, PathFinder 2
    r"NC5",                                   # NC5 controller
    r"CC\d+",                                 # CC100, CC200, CC300
    r"B&R\s*\d+",                             # B&R controllers
]


# ============================================
# EUROMAP PROTOCOL PATTERNS
# ============================================

EUROMAP_SIGNAL_PATTERNS: Dict[str, List[str]] = {
    "euromap_67": [
        # Robot → IMM signals (outputs from robot perspective)
        r"Robot\s*(Ready|Request)",           # Robot Ready, Robot Request
        r"Auto\s*Mode",                        # Auto Mode
        r"Mould\s*Open\s*ACK",                 # Mould Open Acknowledge
        r"Safety\s*Gate\s*Close",              # Safety Gate Close Request
        r"Cycle\s*Start",                      # Cycle Start
        r"Ejector\s*(Advance|Retract)",        # Ejector commands
        r"Core\s*(In|Out)",                    # Core movements

        # IMM → Robot signals (inputs from robot perspective)
        r"Machine\s*(Ready|Auto)",             # Machine Ready, Machine Auto
        r"Mould\s*(Open|Close|Opened|Closed)", # Mould status
        r"Ejector\s*(Forward|Back)",           # Ejector status
        r"Safety\s*Beam\s*(OK|Fault)",         # Safety beam status
        r"Cycle\s*Complete",                   # Cycle Complete
        r"Alarm\s*Active",                     # Machine Alarm
    ],
    "euromap_73": [
        r"Safety\s*(Circuit|Zone)\s*\d*",      # Safety zones
        r"DCS\s*(Zone|Monitor)",               # DCS safety
        r"E-Stop",                             # Emergency stop
        r"Interlock\s*\d*",                    # Safety interlocks
    ],
    "euromap_77": [
        # OPC UA based (later version)
        r"EM77\s*\w+",                         # EM77 namespace
        r"MachineStatus",
        r"CycleData",
        r"ProcessData",
    ],
}

EUROMAP_PROTOCOL_PATTERNS: List[str] = [
    r"Euromap\s*6[7-9](\.\d)?",               # Euromap 67, 67.1, 68, 69
    r"Euromap\s*7[0-9]",                      # Euromap 70-79
    r"Euromap\s*8[0-3]",                      # Euromap 80-83
    r"EM\s*6[7-9]",                           # EM67, EM68, EM69
    r"EM\s*7[0-9]",                           # EM70-79
    r"SPI\s*Interface",                       # SPI (US equivalent)
]


# ============================================
# PROCESS AND DEFECT PATTERNS
# ============================================

PROCESS_VARIABLE_PATTERNS: List[Tuple[str, str]] = [
    # Fill phase
    (r"fill\s*(time|speed|pressure)", "fill_phase"),
    (r"injection\s*(speed|pressure|time)", "fill_phase"),
    (r"velocity\s*profile", "fill_phase"),
    (r"transfer\s*(point|pressure)", "fill_phase"),

    # Pack/hold phase
    (r"pack(ing)?\s*(time|pressure|speed)", "pack_phase"),
    (r"hold(ing)?\s*(time|pressure)", "pack_phase"),
    (r"cushion", "pack_phase"),

    # Cool phase
    (r"cool(ing)?\s*time", "cool_phase"),
    (r"cycle\s*time", "cool_phase"),
    (r"mold\s*temperature", "cool_phase"),

    # Plastication
    (r"screw\s*(speed|RPM|recovery)", "plastication"),
    (r"back\s*pressure", "plastication"),
    (r"melt\s*temperature", "plastication"),
    (r"barrel\s*(temp|zone)", "plastication"),

    # Clamp
    (r"clamp\s*(force|tonnage|pressure)", "clamp"),
    (r"mold\s*protection", "clamp"),
]

DEFECT_PATTERNS: Dict[str, List[str]] = {
    "short_shot": [
        r"short\s*shot",
        r"incomplete\s*(fill|part)",
        r"unfilled\s*(cavity|section)",
        r"non-fill",
    ],
    "flash": [
        r"\bflash\b",
        r"overflow",
        r"parting\s*line\s*(leak|flash)",
    ],
    "sink_marks": [
        r"sink\s*(mark|spot)s?",
        r"depression",
        r"surface\s*depression",
    ],
    "warpage": [
        r"warp(age|ing)?",
        r"distortion",
        r"bow(ing)?",
        r"twist(ing)?",
    ],
    "burn_marks": [
        r"burn\s*(mark|spot)s?",
        r"diesel(ing)?",
        r"gas\s*burn",
        r"degradation",
    ],
    "voids": [
        r"\bvoid(s)?\b",
        r"bubble(s)?",
        r"vacuum\s*void",
        r"air\s*trap",
    ],
    "weld_lines": [
        r"weld\s*line(s)?",
        r"knit\s*line(s)?",
        r"flow\s*line(s)?",
    ],
    "jetting": [
        r"jetting",
        r"snake\s*pattern",
        r"worm\s*track",
    ],
    "splay": [
        r"splay",
        r"silver\s*streak",
        r"moisture\s*streak",
    ],
    "delamination": [
        r"delamination",
        r"layer\s*separation",
        r"peeling",
    ],
    "brittleness": [
        r"brittle(ness)?",
        r"cracking",
        r"stress\s*crack",
    ],
    "discoloration": [
        r"discoloration",
        r"color\s*(shift|variation)",
        r"yellowing",
    ],
    "gate_blush": [
        r"gate\s*blush",
        r"gate\s*(mark|vestige)",
    ],
    "outgassing": [
        r"outgas(sing)?",
        r"gas\s*trap",
        r"vent(ing)?\s*(issue|problem)",
    ],
}

MATERIAL_PATTERNS: List[Tuple[str, str]] = [
    # Commodity plastics
    (r"\bPP\b|polypropylene", "polypropylene"),
    (r"\bPE\b|polyethylene|HDPE|LDPE|LLDPE", "polyethylene"),
    (r"\bPS\b|polystyrene|HIPS", "polystyrene"),
    (r"\bPVC\b|vinyl", "pvc"),

    # Engineering plastics
    (r"\bABS\b", "abs"),
    (r"\bPC\b|polycarbonate", "polycarbonate"),
    (r"\bPA\b|nylon|PA6|PA66|PA12", "nylon"),
    (r"\bPOM\b|acetal|delrin", "acetal"),
    (r"\bPBT\b|PET\b|polyester", "polyester"),
    (r"\bPMMA\b|acrylic", "acrylic"),

    # High-performance
    (r"\bPEEK\b", "peek"),
    (r"\bPEI\b|ultem", "pei"),
    (r"\bPPS\b", "pps"),
    (r"\bLCP\b", "lcp"),
    (r"\bPTFE\b|teflon", "ptfe"),

    # Filled/reinforced
    (r"glass\s*filled|GF\d+%?", "glass_filled"),
    (r"carbon\s*fiber|CF\d+%?", "carbon_filled"),
    (r"mineral\s*filled", "mineral_filled"),
]


# ============================================
# RJG SCIENTIFIC MOLDING PATTERNS
# ============================================

RJG_PATTERNS: List[str] = [
    r"DECOUPLED\s*(MOLDING|I+)?",             # DECOUPLED MOLDING, DECOUPLED II, III
    r"scientific\s*molding",
    r"4\s*plastic\s*variables",
    r"cavity\s*pressure",
    r"eDART",                                  # RJG monitoring system
    r"CoPilot",                               # RJG CoPilot
    r"Mold\s*Analyst",
    r"process\s*window",
    r"viscosity\s*curve",
    r"relative\s*viscosity",
]


# ============================================
# COMPONENT AND PART PATTERNS
# ============================================

IMM_COMPONENT_PATTERNS: List[Tuple[str, str]] = [
    # Injection unit
    (r"screw\s*(tip|check ring|assembly)?", "injection_unit"),
    (r"barrel\s*(heater|zone)?", "injection_unit"),
    (r"nozzle", "injection_unit"),
    (r"shot\s*size", "injection_unit"),
    (r"injection\s*unit", "injection_unit"),
    (r"plastici(z|s)er", "injection_unit"),

    # Clamp unit
    (r"clamp\s*(unit|system)?", "clamp_unit"),
    (r"toggle\s*(clamp)?", "clamp_unit"),
    (r"platen\s*(fixed|moving)?", "clamp_unit"),
    (r"tie\s*bar(s)?", "clamp_unit"),
    (r"mold\s*height", "clamp_unit"),

    # Hydraulics
    (r"hydraulic\s*(pump|system|unit)", "hydraulics"),
    (r"accumulator", "hydraulics"),
    (r"servo\s*valve", "hydraulics"),
    (r"proportional\s*valve", "hydraulics"),

    # Mold
    (r"\bmold\b|\bmould\b", "mold"),
    (r"cavity|cavities", "mold"),
    (r"core\s*(pin|pull)?", "mold"),
    (r"ejector\s*(pin|plate|system)?", "mold"),
    (r"runner\s*(system)?", "mold"),
    (r"hot\s*runner", "mold"),
    (r"gate\s*(type|location)?", "mold"),
    (r"sprue", "mold"),
    (r"cooling\s*(channel|circuit)", "mold"),

    # Safety
    (r"safety\s*(gate|door|fence)", "safety"),
    (r"light\s*curtain", "safety"),
    (r"interlock", "safety"),
    (r"e-stop|emergency\s*stop", "safety"),
]

# Part number patterns by manufacturer
PART_NUMBER_PATTERNS: Dict[str, List[str]] = {
    "kraussmaffei": [
        r"\d{7,10}",                           # 7-10 digit KM part numbers
    ],
    "milacron": [
        r"\d{6}",                              # 6 digit Milacron parts
        r"MIL-\d+",                            # MIL-prefix
    ],
    "vandorn": [
        r"VD-\d+",
        r"SD-\d+",                             # Sumitomo parts
    ],
    "siemens": [
        r"6ES\d+-\d+\w+-\d+\w+",               # Siemens S7/S5 parts
    ],
    "fanuc": [
        r"A\d{2}B-\d{4}-[A-Z]\d+",             # FANUC parts
    ],
}


# ============================================
# MEASUREMENT PATTERNS
# ============================================

IMM_MEASUREMENT_PATTERNS: List[str] = [
    # Force/pressure
    r"\d+\.?\d*\s*(ton(s)?|tonnes?)",          # Clamp tonnage
    r"\d+\.?\d*\s*(psi|bar|MPa|kPa)",          # Pressure
    r"\d+\.?\d*\s*(kN|N)",                     # Force

    # Temperature
    r"\d+\.?\d*\s*(°[CF]|deg\s*[CF])",         # Temperature
    r"\d+\.?\d*\s*degrees",

    # Volume/weight
    r"\d+\.?\d*\s*(oz|g|kg)",                  # Shot weight
    r"\d+\.?\d*\s*(cc|cm³|in³|cu\s*in)",       # Shot volume

    # Speed/time
    r'\d+\.?\d*\s*(mm/s|in/s|ipm|"/s)',        # Injection speed
    r"\d+\.?\d*\s*(sec|s|ms)",                 # Time
    r"\d+\.?\d*\s*(rpm|RPM)",                  # Screw speed

    # Distance
    r'\d+\.?\d*\s*(mm|cm|in|")',               # Position/distance

    # Flow rate
    r"\d+\.?\d*\s*(gal/min|gpm|l/min|lpm)",    # Cooling flow
]


# ============================================
# COMPILED PATTERNS (for performance)
# ============================================

COMPILED_EUROMAP_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in EUROMAP_PROTOCOL_PATTERNS
]

COMPILED_KM_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in KRAUSSMAFFEI_MODEL_PATTERNS
]

COMPILED_CM_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in MILACRON_MODEL_PATTERNS
]

COMPILED_VD_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in VANDORN_MODEL_PATTERNS
]


# ============================================
# DOMAIN SCHEMA DATACLASS
# ============================================

@dataclass
class IMMEntityDef:
    """Definition of an IMM entity type for extraction"""
    name: str
    description: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    context_window: int = 100  # Characters of context to capture


@dataclass
class IMMDomainSchema:
    """Complete IMM domain schema for entity extraction"""
    name: str = "imm_robotics"
    description: str = "Injection molding machine and robot integration"
    entities: List[IMMEntityDef] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    trusted_domains: List[str] = field(default_factory=list)


# ============================================
# SCHEMA FACTORY
# ============================================

def create_imm_domain_schema() -> IMMDomainSchema:
    """
    Create a complete IMM domain schema for entity extraction.

    Returns:
        IMMDomainSchema with all entity definitions and relationships
    """
    entities = [
        # Machine models
        IMMEntityDef(
            name="machine_model",
            description="Injection molding machine models",
            patterns=KRAUSSMAFFEI_MODEL_PATTERNS + MILACRON_MODEL_PATTERNS + VANDORN_MODEL_PATTERNS,
            keywords=[
                "kraussmaffei", "milacron", "cincinnati", "van dorn", "sumitomo",
                "demag", "injection molding machine", "IMM", "toggle", "all-electric"
            ],
            examples=["KM MX1600", "Vista Toggle 440", "Van Dorn 320 HT", "IntElect 100"]
        ),

        # Control systems
        IMMEntityDef(
            name="control_system",
            description="Machine control systems and HMIs",
            patterns=KRAUSSMAFFEI_CONTROL_PATTERNS + MILACRON_CONTROL_PATTERNS + VANDORN_CONTROL_PATTERNS,
            keywords=[
                "controller", "HMI", "PLC", "control system", "operator interface"
            ],
            examples=["MC6", "MOSAIC+", "PathFinder", "Acramatic 2100"]
        ),

        # Euromap protocols
        IMMEntityDef(
            name="euromap_protocol",
            description="Robot-IMM interface protocols",
            patterns=EUROMAP_PROTOCOL_PATTERNS,
            keywords=[
                "euromap", "SPI", "robot interface", "automation interface",
                "EM67", "EM73", "EM77"
            ],
            examples=["Euromap 67", "Euromap 67.1", "Euromap 73", "Euromap 77"]
        ),

        # Euromap signals
        IMMEntityDef(
            name="euromap_signal",
            description="Euromap interface signals",
            patterns=EUROMAP_SIGNAL_PATTERNS["euromap_67"] + EUROMAP_SIGNAL_PATTERNS["euromap_73"],
            keywords=[
                "signal", "I/O", "input", "output", "robot ready", "machine ready",
                "mould open", "cycle start"
            ],
            examples=["Robot Ready", "Machine Auto", "Mould Opened", "Ejector Forward"]
        ),

        # Process variables
        IMMEntityDef(
            name="process_variable",
            description="Injection molding process parameters",
            patterns=[p for p, _ in PROCESS_VARIABLE_PATTERNS],
            keywords=[
                "fill", "pack", "hold", "cool", "injection", "plasticizing",
                "cushion", "transfer", "velocity", "pressure", "temperature"
            ],
            examples=["fill time", "pack pressure", "melt temperature", "back pressure"]
        ),

        # Defects
        IMMEntityDef(
            name="defect",
            description="Injection molding defects",
            patterns=[p for patterns in DEFECT_PATTERNS.values() for p in patterns],
            keywords=list(DEFECT_PATTERNS.keys()) + [
                "defect", "quality issue", "cosmetic", "dimensional"
            ],
            examples=["short shot", "flash", "sink marks", "warpage", "burn marks"]
        ),

        # Materials
        IMMEntityDef(
            name="material",
            description="Plastic materials and resins",
            patterns=[p for p, _ in MATERIAL_PATTERNS],
            keywords=[
                "resin", "plastic", "polymer", "material", "filled", "reinforced",
                "commodity", "engineering plastic"
            ],
            examples=["ABS", "polycarbonate", "nylon 66", "glass filled PP"]
        ),

        # Components
        IMMEntityDef(
            name="component",
            description="Machine and mold components",
            patterns=[p for p, _ in IMM_COMPONENT_PATTERNS],
            keywords=[
                "screw", "barrel", "nozzle", "clamp", "platen", "tie bar",
                "mold", "ejector", "runner", "gate", "hydraulic"
            ],
            examples=["screw tip", "check ring", "tie bar", "hot runner", "ejector pin"]
        ),

        # RJG/Scientific molding
        IMMEntityDef(
            name="scientific_molding",
            description="Scientific molding methodologies",
            patterns=RJG_PATTERNS,
            keywords=[
                "RJG", "scientific molding", "decoupled", "cavity pressure",
                "process window", "viscosity", "systematic"
            ],
            examples=["DECOUPLED MOLDING", "cavity pressure", "eDART", "process window"]
        ),

        # Measurements
        IMMEntityDef(
            name="measurement",
            description="Process measurements and specifications",
            patterns=IMM_MEASUREMENT_PATTERNS,
            keywords=["tonnage", "pressure", "temperature", "speed", "time", "size"],
            examples=["500 ton", "10000 psi", "200°C", "2 seconds"]
        ),
    ]

    relationships = [
        # Machine relationships
        ("machine_model", "uses_control", "control_system"),
        ("machine_model", "has_component", "component"),
        ("machine_model", "compatible_with", "euromap_protocol"),

        # Protocol relationships
        ("euromap_protocol", "defines", "euromap_signal"),
        ("interface", "follows_protocol", "euromap_protocol"),

        # Process relationships
        ("process_variable", "affects", "defect"),
        ("material", "requires", "process_variable"),
        ("defect", "caused_by", "cause"),
        ("cause", "resolved_by", "solution"),

        # Troubleshooting
        ("symptom", "indicates", "cause"),
        ("solution", "requires_procedure", "procedure"),
        ("defect", "prevented_by", "solution"),

        # Component relationships
        ("component", "part_of", "machine_model"),
        ("component", "has_parameter", "process_variable"),
    ]

    trusted_domains = [
        # Official manufacturer sites
        "kraussmaffei.com",
        "milacron.com",
        "sumitomo-shi-demag.us",
        "sumitomo-shi-demag.eu",

        # Standards organizations
        "euromap.org",
        "plastics.org",
        "spi.org",

        # Technical forums
        "injectionmoldingonline.com",
        "robot-forum.com",
        "plctalk.net",
        "practicalmachinist.com",
        "eng-tips.com",

        # Trade publications
        "ptonline.com",
        "plasticstoday.com",
        "plasticsnews.com",

        # Training/education
        "rjginc.com",
        "traininteractive.com",
        "aim.institute",
        "paulsontraining.com",
        "4spe.org",

        # Material suppliers (technical data)
        "basf.com",
        "dupont.com",
        "sabic.com",
        "covestro.com",

        # Parts/service
        "mcspt.com",
        "industrialmanuals.com",
        "controlrepair.com",

        # Robot manufacturers
        "fanuc.eu",
        "fanuc.com",
        "abb.com",
        "kuka.com",
    ]

    return IMMDomainSchema(
        name="imm_robotics",
        description="Injection molding machine and FANUC robot integration for troubleshooting and maintenance",
        entities=entities,
        relationships=relationships,
        trusted_domains=trusted_domains
    )


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def is_imm_query(query: str) -> bool:
    """
    Quick check if a query is IMM/injection molding related.

    Args:
        query: User query string

    Returns:
        True if query contains IMM-specific patterns
    """
    query_lower = query.lower()

    # Check for Euromap mentions (strong signal)
    for pattern in COMPILED_EUROMAP_PATTERNS:
        if pattern.search(query):
            return True

    # Check for manufacturer mentions
    for pattern_list in [COMPILED_KM_PATTERNS, COMPILED_CM_PATTERNS, COMPILED_VD_PATTERNS]:
        for pattern in pattern_list:
            if pattern.search(query):
                return True

    # Check for IMM keywords
    imm_keywords = [
        "injection mold", "injection mould", "molding machine", "moulding machine",
        "kraussmaffei", "milacron", "van dorn", "sumitomo", "demag",
        "euromap", "robot interface", "mc6", "mosaic", "pathfinder",
        "short shot", "flash", "sink mark", "weld line", "burn mark",
        "clamp tonnage", "shot size", "fill time", "pack pressure",
        "decoupled", "scientific molding", "cavity pressure",
        "toggle clamp", "hydraulic", "all-electric", "servo hydraulic"
    ]

    return any(kw in query_lower for kw in imm_keywords)


def detect_manufacturer(query: str) -> Optional[str]:
    """
    Detect which manufacturer is mentioned in a query.

    Args:
        query: User query string

    Returns:
        Manufacturer name or None
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in ["kraussmaffei", "km ", "mc6", "mc5"]):
        return "kraussmaffei"

    if any(kw in query_lower for kw in ["milacron", "cincinnati", "mosaic", "acramatic", "vista toggle"]):
        return "milacron"

    if any(kw in query_lower for kw in ["van dorn", "vandorn", "demag", "sumitomo", "pathfinder"]):
        return "vandorn_sumitomo"

    if any(kw in query_lower for kw in ["fanuc", "roboshot"]):
        return "fanuc"

    return None


def extract_defect_types(text: str) -> List[str]:
    """
    Extract all defect types mentioned in text.

    Args:
        text: Text to search

    Returns:
        List of defect type names found
    """
    defects_found = []
    text_lower = text.lower()

    for defect_name, patterns in DEFECT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                if defect_name not in defects_found:
                    defects_found.append(defect_name)
                break

    return defects_found


def extract_euromap_signals(text: str) -> Dict[str, List[str]]:
    """
    Extract Euromap signal mentions from text.

    Args:
        text: Text to search

    Returns:
        Dict mapping protocol to list of signals found
    """
    signals = {"euromap_67": [], "euromap_73": []}

    for protocol, patterns in EUROMAP_SIGNAL_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and match not in signals[protocol]:
                    signals[protocol].append(match)

    return signals


def get_searcher_engine_config() -> Dict[str, str]:
    """
    Get SearXNG engine configuration for IMM queries.

    Returns:
        Dict mapping query type to engine string
    """
    return {
        "imm_general": "brave,bing,duckduckgo,startpage",
        "imm_technical": "stackoverflow,github,brave,bing",
        "imm_forums": "reddit,brave,bing",
        "imm_academic": "arxiv,semantic_scholar,google_scholar",
        "imm_news": "bing_news,google_news",
    }


# ============================================
# SINGLETON SCHEMA INSTANCE
# ============================================

IMM_SCHEMA = create_imm_domain_schema()


# ============================================
# URL SOURCE REGISTRY
# ============================================

IMM_URL_SOURCES: Dict[str, Dict[str, any]] = {
    # =========================================================================
    # EUROMAP PROTOCOL SPECIFICATIONS (Critical - Primary Standards)
    # =========================================================================
    "euromap_standards": {
        "urls": [
            # Official Euromap Standards (PDF Downloads)
            "https://www.euromap.org/media/recommendations/67/2015/EU%2067_Ver_1.11_May2015.pdf",
            "https://euromap.org/media/recommendations/67_1/2015/EU%2067.1_v1.6_May2015.pdf",
            "https://www.euromap.org/files/EUROMAP_73_1.1.pdf",
            "https://www.euromap.org/technical-issues/technical-recommendations",
            "https://www.plastech.pl/pub/downloads/Euromap67.pdf",
        ],
        "priority": "critical",
        "content_type": "pdf",
        "requires_parsing": True,
    },

    "euromap_guides": {
        "urls": [
            # Euromap Implementation Guides
            "https://www.astor.com.pl/en/articles/euromap-67-communication-interface-for-industrial-robots-cooperating-with-injection-molding-machines/",
            "https://www.zacobria.com/universal-robots-knowledge-base-tech-support-forum-hints-tips-cb2-cb3/index.php/injection-moulding-machine-tending-with-euromap-67-interface/",
            "https://manualzz.com/doc/7094874/universal-robots-euromap67-interface-manual",
        ],
        "priority": "high",
        "content_type": "html",
    },

    # =========================================================================
    # KRAUSSMAFFEI TECHNICAL RESOURCES
    # =========================================================================
    "kraussmaffei_official": {
        "urls": [
            "https://www.kraussmaffei.com/en/download-area",
            "https://trainingacademy.kraussmaffei.com/en/trainings/b1-mc6/",
            "https://press.kraussmaffei.com/Downloadportal/public-brochures/",
            "https://www.kraussmaffei.com/media/download/cms/media/imm/km-imm-br-image-en-view.pdf",
            "https://www.kraussmaffei.com/products/IMM/PX/IMM_BR_PX%20Series_en.pdf",
            "https://www.kraussmaffeichina.com/media/download/cms/media/imm/kraussmaffei/downloads/km-imm-br-portfolio-en.pdf",
        ],
        "priority": "high",
        "content_type": "mixed",
    },

    "kraussmaffei_manuals": {
        "urls": [
            # KraussMaffei Manuals (Third-Party Hosts)
            "https://pdfcoffee.com/mx1600-manual-4-pdf-free.html",
            "https://f.machineryhost.com/47fd3c87f42f55d4b233417d49c34783/ed16b10dfbeb85e5cd5c10d3760e8cca/DOKU%20-%20MC5%20-%20EN%20(VOLL)%20CX%204.X.pdf",
            "https://kubousek.cz/en/skoleni_a_workshopy/operation-and-maintenance-of-kraussmaffei-injection-molding-machines/",
        ],
        "priority": "high",
        "content_type": "pdf",
    },

    # =========================================================================
    # CINCINNATI MILACRON TECHNICAL RESOURCES
    # =========================================================================
    "milacron_official": {
        "urls": [
            "https://www.milacron.com/product/mosaic-control/",
            "https://www.milacron.com/product/m-series/",
            "https://www.milacron.com/services/courtesy/",
            "https://www.milacron.com/services/training/",
        ],
        "priority": "high",
        "content_type": "html",
    },

    "milacron_manuals": {
        "urls": [
            # Cincinnati Milacron Manuals (Purchase/Reference)
            "https://www.mcspt.com/shop/Cincinnati-Milacron-Manuals.html",
            "https://www.mcspt.com/shop/Cincinnati-Milacron-Vista-Toggle-Injection-Molding-Machine-User-Manual.html",
            "https://industrialmanuals.com/Cincinnati%20Milacron-m-371.php",
            "https://www.hawthornecat.com/wp-content/uploads/2022/12/Cincinnati-Milacron-Operators-Instruction-Book.pdf",
            "https://www.scribd.com/doc/51854419/MILACRON-OPERATING-MANUAL-Rexonavn-Com",
            "https://www.scribd.com/document/735189525/Cincinnati-Milacron-Arrow-ERO-500-750-1000-VMC-Service-Manual",
        ],
        "priority": "high",
        "content_type": "mixed",
    },

    # =========================================================================
    # VAN DORN / SUMITOMO DEMAG TECHNICAL RESOURCES
    # =========================================================================
    "vandorn_official": {
        "urls": [
            "https://sumitomo-shi-demag.us/vddlegacyspecs/",
            "https://sumitomo-shi-demag.us/wp-content/uploads/2019/05/HT_rr_Brochure.pdf",
            "https://sumitomo-shi-demag.us/wp-content/uploads/2019/05/Tie_Bar_Stretch.pdf",
            "https://sumitomo-shi-demag.us/",
        ],
        "priority": "high",
        "content_type": "mixed",
    },

    "vandorn_manuals": {
        "urls": [
            # Van Dorn Manuals & Parts
            "https://industrialmanuals.com/van-dorn-m-647.php",
            "https://industrialmanuals.com/dorn-h270-h300a-h400-h300-injection-molding-plus-vickers-manual-year-1963-p-2248.php",
            "https://industrialmanuals.com/dorn-injection-molding-amchine-operations-parts-manual-1997-p-2249.php",
            "https://controlrepair.com/product/manufacturer/van-dorn-demag-2540",
            "https://telarcorp.com/van-dorn-injection-molding-parts/",
        ],
        "priority": "high",
        "content_type": "html",
    },

    # =========================================================================
    # FANUC ROBOT-IMM INTEGRATION
    # =========================================================================
    "fanuc_roboshot": {
        "urls": [
            # FANUC Official ROBOSHOT
            "https://www.fanuc.eu/eu-en/product/roboshot/roboshot-a-s100ib",
            "https://www.fanuc.eu/eu-en/product/roboshot/roboshot-a-s150ib",
            "https://www.fanuc.eu/eu-en/product/roboshot/roboshot-a-s220ib",
            "https://www.fanuc.eu/eu-en/product/roboshot/roboshot-a-s250ib",
        ],
        "priority": "high",
        "content_type": "html",
    },

    "robot_forum_fanuc": {
        "urls": [
            # Robot-Forum.com (FANUC Integration Discussions)
            "https://www.robot-forum.com/robotforum/thread/48854-euromap-67-wiring-for-fanuc/",
            "https://www.robot-forum.com/robotforum/thread/32837-fanuc-and-imm-euromap-67-and-73/",
            "https://www.robot-forum.com/robotforum/thread/33303-choosing-correct-options-for-dcs/",
            "https://forum.diy-robotics.com/hc/en-us/community/posts/360055254571-Euromap-73-on-Fanuc-DIY-Cell",
        ],
        "priority": "high",
        "content_type": "forum",
    },

    "robot_integration_guides": {
        "urls": [
            # Integration Guides
            "https://www.machinebuilding.net/safe-integration-of-tending-robots-with-moulding-machines",
            "https://library.e.abb.com/public/f5f6b3e959e50195c1257187002bde78/Datasheet%20Euromap%20PRINT.pdf",
        ],
        "priority": "high",
        "content_type": "mixed",
    },

    # =========================================================================
    # INJECTION MOLDING TROUBLESHOOTING FORUMS
    # =========================================================================
    "im_online_forum": {
        "urls": [
            # InjectionMoldingOnline.com (Primary Forum)
            "http://www.injectionmoldingonline.com/forum/archive/index.php?f-2-p-5.html",
            "http://www.injectionmoldingonline.com/forum/showthread.php?t=1180",
            "http://www.injectionmoldingonline.com/forum/archive/index.php?t-1019.html",
        ],
        "priority": "high",
        "content_type": "forum",
    },

    "plctalk_forums": {
        "urls": [
            # PLCTalk.net (PLC/Controls Focus)
            "https://www.plctalk.net/threads/injection-molding-machine-programer.33675/",
            "https://support.industry.siemens.com/forum/WW/en/posts/need-help-with-van-dorn-pathfinder-problem/74845",
        ],
        "priority": "high",
        "content_type": "forum",
    },

    # =========================================================================
    # TRADE PUBLICATIONS & TECHNICAL ARTICLES
    # =========================================================================
    "ptonline_troubleshooting": {
        "urls": [
            # Plastics Technology (ptonline.com) - Troubleshooting Articles
            "https://www.ptonline.com/articles",
            "https://www.ptonline.com/columns/troubleshooting-injection-molding-seven-steps-toward-scientific-troubleshooting",
            "https://www.ptonline.com/articles/injection-molding-how-to-get-rid-of-bubbles",
            "https://www.ptonline.com/articles/troubleshooting-bridging",
            "https://www.ptonline.com/articles/how-to-fix-outgassing-problems-in-injection-molding",
            "https://www.ptonline.com/articles/injection-molding-processing-nylonand-other-problems",
            "https://www.ptonline.com/articles/apply-the-power-of-a-troubleshooting-checklist-to-your-process",
            "https://www.ptonline.com/articles/the-butterfly-effect-in-injection-moldinga-connected-process",
            "https://www.ptonline.com/articles/using-data-to-pinpoint-cosmetic-defect-causes-in-injection-molded-parts",
        ],
        "priority": "high",
        "content_type": "article",
    },

    "plastics_publications": {
        "urls": [
            # Other Trade Resources
            "https://www.plasticstoday.com/injection-molding/the-basics-of-plastic-injection-molding-troubleshooting",
            "https://plastics-rubber.basf.com/global/en/performance_polymers/services/product_support_troubleshooting/injection_moulding_troubleshooter",
            "https://www.nexeoplastics.com/blog/technical-solutions/nexeo-plastics-offers-troubleshooting-tips-for-plastic-injection-molding/",
            "https://s3.amazonaws.com/entecpolymers.com/v3/uploads/content/Troubleshooting-Guide-For-Injection-Molding.pdf",
            "https://elastron.com/blog/11-injection-molding-defects-and-troubleshooting/",
            "https://waykenrm.com/blogs/plastic-injection-molding-problems-and-solutions/",
            "https://guanxin-machinery.com/injection-molding-machines-error-codes-and-troubleshooting-solutions/",
        ],
        "priority": "high",
        "content_type": "mixed",
    },

    # =========================================================================
    # RJG SCIENTIFIC MOLDING RESOURCES
    # =========================================================================
    "rjg_training": {
        "urls": [
            "https://rjginc.com/solutions/training/",
            "https://rjginc.com/training/registration/fundamentals-of-systematic-injection-molding/",
            "https://rjginc.com/training/registration/essentials-of-injection-molding/",
            "https://rjginc.learnupon.com/store",
            "https://rjginc.com/bridging-the-skills-gap-in-injection-molding-with-rjgs-max-the-future-of-process-optimization-using-ai/",
            "https://www.traininteractive.com/training/online/injection/decoupled/molding/",
            "https://vitalplastics.com/resources/rjg-decoupled-molding/",
        ],
        "priority": "medium",
        "content_type": "html",
    },

    # =========================================================================
    # MOLD COMPONENT & MAINTENANCE RESOURCES
    # =========================================================================
    "mold_components": {
        "urls": [
            # DME/HASCO (Component Standards)
            "https://www.dme.net/resources/",
            "https://www.hasco.com/",
            # Maintenance Documentation
            "https://www.4spe.org/i4a/pages/index.cfm?pageID=9488",
            "https://aim.institute/courses/pte-program/",
        ],
        "priority": "medium",
        "content_type": "html",
    },

    # =========================================================================
    # CONTROL SYSTEM REPAIR RESOURCES
    # =========================================================================
    "control_repair": {
        "urls": [
            "https://controlrepair.com/product/series/camac-64",
            "https://controlrepair.com/product/manufacturer/van-dorn-demag-2540",
            "https://cincinnatirpt.com/",
        ],
        "priority": "medium",
        "content_type": "html",
    },
}


def get_priority_urls() -> List[str]:
    """Get list of high-priority URLs for initial corpus building."""
    priority_urls = []
    for source in IMM_URL_SOURCES.values():
        if source.get("priority") in ["critical", "high"]:
            priority_urls.extend(source["urls"])
    return priority_urls
