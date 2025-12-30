"""
RJG Scientific Molding Domain Schema for Entity Extraction

Defines entity types, patterns, and relationships specific to RJG Inc.
scientific injection molding methodology and equipment. Used by:
- DocumentGraphService for query enhancement
- DomainCorpus for entity extraction
- UniversalOrchestrator for query routing

Patterns are designed to match content from RJG training materials:
- eDART system terminology (cavity pressure monitoring)
- Scientific molding phases (Fill-Pack-Hold-Cool)
- Process variables (melt temp, mold temp, velocity profiles)
- Defect types and troubleshooting (short shot, flash, sink marks)
- Decoupled molding techniques (DIII, DII)

Usage:
    from agentic.schemas.rjg_schema import (
        RJG_SCHEMA,
        RJG_PROCESS_PATTERNS,
        create_rjg_domain_schema,
        is_scientific_molding_query
    )

    # Check if query is scientific molding related
    if is_scientific_molding_query(query):
        # Route to technical documentation
        ...

    # Create domain schema for corpus
    schema = create_rjg_domain_schema()
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple
from enum import Enum


# ============================================
# ENTITY TYPE DEFINITIONS
# ============================================

class RJGEntityType(str, Enum):
    """Types of entities in RJG scientific molding documentation"""
    PRODUCT = "product"                     # eDART, CoPilot, MAX, HUB
    PROCESS_PHASE = "process_phase"         # Fill, Pack, Hold, Cool
    PROCESS_VARIABLE = "process_variable"   # Melt temp, cushion, velocity
    DEFECT = "defect"                       # Short shot, flash, sink marks
    SENSOR = "sensor"                       # Cavity pressure, temperature
    MACHINE_COMPONENT = "machine_component" # Barrel, screw, nozzle
    MEASUREMENT = "measurement"             # 500 PSI, 450°F, 2.5 sec
    TECHNIQUE = "technique"                 # DIII, Decoupled Molding
    QUALITY_ATTRIBUTE = "quality_attribute" # Dimensional stability, weight
    EUROMAP_SIGNAL = "euromap_signal"       # EM67, EM77 signals
    MATERIAL_PROPERTY = "material_property" # Viscosity, melt flow index


class RJGRelationType(str, Enum):
    """Types of relationships between scientific molding entities"""
    CAUSES = "causes"                       # Process var → Defect
    PREVENTS = "prevents"                   # Technique → Defect
    MEASURES = "measures"                   # Sensor → Process var
    AFFECTS = "affects"                     # Process var → Quality
    PART_OF = "part_of"                     # Component → System
    REQUIRES = "requires"                   # Technique → Equipment
    FOLLOWS = "follows"                     # Phase → Phase
    INDICATES = "indicates"                 # Defect → Cause
    OPTIMIZES = "optimizes"                 # Technique → Process


# ============================================
# REGEX PATTERNS
# ============================================

# RJG Product/System patterns
RJG_PRODUCT_PATTERNS: List[str] = [
    r"\beDART\b",                            # eDART system
    r"\bCoPilot\b",                          # CoPilot dashboard
    r"\bHUB\b",                              # RJG HUB
    r"\bMAX\b",                              # RJG MAX system
    r"\bThe\s*Hub\b",                        # The Hub
    r"\bLynx\s*(Temperature)?\b",            # Lynx system
    r"\bRJG\b",                              # RJG brand
    r"\bCavity\s*Sense\b",                   # CavitySense
]

# Scientific molding phases (Fill-Pack-Hold-Cool)
RJG_PROCESS_PHASE_PATTERNS: List[str] = [
    r"\bfill\s*(phase|time|rate|speed)\b",
    r"\bpack\s*(phase|time|pressure|hold)\b",
    r"\bhold\s*(phase|time|pressure)\b",
    r"\bcool(ing)?\s*(time|phase)\b",
    r"\b(1st|2nd|first|second)\s*stage\b",
    r"\btransfer\s*(point|position)\b",
    r"\bswitch[-\s]?over\b",
    r"\bVP\s*(transfer|switchover)\b",
    r"\bplasticat(e|ion|ing)\b",
    r"\binjection\s*forward\b",
]

# Process variable patterns
RJG_PROCESS_VARIABLE_PATTERNS: List[Tuple[str, str]] = [
    (r"(melt|barrel)\s*temp(erature)?", "melt_temperature"),
    (r"mold\s*temp(erature)?", "mold_temperature"),
    (r"nozzle\s*temp(erature)?", "nozzle_temperature"),
    (r"(injection|fill)\s*(speed|velocity|rate)", "injection_velocity"),
    (r"screw\s*(speed|RPM|rotation)", "screw_speed"),
    (r"back\s*pressure", "back_pressure"),
    (r"pack\s*pressure", "pack_pressure"),
    (r"hold\s*pressure", "hold_pressure"),
    (r"injection\s*pressure", "injection_pressure"),
    (r"clamp\s*(force|tonnage|pressure)", "clamp_force"),
    (r"cushion\s*(size|position)?", "cushion"),
    (r"shot\s*size", "shot_size"),
    (r"cycle\s*time", "cycle_time"),
    (r"cool(ing)?\s*time", "cooling_time"),
    (r"gate\s*(freeze|seal)\s*time", "gate_seal_time"),
    (r"peak\s*(cavity\s*)?pressure", "peak_pressure"),
    (r"end\s*of\s*fill\s*pressure", "end_fill_pressure"),
    (r"(process|viscosity)\s*(index|curve)", "viscosity_index"),
]

# Defect patterns
RJG_DEFECT_PATTERNS: List[Tuple[str, str]] = [
    (r"short\s*shot", "short_shot"),
    (r"\bflash(ing)?\b", "flash"),
    (r"sink\s*mark", "sink_mark"),
    (r"weld\s*line|knit\s*line", "weld_line"),
    (r"burn\s*mark|diesel(ing)?", "burn_mark"),
    (r"jett(ing)?", "jetting"),
    (r"warp(ing|age)?", "warpage"),
    (r"void", "void"),
    (r"bubble", "bubble"),
    (r"splay|silver\s*streak", "splay"),
    (r"delamination", "delamination"),
    (r"brittleness", "brittleness"),
    (r"gloss\s*(variation|difference)", "gloss_variation"),
    (r"dimension(al)?\s*(variation|instability)", "dimensional_variation"),
    (r"part\s*sticking", "part_sticking"),
    (r"gate\s*blush|gate\s*vestige", "gate_blush"),
    (r"flow\s*line|flow\s*mark", "flow_line"),
    (r"record\s*groove", "record_groove"),
    (r"orange\s*peel", "orange_peel"),
    (r"contamination", "contamination"),
    (r"moisture\s*(defect|issue)", "moisture_defect"),
]

# Sensor patterns
RJG_SENSOR_PATTERNS: List[Tuple[str, str]] = [
    (r"cavity\s*pressure\s*(sensor|transducer)?", "cavity_pressure_sensor"),
    (r"piezoelectric\s*(sensor|transducer)?", "piezo_sensor"),
    (r"strain\s*gauge|strain[-\s]?gage", "strain_gauge"),
    (r"melt\s*(pressure\s*)?(sensor|transducer)", "melt_sensor"),
    (r"(indirect|direct)\s*sensor", "sensor_type"),
    (r"post\s*(mount(ed)?|type)", "post_sensor"),
    (r"flush\s*(mount(ed)?|type)", "flush_sensor"),
    (r"ejector\s*pin\s*sensor", "ejector_sensor"),
    (r"temp(erature)?\s*sensor|thermocouple|RTD", "temp_sensor"),
]

# Machine component patterns
RJG_COMPONENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\bbarrel\b", "barrel"),
    (r"\bscrew\b", "screw"),
    (r"\bnozzle\b", "nozzle"),
    (r"hot\s*runner", "hot_runner"),
    (r"cold\s*runner", "cold_runner"),
    (r"\bmold\s*base\b", "mold_base"),
    (r"\b(mold\s*)?cavity\b", "cavity"),
    (r"\bcore\b", "core"),
    (r"\bgate\b", "gate"),
    (r"\brunner\b", "runner"),
    (r"\bsprue\b", "sprue"),
    (r"\bejector\s*(pin|plate)\b", "ejector"),
    (r"clamp\s*(unit|platen)", "clamp_unit"),
    (r"injection\s*unit", "injection_unit"),
    (r"tie\s*bar", "tie_bar"),
    (r"check\s*ring|non[-\s]?return\s*valve", "check_ring"),
    (r"(heater\s*)?band", "heater_band"),
    (r"manifold", "manifold"),
    (r"drop|tip", "drop_tip"),
]

# Measurement patterns (scientific molding units)
RJG_MEASUREMENT_PATTERNS: List[str] = [
    r"\d+\.?\d*\s*(PSI|psi|bar|MPa)",       # Pressure
    r"\d+\.?\d*\s*(°?[FC]|deg\s*[FC])",     # Temperature
    r"\d+\.?\d*\s*(in/s|mm/s|cm/s)",        # Velocity
    r"\d+\.?\d*\s*(cc|cm3|in3|oz)",         # Volume
    r"\d+\.?\d*\s*(tons?|kN)",              # Force
    r"\d+\.?\d*\s*(RPM|rpm)",               # Rotation
    r"\d+\.?\d*\s*(sec|s|ms)\b",            # Time
    r"\d+\.?\d*\s*(in|mm|thou)\b",          # Position
    r"\d+\.?\d*\s*%",                       # Percentage
    r"\d+\.?\d*\s*(g/10\s*min|MFI|MFR)",    # Melt flow
]

# Scientific molding techniques
RJG_TECHNIQUE_PATTERNS: List[str] = [
    r"decoupled\s*(molding|process)?",
    r"DIII|D3|Decoupled\s*III",             # Decoupled III
    r"DII|D2|Decoupled\s*II",               # Decoupled II
    r"scientific\s*molding",
    r"(6|six)\s*step\s*(study|process)",    # 6-step scientific molding
    r"viscosity\s*curve",
    r"pressure\s*(vs\.|versus)\s*time",
    r"cavity\s*balance",
    r"pressure\s*drop\s*study",
    r"gate\s*seal\s*study",
    r"cooling\s*study",
    r"DOE|design\s*of\s*experiments",
    r"process\s*window",
    r"capability\s*study",
    r"CPK|Cpk|process\s*capability",
]

# Quality attribute patterns
RJG_QUALITY_PATTERNS: List[Tuple[str, str]] = [
    (r"part\s*weight", "part_weight"),
    (r"dimension(al)?\s*(stability|accuracy)", "dimensional_stability"),
    (r"surface\s*(finish|quality)", "surface_finish"),
    (r"mechanical\s*(strength|properties)", "mechanical_strength"),
    (r"cycle\s*(consistency|repeatability)", "cycle_consistency"),
    (r"reject\s*rate|scrap\s*rate", "reject_rate"),
    (r"first\s*pass\s*yield|FPY", "first_pass_yield"),
    (r"OEE|overall\s*equipment\s*effectiveness", "oee"),
]

# Euromap signal patterns
RJG_EUROMAP_PATTERNS: List[str] = [
    r"Euromap\s*(67|73|77|82)",
    r"EM\s*(67|73|77|82)",
    r"EUROMAP\s*(67|73|77|82)",
    r"SPI\s*protocol",
    r"machine\s*interface",
]

# Material property patterns
RJG_MATERIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"viscosity", "viscosity"),
    (r"MFI|melt\s*flow\s*index", "melt_flow_index"),
    (r"MFR|melt\s*flow\s*rate", "melt_flow_rate"),
    (r"shear\s*(rate|sensitivity)", "shear_rate"),
    (r"(glass\s*)?transition\s*temp", "glass_transition"),
    (r"crystallinity", "crystallinity"),
    (r"moisture\s*content", "moisture_content"),
    (r"filler\s*(content|loading)", "filler_content"),
]


# ============================================
# COMPILED PATTERNS (for performance)
# ============================================

COMPILED_PRODUCT_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in RJG_PRODUCT_PATTERNS
]

COMPILED_TECHNIQUE_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in RJG_TECHNIQUE_PATTERNS
]

COMPILED_DEFECT_PATTERNS: List[Pattern] = [
    re.compile(p, re.IGNORECASE) for p, _ in RJG_DEFECT_PATTERNS
]


# ============================================
# DOMAIN SCHEMA DATACLASS
# ============================================

@dataclass
class RJGEntityDef:
    """Definition of an RJG entity type for extraction"""
    name: str
    description: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    context_window: int = 50  # Characters of context to capture


@dataclass
class RJGDomainSchema:
    """Complete RJG domain schema for entity extraction"""
    name: str = "rjg_scientific_molding"
    description: str = "RJG Inc. scientific injection molding methodology"
    entities: List[RJGEntityDef] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)


# ============================================
# SCHEMA FACTORY
# ============================================

def create_rjg_domain_schema() -> RJGDomainSchema:
    """
    Create a complete RJG domain schema for entity extraction.

    Returns:
        RJGDomainSchema with all entity definitions and relationships
    """
    entities = [
        RJGEntityDef(
            name="product",
            description="RJG products and systems",
            patterns=RJG_PRODUCT_PATTERNS,
            examples=["eDART", "CoPilot", "HUB", "MAX", "Lynx"]
        ),
        RJGEntityDef(
            name="process_phase",
            description="Injection molding process phases",
            patterns=RJG_PROCESS_PHASE_PATTERNS,
            keywords=[
                "fill phase", "pack phase", "hold phase", "cooling",
                "transfer", "switchover", "plastication"
            ],
            examples=["fill time", "pack pressure", "cooling time", "VP transfer"]
        ),
        RJGEntityDef(
            name="process_variable",
            description="Measurable process parameters",
            patterns=[p for p, _ in RJG_PROCESS_VARIABLE_PATTERNS],
            keywords=[
                "melt temperature", "mold temperature", "injection velocity",
                "back pressure", "cushion", "cycle time", "clamp force"
            ],
            examples=["melt temp 450F", "injection speed 2 in/s", "cushion 0.25 in"]
        ),
        RJGEntityDef(
            name="defect",
            description="Injection molding defects and quality issues",
            patterns=[p for p, _ in RJG_DEFECT_PATTERNS],
            keywords=[
                "short shot", "flash", "sink mark", "weld line",
                "burn mark", "warpage", "void", "splay", "jetting"
            ],
            examples=["short shot", "sink marks near ribs", "flash at parting line"]
        ),
        RJGEntityDef(
            name="sensor",
            description="Cavity pressure and process monitoring sensors",
            patterns=[p for p, _ in RJG_SENSOR_PATTERNS],
            keywords=[
                "cavity pressure sensor", "piezoelectric", "strain gauge",
                "post mount", "flush mount", "thermocouple"
            ],
            examples=["cavity pressure sensor", "piezoelectric transducer", "post-mounted sensor"]
        ),
        RJGEntityDef(
            name="machine_component",
            description="Injection molding machine and mold components",
            patterns=[p for p, _ in RJG_COMPONENT_PATTERNS],
            keywords=[
                "barrel", "screw", "nozzle", "hot runner", "cavity",
                "core", "gate", "runner", "sprue", "ejector"
            ],
            examples=["barrel heater", "check ring", "hot runner manifold"]
        ),
        RJGEntityDef(
            name="measurement",
            description="Numeric measurements with units",
            patterns=RJG_MEASUREMENT_PATTERNS,
            examples=["500 PSI", "450°F", "2.5 in/s", "30 tons", "1500 RPM"]
        ),
        RJGEntityDef(
            name="technique",
            description="Scientific molding techniques and methods",
            patterns=RJG_TECHNIQUE_PATTERNS,
            keywords=[
                "decoupled molding", "DIII", "scientific molding",
                "viscosity curve", "gate seal study", "DOE"
            ],
            examples=["Decoupled III process", "6-step scientific molding", "cavity balance study"]
        ),
        RJGEntityDef(
            name="quality_attribute",
            description="Part quality characteristics",
            patterns=[p for p, _ in RJG_QUALITY_PATTERNS],
            keywords=[
                "part weight", "dimensional stability", "surface finish",
                "reject rate", "Cpk", "OEE"
            ],
            examples=["part weight variation", "dimensional stability", "Cpk > 1.33"]
        ),
        RJGEntityDef(
            name="euromap_signal",
            description="Euromap machine interface signals",
            patterns=RJG_EUROMAP_PATTERNS,
            keywords=["Euromap 67", "Euromap 77", "SPI protocol"],
            examples=["Euromap 67 interface", "EM77 signals"]
        ),
        RJGEntityDef(
            name="material_property",
            description="Plastic material properties",
            patterns=[p for p, _ in RJG_MATERIAL_PATTERNS],
            keywords=[
                "viscosity", "MFI", "melt flow", "shear rate",
                "glass transition", "moisture content"
            ],
            examples=["MFI 12 g/10min", "viscosity curve", "Tg 105°C"]
        ),
    ]

    relationships = [
        # Defect-cause relationships
        ("process_variable", "causes", "defect"),
        ("technique", "prevents", "defect"),
        ("defect", "indicates", "process_variable"),

        # Sensor relationships
        ("sensor", "measures", "process_variable"),
        ("product", "uses", "sensor"),

        # Process flow
        ("process_phase", "follows", "process_phase"),
        ("technique", "optimizes", "process_phase"),

        # Component relationships
        ("sensor", "part_of", "machine_component"),
        ("machine_component", "affects", "process_variable"),

        # Quality relationships
        ("process_variable", "affects", "quality_attribute"),
        ("technique", "improves", "quality_attribute"),

        # Material relationships
        ("material_property", "affects", "process_variable"),
        ("material_property", "causes", "defect"),

        # Euromap relationships
        ("product", "supports", "euromap_signal"),
        ("machine_component", "uses", "euromap_signal"),
    ]

    return RJGDomainSchema(
        name="rjg_scientific_molding",
        description="RJG Inc. scientific injection molding methodology and equipment",
        entities=entities,
        relationships=relationships
    )


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def is_scientific_molding_query(query: str) -> bool:
    """
    Quick check if a query is scientific molding related.

    Args:
        query: User query string

    Returns:
        True if query contains scientific molding patterns
    """
    query_lower = query.lower()

    # Check RJG products first (most definitive)
    for pattern in COMPILED_PRODUCT_PATTERNS:
        if pattern.search(query):
            return True

    # Check techniques
    for pattern in COMPILED_TECHNIQUE_PATTERNS:
        if pattern.search(query):
            return True

    # Check defect patterns
    for pattern in COMPILED_DEFECT_PATTERNS:
        if pattern.search(query):
            return True

    # Check keywords
    scientific_molding_keywords = [
        "cavity pressure", "scientific molding", "decoupled",
        "fill pack hold", "viscosity curve", "gate seal",
        "injection molding", "eDART", "copilot", "rjg",
        "short shot", "flash", "sink mark", "process window"
    ]

    return any(kw in query_lower for kw in scientific_molding_keywords)


def extract_defects(text: str) -> List[str]:
    """
    Extract all injection molding defects from text.

    Args:
        text: Text to search

    Returns:
        List of unique defects found
    """
    defects = set()
    for pattern in COMPILED_DEFECT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            defects.add(matches[0] if isinstance(matches[0], str) else matches[0][0])
    return sorted(defects)


def get_defect_category(defect: str) -> str:
    """
    Get the category for a molding defect.

    Args:
        defect: Defect name

    Returns:
        Category name
    """
    defect_lower = defect.lower()

    categories = {
        # Fill-related
        "short shot": "Fill Defects",
        "jetting": "Fill Defects",
        "flow line": "Fill Defects",
        "weld line": "Fill Defects",

        # Pack-related
        "sink mark": "Pack/Hold Defects",
        "void": "Pack/Hold Defects",
        "flash": "Pack/Hold Defects",

        # Cooling-related
        "warpage": "Cooling Defects",
        "dimensional variation": "Cooling Defects",

        # Material-related
        "splay": "Material Defects",
        "burn mark": "Material Defects",
        "moisture defect": "Material Defects",
        "delamination": "Material Defects",

        # Surface-related
        "gloss variation": "Surface Defects",
        "orange peel": "Surface Defects",
        "gate blush": "Surface Defects",
    }

    for defect_key, category in categories.items():
        if defect_key in defect_lower:
            return category

    return "Unknown Category"


def extract_process_variables(text: str) -> List[Tuple[str, str]]:
    """
    Extract process variables from text.

    Args:
        text: Text to search

    Returns:
        List of (match, variable_type) tuples
    """
    variables = []
    for pattern, var_type in RJG_PROCESS_VARIABLE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                match_str = match if isinstance(match, str) else match[0]
                variables.append((match_str, var_type))
    return variables


# ============================================
# ACRONYM DEFINITIONS (for expansion)
# ============================================

RJG_ACRONYMS: Dict[str, str] = {
    # RJG Products
    "eDART": "Electronic Data Acquisition in Real Time",
    "CoPilot": "Cavity Pressure Monitoring Dashboard",
    "HUB": "RJG HUB Data Collection System",
    "MAX": "RJG MAX Monitoring System",

    # Scientific Molding Terms
    "DIII": "Decoupled III Molding Process",
    "DII": "Decoupled II Molding Process",
    "VP": "Velocity to Pressure (transfer point)",
    "DOE": "Design of Experiments",
    "CPK": "Process Capability Index",
    "OEE": "Overall Equipment Effectiveness",
    "FPY": "First Pass Yield",
    "MFI": "Melt Flow Index",
    "MFR": "Melt Flow Rate",

    # Machine/Mold Terms
    "EOAT": "End of Arm Tooling",
    "RTD": "Resistance Temperature Detector",

    # Standards
    "SPI": "Society of the Plastics Industry",
    "EM67": "Euromap 67 Interface Standard",
    "EM77": "Euromap 77 Interface Standard",
}


# ============================================
# SINGLETON SCHEMA INSTANCE
# ============================================

RJG_SCHEMA = create_rjg_domain_schema()
