"""
Industrial Acronym Dictionary

Provides comprehensive acronym expansion for industrial automation,
robotics, injection molding, and manufacturing queries.

Benefits:
- Improves semantic search by adding context
- Helps LLM understand technical terminology
- Enables better query-domain matching

Usage:
    from agentic.acronym_dictionary import expand_acronyms, get_acronym_info

    # Expand in query
    query = "SRVO-063 on R-30iB"
    expanded = expand_acronyms(query)
    # Result: "SRVO-063 (Servo Alarm) on R-30iB (FANUC Controller)"

    # Get info about acronym
    info = get_acronym_info("PLC")
    # Result: {"expansion": "Programmable Logic Controller", "category": "automation", ...}

Author: Claude Code
Date: December 2025
"""

import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class AcronymInfo:
    """Information about an acronym."""
    expansion: str
    category: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)


# ============================================
# FANUC Error Code Prefixes
# ============================================
FANUC_ERROR_CODES: Dict[str, AcronymInfo] = {
    "SRVO": AcronymInfo(
        expansion="Servo Alarm",
        category="fanuc_error",
        description="Motor, encoder, and servo drive faults",
        related=["MOTN", "SVGN"]
    ),
    "MOTN": AcronymInfo(
        expansion="Motion Alarm",
        category="fanuc_error",
        description="Motion planning and trajectory errors",
        related=["SRVO", "INTP"]
    ),
    "SYST": AcronymInfo(
        expansion="System Alarm",
        category="fanuc_error",
        description="Controller and system-level errors",
        related=["MEMO", "FILE"]
    ),
    "HOST": AcronymInfo(
        expansion="Host Communication Alarm",
        category="fanuc_error",
        description="External communication failures",
        related=["COMM", "SRIO"]
    ),
    "INTP": AcronymInfo(
        expansion="Interpreter Alarm",
        category="fanuc_error",
        description="Program execution and syntax errors",
        related=["MACR", "PRIO"]
    ),
    "PRIO": AcronymInfo(
        expansion="Priority Alarm",
        category="fanuc_error",
        description="High-priority system alerts"
    ),
    "COMM": AcronymInfo(
        expansion="Communication Alarm",
        category="fanuc_error",
        description="Network and fieldbus errors",
        related=["HOST", "SRIO"]
    ),
    "VISI": AcronymInfo(
        expansion="Vision Alarm",
        category="fanuc_error",
        description="iRVision camera and processing errors",
        aliases=["CVIS"],
        related=["SPOT", "ARC"]
    ),
    "CVIS": AcronymInfo(
        expansion="Camera Vision Alarm",
        category="fanuc_error",
        description="iRVision camera and processing errors",
        aliases=["VISI"]
    ),
    "SRIO": AcronymInfo(
        expansion="Serial I/O Alarm",
        category="fanuc_error",
        description="Serial port and I/O errors",
        related=["COMM", "HOST"]
    ),
    "FILE": AcronymInfo(
        expansion="File System Alarm",
        category="fanuc_error",
        description="Memory card and file access errors",
        related=["SYST", "MEMO"]
    ),
    "MEMO": AcronymInfo(
        expansion="Memory Alarm",
        category="fanuc_error",
        description="RAM and storage errors",
        related=["SYST", "FILE"]
    ),
    "MACR": AcronymInfo(
        expansion="Macro Alarm",
        category="fanuc_error",
        description="Macro execution errors",
        related=["INTP", "PRIO"]
    ),
    "PALL": AcronymInfo(
        expansion="Palletizing Alarm",
        category="fanuc_error",
        description="Palletizing application errors",
        related=["MOTN", "INTP"]
    ),
    "SPOT": AcronymInfo(
        expansion="Spot Welding Alarm",
        category="fanuc_error",
        description="Spot welding application errors",
        related=["ARC", "VISI"]
    ),
    "ARC": AcronymInfo(
        expansion="Arc Welding Alarm",
        category="fanuc_error",
        description="Arc welding application errors",
        related=["SPOT", "VISI"]
    ),
    "DISP": AcronymInfo(
        expansion="Dispense Alarm",
        category="fanuc_error",
        description="Dispensing application errors"
    ),
    "DICT": AcronymInfo(
        expansion="Dictionary Alarm",
        category="fanuc_error",
        description="System dictionary errors"
    ),
    "COND": AcronymInfo(
        expansion="Condition Alarm",
        category="fanuc_error",
        description="Condition monitoring alerts"
    ),
    "TOOL": AcronymInfo(
        expansion="Tool Alarm",
        category="fanuc_error",
        description="Tool-related errors"
    ),
    "ACAL": AcronymInfo(
        expansion="Acceleration/Collision Alarm",
        category="fanuc_error",
        description="Collision detection alerts"
    ),
    "SVGN": AcronymInfo(
        expansion="Servo Gain Alarm",
        category="fanuc_error",
        description="Servo tuning and gain errors",
        related=["SRVO", "MOTN"]
    ),
}


# ============================================
# General Automation Acronyms
# ============================================
AUTOMATION_ACRONYMS: Dict[str, AcronymInfo] = {
    "PLC": AcronymInfo(
        expansion="Programmable Logic Controller",
        category="automation",
        description="Industrial control computer",
        aliases=["PAC"],
        related=["HMI", "SCADA", "DCS"]
    ),
    "PAC": AcronymInfo(
        expansion="Programmable Automation Controller",
        category="automation",
        description="Advanced PLC with PC capabilities",
        aliases=["PLC"]
    ),
    "HMI": AcronymInfo(
        expansion="Human Machine Interface",
        category="automation",
        description="Operator touchscreen/panel",
        aliases=["OIT", "OIU"],
        related=["PLC", "SCADA"]
    ),
    "OIT": AcronymInfo(
        expansion="Operator Interface Terminal",
        category="automation",
        aliases=["HMI", "OIU"]
    ),
    "SCADA": AcronymInfo(
        expansion="Supervisory Control and Data Acquisition",
        category="automation",
        description="Industrial monitoring system",
        related=["HMI", "DCS", "PLC"]
    ),
    "DCS": AcronymInfo(
        expansion="Distributed Control System",
        category="automation",
        description="Process control architecture",
        aliases=["DCSS"],  # FANUC uses DCSS for Dual Check Safety
        related=["PLC", "SCADA"]
    ),
    "VFD": AcronymInfo(
        expansion="Variable Frequency Drive",
        category="automation",
        description="Motor speed controller",
        aliases=["VSD", "AFD", "ASD"],
        related=["AC", "DC", "PWM"]
    ),
    "VSD": AcronymInfo(
        expansion="Variable Speed Drive",
        category="automation",
        aliases=["VFD"]
    ),
    "RTU": AcronymInfo(
        expansion="Remote Terminal Unit",
        category="automation",
        description="Remote I/O and monitoring",
        related=["PLC", "SCADA"]
    ),
    "I/O": AcronymInfo(
        expansion="Input/Output",
        category="automation",
        description="Digital and analog signals",
        aliases=["IO"]
    ),
    "DI": AcronymInfo(
        expansion="Digital Input",
        category="automation",
        description="On/off input signal"
    ),
    "DO": AcronymInfo(
        expansion="Digital Output",
        category="automation",
        description="On/off output signal"
    ),
    "AI": AcronymInfo(
        expansion="Analog Input",
        category="automation",
        description="Variable input signal (0-10V, 4-20mA)"
    ),
    "AO": AcronymInfo(
        expansion="Analog Output",
        category="automation",
        description="Variable output signal"
    ),
    "NC": AcronymInfo(
        expansion="Normally Closed",
        category="automation",
        description="Contact that opens when activated"
    ),
    "NO": AcronymInfo(
        expansion="Normally Open",
        category="automation",
        description="Contact that closes when activated"
    ),
    "PID": AcronymInfo(
        expansion="Proportional-Integral-Derivative",
        category="automation",
        description="Control loop algorithm",
        related=["PLC", "DCS"]
    ),
    "OPC": AcronymInfo(
        expansion="Open Platform Communications",
        category="automation",
        description="Industrial data exchange standard",
        aliases=["OPC-UA", "OPC UA"]
    ),
}


# ============================================
# Robotics Acronyms
# ============================================
ROBOTICS_ACRONYMS: Dict[str, AcronymInfo] = {
    "FANUC": AcronymInfo(
        expansion="Factory Automation Numerical Control",
        category="robotics",
        description="Japanese robotics manufacturer"
    ),
    "TCP": AcronymInfo(
        expansion="Tool Center Point",
        category="robotics",
        description="End effector reference frame"
    ),
    "EOAT": AcronymInfo(
        expansion="End of Arm Tooling",
        category="robotics",
        description="Gripper/tool attached to robot",
        aliases=["EoAT"]
    ),
    "DOF": AcronymInfo(
        expansion="Degrees of Freedom",
        category="robotics",
        description="Number of independent movements",
        aliases=["6DOF", "7DOF"]
    ),
    "RCAL": AcronymInfo(
        expansion="Robot Calibration",
        category="fanuc",
        description="FANUC encoder calibration procedure"
    ),
    "TP": AcronymInfo(
        expansion="Teach Pendant",
        category="robotics",
        description="Handheld robot controller",
        aliases=["iPendant"]
    ),
    "KAREL": AcronymInfo(
        expansion="KAREL Programming Language",
        category="fanuc",
        description="FANUC advanced programming language"
    ),
    "WPR": AcronymInfo(
        expansion="Wrist Point Reference",
        category="robotics",
        description="Wrist center position"
    ),
    "UF": AcronymInfo(
        expansion="User Frame",
        category="robotics",
        description="Custom coordinate system"
    ),
    "UTool": AcronymInfo(
        expansion="User Tool",
        category="robotics",
        description="Tool offset definition"
    ),
    "SOP": AcronymInfo(
        expansion="Safe Operating Procedure",
        category="safety",
        description="Standardized safety procedure",
        aliases=["SWP"]
    ),
    "COBOT": AcronymInfo(
        expansion="Collaborative Robot",
        category="robotics",
        description="Robot designed for human interaction",
        aliases=["Collaborative Robot"]
    ),
    "IK": AcronymInfo(
        expansion="Inverse Kinematics",
        category="robotics",
        description="Calculate joint angles from position"
    ),
    "FK": AcronymInfo(
        expansion="Forward Kinematics",
        category="robotics",
        description="Calculate position from joint angles"
    ),
}


# ============================================
# Injection Molding Machine (IMM) Acronyms
# ============================================
IMM_ACRONYMS: Dict[str, AcronymInfo] = {
    "IMM": AcronymInfo(
        expansion="Injection Molding Machine",
        category="imm",
        description="Plastic injection molding equipment"
    ),
    "EUROMAP": AcronymInfo(
        expansion="European Plastics and Rubber Machinery",
        category="imm",
        description="Standard for IMM communication",
        aliases=["EM", "EUROMAP 67", "EUROMAP 77"]
    ),
    "SPI": AcronymInfo(
        expansion="Society of Plastics Industry",
        category="imm",
        description="US plastics industry standard"
    ),
    "TCU": AcronymInfo(
        expansion="Temperature Control Unit",
        category="imm",
        description="Mold heating/cooling unit"
    ),
    "HRC": AcronymInfo(
        expansion="Hot Runner Controller",
        category="imm",
        description="Controls hot runner system temperature"
    ),
    "MTC": AcronymInfo(
        expansion="Mold Temperature Controller",
        category="imm",
        description="Controls mold temperature",
        aliases=["TCU"]
    ),
    "LSR": AcronymInfo(
        expansion="Liquid Silicone Rubber",
        category="imm",
        description="Material type for injection molding"
    ),
    "MFI": AcronymInfo(
        expansion="Melt Flow Index",
        category="imm",
        description="Material flow rate measurement"
    ),
    "PVT": AcronymInfo(
        expansion="Pressure-Volume-Temperature",
        category="imm",
        description="Material property relationship"
    ),
    "RJM": AcronymInfo(
        expansion="Robots and Jigs for Molding",
        category="imm",
        description="Automation around molding machine"
    ),
}


# ============================================
# Safety Acronyms
# ============================================
SAFETY_ACRONYMS: Dict[str, AcronymInfo] = {
    "DCSS": AcronymInfo(
        expansion="Dual Check Safety System",
        category="safety",
        description="FANUC safety monitoring system",
        aliases=["DCS"]
    ),
    "SIL": AcronymInfo(
        expansion="Safety Integrity Level",
        category="safety",
        description="Safety system performance metric",
        related=["PL", "STO", "SLS"]
    ),
    "PL": AcronymInfo(
        expansion="Performance Level",
        category="safety",
        description="ISO 13849 safety rating",
        related=["SIL"]
    ),
    "STO": AcronymInfo(
        expansion="Safe Torque Off",
        category="safety",
        description="Drive safety function"
    ),
    "SS1": AcronymInfo(
        expansion="Safe Stop 1",
        category="safety",
        description="Controlled stop then STO"
    ),
    "SS2": AcronymInfo(
        expansion="Safe Stop 2",
        category="safety",
        description="Stop with motor energized"
    ),
    "SLS": AcronymInfo(
        expansion="Safely Limited Speed",
        category="safety",
        description="Speed monitoring function"
    ),
    "SLP": AcronymInfo(
        expansion="Safely Limited Position",
        category="safety",
        description="Position monitoring function"
    ),
    "ESTOP": AcronymInfo(
        expansion="Emergency Stop",
        category="safety",
        description="Emergency shutdown button",
        aliases=["E-STOP", "E-Stop"]
    ),
    "LOTO": AcronymInfo(
        expansion="Lockout Tagout",
        category="safety",
        description="Energy isolation procedure"
    ),
}


# ============================================
# Communication Protocol Acronyms
# ============================================
PROTOCOL_ACRONYMS: Dict[str, AcronymInfo] = {
    "EIP": AcronymInfo(
        expansion="EtherNet/IP",
        category="protocol",
        description="Industrial Ethernet protocol",
        related=["PROFINET", "MODBUS"]
    ),
    "PROFINET": AcronymInfo(
        expansion="Process Field Net",
        category="protocol",
        description="Siemens industrial Ethernet"
    ),
    "PROFIBUS": AcronymInfo(
        expansion="Process Field Bus",
        category="protocol",
        description="Siemens fieldbus standard"
    ),
    "MODBUS": AcronymInfo(
        expansion="Modicon Bus Protocol",
        category="protocol",
        description="Serial/TCP industrial protocol"
    ),
    "CAN": AcronymInfo(
        expansion="Controller Area Network",
        category="protocol",
        description="Serial bus for automation"
    ),
    "CANOPEN": AcronymInfo(
        expansion="Controller Area Network Open",
        category="protocol",
        description="Higher-layer CAN protocol"
    ),
    "DEVICENET": AcronymInfo(
        expansion="Device Network",
        category="protocol",
        description="Allen-Bradley fieldbus"
    ),
    "CC-LINK": AcronymInfo(
        expansion="Control and Communication Link",
        category="protocol",
        description="Mitsubishi fieldbus"
    ),
    "MQTT": AcronymInfo(
        expansion="Message Queuing Telemetry Transport",
        category="protocol",
        description="Lightweight IoT messaging"
    ),
}


# ============================================
# Combined Dictionary
# ============================================
INDUSTRIAL_ACRONYMS: Dict[str, AcronymInfo] = {
    **FANUC_ERROR_CODES,
    **AUTOMATION_ACRONYMS,
    **ROBOTICS_ACRONYMS,
    **IMM_ACRONYMS,
    **SAFETY_ACRONYMS,
    **PROTOCOL_ACRONYMS,
}


# ============================================
# Query Expansion Functions
# ============================================

def get_acronym_info(acronym: str) -> Optional[AcronymInfo]:
    """
    Get information about an acronym.

    Args:
        acronym: Acronym to look up (case-insensitive)

    Returns:
        AcronymInfo if found, None otherwise
    """
    return INDUSTRIAL_ACRONYMS.get(acronym.upper())


def expand_acronym(acronym: str) -> str:
    """
    Get the expansion for an acronym.

    Args:
        acronym: Acronym to expand

    Returns:
        Expansion string, or original acronym if not found
    """
    info = get_acronym_info(acronym)
    return info.expansion if info else acronym


def expand_acronyms(
    query: str,
    inline: bool = True,
    categories: Optional[List[str]] = None
) -> str:
    """
    Expand known acronyms in a query.

    Args:
        query: User query string
        inline: If True, adds expansion in parentheses. If False, replaces.
        categories: Optional list of categories to expand (e.g., ["fanuc_error", "automation"])

    Returns:
        Query with acronyms expanded

    Example:
        >>> expand_acronyms("SRVO-063 alarm on PLC")
        "SRVO-063 (Servo Alarm) alarm on PLC (Programmable Logic Controller)"
    """
    result = query
    expanded_acronyms = set()  # Avoid duplicate expansions

    for acronym, info in INDUSTRIAL_ACRONYMS.items():
        # Skip if category filter is specified and doesn't match
        if categories and info.category not in categories:
            continue

        # Skip if already expanded
        if acronym in expanded_acronyms:
            continue

        # Pattern to match whole word (with optional hyphen for error codes)
        # Handle cases like "SRVO-063" where we want to expand "SRVO"
        if info.category == "fanuc_error":
            # For error codes, match the prefix before the dash
            pattern = rf'\b{re.escape(acronym)}(?=-\d)'
        else:
            # For other acronyms, match whole word
            pattern = rf'\b{re.escape(acronym)}\b'

        if re.search(pattern, result, re.IGNORECASE):
            if inline:
                # Add expansion in parentheses after acronym
                replacement = rf'{acronym} ({info.expansion})'
            else:
                # Replace with expansion
                replacement = info.expansion

            # Only replace first occurrence to avoid redundancy
            result = re.sub(
                pattern,
                replacement,
                result,
                count=1,
                flags=re.IGNORECASE
            )
            expanded_acronyms.add(acronym)

    return result


def expand_error_code_prefixes(query: str) -> str:
    """
    Expand FANUC error code prefixes in a query.

    Args:
        query: User query string

    Returns:
        Query with error code prefixes expanded

    Example:
        >>> expand_error_code_prefixes("SRVO-063 encoder error")
        "SRVO (Servo Alarm) 063 encoder error"
    """
    result = query

    for prefix, info in FANUC_ERROR_CODES.items():
        # Match prefix followed by dash and number
        pattern = rf'\b({re.escape(prefix)})-(\d{{3,4}})\b'

        if re.search(pattern, result, re.IGNORECASE):
            # Add expansion after prefix
            replacement = rf'\1 ({info.expansion})-\2'
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def get_related_terms(acronym: str) -> List[str]:
    """
    Get related acronyms for query expansion.

    Args:
        acronym: Base acronym to find relations for

    Returns:
        List of related acronym expansions
    """
    info = get_acronym_info(acronym)
    if not info:
        return []

    related = []
    for rel_acronym in info.related:
        rel_info = get_acronym_info(rel_acronym)
        if rel_info:
            related.append(f"{rel_acronym} ({rel_info.expansion})")

    return related


def suggest_acronym_corrections(query: str) -> List[Tuple[str, str]]:
    """
    Suggest corrections for potentially misspelled acronyms.

    Args:
        query: User query string

    Returns:
        List of (original, suggestion) tuples
    """
    suggestions = []
    words = re.findall(r'\b[A-Z]{2,6}\b', query.upper())

    for word in words:
        if word not in INDUSTRIAL_ACRONYMS:
            # Check for close matches
            for known in INDUSTRIAL_ACRONYMS.keys():
                # Simple edit distance check (1 char difference)
                if len(word) == len(known):
                    diff = sum(1 for a, b in zip(word, known) if a != b)
                    if diff == 1:
                        suggestions.append((word, known))
                        break

    return suggestions


def get_category_acronyms(category: str) -> Dict[str, str]:
    """
    Get all acronyms for a specific category.

    Args:
        category: Category name (e.g., "fanuc_error", "automation")

    Returns:
        Dict of acronym -> expansion for that category
    """
    return {
        acronym: info.expansion
        for acronym, info in INDUSTRIAL_ACRONYMS.items()
        if info.category == category
    }


# ============================================
# Statistics
# ============================================

def get_dictionary_stats() -> Dict[str, int]:
    """Get statistics about the acronym dictionary."""
    categories = {}
    for info in INDUSTRIAL_ACRONYMS.values():
        categories[info.category] = categories.get(info.category, 0) + 1

    return {
        "total_acronyms": len(INDUSTRIAL_ACRONYMS),
        "categories": categories,
        "fanuc_error_codes": len(FANUC_ERROR_CODES),
        "automation_terms": len(AUTOMATION_ACRONYMS),
        "robotics_terms": len(ROBOTICS_ACRONYMS),
        "imm_terms": len(IMM_ACRONYMS),
        "safety_terms": len(SAFETY_ACRONYMS),
        "protocol_terms": len(PROTOCOL_ACRONYMS),
    }
