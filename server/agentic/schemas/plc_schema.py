"""
PLC/Automation Domain Schema

Entity patterns and relationships for industrial PLC systems:
- Allen-Bradley/Rockwell Automation (ControlLogix, CompactLogix, PLC-5, SLC)
- Siemens (S7-300/400/1200/1500, TIA Portal)
- AutomationDirect (Click, Do-more, Productivity series)

Covers:
- Fault codes and alarms
- Addressing schemes
- Communication protocols (EtherNet/IP, Profinet, Modbus)
- HMI systems (FactoryTalk, WinCC)
- Programming patterns

Author: Claude Code
Date: December 2025
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================
# ENTITY TYPES
# ============================================

class PLCEntityType(str, Enum):
    """PLC-related entity types"""
    FAULT_CODE = "fault_code"              # PLC fault/alarm codes
    MODULE = "module"                       # I/O modules, CPUs, comm modules
    ADDRESS = "address"                     # PLC memory addresses
    INSTRUCTION = "instruction"             # Ladder logic instructions
    PROTOCOL = "protocol"                   # Communication protocols
    PARAMETER = "parameter"                 # System parameters/settings
    HMI_ELEMENT = "hmi_element"            # HMI screens, tags, alarams
    NETWORK = "network"                     # Network configuration
    PROCEDURE = "procedure"                 # Maintenance/troubleshooting steps


class PLCRelationType(str, Enum):
    """PLC-related relationship types"""
    CAUSES = "causes"                       # Fault causes error
    RESOLVES = "resolves"                   # Procedure resolves fault
    COMMUNICATES_WITH = "communicates_with" # Protocol connects modules
    ADDRESSES = "addresses"                 # Instruction addresses memory
    DISPLAYS = "displays"                   # HMI displays data
    CONTROLS = "controls"                   # PLC controls device
    MONITORS = "monitors"                   # System monitors parameter


# ============================================
# ALLEN-BRADLEY / ROCKWELL PATTERNS
# ============================================

# Major/Minor fault codes (ControlLogix, CompactLogix)
AB_FAULT_PATTERNS: List[Tuple[str, str]] = [
    # Major faults (Type:Code format)
    (r"\b(?:Major\s*)?Fault\s*(?:Type\s*)?(\d{1,2}):(\d{1,3})\b", "major_minor_fault"),
    (r"\bType\s*(\d{1,2})\s*Code\s*(\d{1,3})\b", "type_code_fault"),

    # Specific major fault types
    (r"\bMajor\s*Fault\s*Type\s*1\b", "power_loss_fault"),
    (r"\bMajor\s*Fault\s*Type\s*3\b", "io_fault"),
    (r"\bMajor\s*Fault\s*Type\s*4\b", "program_fault"),
    (r"\bMajor\s*Fault\s*Type\s*6\b", "motion_fault"),
    (r"\bMajor\s*Fault\s*Type\s*7\b", "safety_fault"),

    # Minor faults
    (r"\bMinor\s*Fault\s*(\d{1,3})\b", "minor_fault"),

    # Module faults
    (r"\bModule\s*Fault\s*(\d+)\b", "module_fault"),
    (r"\bConnection\s*Fault\b", "connection_fault"),

    # Servo drive faults (Kinetix)
    (r"\bKinetix\s*Fault\s*(\d+)\b", "kinetix_fault"),
    (r"\bServo\s*Fault\s*(\d+)\b", "servo_fault"),

    # PowerFlex drive faults
    (r"\bPowerFlex\s*Fault\s*(\d+)\b", "powerflex_fault"),
    (r"\bF(\d{3})\b.*(?:PowerFlex|VFD)", "powerflex_f_code"),
]

# Allen-Bradley addressing patterns
AB_ADDRESS_PATTERNS: List[Tuple[str, str]] = [
    # ControlLogix tag-based addressing
    (r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[\d+\])?\b", "tag_address"),

    # PLC-5/SLC-500 file-based addressing
    (r"\b[NIOBSTCRD]\d+[:/]\d+(?:/\d+)?\b", "plc5_slc_address"),
    (r"\bN7:\d+\b", "integer_file"),
    (r"\bB3[:/]\d+(?:/\d+)?\b", "bit_file"),
    (r"\bT4:\d+\b", "timer_file"),
    (r"\bC5:\d+\b", "counter_file"),
    (r"\bF8:\d+\b", "float_file"),
    (r"\bI:\d+(?:/\d+)?\b", "input_address"),
    (r"\bO:\d+(?:/\d+)?\b", "output_address"),

    # Compact I/O addressing
    (r"\bLocal:\d+:[IO]\.Data(?:\.\d+)?\b", "compact_io_address"),
]

# Allen-Bradley module patterns
AB_MODULE_PATTERNS: List[Tuple[str, str]] = [
    # ControlLogix modules
    (r"\b1756-[A-Z]+\d*[A-Z]?\b", "controllogix_module"),
    (r"\b1756-L\d+[A-Z]*\b", "controllogix_cpu"),
    (r"\b1756-EN\d*[A-Z]*\b", "ethernet_module"),
    (r"\b1756-IB\d+\b", "digital_input"),
    (r"\b1756-OB\d+\b", "digital_output"),
    (r"\b1756-IF\d+\b", "analog_input"),
    (r"\b1756-OF\d+\b", "analog_output"),

    # CompactLogix modules
    (r"\b1769-[A-Z]+\d*[A-Z]?\b", "compactlogix_module"),
    (r"\b1769-L\d+[A-Z]*\b", "compactlogix_cpu"),

    # Micro800 series
    (r"\b2080-[A-Z]+\d*\b", "micro800_module"),

    # PLC-5
    (r"\b1785-[A-Z]+\d*\b", "plc5_module"),

    # SLC-500
    (r"\b1747-[A-Z]+\d*\b", "slc500_module"),

    # PowerFlex drives
    (r"\bPowerFlex\s*\d+[A-Z]?\b", "powerflex_drive"),
    (r"\b20F-[A-Z]+\d+\b", "powerflex_700"),
    (r"\b25B-[A-Z]+\d+\b", "powerflex_525"),

    # Kinetix servo
    (r"\bKinetix\s*\d+\b", "kinetix_drive"),
    (r"\b2198-[A-Z]+\d+\b", "kinetix_module"),
]


# ============================================
# SIEMENS PATTERNS
# ============================================

# Siemens fault/diagnostic codes
SIEMENS_FAULT_PATTERNS: List[Tuple[str, str]] = [
    # S7 diagnostic codes
    (r"\bF\d{5}\b", "siemens_f_fault"),  # e.g., F00001
    (r"\bA\d{5}\b", "siemens_alarm"),    # e.g., A00001

    # Event IDs
    (r"\b16#[0-9A-Fa-f]{4,8}\b", "siemens_event_id"),

    # Drive faults (SINAMICS)
    (r"\bF\d{5}\b.*SINAMICS", "sinamics_fault"),
    (r"\bA\d{5}\b.*SINAMICS", "sinamics_alarm"),

    # CPU stop causes
    (r"\bOB\s*\d+\s*(?:not\s*loaded|error)\b", "ob_error"),

    # Safety faults
    (r"\bF-CPU\s*(?:fault|stop)\b", "safety_cpu_fault"),
]

# Siemens addressing patterns
SIEMENS_ADDRESS_PATTERNS: List[Tuple[str, str]] = [
    # Absolute addressing
    (r"\b[IMQPTCD]B?\d+(?:\.\d+)?\b", "absolute_address"),
    (r"\bI\d+\.\d+\b", "input_bit"),
    (r"\bQ\d+\.\d+\b", "output_bit"),
    (r"\bM\d+\.\d+\b", "memory_bit"),
    (r"\bIW\d+\b", "input_word"),
    (r"\bQW\d+\b", "output_word"),
    (r"\bMW\d+\b", "memory_word"),
    (r"\bMD\d+\b", "memory_dword"),

    # DB addressing
    (r"\bDB\d+\.DB[XWDB]\d+(?:\.\d+)?\b", "db_address"),
    (r"\bDB\d+\b", "data_block"),

    # Timer/Counter
    (r"\bT\d+\b", "timer"),
    (r"\bC\d+\b", "counter"),

    # Symbolic addressing (TIA Portal)
    (r'"[A-Za-z_][A-Za-z0-9_]*"(?:\.[A-Za-z_][A-Za-z0-9_]*)*', "symbolic_address"),
]

# Siemens module patterns
SIEMENS_MODULE_PATTERNS: List[Tuple[str, str]] = [
    # S7-1500 series
    (r"\b6ES7\s*5\d{2}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d\b", "s7_1500_module"),
    (r"\bCPU\s*1511\b", "cpu_1511"),
    (r"\bCPU\s*1512\b", "cpu_1512"),
    (r"\bCPU\s*1513\b", "cpu_1513"),
    (r"\bCPU\s*1515\b", "cpu_1515"),
    (r"\bCPU\s*1516\b", "cpu_1516"),
    (r"\bCPU\s*1517\b", "cpu_1517"),
    (r"\bCPU\s*1518\b", "cpu_1518"),

    # S7-1200 series
    (r"\b6ES7\s*2\d{2}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d\b", "s7_1200_module"),
    (r"\bCPU\s*121[0-5]\b", "s7_1200_cpu"),

    # S7-300/400 series
    (r"\b6ES7\s*3\d{2}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d\b", "s7_300_module"),
    (r"\b6ES7\s*4\d{2}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d\b", "s7_400_module"),
    (r"\bCPU\s*31[2-9]\b", "s7_300_cpu"),
    (r"\bCPU\s*41[4-7]\b", "s7_400_cpu"),

    # SINAMICS drives
    (r"\bSINAMICS\s*[SGVF]\d+\b", "sinamics_drive"),
    (r"\b6SL3\d{3}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d\b", "sinamics_module"),

    # ET 200 distributed I/O
    (r"\bET\s*200[A-Z]?[PS]?\b", "et200_system"),
]


# ============================================
# AUTOMATIONDIRECT PATTERNS
# ============================================

AD_FAULT_PATTERNS: List[Tuple[str, str]] = [
    # Click PLC errors
    (r"\bE\d{3}\b.*Click", "click_error"),

    # Do-more errors
    (r"\b(?:Error|Fault)\s*Code\s*\d+\b", "domore_error"),

    # GS series VFD faults
    (r"\bGS\d?\s*(?:Fault|Error)\s*[A-Z]?\d+\b", "gs_vfd_fault"),
    (r"\bOC\d?\b", "overcurrent_fault"),
    (r"\bOU\d?\b", "overvoltage_fault"),
    (r"\bOH\d?\b", "overheat_fault"),
]

AD_ADDRESS_PATTERNS: List[Tuple[str, str]] = [
    # Click PLC addressing
    (r"\b[XYCTSD]\d{1,4}\b", "click_address"),
    (r"\bX\d{1,4}\b", "click_input"),
    (r"\bY\d{1,4}\b", "click_output"),
    (r"\bC\d{1,4}\b", "click_control"),
    (r"\bT\d{1,4}\b", "click_timer"),
    (r"\bCT\d{1,4}\b", "click_counter"),
    (r"\bDS\d{1,4}\b", "click_data"),

    # Do-more/Productivity addressing
    (r"\b[VRMTC]\d+\b", "domore_address"),
]

AD_MODULE_PATTERNS: List[Tuple[str, str]] = [
    # Click series
    (r"\bC0-\d{2}[A-Z]{2,3}-?\d?\b", "click_module"),
    (r"\bC2-\d{2}[A-Z]{2,3}\b", "click2_module"),

    # Do-more series
    (r"\bDO-\d{2}[A-Z]{2,3}\b", "domore_module"),
    (r"\bH2-DM1[EZ]?\b", "domore_cpu"),

    # Productivity series
    (r"\bP[123]-\d{2}[A-Z]{2,3}\b", "productivity_module"),

    # BRX series
    (r"\bBRX[A-Z0-9]+\b", "brx_module"),
]


# ============================================
# COMMUNICATION PROTOCOL PATTERNS
# ============================================

PROTOCOL_PATTERNS: List[Tuple[str, str]] = [
    # Industrial Ethernet
    (r"\bEtherNet/IP\b", "ethernet_ip"),
    (r"\bProfinet\b", "profinet"),
    (r"\bModbus\s*TCP\b", "modbus_tcp"),
    (r"\bEtherCAT\b", "ethercat"),
    (r"\bPowerlink\b", "powerlink"),

    # Serial protocols
    (r"\bModbus\s*RTU\b", "modbus_rtu"),
    (r"\bDF1\b", "df1"),
    (r"\bDH\+\b", "dh_plus"),
    (r"\bDH-485\b", "dh485"),
    (r"\bMPI\b", "mpi"),
    (r"\bProfibus\b", "profibus"),

    # Device protocols
    (r"\bCIP\b", "cip"),
    (r"\bDeviceNet\b", "devicenet"),
    (r"\bControlNet\b", "controlnet"),
    (r"\bASi\b", "asi"),

    # IT protocols in OT
    (r"\bOPC\s*UA\b", "opc_ua"),
    (r"\bOPC\s*DA\b", "opc_da"),
    (r"\bMQTT\b", "mqtt"),
]


# ============================================
# HMI PATTERNS
# ============================================

HMI_PATTERNS: List[Tuple[str, str]] = [
    # Rockwell HMI
    (r"\bFactoryTalk\s*View\b", "factorytalk_view"),
    (r"\bPanelView\s*Plus\b", "panelview_plus"),
    (r"\bPanelView\s*\d+\b", "panelview"),

    # Siemens HMI
    (r"\bWinCC\b", "wincc"),
    (r"\bTIA\s*Portal\b", "tia_portal"),
    (r"\bKTP\d+\b", "siemens_ktp"),
    (r"\bTP\d+\b.*Siemens", "siemens_tp"),
    (r"\bComfort\s*Panel\b", "comfort_panel"),

    # AutomationDirect HMI
    (r"\bC-more\b", "cmore"),
    (r"\bEA\d-[A-Z]\d+[A-Z]+\b", "cmore_panel"),

    # Generic HMI elements
    (r"\balarm\s*(?:history|log|summary)\b", "alarm_screen"),
    (r"\btrend\s*(?:screen|display)\b", "trend_screen"),
    (r"\bfaceplate\b", "faceplate"),
]


# ============================================
# INSTRUCTION PATTERNS
# ============================================

INSTRUCTION_PATTERNS: List[Tuple[str, str]] = [
    # Common ladder logic
    (r"\bXIC\b", "examine_closed"),
    (r"\bXIO\b", "examine_open"),
    (r"\bOTE\b", "output_energize"),
    (r"\bOTL\b", "output_latch"),
    (r"\bOTU\b", "output_unlatch"),
    (r"\bTON\b", "timer_on_delay"),
    (r"\bTOF\b", "timer_off_delay"),
    (r"\bRTO\b", "retentive_timer"),
    (r"\bCTU\b", "count_up"),
    (r"\bCTD\b", "count_down"),
    (r"\bMOV\b", "move"),
    (r"\bCOP\b", "copy"),
    (r"\bCMP\b", "compare"),
    (r"\bADD\b", "add"),
    (r"\bSUB\b", "subtract"),
    (r"\bMUL\b", "multiply"),
    (r"\bDIV\b", "divide"),
    (r"\bJSR\b", "jump_subroutine"),
    (r"\bRET\b", "return"),
    (r"\bMSG\b", "message"),
    (r"\bGSV\b", "get_system_value"),
    (r"\bSSV\b", "set_system_value"),

    # Motion instructions
    (r"\bMAJ\b", "motion_axis_jog"),
    (r"\bMAM\b", "motion_axis_move"),
    (r"\bMAS\b", "motion_axis_stop"),
    (r"\bMAH\b", "motion_axis_home"),
    (r"\bMSO\b", "motion_servo_on"),
    (r"\bMSF\b", "motion_servo_off"),
]


# ============================================
# COMBINED SCHEMA
# ============================================

PLC_SCHEMA: Dict = {
    "domain": "plc_automation",
    "name": "PLC/Automation Systems",
    "description": "Industrial PLC systems from Allen-Bradley, Siemens, and AutomationDirect",
    "manufacturers": ["Allen-Bradley/Rockwell", "Siemens", "AutomationDirect"],
    "entity_types": [e.value for e in PLCEntityType],
    "relation_types": [r.value for r in PLCRelationType],
    "patterns": {
        "allen_bradley": {
            "faults": AB_FAULT_PATTERNS,
            "addresses": AB_ADDRESS_PATTERNS,
            "modules": AB_MODULE_PATTERNS,
        },
        "siemens": {
            "faults": SIEMENS_FAULT_PATTERNS,
            "addresses": SIEMENS_ADDRESS_PATTERNS,
            "modules": SIEMENS_MODULE_PATTERNS,
        },
        "automationdirect": {
            "faults": AD_FAULT_PATTERNS,
            "addresses": AD_ADDRESS_PATTERNS,
            "modules": AD_MODULE_PATTERNS,
        },
        "protocols": PROTOCOL_PATTERNS,
        "hmi": HMI_PATTERNS,
        "instructions": INSTRUCTION_PATTERNS,
    }
}


# ============================================
# PLC ACRONYMS
# ============================================

PLC_ACRONYMS: Dict[str, Dict[str, str]] = {
    # General PLC
    "PLC": {"expansion": "Programmable Logic Controller", "category": "plc"},
    "PAC": {"expansion": "Programmable Automation Controller", "category": "plc"},
    "DCS": {"expansion": "Distributed Control System", "category": "plc"},
    "SCADA": {"expansion": "Supervisory Control and Data Acquisition", "category": "plc"},
    "RTU": {"expansion": "Remote Terminal Unit", "category": "plc"},

    # Allen-Bradley specific
    "RSLogix": {"expansion": "Rockwell Software Logix", "category": "allen_bradley"},
    "RSLinx": {"expansion": "Rockwell Software Linx Communication", "category": "allen_bradley"},
    "AOI": {"expansion": "Add-On Instruction", "category": "allen_bradley"},
    "UDT": {"expansion": "User-Defined Data Type", "category": "allen_bradley"},
    "GSV": {"expansion": "Get System Value", "category": "allen_bradley"},
    "SSV": {"expansion": "Set System Value", "category": "allen_bradley"},
    "MSG": {"expansion": "Message Instruction", "category": "allen_bradley"},
    "CIP": {"expansion": "Common Industrial Protocol", "category": "allen_bradley"},
    "EDS": {"expansion": "Electronic Data Sheet", "category": "allen_bradley"},

    # Siemens specific
    "TIA": {"expansion": "Totally Integrated Automation", "category": "siemens"},
    "WinCC": {"expansion": "Windows Control Center", "category": "siemens"},
    "OB": {"expansion": "Organization Block", "category": "siemens"},
    "FB": {"expansion": "Function Block", "category": "siemens"},
    "FC": {"expansion": "Function", "category": "siemens"},
    "DB": {"expansion": "Data Block", "category": "siemens"},
    "SFB": {"expansion": "System Function Block", "category": "siemens"},
    "SFC": {"expansion": "System Function", "category": "siemens"},
    "MPI": {"expansion": "Multi-Point Interface", "category": "siemens"},
    "PG": {"expansion": "Programming Device", "category": "siemens"},

    # Communication
    "EIP": {"expansion": "EtherNet/IP", "category": "protocol"},
    "OPC": {"expansion": "Open Platform Communications", "category": "protocol"},
    "OPC-UA": {"expansion": "OPC Unified Architecture", "category": "protocol"},

    # I/O and addressing
    "DI": {"expansion": "Digital Input", "category": "io"},
    "DO": {"expansion": "Digital Output", "category": "io"},
    "AI": {"expansion": "Analog Input", "category": "io"},
    "AO": {"expansion": "Analog Output", "category": "io"},
    "HSC": {"expansion": "High-Speed Counter", "category": "io"},
    "PWM": {"expansion": "Pulse Width Modulation", "category": "io"},
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def is_plc_query(query: str) -> bool:
    """
    Check if a query is related to PLC/automation systems.

    Args:
        query: Search query text

    Returns:
        True if query relates to PLC/automation
    """
    query_lower = query.lower()

    # Direct PLC keywords
    plc_keywords = [
        "plc", "programmable logic", "ladder logic", "controllogix",
        "compactlogix", "micrologix", "slc-500", "plc-5",
        "s7-300", "s7-400", "s7-1200", "s7-1500", "simatic",
        "tia portal", "step 7", "wincc",
        "click plc", "do-more", "productivity", "automation direct",
        "factorytalk", "rslogix", "studio 5000",
        "ethernet/ip", "profinet", "modbus", "devicenet",
        "hmi", "panelview", "c-more",
        "fault code", "major fault", "minor fault",
    ]

    for keyword in plc_keywords:
        if keyword in query_lower:
            return True

    # Pattern matching for fault codes
    fault_patterns = [
        r"\btype\s*\d+\s*code\s*\d+",  # Allen-Bradley
        r"\bfault\s*\d+:\d+",           # Major:Minor
        r"\bF\d{5}\b",                  # Siemens
        r"\b1756-",                     # ControlLogix modules
        r"\b6ES7",                      # Siemens modules
    ]

    for pattern in fault_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    return False


def detect_plc_manufacturer(query: str) -> Optional[str]:
    """
    Detect the PLC manufacturer from a query.

    Args:
        query: Search query text

    Returns:
        Manufacturer name or None
    """
    query_lower = query.lower()

    # Allen-Bradley/Rockwell patterns
    ab_patterns = [
        "allen-bradley", "allen bradley", "rockwell", "controllogix",
        "compactlogix", "micrologix", "slc-500", "plc-5",
        "factorytalk", "rslogix", "studio 5000", "panelview",
        "1756-", "1769-", "1747-", "1785-",
        "powerflex", "kinetix"
    ]
    for p in ab_patterns:
        if p in query_lower:
            return "allen_bradley"

    # Siemens patterns
    siemens_patterns = [
        "siemens", "simatic", "s7-", "tia portal", "step 7",
        "wincc", "profinet", "6es7", "sinamics", "et 200"
    ]
    for p in siemens_patterns:
        if p in query_lower:
            return "siemens"

    # AutomationDirect patterns
    ad_patterns = [
        "automationdirect", "automation direct", "click plc",
        "do-more", "productivity", "c-more", "brx"
    ]
    for p in ad_patterns:
        if p in query_lower:
            return "automationdirect"

    return None


def extract_fault_codes(text: str) -> List[Dict[str, str]]:
    """
    Extract PLC fault codes from text.

    Args:
        text: Text to search

    Returns:
        List of extracted fault codes with type and value
    """
    faults = []

    # Allen-Bradley Major:Minor
    for match in re.finditer(r"\b(?:Fault\s*)?(\d{1,2}):(\d{1,3})\b", text, re.IGNORECASE):
        faults.append({
            "manufacturer": "allen_bradley",
            "type": "major_minor",
            "major": match.group(1),
            "minor": match.group(2),
            "raw": match.group(0)
        })

    # Siemens F-faults
    for match in re.finditer(r"\bF(\d{5})\b", text):
        faults.append({
            "manufacturer": "siemens",
            "type": "f_fault",
            "code": match.group(1),
            "raw": match.group(0)
        })

    # Siemens A-alarms
    for match in re.finditer(r"\bA(\d{5})\b", text):
        faults.append({
            "manufacturer": "siemens",
            "type": "alarm",
            "code": match.group(1),
            "raw": match.group(0)
        })

    return faults


def extract_module_numbers(text: str) -> List[Dict[str, str]]:
    """
    Extract PLC module part numbers from text.

    Args:
        text: Text to search

    Returns:
        List of extracted module numbers
    """
    modules = []

    # Allen-Bradley 1756 (ControlLogix)
    for match in re.finditer(r"\b(1756-[A-Z]+\d*[A-Z]?)\b", text):
        modules.append({
            "manufacturer": "allen_bradley",
            "series": "controllogix",
            "part_number": match.group(1)
        })

    # Allen-Bradley 1769 (CompactLogix)
    for match in re.finditer(r"\b(1769-[A-Z]+\d*[A-Z]?)\b", text):
        modules.append({
            "manufacturer": "allen_bradley",
            "series": "compactlogix",
            "part_number": match.group(1)
        })

    # Siemens 6ES7 modules
    for match in re.finditer(r"\b(6ES7\s*\d{3}-\d[A-Z]{2}\d{2}-\d[A-Z]{2}\d)\b", text):
        modules.append({
            "manufacturer": "siemens",
            "series": "s7",
            "part_number": match.group(1)
        })

    return modules


def get_fault_description(fault_type: int, fault_code: int) -> Optional[str]:
    """
    Get description for Allen-Bradley major fault type and code.

    Args:
        fault_type: Major fault type (1-7)
        fault_code: Fault code number

    Returns:
        Fault description or None
    """
    # Common Allen-Bradley major faults
    fault_descriptions = {
        (1, 1): "Power-Up Fault - Nonvolatile memory corrupt",
        (1, 60): "Power-Up Fault - Keyswitch change during execution",
        (1, 61): "Power-Up Fault - Cannot execute task",
        (3, 16): "I/O Fault - Connection lost",
        (3, 20): "I/O Fault - RPI out of range",
        (4, 16): "Program Fault - Unknown instruction",
        (4, 20): "Program Fault - Array subscript out of range",
        (4, 21): "Program Fault - Invalid array tag",
        (4, 31): "Program Fault - JSR recursion too deep",
        (4, 82): "Program Fault - Divide by zero",
        (4, 83): "Program Fault - Invalid value",
        (4, 84): "Program Fault - Invalid string",
        (6, 1): "Motion Fault - Axis safety fault",
        (6, 2): "Motion Fault - Axis servo fault",
        (6, 3): "Motion Fault - Servo on timeout",
        (7, 1): "Safety Fault - Safety task did not complete",
        (7, 2): "Safety Fault - Safety signature mismatch",
    }

    return fault_descriptions.get((fault_type, fault_code))


# ============================================
# DOMAIN SCHEMA CREATION
# ============================================

def create_plc_domain_schema():
    """
    Create a DomainSchema for PLC/automation corpus.

    Returns:
        DomainSchema compatible with DomainCorpus
    """
    from ..domain_corpus import (
        DomainSchema,
        DomainEntityDef,
        DomainRelationDef
    )

    entity_types = [
        DomainEntityDef(
            entity_type="fault_code",
            description="PLC fault and alarm codes",
            extraction_patterns=[p for p, _ in AB_FAULT_PATTERNS + SIEMENS_FAULT_PATTERNS + AD_FAULT_PATTERNS],
            examples=["Type 4 Code 20", "F00001", "Major Fault 3:16"],
            attributes=["manufacturer", "severity", "remedy"]
        ),
        DomainEntityDef(
            entity_type="module",
            description="PLC hardware modules (CPUs, I/O, drives)",
            extraction_patterns=[p for p, _ in AB_MODULE_PATTERNS + SIEMENS_MODULE_PATTERNS + AD_MODULE_PATTERNS],
            examples=["1756-L83E", "CPU 1516", "PowerFlex 525"],
            attributes=["manufacturer", "series", "function"]
        ),
        DomainEntityDef(
            entity_type="address",
            description="PLC memory addressing",
            extraction_patterns=[p for p, _ in AB_ADDRESS_PATTERNS + SIEMENS_ADDRESS_PATTERNS + AD_ADDRESS_PATTERNS],
            examples=["N7:0", "MW100", "DB10.DBX0.0"],
            attributes=["type", "data_type", "scope"]
        ),
        DomainEntityDef(
            entity_type="protocol",
            description="Industrial communication protocols",
            extraction_patterns=[p for p, _ in PROTOCOL_PATTERNS],
            examples=["EtherNet/IP", "Profinet", "Modbus TCP"],
            attributes=["speed", "topology", "compatibility"]
        ),
        DomainEntityDef(
            entity_type="instruction",
            description="PLC programming instructions",
            extraction_patterns=[p for p, _ in INSTRUCTION_PATTERNS],
            examples=["XIC", "OTE", "TON", "MSG"],
            attributes=["category", "usage", "parameters"]
        ),
        DomainEntityDef(
            entity_type="hmi_element",
            description="HMI systems and elements",
            extraction_patterns=[p for p, _ in HMI_PATTERNS],
            examples=["FactoryTalk View", "WinCC", "C-more"],
            attributes=["manufacturer", "screen_type"]
        ),
    ]

    relationships = [
        DomainRelationDef(
            relation_type="causes",
            source_types=["fault_code"],
            target_types=["fault_code", "module"],
            description="Fault causes another issue"
        ),
        DomainRelationDef(
            relation_type="resolves",
            source_types=["procedure"],
            target_types=["fault_code"],
            description="Procedure resolves fault"
        ),
        DomainRelationDef(
            relation_type="communicates_with",
            source_types=["module", "protocol"],
            target_types=["module"],
            description="Communication path"
        ),
        DomainRelationDef(
            relation_type="addresses",
            source_types=["instruction"],
            target_types=["address"],
            description="Instruction uses address"
        ),
        DomainRelationDef(
            relation_type="displays",
            source_types=["hmi_element"],
            target_types=["address", "fault_code"],
            description="HMI displays data"
        ),
    ]

    return DomainSchema(
        domain_id="plc_automation",
        domain_name="PLC/Automation Systems",
        description="Industrial PLC troubleshooting for Allen-Bradley, Siemens, and AutomationDirect",
        entity_types=entity_types,
        relationships=relationships,
        extraction_hints={
            "fault_patterns": [p for p, _ in AB_FAULT_PATTERNS + SIEMENS_FAULT_PATTERNS],
            "module_patterns": [p for p, _ in AB_MODULE_PATTERNS + SIEMENS_MODULE_PATTERNS],
            "address_patterns": [p for p, _ in AB_ADDRESS_PATTERNS + SIEMENS_ADDRESS_PATTERNS],
        },
        priority_patterns=[
            "fault", "alarm", "error", "1756", "6ES7", "controllogix",
            "s7-1500", "ethernet/ip", "profinet", "hmi", "troubleshoot"
        ]
    )


__all__ = [
    "PLCEntityType",
    "PLCRelationType",
    "PLC_SCHEMA",
    "PLC_ACRONYMS",
    "AB_FAULT_PATTERNS",
    "AB_ADDRESS_PATTERNS",
    "AB_MODULE_PATTERNS",
    "SIEMENS_FAULT_PATTERNS",
    "SIEMENS_ADDRESS_PATTERNS",
    "SIEMENS_MODULE_PATTERNS",
    "AD_FAULT_PATTERNS",
    "AD_ADDRESS_PATTERNS",
    "AD_MODULE_PATTERNS",
    "PROTOCOL_PATTERNS",
    "HMI_PATTERNS",
    "INSTRUCTION_PATTERNS",
    "is_plc_query",
    "detect_plc_manufacturer",
    "extract_fault_codes",
    "extract_module_numbers",
    "get_fault_description",
    "create_plc_domain_schema",
]
