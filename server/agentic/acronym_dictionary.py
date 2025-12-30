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
    "OPC-UA": AcronymInfo(
        expansion="Open Platform Communications Unified Architecture",
        category="protocol",
        description="Industrial interoperability standard",
        aliases=["OPC UA", "OPC"],
        related=["PROFINET", "EIP"]
    ),
    "CIP": AcronymInfo(
        expansion="Common Industrial Protocol",
        category="protocol",
        description="Base for EtherNet/IP, DeviceNet, ControlNet"
    ),
}


# ============================================
# Controller System Acronyms
# ============================================
CONTROLLER_ACRONYMS: Dict[str, AcronymInfo] = {
    # Cincinnati Milacron Controllers
    "CAMAC": AcronymInfo(
        expansion="Cincinnati Milacron Controller",
        category="milacron",
        description="Legacy IMM controller (486, VEL, VSX, VTL variants)",
        aliases=["CAMAC 486", "CAMAC VEL"]
    ),
    "MOSAIC": AcronymInfo(
        expansion="Milacron Operator Supervisory and Control",
        category="milacron",
        description="Current Milacron IMM controller platform",
        aliases=["MOSAIC+", "MOSAIC Plus"]
    ),
    # Van Dorn Controllers
    "PATHFINDER": AcronymInfo(
        expansion="Van Dorn PathFinder Controller",
        category="van_dorn",
        description="Van Dorn IMM controller series (1000-5000)",
        aliases=["PathFinder", "Pathfinder 3000", "Pathfinder 5000"]
    ),
    # KraussMaffei Controllers
    "MC": AcronymInfo(
        expansion="Motion Control (KraussMaffei)",
        category="kraussmaffei",
        description="KraussMaffei controller series (MC1-MC6)",
        aliases=["MC4", "MC5", "MC6"]
    ),
    # Real-Time Operating Systems
    "QNX": AcronymInfo(
        expansion="QNX Neutrino RTOS",
        category="rtos",
        description="Real-time operating system for industrial controllers",
        aliases=["Neutrino", "QNX Neutrino"]
    ),
    # FANUC Controller Models
    "R-30iB": AcronymInfo(
        expansion="FANUC R-30iB Controller",
        category="fanuc",
        description="FANUC robot controller platform",
        aliases=["R30iB", "R-30iB Plus", "R-30iB Mate"]
    ),
    "R-J3iB": AcronymInfo(
        expansion="FANUC R-J3iB Controller",
        category="fanuc",
        description="Legacy FANUC robot controller"
    ),
}


# ============================================
# HMI System Acronyms
# ============================================
HMI_ACRONYMS: Dict[str, AcronymInfo] = {
    "FACTORYTALK": AcronymInfo(
        expansion="Rockwell FactoryTalk Software Suite",
        category="hmi",
        description="Allen-Bradley/Rockwell HMI and MES platform",
        aliases=["FactoryTalk View", "FT View"]
    ),
    "WINCC": AcronymInfo(
        expansion="Siemens WinCC HMI Software",
        category="hmi",
        description="Siemens visualization and SCADA system",
        aliases=["WinCC Flexible", "WinCC Professional"]
    ),
    "PANELVIEW": AcronymInfo(
        expansion="Allen-Bradley PanelView HMI",
        category="hmi",
        description="Rockwell operator interface terminal",
        aliases=["PanelView Plus", "PanelView 5000"]
    ),
    "KEPSERVER": AcronymInfo(
        expansion="Kepware OPC Server",
        category="hmi",
        description="Industrial connectivity platform",
        aliases=["KEPServerEX"]
    ),
    "IGNITION": AcronymInfo(
        expansion="Inductive Automation Ignition",
        category="hmi",
        description="SCADA and industrial automation platform"
    ),
}


# ============================================
# RJG Scientific Molding Acronyms
# ============================================
SCIENTIFIC_MOLDING_ACRONYMS: Dict[str, AcronymInfo] = {
    # RJG Products
    "eDART": AcronymInfo(
        expansion="Electronic Data Acquisition in Real Time",
        category="rjg",
        description="RJG cavity pressure monitoring system"
    ),
    "COPILOT": AcronymInfo(
        expansion="CoPilot Process Monitoring Dashboard",
        category="rjg",
        description="RJG real-time process visualization",
        aliases=["CoPilot"]
    ),
    "LEYII": AcronymInfo(
        expansion="Lynx Temperature Controller",
        category="rjg",
        description="RJG intelligent temperature controller"
    ),
    # Scientific Molding Methodology
    "DIII": AcronymInfo(
        expansion="Decoupled III Molding Process",
        category="scientific_molding",
        description="RJG advanced velocity-to-pressure transfer method",
        aliases=["D3", "Decoupled III"]
    ),
    "DII": AcronymInfo(
        expansion="Decoupled II Molding Process",
        category="scientific_molding",
        description="Velocity/pressure decoupled molding",
        aliases=["D2", "Decoupled II"]
    ),
    "VP": AcronymInfo(
        expansion="Velocity-to-Pressure Transfer",
        category="scientific_molding",
        description="Switchover point from fill to pack phase"
    ),
    "FPH": AcronymInfo(
        expansion="Fill-Pack-Hold Process Phases",
        category="scientific_molding",
        description="Three main injection phases"
    ),
    # Quality and Statistics
    "CPK": AcronymInfo(
        expansion="Process Capability Index",
        category="quality",
        description="Statistical measure of process capability (target: >1.33)",
        aliases=["Cpk", "Cp"]
    ),
    "SPC": AcronymInfo(
        expansion="Statistical Process Control",
        category="quality",
        description="Statistical monitoring of process variation"
    ),
    "DOE": AcronymInfo(
        expansion="Design of Experiments",
        category="quality",
        description="Systematic method to determine process relationships"
    ),
    "OEE": AcronymInfo(
        expansion="Overall Equipment Effectiveness",
        category="quality",
        description="Machine availability × performance × quality"
    ),
    "FPY": AcronymInfo(
        expansion="First Pass Yield",
        category="quality",
        description="Percentage of parts good on first attempt"
    ),
    "PPM": AcronymInfo(
        expansion="Parts Per Million Defects",
        category="quality",
        description="Quality metric for defect rate"
    ),
    # Material Properties
    "MFR": AcronymInfo(
        expansion="Melt Flow Rate",
        category="material",
        description="Material flow property (g/10 min)",
        aliases=["MFI", "MIF"]
    ),
    "MVR": AcronymInfo(
        expansion="Melt Volume Rate",
        category="material",
        description="Volumetric flow rate (cm³/10 min)"
    ),
    "TG": AcronymInfo(
        expansion="Glass Transition Temperature",
        category="material",
        description="Temperature where amorphous polymer becomes rubbery",
        aliases=["Tg"]
    ),
    "TM": AcronymInfo(
        expansion="Melt Temperature",
        category="material",
        description="Temperature where crystalline polymer flows",
        aliases=["Tm"]
    ),
    # Cavity Pressure Sensors
    "CPT": AcronymInfo(
        expansion="Cavity Pressure Transducer",
        category="sensor",
        description="Sensor measuring pressure inside mold cavity"
    ),
    "PSI": AcronymInfo(
        expansion="Pounds per Square Inch",
        category="measurement",
        description="Pressure unit commonly used in US molding"
    ),
    # Machine Interface
    "EM67": AcronymInfo(
        expansion="Euromap 67 Interface Standard",
        category="interface",
        description="Robot-IMM communication standard"
    ),
    "EM77": AcronymInfo(
        expansion="Euromap 77 Interface Standard",
        category="interface",
        description="Real-time data acquisition standard"
    ),
    "EM82": AcronymInfo(
        expansion="Euromap 82 Interface Standard",
        category="interface",
        description="OPC-UA based machine communication"
    ),
    # Process Terms
    "SVP": AcronymInfo(
        expansion="Specific Viscosity Profile",
        category="scientific_molding",
        description="Viscosity curve during injection"
    ),
    "GST": AcronymInfo(
        expansion="Gate Seal Time",
        category="scientific_molding",
        description="Time for gate to freeze off"
    ),
    "CTI": AcronymInfo(
        expansion="Cooling Time Index",
        category="scientific_molding",
        description="Standardized cooling time calculation"
    ),
    # Defect Terms
    "BSR": AcronymInfo(
        expansion="Burn/Scorch/Record Groove",
        category="defect",
        description="Thermal degradation defects"
    ),
}


# ============================================
# PLC/Automation Acronyms (Allen-Bradley, Siemens, AutomationDirect)
# ============================================
PLC_ACRONYMS: Dict[str, AcronymInfo] = {
    # General PLC
    "PLC": AcronymInfo(
        expansion="Programmable Logic Controller",
        category="plc",
        description="Industrial control computer for automation"
    ),
    "PAC": AcronymInfo(
        expansion="Programmable Automation Controller",
        category="plc",
        description="Advanced PLC with PC-like capabilities"
    ),
    "DCS": AcronymInfo(
        expansion="Distributed Control System",
        category="plc",
        description="Process control system with distributed I/O",
        aliases=["Distributed Controls"]
    ),
    "SCADA": AcronymInfo(
        expansion="Supervisory Control and Data Acquisition",
        category="plc",
        description="Industrial monitoring and control system"
    ),
    "RTU": AcronymInfo(
        expansion="Remote Terminal Unit",
        category="plc",
        description="Remote I/O controller for SCADA systems"
    ),

    # Allen-Bradley/Rockwell Specific
    "RSLogix": AcronymInfo(
        expansion="Rockwell Software Logix Programming",
        category="allen_bradley",
        description="PLC programming software for Allen-Bradley"
    ),
    "RSLinx": AcronymInfo(
        expansion="Rockwell Software Linx Communication",
        category="allen_bradley",
        description="Communication server for Allen-Bradley PLCs"
    ),
    "AOI": AcronymInfo(
        expansion="Add-On Instruction",
        category="allen_bradley",
        description="Reusable custom instruction in ControlLogix/CompactLogix",
        aliases=["Add On Instruction"]
    ),
    "UDT": AcronymInfo(
        expansion="User-Defined Data Type",
        category="allen_bradley",
        description="Custom data structure in Allen-Bradley PLCs"
    ),
    "GSV": AcronymInfo(
        expansion="Get System Value",
        category="allen_bradley",
        description="Instruction to read controller system data"
    ),
    "SSV": AcronymInfo(
        expansion="Set System Value",
        category="allen_bradley",
        description="Instruction to write controller system data"
    ),
    "MSG": AcronymInfo(
        expansion="Message Instruction",
        category="allen_bradley",
        description="Communication instruction between PLCs"
    ),
    "CIP": AcronymInfo(
        expansion="Common Industrial Protocol",
        category="allen_bradley",
        description="Protocol for EtherNet/IP and DeviceNet"
    ),
    "EDS": AcronymInfo(
        expansion="Electronic Data Sheet",
        category="allen_bradley",
        description="Device configuration file for network devices"
    ),
    "ACD": AcronymInfo(
        expansion="Archive Copy of Device",
        category="allen_bradley",
        description="Studio 5000 project file format"
    ),
    "RPI": AcronymInfo(
        expansion="Requested Packet Interval",
        category="allen_bradley",
        description="Communication scan rate in milliseconds"
    ),

    # Siemens Specific
    "TIA": AcronymInfo(
        expansion="Totally Integrated Automation",
        category="siemens",
        description="Siemens integrated automation framework"
    ),
    "WinCC": AcronymInfo(
        expansion="Windows Control Center",
        category="siemens",
        description="Siemens SCADA and HMI software"
    ),
    "OB": AcronymInfo(
        expansion="Organization Block",
        category="siemens",
        description="Siemens S7 main program structure",
        aliases=["Organisation Block"]
    ),
    "FB": AcronymInfo(
        expansion="Function Block",
        category="siemens",
        description="Siemens S7 reusable logic with instance data"
    ),
    "FC": AcronymInfo(
        expansion="Function",
        category="siemens",
        description="Siemens S7 reusable logic without instance data"
    ),
    "DB": AcronymInfo(
        expansion="Data Block",
        category="siemens",
        description="Siemens S7 data storage area"
    ),
    "SFB": AcronymInfo(
        expansion="System Function Block",
        category="siemens",
        description="Built-in Siemens function block"
    ),
    "SFC": AcronymInfo(
        expansion="System Function",
        category="siemens",
        description="Built-in Siemens function"
    ),
    "MPI": AcronymInfo(
        expansion="Multi-Point Interface",
        category="siemens",
        description="Siemens S7 programming interface"
    ),
    "PG": AcronymInfo(
        expansion="Programming Device",
        category="siemens",
        description="Computer for programming Siemens PLCs"
    ),
    "CPU": AcronymInfo(
        expansion="Central Processing Unit",
        category="plc",
        description="Main processor module in PLC"
    ),
    "IM": AcronymInfo(
        expansion="Interface Module",
        category="siemens",
        description="Siemens rack interconnect module"
    ),

    # Communication Protocols
    "EIP": AcronymInfo(
        expansion="EtherNet/IP",
        category="protocol",
        description="Industrial Ethernet protocol (Rockwell, ODVA)"
    ),
    "PN": AcronymInfo(
        expansion="Profinet",
        category="protocol",
        description="Siemens industrial Ethernet standard"
    ),
    "DP": AcronymInfo(
        expansion="Decentralized Periphery",
        category="protocol",
        description="Profibus DP remote I/O"
    ),
    "OPC": AcronymInfo(
        expansion="Open Platform Communications",
        category="protocol",
        description="Industrial data exchange standard"
    ),
    "OPC-UA": AcronymInfo(
        expansion="OPC Unified Architecture",
        category="protocol",
        description="Modern secure industrial protocol",
        aliases=["OPCUA", "OPC UA"]
    ),

    # I/O and Addressing
    "DI": AcronymInfo(
        expansion="Digital Input",
        category="io",
        description="Discrete/binary input signal"
    ),
    "DO": AcronymInfo(
        expansion="Digital Output",
        category="io",
        description="Discrete/binary output signal"
    ),
    "AI": AcronymInfo(
        expansion="Analog Input",
        category="io",
        description="4-20mA or 0-10V input signal"
    ),
    "AO": AcronymInfo(
        expansion="Analog Output",
        category="io",
        description="4-20mA or 0-10V output signal"
    ),
    "HSC": AcronymInfo(
        expansion="High-Speed Counter",
        category="io",
        description="Fast pulse counting input"
    ),
    "PWM": AcronymInfo(
        expansion="Pulse Width Modulation",
        category="io",
        description="Variable duty cycle output"
    ),

    # Programming Instructions
    "XIC": AcronymInfo(
        expansion="Examine If Closed",
        category="ladder_logic",
        description="Normally open contact instruction"
    ),
    "XIO": AcronymInfo(
        expansion="Examine If Open",
        category="ladder_logic",
        description="Normally closed contact instruction"
    ),
    "OTE": AcronymInfo(
        expansion="Output Energize",
        category="ladder_logic",
        description="Standard output coil instruction"
    ),
    "OTL": AcronymInfo(
        expansion="Output Latch",
        category="ladder_logic",
        description="Latching output instruction"
    ),
    "OTU": AcronymInfo(
        expansion="Output Unlatch",
        category="ladder_logic",
        description="Unlatching output instruction"
    ),
    "TON": AcronymInfo(
        expansion="Timer On-Delay",
        category="ladder_logic",
        description="Delays turning on"
    ),
    "TOF": AcronymInfo(
        expansion="Timer Off-Delay",
        category="ladder_logic",
        description="Delays turning off"
    ),
    "RTO": AcronymInfo(
        expansion="Retentive Timer On",
        category="ladder_logic",
        description="Accumulating timer"
    ),
    "CTU": AcronymInfo(
        expansion="Count Up",
        category="ladder_logic",
        description="Up counter instruction"
    ),
    "CTD": AcronymInfo(
        expansion="Count Down",
        category="ladder_logic",
        description="Down counter instruction"
    ),
    "JSR": AcronymInfo(
        expansion="Jump to Subroutine",
        category="ladder_logic",
        description="Call subroutine/routine"
    ),
    "RET": AcronymInfo(
        expansion="Return",
        category="ladder_logic",
        description="Return from subroutine"
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
    **CONTROLLER_ACRONYMS,
    **HMI_ACRONYMS,
    **SCIENTIFIC_MOLDING_ACRONYMS,
    **PLC_ACRONYMS,
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
        "controller_terms": len(CONTROLLER_ACRONYMS),
        "hmi_terms": len(HMI_ACRONYMS),
        "scientific_molding_terms": len(SCIENTIFIC_MOLDING_ACRONYMS),
        "plc_terms": len(PLC_ACRONYMS),
    }
