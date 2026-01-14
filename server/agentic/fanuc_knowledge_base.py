"""
FANUC Robot Domain Knowledge Base.

Provides structured knowledge for FANUC robot programming, motion types,
I/O timing, and troubleshooting patterns. Used by the RAG pipeline to
enhance responses with accurate technical information.

Part of the industrial robotics pipeline improvements.

Key Topics:
1. Motion Types (CNT, FINE, CR)
2. I/O Timing Triggers (DB, TB, TA)
3. Output Signal Behavior
4. Speed-Dependent Issues
5. Common Troubleshooting Patterns
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Motion Type Knowledge
# =============================================================================

class MotionType(Enum):
    """FANUC motion termination types."""
    FINE = "fine"          # Full stop at position
    CNT = "cnt"            # Continuous motion (corner smoothing)
    CR = "cr"              # Circular radius (corner rounding)


@dataclass
class MotionTypeInfo:
    """Information about a FANUC motion type."""
    name: str
    description: str
    output_trigger: str
    speed_impact: str
    use_cases: List[str]
    timing_notes: str


MOTION_TYPES: Dict[str, MotionTypeInfo] = {
    "FINE": MotionTypeInfo(
        name="FINE (Full Stop)",
        description=(
            "Robot decelerates and comes to a complete stop at the programmed position. "
            "The servo confirms position is achieved within tolerance before proceeding."
        ),
        output_trigger=(
            "Digital outputs (DO/RO) programmed on the same line execute AFTER the robot "
            "has reached the position and confirmed within tolerance (In-Position check)."
        ),
        speed_impact=(
            "Slowest motion type. Total cycle time increases significantly with multiple "
            "FINE points. However, output timing is PREDICTABLE and position-accurate."
        ),
        use_cases=[
            "Precision pick/place operations",
            "Welding start/end points",
            "Position-critical tool actuation",
            "Dispensing applications",
            "Quality inspection positions"
        ],
        timing_notes=(
            "In-Position time typically 20-50ms depending on servo tuning. "
            "Outputs fire only after In-Position flag is TRUE. "
            "Use FINE when tool must activate at exact position."
        )
    ),
    "CNT": MotionTypeInfo(
        name="CNT (Continuous Path)",
        description=(
            "Robot does NOT stop at programmed position. Instead, it blends the motion "
            "into the next point, creating a smooth corner. CNT value (0-100) controls "
            "the blend radius - higher values = larger corner radius."
        ),
        output_trigger=(
            "Digital outputs (DO/RO) programmed on the same line execute when the robot "
            "APPROACHES the position, NOT when it reaches it. The exact trigger point "
            "depends on the CNT value and robot speed."
        ),
        speed_impact=(
            "Fastest motion type. Eliminates deceleration/acceleration at each point. "
            "However, output timing is UNPREDICTABLE and varies with speed/CNT value."
        ),
        use_cases=[
            "High-speed palletizing",
            "Material handling (non-critical positions)",
            "Arc welding paths (continuous motion)",
            "Painting/coating applications",
            "Path smoothing for cycle time reduction"
        ],
        timing_notes=(
            "WARNING: With CNT motion, outputs may fire 50-200mm BEFORE the robot "
            "reaches the programmed position, depending on speed. At 60%+ override, "
            "this distance increases significantly. Use DB/TB triggers instead."
        )
    ),
    "CR": MotionTypeInfo(
        name="CR (Corner Region)",
        description=(
            "Similar to CNT but uses a fixed radius value in mm instead of percentage. "
            "Provides more consistent corner rounding regardless of point spacing."
        ),
        output_trigger=(
            "Same as CNT - outputs fire during approach, not at position. "
            "Timing varies with speed."
        ),
        speed_impact="Similar to CNT but with fixed radius geometry.",
        use_cases=[
            "Applications requiring consistent corner radii",
            "High-speed paths with known geometry",
            "Arc welding with fixed fillet requirements"
        ],
        timing_notes=(
            "CR values in mm. Common values: CR5=5mm radius, CR10=10mm radius. "
            "Same output timing caveats as CNT apply."
        )
    )
}


# =============================================================================
# I/O Timing Trigger Knowledge
# =============================================================================

@dataclass
class IOTriggerInfo:
    """Information about FANUC I/O timing triggers."""
    name: str
    syntax: str
    description: str
    when_to_use: str
    example: str
    calculation: str


IO_TRIGGERS: Dict[str, IOTriggerInfo] = {
    "DB": IOTriggerInfo(
        name="Distance Before (DB)",
        syntax="DO[n]=ON,DB=x.x",
        description=(
            "Triggers output when robot is a specified DISTANCE (mm) before "
            "reaching the programmed position. Distance is measured along the path."
        ),
        when_to_use=(
            "Use when tool/actuator has known response time and you need position-accurate "
            "actuation regardless of robot speed. Calculate: DB = (response_time * speed)."
        ),
        example=(
            "For a pneumatic valve with 30ms response time at 500mm/s:\n"
            "DB = 0.030s × 500mm/s = 15mm\n"
            "Syntax: DO[1]=ON,DB=15.0"
        ),
        calculation=(
            "DB (mm) = actuator_response_time (sec) × robot_speed (mm/sec)\n"
            "Note: Robot speed varies with override percentage!"
        )
    ),
    "TB": IOTriggerInfo(
        name="Time Before (TB)",
        syntax="DO[n]=ON,TB=x.xx",
        description=(
            "Triggers output a specified TIME (seconds) before the robot would "
            "reach the programmed position at current speed."
        ),
        when_to_use=(
            "Use when actuator response time is fixed and you want automatic "
            "compensation regardless of robot speed. TB = actuator_response_time."
        ),
        example=(
            "For a pneumatic valve with 30ms response time:\n"
            "TB = 0.03 (seconds)\n"
            "Syntax: DO[1]=ON,TB=0.03"
        ),
        calculation=(
            "TB (sec) = actuator_response_time (sec)\n"
            "Robot automatically calculates distance based on current speed."
        )
    ),
    "TA": IOTriggerInfo(
        name="Time After (TA)",
        syntax="DO[n]=ON,TA=x.xx",
        description=(
            "Triggers output a specified TIME (seconds) AFTER the robot reaches "
            "or passes the programmed position."
        ),
        when_to_use=(
            "Use when output should activate after reaching position, but a delay "
            "is needed (e.g., settling time, dwell for dispensing)."
        ),
        example=(
            "For dispensing that needs 100ms settle time:\n"
            "TA = 0.10 (seconds)\n"
            "Syntax: DO[1]=ON,TA=0.10"
        ),
        calculation="TA (sec) = desired_delay_after_position"
    ),
    "PULSE": IOTriggerInfo(
        name="Pulse Output",
        syntax="DO[n]=PULSE,x.xx sec",
        description=(
            "Turns output ON for specified duration, then automatically turns OFF. "
            "Useful for momentary triggers."
        ),
        when_to_use=(
            "Use for momentary actuation (triggers, counters, sensors). "
            "Eliminates need for separate OFF instruction."
        ),
        example="DO[1]=PULSE,0.5sec  (500ms pulse)",
        calculation="Duration = time actuator needs to complete action"
    )
}


# =============================================================================
# Speed-Dependent Troubleshooting Patterns
# =============================================================================

@dataclass
class TroubleshootingPattern:
    """Common troubleshooting pattern for speed-dependent issues."""
    symptom: str
    root_cause: str
    explanation: str
    solutions: List[str]
    technical_details: str
    related_topics: List[str]


TROUBLESHOOTING_PATTERNS: Dict[str, TroubleshootingPattern] = {
    "speed_dependent_output_failure": TroubleshootingPattern(
        symptom=(
            "Robot output (DO/RO) fails to actuate tool/device at high speeds, "
            "but works at low speeds. Failure increases as speed increases."
        ),
        root_cause=(
            "CNT motion combined with position-dependent actuator timing. "
            "At high speeds, output fires too early relative to actual position."
        ),
        explanation=(
            "With CNT motion, outputs fire when the robot APPROACHES the position, "
            "not when it reaches it. At 20% override, the robot may be 20mm away. "
            "At 80% override, the robot may be 100mm+ away when the output fires.\n\n"
            "If the actuator (gripper, valve, tool) takes 20-50ms to respond, "
            "and the robot continues moving during that time, the actual actuation "
            "occurs at a different position than programmed.\n\n"
            "At certain positions (e.g., extended reach, specific orientations), "
            "this timing mismatch causes failure. Other positions may work because "
            "the geometry tolerates the offset."
        ),
        solutions=[
            "SOLUTION 1 (Best): Change motion type to FINE for actuation points. "
            "Robot will stop and confirm position before firing output.",

            "SOLUTION 2: Use Time Before (TB) trigger. Set TB = actuator response time. "
            "Example: RO[1]=ON,TB=0.03 for 30ms valve response.",

            "SOLUTION 3: Use Distance Before (DB) trigger with calculated offset. "
            "DB = response_time × max_expected_speed. Provides margin at all speeds.",

            "SOLUTION 4: Add separate WAIT instruction after output for CNT motions. "
            "Example: RO[1]=ON ; WAIT 0.05sec",

            "SOLUTION 5: Reduce CNT value to decrease corner smoothing, "
            "which reduces the early-trigger distance."
        ],
        technical_details=(
            "I/O Update Rate: FANUC robots typically update I/O every 4-8ms (ITP).\n"
            "Valve Response: Pneumatic valves typically 20-50ms, some up to 100ms.\n"
            "Position Error: At 1000mm/s with 50ms response = 50mm position error.\n"
            "CNT Trigger Distance: Roughly proportional to CNT value and speed.\n\n"
            "Formula for DB calculation:\n"
            "  DB (mm) = valve_response_time (s) × robot_TCP_speed (mm/s)\n"
            "  Example: 0.030s × 600mm/s = 18mm\n\n"
            "Robot TCP speed at override:\n"
            "  Actual_speed = Programmed_speed × (Override% / 100)"
        ),
        related_topics=[
            "Motion termination types",
            "I/O timing triggers",
            "Pneumatic valve response",
            "In-Position check",
            "Servo settling time"
        ]
    ),

    "position_dependent_failure": TroubleshootingPattern(
        symptom=(
            "Tool actuation fails at some robot positions but not others, "
            "even at the same speed."
        ),
        root_cause=(
            "Robot TCP speed varies with joint configuration and distance from base. "
            "Extended positions have different dynamics than retracted positions."
        ),
        explanation=(
            "Robot TCP speed is not uniform - it depends on:\n"
            "1. Distance from robot base (extended = faster at same joint speed)\n"
            "2. Joint configuration (singularities, joint limits)\n"
            "3. Acceleration capabilities vary with payload and position\n\n"
            "At extended positions, the same override percentage results in higher "
            "TCP speeds, amplifying timing issues."
        ),
        solutions=[
            "Use TB (Time Before) trigger instead of DB for automatic speed compensation",
            "Add position-specific offsets for critical actuation points",
            "Consider robot positioning to keep critical operations in optimal zone",
            "Use FINE motion for position-critical actuations"
        ],
        technical_details=(
            "Jacobian matrix determines TCP speed from joint speeds.\n"
            "Near singularities, small joint movements = large TCP movements.\n"
            "Extended reach typically has 20-40% higher TCP speed than retracted."
        ),
        related_topics=[
            "Robot kinematics",
            "Jacobian matrix",
            "Singularities",
            "Payload effects"
        ]
    ),

    "intermittent_output_failure": TroubleshootingPattern(
        symptom=(
            "Output sometimes works, sometimes fails, even at the same position and speed."
        ),
        root_cause=(
            "Marginal timing - the system is on the edge of success/failure, "
            "and small variations cause intermittent issues."
        ),
        explanation=(
            "When output timing is marginal, small variations in:\n"
            "- Valve response time (temperature, wear, pressure)\n"
            "- I/O processing latency\n"
            "- Path planning variations\n"
            "can push the system between success and failure."
        ),
        solutions=[
            "Add timing margin: increase TB/DB values by 20-50%",
            "Check pneumatic pressure and flow rates",
            "Verify valve condition and response time",
            "Consider faster actuator if cycle time is critical"
        ],
        technical_details=(
            "Valve response varies with:\n"
            "- Supply pressure: +10% pressure ≈ -15% response time\n"
            "- Temperature: cold = slower response\n"
            "- Wear: worn seals = inconsistent response\n"
            "- Flow restriction: long hoses/small fittings = slower response"
        ),
        related_topics=[
            "Pneumatic system design",
            "Valve specifications",
            "System reliability",
            "Preventive maintenance"
        ]
    )
}


# =============================================================================
# Robot Output Types Knowledge
# =============================================================================

@dataclass
class OutputTypeInfo:
    """Information about FANUC output types."""
    name: str
    description: str
    timing: str
    use_cases: List[str]


OUTPUT_TYPES: Dict[str, OutputTypeInfo] = {
    "RO": OutputTypeInfo(
        name="Robot Output (RO)",
        description=(
            "Dedicated robot outputs, typically faster I/O response than DO. "
            "Directly connected to robot controller I/O board."
        ),
        timing="Typically 2-4ms update rate, processed at servo level",
        use_cases=[
            "Time-critical tool actuation",
            "Servo-synchronized operations",
            "Welding gun control",
            "High-speed gripper control"
        ]
    ),
    "DO": OutputTypeInfo(
        name="Digital Output (DO)",
        description=(
            "General purpose digital outputs. May be local or remote (fieldbus). "
            "Update rate depends on I/O configuration."
        ),
        timing="Local: 4-8ms, Remote/Fieldbus: 10-50ms depending on network",
        use_cases=[
            "General automation signals",
            "PLC communication",
            "Status indicators",
            "Non-critical device control"
        ]
    ),
    "GO": OutputTypeInfo(
        name="Group Output (GO)",
        description=(
            "Multiple digital outputs treated as a group for binary/analog values. "
            "Useful for sending numeric values to external devices."
        ),
        timing="Same as DO (4-50ms depending on configuration)",
        use_cases=[
            "Part type selection",
            "Numeric data to PLC",
            "Multi-state device control"
        ]
    ),
    "AO": OutputTypeInfo(
        name="Analog Output (AO)",
        description=(
            "Analog voltage or current output (0-10V, 4-20mA). "
            "For proportional control of devices."
        ),
        timing="Typically 10-20ms update rate",
        use_cases=[
            "Welding voltage control",
            "Dispensing flow rate",
            "Speed control of external devices"
        ]
    )
}


# =============================================================================
# Knowledge Query Functions
# =============================================================================

def get_motion_type_info(motion_type: str) -> Optional[MotionTypeInfo]:
    """Get information about a specific motion type."""
    return MOTION_TYPES.get(motion_type.upper())


def get_io_trigger_info(trigger_type: str) -> Optional[IOTriggerInfo]:
    """Get information about a specific I/O trigger."""
    return IO_TRIGGERS.get(trigger_type.upper())


def get_troubleshooting_pattern(pattern_key: str) -> Optional[TroubleshootingPattern]:
    """Get a specific troubleshooting pattern."""
    return TROUBLESHOOTING_PATTERNS.get(pattern_key)


def get_output_type_info(output_type: str) -> Optional[OutputTypeInfo]:
    """Get information about a specific output type."""
    return OUTPUT_TYPES.get(output_type.upper())


def search_knowledge_base(query: str) -> List[Dict[str, Any]]:
    """
    Search the FANUC knowledge base for relevant information.

    Args:
        query: Search query (e.g., "CNT motion output timing")

    Returns:
        List of relevant knowledge entries with scores
    """
    results = []
    query_lower = query.lower()

    # Search motion types
    for key, info in MOTION_TYPES.items():
        score = 0
        if key.lower() in query_lower:
            score += 2.0
        if "motion" in query_lower:
            score += 0.5
        if "output" in query_lower and "output" in info.output_trigger.lower():
            score += 1.0
        if "timing" in query_lower:
            score += 0.5

        if score > 0:
            results.append({
                "type": "motion_type",
                "key": key,
                "content": f"{info.name}\n{info.description}\n\nOutput Trigger: {info.output_trigger}\n\nTiming Notes: {info.timing_notes}",
                "score": score
            })

    # Search I/O triggers
    for key, info in IO_TRIGGERS.items():
        score = 0
        if key.lower() in query_lower:
            score += 2.0
        if "trigger" in query_lower or "timing" in query_lower:
            score += 1.0
        if "distance" in query_lower and key == "DB":
            score += 1.5
        if "time" in query_lower and key in ("TB", "TA"):
            score += 1.5

        if score > 0:
            results.append({
                "type": "io_trigger",
                "key": key,
                "content": f"{info.name}\nSyntax: {info.syntax}\n\n{info.description}\n\nWhen to Use: {info.when_to_use}\n\nExample: {info.example}",
                "score": score
            })

    # Search troubleshooting patterns
    for key, pattern in TROUBLESHOOTING_PATTERNS.items():
        score = 0
        if "speed" in query_lower and "speed" in pattern.symptom.lower():
            score += 1.5
        if "fail" in query_lower and "fail" in pattern.symptom.lower():
            score += 1.0
        if "output" in query_lower and "output" in pattern.symptom.lower():
            score += 1.0
        if "position" in query_lower and "position" in pattern.symptom.lower():
            score += 0.8
        if "troubleshoot" in query_lower:
            score += 0.5

        if score > 0:
            solutions_text = "\n".join(f"- {s}" for s in pattern.solutions)
            results.append({
                "type": "troubleshooting",
                "key": key,
                "content": f"Symptom: {pattern.symptom}\n\nRoot Cause: {pattern.root_cause}\n\nExplanation: {pattern.explanation}\n\nSolutions:\n{solutions_text}\n\nTechnical Details: {pattern.technical_details}",
                "score": score
            })

    # Search output types
    for key, info in OUTPUT_TYPES.items():
        score = 0
        if key.lower() in query_lower:
            score += 2.0
        if "output" in query_lower:
            score += 0.3
        if "ro" in query_lower and key == "RO":
            score += 2.0
        if "do" in query_lower and key == "DO":
            score += 2.0

        if score > 0:
            results.append({
                "type": "output_type",
                "key": key,
                "content": f"{info.name}\n{info.description}\n\nTiming: {info.timing}\n\nUse Cases: {', '.join(info.use_cases)}",
                "score": score
            })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_all_knowledge_summary() -> Dict[str, Any]:
    """Get a summary of all knowledge in the base."""
    return {
        "motion_types": list(MOTION_TYPES.keys()),
        "io_triggers": list(IO_TRIGGERS.keys()),
        "troubleshooting_patterns": list(TROUBLESHOOTING_PATTERNS.keys()),
        "output_types": list(OUTPUT_TYPES.keys()),
        "total_entries": (
            len(MOTION_TYPES) + len(IO_TRIGGERS) +
            len(TROUBLESHOOTING_PATTERNS) + len(OUTPUT_TYPES)
        )
    }


# =============================================================================
# Query Enhancement for Robotics
# =============================================================================

# Robotics-specific query expansion terms
ROBOTICS_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    # FANUC-specific terms
    "cnt": ["continuous motion", "corner smoothing", "cnt100", "cnt50"],
    "fine": ["fine motion", "full stop", "in-position", "positioning accuracy"],
    "ro": ["robot output", "RO signal", "digital output", "tool output"],
    "do": ["digital output", "DO signal", "I/O output"],
    "tb": ["time before", "timing trigger", "anticipation output"],
    "db": ["distance before", "position trigger", "anticipation"],
    "override": ["speed override", "robot speed", "feedrate override"],
    "teach pendant": ["TP", "pendant", "operator panel"],

    # Motion/kinematics
    "tcp": ["tool center point", "end effector", "tool tip"],
    "singularity": ["singular position", "wrist flip", "axis alignment"],
    "payload": ["load capacity", "weight capacity", "inertia"],

    # Pneumatics
    "pneumatic": ["air cylinder", "pneumatic actuator", "air valve"],
    "solenoid": ["solenoid valve", "directional valve", "pneumatic valve"],
    "gripper": ["end effector", "gripper fingers", "pneumatic gripper"],

    # Alarms/errors
    "srvo": ["servo alarm", "servo error", "SRVO-"],
    "motn": ["motion alarm", "motion error", "MOTN-"],
    "syst": ["system alarm", "system error", "SYST-"],

    # Programming
    "karel": ["KAREL program", "PC program", "robot programming"],
    "tp program": ["teach pendant program", "motion program", "job program"],
    "macro": ["macro program", "subprogram", "utility program"]
}


def expand_robotics_query(query: str) -> str:
    """
    Expand a robotics query with domain-specific synonyms.

    Args:
        query: Original search query

    Returns:
        Expanded query with additional terms
    """
    query_lower = query.lower()
    expansions = []

    for term, synonyms in ROBOTICS_QUERY_EXPANSIONS.items():
        if term in query_lower:
            # Add first synonym that's not already in query
            for synonym in synonyms[:2]:  # Limit to 2 expansions per term
                if synonym.lower() not in query_lower:
                    expansions.append(synonym)
                    break

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# =============================================================================
# Singleton and Initialization
# =============================================================================

_knowledge_base_initialized = False


def initialize_knowledge_base() -> bool:
    """Initialize the FANUC knowledge base."""
    global _knowledge_base_initialized

    if _knowledge_base_initialized:
        return True

    logger.info("Initializing FANUC knowledge base...")
    logger.info(f"  Motion types: {len(MOTION_TYPES)}")
    logger.info(f"  I/O triggers: {len(IO_TRIGGERS)}")
    logger.info(f"  Troubleshooting patterns: {len(TROUBLESHOOTING_PATTERNS)}")
    logger.info(f"  Output types: {len(OUTPUT_TYPES)}")
    logger.info(f"  Query expansions: {len(ROBOTICS_QUERY_EXPANSIONS)}")

    _knowledge_base_initialized = True
    return True


# Initialize on import
initialize_knowledge_base()
