"""
Domain-Specific Benchmark Suite for Agentic Search

Provides systematic quality measurement for industrial automation queries,
with focus on FANUC robotics troubleshooting.

Components:
1. FANUC_BENCHMARK: 70 test cases with expected entities and domains (G.1.4)
2. TechnicalAccuracyScorer: Multi-dimensional answer quality assessment
3. BenchmarkRunner: Automated test execution and reporting

Usage:
    from agentic.benchmark import (
        run_benchmark,
        TechnicalAccuracyScorer,
        FANUC_BENCHMARK
    )

    # Run full benchmark
    results = await run_benchmark(orchestrator, preset="balanced")

    # Score a single answer
    scorer = TechnicalAccuracyScorer()
    score = scorer.score(answer, ground_truth, query)

Author: Claude Code
Date: December 2025
"""

import re
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================
# BENCHMARK DATA STRUCTURES
# ============================================

class QueryDifficulty(str, Enum):
    """Difficulty level for benchmark queries."""
    EASY = "easy"           # Single error code, direct answer
    MEDIUM = "medium"       # Multiple aspects, some reasoning
    HARD = "hard"           # Complex troubleshooting, multiple causes
    EXPERT = "expert"       # Requires deep domain knowledge


class QueryCategory(str, Enum):
    """Category of benchmark query."""
    ERROR_CODE = "error_code"           # Specific alarm/error lookup
    TROUBLESHOOTING = "troubleshooting" # Problem diagnosis
    PROCEDURE = "procedure"             # Step-by-step instructions
    COMPARISON = "comparison"           # Compare options/approaches
    CONCEPTUAL = "conceptual"           # Explain concept/theory
    PARAMETER = "parameter"             # System settings/configuration


@dataclass
class BenchmarkQuery:
    """A single benchmark test case."""
    query: str
    category: QueryCategory
    difficulty: QueryDifficulty
    expected_entities: List[str]        # Must appear in answer
    expected_domains: List[str]         # Preferred source domains
    required_concepts: List[str]        # Key concepts to cover
    safety_critical: bool = False       # Must include safety warnings
    ground_truth_summary: str = ""      # Brief expected answer
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark query."""
    query: BenchmarkQuery
    answer: str
    sources: List[str]
    confidence: float
    execution_time_ms: int
    entity_coverage: float              # 0-1, entities found
    domain_match_rate: float            # 0-1, sources from expected domains
    concept_coverage: float             # 0-1, concepts covered
    safety_present: bool                # Safety warnings included
    technical_accuracy: float           # 0-1, overall accuracy
    passed: bool                        # Met minimum thresholds


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results."""
    total_queries: int
    passed_queries: int
    pass_rate: float
    avg_confidence: float
    avg_entity_coverage: float
    avg_domain_match: float
    avg_concept_coverage: float
    avg_technical_accuracy: float
    avg_execution_time_ms: float
    by_category: Dict[str, Dict[str, float]]
    by_difficulty: Dict[str, Dict[str, float]]
    timestamp: str
    preset: str
    results: List[BenchmarkResult]


# ============================================
# FANUC BENCHMARK TEST CASES
# ============================================

# Realistic FANUC-related domains that web searches actually return
# These include official sources, forums, integrators, and documentation aggregators
FANUC_TRUSTED_DOMAINS: List[str] = [
    # Official FANUC sources
    "fanucamerica.com",
    "techtransfer.fanucamerica.com",
    "crc2.frc.com",
    "fanuc-academy.uk",
    # Industrial robotics forums
    "robot-forum.com",
    "plctalk.net",
    "practicalmachinist.com",
    "eng-tips.com",
    "forum.diy-robotics.com",
    # FANUC service providers/integrators
    "2rirobotics.com",
    "aerobotix.net",
    "robotic.support",
    "robotworx.com",
    # Documentation aggregators
    "everythingaboutrobots.com",
    "manualslib.com",
    "pdfcoffee.com",
    # Technical communities
    "reddit.com",
    "emastercam.com",
    "cnczone.com",
    # General technical
    "stackoverflow.com",
    "electronics.stackexchange.com",
    "robotics.stackexchange.com",
]

FANUC_BENCHMARK: List[BenchmarkQuery] = [
    # === EASY: Direct Error Code Lookups ===
    BenchmarkQuery(
        query="What causes SRVO-063 alarm?",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["SRVO-063", "encoder", "pulsecoder", "calibration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["encoder failure", "RCAL", "mastering"],
        ground_truth_summary="SRVO-063 is a servo alarm indicating encoder/pulsecoder issues requiring recalibration",
        tags=["servo", "encoder", "calibration"]
    ),
    BenchmarkQuery(
        query="MOTN-023 alarm meaning",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["MOTN-023", "motion", "position"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion planning", "position error"],
        ground_truth_summary="MOTN-023 is a motion alarm related to position/trajectory planning",
        tags=["motion", "trajectory"]
    ),
    BenchmarkQuery(
        query="SYST-100 system alarm",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["SYST-100", "system", "controller"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["system error", "controller"],
        ground_truth_summary="SYST-100 is a system-level alarm in the controller",
        tags=["system", "controller"]
    ),
    BenchmarkQuery(
        query="HOST-001 communication error",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["HOST-001", "communication", "network"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["host communication", "ethernet", "timeout"],
        ground_truth_summary="HOST-001 indicates host communication failure",
        tags=["communication", "network"]
    ),
    BenchmarkQuery(
        query="INTP-127 interpreter alarm",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["INTP-127", "interpreter", "program"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["program error", "syntax", "execution"],
        ground_truth_summary="INTP-127 is an interpreter alarm during program execution",
        tags=["interpreter", "program"]
    ),

    # === MEDIUM: Troubleshooting Queries ===
    BenchmarkQuery(
        query="Robot arm jerking during motion, possible causes?",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["servo", "motor", "encoder", "vibration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["servo tuning", "mechanical wear", "encoder feedback"],
        ground_truth_summary="Jerking motion can be caused by servo tuning, encoder issues, or mechanical wear",
        tags=["motion", "servo", "mechanical"]
    ),
    BenchmarkQuery(
        query="How to diagnose intermittent servo overcurrent",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["SRVO", "overcurrent", "amplifier", "motor"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["current monitoring", "amplifier", "motor load"],
        safety_critical=True,
        ground_truth_summary="Intermittent overcurrent requires checking amplifier, motor, cabling, and load conditions",
        tags=["servo", "overcurrent", "diagnosis"]
    ),
    BenchmarkQuery(
        query="Vision system drift calibration issues",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["iRVision", "camera", "calibration", "drift"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["vision calibration", "camera mounting", "lighting"],
        ground_truth_summary="Vision drift can be caused by camera mounting, lighting, or calibration decay",
        tags=["vision", "calibration", "iRVision"]
    ),
    BenchmarkQuery(
        query="Robot not finding home position",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["home", "position", "mastering", "encoder"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["mastering", "reference position", "encoder reset"],
        ground_truth_summary="Home position issues typically require remastering or encoder reference reset",
        tags=["mastering", "home", "position"]
    ),
    BenchmarkQuery(
        query="Payload identification failure",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["payload", "identification", "inertia", "mass"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["payload calculation", "inertia estimation", "tool weight"],
        ground_truth_summary="Payload identification requires proper motion conditions and tool mounting",
        tags=["payload", "calibration", "performance"]
    ),

    # === MEDIUM: Procedure Queries ===
    BenchmarkQuery(
        query="Steps to perform RCAL calibration",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["RCAL", "calibration", "encoder", "mastering"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["calibration procedure", "encoder reset", "reference position"],
        safety_critical=True,
        ground_truth_summary="RCAL is performed via MENU > SYSTEM > MASTER/CAL with specific axis motions",
        tags=["calibration", "procedure", "RCAL"]
    ),
    BenchmarkQuery(
        query="How to backup and restore robot programs",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["backup", "restore", "program", "USB", "memory"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["file management", "USB transfer", "program files"],
        ground_truth_summary="Backup via FILE > BACKUP, restore via FILE > RESTORE using USB or network",
        tags=["backup", "restore", "program"]
    ),
    BenchmarkQuery(
        query="Zero position mastering procedure",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["zero", "mastering", "position", "fixture"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["mastering fixture", "zero position", "encoder reset"],
        safety_critical=True,
        ground_truth_summary="Zero mastering uses fixtures to align axes at known positions",
        tags=["mastering", "zero", "procedure"]
    ),
    BenchmarkQuery(
        query="Setting up user frames",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["user frame", "coordinate", "TCP", "teaching"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["coordinate system", "three-point method", "frame setup"],
        ground_truth_summary="User frames defined via three-point or four-point teaching method",
        tags=["frames", "coordinates", "setup"]
    ),
    BenchmarkQuery(
        query="Tool center point TCP calibration",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["TCP", "tool", "calibration", "offset"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["TCP definition", "six-point method", "tool offset"],
        ground_truth_summary="TCP calibration via six-point method touching reference point from different angles",
        tags=["TCP", "tool", "calibration"]
    ),

    # === HARD: Complex Troubleshooting ===
    BenchmarkQuery(
        query="Multiple axis collision detection triggering falsely",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["collision", "detection", "ACAL", "sensitivity", "torque"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["collision detection", "sensitivity tuning", "payload", "friction"],
        safety_critical=True,
        ground_truth_summary="False collision detection requires tuning sensitivity, payload verification, and friction compensation",
        tags=["collision", "safety", "tuning"]
    ),
    BenchmarkQuery(
        query="DCS Safe Position monitoring errors",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["DCS", "safe position", "safety", "monitoring"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["DCS configuration", "safe zones", "position monitoring"],
        safety_critical=True,
        ground_truth_summary="DCS Safe Position errors require checking configuration, encoder accuracy, and zone definitions",
        tags=["DCS", "safety", "monitoring"]
    ),
    BenchmarkQuery(
        query="Servo motor overheating under continuous operation",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["motor", "overheating", "temperature", "duty cycle"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["thermal management", "duty cycle", "current limiting", "cooling"],
        safety_critical=True,
        ground_truth_summary="Motor overheating can be caused by excessive duty cycle, undersized motor, or cooling issues",
        tags=["thermal", "motor", "performance"]
    ),
    BenchmarkQuery(
        query="Sporadic communication drops with PLC via EtherNet/IP",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["EtherNet/IP", "communication", "PLC", "network", "timeout"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["network diagnostics", "RPI settings", "timeout configuration"],
        ground_truth_summary="Sporadic drops require checking network infrastructure, RPI timing, and timeout settings",
        tags=["network", "communication", "PLC"]
    ),
    BenchmarkQuery(
        query="Cycle time optimization for pick and place application",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["cycle time", "speed", "acceleration", "CNT", "motion"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion optimization", "CNT settings", "speed override", "path planning"],
        ground_truth_summary="Cycle time optimization involves CNT settings, speed adjustments, and path optimization",
        tags=["performance", "optimization", "motion"]
    ),

    # === EXPERT: Deep Domain Knowledge ===
    BenchmarkQuery(
        query="KAREL program memory management best practices",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["KAREL", "memory", "program", "variables", "allocation"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["memory allocation", "variable scope", "program structure"],
        ground_truth_summary="KAREL memory management involves proper variable scoping, array sizing, and routine structure",
        tags=["KAREL", "programming", "advanced"]
    ),
    BenchmarkQuery(
        query="Servo gain tuning for high-speed applications",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["servo", "gain", "tuning", "SVGN", "bandwidth"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["position gain", "velocity gain", "feedforward", "stability"],
        safety_critical=True,
        ground_truth_summary="Servo tuning involves adjusting position, velocity, and integral gains with stability monitoring",
        tags=["servo", "tuning", "performance"]
    ),
    BenchmarkQuery(
        query="Multi-robot coordination and interference avoidance",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["multi-robot", "coordination", "interference", "zone", "semaphore"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["zone monitoring", "interference regions", "synchronization"],
        safety_critical=True,
        ground_truth_summary="Multi-robot coordination uses zones, semaphores, and interference checking for safe operation",
        tags=["multi-robot", "coordination", "safety"]
    ),
    BenchmarkQuery(
        query="iRVision 3D sensor integration and calibration",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["iRVision", "3D", "sensor", "calibration", "point cloud"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["3D calibration", "point cloud processing", "sensor mounting"],
        ground_truth_summary="3D vision integration requires sensor calibration, mounting alignment, and point cloud configuration",
        tags=["vision", "3D", "integration"]
    ),
    BenchmarkQuery(
        query="Conveyor tracking synchronization issues",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["conveyor", "tracking", "encoder", "synchronization", "line tracking"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["line tracking", "encoder feedback", "trigger timing"],
        ground_truth_summary="Conveyor tracking issues involve encoder setup, trigger timing, and synchronization window",
        tags=["conveyor", "tracking", "synchronization"]
    ),

    # === COMPARISON Queries ===
    BenchmarkQuery(
        query="R-30iA vs R-30iB controller differences",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["R-30iA", "R-30iB", "controller", "features"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["controller generations", "feature comparison", "compatibility"],
        ground_truth_summary="R-30iB has improved processing, safety features, and network capabilities over R-30iA",
        tags=["controller", "comparison", "features"]
    ),
    BenchmarkQuery(
        query="Collaborative robot vs standard industrial robot applications",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["collaborative", "cobot", "industrial", "safety"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["safety requirements", "payload", "speed", "application suitability"],
        safety_critical=True,
        ground_truth_summary="Cobots suit human-collaborative work with lower payloads; industrial robots for high-speed isolated work",
        tags=["cobot", "comparison", "safety"]
    ),

    # === PARAMETER Queries ===
    BenchmarkQuery(
        query="$MCR_GRP system variables explained",
        category=QueryCategory.PARAMETER,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["$MCR_GRP", "system variable", "configuration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["system variables", "group configuration", "motion control"],
        ground_truth_summary="$MCR_GRP controls motion group settings including acceleration and deceleration",
        tags=["parameters", "system variables"]
    ),
    BenchmarkQuery(
        query="Adjusting $SPEED and $MOTYPE parameters",
        category=QueryCategory.PARAMETER,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["$SPEED", "$MOTYPE", "motion", "override"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["speed override", "motion type", "program control"],
        ground_truth_summary="$SPEED controls motion speed override; $MOTYPE sets motion type (joint/linear)",
        tags=["parameters", "motion", "speed"]
    ),

    # === Additional Edge Cases ===
    BenchmarkQuery(
        query="Robot stops randomly with no alarm",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["stop", "pause", "signal", "interlock"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["external signals", "interlocks", "hold conditions"],
        ground_truth_summary="Random stops without alarms typically caused by external hold signals or interlocks",
        tags=["troubleshooting", "signals", "interlocks"]
    ),
    BenchmarkQuery(
        query="Battery low warning recovery procedure",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["battery", "backup", "encoder", "memory"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["battery replacement", "data backup", "encoder position"],
        safety_critical=True,
        ground_truth_summary="Replace batteries while powered to preserve encoder positions; backup data first",
        tags=["maintenance", "battery", "backup"]
    ),

    # ============================================
    # EXPANDED BENCHMARK SET (G.1.4 Golden Queries)
    # Total target: 70 queries
    # ============================================

    # === EASY: Additional Error Codes ===
    BenchmarkQuery(
        query="PRIO-001 priority error alarm",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["PRIO-001", "priority", "task"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["task priority", "system resources"],
        ground_truth_summary="PRIO-001 indicates task priority scheduling conflict",
        tags=["priority", "task", "system"]
    ),
    BenchmarkQuery(
        query="TOOL-001 tool frame error",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["TOOL-001", "tool", "frame"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["tool frame", "TCP offset"],
        ground_truth_summary="TOOL-001 indicates tool frame definition error",
        tags=["tool", "frame", "tcp"]
    ),
    BenchmarkQuery(
        query="FILE-032 file access error",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["FILE-032", "file", "access", "storage"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["file system", "memory card", "storage"],
        ground_truth_summary="FILE-032 indicates file access or storage error",
        tags=["file", "storage", "memory"]
    ),
    BenchmarkQuery(
        query="SRVO-001 servo failure alarm",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["SRVO-001", "servo", "overcurrent"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["servo overcurrent", "motor protection"],
        ground_truth_summary="SRVO-001 indicates servo overcurrent protection triggered",
        tags=["servo", "overcurrent", "protection"]
    ),
    BenchmarkQuery(
        query="MOTN-063 motion error alarm",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["MOTN-063", "motion", "trajectory"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion planning", "trajectory error"],
        ground_truth_summary="MOTN-063 indicates trajectory planning or execution error",
        tags=["motion", "trajectory"]
    ),
    BenchmarkQuery(
        query="CVIS-010 vision communication error",
        category=QueryCategory.ERROR_CODE,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["CVIS-010", "vision", "communication"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["iRVision", "camera communication"],
        ground_truth_summary="CVIS-010 indicates vision system communication failure",
        tags=["vision", "communication"]
    ),

    # === MEDIUM: Additional Troubleshooting ===
    BenchmarkQuery(
        query="Robot vibration at specific positions",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["vibration", "resonance", "servo", "position"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["resonance", "servo tuning", "mechanical coupling"],
        ground_truth_summary="Position-specific vibration often caused by resonance or servo tuning issues",
        tags=["vibration", "resonance", "servo"]
    ),
    BenchmarkQuery(
        query="Arc welding quality issues inconsistent bead",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["welding", "arc", "bead", "quality"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["weld parameters", "torch angle", "travel speed"],
        ground_truth_summary="Inconsistent weld bead caused by speed, angle, or parameter variations",
        tags=["welding", "arc", "quality"]
    ),
    BenchmarkQuery(
        query="Robot slow response to teach pendant commands",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["teach pendant", "response", "communication", "lag"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["pendant communication", "system load", "cable"],
        ground_truth_summary="Slow response may be due to communication issues or system overload",
        tags=["teach pendant", "communication", "response"]
    ),
    BenchmarkQuery(
        query="Gripper not opening or closing properly",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["gripper", "pneumatic", "IO", "signal"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["I/O signals", "pneumatic pressure", "solenoid"],
        ground_truth_summary="Gripper issues caused by I/O, pneumatics, or mechanical problems",
        tags=["gripper", "pneumatic", "io"]
    ),
    BenchmarkQuery(
        query="Spot welding electrode wear compensation",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["spot welding", "electrode", "wear", "compensation"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["tip dress", "wear compensation", "electrode life"],
        ground_truth_summary="Electrode wear requires tip dressing and automatic compensation setup",
        tags=["welding", "spot", "electrode"]
    ),

    # === MEDIUM: Additional Procedures ===
    BenchmarkQuery(
        query="How to configure digital I/O signals",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["digital", "I/O", "signal", "configuration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["I/O assignment", "rack/slot", "signal mapping"],
        ground_truth_summary="Digital I/O configured via MENU > I/O > Digital with rack/slot assignment",
        tags=["io", "digital", "configuration"]
    ),
    BenchmarkQuery(
        query="Steps to create macro program",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["macro", "program", "MR", "routine"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["macro programming", "call instruction", "arguments"],
        ground_truth_summary="Macros created as MR[ ] programs and called with CALL instruction",
        tags=["macro", "programming"]
    ),
    BenchmarkQuery(
        query="Palletizing setup and configuration",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["palletizing", "pallet", "pattern", "stack"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["pallet pattern", "stack configuration", "index calculation"],
        ground_truth_summary="Palletizing uses PalletTool option with pattern and layer configuration",
        tags=["palletizing", "application", "setup"]
    ),
    BenchmarkQuery(
        query="How to set up robot interference zones",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["interference", "zone", "space", "check"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["space definition", "zone checking", "interference avoidance"],
        safety_critical=True,
        ground_truth_summary="Interference zones defined via MENU > SETUP > Space with box/plane regions",
        tags=["interference", "safety", "zones"]
    ),

    # === MEDIUM: Additional Parameters ===
    BenchmarkQuery(
        query="$SPEED_CTRL system variables for motion",
        category=QueryCategory.PARAMETER,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["$SPEED_CTRL", "system variable", "motion"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["speed control", "motion limits"],
        ground_truth_summary="$SPEED_CTRL controls motion speed limits and acceleration profiles",
        tags=["parameters", "speed", "motion"]
    ),
    BenchmarkQuery(
        query="$SAFE_IO configuration for safety signals",
        category=QueryCategory.PARAMETER,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["$SAFE_IO", "safety", "signal", "configuration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["safety I/O", "dual-channel", "safety signals"],
        safety_critical=True,
        ground_truth_summary="$SAFE_IO configures safety-rated I/O for DCS and external safety",
        tags=["safety", "io", "parameters"]
    ),
    BenchmarkQuery(
        query="$COLL_DET collision detection parameters",
        category=QueryCategory.PARAMETER,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["$COLL_DET", "collision", "detection", "sensitivity"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["collision sensitivity", "torque threshold", "detection level"],
        safety_critical=True,
        ground_truth_summary="$COLL_DET controls collision detection sensitivity and response",
        tags=["collision", "safety", "parameters"]
    ),

    # === HARD: Complex Scenarios ===
    BenchmarkQuery(
        query="Multi-group coordinated motion synchronization issues",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["multi-group", "coordinated", "motion", "synchronization"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["group coordination", "sync timing", "master-slave"],
        ground_truth_summary="Multi-group sync requires proper group configuration and motion timing",
        tags=["multi-group", "coordination", "motion"]
    ),
    BenchmarkQuery(
        query="External axis setup and calibration with positioner",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["external axis", "positioner", "calibration", "E1"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["extended axis", "positioner setup", "kinematics"],
        ground_truth_summary="External axes require group configuration, calibration, and coordinated motion setup",
        tags=["external axis", "positioner", "calibration"]
    ),
    BenchmarkQuery(
        query="Robot path accuracy degradation over time",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["path", "accuracy", "degradation", "calibration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["mechanical wear", "calibration drift", "thermal effects"],
        ground_truth_summary="Path accuracy degradation caused by wear, thermal effects, or calibration drift",
        tags=["accuracy", "wear", "calibration"]
    ),
    BenchmarkQuery(
        query="Force sensing integration for assembly tasks",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["force", "sensing", "assembly", "compliance"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["force control", "compliance", "insertion"],
        ground_truth_summary="Force sensing uses Force Sensor package with compliance and search patterns",
        tags=["force", "sensing", "assembly"]
    ),
    BenchmarkQuery(
        query="Arc welding seam tracking adaptive control",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["seam tracking", "arc", "adaptive", "TAST"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["TAST", "seam tracking", "adaptive control"],
        ground_truth_summary="Seam tracking uses TAST or vision for real-time path correction",
        tags=["welding", "seam tracking", "adaptive"]
    ),
    BenchmarkQuery(
        query="Robot cell network architecture design",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["network", "EtherNet/IP", "architecture", "PLC"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["network topology", "protocol selection", "scan rate"],
        ground_truth_summary="Cell network uses EtherNet/IP or DeviceNet with proper topology design",
        tags=["network", "architecture", "integration"]
    ),

    # === EXPERT: Additional Deep Knowledge ===
    BenchmarkQuery(
        query="Custom inverse kinematics for non-standard robot configurations",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["inverse kinematics", "configuration", "IK", "joint"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["kinematics", "joint solutions", "configuration space"],
        ground_truth_summary="Custom IK requires understanding of robot geometry and configuration options",
        tags=["kinematics", "advanced", "configuration"]
    ),
    BenchmarkQuery(
        query="Robot simulation to production path matching",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["simulation", "production", "offline", "calibration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["OLP calibration", "cell calibration", "path verification"],
        ground_truth_summary="Simulation matching requires calibration of virtual cell to physical cell",
        tags=["simulation", "offline", "calibration"]
    ),
    BenchmarkQuery(
        query="High-speed motion optimization for material handling",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["high-speed", "motion", "optimization", "payload"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion profile", "CNT optimization", "acceleration limits"],
        ground_truth_summary="High-speed optimization involves motion profiles, CNT values, and payload balancing",
        tags=["high-speed", "optimization", "performance"]
    ),
    BenchmarkQuery(
        query="Robot cell digital twin synchronization",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["digital twin", "synchronization", "OPC-UA", "real-time"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["digital twin", "data synchronization", "real-time monitoring"],
        ground_truth_summary="Digital twin uses OPC-UA or MQTT for real-time state synchronization",
        tags=["digital twin", "industry 4.0", "monitoring"]
    ),
    BenchmarkQuery(
        query="Servo drive tuning for high-inertia loads",
        category=QueryCategory.PROCEDURE,
        difficulty=QueryDifficulty.EXPERT,
        expected_entities=["servo", "tuning", "inertia", "gain"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["inertia ratio", "gain tuning", "stability margins"],
        safety_critical=True,
        ground_truth_summary="High-inertia loads require adjusted servo gains and inertia compensation",
        tags=["servo", "tuning", "inertia"]
    ),

    # === COMPARISON: Additional Comparisons ===
    BenchmarkQuery(
        query="Joint motion J vs linear motion L differences",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["joint", "linear", "motion", "J", "L"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion types", "interpolation", "path control"],
        ground_truth_summary="J motion moves joints directly; L motion moves TCP linearly",
        tags=["motion", "comparison", "basic"]
    ),
    BenchmarkQuery(
        query="ROBOGUIDE vs NC Builder simulation tools",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["ROBOGUIDE", "NC Builder", "simulation", "offline"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["simulation software", "offline programming", "features"],
        ground_truth_summary="ROBOGUIDE for robots, NC Builder for CNC; different target machines",
        tags=["simulation", "software", "comparison"]
    ),
    BenchmarkQuery(
        query="DeviceNet vs EtherNet/IP for robot communication",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["DeviceNet", "EtherNet/IP", "communication", "fieldbus"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["fieldbus", "protocol comparison", "bandwidth"],
        ground_truth_summary="EtherNet/IP faster and more common; DeviceNet older but still used",
        tags=["communication", "fieldbus", "comparison"]
    ),
    BenchmarkQuery(
        query="Single vs dual check safety configurations",
        category=QueryCategory.COMPARISON,
        difficulty=QueryDifficulty.HARD,
        expected_entities=["single check", "dual check", "safety", "DCS"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["safety categories", "redundancy", "SIL"],
        safety_critical=True,
        ground_truth_summary="Dual check provides redundancy for higher safety categories (PLd/Cat3+)",
        tags=["safety", "DCS", "comparison"]
    ),

    # === CONCEPTUAL: Additional Theory ===
    BenchmarkQuery(
        query="Robot singularity and wrist flip explained",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["singularity", "wrist flip", "joint", "configuration"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["singularity", "configuration", "arm solution"],
        ground_truth_summary="Singularity occurs when robot loses a degree of freedom; wrist flip at J5=0",
        tags=["singularity", "kinematics", "concept"]
    ),
    BenchmarkQuery(
        query="What is FINE vs CNT termination type",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.EASY,
        expected_entities=["FINE", "CNT", "termination", "motion"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["motion termination", "path blending", "position accuracy"],
        ground_truth_summary="FINE stops at position; CNT blends through for faster cycle time",
        tags=["motion", "termination", "concept"]
    ),
    BenchmarkQuery(
        query="RPI and implicit messaging in EtherNet/IP",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["RPI", "implicit messaging", "EtherNet/IP", "I/O"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["Requested Packet Interval", "cyclic data", "I/O connections"],
        ground_truth_summary="RPI sets cyclic data rate; implicit messaging for real-time I/O exchange",
        tags=["communication", "EtherNet/IP", "concept"]
    ),
    BenchmarkQuery(
        query="Robot arm reach and work envelope calculation",
        category=QueryCategory.CONCEPTUAL,
        difficulty=QueryDifficulty.MEDIUM,
        expected_entities=["reach", "work envelope", "workspace", "kinematics"],
        expected_domains=FANUC_TRUSTED_DOMAINS,
        required_concepts=["robot workspace", "reach calculation", "interference zones"],
        ground_truth_summary="Work envelope determined by robot kinematics and joint limits",
        tags=["workspace", "reach", "concept"]
    ),
]


# ============================================
# TECHNICAL ACCURACY SCORER
# ============================================

class TechnicalAccuracyScorer:
    """
    Multi-dimensional scorer for technical answer quality.

    Evaluates:
    1. Entity coverage - Required technical terms present
    2. Procedure completeness - Steps are complete and ordered
    3. Safety warnings - Critical safety info included
    4. Technical term accuracy - Terms used correctly
    5. Concept coverage - Key concepts addressed
    """

    # Safety-related keywords to check
    SAFETY_KEYWORDS = [
        "safety", "caution", "warning", "danger", "hazard",
        "lockout", "tagout", "LOTO", "e-stop", "emergency",
        "power off", "de-energize", "protective", "guard",
        "before", "ensure", "verify", "check", "confirm"
    ]

    # Procedure indicator patterns
    PROCEDURE_PATTERNS = [
        r"\b(step\s*\d+|first|second|third|then|next|finally)\b",
        r"\b(press|select|navigate|go to|click|choose)\b",
        r"\b(menu|screen|option|button|key)\b",
        r"\d+\.\s+\w+",  # Numbered steps
    ]

    def __init__(self):
        self.compiled_procedure_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROCEDURE_PATTERNS
        ]

    def score(
        self,
        answer: str,
        benchmark: BenchmarkQuery,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Score a technical answer against benchmark expectations.

        Args:
            answer: Generated answer text
            benchmark: Benchmark query with expectations
            sources: Optional list of source URLs/domains

        Returns:
            Dict with individual scores and overall accuracy
        """
        answer_lower = answer.lower()

        # 1. Entity Coverage
        entity_coverage = self._score_entity_coverage(
            answer_lower, benchmark.expected_entities
        )

        # 2. Concept Coverage
        concept_coverage = self._score_concept_coverage(
            answer_lower, benchmark.required_concepts
        )

        # 3. Domain Match (if sources provided)
        domain_match = self._score_domain_match(
            sources, benchmark.expected_domains
        ) if sources else 0.5

        # 4. Safety Presence (if safety-critical)
        safety_present = self._check_safety_presence(answer_lower)
        safety_score = 1.0 if not benchmark.safety_critical else (
            1.0 if safety_present else 0.3
        )

        # 5. Procedure Completeness (for procedure queries)
        procedure_score = 1.0
        if benchmark.category == QueryCategory.PROCEDURE:
            procedure_score = self._score_procedure_completeness(answer)

        # 6. Technical Term Accuracy (basic check)
        term_accuracy = self._score_term_accuracy(answer, benchmark)

        # Calculate overall accuracy (weighted)
        overall = (
            entity_coverage * 0.25 +
            concept_coverage * 0.25 +
            domain_match * 0.15 +
            safety_score * 0.15 +
            procedure_score * 0.10 +
            term_accuracy * 0.10
        )

        return {
            "entity_coverage": entity_coverage,
            "concept_coverage": concept_coverage,
            "domain_match": domain_match,
            "safety_present": safety_present,
            "safety_score": safety_score,
            "procedure_completeness": procedure_score,
            "term_accuracy": term_accuracy,
            "overall_accuracy": overall,
            "passed": overall >= 0.6 and (not benchmark.safety_critical or safety_present)
        }

    def _score_entity_coverage(
        self,
        answer: str,
        expected_entities: List[str]
    ) -> float:
        """Calculate what fraction of expected entities appear in answer."""
        if not expected_entities:
            return 1.0

        found = 0
        for entity in expected_entities:
            # Check for entity or close variations
            entity_lower = entity.lower()
            if entity_lower in answer:
                found += 1
            # Check without hyphens/spaces
            elif entity_lower.replace("-", "").replace(" ", "") in answer.replace("-", "").replace(" ", ""):
                found += 0.8

        return found / len(expected_entities)

    def _score_concept_coverage(
        self,
        answer: str,
        required_concepts: List[str]
    ) -> float:
        """Calculate coverage of required concepts."""
        if not required_concepts:
            return 1.0

        found = 0
        for concept in required_concepts:
            concept_lower = concept.lower()
            # Check exact match
            if concept_lower in answer:
                found += 1
            else:
                # Check individual words (at least half must match)
                words = concept_lower.split()
                matched = sum(1 for w in words if w in answer)
                if matched >= len(words) / 2:
                    found += 0.7

        return found / len(required_concepts)

    def _score_domain_match(
        self,
        sources: List[str],
        expected_domains: List[str]
    ) -> float:
        """Calculate what fraction of sources are from expected domains."""
        if not sources or not expected_domains:
            return 0.5

        expected_lower = [d.lower() for d in expected_domains]
        matched = 0

        for source in sources:
            source_lower = source.lower()
            for domain in expected_lower:
                if domain in source_lower:
                    matched += 1
                    break

        return matched / len(sources)

    def _check_safety_presence(self, answer: str) -> bool:
        """Check if safety-related keywords are present."""
        return any(kw in answer for kw in self.SAFETY_KEYWORDS)

    def _score_procedure_completeness(self, answer: str) -> float:
        """Score completeness of procedural answers."""
        matches = 0
        for pattern in self.compiled_procedure_patterns:
            if pattern.search(answer):
                matches += 1

        # Normalize: having 2+ procedure indicators is good
        return min(1.0, matches / 2)

    def _score_term_accuracy(
        self,
        answer: str,
        benchmark: BenchmarkQuery
    ) -> float:
        """Basic check for technical term accuracy."""
        # Check if query terms appear in answer (basic relevance)
        query_terms = set(benchmark.query.lower().split())
        answer_terms = set(answer.lower().split())

        # Remove common words
        common = {"the", "a", "an", "is", "are", "what", "how", "to", "for", "and", "or"}
        query_terms -= common

        if not query_terms:
            return 1.0

        overlap = len(query_terms & answer_terms)
        return overlap / len(query_terms)


# ============================================
# BENCHMARK RUNNER
# ============================================

class BenchmarkRunner:
    """
    Runs benchmark test suite against an orchestrator.
    """

    def __init__(
        self,
        scorer: Optional[TechnicalAccuracyScorer] = None,
        min_pass_threshold: float = 0.6
    ):
        self.scorer = scorer or TechnicalAccuracyScorer()
        self.min_pass_threshold = min_pass_threshold

    async def run_single(
        self,
        orchestrator,
        query: BenchmarkQuery,
        preset: str = "balanced"
    ) -> BenchmarkResult:
        """Run a single benchmark query."""
        from .models import SearchRequest

        start_time = datetime.now(timezone.utc)

        try:
            request = SearchRequest(
                query=query.query,
                max_iterations=5,
                min_confidence=0.5
            )
            response = await orchestrator.search(request)

            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            # Extract sources
            sources = []
            if hasattr(response, 'sources') and response.sources:
                sources = [s.get('url', s.get('source', '')) for s in response.sources]

            # Score the answer
            answer = response.synthesis if hasattr(response, 'synthesis') else str(response)
            scores = self.scorer.score(answer, query, sources)

            return BenchmarkResult(
                query=query,
                answer=answer,
                sources=sources,
                confidence=getattr(response, 'confidence', 0.5),
                execution_time_ms=execution_time,
                entity_coverage=scores["entity_coverage"],
                domain_match_rate=scores["domain_match"],
                concept_coverage=scores["concept_coverage"],
                safety_present=scores["safety_present"],
                technical_accuracy=scores["overall_accuracy"],
                passed=scores["passed"]
            )

        except Exception as e:
            logger.error(f"Benchmark query failed: {query.query[:50]}... - {e}")
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            return BenchmarkResult(
                query=query,
                answer=f"Error: {str(e)}",
                sources=[],
                confidence=0.0,
                execution_time_ms=execution_time,
                entity_coverage=0.0,
                domain_match_rate=0.0,
                concept_coverage=0.0,
                safety_present=False,
                technical_accuracy=0.0,
                passed=False
            )

    async def run_benchmark(
        self,
        orchestrator,
        queries: Optional[List[BenchmarkQuery]] = None,
        preset: str = "balanced",
        categories: Optional[List[QueryCategory]] = None,
        difficulties: Optional[List[QueryDifficulty]] = None,
        max_queries: Optional[int] = None
    ) -> BenchmarkReport:
        """
        Run full benchmark suite.

        Args:
            orchestrator: Orchestrator instance to test
            queries: Custom queries or None for FANUC_BENCHMARK
            preset: Orchestrator preset to use
            categories: Filter by categories
            difficulties: Filter by difficulties
            max_queries: Maximum queries to run

        Returns:
            BenchmarkReport with aggregate results
        """
        test_queries = queries or FANUC_BENCHMARK

        # Apply filters
        if categories:
            test_queries = [q for q in test_queries if q.category in categories]
        if difficulties:
            test_queries = [q for q in test_queries if q.difficulty in difficulties]
        if max_queries:
            test_queries = test_queries[:max_queries]

        logger.info(f"Running benchmark with {len(test_queries)} queries using preset '{preset}'")

        results = []
        for i, query in enumerate(test_queries):
            logger.info(f"[{i+1}/{len(test_queries)}] Testing: {query.query[:50]}...")
            result = await self.run_single(orchestrator, query, preset)
            results.append(result)
            logger.info(f"  -> {'PASS' if result.passed else 'FAIL'} "
                       f"(accuracy={result.technical_accuracy:.2f}, "
                       f"confidence={result.confidence:.2f})")

        # Calculate aggregate metrics
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        # By category
        by_category = {}
        for cat in QueryCategory:
            cat_results = [r for r in results if r.query.category == cat]
            if cat_results:
                by_category[cat.value] = {
                    "count": len(cat_results),
                    "passed": sum(1 for r in cat_results if r.passed),
                    "avg_accuracy": sum(r.technical_accuracy for r in cat_results) / len(cat_results)
                }

        # By difficulty
        by_difficulty = {}
        for diff in QueryDifficulty:
            diff_results = [r for r in results if r.query.difficulty == diff]
            if diff_results:
                by_difficulty[diff.value] = {
                    "count": len(diff_results),
                    "passed": sum(1 for r in diff_results if r.passed),
                    "avg_accuracy": sum(r.technical_accuracy for r in diff_results) / len(diff_results)
                }

        return BenchmarkReport(
            total_queries=total,
            passed_queries=passed,
            pass_rate=passed / total if total > 0 else 0.0,
            avg_confidence=sum(r.confidence for r in results) / total if total > 0 else 0.0,
            avg_entity_coverage=sum(r.entity_coverage for r in results) / total if total > 0 else 0.0,
            avg_domain_match=sum(r.domain_match_rate for r in results) / total if total > 0 else 0.0,
            avg_concept_coverage=sum(r.concept_coverage for r in results) / total if total > 0 else 0.0,
            avg_technical_accuracy=sum(r.technical_accuracy for r in results) / total if total > 0 else 0.0,
            avg_execution_time_ms=sum(r.execution_time_ms for r in results) / total if total > 0 else 0.0,
            by_category=by_category,
            by_difficulty=by_difficulty,
            timestamp=datetime.now(timezone.utc).isoformat(),
            preset=preset,
            results=results
        )


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

async def run_benchmark(
    orchestrator,
    preset: str = "balanced",
    categories: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_queries: Optional[int] = None
) -> BenchmarkReport:
    """
    Convenience function to run FANUC benchmark.

    Args:
        orchestrator: UniversalOrchestrator instance
        preset: Preset to test with
        categories: List of category names to filter
        difficulties: List of difficulty names to filter
        max_queries: Maximum number of queries to run

    Returns:
        BenchmarkReport with results
    """
    runner = BenchmarkRunner()

    cat_enums = [QueryCategory(c) for c in categories] if categories else None
    diff_enums = [QueryDifficulty(d) for d in difficulties] if difficulties else None

    return await runner.run_benchmark(
        orchestrator=orchestrator,
        preset=preset,
        categories=cat_enums,
        difficulties=diff_enums,
        max_queries=max_queries
    )


def get_benchmark_stats() -> Dict[str, Any]:
    """Get statistics about the benchmark suite."""
    by_category = {}
    for cat in QueryCategory:
        by_category[cat.value] = sum(1 for q in FANUC_BENCHMARK if q.category == cat)

    by_difficulty = {}
    for diff in QueryDifficulty:
        by_difficulty[diff.value] = sum(1 for q in FANUC_BENCHMARK if q.difficulty == diff)

    safety_critical = sum(1 for q in FANUC_BENCHMARK if q.safety_critical)

    return {
        "total_queries": len(FANUC_BENCHMARK),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "safety_critical_count": safety_critical,
        "all_tags": list(set(tag for q in FANUC_BENCHMARK for tag in q.tags))
    }


def filter_benchmark(
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    safety_only: bool = False
) -> List[BenchmarkQuery]:
    """
    Filter benchmark queries by criteria.

    Args:
        tags: Filter by tags (any match)
        category: Filter by category name
        difficulty: Filter by difficulty name
        safety_only: Only safety-critical queries

    Returns:
        Filtered list of BenchmarkQuery
    """
    result = FANUC_BENCHMARK

    if tags:
        result = [q for q in result if any(t in q.tags for t in tags)]
    if category:
        result = [q for q in result if q.category.value == category]
    if difficulty:
        result = [q for q in result if q.difficulty.value == difficulty]
    if safety_only:
        result = [q for q in result if q.safety_critical]

    return result
