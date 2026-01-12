#!/usr/bin/env python3
"""
Comprehensive Agent Benchmarking Framework

Benchmarks ALL LLM calls in the agentic pipeline:
- Analyzer (query analysis/classification)
- CRAG Evaluator (retrieval quality evaluation)
- Verifier (claim extraction and verification)
- Self-Reflection (ISREL/ISSUP/ISUSE)
- URL Relevance Filter
- Cross-Domain Validator
- Entity Grounding
- HyDE Expander
- Planner (query decomposition)
- Experience Distiller

Each agent is tested with:
- Multiple model variants (parameter sizes, quantizations)
- Multiple test contexts (different query types, difficulties)
- Quality metrics specific to each agent's task

Usage:
    python tests/data/agent_benchmarks.py --agent analyzer --models all
    python tests/data/agent_benchmarks.py --agent crag --models "qwen3:8b,gemma3:4b"
    python tests/data/agent_benchmarks.py --rankings
"""

import asyncio
import sqlite3
import json
import time
import subprocess
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import httpx


class AgentType(Enum):
    """All agent types in the pipeline."""
    ANALYZER = "analyzer"
    CRAG_EVALUATOR = "crag_evaluator"
    VERIFIER = "verifier"
    SELF_REFLECTION = "self_reflection"
    URL_FILTER = "url_relevance_filter"
    CROSS_DOMAIN = "cross_domain_validator"
    ENTITY_GROUNDING = "entity_grounding"
    HYDE_EXPANDER = "hyde_expander"
    PLANNER = "planner"
    EXPERIENCE_DISTILLER = "experience_distiller"
    ENTROPY_MONITOR = "entropy_monitor"
    RAGAS_EVALUATOR = "ragas_evaluator"


# Model variants to benchmark for each agent type
# Organized by: base model -> parameter variants -> quantization variants
MODEL_VARIANTS = {
    # Fast utility models (for analysis, filtering, evaluation)
    "utility": [
        # Gemma family
        "gemma3:1b",
        "gemma3:4b",
        "gemma3:4b-it-q4_K_M",
        "gemma3:4b-it-q8_0",
        # Qwen family
        "qwen3:1.7b",
        "qwen3:4b",
        "qwen3:8b",
        "qwen3:8b-q4_K_M",
        "qwen3:8b-q8_0",
        # Llama family
        "llama3.2:1b",
        "llama3.2:3b",
        # Ministral
        "ministral-3:3b",
        # Phi
        "phi4-mini:3.8b",
    ],

    # Thinking/reasoning models (for complex evaluation, reflection)
    "thinking": [
        "deepseek-r1:1.5b",
        "deepseek-r1:7b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        "deepseek-r1:14b-qwen-distill-q8_0",
        "deepseek-r1:32b",
        "qwq:32b",
        "cogito:3b",
        "cogito:8b",
        "cogito:14b",
        "phi4-reasoning:14b",
        "openthinker:7b",
        "openthinker:32b",
    ],

    # Standard models (for general tasks)
    "standard": [
        "qwen3:8b",
        "qwen3:14b",
        "qwen3:30b-a3b",
        "gemma3:12b",
        "llama3.3:70b",
        "mistral:7b",
        "mistral-nemo:12b",
    ],
}

# Map agent types to recommended model categories
AGENT_MODEL_MAP = {
    AgentType.ANALYZER: ["utility", "standard"],
    AgentType.CRAG_EVALUATOR: ["utility", "thinking"],
    AgentType.VERIFIER: ["utility", "standard"],
    AgentType.SELF_REFLECTION: ["utility", "thinking"],
    AgentType.URL_FILTER: ["utility"],
    AgentType.CROSS_DOMAIN: ["utility", "standard"],
    AgentType.ENTITY_GROUNDING: ["utility"],
    AgentType.HYDE_EXPANDER: ["utility", "standard"],
    AgentType.PLANNER: ["utility", "thinking"],
    AgentType.EXPERIENCE_DISTILLER: ["utility"],
    AgentType.ENTROPY_MONITOR: ["utility"],
    AgentType.RAGAS_EVALUATOR: ["utility", "thinking"],
}


@dataclass
class AgentBenchmarkResult:
    """Result from a single agent benchmark run."""
    agent_type: str
    model: str
    test_context_id: str
    success: bool
    ttfs_ms: Optional[float]
    total_duration_ms: float
    vram_before_mb: int
    vram_after_mb: int
    vram_peak_mb: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    thinking_tokens: Optional[int]
    context_length: int
    output_preview: str
    # Quality metrics (agent-specific)
    accuracy_score: Optional[float]  # How correct was the output?
    completeness_score: Optional[float]  # Did it cover all aspects?
    format_score: Optional[float]  # Was output well-structured?
    efficiency_score: Optional[float]  # Quality per time per VRAM
    # Metadata
    model_size_gb: float
    quantization: str
    parameter_count: str
    is_thinking_model: bool
    error_message: Optional[str]
    notes: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentTestContext:
    """Test context for agent benchmarking."""
    context_id: str
    agent_type: str
    name: str
    difficulty: str  # easy, medium, hard, expert
    # Input for the agent
    query: str
    input_data: Dict[str, Any]  # Agent-specific input
    # Expected outputs for validation
    expected_output_type: str  # json, text, structured
    expected_fields: List[str]
    expected_values: Optional[Dict[str, Any]]
    validation_keywords: List[str]


class AgentBenchmark:
    """Comprehensive agent benchmarking framework."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "agent_benchmarks.db"
        self.db_path = Path(db_path)
        self.ollama_url = "http://localhost:11434"
        self._init_db()
        self._load_test_contexts()

    def _init_db(self):
        """Initialize agent benchmark database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agent_benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    model TEXT NOT NULL,
                    test_context_id TEXT,
                    success INTEGER NOT NULL,
                    ttfs_ms REAL,
                    total_duration_ms REAL NOT NULL,
                    vram_before_mb INTEGER,
                    vram_after_mb INTEGER,
                    vram_peak_mb INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    thinking_tokens INTEGER,
                    context_length INTEGER,
                    output_preview TEXT,
                    output_full TEXT,
                    accuracy_score REAL,
                    completeness_score REAL,
                    format_score REAL,
                    efficiency_score REAL,
                    model_size_gb REAL,
                    quantization TEXT,
                    parameter_count TEXT,
                    is_thinking_model INTEGER,
                    error_message TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_test_contexts (
                    context_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    query TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    expected_output_type TEXT,
                    expected_fields TEXT,
                    expected_values TEXT,
                    validation_keywords TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_model_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    updated_at TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    model TEXT NOT NULL,
                    avg_duration_ms REAL,
                    avg_ttfs_ms REAL,
                    avg_vram_delta_mb REAL,
                    success_rate REAL,
                    avg_accuracy REAL,
                    avg_completeness REAL,
                    avg_format REAL,
                    avg_efficiency REAL,
                    run_count INTEGER,
                    composite_score REAL,
                    rank INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_agent ON agent_benchmark_runs(agent_type);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_model ON agent_benchmark_runs(model);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_timestamp ON agent_benchmark_runs(timestamp);
            """)

    def _load_test_contexts(self):
        """Load predefined test contexts for all agents."""
        self.test_contexts = self._define_test_contexts()

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            for ctx in self.test_contexts:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_test_contexts
                    (context_id, agent_type, name, difficulty, query, input_data,
                     expected_output_type, expected_fields, expected_values,
                     validation_keywords, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ctx.context_id, ctx.agent_type, ctx.name, ctx.difficulty,
                    ctx.query, json.dumps(ctx.input_data),
                    ctx.expected_output_type, json.dumps(ctx.expected_fields),
                    json.dumps(ctx.expected_values) if ctx.expected_values else None,
                    json.dumps(ctx.validation_keywords),
                    datetime.now(timezone.utc).isoformat()
                ))

    def _define_test_contexts(self) -> List[AgentTestContext]:
        """Define test contexts for all agent types."""
        contexts = []

        # ============================================
        # ANALYZER TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="analyzer_fanuc_servo",
                agent_type="analyzer",
                name="FANUC Servo Error Analysis",
                difficulty="medium",
                query="FANUC R-30iB SRVO-062 alarm on J2 axis during welding cycle",
                input_data={"query": "FANUC R-30iB SRVO-062 alarm on J2 axis during welding cycle"},
                expected_output_type="json",
                expected_fields=["query_type", "requires_search", "complexity", "key_topics", "suggested_domains"],
                expected_values={"query_type": "troubleshooting", "requires_search": True},
                validation_keywords=["FANUC", "servo", "alarm", "troubleshoot", "J2", "axis"]
            ),
            AgentTestContext(
                context_id="analyzer_plc_network",
                agent_type="analyzer",
                name="PLC Network Fault Analysis",
                difficulty="hard",
                query="ControlLogix 1756-L73 communication fault 1:13 with remote I/O after power dip",
                input_data={"query": "ControlLogix 1756-L73 communication fault 1:13 with remote I/O after power dip"},
                expected_output_type="json",
                expected_fields=["query_type", "requires_search", "complexity", "key_topics"],
                expected_values={"query_type": "troubleshooting", "requires_search": True},
                validation_keywords=["ControlLogix", "fault", "communication", "I/O", "power"]
            ),
            AgentTestContext(
                context_id="analyzer_conceptual",
                agent_type="analyzer",
                name="Conceptual Query Analysis",
                difficulty="easy",
                query="What is the difference between FINE and CNT motion termination types?",
                input_data={"query": "What is the difference between FINE and CNT motion termination types?"},
                expected_output_type="json",
                expected_fields=["query_type", "requires_search", "complexity"],
                expected_values={"query_type": "conceptual", "complexity": "low"},
                validation_keywords=["FINE", "CNT", "motion", "termination", "compare"]
            ),
            AgentTestContext(
                context_id="analyzer_procedure",
                agent_type="analyzer",
                name="Procedure Query Analysis",
                difficulty="medium",
                query="Step by step procedure for FANUC robot master calibration (RCAL)",
                input_data={"query": "Step by step procedure for FANUC robot master calibration (RCAL)"},
                expected_output_type="json",
                expected_fields=["query_type", "requires_search", "complexity", "key_topics"],
                expected_values={"query_type": "procedural"},
                validation_keywords=["procedure", "calibration", "RCAL", "step"]
            ),
        ])

        # ============================================
        # CRAG EVALUATOR TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="crag_relevant_sources",
                agent_type="crag_evaluator",
                name="High Relevance Evaluation",
                difficulty="medium",
                query="FANUC SRVO-062 battery alarm troubleshooting",
                input_data={
                    "query": "FANUC SRVO-062 battery alarm troubleshooting",
                    "sources": [
                        {"title": "FANUC SRVO Alarm Codes", "url": "fanucamerica.com/alarms", "snippet": "SRVO-062 is issued when the battery for backing up the absolute position data of the pulsecoder is not connected or empty."},
                        {"title": "Robot Forum - SRVO-062", "url": "robot-forum.com/thread/1234", "snippet": "Replace the pulse coder battery and reset the alarm using $MCR.$SPC_RESET."},
                    ]
                },
                expected_output_type="json",
                expected_fields=["quality", "recommended_action", "reasoning"],
                expected_values={"quality": "correct", "recommended_action": "proceed"},
                validation_keywords=["relevant", "correct", "proceed", "battery", "pulsecoder"]
            ),
            AgentTestContext(
                context_id="crag_ambiguous_sources",
                agent_type="crag_evaluator",
                name="Ambiguous Sources Evaluation",
                difficulty="hard",
                query="FANUC M-16 robot not moving",
                input_data={
                    "query": "FANUC M-16 robot not moving",
                    "sources": [
                        {"title": "English StackExchange", "url": "english.stackexchange.com/q/123", "snippet": "What is the distinction between 'robot' and 'automaton'?"},
                        {"title": "TimeAndDate.com", "url": "timeanddate.com/countdown", "snippet": "Countdown to new year celebration"},
                    ]
                },
                expected_output_type="json",
                expected_fields=["quality", "recommended_action", "reasoning"],
                expected_values={"quality": "ambiguous", "recommended_action": "refine_query"},
                validation_keywords=["irrelevant", "ambiguous", "refine", "off-topic"]
            ),
            AgentTestContext(
                context_id="crag_partial_sources",
                agent_type="crag_evaluator",
                name="Partial Relevance Evaluation",
                difficulty="medium",
                query="Allen-Bradley PowerFlex 525 fault code F064",
                input_data={
                    "query": "Allen-Bradley PowerFlex 525 fault code F064",
                    "sources": [
                        {"title": "PowerFlex 525 Manual", "url": "rockwellautomation.com/manuals", "snippet": "F064 - Motor Stall fault. Check motor connections and parameters."},
                        {"title": "General VFD Troubleshooting", "url": "automationdirect.com/blog", "snippet": "Variable frequency drives require proper motor parameters."},
                    ]
                },
                expected_output_type="json",
                expected_fields=["quality", "recommended_action", "reasoning"],
                expected_values={"quality": "correct"},
                validation_keywords=["F064", "motor", "stall", "PowerFlex"]
            ),
        ])

        # ============================================
        # VERIFIER TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="verifier_extract_claims",
                agent_type="verifier",
                name="Extract Claims from Technical Content",
                difficulty="medium",
                query="Extract verifiable claims from FANUC troubleshooting content",
                input_data={
                    "operation": "extract_claims",
                    "content": """SRVO-062 BZAL Alarm occurs when the pulse coder battery voltage drops below 5.5V.
                    To resolve this alarm:
                    1. Replace the battery (CR2032 or equivalent)
                    2. Access Menu > System > Master Cal
                    3. Press F3 to reset pulse coder alarm (RES_PCA)
                    4. Cycle power using FCTN > 0 > 8
                    The battery should be replaced every 2-3 years as preventive maintenance."""
                },
                expected_output_type="json",
                expected_fields=["claims"],
                expected_values=None,
                validation_keywords=["5.5V", "CR2032", "RES_PCA", "2-3 years", "battery"]
            ),
            AgentTestContext(
                context_id="verifier_verify_claims",
                agent_type="verifier",
                name="Verify Claims Against Sources",
                difficulty="hard",
                query="Verify extracted claims against source documents",
                input_data={
                    "operation": "verify_claims",
                    "claims": [
                        "SRVO-062 occurs when battery voltage is below 5.5V",
                        "The pulse coder battery should be replaced every 2-3 years",
                        "RES_PCA resets the pulse coder alarm"
                    ],
                    "sources": [
                        {"content": "SRVO-062 BZAL alarm is issued when the battery for backing up the absolute position data of the pulsecoder is not connected or is empty. Battery voltage threshold: 5.5V."},
                        {"content": "Preventive maintenance schedule: Replace pulse coder batteries every 2-3 years to prevent BZAL alarms."}
                    ]
                },
                expected_output_type="json",
                expected_fields=["verified_claims", "confidence"],
                expected_values=None,
                validation_keywords=["verified", "5.5V", "true", "supported"]
            ),
        ])

        # ============================================
        # SELF-REFLECTION TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="self_reflection_good_synthesis",
                agent_type="self_reflection",
                name="Evaluate Good Synthesis",
                difficulty="medium",
                query="Evaluate synthesis quality for SRVO-062 troubleshooting",
                input_data={
                    "query": "How to troubleshoot FANUC SRVO-062 alarm?",
                    "synthesis": """## SRVO-062 BZAL Alarm Troubleshooting

**Cause:** The SRVO-062 alarm indicates the pulse coder battery is depleted or disconnected.

**Resolution Steps:**
1. Check battery voltage (should be above 5.5V)
2. Replace CR2032 battery if needed
3. Navigate to Menu > System > Master Cal
4. Press F3 to execute RES_PCA
5. Cycle power via FCTN > 0 > 8

**Prevention:** Replace batteries every 2-3 years.

*Sources: FANUC Alarm Reference, Robot-Forum.com*""",
                    "sources": [
                        {"title": "FANUC Alarm Reference", "url": "fanucamerica.com/alarms", "relevant": True},
                        {"title": "Robot Forum", "url": "robot-forum.com/thread/1234", "relevant": True}
                    ]
                },
                expected_output_type="json",
                expected_fields=["ISREL", "ISSUP", "ISUSE", "overall_quality", "suggestions"],
                expected_values={"overall_quality": "good"},
                validation_keywords=["relevant", "supported", "useful", "good", "high"]
            ),
            AgentTestContext(
                context_id="self_reflection_poor_synthesis",
                agent_type="self_reflection",
                name="Evaluate Poor Synthesis",
                difficulty="hard",
                query="Evaluate synthesis that drifts off-topic",
                input_data={
                    "query": "How to troubleshoot FANUC SRVO-062 alarm?",
                    "synthesis": """FANUC robots are widely used in automotive manufacturing.
                    The company was founded in 1956 in Japan.
                    They offer various robot models including the M-16, R-2000, and LR Mate series.
                    Industrial robots have revolutionized manufacturing processes worldwide.""",
                    "sources": [
                        {"title": "Wikipedia - FANUC", "url": "wikipedia.org/fanuc", "relevant": False},
                    ]
                },
                expected_output_type="json",
                expected_fields=["ISREL", "ISSUP", "ISUSE", "overall_quality", "suggestions"],
                expected_values={"overall_quality": "poor"},
                validation_keywords=["off-topic", "not relevant", "poor", "low", "missing"]
            ),
        ])

        # ============================================
        # URL RELEVANCE FILTER TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="url_filter_mixed",
                agent_type="url_relevance_filter",
                name="Filter Mixed URL Set",
                difficulty="medium",
                query="FANUC SRVO-062 alarm troubleshooting",
                input_data={
                    "query": "FANUC SRVO-062 alarm troubleshooting",
                    "urls": [
                        {"url": "fanucamerica.com/alarms/srvo-062", "title": "FANUC SRVO-062 Alarm Reference", "snippet": "SRVO-062 BZAL alarm troubleshooting guide"},
                        {"url": "robot-forum.com/thread/1234", "title": "SRVO-062 Battery Issue", "snippet": "How to replace pulse coder battery"},
                        {"url": "english.stackexchange.com/q/123", "title": "Meaning of robot", "snippet": "Etymology of the word robot"},
                        {"url": "timeanddate.com/countdown", "title": "Countdown Timer", "snippet": "New year countdown"},
                        {"url": "everythingaboutrobots.com/fanuc-alarms", "title": "FANUC Alarm Codes", "snippet": "Complete list of FANUC servo alarms"},
                    ]
                },
                expected_output_type="json",
                expected_fields=["relevant_indices", "reasoning"],
                expected_values={"relevant_indices": [0, 1, 4]},
                validation_keywords=["fanucamerica", "robot-forum", "everythingaboutrobots", "relevant", "SRVO-062"]
            ),
            AgentTestContext(
                context_id="url_filter_all_irrelevant",
                agent_type="url_relevance_filter",
                name="Filter All Irrelevant URLs",
                difficulty="easy",
                query="How to calibrate FANUC robot TCP",
                input_data={
                    "query": "How to calibrate FANUC robot TCP",
                    "urls": [
                        {"url": "cooking.com/recipes", "title": "Best Recipes", "snippet": "Top 10 recipes for dinner"},
                        {"url": "weather.com/forecast", "title": "Weather Forecast", "snippet": "Weekly weather update"},
                        {"url": "sports.com/scores", "title": "Game Scores", "snippet": "Latest sports scores"},
                    ]
                },
                expected_output_type="json",
                expected_fields=["relevant_indices", "reasoning"],
                expected_values={"relevant_indices": []},
                validation_keywords=["none", "irrelevant", "off-topic", "no relevant"]
            ),
        ])

        # ============================================
        # HYDE EXPANDER TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="hyde_troubleshooting",
                agent_type="hyde_expander",
                name="HyDE for Troubleshooting Query",
                difficulty="medium",
                query="FANUC robot not moving after power cycle",
                input_data={"query": "FANUC robot not moving after power cycle"},
                expected_output_type="text",
                expected_fields=[],
                expected_values=None,
                validation_keywords=["servo", "alarm", "mastering", "brake", "enable", "deadman", "e-stop"]
            ),
            AgentTestContext(
                context_id="hyde_conceptual",
                agent_type="hyde_expander",
                name="HyDE for Conceptual Query",
                difficulty="easy",
                query="What is robot singularity?",
                input_data={"query": "What is robot singularity?"},
                expected_output_type="text",
                expected_fields=[],
                expected_values=None,
                validation_keywords=["singularity", "joint", "velocity", "infinite", "configuration", "axis"]
            ),
        ])

        # ============================================
        # CROSS-DOMAIN VALIDATOR TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="cross_domain_valid",
                agent_type="cross_domain_validator",
                name="Validate Valid Cross-Domain Claims",
                difficulty="medium",
                query="Validate synthesis about FANUC robot with Rockwell PLC integration",
                input_data={
                    "synthesis": """The FANUC robot communicates with the Allen-Bradley ControlLogix PLC via EtherNet/IP.
                    The RJ3-iC controller sends digital I/O signals through the DeviceNet module.
                    When SRVO-062 occurs, the PLC receives a fault signal on input I:1/0.""",
                    "domains": ["fanuc", "rockwell", "plc"]
                },
                expected_output_type="json",
                expected_fields=["valid_claims", "invalid_claims", "warnings"],
                expected_values=None,
                validation_keywords=["valid", "EtherNet/IP", "DeviceNet", "compatible"]
            ),
            AgentTestContext(
                context_id="cross_domain_invalid",
                agent_type="cross_domain_validator",
                name="Detect Invalid Cross-Domain Claims",
                difficulty="hard",
                query="Detect hallucinated cross-domain causation",
                input_data={
                    "synthesis": """The SRVO-068 robot servo alarm causes hydraulic pressure fluctuations in the IMM.
                    This triggers the eDart cavity pressure sensor to malfunction.
                    The robot's motor overload directly affects the injection unit barrel heater zones.""",
                    "domains": ["fanuc", "imm", "edart"]
                },
                expected_output_type="json",
                expected_fields=["valid_claims", "invalid_claims", "warnings"],
                expected_values=None,
                validation_keywords=["invalid", "hallucination", "impossible", "no connection", "false"]
            ),
        ])

        # ============================================
        # PLANNER TEST CONTEXTS
        # ============================================
        contexts.extend([
            AgentTestContext(
                context_id="planner_simple",
                agent_type="planner",
                name="Simple Query Decomposition",
                difficulty="easy",
                query="What is FANUC SRVO-062 alarm?",
                input_data={"query": "What is FANUC SRVO-062 alarm?"},
                expected_output_type="json",
                expected_fields=["search_queries", "strategy"],
                expected_values=None,
                validation_keywords=["SRVO-062", "FANUC", "alarm", "search"]
            ),
            AgentTestContext(
                context_id="planner_complex",
                agent_type="planner",
                name="Complex Multi-Part Query Decomposition",
                difficulty="hard",
                query="Compare FANUC R-30iA vs R-30iB controller capabilities and troubleshoot common upgrade issues",
                input_data={"query": "Compare FANUC R-30iA vs R-30iB controller capabilities and troubleshoot common upgrade issues"},
                expected_output_type="json",
                expected_fields=["search_queries", "strategy", "sub_queries"],
                expected_values=None,
                validation_keywords=["R-30iA", "R-30iB", "compare", "upgrade", "capabilities", "multiple"]
            ),
        ])

        return contexts

    def get_gpu_stats(self) -> Dict[str, int]:
        """Get current GPU memory stats."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "used_mb": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2])
                }
        except Exception:
            pass
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

    def parse_model_info(self, model_name: str) -> Tuple[str, str, float, bool]:
        """Parse model name to extract parameter count, quantization, size estimate, is_thinking."""
        # Extract parameter count
        param_match = re.search(r'(\d+\.?\d*)b', model_name.lower())
        param_count = param_match.group(1) + "B" if param_match else "unknown"

        # Extract quantization
        quant_patterns = ['q8_0', 'q4_k_m', 'q4_0', 'q5_k_m', 'q6_k', 'fp16', 'bf16', 'f16']
        quantization = "default"
        for q in quant_patterns:
            if q in model_name.lower():
                quantization = q.upper()
                break

        # Estimate size (rough)
        size_gb = 0.0
        if param_match:
            params = float(param_match.group(1))
            if 'fp16' in model_name.lower() or 'bf16' in model_name.lower():
                size_gb = params * 2
            elif 'q8' in model_name.lower():
                size_gb = params * 1
            elif 'q4' in model_name.lower():
                size_gb = params * 0.5
            else:
                size_gb = params * 0.6

        # Detect thinking model
        thinking_indicators = ['deepseek-r1', 'qwq', 'cogito', 'openthinker', 'reasoning', 'think']
        is_thinking = any(ind in model_name.lower() for ind in thinking_indicators)

        return param_count, quantization, size_gb, is_thinking

    async def unload_all_models(self):
        """Unload all models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={"model": model["name"], "keep_alive": 0}
                        )
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Warning: Could not unload models: {e}")

    def _build_agent_prompt(self, agent_type: str, test_context: AgentTestContext) -> str:
        """Build appropriate prompt for each agent type."""

        if agent_type == "analyzer":
            return f"""Analyze the following query and determine its characteristics.

Query: {test_context.query}

Respond with a JSON object containing:
- query_type: One of [troubleshooting, procedural, conceptual, comparison, factual]
- requires_search: Boolean indicating if web search is needed
- complexity: One of [low, medium, high]
- key_topics: List of main topics/entities in the query
- suggested_domains: List of domains likely to have relevant information

JSON Response:"""

        elif agent_type == "crag_evaluator":
            sources = test_context.input_data.get("sources", [])
            source_text = "\n".join([f"- [{s['title']}]({s['url']}): {s['snippet']}" for s in sources])
            return f"""Evaluate the quality and relevance of retrieved sources for the query.

Query: {test_context.input_data['query']}

Retrieved Sources:
{source_text}

Evaluate and respond with JSON:
- quality: One of [correct, ambiguous, incorrect]
- recommended_action: One of [proceed, refine_query, web_search_fallback]
- reasoning: Brief explanation of your evaluation

JSON Response:"""

        elif agent_type == "verifier":
            if test_context.input_data.get("operation") == "extract_claims":
                return f"""Extract verifiable factual claims from the following technical content.

Content:
{test_context.input_data['content']}

Extract specific, verifiable claims as a JSON array:
{{"claims": ["claim 1", "claim 2", ...]}}

JSON Response:"""
            else:
                claims = test_context.input_data.get("claims", [])
                sources = test_context.input_data.get("sources", [])
                claims_text = "\n".join([f"- {c}" for c in claims])
                sources_text = "\n".join([f"Source: {s['content']}" for s in sources])
                return f"""Verify the following claims against the provided sources.

Claims to verify:
{claims_text}

Sources:
{sources_text}

Respond with JSON:
- verified_claims: List of claims that are supported by sources
- unverified_claims: List of claims not supported
- confidence: Overall confidence score (0-1)

JSON Response:"""

        elif agent_type == "self_reflection":
            return f"""Evaluate the quality of the following synthesis for the given query.

Query: {test_context.input_data['query']}

Synthesis:
{test_context.input_data['synthesis']}

Sources consulted: {len(test_context.input_data.get('sources', []))}

Evaluate using Self-RAG criteria and respond with JSON:
- ISREL: Is the synthesis relevant to the query? (0-1)
- ISSUP: Is the synthesis supported by sources? (0-1)
- ISUSE: Is the synthesis useful and actionable? (0-1)
- overall_quality: One of [poor, fair, good, excellent]
- suggestions: List of improvement suggestions

JSON Response:"""

        elif agent_type == "url_relevance_filter":
            urls = test_context.input_data.get("urls", [])
            urls_text = "\n".join([f"{i}. [{u['title']}]({u['url']}): {u['snippet']}"
                                   for i, u in enumerate(urls)])
            return f"""Evaluate which URLs are relevant to the query.

Query: {test_context.input_data['query']}

URLs to evaluate:
{urls_text}

Respond with JSON:
- relevant_indices: List of indices (0-based) of relevant URLs
- reasoning: Brief explanation

JSON Response:"""

        elif agent_type == "hyde_expander":
            return f"""Generate a hypothetical document that would perfectly answer the following query.
This document should contain technical details, specific terms, and information that would be found in an authoritative source.

Query: {test_context.input_data['query']}

Hypothetical Document:"""

        elif agent_type == "cross_domain_validator":
            return f"""Validate the following synthesis for cross-domain hallucinations.
Check if claims about interactions between different industrial systems are technically valid.

Synthesis:
{test_context.input_data['synthesis']}

Domains involved: {test_context.input_data['domains']}

Respond with JSON:
- valid_claims: List of technically valid claims
- invalid_claims: List of hallucinated or impossible claims
- warnings: List of claims that need verification

JSON Response:"""

        elif agent_type == "planner":
            return f"""Decompose the following query into search sub-queries and plan the search strategy.

Query: {test_context.input_data['query']}

Respond with JSON:
- search_queries: List of specific search queries to execute
- strategy: Brief description of search approach
- sub_queries: If complex, list of sub-questions to answer

JSON Response:"""

        else:
            return f"Process the following: {test_context.query}"

    def _evaluate_output(
        self,
        agent_type: str,
        output: str,
        test_context: AgentTestContext
    ) -> Tuple[float, float, float]:
        """Evaluate output quality. Returns (accuracy, completeness, format) scores."""

        accuracy = 0.0
        completeness = 0.0
        format_score = 0.0

        # Check if output is valid JSON (for agents expecting JSON)
        if test_context.expected_output_type == "json":
            try:
                # Extract JSON from output
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    format_score = 0.8

                    # Check expected fields
                    if test_context.expected_fields:
                        found_fields = sum(1 for f in test_context.expected_fields if f in parsed)
                        completeness = found_fields / len(test_context.expected_fields)

                    # Check expected values
                    if test_context.expected_values:
                        matched_values = sum(1 for k, v in test_context.expected_values.items()
                                           if parsed.get(k) == v)
                        accuracy = matched_values / len(test_context.expected_values)
                    else:
                        accuracy = 0.7  # Default if no expected values

            except (json.JSONDecodeError, AttributeError):
                format_score = 0.2
        else:
            format_score = 0.7  # Text output is acceptable

        # Check validation keywords
        output_lower = output.lower()
        if test_context.validation_keywords:
            found_keywords = sum(1 for kw in test_context.validation_keywords
                               if kw.lower() in output_lower)
            keyword_score = found_keywords / len(test_context.validation_keywords)
            accuracy = max(accuracy, keyword_score * 0.8)
            completeness = max(completeness, keyword_score)

        return accuracy, completeness, format_score

    async def benchmark_agent(
        self,
        agent_type: str,
        model: str,
        test_context: AgentTestContext,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        unload_first: bool = True
    ) -> AgentBenchmarkResult:
        """Benchmark a single agent with a specific model and context."""

        if unload_first:
            await self.unload_all_models()
            await asyncio.sleep(3)

        vram_before = self.get_gpu_stats()["used_mb"]
        param_count, quantization, size_gb, is_thinking = self.parse_model_info(model)

        prompt = self._build_agent_prompt(agent_type, test_context)

        start_time = time.time()
        ttfs_ms = None
        output_text = ""
        error_msg = None
        success = False
        input_tokens = None
        output_tokens = None
        thinking_tokens = 0

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if first_token_time is None and data.get("response"):
                                    first_token_time = time.time()
                                    ttfs_ms = (first_token_time - start_time) * 1000

                                output_text += data.get("response", "")

                                if data.get("done"):
                                    input_tokens = data.get("prompt_eval_count")
                                    output_tokens = data.get("eval_count")
                            except json.JSONDecodeError:
                                continue

                success = len(output_text) > 20

                # Count thinking tokens
                thinking_patterns = [r'<think>(.*?)</think>', r'<thinking>(.*?)</thinking>']
                for pattern in thinking_patterns:
                    for match in re.findall(pattern, output_text, re.DOTALL | re.IGNORECASE):
                        thinking_tokens += len(match.split())

        except Exception as e:
            error_msg = str(e)
            success = False

        total_duration_ms = (time.time() - start_time) * 1000
        vram_after = self.get_gpu_stats()["used_mb"]
        vram_peak = max(vram_before, vram_after)

        # Evaluate output quality
        accuracy, completeness, format_score = self._evaluate_output(agent_type, output_text, test_context)

        # Calculate efficiency score
        vram_delta = max(abs(vram_after - vram_before), 100)  # Minimum 100MB
        avg_quality = (accuracy + completeness + format_score) / 3
        efficiency = (avg_quality * 1000000) / (total_duration_ms * vram_delta) if total_duration_ms > 0 else 0

        result = AgentBenchmarkResult(
            agent_type=agent_type,
            model=model,
            test_context_id=test_context.context_id,
            success=success,
            ttfs_ms=ttfs_ms,
            total_duration_ms=total_duration_ms,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
            vram_peak_mb=vram_peak,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            context_length=len(prompt),
            output_preview=output_text[:500] if output_text else "",
            accuracy_score=accuracy,
            completeness_score=completeness,
            format_score=format_score,
            efficiency_score=efficiency,
            model_size_gb=size_gb,
            quantization=quantization,
            parameter_count=param_count,
            is_thinking_model=is_thinking,
            error_message=error_msg,
            notes=None
        )

        # Store result
        self._store_result(result, output_text)

        return result

    def _store_result(self, result: AgentBenchmarkResult, full_output: str):
        """Store benchmark result in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_benchmark_runs (
                    timestamp, agent_type, model, test_context_id, success,
                    ttfs_ms, total_duration_ms, vram_before_mb, vram_after_mb, vram_peak_mb,
                    input_tokens, output_tokens, thinking_tokens, context_length,
                    output_preview, output_full, accuracy_score, completeness_score,
                    format_score, efficiency_score, model_size_gb, quantization,
                    parameter_count, is_thinking_model, error_message, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp, result.agent_type, result.model, result.test_context_id,
                1 if result.success else 0, result.ttfs_ms, result.total_duration_ms,
                result.vram_before_mb, result.vram_after_mb, result.vram_peak_mb,
                result.input_tokens, result.output_tokens, result.thinking_tokens,
                result.context_length, result.output_preview, full_output,
                result.accuracy_score, result.completeness_score, result.format_score,
                result.efficiency_score, result.model_size_gb, result.quantization,
                result.parameter_count, 1 if result.is_thinking_model else 0,
                result.error_message, result.notes
            ))

    def get_models_for_agent(self, agent_type: AgentType, include_all: bool = False) -> List[str]:
        """Get list of models to benchmark for an agent type."""
        if include_all:
            all_models = []
            for category in MODEL_VARIANTS.values():
                all_models.extend(category)
            return list(set(all_models))

        categories = AGENT_MODEL_MAP.get(agent_type, ["utility"])
        models = []
        for cat in categories:
            models.extend(MODEL_VARIANTS.get(cat, []))
        return list(set(models))

    def get_test_contexts_for_agent(self, agent_type: str) -> List[AgentTestContext]:
        """Get test contexts for a specific agent type."""
        return [ctx for ctx in self.test_contexts if ctx.agent_type == agent_type]

    def calculate_rankings(self, agent_type: Optional[str] = None) -> List[Dict]:
        """Calculate model rankings for an agent type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            where_clause = "WHERE agent_type = ?" if agent_type else ""
            params = (agent_type,) if agent_type else ()

            cursor = conn.execute(f"""
                SELECT
                    agent_type,
                    model,
                    COUNT(*) as run_count,
                    AVG(total_duration_ms) as avg_duration_ms,
                    AVG(ttfs_ms) as avg_ttfs_ms,
                    AVG(vram_after_mb - vram_before_mb) as avg_vram_delta_mb,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(completeness_score) as avg_completeness,
                    AVG(format_score) as avg_format,
                    AVG(efficiency_score) as avg_efficiency,
                    quantization,
                    parameter_count,
                    MAX(is_thinking_model) as is_thinking
                FROM agent_benchmark_runs
                {where_clause}
                GROUP BY agent_type, model
                HAVING run_count >= 1
                ORDER BY agent_type, avg_efficiency DESC
            """, params)

            results = []
            current_agent = None
            rank = 0

            for row in cursor.fetchall():
                r = dict(row)
                if r["agent_type"] != current_agent:
                    current_agent = r["agent_type"]
                    rank = 0
                rank += 1

                # Calculate composite score
                r["composite_score"] = (
                    (r["avg_accuracy"] or 0) * 0.35 +
                    (r["avg_completeness"] or 0) * 0.25 +
                    (r["avg_format"] or 0) * 0.15 +
                    (r["success_rate"] or 0) * 0.25
                )
                r["rank"] = rank
                results.append(r)

            return results

    def print_rankings(self, agent_type: Optional[str] = None):
        """Print formatted rankings table."""
        rankings = self.calculate_rankings(agent_type)

        if not rankings:
            print("\nNo benchmark results yet.")
            return

        current_agent = None

        for r in rankings:
            if r["agent_type"] != current_agent:
                current_agent = r["agent_type"]
                print(f"\n{'='*120}")
                print(f"RANKINGS FOR: {current_agent.upper()}")
                print(f"{'='*120}")
                print(f"{'Rank':<5} {'Model':<35} {'Duration':<10} {'TTFS':<8} {'VRAM Δ':<9} {'Success':<8} {'Accuracy':<9} {'Complete':<9} {'Format':<8} {'Score':<7}")
                print(f"{'-'*120}")

            duration = f"{r['avg_duration_ms']:.0f}ms" if r['avg_duration_ms'] else "N/A"
            ttfs = f"{r['avg_ttfs_ms']:.0f}ms" if r['avg_ttfs_ms'] else "N/A"
            vram = f"{r['avg_vram_delta_mb']:.0f}MB" if r['avg_vram_delta_mb'] else "N/A"
            success = f"{r['success_rate']*100:.0f}%" if r['success_rate'] else "N/A"
            accuracy = f"{r['avg_accuracy']:.2f}" if r['avg_accuracy'] else "N/A"
            completeness = f"{r['avg_completeness']:.2f}" if r['avg_completeness'] else "N/A"
            format_s = f"{r['avg_format']:.2f}" if r['avg_format'] else "N/A"
            score = f"{r['composite_score']:.2f}" if r['composite_score'] else "N/A"

            think = "🧠" if r['is_thinking'] else ""
            print(f"{r['rank']:<5} {r['model']:<33}{think:<2} {duration:<10} {ttfs:<8} {vram:<9} {success:<8} {accuracy:<9} {completeness:<9} {format_s:<8} {score:<7}")


async def run_agent_benchmarks(
    agent_types: List[str] = None,
    models: List[str] = None,
    max_contexts: int = 3,
    unload_between: bool = True,
    quick: bool = False
):
    """Run comprehensive agent benchmarks."""

    bench = AgentBenchmark()

    # Default to all agents if not specified
    if agent_types is None:
        agent_types = ["analyzer", "crag_evaluator", "verifier", "self_reflection", "url_relevance_filter"]

    total_runs = 0

    for agent_type in agent_types:
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {agent_type.upper()}")
        print(f"{'='*80}")

        # Get test contexts for this agent
        contexts = bench.get_test_contexts_for_agent(agent_type)[:max_contexts]

        if not contexts:
            print(f"No test contexts for {agent_type}, skipping...")
            continue

        # Get models to test
        if models:
            test_models = models
        elif quick:
            # Quick mode: just test 2-3 key models
            test_models = ["gemma3:4b", "qwen3:8b"]
        else:
            test_models = bench.get_models_for_agent(AgentType(agent_type))

        print(f"Testing {len(test_models)} models on {len(contexts)} contexts")

        for ctx in contexts:
            print(f"\n--- Context: {ctx.name} ({ctx.difficulty}) ---")
            print(f"Query: {ctx.query[:60]}...")

            for model in test_models:
                print(f"\n  Testing: {model}")

                try:
                    result = await bench.benchmark_agent(
                        agent_type=agent_type,
                        model=model,
                        test_context=ctx,
                        unload_first=unload_between
                    )

                    if result.success:
                        print(f"    ✓ Duration: {result.total_duration_ms:.0f}ms | "
                              f"TTFS: {result.ttfs_ms:.0f}ms | "
                              f"Accuracy: {result.accuracy_score:.2f} | "
                              f"Complete: {result.completeness_score:.2f}")
                        total_runs += 1
                    else:
                        print(f"    ✗ FAILED: {result.error_message}")

                except Exception as e:
                    print(f"    ✗ ERROR: {e}")

                await asyncio.sleep(2)

    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE: {total_runs} successful runs")
    print(f"{'='*80}")

    bench.print_rankings()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Benchmark Framework")
    parser.add_argument("--agent", type=str, help="Specific agent to benchmark (analyzer, crag_evaluator, verifier, etc.)")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to test")
    parser.add_argument("--all-models", action="store_true", help="Test all available models")
    parser.add_argument("--quick", action="store_true", help="Quick mode: test only key models")
    parser.add_argument("--max-contexts", type=int, default=3, help="Max contexts per agent")
    parser.add_argument("--no-unload", action="store_true", help="Don't unload models between tests")
    parser.add_argument("--rankings", action="store_true", help="Show current rankings")
    parser.add_argument("--rankings-agent", type=str, help="Show rankings for specific agent")
    args = parser.parse_args()

    bench = AgentBenchmark()

    if args.rankings or args.rankings_agent:
        bench.print_rankings(args.rankings_agent)
        return

    agent_types = None
    if args.agent:
        agent_types = [args.agent]

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.all_models:
        # Get all models from all categories
        all_models = []
        for category in MODEL_VARIANTS.values():
            all_models.extend(category)
        models = list(set(all_models))

    await run_agent_benchmarks(
        agent_types=agent_types,
        models=models,
        max_contexts=args.max_contexts,
        unload_between=not args.no_unload,
        quick=args.quick
    )


if __name__ == "__main__":
    asyncio.run(main())
