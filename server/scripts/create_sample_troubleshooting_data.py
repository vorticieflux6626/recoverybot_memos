#!/usr/bin/env python3
"""
Create Sample Troubleshooting Data for Recovery Bot

Phase 6: Quest System Re-Implementation Plan
Creates sample workflows, tasks, user expertise, and achievements
for testing the troubleshooting task tracker system.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python scripts/create_sample_troubleshooting_data.py [--reset]

Options:
    --reset     Drop and recreate all troubleshooting tables before populating
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from config.database import AsyncSessionLocal, async_engine
from models.troubleshooting import (
    TroubleshootingWorkflow,
    WorkflowTask,
    TroubleshootingSession,
    TaskExecution,
    UserExpertise,
    TroubleshootingAchievement,
    TroubleshootingCategory,
    TaskExecutionType,
    ExpertiseLevel,
    TroubleshootingDomain,
    SessionState,
    TaskState,
)


# =============================================================================
# SAMPLE WORKFLOWS
# =============================================================================

SAMPLE_WORKFLOWS = [
    # FANUC Servo Error Workflows
    {
        "name": "SRVO-062 BZAL Alarm Diagnosis",
        "description": "Diagnose FANUC SRVO-062 (BZAL alarm) - Servo motor encoder error typically caused by cable issues or encoder failure.",
        "category": TroubleshootingCategory.ERROR_DIAGNOSIS,
        "domain": TroubleshootingDomain.FANUC_SERVO,
        "expertise_points": 25,
        "estimated_duration_minutes": 15,
        "error_code_patterns": ["SRVO-062", "SRVO-0062"],
        "symptom_keywords": ["bzal", "encoder", "battery", "backup"],
        "entry_node_types": ["error_code"],
        "target_node_types": ["remedy", "procedure"],
        "traversal_mode": "semantic_astar",
        "max_hops": 4,
        "beam_width": 10,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Entity Extraction", "pipeline_hook": "_extract_entities", "order": 1},
            {"name": "Technical Doc Search", "pipeline_hook": "_search_technical_docs", "order": 2},
            {"name": "Cross-Domain Validation", "pipeline_hook": "_validate_cross_domain", "order": 3},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 4},
            {"name": "User Verification", "type": "user_action", "prompt": "Did the suggested solution resolve the SRVO-062 alarm?", "order": 5},
        ]
    },
    {
        "name": "SRVO-063 BLAL Alarm Diagnosis",
        "description": "Diagnose FANUC SRVO-063 (BLAL alarm) - Low battery alarm for servo motor encoder backup.",
        "category": TroubleshootingCategory.ERROR_DIAGNOSIS,
        "domain": TroubleshootingDomain.FANUC_SERVO,
        "expertise_points": 20,
        "estimated_duration_minutes": 10,
        "error_code_patterns": ["SRVO-063", "SRVO-0063"],
        "symptom_keywords": ["blal", "low battery", "encoder battery"],
        "entry_node_types": ["error_code"],
        "target_node_types": ["remedy", "procedure"],
        "traversal_mode": "semantic_astar",
        "max_hops": 4,
        "beam_width": 10,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Technical Doc Search", "pipeline_hook": "_search_technical_docs", "order": 1},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 2},
            {"name": "Battery Replacement", "type": "user_action", "prompt": "Replace the encoder battery following the procedure. Did this resolve the alarm?", "order": 3},
        ]
    },
    {
        "name": "SRVO-068 DTERR Alarm Diagnosis",
        "description": "Diagnose FANUC SRVO-068 (DTERR alarm) - Servo motor disconnect error, typically cable or connection issues.",
        "category": TroubleshootingCategory.ERROR_DIAGNOSIS,
        "domain": TroubleshootingDomain.FANUC_SERVO,
        "expertise_points": 30,
        "estimated_duration_minutes": 20,
        "error_code_patterns": ["SRVO-068", "SRVO-0068"],
        "symptom_keywords": ["dterr", "disconnect", "cable", "connection"],
        "entry_node_types": ["error_code"],
        "target_node_types": ["remedy", "procedure"],
        "traversal_mode": "semantic_astar",
        "max_hops": 5,
        "beam_width": 15,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Entity Extraction", "pipeline_hook": "_extract_entities", "order": 1},
            {"name": "Technical Doc Search", "pipeline_hook": "_search_technical_docs", "order": 2},
            {"name": "Circuit Diagram Lookup", "pipeline_hook": "_get_circuit_diagram", "order": 3},
            {"name": "Cross-Domain Validation", "pipeline_hook": "_validate_cross_domain", "order": 4},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 5},
            {"name": "Cable Inspection", "type": "user_action", "prompt": "Inspect the encoder cable and connections. Are there any visible issues?", "order": 6},
        ]
    },
    # FANUC Motion Error Workflows
    {
        "name": "MOTN-017 Overrun Alarm Diagnosis",
        "description": "Diagnose FANUC MOTN-017 - Position overrun alarm during motion execution.",
        "category": TroubleshootingCategory.ERROR_DIAGNOSIS,
        "domain": TroubleshootingDomain.FANUC_MOTION,
        "expertise_points": 25,
        "estimated_duration_minutes": 15,
        "error_code_patterns": ["MOTN-017", "MOTN-0017"],
        "symptom_keywords": ["overrun", "position", "motion limit"],
        "entry_node_types": ["error_code"],
        "target_node_types": ["remedy", "procedure"],
        "traversal_mode": "semantic_astar",
        "max_hops": 4,
        "beam_width": 10,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Technical Doc Search", "pipeline_hook": "_search_technical_docs", "order": 1},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 2},
        ]
    },
    # IMM Defect Workflows
    {
        "name": "Short Shot Defect Analysis",
        "description": "Analyze and resolve injection molding short shot defects - incomplete part fill.",
        "category": TroubleshootingCategory.SYMPTOM_ANALYSIS,
        "domain": TroubleshootingDomain.IMM_DEFECTS,
        "expertise_points": 20,
        "estimated_duration_minutes": 15,
        "error_code_patterns": [],
        "symptom_keywords": ["short shot", "incomplete fill", "underfill", "not filling"],
        "entry_node_types": ["symptom", "defect"],
        "target_node_types": ["remedy", "parameter"],
        "traversal_mode": "flow_based",
        "max_hops": 5,
        "beam_width": 15,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Entity Extraction", "pipeline_hook": "_extract_entities", "order": 1},
            {"name": "Defect Pattern Search", "pipeline_hook": "_search_technical_docs", "order": 2},
            {"name": "Process Parameter Analysis", "pipeline_hook": "_analyze_process_params", "order": 3},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 4},
            {"name": "Parameter Adjustment", "type": "user_action", "prompt": "Adjust the suggested process parameters. Did defect improve?", "order": 5},
        ]
    },
    {
        "name": "Flash Defect Analysis",
        "description": "Analyze and resolve injection molding flash defects - excess material at parting line.",
        "category": TroubleshootingCategory.SYMPTOM_ANALYSIS,
        "domain": TroubleshootingDomain.IMM_DEFECTS,
        "expertise_points": 20,
        "estimated_duration_minutes": 15,
        "error_code_patterns": [],
        "symptom_keywords": ["flash", "excess material", "parting line", "burr"],
        "entry_node_types": ["symptom", "defect"],
        "target_node_types": ["remedy", "parameter"],
        "traversal_mode": "flow_based",
        "max_hops": 5,
        "beam_width": 15,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Defect Pattern Search", "pipeline_hook": "_search_technical_docs", "order": 1},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 2},
        ]
    },
    # Procedure Execution Workflows
    {
        "name": "Robot Mastering Procedure",
        "description": "Step-by-step procedure for FANUC robot axis mastering after encoder battery replacement.",
        "category": TroubleshootingCategory.PROCEDURE_EXECUTION,
        "domain": TroubleshootingDomain.FANUC_SERVO,
        "expertise_points": 50,
        "estimated_duration_minutes": 45,
        "error_code_patterns": [],
        "symptom_keywords": ["mastering", "zero position", "calibration", "home position"],
        "entry_node_types": ["procedure"],
        "target_node_types": ["step", "warning"],
        "traversal_mode": "multi_hop",
        "max_hops": 6,
        "beam_width": 20,
        "tasks": [
            {"name": "Query Analysis", "pipeline_hook": "_analyze_query", "order": 0},
            {"name": "Procedure Lookup", "pipeline_hook": "_search_technical_docs", "order": 1},
            {"name": "Diagram Generation", "pipeline_hook": "_generate_diagram", "order": 2},
            {"name": "Synthesis", "pipeline_hook": "_synthesize", "order": 3},
            {"name": "Backup Current Position", "type": "user_action", "prompt": "Record current axis positions before mastering.", "order": 4},
            {"name": "Execute Mastering", "type": "user_action", "prompt": "Follow the mastering procedure steps. Confirm completion.", "order": 5},
            {"name": "Verify Positions", "type": "user_action", "prompt": "Verify robot reaches expected positions after mastering.", "order": 6},
        ]
    },
]


# =============================================================================
# SAMPLE ACHIEVEMENTS
# =============================================================================

SAMPLE_ACHIEVEMENTS = [
    # Sessions milestones
    {"title": "First Resolution", "description": "Complete your first troubleshooting session", "criteria_type": "sessions_completed", "criteria_value": 1, "domain": None},
    {"title": "Problem Solver", "description": "Complete 10 troubleshooting sessions", "criteria_type": "sessions_completed", "criteria_value": 10, "domain": None},
    {"title": "Expert Troubleshooter", "description": "Complete 50 troubleshooting sessions", "criteria_type": "sessions_completed", "criteria_value": 50, "domain": None},

    # Domain mastery
    {"title": "FANUC Servo Initiate", "description": "Earn 50 points in FANUC Servo troubleshooting", "criteria_type": "domain_points", "criteria_value": 50, "domain": TroubleshootingDomain.FANUC_SERVO},
    {"title": "FANUC Servo Technician", "description": "Earn 200 points in FANUC Servo troubleshooting", "criteria_type": "domain_points", "criteria_value": 200, "domain": TroubleshootingDomain.FANUC_SERVO},
    {"title": "FANUC Servo Expert", "description": "Earn 500 points in FANUC Servo troubleshooting", "criteria_type": "domain_points", "criteria_value": 500, "domain": TroubleshootingDomain.FANUC_SERVO},
    {"title": "IMM Defect Initiate", "description": "Earn 50 points in IMM defect analysis", "criteria_type": "domain_points", "criteria_value": 50, "domain": TroubleshootingDomain.IMM_DEFECTS},
    {"title": "IMM Defect Technician", "description": "Earn 200 points in IMM defect analysis", "criteria_type": "domain_points", "criteria_value": 200, "domain": TroubleshootingDomain.IMM_DEFECTS},

    # Streak achievements
    {"title": "Consistent Learner", "description": "Maintain a 7-day activity streak", "criteria_type": "streak_days", "criteria_value": 7, "domain": None},
    {"title": "Dedicated Technician", "description": "Maintain a 30-day activity streak", "criteria_type": "streak_days", "criteria_value": 30, "domain": None},

    # Level achievements
    {"title": "Technician Rank", "description": "Reach Technician expertise level", "criteria_type": "level_reached", "criteria_value": 100, "domain": None},
    {"title": "Specialist Rank", "description": "Reach Specialist expertise level", "criteria_type": "level_reached", "criteria_value": 500, "domain": None},
    {"title": "Expert Rank", "description": "Reach Expert expertise level", "criteria_type": "level_reached", "criteria_value": 2000, "domain": None},
]


# =============================================================================
# SAMPLE USERS
# =============================================================================

SAMPLE_USERS = [
    {
        "user_id": "sparkone",
        "total_expertise_points": 175,
        "expertise_level": ExpertiseLevel.TECHNICIAN,
        "domain_points": {
            TroubleshootingDomain.FANUC_SERVO: 100,
            TroubleshootingDomain.IMM_DEFECTS: 50,
            TroubleshootingDomain.FANUC_MOTION: 25,
        },
        "domains_mastered": [TroubleshootingDomain.FANUC_SERVO],
        "total_sessions": 12,
        "successful_resolutions": 10,
        "avg_resolution_time_seconds": 480.0,
        "current_streak_days": 3,
        "longest_streak_days": 7,
    },
    {
        "user_id": "test_novice",
        "total_expertise_points": 25,
        "expertise_level": ExpertiseLevel.NOVICE,
        "domain_points": {
            TroubleshootingDomain.FANUC_SERVO: 25,
        },
        "domains_mastered": [],
        "total_sessions": 2,
        "successful_resolutions": 1,
        "avg_resolution_time_seconds": 900.0,
        "current_streak_days": 1,
        "longest_streak_days": 1,
    },
    {
        "user_id": "test_expert",
        "total_expertise_points": 2500,
        "expertise_level": ExpertiseLevel.EXPERT,
        "domain_points": {
            TroubleshootingDomain.FANUC_SERVO: 800,
            TroubleshootingDomain.FANUC_MOTION: 600,
            TroubleshootingDomain.IMM_DEFECTS: 500,
            TroubleshootingDomain.IMM_PROCESS: 400,
            TroubleshootingDomain.ELECTRICAL: 200,
        },
        "domains_mastered": [
            TroubleshootingDomain.FANUC_SERVO,
            TroubleshootingDomain.FANUC_MOTION,
            TroubleshootingDomain.IMM_DEFECTS,
        ],
        "total_sessions": 150,
        "successful_resolutions": 140,
        "avg_resolution_time_seconds": 240.0,
        "current_streak_days": 14,
        "longest_streak_days": 45,
    },
]


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def reset_tables(session: AsyncSession):
    """Drop and recreate troubleshooting tables."""
    print("⚠️  Resetting troubleshooting tables...")

    # Delete in reverse dependency order
    await session.execute(text("DELETE FROM task_executions"))
    await session.execute(text("DELETE FROM troubleshooting_sessions"))
    await session.execute(text("DELETE FROM workflow_tasks"))
    await session.execute(text("DELETE FROM troubleshooting_workflows"))
    await session.execute(text("DELETE FROM user_expertise"))
    await session.execute(text("DELETE FROM troubleshooting_achievements"))
    await session.commit()

    print("✅ Tables cleared")


async def create_workflows(session: AsyncSession) -> dict:
    """Create sample workflows with tasks."""
    print("\n📋 Creating sample workflows...")

    workflow_ids = {}

    for wf_data in SAMPLE_WORKFLOWS:
        # Extract tasks before creating workflow
        tasks_data = wf_data.pop("tasks", [])

        workflow = TroubleshootingWorkflow(
            id=uuid.uuid4(),
            name=wf_data["name"],
            description=wf_data["description"],
            category=wf_data["category"].value if isinstance(wf_data["category"], TroubleshootingCategory) else wf_data["category"],
            domain=wf_data["domain"].value if isinstance(wf_data["domain"], TroubleshootingDomain) else wf_data["domain"],
            expertise_points=wf_data["expertise_points"],
            estimated_duration_minutes=wf_data["estimated_duration_minutes"],
            error_code_patterns=wf_data["error_code_patterns"],
            symptom_keywords=wf_data["symptom_keywords"],
            entry_node_types=wf_data["entry_node_types"],
            target_node_types=wf_data["target_node_types"],
            traversal_mode=wf_data["traversal_mode"],
            max_hops=wf_data["max_hops"],
            beam_width=wf_data["beam_width"],
            is_active=True,
        )
        session.add(workflow)
        workflow_ids[wf_data["name"]] = workflow.id

        # Create tasks
        for task_data in tasks_data:
            exec_type = TaskExecutionType.USER_ACTION if task_data.get("type") == "user_action" else TaskExecutionType.AUTOMATIC

            task = WorkflowTask(
                id=uuid.uuid4(),
                workflow_id=workflow.id,
                name=task_data["name"],
                description=task_data.get("description", f"Execute {task_data['name']} step"),
                execution_type=exec_type.value,
                order_index=task_data["order"],
                is_required=True,
                pipeline_hook=task_data.get("pipeline_hook"),
                user_prompt=task_data.get("prompt"),
                timeout_seconds=60,
            )
            session.add(task)

        print(f"  ✅ {wf_data['name']} ({len(tasks_data)} tasks)")

    await session.commit()
    print(f"✅ Created {len(SAMPLE_WORKFLOWS)} workflows")

    return workflow_ids


async def create_achievements(session: AsyncSession):
    """Create sample achievements."""
    print("\n🏆 Creating sample achievements...")

    for ach_data in SAMPLE_ACHIEVEMENTS:
        achievement = TroubleshootingAchievement(
            id=uuid.uuid4(),
            title=ach_data["title"],
            description=ach_data["description"],
            domain=ach_data["domain"].value if isinstance(ach_data["domain"], TroubleshootingDomain) else ach_data["domain"],
            criteria_type=ach_data["criteria_type"],
            criteria_value=ach_data["criteria_value"],
            is_active=True,
        )
        session.add(achievement)
        print(f"  ✅ {ach_data['title']}")

    await session.commit()
    print(f"✅ Created {len(SAMPLE_ACHIEVEMENTS)} achievements")


async def create_user_expertise(session: AsyncSession):
    """Create sample user expertise data."""
    print("\n👤 Creating sample user expertise...")

    for user_data in SAMPLE_USERS:
        # Convert enum keys to strings for domain_points
        domain_points = {}
        for k, v in user_data["domain_points"].items():
            key = k.value if isinstance(k, TroubleshootingDomain) else k
            domain_points[key] = v

        domains_mastered = []
        for d in user_data["domains_mastered"]:
            domains_mastered.append(d.value if isinstance(d, TroubleshootingDomain) else d)

        expertise = UserExpertise(
            user_id=user_data["user_id"],
            total_expertise_points=user_data["total_expertise_points"],
            expertise_level=user_data["expertise_level"].value if isinstance(user_data["expertise_level"], ExpertiseLevel) else user_data["expertise_level"],
            domain_points=domain_points,
            domains_mastered=domains_mastered,
            total_sessions=user_data["total_sessions"],
            successful_resolutions=user_data["successful_resolutions"],
            avg_resolution_time_seconds=user_data["avg_resolution_time_seconds"],
            current_streak_days=user_data["current_streak_days"],
            longest_streak_days=user_data["longest_streak_days"],
            last_activity_date=datetime.now(timezone.utc).date(),
        )
        session.add(expertise)
        print(f"  ✅ {user_data['user_id']} ({user_data['expertise_level'].value if isinstance(user_data['expertise_level'], ExpertiseLevel) else user_data['expertise_level']}, {user_data['total_expertise_points']} pts)")

    await session.commit()
    print(f"✅ Created {len(SAMPLE_USERS)} user profiles")


async def create_sample_session(session: AsyncSession, workflow_ids: dict):
    """Create a sample completed session for testing."""
    print("\n📊 Creating sample session history...")

    # Get the SRVO-063 workflow
    workflow_id = workflow_ids.get("SRVO-063 BLAL Alarm Diagnosis")
    if not workflow_id:
        print("  ⚠️ SRVO-063 workflow not found, skipping sample session")
        return

    # Create a completed session for sparkone
    ts_session = TroubleshootingSession(
        id=uuid.uuid4(),
        user_id="sparkone",
        workflow_id=workflow_id,
        original_query="SRVO-063 low battery alarm",
        detected_error_codes=["SRVO-063"],
        detected_symptoms=["low battery", "encoder battery"],
        entry_type="error_code",
        domain=TroubleshootingDomain.FANUC_SERVO.value,
        state=SessionState.RESOLVED.value,
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        completed_at=datetime.now(timezone.utc) - timedelta(hours=1, minutes=45),
        resolution_type="self_resolved",
        total_tasks=4,
        completed_tasks=4,
        expertise_points_earned=20,
        resolution_time_seconds=900.0,
        user_rating=5,
        user_feedback="Clear instructions, battery replacement was straightforward.",
    )
    session.add(ts_session)

    # Create task executions
    task_names = [
        ("Query Analysis", "_analyze_query", 150),
        ("Technical Doc Search", "_search_technical_docs", 2500),
        ("Synthesis", "_synthesize", 3000),
        ("Battery Replacement", None, 600000),  # User action
    ]

    for i, (name, hook, duration_ms) in enumerate(task_names):
        execution = TaskExecution(
            id=uuid.uuid4(),
            session_id=ts_session.id,
            state=TaskState.COMPLETED.value,
            started_at=ts_session.started_at + timedelta(milliseconds=i * 5000),
            completed_at=ts_session.started_at + timedelta(milliseconds=i * 5000 + duration_ms),
            input_data={"task_name": name},
            output_data={"success": True},
            metrics={"latency_ms": duration_ms, "confidence": 0.85 + (i * 0.03)},
            verification_passed=True,
        )
        session.add(execution)

    await session.commit()
    print(f"  ✅ Created sample session: SRVO-063 diagnosis (resolved)")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("🔧 Troubleshooting Sample Data Creator")
    print("=" * 60)

    reset = "--reset" in sys.argv

    async with AsyncSessionLocal() as session:
        try:
            if reset:
                await reset_tables(session)

            workflow_ids = await create_workflows(session)
            await create_achievements(session)
            await create_user_expertise(session)
            await create_sample_session(session, workflow_ids)

            print("\n" + "=" * 60)
            print("✅ Sample data creation complete!")
            print("=" * 60)
            print("\nSummary:")
            print(f"  • {len(SAMPLE_WORKFLOWS)} workflows with tasks")
            print(f"  • {len(SAMPLE_ACHIEVEMENTS)} achievements")
            print(f"  • {len(SAMPLE_USERS)} user profiles")
            print(f"  • 1 sample completed session")
            print("\nTest with:")
            print("  curl http://localhost:8001/api/v1/troubleshooting/workflows")
            print("  curl 'http://localhost:8001/api/v1/troubleshooting/expertise?user_id=sparkone'")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            await session.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(main())
