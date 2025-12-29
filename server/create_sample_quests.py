#!/usr/bin/env python3
"""
Create Sample Quests for Recovery Bot memOS
Populates the database with initial quests for testing and demonstration
"""

import asyncio
import json
from uuid import uuid4
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from config.database import AsyncSessionLocal
from models.quest import Quest, QuestTask, QuestCategory, VerificationType
from core.quest_service_fixed import quest_service_fixed as quest_service

# Sample quests organized by category
SAMPLE_QUESTS = [
    # Daily Quests
    {
        "title": "Morning Gratitude",
        "description": "Start your day with gratitude by listing three things you're thankful for",
        "category": QuestCategory.DAILY,
        "points": 50,
        "max_active_days": 1,
        "cooldown_hours": 20,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"time_of_day": "morning", "duration_minutes": 5},
        "tasks": [
            {
                "title": "Write down three things you're grateful for",
                "description": "Take a moment to reflect on positive aspects of your life",
                "is_required": True
            }
        ]
    },
    {
        "title": "Daily Check-In",
        "description": "Complete your daily recovery check-in and mood assessment",
        "category": QuestCategory.DAILY,
        "points": 75,
        "max_active_days": 1,
        "cooldown_hours": 20,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"includes_mood_tracking": True},
        "tasks": [
            {
                "title": "Rate your mood on a scale of 1-10",
                "is_required": True
            },
            {
                "title": "Journal about your day",
                "description": "Write at least 3 sentences about your recovery journey today",
                "is_required": True
            },
            {
                "title": "Set an intention for tomorrow",
                "is_required": False
            }
        ]
    },
    {
        "title": "Healthy Meal Planning",
        "description": "Plan and prepare a nutritious meal",
        "category": QuestCategory.DAILY,
        "points": 60,
        "max_active_days": 1,
        "cooldown_hours": 12,
        "verification_type": VerificationType.PHOTO_EVIDENCE,
        "tasks": [
            {
                "title": "Plan a healthy meal",
                "is_required": True
            },
            {
                "title": "Shop for ingredients",
                "is_required": True
            },
            {
                "title": "Prepare and enjoy the meal",
                "description": "Take a photo of your prepared meal",
                "is_required": True,
                "verification_data": {"requires_photo": True}
            }
        ]
    },
    
    # Weekly Quests
    {
        "title": "Meeting Attendance",
        "description": "Attend a recovery support meeting this week",
        "category": QuestCategory.WEEKLY,
        "points": 150,
        "max_active_days": 7,
        "cooldown_hours": 144,  # 6 days
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"meeting_types": ["AA", "NA", "SMART Recovery", "Other"]},
        "tasks": [
            {
                "title": "Find a meeting to attend",
                "description": "Use the app to locate a meeting near you",
                "is_required": True
            },
            {
                "title": "Attend the meeting",
                "is_required": True
            },
            {
                "title": "Share during the meeting",
                "description": "Optional: Share your experience if comfortable",
                "is_required": False
            },
            {
                "title": "Reflect on the experience",
                "description": "Journal about what you learned",
                "is_required": True
            }
        ]
    },
    {
        "title": "Physical Wellness Week",
        "description": "Complete physical activities 3 times this week",
        "category": QuestCategory.WEEKLY,
        "points": 200,
        "max_active_days": 7,
        "cooldown_hours": 144,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "Day 1: 20 minutes of physical activity",
                "description": "Walking, running, yoga, or any exercise",
                "is_required": True
            },
            {
                "title": "Day 3: 20 minutes of physical activity",
                "is_required": True
            },
            {
                "title": "Day 5: 20 minutes of physical activity",
                "is_required": True
            },
            {
                "title": "Bonus: Try a new type of exercise",
                "is_required": False
            }
        ]
    },
    
    # Milestone Quests
    {
        "title": "30 Days of Recovery",
        "description": "Celebrate reaching 30 days in recovery",
        "category": QuestCategory.MILESTONE,
        "points": 500,
        "min_recovery_stage": "30_days",
        "verification_type": VerificationType.AUTO_VERIFY,
        "metadata": {"milestone_days": 30, "celebration": True},
        "tasks": [
            {
                "title": "Reflect on your journey",
                "description": "Write about your experience over the past 30 days",
                "is_required": True
            },
            {
                "title": "Share with someone you trust",
                "description": "Tell someone about your achievement",
                "is_required": True
            },
            {
                "title": "Reward yourself",
                "description": "Do something special (non-substance related) to celebrate",
                "is_required": True
            }
        ]
    },
    {
        "title": "90 Days Strong",
        "description": "Commemorate 90 days of recovery",
        "category": QuestCategory.MILESTONE,
        "points": 1000,
        "min_recovery_stage": "90_days",
        "verification_type": VerificationType.AUTO_VERIFY,
        "metadata": {"milestone_days": 90},
        "tasks": [
            {
                "title": "Write a letter to your past self",
                "is_required": True
            },
            {
                "title": "Create a recovery timeline",
                "description": "Document your journey with key moments",
                "is_required": True
            },
            {
                "title": "Plan your next 90 days",
                "is_required": True
            }
        ]
    },
    
    # Life Skills Quests
    {
        "title": "Budget Basics",
        "description": "Create a simple monthly budget",
        "category": QuestCategory.LIFE_SKILLS,
        "points": 150,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "List all sources of income",
                "is_required": True
            },
            {
                "title": "Track expenses for one week",
                "is_required": True
            },
            {
                "title": "Create categories for spending",
                "is_required": True
            },
            {
                "title": "Set one savings goal",
                "is_required": True
            }
        ]
    },
    {
        "title": "Job Application Workshop",
        "description": "Prepare and submit a job application",
        "category": QuestCategory.LIFE_SKILLS,
        "points": 250,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"career_focused": True},
        "tasks": [
            {
                "title": "Update or create your resume",
                "is_required": True
            },
            {
                "title": "Find 3 job openings that interest you",
                "is_required": True
            },
            {
                "title": "Submit at least one application",
                "is_required": True
            },
            {
                "title": "Practice interview questions",
                "is_required": False
            }
        ]
    },
    
    # Community Quests
    {
        "title": "Recovery Buddy",
        "description": "Connect with another person in recovery",
        "category": QuestCategory.COMMUNITY,
        "points": 100,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "Reach out to someone from your meeting",
                "is_required": True
            },
            {
                "title": "Exchange contact information",
                "is_required": True
            },
            {
                "title": "Check in with them this week",
                "is_required": True
            }
        ]
    },
    {
        "title": "Service Work",
        "description": "Volunteer to help at a recovery meeting or event",
        "category": QuestCategory.COMMUNITY,
        "points": 200,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"service_type": "meeting_support"},
        "tasks": [
            {
                "title": "Volunteer for a service position",
                "description": "Greeter, coffee maker, or cleanup",
                "is_required": True
            },
            {
                "title": "Complete your service commitment",
                "is_required": True
            },
            {
                "title": "Reflect on the experience",
                "is_required": True
            }
        ]
    },
    
    # Emergency/Crisis Quests
    {
        "title": "Crisis Safety Plan",
        "description": "Create a personal crisis intervention plan",
        "category": QuestCategory.EMERGENCY,
        "points": 300,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"crisis_preparedness": True},
        "tasks": [
            {
                "title": "List warning signs of crisis",
                "is_required": True
            },
            {
                "title": "Identify coping strategies",
                "is_required": True
            },
            {
                "title": "List emergency contacts",
                "description": "Include sponsor, counselor, crisis hotline",
                "is_required": True
            },
            {
                "title": "Share plan with trusted person",
                "is_required": True
            }
        ]
    },
    
    # Wellness Quests
    {
        "title": "Mindfulness Journey",
        "description": "Practice mindfulness meditation for 7 days",
        "category": QuestCategory.WELLNESS,
        "points": 175,
        "max_active_days": 10,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "Day 1: 5-minute breathing exercise",
                "is_required": True
            },
            {
                "title": "Day 2: Body scan meditation",
                "is_required": True
            },
            {
                "title": "Day 3: Mindful walking",
                "is_required": True
            },
            {
                "title": "Day 4: Loving-kindness meditation",
                "is_required": True
            },
            {
                "title": "Day 5: Mindful eating",
                "is_required": True
            },
            {
                "title": "Day 6: Gratitude meditation",
                "is_required": True
            },
            {
                "title": "Day 7: Choose your favorite practice",
                "is_required": True
            }
        ]
    },
    {
        "title": "Sleep Hygiene Challenge",
        "description": "Improve your sleep habits over one week",
        "category": QuestCategory.WELLNESS,
        "points": 150,
        "max_active_days": 7,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "Set consistent sleep/wake times",
                "is_required": True
            },
            {
                "title": "Create bedtime routine",
                "description": "No screens 30 minutes before bed",
                "is_required": True
            },
            {
                "title": "Track sleep quality for 7 nights",
                "is_required": True
            },
            {
                "title": "Adjust environment",
                "description": "Temperature, darkness, comfort",
                "is_required": False
            }
        ]
    },
    
    # Spiritual Quests
    {
        "title": "Spiritual Exploration",
        "description": "Explore spiritual practices that resonate with you",
        "category": QuestCategory.SPIRITUAL,
        "points": 125,
        "verification_type": VerificationType.SELF_REPORT,
        "tasks": [
            {
                "title": "Try a new spiritual practice",
                "description": "Prayer, meditation, nature walk, etc.",
                "is_required": True
            },
            {
                "title": "Read spiritual literature",
                "description": "Any text that inspires you",
                "is_required": True
            },
            {
                "title": "Reflect on your spiritual journey",
                "is_required": True
            }
        ]
    },
    {
        "title": "Forgiveness Practice",
        "description": "Work on forgiveness - of self and others",
        "category": QuestCategory.SPIRITUAL,
        "points": 200,
        "verification_type": VerificationType.SELF_REPORT,
        "metadata": {"emotional_intensity": "high"},
        "tasks": [
            {
                "title": "List resentments you're holding",
                "is_required": True
            },
            {
                "title": "Write a forgiveness letter (don't send)",
                "is_required": True
            },
            {
                "title": "Practice self-forgiveness meditation",
                "is_required": True
            },
            {
                "title": "Share with sponsor or counselor",
                "is_required": False
            }
        ]
    }
]


async def create_sample_quests():
    """Create all sample quests in the database"""
    print("🎯 Creating Sample Quests for Recovery Bot")
    print("=" * 60)
    
    created_count = 0
    
    try:
        async with AsyncSessionLocal() as session:
            for quest_data in SAMPLE_QUESTS:
                try:
                    # Extract tasks
                    tasks = quest_data.pop("tasks", [])
                    
                    # Create quest
                    quest = Quest(
                        id=uuid4(),
                        title=quest_data["title"],
                        description=quest_data["description"],
                        category=quest_data["category"],
                        points=quest_data.get("points", 0),
                        min_recovery_stage=quest_data.get("min_recovery_stage"),
                        max_active_days=quest_data.get("max_active_days"),
                        cooldown_hours=quest_data.get("cooldown_hours", 0),
                        prerequisites=quest_data.get("prerequisites", []),
                        verification_type=quest_data.get("verification_type", VerificationType.SELF_REPORT),
                        quest_metadata=quest_data.get("metadata", {}),
                        is_active=True
                    )
                    
                    session.add(quest)
                    
                    # Create tasks
                    for idx, task_data in enumerate(tasks):
                        task = QuestTask(
                            id=uuid4(),
                            quest_id=quest.id,
                            title=task_data["title"],
                            description=task_data.get("description"),
                            order_index=idx,
                            is_required=task_data.get("is_required", True),
                            verification_data=task_data.get("verification_data", {})
                        )
                        session.add(task)
                    
                    await session.commit()
                    created_count += 1
                    
                    print(f"✅ Created: {quest.title} ({quest.category}) - {len(tasks)} tasks")
                    
                except Exception as e:
                    print(f"❌ Failed to create quest '{quest_data.get('title', 'Unknown')}': {e}")
                    await session.rollback()
            
        print("\n" + "=" * 60)
        print(f"🎉 Created {created_count} sample quests!")
        
        # Verify quest counts by category
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, func
            
            result = await session.execute(
                select(Quest.category, func.count(Quest.id))
                .group_by(Quest.category)
                .order_by(Quest.category)
            )
            
            print("\n📊 Quests by Category:")
            for category, count in result.fetchall():
                print(f"   {category}: {count} quests")
        
        return created_count
        
    except Exception as e:
        print(f"❌ Error creating sample quests: {e}")
        import traceback
        traceback.print_exc()
        return 0


async def list_existing_quests():
    """List all existing quests in the database"""
    print("\n📋 Existing Quests in Database:")
    print("=" * 60)
    
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            result = await session.execute(
                select(Quest).order_by(Quest.category, Quest.title)
            )
            quests = result.scalars().all()
            
            if not quests:
                print("No quests found in database.")
                return
            
            current_category = None
            for quest in quests:
                if quest.category != current_category:
                    current_category = quest.category
                    print(f"\n{current_category.upper()}:")
                
                # Count tasks
                task_count = len(quest.tasks) if quest.tasks else 0
                print(f"  - {quest.title} ({quest.points} pts, {task_count} tasks)")
                
    except Exception as e:
        print(f"Error listing quests: {e}")


async def clear_existing_quests():
    """Clear all existing quests from the database"""
    print("\n🗑️  Clearing existing quests...")
    
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete
            
            # Delete all quests (cascades to tasks and user quests)
            await session.execute(delete(Quest))
            await session.commit()
            
            print("✅ All existing quests cleared")
            
    except Exception as e:
        print(f"❌ Error clearing quests: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample Quest Management")
    parser.add_argument("--list", action="store_true", help="List existing quests")
    parser.add_argument("--clear", action="store_true", help="Clear all existing quests")
    parser.add_argument("--create", action="store_true", help="Create sample quests")
    
    args = parser.parse_args()
    
    if args.list:
        asyncio.run(list_existing_quests())
    elif args.clear:
        asyncio.run(clear_existing_quests())
    elif args.create:
        asyncio.run(create_sample_quests())
    else:
        # Default: create sample quests
        asyncio.run(create_sample_quests())