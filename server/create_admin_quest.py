#!/usr/bin/env python3
"""
Create a quest via admin API
For testing quest creation through API endpoints
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8001"

# Sample quest for testing
TEST_QUEST = {
    "title": "Recovery Check-In Challenge",
    "description": "Complete daily check-ins for 7 days straight",
    "category": "daily",
    "points": 300,
    "max_active_days": 10,
    "cooldown_hours": 168,  # 7 days
    "verification_type": "self_report",
    "metadata": {
        "streak_required": 7,
        "reward_type": "badge"
    },
    "tasks": [
        {
            "title": "Day 1: Morning check-in",
            "description": "Complete your morning recovery check-in",
            "is_required": True
        },
        {
            "title": "Day 2: Evening reflection",
            "description": "Reflect on your day and journal your thoughts",
            "is_required": True
        },
        {
            "title": "Day 3: Gratitude practice",
            "description": "List 5 things you're grateful for",
            "is_required": True
        },
        {
            "title": "Day 4: Connect with support",
            "description": "Reach out to your sponsor or support group",
            "is_required": True
        },
        {
            "title": "Day 5: Self-care activity",
            "description": "Do something nice for yourself",
            "is_required": True
        },
        {
            "title": "Day 6: Share your progress",
            "description": "Share your journey with someone",
            "is_required": True
        },
        {
            "title": "Day 7: Celebrate your streak!",
            "description": "Acknowledge your achievement",
            "is_required": True
        }
    ]
}


async def create_quest():
    print("🎯 Creating Quest via Admin API")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # Create quest
        print(f"Quest: {TEST_QUEST['title']}")
        print(f"Category: {TEST_QUEST['category']}")
        print(f"Points: {TEST_QUEST['points']}")
        print(f"Tasks: {len(TEST_QUEST['tasks'])}")
        
        response = await client.post(
            f"{BASE_URL}/api/v1/quests/admin/create",
            json=TEST_QUEST
        )
        
        if response.status_code == 200:
            quest = response.json()
            print("\n✅ Quest created successfully!")
            print(f"Quest ID: {quest['id']}")
            print(f"Created at: {quest['created_at']}")
            
            # Verify quest is available
            print("\n📋 Verifying quest availability...")
            response = await client.get(
                f"{BASE_URL}/api/v1/quests/available",
                params={"user_id": "test_user", "limit": 10}
            )
            
            if response.status_code == 200:
                quests = response.json()
                matching = [q for q in quests if q['id'] == quest['id']]
                if matching:
                    print("✅ Quest is available for users!")
                else:
                    print("⚠️  Quest created but not showing as available")
            else:
                print(f"❌ Failed to verify availability: {response.status_code}")
                
        else:
            print(f"❌ Failed to create quest: {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(create_quest())