# TELEPHONE.md - Communication File for memOS Server Team

> **Updated**: 2025-12-30 | **Parent**: [CLAUDE.md](./CLAUDE.md) | **Status**: Historical (July 2025)

## Current Status: REST API Complete ✅

The core REST API for memory management is now complete and tested. All endpoints are working with:
- HIPAA-compliant encryption and audit logging
- Semantic search via Ollama embeddings
- User settings and consent management
- PostgreSQL with pgvector for similarity search

## Next Phase: Quest & Task Management System 🎯

### Overview
We need to implement a CMS-based quest and task management system to help Recovery Bot users track their recovery journey through gamified goals and achievements.

### Server Requirements for Quest System

#### 1. Database Schema (PostgreSQL)

```sql
-- Quest definitions (managed by CMS)
CREATE TABLE quests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL, -- daily, weekly, milestone, life_skills
    points INTEGER DEFAULT 0,
    min_recovery_stage VARCHAR(50), -- detox, early_recovery, maintenance, etc.
    max_active_days INTEGER, -- how long quest stays active
    cooldown_hours INTEGER, -- time before quest can be repeated
    prerequisites JSON, -- array of quest IDs that must be completed first
    verification_type VARCHAR(50), -- self_report, peer_verify, counselor_approve, auto_verify
    metadata JSON, -- additional quest-specific data
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual tasks within quests
CREATE TABLE quest_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quest_id UUID REFERENCES quests(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    order_index INTEGER NOT NULL,
    is_required BOOLEAN DEFAULT true,
    verification_data JSON, -- what's needed to verify completion
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User quest assignments and progress
CREATE TABLE user_quests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    quest_id UUID REFERENCES quests(id),
    state VARCHAR(50) NOT NULL, -- assigned, in_progress, completed, verified, rewarded
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    verified_at TIMESTAMP,
    verified_by VARCHAR(255), -- user_id of verifier if applicable
    progress_data JSON, -- quest-specific progress tracking
    points_earned INTEGER DEFAULT 0,
    UNIQUE(user_id, quest_id, started_at) -- prevent duplicate active quests
);

-- Task completion tracking
CREATE TABLE user_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_quest_id UUID REFERENCES user_quests(id) ON DELETE CASCADE,
    task_id UUID REFERENCES quest_tasks(id),
    user_id VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL, -- pending, completed, skipped
    completed_at TIMESTAMP,
    evidence_data JSON, -- photos, GPS coords, text notes, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Achievement definitions
CREATE TABLE achievements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    icon_url VARCHAR(500),
    category VARCHAR(50),
    criteria_type VARCHAR(50), -- points_earned, quests_completed, streak_days, etc.
    criteria_value INTEGER,
    criteria_data JSON, -- complex criteria like "complete 5 daily quests"
    badge_color VARCHAR(7), -- hex color
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User achievement tracking
CREATE TABLE user_achievements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    achievement_id UUID REFERENCES achievements(id),
    earned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    points_awarded INTEGER DEFAULT 0,
    UNIQUE(user_id, achievement_id)
);

-- Daily streaks and statistics
CREATE TABLE user_quest_stats (
    user_id VARCHAR(255) PRIMARY KEY,
    total_points INTEGER DEFAULT 0,
    current_streak_days INTEGER DEFAULT 0,
    longest_streak_days INTEGER DEFAULT 0,
    last_activity_date DATE,
    total_quests_completed INTEGER DEFAULT 0,
    level VARCHAR(50) DEFAULT 'newcomer',
    weekly_points INTEGER DEFAULT 0,
    monthly_points INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_quests_category ON quests(category) WHERE is_active = true;
CREATE INDEX idx_user_quests_user_state ON user_quests(user_id, state);
CREATE INDEX idx_user_quests_completed ON user_quests(completed_at) WHERE completed_at IS NOT NULL;
CREATE INDEX idx_user_tasks_state ON user_tasks(user_id, state);
CREATE INDEX idx_user_achievements_user ON user_achievements(user_id);
CREATE INDEX idx_user_quest_stats_points ON user_quest_stats(total_points DESC);
```

#### 2. API Endpoints Needed

```python
# Quest Management (CMS/Admin)
POST   /api/v1/admin/quests                 # Create new quest
PUT    /api/v1/admin/quests/{id}           # Update quest
DELETE /api/v1/admin/quests/{id}           # Deactivate quest
POST   /api/v1/admin/quests/{id}/publish   # Make quest available

# Quest Discovery & Assignment
GET    /api/v1/quests/available            # Get quests user can start
GET    /api/v1/quests/categories           # List quest categories
GET    /api/v1/quests/{id}                 # Get quest details
POST   /api/v1/quests/{id}/assign          # Start a quest

# Quest Progress
GET    /api/v1/users/{user_id}/quests     # Get user's active/completed quests
PUT    /api/v1/user-quests/{id}/progress   # Update quest progress
POST   /api/v1/user-tasks/{id}/complete    # Mark task complete
POST   /api/v1/user-quests/{id}/complete   # Submit quest for completion
POST   /api/v1/user-quests/{id}/verify     # Counselor verification

# Achievements & Stats
GET    /api/v1/users/{user_id}/achievements # Get user achievements
GET    /api/v1/users/{user_id}/stats       # Get user statistics
GET    /api/v1/achievements/available       # Get available achievements
GET    /api/v1/leaderboard                  # Community leaderboard (anonymized)

# Daily Activities
GET    /api/v1/users/{user_id}/daily       # Today's tasks & quests
POST   /api/v1/users/{user_id}/checkin     # Daily check-in (maintains streaks)
```

#### 3. Service Layer Requirements

```python
# QuestService
- create_quest(quest_data) -> Quest
- get_available_quests(user_id, filters) -> List[Quest]
- assign_quest(user_id, quest_id) -> UserQuest
- check_prerequisites(user_id, quest_id) -> bool
- calculate_quest_points(quest, completion_data) -> int

# ProgressService  
- update_task_progress(user_id, task_id, evidence) -> UserTask
- validate_quest_completion(user_quest_id) -> bool
- calculate_streak(user_id) -> int
- check_daily_goals(user_id) -> Dict

# AchievementService
- check_achievements(user_id) -> List[Achievement]
- award_achievement(user_id, achievement_id) -> UserAchievement
- calculate_user_level(total_points) -> str

# VerificationService
- submit_evidence(task_id, evidence_data) -> bool
- peer_verify(user_quest_id, verifier_id) -> bool
- counselor_approve(user_quest_id, counselor_id) -> bool
- auto_verify(task_id, criteria) -> bool

# NotificationService (integrate with existing)
- send_quest_reminder(user_id, quest_id)
- send_achievement_notification(user_id, achievement_id)
- send_streak_reminder(user_id, streak_days)
```

#### 4. Integration Points

1. **Memory Service Integration**
   - Store quest completions as memories
   - Link achievements to memory milestones
   - Use memory search for quest recommendations

2. **Authentication/User Service**
   - Extend user settings for quest preferences
   - Add counselor role for verification
   - Track user recovery stage for quest matching

3. **Notification System**
   - Daily quest reminders
   - Achievement celebrations
   - Streak maintenance alerts

#### 5. CMS Requirements

For the admin interface, we need:
- Quest builder with task management
- Template system for common quest types
- Analytics dashboard for quest performance
- User progress monitoring
- Bulk quest assignment tools

#### 6. Security & Privacy Considerations

- Anonymized leaderboards (no real names)
- Optional privacy settings for achievements
- HIPAA compliance for quest data
- Audit logging for all quest activities
- Secure evidence storage (encrypted)

#### 7. Performance Requirements

- Quest queries < 100ms
- Bulk progress updates for offline sync
- Caching for frequently accessed quests
- Background jobs for achievement checking
- Optimistic UI updates with eventual consistency

### Priority Order for Implementation

1. **Phase 1** (Week 1): Database schema and basic CRUD
2. **Phase 2** (Week 2): Quest assignment and progress tracking
3. **Phase 3** (Week 3): Achievements and statistics
4. **Phase 4** (Week 4): Verification system and notifications

### Questions for Android Team

1. How should we handle offline quest progress?
2. What's the preferred UI for quest selection?
3. Should we support quest sharing between users?
4. How detailed should progress tracking be?
5. What notification preferences do we need?

---

*Last Updated: 2025-07-13 by Claude*
*Next Sync Meeting: TBD*

## Quest System Implementation Status

### ✅ Completed (2025-07-13)

1. **Database Schema** - All quest-related tables created:
   - `quests` - Quest templates with categories and rewards
   - `quest_tasks` - Individual tasks within quests
   - `user_quests` - User's quest assignments and progress
   - `user_tasks` - Task completion tracking
   - `achievements` - Achievement definitions
   - `user_achievements` - Earned achievements
   - `user_quest_stats` - User statistics and levels

2. **Core Services** - Quest management service implemented:
   - Quest assignment with prerequisite checking
   - Progress tracking and task completion
   - Achievement checking and reward system
   - Streak tracking and level calculation
   - Integration with memory service for milestones

3. **REST API Endpoints** - Complete quest API:
   - `GET /api/v1/quests/categories` - List quest categories
   - `GET /api/v1/quests/available` - Get available quests for user
   - `POST /api/v1/quests/{quest_id}/assign` - Assign quest to user
   - `GET /api/v1/quests/users/{user_id}/quests` - Get user's quests
   - `GET /api/v1/quests/users/{user_id}/daily` - Daily quest view
   - `GET /api/v1/quests/users/{user_id}/stats` - User statistics
   - `PUT /api/v1/quests/tasks/{task_id}/complete` - Complete task

4. **Sample Data** - Created 16 sample quests across categories:
   - Daily (3): Morning Gratitude, Daily Check-In, Healthy Meal Planning
   - Weekly (2): Meeting Attendance, Physical Wellness Week
   - Milestone (2): 30 Days of Recovery, 90 Days Strong
   - Life Skills (2): Budget Basics, Job Application Workshop
   - Community (2): Recovery Buddy, Service Work
   - Emergency (1): Crisis Safety Plan
   - Wellness (2): Mindfulness Journey, Sleep Hygiene Challenge
   - Spiritual (2): Spiritual Exploration, Forgiveness Practice

### 🔧 Known Issues

1. **Event Loop Errors** - Some async operations have event loop conflicts
   - Affects user stats endpoint
   - Workaround: Use synchronous operations where possible

2. **Mem0 Client Warnings** - Configuration warnings on startup
   - Does not affect quest functionality
   - Can be ignored for now

### 📱 Android Integration Guide

#### Quest List Screen
```kotlin
// Fetch available quests
GET /api/v1/quests/available?user_id={userId}&category={category}

// Display in RecyclerView with categories
data class Quest(
    val id: String,
    val title: String,
    val description: String,
    val category: String,
    val points: Int,
    val taskCount: Int
)
```

#### Quest Assignment
```kotlin
// Assign quest to user
POST /api/v1/quests/{questId}/assign?user_id={userId}

// Handle response
data class UserQuest(
    val id: String,
    val quest: Quest,
    val state: String,
    val progressPercentage: Float,
    val tasksCompleted: Int,
    val totalTasks: Int
)
```

#### Progress Tracking
```kotlin
// Complete a task
PUT /api/v1/quests/tasks/{taskId}/complete?user_id={userId}
Body: { "evidence_data": {} }

// Get user stats
GET /api/v1/quests/users/{userId}/stats
```

### 🚀 Next Steps for Android

1. **UI Components**:
   - Quest category tabs/chips
   - Quest cards with progress indicators
   - Task checklist within quest details
   - User level/points display in app bar

2. **State Management**:
   - Cache active quests locally
   - Sync progress on app resume
   - Handle offline task completion

3. **Gamification Elements**:
   - Achievement notifications
   - Streak counter widget
   - Level up animations
   - Points earned toast messages

4. **Testing**:
   - Test quest assignment flow
   - Verify task completion updates
   - Check streak calculations
   - Validate achievement triggers

## Authentication System Analysis (2025-07-13)

### 🔴 CRITICAL: Missing Authentication Layer

The memOS server currently has **NO authentication implementation** despite having JWT configuration in settings.

#### Current State:
- ❌ No auth endpoints (`/api/v1/auth/login`, `/register`, `/refresh`, `/logout`)
- ❌ No authentication middleware or guards
- ❌ All endpoints accept `user_id` as query parameter without validation
- ❌ No Bearer token validation despite Android client sending them
- ⚠️  JWT settings configured but unused

#### Security Implications:
- Any client can access any user's data by providing their user_id
- No session management or token validation
- Complete bypass of intended security model

### Proposed Authentication Architecture

#### Option 1: Centralized Auth (Recommended)
```
Android App → PHP Backend (auth) → JWT Token
     ↓              ↓
     └──────→ memOS Server (validate JWT)
```

Benefits:
- Single source of truth for users
- PHP backend already has user management
- Consistent auth across all services

Implementation:
1. PHP backend issues JWT tokens with shared secret
2. memOS validates tokens using same secret
3. Extract user_id from validated token
4. Remove user_id query parameters

#### Option 2: memOS Native Auth
```python
# Add to memOS server
@router.post("/api/v1/auth/login")
async def login(credentials: LoginCredentials):
    # Validate against PHP backend API
    user = await validate_with_php_backend(credentials)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    
    # Generate JWT
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}
```

#### Option 3: Service Mesh Pattern
- Keep memOS internal only
- PHP backend proxies all memOS requests
- Android only talks to PHP backend
- Inter-service auth between PHP↔memOS

### Immediate Security Fixes Needed

1. **Add JWT Validation Middleware**:
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# Apply to all endpoints
@router.get("/api/v1/memory/list/{user_id}")
async def list_memories(
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(403, "Cannot access other user's data")
```

2. **Remove user_id Query Parameters**:
- Get user_id from JWT token instead
- Prevents user impersonation
- Maintains REST principles

3. **Add Rate Limiting**:
- Prevent brute force attacks
- Use Redis for distributed rate limiting
- Apply per-user and per-IP limits

### Integration Timeline

- **Week 1**: Implement JWT validation middleware
- **Week 2**: Add auth endpoints or proxy system  
- **Week 3**: Remove query param auth, test with Android
- **Week 4**: Security audit and penetration testing

### Questions for PHP Backend Team

1. Can we share the JWT secret key between services?
2. What user fields should be in the JWT payload?
3. Should memOS validate users against PHP backend?
4. Preferred token expiration and refresh strategy?
5. How to handle service-to-service authentication?

---
*Authentication analysis added by memOS Agent*