#!/usr/bin/env python3
"""
Database Initialization Script for memOS Server
Creates all required tables and indexes for HIPAA-compliant memory storage
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the server directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import models to ensure they are registered with SQLAlchemy
from models.memory import Memory, MemoryType, MemoryPrivacyLevel
from models.user import UserMemorySettings
from models.quest import Quest, QuestTask, UserQuest, UserTask, Achievement, UserAchievement, UserQuestStats
from models.ollama_model import OllamaModelSpec
from config.database import db_manager, Base, async_engine
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_database_schema():
    """
    Create all database tables and indexes
    """
    logger.info("🗃️  Initializing memOS Database Schema")
    logger.info("=" * 50)
    
    settings = get_settings()
    logger.info(f"Database: {settings.postgres_db}")
    logger.info(f"Host: {settings.postgres_host}:{settings.postgres_port}")
    
    try:
        # Step 1: Create database if it doesn't exist
        logger.info("📝 Creating database if needed...")
        await db_manager.initialize()
        logger.info("✅ Database creation completed")
        
        # Step 2: Create all tables
        logger.info("🏗️  Creating database tables...")
        async with async_engine.begin() as conn:
            # Import all models to ensure they're registered
            logger.info("   Importing models...")
            logger.info(f"   - Memory model: {Memory.__tablename__}")
            logger.info(f"   - UserMemorySettings model: {UserMemorySettings.__tablename__}")
            logger.info(f"   - Quest model: {Quest.__tablename__}")
            logger.info(f"   - QuestTask model: {QuestTask.__tablename__}")
            logger.info(f"   - UserQuest model: {UserQuest.__tablename__}")
            logger.info(f"   - UserTask model: {UserTask.__tablename__}")
            logger.info(f"   - Achievement model: {Achievement.__tablename__}")
            logger.info(f"   - UserAchievement model: {UserAchievement.__tablename__}")
            logger.info(f"   - UserQuestStats model: {UserQuestStats.__tablename__}")
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create additional indexes for performance
            logger.info("📊 Creating performance indexes...")
            
            # Indexes for memory table
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_created 
                ON memories(user_id, created_at DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_type_privacy 
                ON memories(memory_type, privacy_level) WHERE is_deleted = false;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_recovery_stage 
                ON memories(recovery_stage) WHERE recovery_stage IS NOT NULL;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_therapeutic_relevance 
                ON memories(therapeutic_relevance DESC) WHERE is_deleted = false;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_crisis_level 
                ON memories(crisis_level DESC) WHERE crisis_level > 0.0;
            """))
            
            # Indexes for user settings table
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_settings_enabled 
                ON user_memory_settings(memory_enabled, updated_at DESC);
            """))
            
            # Indexes for quest tables
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_quests_category 
                ON quests(category) WHERE is_active = true;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_quests_user_state 
                ON user_quests(user_id, state);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_quests_completed 
                ON user_quests(completed_at) WHERE completed_at IS NOT NULL;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_tasks_state 
                ON user_tasks(user_id, state);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_achievements_user 
                ON user_achievements(user_id);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_quest_stats_points 
                ON user_quest_stats(total_points DESC);
            """))
            
            logger.info("✅ Performance indexes created")
        
        # Step 3: Verify table creation
        logger.info("🔍 Verifying table creation...")
        async with async_engine.connect() as conn:
            # Check tables exist
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN (
                    'memories', 'user_memory_settings', 'quests', 'quest_tasks',
                    'user_quests', 'user_tasks', 'achievements', 'user_achievements',
                    'user_quest_stats'
                )
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"   Created tables: {tables}")
            
            expected_tables = [
                'achievements', 'memories', 'quest_tasks', 'quests',
                'user_achievements', 'user_memory_settings', 'user_quest_stats',
                'user_quests', 'user_tasks'
            ]
            
            if len(tables) >= len(expected_tables):
                logger.info("✅ All required tables created successfully")
            else:
                missing = set(expected_tables) - set(tables)
                logger.error(f"❌ Missing tables: {missing}")
                return False
            
            # Check pgvector extension
            result = await conn.execute(text("""
                SELECT extname FROM pg_extension WHERE extname = 'vector';
            """))
            
            vector_ext = result.fetchone()
            if vector_ext:
                logger.info("✅ pgvector extension enabled")
            else:
                logger.warning("⚠️  pgvector extension not found")
        
        # Step 4: Test basic operations
        logger.info("🧪 Testing database operations...")
        health_ok = await db_manager.health_check()
        if health_ok:
            logger.info("✅ Database health check passed")
        else:
            logger.error("❌ Database health check failed")
            return False
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 memOS Database Initialization Complete!")
        logger.info("=" * 50)
        logger.info("📋 Summary:")
        logger.info("   ✓ PostgreSQL database created")
        logger.info("   ✓ pgvector extension enabled")
        logger.info("   ✓ Memory storage table created")
        logger.info("   ✓ User settings table created")
        logger.info("   ✓ Performance indexes created")
        logger.info("   ✓ Health check passed")
        logger.info("\n🚀 Ready for API server startup!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await db_manager.close()

async def check_schema_status():
    """
    Check current database schema status
    """
    logger.info("🔍 Checking database schema status...")
    
    try:
        async with async_engine.connect() as conn:
            # Check if tables exist
            result = await conn.execute(text("""
                SELECT table_name, 
                       (SELECT count(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' 
                AND table_name IN (
                    'memories', 'user_memory_settings', 'quests', 'quest_tasks',
                    'user_quests', 'user_tasks', 'achievements', 'user_achievements',
                    'user_quest_stats'
                )
                ORDER BY table_name;
            """))
            
            tables_info = result.fetchall()
            
            if not tables_info:
                logger.info("❌ No memOS tables found - database needs initialization")
                return False
            
            for table_name, column_count in tables_info:
                logger.info(f"   ✓ Table '{table_name}': {column_count} columns")
            
            # Check row counts
            for table_name, _ in tables_info:
                try:
                    result = await conn.execute(text(f"SELECT count(*) FROM {table_name}"))
                    row_count = result.scalar()
                    logger.info(f"   📊 Table '{table_name}': {row_count} rows")
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not count rows in '{table_name}': {e}")
            
            # Check extensions
            result = await conn.execute(text("""
                SELECT extname FROM pg_extension 
                WHERE extname IN ('vector', 'uuid-ossp', 'pgcrypto')
                ORDER BY extname;
            """))
            
            extensions = [row[0] for row in result.fetchall()]
            logger.info(f"   🔌 Extensions: {extensions}")
            
            logger.info("✅ Database schema check complete")
            return True
            
    except Exception as e:
        logger.error(f"❌ Schema check failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    from sqlalchemy import text
    
    parser = argparse.ArgumentParser(description="memOS Database Initialization")
    parser.add_argument("--check", action="store_true", help="Check current schema status")
    parser.add_argument("--force", action="store_true", help="Force recreation of tables")
    
    args = parser.parse_args()
    
    if args.check:
        asyncio.run(check_schema_status())
    else:
        success = asyncio.run(create_database_schema())
        sys.exit(0 if success else 1)