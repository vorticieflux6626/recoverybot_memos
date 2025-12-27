"""
Database Configuration for memOS Server
Handles PostgreSQL with pgvector extension setup
"""

import asyncio
import logging
from typing import AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Database URL construction
def get_database_url(async_mode: bool = False) -> str:
    """Get database URL for sync or async operations"""
    base_url = settings.database_url
    if async_mode:
        return base_url.replace("postgresql://", "postgresql+asyncpg://")
    return base_url.replace("postgresql://", "postgresql+psycopg2://")


# SQLAlchemy setup
Base = declarative_base()

# Sync engine for migrations and setup
sync_engine = create_engine(
    get_database_url(async_mode=False),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug
)

# Async engine for main operations
async_engine = create_async_engine(
    get_database_url(async_mode=True),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug
)

# Session makers
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with proper cleanup"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


def get_sync_db():
    """Get sync database session (for migrations)"""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Sync database session error: {e}")
        raise
    finally:
        db.close()


async def init_database():
    """Initialize database with required extensions and tables"""
    logger.info("Initializing memOS database...")
    
    try:
        async with async_engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Enable other useful extensions for HIPAA compliance
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            
            logger.info("Database extensions enabled successfully")
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def check_database_connection():
    """Check if database connection is working"""
    try:
        async with get_async_db() as db:
            result = await db.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    import asyncpg
    
    try:
        # Connect to default postgres database to create our database
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database="postgres"
        )
        
        # Check if database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            settings.postgres_db
        )
        
        if not db_exists:
            await conn.execute(f"CREATE DATABASE {settings.postgres_db}")
            logger.info(f"Created database: {settings.postgres_db}")
        else:
            logger.info(f"Database {settings.postgres_db} already exists")
            
        await conn.close()
        
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        raise


class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self):
        self.async_engine = async_engine
        self.sync_engine = sync_engine
    
    async def health_check(self) -> bool:
        """Check database health"""
        return await check_database_connection()
    
    async def initialize(self):
        """Initialize database with all requirements"""
        await create_database_if_not_exists()
        await init_database()
    
    async def close(self):
        """Close database connections"""
        await self.async_engine.dispose()
        self.sync_engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI endpoints"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {type(e).__name__}: {str(e)}")
            raise
        finally:
            await session.close()


if __name__ == "__main__":
    # Test database connection
    async def test_connection():
        print("Testing database connection...")
        await create_database_if_not_exists()
        await init_database()
        
        is_connected = await check_database_connection()
        print(f"Database connection: {'✓ Success' if is_connected else '✗ Failed'}")
        
        await db_manager.close()
    
    asyncio.run(test_connection())