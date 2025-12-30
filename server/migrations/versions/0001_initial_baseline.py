"""Initial baseline migration

This is an empty migration that marks the existing database state
as the baseline for future migrations. All tables and indexes were
created via init_database.py before Alembic was set up.

Revision ID: 0001_initial
Revises:
Create Date: 2025-12-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    This migration is a baseline marker only.

    The following tables already exist in the database:
    - memories (with pgvector embedding_vector column)
    - user_memory_settings
    - quests
    - quest_tasks
    - user_quests
    - user_tasks
    - achievements
    - user_achievements
    - user_quest_stats
    - ollama_model_specs

    The following indexes already exist:
    - idx_memories_embedding_hnsw (HNSW vector index)
    - idx_memories_tags_gin (GIN index)
    - idx_memories_entities_gin (GIN index)
    - Various composite and partial indexes

    Extensions enabled:
    - vector (pgvector)
    - uuid-ossp
    - pgcrypto
    """
    pass


def downgrade() -> None:
    """No downgrade - this is the baseline"""
    pass
