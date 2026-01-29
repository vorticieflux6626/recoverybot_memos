"""Quest System to Troubleshooting Task Tracker Migration

Transforms the quest system from recovery-focused gamification into
a troubleshooting graph traversal task tracker for industrial diagnostics.

New tables:
- troubleshooting_workflows (replaces quests)
- workflow_tasks (replaces quest_tasks)
- troubleshooting_sessions (replaces user_quests)
- task_executions (replaces user_tasks)
- user_expertise (replaces user_quest_stats)
- troubleshooting_achievements (replaces achievements)
- user_troubleshooting_achievements (replaces user_achievements)

See: QUEST_SYSTEM_REIMPLEMENTATION_PLAN.md for full architecture.

Revision ID: 0002_troubleshooting
Revises: 0001_initial
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON


# revision identifiers, used by Alembic.
revision: str = '0002_troubleshooting'
down_revision: Union[str, None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create troubleshooting task tracker tables.

    Old quest tables are preserved for backward compatibility during
    the migration period. They will be dropped in a future migration
    once all clients have migrated to the new API.
    """

    # =========================================================================
    # Table: troubleshooting_workflows (replaces quests)
    # =========================================================================
    op.create_table(
        'troubleshooting_workflows',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('category', sa.String(50), nullable=False),

        # Graph integration
        sa.Column('entry_node_types', JSON, server_default='[]'),
        sa.Column('target_node_types', JSON, server_default='[]'),
        sa.Column('traversal_mode', sa.String(50), server_default='semantic_astar'),
        sa.Column('max_hops', sa.Integer, server_default='5'),
        sa.Column('beam_width', sa.Integer, server_default='10'),

        # Domain and expertise
        sa.Column('domain', sa.String(100)),
        sa.Column('expertise_points', sa.Integer, server_default='10'),
        sa.Column('estimated_duration_minutes', sa.Integer),

        # Entry patterns for auto-detection
        sa.Column('error_code_patterns', JSON, server_default='[]'),
        sa.Column('symptom_keywords', JSON, server_default='[]'),

        # Lifecycle
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Index for domain-based queries
    op.create_index('idx_workflow_domain', 'troubleshooting_workflows', ['domain'])
    op.create_index('idx_workflow_category', 'troubleshooting_workflows', ['category'])
    op.create_index('idx_workflow_active', 'troubleshooting_workflows', ['is_active'])

    # =========================================================================
    # Table: workflow_tasks (replaces quest_tasks)
    # =========================================================================
    op.create_table(
        'workflow_tasks',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('workflow_id', UUID(as_uuid=True),
                  sa.ForeignKey('troubleshooting_workflows.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('execution_type', sa.String(50), nullable=False,
                  server_default='automatic'),
        sa.Column('order_index', sa.Integer, nullable=False),
        sa.Column('is_required', sa.Boolean, server_default='true'),

        # Pipeline integration
        sa.Column('pipeline_hook', sa.String(100)),
        sa.Column('timeout_seconds', sa.Integer, server_default='60'),

        # Verification criteria
        sa.Column('verification_criteria', JSON, server_default='{}'),

        # User action config
        sa.Column('user_prompt', sa.Text),
        sa.Column('user_options', JSON, server_default='[]'),

        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Index for task ordering
    op.create_index('idx_task_workflow_order', 'workflow_tasks',
                    ['workflow_id', 'order_index'])

    # =========================================================================
    # Table: troubleshooting_sessions (replaces user_quests)
    # =========================================================================
    op.create_table(
        'troubleshooting_sessions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('workflow_id', UUID(as_uuid=True),
                  sa.ForeignKey('troubleshooting_workflows.id')),

        # Query context
        sa.Column('original_query', sa.Text, nullable=False),
        sa.Column('detected_error_codes', JSON, server_default='[]'),
        sa.Column('detected_symptoms', JSON, server_default='[]'),
        sa.Column('entry_type', sa.String(50)),
        sa.Column('domain', sa.String(100)),

        # State tracking
        sa.Column('state', sa.String(50), server_default='initiated'),
        sa.Column('started_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('resolution_type', sa.String(50)),

        # Graph traversal tracking
        sa.Column('selected_path_id', sa.String(100)),
        sa.Column('paths_presented', JSON, server_default='[]'),
        sa.Column('current_step_index', sa.Integer, server_default='0'),
        sa.Column('total_steps', sa.Integer, server_default='0'),
        sa.Column('completed_steps', JSON, server_default='[]'),

        # Task tracking
        sa.Column('total_tasks', sa.Integer, server_default='0'),
        sa.Column('completed_tasks', sa.Integer, server_default='0'),

        # Expertise & metrics
        sa.Column('expertise_points_earned', sa.Integer, server_default='0'),
        sa.Column('resolution_time_seconds', sa.Float),

        # User feedback
        sa.Column('user_rating', sa.Integer),
        sa.Column('user_feedback', sa.Text),

        # Metadata
        sa.Column('session_metadata', JSON, server_default='{}'),
    )

    # Indexes for session queries
    op.create_index('idx_session_user_state', 'troubleshooting_sessions',
                    ['user_id', 'state'])
    op.create_index('idx_session_domain', 'troubleshooting_sessions', ['domain'])
    op.create_index('idx_session_user_id', 'troubleshooting_sessions', ['user_id'])

    # =========================================================================
    # Table: task_executions (replaces user_tasks)
    # =========================================================================
    op.create_table(
        'task_executions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', UUID(as_uuid=True),
                  sa.ForeignKey('troubleshooting_sessions.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('task_id', UUID(as_uuid=True),
                  sa.ForeignKey('workflow_tasks.id')),

        # Execution state
        sa.Column('state', sa.String(50), server_default='pending'),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),

        # Execution data
        sa.Column('input_data', JSON, server_default='{}'),
        sa.Column('output_data', JSON, server_default='{}'),
        sa.Column('metrics', JSON, server_default='{}'),

        # Verification
        sa.Column('verification_passed', sa.Boolean),
        sa.Column('verification_details', JSON, server_default='{}'),

        # User interaction
        sa.Column('user_input', JSON),
        sa.Column('user_notes', sa.Text),

        # Error tracking
        sa.Column('error_message', sa.Text),
        sa.Column('error_recoverable', sa.Boolean, server_default='true'),
    )

    # Index for execution queries
    op.create_index('idx_execution_session_state', 'task_executions',
                    ['session_id', 'state'])

    # =========================================================================
    # Table: user_expertise (replaces user_quest_stats)
    # =========================================================================
    op.create_table(
        'user_expertise',
        sa.Column('user_id', sa.String(255), primary_key=True),

        # Overall expertise
        sa.Column('total_expertise_points', sa.Integer, server_default='0'),
        sa.Column('expertise_level', sa.String(50), server_default='novice'),

        # Domain-specific expertise
        sa.Column('domain_points', JSON, server_default='{}'),
        sa.Column('domains_mastered', JSON, server_default='[]'),

        # Performance metrics
        sa.Column('total_sessions', sa.Integer, server_default='0'),
        sa.Column('successful_resolutions', sa.Integer, server_default='0'),
        sa.Column('avg_resolution_time_seconds', sa.Float),
        sa.Column('fastest_resolution_seconds', sa.Float),

        # Engagement metrics
        sa.Column('current_streak_days', sa.Integer, server_default='0'),
        sa.Column('longest_streak_days', sa.Integer, server_default='0'),
        sa.Column('last_activity_date', sa.Date),

        # Periodic tracking
        sa.Column('weekly_sessions', sa.Integer, server_default='0'),
        sa.Column('monthly_sessions', sa.Integer, server_default='0'),
        sa.Column('weekly_resolutions', sa.Integer, server_default='0'),
        sa.Column('monthly_resolutions', sa.Integer, server_default='0'),

        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Index for leaderboard queries
    op.create_index('idx_expertise_level', 'user_expertise',
                    ['expertise_level', 'total_expertise_points'])

    # =========================================================================
    # Table: troubleshooting_achievements (replaces achievements)
    # =========================================================================
    op.create_table(
        'troubleshooting_achievements',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('icon_url', sa.String(500)),

        # Achievement criteria
        sa.Column('domain', sa.String(100)),
        sa.Column('criteria_type', sa.String(50)),
        sa.Column('criteria_value', sa.Integer),
        sa.Column('criteria_data', JSON, server_default='{}'),

        # Display
        sa.Column('badge_color', sa.String(7)),
        sa.Column('tier', sa.String(20), server_default='bronze'),

        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Index for achievement queries
    op.create_index('idx_achievement_domain', 'troubleshooting_achievements',
                    ['domain'])
    op.create_index('idx_achievement_tier', 'troubleshooting_achievements',
                    ['tier'])

    # =========================================================================
    # Table: user_troubleshooting_achievements (replaces user_achievements)
    # =========================================================================
    op.create_table(
        'user_troubleshooting_achievements',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('achievement_id', UUID(as_uuid=True),
                  sa.ForeignKey('troubleshooting_achievements.id'),
                  nullable=False),
        sa.Column('earned_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('points_awarded', sa.Integer, server_default='0'),

        # Context
        sa.Column('session_id', UUID(as_uuid=True)),
        sa.Column('domain', sa.String(100)),
    )

    # Index for user achievement queries
    op.create_index('idx_user_achievement_user', 'user_troubleshooting_achievements',
                    ['user_id'])
    op.create_index('idx_user_achievement_earned', 'user_troubleshooting_achievements',
                    ['user_id', 'earned_at'])


def downgrade() -> None:
    """
    Drop troubleshooting tables in reverse order of creation.

    WARNING: This will delete all troubleshooting data.
    Old quest tables are preserved for potential data recovery.
    """
    # Drop indexes first (some are created implicitly, but be explicit)
    op.drop_index('idx_user_achievement_earned', 'user_troubleshooting_achievements')
    op.drop_index('idx_user_achievement_user', 'user_troubleshooting_achievements')
    op.drop_index('idx_achievement_tier', 'troubleshooting_achievements')
    op.drop_index('idx_achievement_domain', 'troubleshooting_achievements')
    op.drop_index('idx_expertise_level', 'user_expertise')
    op.drop_index('idx_execution_session_state', 'task_executions')
    op.drop_index('idx_session_user_id', 'troubleshooting_sessions')
    op.drop_index('idx_session_domain', 'troubleshooting_sessions')
    op.drop_index('idx_session_user_state', 'troubleshooting_sessions')
    op.drop_index('idx_task_workflow_order', 'workflow_tasks')
    op.drop_index('idx_workflow_active', 'troubleshooting_workflows')
    op.drop_index('idx_workflow_category', 'troubleshooting_workflows')
    op.drop_index('idx_workflow_domain', 'troubleshooting_workflows')

    # Drop tables in reverse dependency order
    op.drop_table('user_troubleshooting_achievements')
    op.drop_table('troubleshooting_achievements')
    op.drop_table('user_expertise')
    op.drop_table('task_executions')
    op.drop_table('troubleshooting_sessions')
    op.drop_table('workflow_tasks')
    op.drop_table('troubleshooting_workflows')
