"""Add billing_audit_log table for comprehensive billing entity audit logging

This migration implements Phase 3 of the Token Stats Plan:
- Creates billing_audit_log table for tracking all billing-related changes
- Tracks changes to API keys, budgets, pricing, and usage records
- Supports rich context including actor, reason, and request metadata

Revision ID: 019_billing_audit_log
Revises: 018_pricing_audit_log
Create Date: 2025-01-15
"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import (
    is_postgresql, uuid_column, inet_column, jsonb_column, create_index
)


# revision identifiers, used by Alembic.
revision = '019_billing_audit_log'
down_revision = '018_pricing_audit_log'
branch_labels = None
depends_on = None


def upgrade():
    """Create billing_audit_log table for comprehensive audit trail."""

    # Create billing_audit_log table
    op.create_table(
        'billing_audit_log',
        # Primary key
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),

        # Entity identification
        sa.Column('entity_type', sa.String(30), nullable=False),  # 'api_key', 'budget', 'pricing', 'usage_record'
        sa.Column('entity_id', sa.String(100), nullable=False),

        # Action type
        sa.Column('action', sa.String(30), nullable=False),

        # Changes stored as JSON: {"field": {"old": x, "new": y}, ...}
        sa.Column('changes', jsonb_column(), nullable=False),

        # Actor information
        sa.Column('actor_type', sa.String(20), nullable=False),  # 'user', 'system', 'api_sync'
        sa.Column('actor_user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('actor_description', sa.Text(), nullable=True),

        # Context
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('ip_address', inet_column(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('request_id', uuid_column(), nullable=True),

        # Related entities for efficient querying
        sa.Column('related_api_key_id', sa.Integer(), nullable=True),
        sa.Column('related_budget_id', sa.Integer(), nullable=True),
        sa.Column('related_user_id', sa.Integer(), nullable=True),

        # Timestamp
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),

        # Primary key
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for efficient querying

    # Main lookup index: entity type + entity ID + time (DESC)
    create_index(
        'idx_billing_audit_entity',
        'billing_audit_log',
        ['entity_type', 'entity_id', 'created_at'],
        postgresql_ops={'created_at': 'DESC'},
    )

    # Partial indexes (PostgreSQL only) or regular indexes (SQLite)
    if is_postgresql():
        op.execute("""
            CREATE INDEX idx_billing_audit_api_key
            ON billing_audit_log(related_api_key_id, created_at DESC)
            WHERE related_api_key_id IS NOT NULL
        """)
        op.execute("""
            CREATE INDEX idx_billing_audit_budget
            ON billing_audit_log(related_budget_id, created_at DESC)
            WHERE related_budget_id IS NOT NULL
        """)
        op.execute("""
            CREATE INDEX idx_billing_audit_actor
            ON billing_audit_log(actor_user_id, created_at DESC)
            WHERE actor_user_id IS NOT NULL
        """)
    else:
        op.create_index('idx_billing_audit_api_key', 'billing_audit_log',
                        ['related_api_key_id', 'created_at'])
        op.create_index('idx_billing_audit_budget', 'billing_audit_log',
                        ['related_budget_id', 'created_at'])
        op.create_index('idx_billing_audit_actor', 'billing_audit_log',
                        ['actor_user_id', 'created_at'])

    # Index for user audit trail queries (all actions by or affecting a user)
    create_index(
        'idx_billing_audit_user',
        'billing_audit_log',
        ['related_user_id', 'created_at'],
        postgresql_ops={'created_at': 'DESC'},
    )

    # Time-based queries (recent activity)
    create_index(
        'idx_billing_audit_time',
        'billing_audit_log',
        ['created_at'],
        postgresql_ops={'created_at': 'DESC'},
    )


def downgrade():
    """Remove billing_audit_log table."""

    # Drop indexes (handle both PostgreSQL and SQLite)
    try:
        op.drop_index('idx_billing_audit_actor', table_name='billing_audit_log')
    except Exception:
        pass
    op.drop_index('idx_billing_audit_time', table_name='billing_audit_log')
    op.drop_index('idx_billing_audit_user', table_name='billing_audit_log')
    try:
        op.drop_index('idx_billing_audit_budget', table_name='billing_audit_log')
    except Exception:
        pass
    try:
        op.drop_index('idx_billing_audit_api_key', table_name='billing_audit_log')
    except Exception:
        pass
    op.drop_index('idx_billing_audit_entity', table_name='billing_audit_log')

    # Drop table
    op.drop_table('billing_audit_log')
