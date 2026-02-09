"""Add pricing_audit_log table for tracking pricing changes

This migration implements Phase 2 of the Token Stats Plan:
- Creates pricing_audit_log table for tracking all pricing changes
- Adds indexes for efficient querying by provider/model and user

Revision ID: 018_pricing_audit_log
Revises: 017_usage_records
Create Date: 2025-01-15
"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_postgresql, uuid_column, jsonb_column, create_index


# revision identifiers, used by Alembic.
revision = '018_pricing_audit_log'
down_revision = '017_usage_records'
branch_labels = None
depends_on = None


def upgrade():
    """Create pricing_audit_log table for tracking pricing changes."""

    # Create pricing_audit_log table
    op.create_table(
        'pricing_audit_log',
        # Primary key
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),

        # Provider and model identification
        sa.Column('provider_id', sa.String(50), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),

        # Action type
        sa.Column('action', sa.String(20), nullable=False),  # 'create', 'update', 'sync', 'override', 'remove_override'

        # Old pricing values (null for create actions)
        sa.Column('old_input_price_per_million_cents', sa.BigInteger(), nullable=True),
        sa.Column('old_output_price_per_million_cents', sa.BigInteger(), nullable=True),

        # New pricing values
        sa.Column('new_input_price_per_million_cents', sa.BigInteger(), nullable=False),
        sa.Column('new_output_price_per_million_cents', sa.BigInteger(), nullable=False),

        # Source tracking
        sa.Column('change_source', sa.String(20), nullable=False),  # 'api_sync', 'admin_manual', 'system_default'

        # User who made the change (null for system/sync operations)
        sa.Column('changed_by_user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),

        # Additional context
        sa.Column('change_reason', sa.Text(), nullable=True),
        sa.Column('sync_job_id', uuid_column(), nullable=True),
        sa.Column('api_response_snapshot', jsonb_column(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),

        # Primary key
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for pricing_audit_log
    # Note: postgresql_ops is only used on PostgreSQL
    create_index(
        'idx_pricing_audit_log_provider_model',
        'pricing_audit_log',
        ['provider_id', 'model_id', 'created_at'],
        postgresql_ops={'created_at': 'DESC'},
    )

    # Partial index for queries by user (PostgreSQL only)
    if is_postgresql():
        op.execute("""
            CREATE INDEX idx_pricing_audit_log_changed_by
            ON pricing_audit_log(changed_by_user_id)
            WHERE changed_by_user_id IS NOT NULL
        """)
    else:
        op.create_index('idx_pricing_audit_log_changed_by', 'pricing_audit_log', ['changed_by_user_id'])

    # Index for sync job queries
    op.create_index(
        'idx_pricing_audit_log_sync_job',
        'pricing_audit_log',
        ['sync_job_id'],
    )


def downgrade():
    """Remove pricing_audit_log table."""

    # Drop indexes
    op.drop_index('idx_pricing_audit_log_sync_job', table_name='pricing_audit_log')
    try:
        op.drop_index('idx_pricing_audit_log_changed_by', table_name='pricing_audit_log')
    except Exception:
        pass
    op.drop_index('idx_pricing_audit_log_provider_model', table_name='pricing_audit_log')

    # Drop table
    op.drop_table('pricing_audit_log')
