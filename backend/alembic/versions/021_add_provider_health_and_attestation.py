"""Add provider_health and attestation_history tables

This migration adds tables for tracking provider health and attestation history:
- provider_health: Tracks current health status of inference providers
- attestation_history: Historical log of attestation verification attempts

Revision ID: 021_provider_health_attestation
Revises: 020_budget_reconciliation
Create Date: 2025-01-16
"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import (
    is_postgresql, uuid_column, jsonb_column, timestamp_default
)


# revision identifiers, used by Alembic.
revision = '021_provider_health_attestation'
down_revision = '020_budget_reconciliation'
branch_labels = None
depends_on = None


def upgrade():
    """Add provider health and attestation history tables."""

    # Create provider_health table
    # Note: SQLite doesn't support gen_random_uuid() or onupdate, handle in application layer
    op.create_table(
        'provider_health',
        sa.Column('id', uuid_column(), primary_key=True),
        sa.Column('provider_id', sa.String(length=50), nullable=False, unique=True),
        sa.Column('healthy', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_check_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_healthy_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('last_attestation_json', jsonb_column(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=timestamp_default()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=timestamp_default()),
        sa.PrimaryKeyConstraint('id')
    )

    # Create index for provider_id (unique lookup)
    op.create_index('ix_provider_health_provider_id', 'provider_health', ['provider_id'], unique=True)

    # Create attestation_history table
    op.create_table(
        'attestation_history',
        sa.Column('id', uuid_column(), primary_key=True),
        sa.Column('provider_id', sa.String(length=50), nullable=False),
        sa.Column('model', sa.String(length=255), nullable=False),
        sa.Column('verified', sa.Boolean(), nullable=False),
        sa.Column('signing_address', sa.String(length=100), nullable=True),
        sa.Column('intel_tdx_verified', sa.Boolean(), nullable=True),
        sa.Column('gpu_attestation_verified', sa.Boolean(), nullable=True),
        sa.Column('nonce_binding_verified', sa.Boolean(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=timestamp_default()),
        sa.PrimaryKeyConstraint('id')
    )

    # Create index for querying attestation history by provider
    op.create_index(
        'idx_attestation_history_provider',
        'attestation_history',
        ['provider_id', 'created_at']
    )


def downgrade():
    """Remove provider health and attestation history tables."""

    # Drop attestation_history table
    op.drop_index('idx_attestation_history_provider', table_name='attestation_history')
    op.drop_table('attestation_history')

    # Drop provider_health table
    op.drop_index('ix_provider_health_provider_id', table_name='provider_health')
    op.drop_table('provider_health')
