"""Add provider preferences to chatbot_instances table

This migration adds columns to chatbot_instances table for provider selection:
- preferred_provider_id: Preferred inference provider for this chatbot
- allowed_providers: List of allowed providers (empty = all allowed)

Revision ID: 022_chatbot_provider_preferences
Revises: 021_provider_health_attestation
Create Date: 2025-01-16
"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_postgresql, string_array_column


# revision identifiers, used by Alembic.
revision = '022_chatbot_provider_preferences'
down_revision = '021_provider_health_attestation'
branch_labels = None
depends_on = None


def upgrade():
    """Add provider preference columns to chatbot_instances table."""

    # Add preferred_provider_id column
    op.add_column(
        'chatbot_instances',
        sa.Column('preferred_provider_id', sa.String(length=50), nullable=True)
    )

    # Add allowed_providers column (array on PostgreSQL, JSON text on SQLite)
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import ARRAY
        op.add_column(
            'chatbot_instances',
            sa.Column(
                'allowed_providers',
                ARRAY(sa.String(length=50)),
                nullable=True,
                server_default=sa.text("ARRAY[]::VARCHAR[]")
            )
        )
    else:
        # SQLite: use Text with JSON serialization, default to empty JSON array
        op.add_column(
            'chatbot_instances',
            sa.Column(
                'allowed_providers',
                sa.Text(),
                nullable=True,
                server_default='[]'
            )
        )

    # Create index for querying by preferred provider
    op.create_index(
        'idx_chatbot_instances_preferred_provider',
        'chatbot_instances',
        ['preferred_provider_id']
    )


def downgrade():
    """Remove provider preference columns from chatbot_instances table."""

    # Drop index
    op.drop_index('idx_chatbot_instances_preferred_provider', table_name='chatbot_instances')

    # Drop columns
    op.drop_column('chatbot_instances', 'allowed_providers')
    op.drop_column('chatbot_instances', 'preferred_provider_id')
