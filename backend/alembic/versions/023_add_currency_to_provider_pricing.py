"""Add currency field to provider_pricing table

This migration adds a currency column to support multi-currency pricing:
- PrivateMode.ai uses EUR
- RedPill.ai uses USD

Revision ID: 023_add_currency_provider_pricing
Revises: 022_chatbot_provider_preferences
Create Date: 2025-01-16
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '023_add_pricing_currency'
down_revision = '022_chatbot_provider_preferences'
branch_labels = None
depends_on = None


def upgrade():
    """Add currency column to provider_pricing table."""

    # Add currency column with default 'USD'
    op.add_column(
        'provider_pricing',
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD')
    )

    # Update existing privatemode entries to EUR
    op.execute("""
        UPDATE provider_pricing
        SET currency = 'EUR'
        WHERE provider_id = 'privatemode'
    """)


def downgrade():
    """Remove currency column from provider_pricing table."""
    op.drop_column('provider_pricing', 'currency')
