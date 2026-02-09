"""Security fix: Change Integer to BigInteger for counters to prevent overflow

This migration changes counter columns from Integer to BigInteger to prevent
integer overflow attacks or issues with high-volume usage.

Security mitigation #35: Budget counter overflow risk

Affected tables:
- budgets: limit_cents, warning_threshold_cents, current_usage_cents
- api_keys: total_requests, total_tokens, total_cost
- usage_tracking: request_tokens, response_tokens, total_tokens, cost_cents

Revision ID: 024_biginteger_counters
Revises: 023_add_pricing_currency
Create Date: 2026-01-19
"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_sqlite


# revision identifiers, used by Alembic.
revision = '024_biginteger_counters'
down_revision = '023_add_pricing_currency'
branch_labels = None
depends_on = None


def upgrade():
    """Change Integer columns to BigInteger for counter fields."""
    # SQLite INTEGER is already 64-bit, no type change needed
    if is_sqlite():
        return

    # Budget table - limit and usage counters
    op.alter_column(
        'budgets',
        'limit_cents',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=False
    )
    op.alter_column(
        'budgets',
        'warning_threshold_cents',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'budgets',
        'current_usage_cents',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )

    # API Keys table - usage tracking counters
    op.alter_column(
        'api_keys',
        'total_requests',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'api_keys',
        'total_tokens',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'api_keys',
        'total_cost',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )

    # Usage Tracking table - token and cost counters
    op.alter_column(
        'usage_tracking',
        'request_tokens',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'response_tokens',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'total_tokens',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'cost_cents',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
        existing_nullable=True
    )


def downgrade():
    """Revert BigInteger columns back to Integer.

    WARNING: This may cause data loss if values exceed Integer range.
    """
    # SQLite INTEGER is already 64-bit, no type change needed
    if is_sqlite():
        return

    # Usage Tracking table
    op.alter_column(
        'usage_tracking',
        'cost_cents',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'total_tokens',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'response_tokens',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'usage_tracking',
        'request_tokens',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )

    # API Keys table
    op.alter_column(
        'api_keys',
        'total_cost',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'api_keys',
        'total_tokens',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'api_keys',
        'total_requests',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )

    # Budget table
    op.alter_column(
        'budgets',
        'current_usage_cents',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'budgets',
        'warning_threshold_cents',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=True
    )
    op.alter_column(
        'budgets',
        'limit_cents',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
        existing_nullable=False
    )
