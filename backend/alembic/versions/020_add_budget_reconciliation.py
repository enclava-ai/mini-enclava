"""Add budget reconciliation tracking

This migration adds:
- last_reconciled_at column to budgets table for tracking reconciliation runs
- last_reconciliation_diff_cents for tracking drift corrections

Revision ID: 020_budget_reconciliation
Revises: 019_add_billing_audit_log
Create Date: 2025-01-15
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '020_budget_reconciliation'
down_revision = '019_billing_audit_log'
branch_labels = None
depends_on = None


def upgrade():
    """Add budget reconciliation columns."""

    # Add last_reconciled_at to track when budget was last reconciled
    op.add_column('budgets', sa.Column('last_reconciled_at', sa.DateTime(), nullable=True))

    # Add last_reconciliation_diff_cents to track drift found in last reconciliation
    op.add_column('budgets', sa.Column('last_reconciliation_diff_cents', sa.Integer(), nullable=True))

    # Create index for finding budgets that need reconciliation
    op.create_index(
        'idx_budgets_reconciliation',
        'budgets',
        ['is_active', 'last_reconciled_at'],
    )


def downgrade():
    """Remove budget reconciliation columns."""

    op.drop_index('idx_budgets_reconciliation', table_name='budgets')
    op.drop_column('budgets', 'last_reconciliation_diff_cents')
    op.drop_column('budgets', 'last_reconciled_at')
