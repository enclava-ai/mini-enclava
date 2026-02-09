"""Add context_schema to extract_templates

Revision ID: 027_add_context_schema
Revises: 026_remove_extract_template_name
Create Date: 2026-01-22

"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import jsonb_column


# revision identifiers, used by Alembic.
revision = '027_add_context_schema'
down_revision = '026_remove_extract_template_name'
branch_labels = None
depends_on = None


def upgrade():
    # Add context_schema column to extract_templates
    # This allows templates to define custom context variables like company_name, currency, etc.
    op.add_column('extract_templates',
        sa.Column('context_schema', jsonb_column(), nullable=True)
    )


def downgrade():
    # Remove the context_schema column if rolling back
    op.drop_column('extract_templates', 'context_schema')
