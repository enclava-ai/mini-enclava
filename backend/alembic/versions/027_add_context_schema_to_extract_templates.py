"""Add context_schema to extract_templates

Revision ID: 027_add_context_schema_to_extract_templates
Revises: 026_remove_extract_template_name
Create Date: 2026-01-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = '027_add_context_schema_to_extract_templates'
down_revision = '026_remove_extract_template_name'
branch_labels = None
depends_on = None


def upgrade():
    # Add context_schema column to extract_templates
    # This allows templates to define custom context variables like company_name, currency, etc.
    op.add_column('extract_templates',
        sa.Column('context_schema', JSONB, nullable=True)
    )


def downgrade():
    # Remove the context_schema column if rolling back
    op.drop_column('extract_templates', 'context_schema')
