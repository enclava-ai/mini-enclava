"""Add extract API key support

Revision ID: 028_add_extract_api_key_support
Revises: 027_add_context_schema
Create Date: 2026-01-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '028_add_extract_api_key_support'
down_revision = '027_add_context_schema'
branch_labels = None
depends_on = None


def upgrade():
    # Add allowed_extract_templates column to api_keys table
    op.add_column(
        'api_keys',
        sa.Column(
            'allowed_extract_templates',
            postgresql.JSON,
            nullable=False,
            server_default='[]'
        )
    )


def downgrade():
    # Remove allowed_extract_templates column
    op.drop_column('api_keys', 'allowed_extract_templates')
