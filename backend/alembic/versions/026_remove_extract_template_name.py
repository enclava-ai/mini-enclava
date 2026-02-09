"""Remove name column from extract_templates

Revision ID: 026_remove_extract_template_name
Revises: 025_add_extract_tables
Create Date: 2026-01-22

"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_sqlite


# revision identifiers, used by Alembic.
revision = '026_remove_extract_template_name'
down_revision = '025_add_extract_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Remove the name column from extract_templates
    # The template ID now serves as both identifier and display name
    op.drop_column('extract_templates', 'name')


def downgrade():
    # Add the name column back if we need to rollback
    op.add_column('extract_templates',
        sa.Column('name', sa.String(255), nullable=False, server_default='')
    )
    # Update existing rows to use id as name
    op.execute("UPDATE extract_templates SET name = id WHERE name = ''")
    # Remove server default
    if is_sqlite():
        with op.batch_alter_table('extract_templates') as batch_op:
            batch_op.alter_column('name', server_default=None)
    else:
        op.alter_column('extract_templates', 'name', server_default=None)
