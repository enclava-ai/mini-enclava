"""Add extract model settings

Revision ID: 029
Revises: 028
Create Date: 2026-01-23
"""

from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_postgresql, timestamp_default


# revision identifiers, used by Alembic.
revision = "029_add_extract_model_settings"
down_revision = "028_add_extract_api_key_support"
branch_labels = None
depends_on = None


def upgrade():
    # Add model column to extract_templates
    op.add_column(
        "extract_templates",
        sa.Column("model", sa.String(length=100), nullable=True),
    )

    # Create extract_settings table
    op.create_table(
        "extract_settings",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column(
            "default_model", sa.String(length=100), nullable=True  # Will be set by initialization
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=timestamp_default(),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Insert settings row with null default_model (will be populated from available models)
    # Use dialect-appropriate syntax for upsert
    if is_postgresql():
        op.execute(
            """
            INSERT INTO extract_settings (id, default_model)
            VALUES (1, NULL)
            ON CONFLICT (id) DO NOTHING
            """
        )
    else:
        # SQLite syntax
        op.execute(
            """
            INSERT OR IGNORE INTO extract_settings (id, default_model)
            VALUES (1, NULL)
            """
        )


def downgrade():
    op.drop_table("extract_settings")
    op.drop_column("extract_templates", "model")
