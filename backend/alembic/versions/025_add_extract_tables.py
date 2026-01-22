"""add extract tables

Revision ID: 025_add_extract_tables
Revises: 024_biginteger_counters, fd999a559a35
Create Date: 2025-01-21

This migration merges two branches and adds Extract tables.

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "025_add_extract_tables"
down_revision = ("024_biginteger_counters", "fd999a559a35")  # Merge two branches
branch_labels = None
depends_on = None


def upgrade():
    # Extract Templates
    op.create_table(
        "extract_templates",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("system_prompt", sa.Text, nullable=False),
        sa.Column("user_prompt", sa.Text, nullable=False),
        sa.Column("output_schema", postgresql.JSONB, nullable=True),
        sa.Column("is_default", sa.Boolean, default=False, nullable=False),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index(
        "idx_extract_templates_is_active", "extract_templates", ["is_active"]
    )

    # Extract Jobs
    op.create_table(
        "extract_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "api_key_id", sa.Integer, sa.ForeignKey("api_keys.id"), nullable=True
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(50), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False),
        sa.Column("num_pages", sa.Integer, default=1),
        sa.Column("status", sa.String(50), default="pending", nullable=False),
        sa.Column(
            "template_id",
            sa.String(100),
            sa.ForeignKey("extract_templates.id"),
            nullable=False,
        ),
        sa.Column("buyer_context", sa.Text, nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("total_cost_cents", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("idx_extract_jobs_user_id", "extract_jobs", ["user_id"])
    op.create_index("idx_extract_jobs_status", "extract_jobs", ["status"])
    op.create_index("idx_extract_jobs_created_at", "extract_jobs", ["created_at"])

    # Extract Results
    op.create_table(
        "extract_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("extract_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer, default=1, nullable=False),
        sa.Column("raw_response", sa.Text, nullable=True),
        sa.Column("parsed_data", postgresql.JSONB, nullable=True),
        sa.Column("validation_errors", postgresql.JSONB, default=[], nullable=False),
        sa.Column("validation_warnings", postgresql.JSONB, default=[], nullable=False),
        sa.Column("is_final", sa.Boolean, default=False, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )
    op.create_index("idx_extract_results_job_id", "extract_results", ["job_id"])


def downgrade():
    op.drop_table("extract_results")
    op.drop_table("extract_jobs")
    op.drop_table("extract_templates")
