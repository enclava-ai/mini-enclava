"""Add tool support to chatbot messages

Revision ID: 007_add_chatbot_tool_support
Revises: 006_add_source_url_to_rag_docs
Create Date: 2024-12-16 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from app.db.migrations import is_sqlite


# revision identifiers, used by Alembic.
revision = '007_add_chatbot_tool_support'
down_revision = '006_add_source_url_to_rag_docs'
branch_labels = None
depends_on = None


def upgrade():
    """Add tool-related columns to chatbot_messages table."""
    # Make content nullable (required for tool-call messages with no text)
    if is_sqlite():
        with op.batch_alter_table('chatbot_messages') as batch_op:
            batch_op.alter_column('content',
                                  existing_type=sa.Text(),
                                  nullable=True)
    else:
        op.alter_column('chatbot_messages', 'content',
                        existing_type=sa.Text(),
                        nullable=True)

    # Add tool-related columns
    op.add_column('chatbot_messages', sa.Column('tool_calls', sa.JSON(), nullable=True))
    op.add_column('chatbot_messages', sa.Column('tool_call_id', sa.String(100), nullable=True))
    op.add_column('chatbot_messages', sa.Column('tool_name', sa.String(100), nullable=True))


def downgrade():
    """Remove tool support from chatbot_messages table."""
    # Drop tool columns
    op.drop_column('chatbot_messages', 'tool_name')
    op.drop_column('chatbot_messages', 'tool_call_id')
    op.drop_column('chatbot_messages', 'tool_calls')

    # Restore NOT NULL constraint on content
    if is_sqlite():
        with op.batch_alter_table('chatbot_messages') as batch_op:
            batch_op.alter_column('content',
                                  existing_type=sa.Text(),
                                  nullable=False)
    else:
        op.alter_column('chatbot_messages', 'content',
                        existing_type=sa.Text(),
                        nullable=False)
