"""Add usage_records and provider_pricing tables for token stats

This migration implements Phase 1 of the Token Stats Plan:
- Creates usage_records table for tracking individual LLM requests
- Creates provider_pricing table for model pricing management
- Adds soft-delete columns to api_keys table

Revision ID: 017_usage_records
Revises: 016_remove_display_name
Create Date: 2025-01-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '017_usage_records'
down_revision = '016_remove_display_name'
branch_labels = None
depends_on = None


def upgrade():
    """Create usage_records and provider_pricing tables, add soft-delete to api_keys."""

    # Create usage_records table
    op.create_table(
        'usage_records',
        # Primary key
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),

        # Unique request identifier for tracing
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),

        # Ownership
        sa.Column('api_key_id', sa.Integer(), sa.ForeignKey('api_keys.id'), nullable=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),

        # Provider Information
        sa.Column('provider_id', sa.String(50), nullable=False),
        sa.Column('provider_model', sa.String(255), nullable=False),
        sa.Column('normalized_model', sa.String(255), nullable=False),

        # Token Metrics
        sa.Column('input_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('output_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),

        # Cost in cents
        sa.Column('input_cost_cents', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('output_cost_cents', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('total_cost_cents', sa.BigInteger(), nullable=False, server_default='0'),

        # Pricing snapshot
        sa.Column('input_price_per_million_cents', sa.BigInteger(), nullable=False),
        sa.Column('output_price_per_million_cents', sa.BigInteger(), nullable=False),
        sa.Column('pricing_source', sa.String(20), nullable=False),
        sa.Column('pricing_effective_from', sa.DateTime(), nullable=False),

        # Request Context
        sa.Column('endpoint', sa.String(255), nullable=False),
        sa.Column('method', sa.String(10), nullable=False, server_default='POST'),
        sa.Column('chatbot_id', sa.String(50), nullable=True),
        sa.Column('agent_config_id', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=True),

        # Request Characteristics
        sa.Column('is_streaming', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_tool_calling', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('message_count', sa.Integer(), nullable=False, server_default='0'),

        # Performance Metrics
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('ttft_ms', sa.Integer(), nullable=True),

        # Status
        sa.Column('status', sa.String(20), nullable=False, server_default='success'),
        sa.Column('error_type', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),

        # Client Info
        sa.Column('ip_address', postgresql.INET(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),

        # Primary key and constraints
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_id'),
        sa.CheckConstraint(
            'api_key_id IS NOT NULL OR user_id IS NOT NULL',
            name='usage_records_api_key_or_user'
        ),
    )

    # Create indexes for usage_records
    op.create_index('idx_usage_records_request_id', 'usage_records', ['request_id'])
    op.create_index('idx_usage_records_api_key_id', 'usage_records', ['api_key_id'])
    op.create_index('idx_usage_records_user_id', 'usage_records', ['user_id'])
    op.create_index('idx_usage_records_provider_id', 'usage_records', ['provider_id'])
    op.create_index('idx_usage_records_normalized_model', 'usage_records', ['normalized_model'])
    op.create_index('idx_usage_records_created_at', 'usage_records', ['created_at'])
    op.create_index('idx_usage_records_api_key_created', 'usage_records', ['api_key_id', 'created_at'])
    op.create_index('idx_usage_records_user_created', 'usage_records', ['user_id', 'created_at'])
    op.create_index('idx_usage_records_provider_created', 'usage_records', ['provider_id', 'created_at'])

    # Create partial index for billing aggregations (only successful requests)
    op.execute("""
        CREATE INDEX idx_usage_records_billing
        ON usage_records(api_key_id, provider_id, normalized_model, created_at)
        WHERE status = 'success'
    """)

    # Create partial index for chatbot queries
    op.execute("""
        CREATE INDEX idx_usage_records_chatbot_created
        ON usage_records(chatbot_id, created_at DESC)
        WHERE chatbot_id IS NOT NULL
    """)

    # Create provider_pricing table
    op.create_table(
        'provider_pricing',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),

        # Provider and model identification
        sa.Column('provider_id', sa.String(50), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=True),

        # Pricing (in cents per 1M tokens)
        sa.Column('input_price_per_million_cents', sa.BigInteger(), nullable=False),
        sa.Column('output_price_per_million_cents', sa.BigInteger(), nullable=False),

        # Source tracking
        sa.Column('price_source', sa.String(20), nullable=False),
        sa.Column('source_api_response', postgresql.JSONB(), nullable=True),

        # Override support
        sa.Column('is_override', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('override_reason', sa.Text(), nullable=True),
        sa.Column('override_by_user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),

        # Model metadata
        sa.Column('context_length', sa.Integer(), nullable=True),
        sa.Column('architecture', postgresql.JSONB(), nullable=True),
        sa.Column('quantization', sa.String(20), nullable=True),

        # Validity period
        sa.Column('effective_from', sa.DateTime(), nullable=False),
        sa.Column('effective_until', sa.DateTime(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider_id', 'model_id', 'effective_from', name='provider_pricing_unique'),
    )

    # Create indexes for provider_pricing
    op.create_index(
        'idx_provider_pricing_lookup',
        'provider_pricing',
        ['provider_id', 'model_id', 'effective_from'],
    )

    # Create partial index for current pricing
    op.execute("""
        CREATE INDEX idx_provider_pricing_current
        ON provider_pricing(provider_id, model_id, effective_from DESC)
        WHERE effective_until IS NULL
    """)

    # Create partial index for override management
    op.execute("""
        CREATE INDEX idx_provider_pricing_overrides
        ON provider_pricing(provider_id, is_override)
        WHERE is_override = TRUE
    """)

    # Add soft-delete columns to api_keys table
    op.add_column('api_keys', sa.Column('deleted_at', sa.DateTime(), nullable=True))
    op.add_column('api_keys', sa.Column('deleted_by_user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True))
    op.add_column('api_keys', sa.Column('deletion_reason', sa.Text(), nullable=True))

    # Create indexes for soft-delete on api_keys
    op.execute("""
        CREATE INDEX idx_api_keys_active
        ON api_keys(user_id, is_active)
        WHERE deleted_at IS NULL
    """)
    op.create_index('idx_api_keys_all', 'api_keys', ['user_id', 'created_at'])


def downgrade():
    """Remove usage_records and provider_pricing tables, remove soft-delete from api_keys."""

    # Drop indexes from api_keys
    op.execute("DROP INDEX IF EXISTS idx_api_keys_active")
    op.drop_index('idx_api_keys_all', table_name='api_keys')

    # Drop soft-delete columns from api_keys
    op.drop_column('api_keys', 'deletion_reason')
    op.drop_column('api_keys', 'deleted_by_user_id')
    op.drop_column('api_keys', 'deleted_at')

    # Drop provider_pricing indexes and table
    op.execute("DROP INDEX IF EXISTS idx_provider_pricing_overrides")
    op.execute("DROP INDEX IF EXISTS idx_provider_pricing_current")
    op.drop_index('idx_provider_pricing_lookup', table_name='provider_pricing')
    op.drop_table('provider_pricing')

    # Drop usage_records indexes and table
    op.execute("DROP INDEX IF EXISTS idx_usage_records_chatbot_created")
    op.execute("DROP INDEX IF EXISTS idx_usage_records_billing")
    op.drop_index('idx_usage_records_provider_created', table_name='usage_records')
    op.drop_index('idx_usage_records_user_created', table_name='usage_records')
    op.drop_index('idx_usage_records_api_key_created', table_name='usage_records')
    op.drop_index('idx_usage_records_created_at', table_name='usage_records')
    op.drop_index('idx_usage_records_normalized_model', table_name='usage_records')
    op.drop_index('idx_usage_records_provider_id', table_name='usage_records')
    op.drop_index('idx_usage_records_user_id', table_name='usage_records')
    op.drop_index('idx_usage_records_api_key_id', table_name='usage_records')
    op.drop_index('idx_usage_records_request_id', table_name='usage_records')
    op.drop_table('usage_records')
