"""Update extract scopes to single scope

Revision ID: 030_update_extract_scopes
Revises: 029_add_extract_model_settings
Create Date: 2026-01-23

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import json


# revision identifiers, used by Alembic.
revision = '030_update_extract_scopes'
down_revision = '029_add_extract_model_settings'
branch_labels = None
depends_on = None


def upgrade():
    """
    Update API keys to use single 'extract' scope instead of granular scopes.

    Changes:
    - extract.process, extract.jobs, extract.templates -> extract
    """
    # Use raw SQL to update scopes
    conn = op.get_bind()

    # Get all API keys
    result = conn.execute(sa.text("SELECT id, scopes FROM api_keys"))

    for row in result:
        api_key_id = row[0]
        scopes = row[1] if row[1] else []

        # Check if this API key has any of the old extract scopes
        old_extract_scopes = ['extract.process', 'extract.jobs', 'extract.templates']
        has_old_scopes = any(scope in scopes for scope in old_extract_scopes)

        if has_old_scopes:
            # Remove old extract scopes
            new_scopes = [scope for scope in scopes if scope not in old_extract_scopes]

            # Add new single extract scope if not already present
            if 'extract' not in new_scopes:
                new_scopes.append('extract')

            # Update the API key (PostgreSQL requires JSON to be a string)
            conn.execute(
                sa.text("UPDATE api_keys SET scopes = CAST(:scopes AS JSON) WHERE id = :id"),
                {"scopes": json.dumps(new_scopes), "id": api_key_id}
            )


def downgrade():
    """
    Revert single 'extract' scope back to granular scopes.

    Note: This is a lossy downgrade - we assume all extract permissions.
    """
    conn = op.get_bind()

    # Get all API keys
    result = conn.execute(sa.text("SELECT id, scopes FROM api_keys"))

    for row in result:
        api_key_id = row[0]
        scopes = row[1] if row[1] else []

        # Check if this API key has the new extract scope
        if 'extract' in scopes:
            # Remove new extract scope
            new_scopes = [scope for scope in scopes if scope != 'extract']

            # Add old granular scopes
            old_scopes = ['extract.process', 'extract.jobs', 'extract.templates']
            for scope in old_scopes:
                if scope not in new_scopes:
                    new_scopes.append(scope)

            # Update the API key (PostgreSQL requires JSON to be a string)
            conn.execute(
                sa.text("UPDATE api_keys SET scopes = CAST(:scopes AS JSON) WHERE id = :id"),
                {"scopes": json.dumps(new_scopes), "id": api_key_id}
            )
