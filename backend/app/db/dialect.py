"""
Database dialect detection and cross-database SQL helpers.

Use these helpers for any SQL that differs between PostgreSQL and SQLite.
"""
from functools import lru_cache
from typing import Any
from sqlalchemy import text
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_dialect_name() -> str:
    """
    Get current database dialect name.

    Returns: 'postgresql' or 'sqlite'
    """
    from app.core.config import settings
    db_url = settings.DATABASE_URL or ""
    if db_url.startswith("sqlite"):
        return "sqlite"
    return "postgresql"


def is_postgresql() -> bool:
    """Check if using PostgreSQL backend."""
    return get_dialect_name() == "postgresql"


def is_sqlite() -> bool:
    """Check if using SQLite backend."""
    return get_dialect_name() == "sqlite"


# =============================================================================
# SQL Query Helpers
# =============================================================================

def sql_table_exists(table_name: str) -> str:
    """
    Generate SQL to check if a table exists.

    Returns SQL that produces:
    - PostgreSQL: Boolean (true/false)
    - SQLite: Row if exists, empty if not

    Use parse_table_exists_result() to interpret the result.
    """
    if is_sqlite():
        return f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    return f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"


def parse_table_exists_result(result) -> bool:
    """Parse result from sql_table_exists query."""
    if is_sqlite():
        return result.scalar() is not None
    return bool(result.scalar())


def sql_table_count() -> str:
    """Generate SQL to count tables in database."""
    if is_sqlite():
        return "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    return "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"


def sql_current_timestamp() -> str:
    """
    Get SQL expression for current timestamp.

    For use in server_default in migrations.
    """
    if is_sqlite():
        return "CURRENT_TIMESTAMP"
    return "now()"


def sql_gen_random_uuid() -> str:
    """
    Get SQL expression for generating a UUID.

    Note: SQLite doesn't have built-in UUID generation.
    For SQLite, use Python uuid.uuid4() instead of server-side generation.
    """
    if is_sqlite():
        raise NotImplementedError(
            "SQLite doesn't support server-side UUID generation. "
            "Use Python's uuid.uuid4() with default= in Column definition."
        )
    return "gen_random_uuid()"
