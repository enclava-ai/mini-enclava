"""
Cross-database migration helpers for Alembic.

Use these in migrations to write dialect-aware schema changes.
"""
from alembic import op, context
import sqlalchemy as sa
from typing import Optional, List, Any


def get_dialect() -> str:
    """Get current migration dialect name."""
    return context.get_context().dialect.name


def is_postgresql() -> bool:
    return get_dialect() == "postgresql"


def is_sqlite() -> bool:
    return get_dialect() == "sqlite"


# =============================================================================
# Type Helpers
# =============================================================================

def uuid_column() -> sa.types.TypeEngine:
    """
    Get appropriate UUID column type for current dialect.

    - PostgreSQL: UUID
    - SQLite: String(36)
    """
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import UUID
        return UUID(as_uuid=True)
    return sa.String(36)


def jsonb_column() -> sa.types.TypeEngine:
    """
    Get appropriate JSON column type for current dialect.

    - PostgreSQL: JSONB
    - SQLite: JSON
    """
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import JSONB
        return JSONB
    return sa.JSON


def inet_column() -> sa.types.TypeEngine:
    """
    Get appropriate IP address column type for current dialect.

    - PostgreSQL: INET
    - SQLite: String(45)
    """
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import INET
        return INET
    return sa.String(45)


def string_array_column() -> sa.types.TypeEngine:
    """
    Get appropriate string array column type for current dialect.

    - PostgreSQL: ARRAY(String)
    - SQLite: Text (JSON-serialized)
    """
    if is_postgresql():
        return sa.ARRAY(sa.String(255))
    return sa.Text


def timestamp_default() -> sa.sql.elements.TextClause:
    """
    Get appropriate timestamp default for current dialect.

    - PostgreSQL: now()
    - SQLite: CURRENT_TIMESTAMP
    """
    if is_postgresql():
        return sa.text("now()")
    return sa.text("CURRENT_TIMESTAMP")


def autoincrement_int_column() -> sa.types.TypeEngine:
    """
    Get appropriate auto-incrementing integer primary key type.

    - PostgreSQL: BigInteger (supports BIGSERIAL)
    - SQLite: Integer (only INTEGER PRIMARY KEY auto-increments in SQLite)

    Note: SQLite only auto-increments INTEGER PRIMARY KEY columns,
    not BIGINT. Use this helper for auto-incrementing PKs.
    """
    if is_postgresql():
        return sa.BigInteger()
    return sa.Integer()


# =============================================================================
# Enum Helpers
# =============================================================================

def create_enum(name: str, values: List[str]) -> Optional[Any]:
    """
    Create an enum type (PostgreSQL only).

    For SQLite, returns None - use String column with CHECK constraint instead
    to preserve integrity.

    Returns:
        PostgreSQL: ENUM type
        SQLite: None
    """
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import ENUM
        enum_type = ENUM(*values, name=name, create_type=False)
        op.execute(f"CREATE TYPE {name} AS ENUM ({', '.join(repr(v) for v in values)})")
        return enum_type
    return None


def drop_enum(name: str):
    """Drop an enum type (PostgreSQL only, no-op on SQLite)."""
    if is_postgresql():
        op.execute(f"DROP TYPE IF EXISTS {name}")


def enum_column(enum_name: str, values: List[str]) -> sa.types.TypeEngine:
    """
    Get column type for enum values.

    - PostgreSQL: Uses native ENUM type
    - SQLite: Uses String with length based on longest value
    """
    if is_postgresql():
        from sqlalchemy.dialects.postgresql import ENUM
        return ENUM(*values, name=enum_name, create_type=False)
    max_len = max(len(v) for v in values)
    return sa.String(max_len + 10)  # Some buffer


def enum_check_constraint(
    column_name: str,
    values: List[str],
    name: Optional[str] = None
) -> Optional[sa.CheckConstraint]:
    """
    Create a CHECK constraint for enum values on SQLite.

    - PostgreSQL: returns None (native ENUM enforces integrity)
    - SQLite: returns CHECK constraint enforcing allowed values
    """
    if is_postgresql():
        return None
    constraint_name = name or f"ck_{column_name}_values"
    values_sql = ", ".join(repr(v) for v in values)
    return sa.CheckConstraint(
        f"{column_name} IN ({values_sql})",
        name=constraint_name
    )


# =============================================================================
# Index Helpers
# =============================================================================

def create_index(
    index_name: str,
    table_name: str,
    columns: List[str],
    unique: bool = False,
    postgresql_where: Optional[str] = None,
    postgresql_ops: Optional[dict] = None
):
    """
    Create an index with optional PostgreSQL-specific features.

    Args:
        index_name: Name of the index
        table_name: Table to index
        columns: Columns to include
        unique: Whether index is unique
        postgresql_where: Partial index condition (PostgreSQL only)
        postgresql_ops: Column operators (PostgreSQL only)

    Note: postgresql_where and postgresql_ops are silently ignored on SQLite.
    SQLite does support WHERE in indexes, but the syntax differs.
    """
    kwargs = {"unique": unique}

    if is_postgresql():
        if postgresql_where:
            kwargs["postgresql_where"] = sa.text(postgresql_where)
        if postgresql_ops:
            kwargs["postgresql_ops"] = postgresql_ops

    op.create_index(index_name, table_name, columns, **kwargs)
