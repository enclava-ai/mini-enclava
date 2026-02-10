"""
Cross-database upsert (INSERT ... ON CONFLICT) operations.

Both PostgreSQL and SQLite (3.24+) support upsert, but with
slightly different import paths.
"""
from typing import Dict, List, Optional, Any
from sqlalchemy.sql import Insert

from app.db.dialect import is_sqlite


def upsert(
    table,
    values: Dict[str, Any],
    index_elements: List[str],
    update_set: Optional[Dict[str, Any]] = None
) -> Insert:
    """
    Create a cross-database upsert statement.

    Args:
        table: SQLAlchemy Table or Model class
        values: Column values to insert
        index_elements: Columns forming the unique constraint
        update_set: Columns to update on conflict (None = do nothing)

    Returns:
        Executable INSERT ... ON CONFLICT statement

    Example:
        stmt = upsert(
            ExtractSettings,
            values={"id": 1, "default_model": "gpt-4"},
            index_elements=["id"],
            update_set={"default_model": "gpt-4"}
        )
        await db.execute(stmt)
    """
    if is_sqlite():
        from sqlalchemy.dialects.sqlite import insert
    else:
        from sqlalchemy.dialects.postgresql import insert

    stmt = insert(table).values(**values)

    if update_set:
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=update_set
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)

    return stmt
