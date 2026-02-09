"""
Cross-database compatible column types.

IMPORTANT: Use these types instead of PostgreSQL-specific imports.
- Use GUID instead of postgresql.UUID
- Use JSONB instead of postgresql.JSONB
- Use INET instead of postgresql.INET
- Use StringArray instead of ARRAY(String)

These types use native PostgreSQL types when available, falling back
to SQLite-compatible alternatives.
"""
from sqlalchemy import TypeDecorator, String, Text, JSON
import uuid
import json
from typing import Any, List, Optional


class GUID(TypeDecorator):
    """
    Platform-independent UUID type.

    - PostgreSQL: Uses native UUID type
    - SQLite: Uses String(36)

    Always returns Python uuid.UUID objects.
    """
    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import UUID as PGUUID
            return dialect.type_descriptor(PGUUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
        return str(value) if isinstance(value, uuid.UUID) else value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class JSONB(TypeDecorator):
    """
    Cross-platform JSON with binary storage optimization.

    - PostgreSQL: Uses JSONB (binary, indexable)
    - SQLite: Uses JSON (text-based, JSON1 extension)

    Note: JSONB GIN indexes only work on PostgreSQL.
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import JSONB as PGJSONB
            return dialect.type_descriptor(PGJSONB)
        return dialect.type_descriptor(JSON)


class INET(TypeDecorator):
    """
    IP address storage type.

    - PostgreSQL: Uses native INET type with validation
    - SQLite: Uses String(45) - max IPv6 length

    Note: PostgreSQL validates IP format; SQLite stores as plain string.
    Application-level validation recommended for SQLite.
    """
    impl = String(45)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import INET as PGINET
            return dialect.type_descriptor(PGINET)
        return dialect.type_descriptor(String(45))


class StringArray(TypeDecorator):
    """
    Array of strings type.

    - PostgreSQL: Uses native ARRAY(VARCHAR)
    - SQLite: Uses JSON-serialized list

    Always returns Python list of strings.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy import ARRAY, String as SAString
            return dialect.type_descriptor(ARRAY(SAString(255)))
        return dialect.type_descriptor(Text)

    def process_bind_param(self, value: Optional[List[str]], dialect) -> Optional[Any]:
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.dumps(value)

    def process_result_value(self, value: Optional[Any], dialect) -> Optional[List[str]]:
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value
