"""
Database connection and session management

This module manages database connections with optimized pool settings:
- PostgreSQL: Primary async pool (asyncpg): 30 + 50 overflow = 80 max connections
              Legacy sync pool (psycopg2): 5 + 10 overflow = 15 max connections
              Total: 95 max connections (under PostgreSQL default of 100)
- SQLite: StaticPool for single-connection access with foreign key support

Pool monitoring is available via get_pool_status() function (PostgreSQL only).
"""

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Dialect Detection
# ============================================================================

def _is_sqlite() -> bool:
    """Check if using SQLite backend."""
    return settings.DATABASE_URL and settings.DATABASE_URL.startswith("sqlite")


def _is_postgresql() -> bool:
    """Check if using PostgreSQL backend."""
    return not _is_sqlite()


# ============================================================================
# SQLite PRAGMA Configuration
# ============================================================================

@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key enforcement for SQLite."""
    conn_type = str(type(dbapi_connection))
    if 'sqlite' in conn_type.lower():
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ============================================================================
# Dialect-aware Table Existence Helpers
# ============================================================================

def _sql_table_exists(table_name: str) -> str:
    """Generate table existence check SQL."""
    if _is_sqlite():
        return f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    return f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"


def _parse_table_exists(result) -> bool:
    """Parse table existence check result."""
    if _is_sqlite():
        return result.scalar() is not None
    return bool(result.scalar())


# ============================================================================
# Engine Factories
# ============================================================================

# Pool metrics tracking
_pool_metrics = {
    "async_checkouts": 0,
    "async_checkins": 0,
    "async_overflow": 0,
    "sync_checkouts": 0,
    "sync_checkins": 0,
    "sync_overflow": 0,
}


def _create_postgresql_engines():
    """Create PostgreSQL engines with optimized connection pooling."""
    # Create async engine with optimized connection pooling
    # This is the PRIMARY engine - most operations should use async sessions
    # Pool sizing: 30 base + 50 overflow = 80 max connections
    # Note: PostgreSQL default max_connections=100, leave headroom for admin/monitoring
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=settings.APP_DEBUG,
        future=True,
        pool_pre_ping=True,
        pool_size=30,  # Base pool size for steady-state operations
        max_overflow=50,  # Burst capacity for high load
        pool_recycle=3600,  # Recycle connections every hour
        pool_timeout=30,  # Max time to get connection from pool
        connect_args={
            "timeout": 5,
            "command_timeout": 5,
            "server_settings": {
                "application_name": "enclava_backend",
            },
        },
    )

    # Create synchronous engine for legacy code paths and startup operations
    # IMPORTANT: This pool should be MINIMAL - prefer async operations for all new code
    # Most budget enforcement, chatbot, and API operations now use async sessions
    # Pool sizing: 5 base + 10 overflow = 15 max connections
    sync_engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.APP_DEBUG,
        future=True,
        pool_pre_ping=True,
        pool_size=5,  # Minimal - only for startup/migrations/legacy paths
        max_overflow=10,  # Small burst for edge cases
        pool_recycle=3600,  # Recycle connections every hour
        pool_timeout=30,  # Max time to get connection from pool
        connect_args={
            "connect_timeout": 5,
            "application_name": "enclava_backend_sync",
        },
    )

    logger.info("Database: PostgreSQL")
    return async_engine, sync_engine


def _create_sqlite_engines():
    """Create SQLite engines."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url[10:]
    else:
        db_path = "./data/enclava.db"

    async_url = f"sqlite+aiosqlite:///{db_path}"

    async_engine = create_async_engine(
        async_url,
        echo=settings.APP_DEBUG,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    sync_engine = create_engine(
        db_url,
        echo=settings.APP_DEBUG,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    logger.info(f"Database: SQLite ({db_path})")
    return async_engine, sync_engine


# ============================================================================
# Engine Creation
# ============================================================================

if _is_sqlite():
    engine, sync_engine = _create_sqlite_engines()
else:
    engine, sync_engine = _create_postgresql_engines()

# Create async session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create sync session factory
# NOTE: Prefer async_session_factory for all new code
SessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
)

# Create base class for models
Base = declarative_base()


def utc_now() -> datetime:
    """
    Return current UTC time as a naive datetime (no timezone info).

    This is required because our database columns use TIMESTAMP WITHOUT TIME ZONE.
    PostgreSQL/asyncpg cannot accept timezone-aware Python datetimes for these columns.

    Usage in models:
        from app.db.database import Base, utc_now
        created_at = Column(DateTime, default=utc_now)
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ============================================================================
# Pool Monitoring (PostgreSQL only)
# ============================================================================

def _setup_pool_monitoring():
    """Set up event listeners for pool monitoring (PostgreSQL only)"""

    # Sync engine pool events
    @event.listens_for(sync_engine, "checkout")
    def sync_checkout(dbapi_conn, connection_record, connection_proxy):
        _pool_metrics["sync_checkouts"] += 1
        pool = sync_engine.pool
        if pool.overflow() > 0:
            _pool_metrics["sync_overflow"] = pool.overflow()
            logger.debug(f"Sync pool checkout (overflow: {pool.overflow()})")

    @event.listens_for(sync_engine, "checkin")
    def sync_checkin(dbapi_conn, connection_record):
        _pool_metrics["sync_checkins"] += 1

    # Note: Async engine pool events work on the underlying sync engine
    # We access it via engine.sync_engine for event registration
    try:
        sync_async_engine = engine.sync_engine

        @event.listens_for(sync_async_engine, "checkout")
        def async_checkout(dbapi_conn, connection_record, connection_proxy):
            _pool_metrics["async_checkouts"] += 1
            pool = sync_async_engine.pool
            if pool.overflow() > 0:
                _pool_metrics["async_overflow"] = pool.overflow()
                logger.debug(f"Async pool checkout (overflow: {pool.overflow()})")

        @event.listens_for(sync_async_engine, "checkin")
        def async_checkin(dbapi_conn, connection_record):
            _pool_metrics["async_checkins"] += 1

    except Exception as e:
        logger.warning(f"Could not set up async pool monitoring: {e}")


def get_pool_status() -> Dict[str, Any]:
    """
    Get current status of database connection pools.

    Returns:
        Dict containing pool statistics for both async and sync engines
    """
    if _is_sqlite():
        return {
            "backend": "sqlite",
            "message": "Pool monitoring not applicable for SQLite (uses StaticPool)",
        }

    try:
        # Get async pool status
        async_pool = engine.sync_engine.pool
        async_status = {
            "size": async_pool.size(),
            "checked_in": async_pool.checkedin(),
            "checked_out": async_pool.checkedout(),
            "overflow": async_pool.overflow(),
            "invalid": async_pool.invalidatedcount() if hasattr(async_pool, 'invalidatedcount') else 0,
        }
    except Exception as e:
        async_status = {"error": str(e)}

    try:
        # Get sync pool status
        sync_pool = sync_engine.pool
        sync_status = {
            "size": sync_pool.size(),
            "checked_in": sync_pool.checkedin(),
            "checked_out": sync_pool.checkedout(),
            "overflow": sync_pool.overflow(),
            "invalid": sync_pool.invalidatedcount() if hasattr(sync_pool, 'invalidatedcount') else 0,
        }
    except Exception as e:
        sync_status = {"error": str(e)}

    return {
        "backend": "postgresql",
        "async_pool": async_status,
        "sync_pool": sync_status,
        "metrics": _pool_metrics.copy(),
        "config": {
            "async_pool_size": 30,
            "async_max_overflow": 50,
            "async_max_connections": 80,
            "sync_pool_size": 5,
            "sync_max_overflow": 10,
            "sync_max_connections": 15,
            "total_max_connections": 95,
        }
    }


def log_pool_status():
    """Log current pool status (useful for debugging)"""
    status = get_pool_status()
    logger.info(f"Database pool status: {status}")


# Initialize pool monitoring (PostgreSQL only)
if _is_postgresql():
    _setup_pool_monitoring()

# Metadata for migrations
metadata = MetaData()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    from fastapi import HTTPException
    from starlette.exceptions import HTTPException as StarletteHTTPException

    async with async_session_factory() as session:
        try:
            yield session
        except (HTTPException, StarletteHTTPException):
            # Don't log HTTP exceptions - these are normal API responses (401, 403, 404, etc.)
            # Just rollback any pending transaction and re-raise
            await session.rollback()
            raise
        except SQLAlchemyError as e:
            # Log actual database errors
            logger.error(f"Database error during request: {e}")
            await session.rollback()
            raise
        except Exception as e:
            # Log unexpected errors but don't treat them as database failures
            logger.warning(f"Request error (non-database): {type(e).__name__}")
            await session.rollback()
            raise


async def init_db():
    """Initialize database"""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models.user import User
            from app.models.role import Role
            from app.models.api_key import APIKey
            from app.models.usage_tracking import UsageTracking

            # Import additional models - these are available
            try:
                from app.models.budget import Budget
            except ImportError:
                logger.warning("Budget model not available yet")

            try:
                from app.models.audit_log import AuditLog
            except ImportError:
                logger.warning("AuditLog model not available yet")

            try:
                from app.models.module import Module
            except ImportError:
                logger.warning("Module model not available yet")

            # Tables are now created via migration container - no need to create here
            # await conn.run_sync(Base.metadata.create_all)  # DISABLED - migrations handle this

        # Create default roles if they don't exist
        await create_default_roles()

        # Create default admin user if no admin exists
        await create_default_admin()

        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def create_default_roles():
    """Create default roles if they don't exist"""
    from app.models.role import Role, RoleLevel
    from sqlalchemy import select, text
    from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

    try:
        async with async_session_factory() as session:
            # Check if the roles table exists first
            check_table = text(_sql_table_exists('roles'))
            result = await session.execute(check_table)
            table_exists = _parse_table_exists(result)

            if not table_exists:
                logger.warning("Roles table does not exist yet - waiting for migrations")
                return

            # Check if any roles exist
            stmt = select(Role).limit(1)
            result = await session.execute(stmt)
            existing_role = result.scalar_one_or_none()

            if existing_role:
                logger.info("Roles already exist - skipping default role creation")
                return

            # Create default roles using the Role.create_default_roles class method
            default_roles = Role.create_default_roles()

            for role in default_roles:
                session.add(role)

            await session.commit()

            logger.info("Created default roles: read_only, user, admin, super_admin")

    except ProgrammingError as e:
        if "does not exist" in str(e):
            logger.warning("Roles table does not exist yet - waiting for migrations")
        else:
            logger.error(f"Failed to create default roles due to database error: {e}")
            raise
    except SQLAlchemyError as e:
        logger.error(f"Failed to create default roles due to database error: {e}")
        raise


async def create_default_admin():
    """Create default admin user if user with ADMIN_EMAIL doesn't exist"""
    from app.models.user import User
    from app.models.role import Role
    from app.core.security import get_password_hash
    from app.core.config import settings
    from sqlalchemy import select, text
    from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

    try:
        admin_email = settings.ADMIN_EMAIL
        admin_password = settings.ADMIN_PASSWORD

        if not admin_email or not admin_password:
            logger.info("Admin bootstrap skipped: ADMIN_EMAIL or ADMIN_PASSWORD unset")
            return

        async with async_session_factory() as session:
            # Check if required tables exist first
            check_users = text(_sql_table_exists('users'))
            check_roles = text(_sql_table_exists('roles'))

            result_users = await session.execute(check_users)
            result_roles = await session.execute(check_roles)

            users_exist = _parse_table_exists(result_users)
            roles_exist = _parse_table_exists(result_roles)

            if not users_exist or not roles_exist:
                logger.warning("Users/roles tables do not exist yet - waiting for migrations")
                return

            # Check if user with ADMIN_EMAIL exists
            stmt = select(User).where(User.email == admin_email)
            result = await session.execute(stmt)
            existing_user = result.scalar_one_or_none()

            if existing_user:
                logger.info(
                    f"User with email {admin_email} already exists - skipping admin creation"
                )
                return

            # Get the super_admin role
            stmt = select(Role).where(Role.name == "super_admin")
            result = await session.execute(stmt)
            super_admin_role = result.scalar_one_or_none()

            if not super_admin_role:
                logger.error("Super admin role not found - cannot create admin user")
                return

            # Create admin user from environment variables
            # Generate username from email (part before @)
            admin_username = admin_email.split("@")[0]

            admin_user = User.create_default_admin(
                email=admin_email,
                username=admin_username,
                password_hash=get_password_hash(admin_password),
            )

            # Assign the super_admin role
            admin_user.role_id = super_admin_role.id

            session.add(admin_user)
            await session.commit()

            logger.warning("=" * 60)
            logger.warning("ADMIN USER CREATED FROM ENVIRONMENT")
            logger.warning(f"Email: {admin_email}")
            logger.warning(f"Username: {admin_username}")
            logger.warning("Role: Super Administrator")
            logger.warning(
                "Password: [Set via ADMIN_PASSWORD - only used on first creation]"
            )
            logger.warning("PLEASE CHANGE THE PASSWORD AFTER FIRST LOGIN")
            logger.warning("=" * 60)

    except ProgrammingError as e:
        if "does not exist" in str(e):
            logger.warning("Users/roles tables do not exist yet - waiting for migrations")
        else:
            logger.error(f"Failed to create default admin user due to database error: {e}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to create default admin user due to database error: {e}")
    except AttributeError as e:
        logger.error(
            f"Failed to create default admin user: invalid ADMIN_EMAIL '{settings.ADMIN_EMAIL}'"
        )
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")
        # Don't raise here as this shouldn't block the application startup
