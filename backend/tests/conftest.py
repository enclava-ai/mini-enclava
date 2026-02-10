"""
Pytest configuration and shared fixtures for all tests.

Supports both PostgreSQL and SQLite backends for testing:
- Set TEST_DATABASE_BACKEND=sqlite for fast local testing (default)
- Set TEST_DATABASE_BACKEND=postgresql for full integration tests

Environment Variables:
- TEST_DATABASE_BACKEND: "sqlite" (default) or "postgresql"
- TEST_DATABASE_URL: Override for PostgreSQL connection string
- SQLITE_TEST_DB_PATH: Path for SQLite test database (default: :memory:)
"""
import os
import sys
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool, StaticPool
from sqlalchemy import event
import aiohttp
from httpx import AsyncClient
import uuid

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import Base, get_db
from app.core.config import settings
from app.main import app


# ============================================================================
# Test Database Configuration
# ============================================================================

# Determine which database backend to use for testing
TEST_DATABASE_BACKEND = os.getenv("TEST_DATABASE_BACKEND", "sqlite").lower()

# PostgreSQL test configuration
POSTGRESQL_TEST_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://enclava_user:enclava_pass@localhost:5432/enclava_test_db"
)

# SQLite test configuration
SQLITE_TEST_DB_PATH = os.getenv("SQLITE_TEST_DB_PATH", ":memory:")
SQLITE_ASYNC_TEST_URL = f"sqlite+aiosqlite:///{SQLITE_TEST_DB_PATH}"


def _is_sqlite_test() -> bool:
    """Check if tests are running with SQLite backend."""
    return TEST_DATABASE_BACKEND == "sqlite"


def _is_postgresql_test() -> bool:
    """Check if tests are running with PostgreSQL backend."""
    return TEST_DATABASE_BACKEND == "postgresql"


# ============================================================================
# Engine Creation
# ============================================================================

def _create_test_engine():
    """Create test engine based on selected backend."""
    if _is_sqlite_test():
        # SQLite test engine with StaticPool for in-memory database
        engine = create_async_engine(
            SQLITE_ASYNC_TEST_URL,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Enable foreign key enforcement for SQLite
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    else:
        # PostgreSQL test engine with NullPool for isolation
        return create_async_engine(
            POSTGRESQL_TEST_URL,
            echo=False,
            pool_pre_ping=True,
            poolclass=NullPool
        )


# Create the test engine
test_engine = _create_test_engine()

# Create test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "postgresql_only: mark test as PostgreSQL-only (skipped on SQLite)"
    )
    config.addinivalue_line(
        "markers", "sqlite_only: mark test as SQLite-only (skipped on PostgreSQL)"
    )
    config.addinivalue_line(
        "markers", "db_backend(backend): mark test for specific database backend"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on database backend markers."""
    for item in items:
        # Skip postgresql_only tests when running SQLite
        if _is_sqlite_test() and "postgresql_only" in [m.name for m in item.iter_markers()]:
            item.add_marker(pytest.mark.skip(reason="Test requires PostgreSQL backend"))

        # Skip sqlite_only tests when running PostgreSQL
        if _is_postgresql_test() and "sqlite_only" in [m.name for m in item.iter_markers()]:
            item.add_marker(pytest.mark.skip(reason="Test requires SQLite backend"))

        # Check db_backend marker for specific backend requirements
        for marker in item.iter_markers(name="db_backend"):
            required_backend = marker.args[0] if marker.args else None
            if required_backend and required_backend.lower() != TEST_DATABASE_BACKEND:
                item.add_marker(
                    pytest.mark.skip(reason=f"Test requires {required_backend} backend")
                )


# ============================================================================
# Core Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def db_backend():
    """Return the current test database backend name."""
    return TEST_DATABASE_BACKEND


@pytest.fixture(scope="session")
def is_sqlite():
    """Return True if running with SQLite backend."""
    return _is_sqlite_test()


@pytest.fixture(scope="session")
def is_postgresql():
    """Return True if running with PostgreSQL backend."""
    return _is_postgresql_test()


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with automatic rollback."""
    async with test_engine.begin() as conn:
        # Create all tables for this test
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session
        # Rollback any changes made during the test
        await session.rollback()

    # Clean up tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing FastAPI endpoints."""
    async def override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def authenticated_client(async_client: AsyncClient, test_user_token: str) -> AsyncClient:
    """Create an authenticated async client with JWT token."""
    async_client.headers.update({"Authorization": f"Bearer {test_user_token}"})
    return async_client


@pytest_asyncio.fixture(scope="function")
async def api_key_client(async_client: AsyncClient, test_api_key: str) -> AsyncClient:
    """Create an async client authenticated with API key."""
    async_client.headers.update({"Authorization": f"Bearer {test_api_key}"})
    return async_client


@pytest_asyncio.fixture(scope="function")
async def nginx_client() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Create an aiohttp client for testing through nginx proxy."""
    async with aiohttp.ClientSession() as session:
        yield session


# ============================================================================
# Qdrant Fixtures (PostgreSQL/full integration only)
# ============================================================================

@pytest.fixture(scope="function")
def qdrant_client():
    """Create a Qdrant client for testing."""
    # Only import and create if not running SQLite-only tests
    if _is_sqlite_test() and not os.getenv("QDRANT_HOST"):
        pytest.skip("Qdrant not available in SQLite-only test mode")

    from qdrant_client import QdrantClient
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )


@pytest_asyncio.fixture(scope="function")
async def test_qdrant_collection(qdrant_client) -> str:
    """Create a test Qdrant collection."""
    from qdrant_client.models import Distance, VectorParams

    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


# ============================================================================
# User & Auth Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_user(test_db: AsyncSession) -> dict:
    """Create a test user."""
    from app.models.user import User
    from app.core.security import get_password_hash

    user = User(
        email="testuser@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpass123"),
        is_active=True,
        is_verified=True
    )

    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)

    return {
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
        "password": "testpass123"
    }


@pytest_asyncio.fixture(scope="function")
async def test_user_token(test_user: dict) -> str:
    """Create a JWT token for test user."""
    from app.core.security import create_access_token

    token_data = {"sub": test_user["email"], "user_id": test_user["id"]}
    return create_access_token(data=token_data)


@pytest_asyncio.fixture(scope="function")
async def test_api_key(test_db: AsyncSession, test_user: dict) -> str:
    """Create a test API key."""
    from app.models.api_key import APIKey
    from app.models.budget import Budget
    import secrets

    # Create budget
    budget = Budget(
        id=str(uuid.uuid4()),
        user_id=test_user["id"],
        limit_amount=100.0,
        period="monthly",
        current_usage=0.0,
        is_active=True
    )
    test_db.add(budget)

    # Create API key
    key = f"sk-test-{secrets.token_urlsafe(32)}"
    api_key = APIKey(
        id=str(uuid.uuid4()),
        key_hash=key,  # In real code, this would be hashed
        name="Test API Key",
        user_id=test_user["id"],
        scopes=["llm.chat", "llm.embeddings"],
        budget_id=budget.id,
        is_active=True
    )
    test_db.add(api_key)
    await test_db.commit()

    return key


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_documents_dir() -> Path:
    """Get the test documents directory."""
    return Path(__file__).parent / "data" / "documents"


@pytest.fixture(scope="session")
def sample_text_path(test_documents_dir: Path) -> Path:
    """Get path to sample text file for testing."""
    text_path = test_documents_dir / "sample.txt"
    if not text_path.exists():
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("""
        Enclava Platform Documentation

        This is a sample document for testing the RAG system.
        It contains information about the Enclava platform's features and capabilities.

        Features:
        - Secure LLM access through PrivateMode.ai
        - Chatbot creation and management
        - RAG (Retrieval Augmented Generation) support
        - OpenAI-compatible API endpoints
        - Budget management and API key controls
        """)
    return text_path


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LLM_PROMPTS"] = "true"
    os.environ["APP_DEBUG"] = "true"

    # Log which backend is being used
    print(f"\n[Test Config] Database backend: {TEST_DATABASE_BACKEND}")
    if _is_sqlite_test():
        print(f"[Test Config] SQLite URL: {SQLITE_ASYNC_TEST_URL}")
    else:
        print(f"[Test Config] PostgreSQL URL: {POSTGRESQL_TEST_URL}")

    yield

    # Cleanup
    os.environ.pop("TESTING", None)


# ============================================================================
# Parametrized Backend Fixtures (for running same test on both backends)
# ============================================================================

@pytest.fixture(params=["sqlite", "postgresql"])
def db_backend_param(request):
    """
    Parametrize tests to potentially run on both backends.

    Note: This fixture is for documentation/future use. In practice,
    tests run on a single backend per test session. To run on both:

    1. Run: TEST_DATABASE_BACKEND=sqlite pytest tests/
    2. Run: TEST_DATABASE_BACKEND=postgresql pytest tests/

    Usage in tests:
        @pytest.mark.parametrize("db_backend_param", ["sqlite", "postgresql"], indirect=True)
        def test_something(db_backend_param, test_db):
            # Test runs once per backend
            pass
    """
    backend = request.param
    current_backend = TEST_DATABASE_BACKEND

    if backend != current_backend:
        pytest.skip(f"Test backend {backend} != current backend {current_backend}")

    return backend


# ============================================================================
# Helper Functions for Tests
# ============================================================================

def skip_if_sqlite(reason: str = "Feature not supported on SQLite"):
    """Decorator to skip a test if running on SQLite."""
    return pytest.mark.skipif(
        _is_sqlite_test(),
        reason=reason
    )


def skip_if_postgresql(reason: str = "Feature not supported on PostgreSQL"):
    """Decorator to skip a test if running on PostgreSQL."""
    return pytest.mark.skipif(
        _is_postgresql_test(),
        reason=reason
    )
