"""Session-based Web Authentication for HTMX Frontend"""

import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.user import User


class WebAuthError(Exception):
    """Web authentication error."""

    pass


async def get_current_user_from_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get user from session cookie.
    Raises HTTPException with redirect if not authenticated.
    """
    user_id = request.session.get("user_id")

    if not user_id:
        # For HTMX requests, return 401 to trigger client-side redirect
        if request.headers.get("HX-Request"):
            raise HTTPException(
                status_code=401,
                headers={"HX-Redirect": "/login"},
            )
        # For regular requests, redirect to login
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
        )

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        # Invalid session, clear it
        request.session.clear()
        if request.headers.get("HX-Request"):
            raise HTTPException(
                status_code=401,
                headers={"HX-Redirect": "/login"},
            )
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
        )

    if not user.is_active:
        request.session.clear()
        raise HTTPException(
            status_code=403,
            detail="Account is disabled",
        )

    return user


async def get_optional_user_from_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get user from session if logged in, None otherwise.
    Does not raise exceptions for unauthenticated requests.
    """
    user_id = request.session.get("user_id")

    if not user_id:
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        return None

    return user


def generate_csrf_token(request: Request) -> str:
    """Generate and store CSRF token in session."""
    token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = token
    return token


def get_csrf_token(request: Request) -> str:
    """Get existing CSRF token or generate new one."""
    token = request.session.get("csrf_token")
    if not token:
        token = generate_csrf_token(request)
    return token


def verify_csrf_token(request: Request, token: str) -> bool:
    """Verify CSRF token matches session."""
    session_token = request.session.get("csrf_token")
    if not session_token or not token:
        return False
    return secrets.compare_digest(session_token, token)


async def require_csrf_token(request: Request) -> None:
    """
    Dependency to verify CSRF token for POST/PUT/DELETE requests.
    Token should be in form data as 'csrf_token' or in header as 'X-CSRF-Token'.
    """
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        # Try to get token from form data
        token = None

        # Check header first (for HTMX requests)
        token = request.headers.get("X-CSRF-Token")

        if not token:
            # Try form data
            try:
                form = await request.form()
                token = form.get("csrf_token")
            except Exception:
                pass

        if not token or not verify_csrf_token(request, token):
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing CSRF token",
            )


def login_user(request: Request, user: User) -> None:
    """Log in a user by setting session data."""
    request.session["user_id"] = user.id
    request.session["logged_in_at"] = datetime.now(timezone.utc).isoformat()
    # Regenerate CSRF token on login
    generate_csrf_token(request)


def logout_user(request: Request) -> None:
    """Log out a user by clearing session."""
    request.session.clear()


def redirect_to_login(message: str = "") -> RedirectResponse:
    """Create redirect response to login page."""
    url = "/login"
    if message:
        url += f"?message={message}"
    return RedirectResponse(url=url, status_code=303)


def redirect_to_dashboard() -> RedirectResponse:
    """Create redirect response to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=303)
