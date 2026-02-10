"""Authentication Web Routes"""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.templates import templates
from app.core.web_auth import (
    get_csrf_token,
    get_optional_user_from_session,
    login_user,
    logout_user,
    redirect_to_dashboard,
    verify_csrf_token,
)
from app.core.security import verify_password
from app.db.database import get_db
from app.models.user import User
from sqlalchemy import select

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    message: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Render login page."""
    # Check if already logged in
    user = await get_optional_user_from_session(request, db)
    if user:
        return redirect_to_dashboard()

    csrf_token = get_csrf_token(request)

    return templates.TemplateResponse(
        "pages/auth/login.html",
        {
            "request": request,
            "csrf_token": csrf_token,
            "message": message,
            "user": None,
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Process login form submission."""
    # Verify CSRF token
    if not verify_csrf_token(request, csrf_token):
        return templates.TemplateResponse(
            "pages/auth/login.html",
            {
                "request": request,
                "csrf_token": get_csrf_token(request),
                "error": "Invalid request. Please try again.",
                "email": email,
                "user": None,
            },
            status_code=400,
        )

    # Find user by email
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()

    # Verify credentials
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            "pages/auth/login.html",
            {
                "request": request,
                "csrf_token": get_csrf_token(request),
                "error": "Invalid email or password.",
                "email": email,
                "user": None,
            },
            status_code=401,
        )

    # Check if user is active
    if not user.is_active:
        return templates.TemplateResponse(
            "pages/auth/login.html",
            {
                "request": request,
                "csrf_token": get_csrf_token(request),
                "error": "Your account is disabled. Please contact support.",
                "email": email,
                "user": None,
            },
            status_code=403,
        )

    # Log in user
    login_user(request, user)

    # Redirect to dashboard
    return redirect_to_dashboard()


@router.post("/logout")
async def logout(
    request: Request,
    csrf_token: str = Form(...),
):
    """Process logout."""
    if verify_csrf_token(request, csrf_token):
        logout_user(request)

    return RedirectResponse(url="/login", status_code=303)


@router.get("/", response_class=HTMLResponse)
async def root_redirect(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Redirect root to dashboard or login."""
    user = await get_optional_user_from_session(request, db)
    if user:
        return redirect_to_dashboard()
    return RedirectResponse(url="/login", status_code=303)
