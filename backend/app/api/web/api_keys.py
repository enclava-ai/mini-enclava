"""API Keys Management Web Routes"""

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import Optional
import secrets

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.core.security import get_api_key_hash
from app.db.database import get_db
from app.models.user import User
from app.models.api_key import APIKey

router = APIRouter()


def generate_api_key() -> tuple[str, str]:
    """Generate a secure API key and its hash.

    Returns:
        Tuple of (full_key, key_hash)
    """
    full_key = f"enc_{secrets.token_urlsafe(32)}"
    key_hash = get_api_key_hash(full_key)
    return full_key, key_hash


@router.get("/api-keys", response_class=HTMLResponse)
async def api_keys_page(
    request: Request,
    page: int = 1,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render API keys management page."""
    csrf_token = get_csrf_token(request)

    # Get API keys with pagination
    per_page = 10
    offset = (page - 1) * per_page

    # Count total
    count_result = await db.execute(
        select(APIKey).where(APIKey.user_id == user.id)
    )
    total = len(count_result.scalars().all())
    total_pages = (total + per_page - 1) // per_page

    # Get page of keys
    keys_result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == user.id)
        .order_by(APIKey.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    api_keys = keys_result.scalars().all()

    # For HTMX table updates
    if request.headers.get("HX-Request") and request.headers.get("HX-Target") == "table-container":
        return templates.TemplateResponse(
            "pages/api_keys/_list.html",
            {
                "request": request,
                "user": user,
                "csrf_token": csrf_token,
                "api_keys": api_keys,
                "page": page,
                "total_pages": total_pages,
            },
        )

    return templates.TemplateResponse(
        "pages/api_keys/index.html",
        {
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "api_keys": api_keys,
            "page": page,
            "total_pages": total_pages,
        },
    )


@router.post("/api-keys", response_class=HTMLResponse)
async def create_api_key(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    expires_at: Optional[str] = Form(None),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # Generate the key and hash it
    key_value, key_hash = generate_api_key()

    # Parse expiration date if provided
    expiration = None
    if expires_at:
        try:
            expiration = datetime.fromisoformat(expires_at).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Create the API key with hashed key
    api_key = APIKey(
        name=name,
        description=description,
        key_hash=key_hash,
        key_prefix=key_value[:12],
        user_id=user.id,
        is_active=True,
        expires_at=expiration,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    # Return the created key (show full key once)
    return templates.TemplateResponse(
        "pages/api_keys/_created.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "api_key": api_key,
            "full_key": key_value,  # Only shown once
        },
    )


@router.post("/api-keys/{key_id}/toggle", response_class=HTMLResponse)
async def toggle_api_key(
    request: Request,
    key_id: str,
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Toggle API key active status."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = not api_key.is_active
    await db.commit()

    # Return updated row
    return templates.TemplateResponse(
        "pages/api_keys/_row.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "api_key": api_key,
        },
    )


@router.post("/api-keys/{key_id}/regenerate", response_class=HTMLResponse)
async def regenerate_api_key(
    request: Request,
    key_id: str,
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Regenerate an API key."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Generate new key and hash it
    new_key, new_key_hash = generate_api_key()
    api_key.key_hash = new_key_hash
    api_key.key_prefix = new_key[:12]
    await db.commit()

    # Return the regenerated key (show full key once)
    return templates.TemplateResponse(
        "pages/api_keys/_created.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "api_key": api_key,
            "full_key": new_key,
            "regenerated": True,
        },
    )


@router.delete("/api-keys/{key_id}", response_class=HTMLResponse)
async def delete_api_key(
    request: Request,
    key_id: str,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Delete an API key."""
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    await db.delete(api_key)
    await db.commit()

    # Return empty response (row will be removed via hx-swap)
    return HTMLResponse(content="", status_code=200)
