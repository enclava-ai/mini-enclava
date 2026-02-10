"""Dashboard Web Routes"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta, timezone

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.models.api_key import APIKey
from app.models.extract_job import ExtractJob

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render dashboard page with stats."""
    csrf_token = get_csrf_token(request)

    # Get stats
    # Active API keys count
    api_keys_result = await db.execute(
        select(func.count(APIKey.id)).where(
            APIKey.user_id == user.id,
            APIKey.is_active == True,
        )
    )
    active_api_keys = api_keys_result.scalar() or 0

    # Extract jobs in last 24 hours
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    extract_jobs_result = await db.execute(
        select(func.count(ExtractJob.id)).where(
            ExtractJob.user_id == user.id,
            ExtractJob.created_at >= yesterday,
        )
    )
    recent_extract_jobs = extract_jobs_result.scalar() or 0

    # Total extract jobs
    total_extract_jobs_result = await db.execute(
        select(func.count(ExtractJob.id)).where(ExtractJob.user_id == user.id)
    )
    total_extract_jobs = total_extract_jobs_result.scalar() or 0

    return templates.TemplateResponse(
        "pages/dashboard/index.html",
        {
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "stats": {
                "active_api_keys": active_api_keys,
                "recent_extract_jobs": recent_extract_jobs,
                "total_extract_jobs": total_extract_jobs,
                "budget_used": 0,  # TODO: Implement budget tracking
                "budget_total": 100,
            },
        },
    )
