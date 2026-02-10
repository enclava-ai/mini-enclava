"""Analytics Web Routes"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.models.usage_tracking import UsageTracking

router = APIRouter()


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    days: int = 7,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render analytics page."""
    csrf_token = get_csrf_token(request)

    # Calculate date range (using naive datetime to match database)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Get analytics data from UsageTracking
    # Total requests
    total_requests_result = await db.execute(
        select(func.count(UsageTracking.id)).where(
            UsageTracking.user_id == user.id,
            UsageTracking.created_at >= start_date,
        )
    )
    total_requests = total_requests_result.scalar() or 0

    # Total tokens
    total_tokens_result = await db.execute(
        select(func.sum(UsageTracking.total_tokens)).where(
            UsageTracking.user_id == user.id,
            UsageTracking.created_at >= start_date,
        )
    )
    total_tokens = total_tokens_result.scalar() or 0

    # Total cost (in cents, convert to dollars)
    total_cost_result = await db.execute(
        select(func.sum(UsageTracking.cost_cents)).where(
            UsageTracking.user_id == user.id,
            UsageTracking.created_at >= start_date,
        )
    )
    total_cost_cents = total_cost_result.scalar() or 0
    total_cost = total_cost_cents / 100  # Convert cents to dollars

    # Usage by endpoint
    endpoint_usage_result = await db.execute(
        select(
            UsageTracking.endpoint,
            func.count(UsageTracking.id).label("count"),
        )
        .where(
            UsageTracking.user_id == user.id,
            UsageTracking.created_at >= start_date,
        )
        .group_by(UsageTracking.endpoint)
        .order_by(func.count(UsageTracking.id).desc())
        .limit(10)
    )
    endpoint_usage = endpoint_usage_result.all()

    # Usage by model
    model_usage_result = await db.execute(
        select(
            UsageTracking.model,
            func.count(UsageTracking.id).label("count"),
            func.sum(UsageTracking.total_tokens).label("tokens"),
        )
        .where(
            UsageTracking.user_id == user.id,
            UsageTracking.created_at >= start_date,
            UsageTracking.model.isnot(None),
        )
        .group_by(UsageTracking.model)
        .order_by(func.count(UsageTracking.id).desc())
        .limit(10)
    )
    model_usage = model_usage_result.all()

    return templates.TemplateResponse(
        "pages/analytics/index.html",
        {
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "days": days,
            "stats": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost": float(total_cost) if total_cost else 0,
            },
            "endpoint_usage": endpoint_usage,
            "model_usage": model_usage,
        },
    )
