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
from app.models.usage_record import UsageRecord

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

    # Get analytics data from UsageRecord (the actual usage table)
    # Total requests
    total_requests_result = await db.execute(
        select(func.count(UsageRecord.id)).where(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= start_date,
        )
    )
    total_requests = total_requests_result.scalar() or 0

    # Total tokens
    total_tokens_result = await db.execute(
        select(func.sum(UsageRecord.total_tokens)).where(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= start_date,
        )
    )
    total_tokens = total_tokens_result.scalar() or 0

    # Total cost (in cents, convert to dollars)
    total_cost_result = await db.execute(
        select(func.sum(UsageRecord.total_cost_cents)).where(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= start_date,
        )
    )
    total_cost_cents = total_cost_result.scalar() or 0
    total_cost = total_cost_cents / 100  # Convert cents to dollars

    # Usage by endpoint
    endpoint_usage_result = await db.execute(
        select(
            UsageRecord.endpoint,
            func.count(UsageRecord.id).label("count"),
        )
        .where(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= start_date,
        )
        .group_by(UsageRecord.endpoint)
        .order_by(func.count(UsageRecord.id).desc())
        .limit(10)
    )
    endpoint_usage = endpoint_usage_result.all()

    # Usage by model
    model_usage_result = await db.execute(
        select(
            UsageRecord.normalized_model,
            func.count(UsageRecord.id).label("count"),
            func.sum(UsageRecord.total_tokens).label("tokens"),
        )
        .where(
            UsageRecord.user_id == user.id,
            UsageRecord.created_at >= start_date,
            UsageRecord.normalized_model.isnot(None),
        )
        .group_by(UsageRecord.normalized_model)
        .order_by(func.count(UsageRecord.id).desc())
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
