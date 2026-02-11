"""Budget Management Web Routes"""

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from decimal import Decimal
from datetime import datetime, timedelta

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.db.database import get_db, utc_now
from app.models.user import User
from app.models.budget import Budget

router = APIRouter()


def _calculate_period_dates(period_type: str) -> tuple[datetime, datetime]:
    """Calculate period start and end dates based on period type."""
    now = utc_now()
    if period_type == "daily":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period_type == "weekly":
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(weeks=1)
    elif period_type == "yearly":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1)
    else:  # monthly (default)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    return start, end


@router.get("/budgets", response_class=HTMLResponse)
async def budgets_page(
    request: Request,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render budgets management page."""
    csrf_token = get_csrf_token(request)

    # Get user's budgets
    budgets_result = await db.execute(
        select(Budget)
        .where(Budget.user_id == user.id)
        .order_by(Budget.created_at.desc())
    )
    budgets = budgets_result.scalars().all()

    return templates.TemplateResponse(
        "pages/budgets/index.html",
        {
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "budgets": budgets,
        },
    )


@router.post("/budgets", response_class=HTMLResponse)
async def create_budget(
    request: Request,
    name: str = Form(...),
    amount: str = Form(...),
    period: str = Form(...),
    warning_threshold: str = Form("80"),
    auto_renew: bool = Form(False),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Create a new budget."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    try:
        # Convert dollars to cents
        amount_dollars = Decimal(amount)
        limit_cents = int(amount_dollars * 100)

        # Convert warning threshold percentage to cents
        warning_pct = Decimal(warning_threshold)
        warning_threshold_cents = int((warning_pct / 100) * limit_cents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount or threshold")

    # Calculate period dates
    period_start, period_end = _calculate_period_dates(period)

    budget = Budget(
        name=name,
        limit_cents=limit_cents,
        period_type=period,
        warning_threshold_cents=warning_threshold_cents,
        auto_renew=auto_renew,
        user_id=user.id,
        current_usage_cents=0,
        period_start=period_start,
        period_end=period_end,
    )
    db.add(budget)
    await db.commit()
    await db.refresh(budget)

    # Return updated budget list
    return await get_budget_list(request, user, db)


@router.get("/budgets/list", response_class=HTMLResponse)
async def get_budget_list(
    request: Request,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Get budget list partial."""
    budgets_result = await db.execute(
        select(Budget)
        .where(Budget.user_id == user.id)
        .order_by(Budget.created_at.desc())
    )
    budgets = budgets_result.scalars().all()

    return templates.TemplateResponse(
        "pages/budgets/_list.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "budgets": budgets,
        },
    )


@router.put("/budgets/{budget_id}", response_class=HTMLResponse)
async def update_budget(
    request: Request,
    budget_id: str,
    name: str = Form(...),
    amount: str = Form(...),
    period: str = Form(...),
    warning_threshold: str = Form("80"),
    auto_renew: bool = Form(False),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Update a budget."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    result = await db.execute(
        select(Budget).where(
            Budget.id == budget_id,
            Budget.user_id == user.id,
        )
    )
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")

    try:
        # Convert dollars to cents
        amount_dollars = Decimal(amount)
        limit_cents = int(amount_dollars * 100)

        # Convert warning threshold percentage to cents
        warning_pct = Decimal(warning_threshold)
        warning_threshold_cents = int((warning_pct / 100) * limit_cents)

        budget.name = name
        budget.limit_cents = limit_cents
        budget.period_type = period
        budget.warning_threshold_cents = warning_threshold_cents
        budget.auto_renew = auto_renew

        # Update period dates if period type changed
        if budget.period_type != period:
            period_start, period_end = _calculate_period_dates(period)
            budget.period_start = period_start
            budget.period_end = period_end

        await db.commit()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid budget data: {str(e)}")

    # Return updated budget list
    return await get_budget_list(request, user, db)


@router.delete("/budgets/{budget_id}", response_class=HTMLResponse)
async def delete_budget(
    request: Request,
    budget_id: str,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Delete a budget."""
    result = await db.execute(
        select(Budget).where(
            Budget.id == budget_id,
            Budget.user_id == user.id,
        )
    )
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")

    await db.delete(budget)
    await db.commit()

    # Return empty response
    return HTMLResponse(content="", status_code=200)
