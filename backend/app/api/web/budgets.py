"""Budget Management Web Routes"""

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from decimal import Decimal

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.models.budget import Budget

router = APIRouter()


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
        amount_decimal = Decimal(amount)
        warning_decimal = Decimal(warning_threshold)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount or threshold")

    budget = Budget(
        name=name,
        amount=amount_decimal,
        period=period,
        warning_threshold=warning_decimal,
        auto_renew=auto_renew,
        user_id=user.id,
        spent=Decimal("0"),
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
        budget.name = name
        budget.amount = Decimal(amount)
        budget.period = period
        budget.warning_threshold = Decimal(warning_threshold)
        budget.auto_renew = auto_renew
        await db.commit()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid budget data")

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
