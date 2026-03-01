"""Settings Web Routes"""

import logging

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.services.llm.service import llm_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render settings page."""
    csrf_token = get_csrf_token(request)

    # Get module status from app state
    module_manager = request.app.state.module_manager
    modules = {}
    if module_manager:
        for name, module in module_manager.modules.items():
            modules[name] = {
                "name": name,
                "enabled": module.enabled if hasattr(module, "enabled") else True,
                "healthy": True,  # TODO: Add health check
            }

    # Get LLM provider status
    llm_providers = []
    try:
        provider_health = await llm_service.get_providers_health(db)
        for provider_info in provider_health:
            models = []
            for model in provider_info.get("models", []):
                model_id = model.get("id")
                models.append({
                    "id": model_id,
                    "name": model_id,
                })

            is_healthy = bool(provider_info.get("healthy"))
            llm_providers.append({
                "name": provider_info.get("provider_id", "unknown"),
                "healthy": is_healthy,
                "status": "healthy" if is_healthy else "unhealthy",
                "models": models,
                "model_count": len(models),
            })
    except Exception as exc:
        logger.warning("Failed to fetch LLM provider status for settings page: %s", exc)

    return templates.TemplateResponse(
        "pages/settings/index.html",
        {
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "modules": modules,
            "llm_providers": llm_providers,
        },
    )


@router.post("/settings/profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    name: str = Form(...),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Update user profile."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    user.name = name
    await db.commit()

    return templates.TemplateResponse(
        "pages/settings/_profile_form.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "success": True,
            "message": "Profile updated successfully",
        },
    )
