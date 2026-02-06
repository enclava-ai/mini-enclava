"""Settings Web Routes"""

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.services.llm.service import llm_service

router = APIRouter()


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
        provider_health = await llm_service.get_provider_health()
        for provider_name, health_info in provider_health.items():
            models = []
            try:
                provider_models = await llm_service.get_models(provider_name)
                models = [{"id": m.id, "name": m.name or m.id} for m in provider_models]
            except Exception:
                pass

            llm_providers.append({
                "name": provider_name,
                "healthy": health_info.get("status") == "healthy",
                "status": health_info.get("status", "unknown"),
                "models": models,
                "model_count": len(models),
            })
    except Exception:
        pass

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
