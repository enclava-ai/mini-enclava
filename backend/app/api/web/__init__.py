"""Web Routes for HTMX/Jinja2 Frontend"""

from fastapi import APIRouter
from fastapi.responses import Response

from app.api.web.auth import router as auth_router
from app.api.web.dashboard import router as dashboard_router
from app.api.web.extract import router as extract_router
from app.api.web.api_keys import router as api_keys_router
from app.api.web.budgets import router as budgets_router
from app.api.web.analytics import router as analytics_router
from app.api.web.settings import router as settings_router

# Main web router
web_router = APIRouter(tags=["web"])


# Favicon route to suppress 404
@web_router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Return empty response - browsers will use default/no icon
    return Response(status_code=204)

# Include all sub-routers
web_router.include_router(auth_router)
web_router.include_router(dashboard_router)
web_router.include_router(extract_router)
web_router.include_router(api_keys_router)
web_router.include_router(budgets_router)
web_router.include_router(analytics_router)
web_router.include_router(settings_router)
