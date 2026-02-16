"""
API v1 package
"""

from fastapi import APIRouter
from app.core.config import settings as app_settings
from .auth import router as auth_router
from .llm import router as llm_router
from .modules import router as modules_router
from .platform import router as platform_router
from .users import router as users_router
from .api_keys import router as api_keys_router
from .budgets import router as budgets_router
from .settings import router as settings_router
from .prompt_templates import router as prompt_templates_router
from .endpoints.tools import router as tools_router
from .endpoints.tool_calling import router as tool_calling_router
from .endpoints.user_management import router as user_management_router

if app_settings.AUDIT_ENABLED:
    from .audit import router as audit_router

if app_settings.ANALYTICS_ENABLED:
    from .analytics import router as analytics_router

if app_settings.RAG_ENABLED:
    from .rag import router as rag_router

if app_settings.CHATBOTS_ENABLED:
    from .chatbot import router as chatbot_router

if app_settings.EXTRACT_ENABLED:
    from .extract import router as extract_router

if app_settings.PLUGINS_ENABLED:
    from .plugin_registry import router as plugin_registry_router

# Create main API router
api_router = APIRouter()

# Include authentication routes
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])

# Include LLM proxy routes
api_router.include_router(llm_router, prefix="/llm", tags=["llm"])

# Include modules routes
api_router.include_router(modules_router, prefix="/modules", tags=["modules"])

# Include platform routes
api_router.include_router(platform_router, prefix="/platform", tags=["platform"])

# Include user management routes
api_router.include_router(users_router, prefix="/users", tags=["users"])

# Include API key management routes
api_router.include_router(api_keys_router, prefix="/api-keys", tags=["api-keys"])

# Include budget management routes
api_router.include_router(budgets_router, prefix="/budgets", tags=["budgets"])

# Include settings management routes
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])

if app_settings.ANALYTICS_ENABLED:
    # Include analytics routes
    api_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])

if app_settings.RAG_ENABLED:
    # Include RAG routes
    api_router.include_router(rag_router, prefix="/rag", tags=["rag"])

if app_settings.CHATBOTS_ENABLED:
    # Include chatbot routes
    api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])

if app_settings.EXTRACT_ENABLED:
    # Include extract routes
    api_router.include_router(extract_router, prefix="/extract", tags=["extract"])

# Include prompt template routes
api_router.include_router(
    prompt_templates_router, prefix="/prompt-templates", tags=["prompt-templates"]
)


# Include plugin registry routes
if app_settings.PLUGINS_ENABLED:
    # Include plugin registry routes
    api_router.include_router(
        plugin_registry_router, prefix="/plugins", tags=["plugins"]
    )

if app_settings.BUILTIN_TOOLS_ENABLED:
    # Include built-in tool management routes
    api_router.include_router(tools_router, prefix="/tools", tags=["tools"])
    api_router.include_router(
        tool_calling_router, prefix="/tool-calling", tags=["tool-calling"]
    )

# Include tool management routes
if app_settings.AUDIT_ENABLED:
    # Include audit log routes
    api_router.include_router(audit_router, prefix="/audit", tags=["audit"])

# Include admin user management routes
api_router.include_router(
    user_management_router, prefix="/admin/user-management", tags=["admin", "user-management"]
)
