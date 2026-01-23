"""Extract API routes."""

import json
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, get_extract_auth_context
from app.db.database import get_db
from app.models.api_key import APIKey
from app.models.extract_settings import ExtractSettings
from app.models.user import User
from app.modules.extract.exceptions import TemplateExistsError, TemplateNotFoundError
from app.modules.extract.schemas import (
    ExtractSettingsResponse,
    ExtractSettingsUpdate,
    JobDetailResponse,
    JobListResponse,
    ProcessResponse,
    TemplateCreate,
    TemplateListResponse,
    TemplateResponse,
    TemplateUpdate,
)
from app.modules.extract.services.extract_service import ExtractService
from app.modules.extract.services.template_wizard import TemplateWizardService
from app.modules.extract.services.document_processor import DocumentProcessor
from app.modules.extract.templates.manager import TemplateManager

router = APIRouter()
extract_service = ExtractService()
wizard_service = TemplateWizardService()
doc_processor = DocumentProcessor()


# --- Document Processing ---


@router.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="Document to process (PDF, JPG, PNG)"),
    template: Optional[str] = Form(None, description="Template ID to use"),
    context: Optional[str] = Form(None, description="JSON context for template placeholders"),
    db: AsyncSession = Depends(get_db),
    auth: tuple[Dict[str, Any], Optional[APIKey]] = Depends(get_extract_auth_context),
):
    """
    Process a document and extract structured data.

    Supports PDF (multi-page), JPG, and PNG files up to 10MB.

    The extraction uses the specified template or the default template
    if not specified. Context can include variables like company_name, currency, etc.

    Authentication: JWT (frontend) or API key (external)
    Required scope (API key): extract.process
    """
    current_user, api_key = auth

    # Parse context JSON
    context_dict = None
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in context parameter")

    # Get configuration (use default config for now)
    config = {
        "default_template": "detailed_invoice",
        "vision_model": "gpt-4o",
        "max_file_size_mb": 10,
        "image_max_dimension": 1024,
        "auto_reprocess": True,
        "max_retries": 2,
    }
    template_id = template or config.get("default_template", "detailed_invoice")

    # API key permission checks
    if api_key:
        # Check scope
        if not api_key.has_scope("extract.process"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Scope 'extract.process' required"
            )

        # Check template access
        if not api_key.can_access_template(template_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to use template '{template_id}'"
            )

    # Convert user dict to User object for extract_service
    user_obj = User(
        id=current_user["id"],
        email=current_user.get("email"),
        is_superuser=current_user.get("is_superuser", False),
    )

    result = await extract_service.process_document(
        db=db,
        file=file,
        template_id=template_id,
        context=context_dict,
        current_user=user_obj,
        api_key=api_key,
        config=config,
    )

    return result


# --- Job Management ---


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
    auth: tuple[Dict[str, Any], Optional[APIKey]] = Depends(get_extract_auth_context),
):
    """
    List Extract jobs for the current user.

    Authentication: JWT (frontend) or API key (external)
    Required scope (API key): extract.jobs
    """
    current_user, api_key = auth

    # API key scope check
    if api_key and not api_key.has_scope("extract.jobs"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scope 'extract.jobs' required"
        )

    return await extract_service.list_jobs(db, current_user["id"], limit, offset, status)


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    auth: tuple[Dict[str, Any], Optional[APIKey]] = Depends(get_extract_auth_context),
):
    """
    Get job details and extraction result.

    Authentication: JWT (frontend) or API key (external)
    Required scope (API key): extract.jobs
    """
    current_user, api_key = auth

    # API key scope check
    if api_key and not api_key.has_scope("extract.jobs"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scope 'extract.jobs' required"
        )

    return await extract_service.get_job(db, str(job_id), current_user["id"])


# --- Template Management ---


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    db: AsyncSession = Depends(get_db),
    auth: tuple[Dict[str, Any], Optional[APIKey]] = Depends(get_extract_auth_context),
):
    """
    List all extraction templates.

    Authentication: JWT (frontend) or API key (external)
    Required scope (API key): extract.templates
    """
    current_user, api_key = auth

    # API key scope check
    if api_key and not api_key.has_scope("extract.templates"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scope 'extract.templates' required"
        )

    manager = TemplateManager(db)
    templates = await manager.list_templates()
    return TemplateListResponse(templates=templates)


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a template with full details including prompts."""
    manager = TemplateManager(db)
    try:
        template = await manager.get_template(template_id)
        return template
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/templates", response_model=TemplateResponse, status_code=201)
async def create_template(
    template: TemplateCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new extraction template."""
    manager = TemplateManager(db)
    try:
        result = await manager.create_template(template)
        return result
    except TemplateExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    template: TemplateUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update any template (including defaults)."""
    manager = TemplateManager(db)
    try:
        result = await manager.update_template(template_id, template)
        return result
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/templates/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete any template."""
    manager = TemplateManager(db)
    try:
        await manager.delete_template(template_id)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/templates/reset-defaults", status_code=200)
async def reset_default_templates(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Reset all default templates to their original state."""
    manager = TemplateManager(db)
    await manager.reset_defaults()
    return {"status": "defaults_restored"}


@router.post("/templates/wizard")
async def template_wizard(
    file: UploadFile = File(..., description="Sample document to analyze"),
    model: Optional[str] = Form("gpt-4o", description="Vision model to use"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyze a sample document and generate a template automatically.

    Upload an invoice, receipt, contract, or other document and the wizard
    will analyze it to suggest an appropriate extraction template.
    """
    # Get configuration
    config = {
        "vision_model": model or "gpt-4o",
        "max_file_size_mb": 10,
        "image_max_dimension": 1024,
    }

    # Process file to image
    images = await doc_processor.process_file(
        file,
        max_size_mb=config.get("max_file_size_mb", 10),
        max_dimension=config.get("image_max_dimension", 1024),
    )

    # Use first page/image for analysis
    if not images:
        raise HTTPException(status_code=400, detail="No valid images found in file")

    # Analyze with wizard
    template_data = await wizard_service.analyze_document(
        images[0],
        model_name=config.get("vision_model", "gpt-4o")
    )

    # Validate template
    if not wizard_service.validate_template(template_data):
        raise HTTPException(
            status_code=500,
            detail="Generated template is invalid"
        )

    # Format for frontend
    formatted_template = wizard_service.format_template_for_creation(template_data)

    return {
        "success": True,
        "template": formatted_template,
        "analysis": {
            "document_type": template_data.get("document_type", "unknown"),
            "fields": template_data.get("fields", []),
        }
    }


# --- Settings ---


@router.get("/models")
async def get_vision_models():
    """
    Get available vision-capable models from the platform.

    Public endpoint (no auth required for discovery).
    """
    from app.services.llm.service import llm_service

    try:
        all_models = await llm_service.get_models()
        vision_models = [m for m in all_models if "vision" in m.capabilities]

        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.id,  # Use id as name for now
                    "provider": m.provider,
                    "supports_vision": True,
                }
                for m in vision_models
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load available models: {str(e)}",
        )


@router.get("/settings", response_model=ExtractSettingsResponse)
async def get_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get Extract module settings.

    JWT authentication only (not available via API key).
    Auto-populates default_model with first available vision model if not set.
    """
    from app.services.llm.service import llm_service

    stmt = select(ExtractSettings).where(ExtractSettings.id == 1)
    result = await db.execute(stmt)
    settings = result.scalar_one_or_none()

    # Auto-populate default_model if not set
    if not settings or not settings.default_model:
        # Get first available vision model
        try:
            all_models = await llm_service.get_models()
            vision_models = [m for m in all_models if "vision" in m.capabilities]

            if not vision_models:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No vision-capable models available in the platform",
                )

            first_model = vision_models[0].id

            if not settings:
                settings = ExtractSettings(id=1, default_model=first_model)
                db.add(settings)
            else:
                settings.default_model = first_model

            await db.commit()
            await db.refresh(settings)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize settings: {str(e)}",
            )

    return settings


@router.put("/settings", response_model=ExtractSettingsResponse)
async def update_settings(
    settings_data: ExtractSettingsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update Extract module settings.

    JWT authentication only (not available via API key).
    """
    stmt = select(ExtractSettings).where(ExtractSettings.id == 1)
    result = await db.execute(stmt)
    settings = result.scalar_one_or_none()

    if not settings:
        settings = ExtractSettings(id=1)
        db.add(settings)

    settings.default_model = settings_data.default_model
    await db.commit()
    await db.refresh(settings)

    return settings


# --- Health ---


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "module": "extract",
        "version": "1.0.0",
    }
