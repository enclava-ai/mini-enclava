"""Extract Module Web Routes"""

from fastapi import APIRouter, Depends, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import json

from app.core.templates import templates
from app.core.web_auth import get_current_user_from_session, get_csrf_token, verify_csrf_token
from app.db.database import get_db
from app.models.user import User
from app.models.extract_template import ExtractTemplate
from app.models.extract_job import ExtractJob
from app.services.llm.service import llm_service

router = APIRouter()


@router.get("/extract", response_class=HTMLResponse)
async def extract_page(
    request: Request,
    tab: str = "process",
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Render extract module page with tabs."""
    csrf_token = get_csrf_token(request)

    # Get templates for the dropdown (templates are global, filter by is_active)
    templates_result = await db.execute(
        select(ExtractTemplate).where(
            ExtractTemplate.is_active == True
        ).order_by(ExtractTemplate.id)
    )
    extract_templates = templates_result.scalars().all()

    # Get recent jobs for the process tab
    jobs_result = await db.execute(
        select(ExtractJob)
        .where(ExtractJob.user_id == user.id)
        .order_by(ExtractJob.created_at.desc())
        .limit(10)
    )
    recent_jobs = jobs_result.scalars().all()

    # Fetch available vision models from LLM service
    vision_models = []
    try:
        all_models = await llm_service.get_models()
        vision_models = [
            {"value": m.id, "label": m.id}
            for m in all_models
            if "vision" in m.capabilities
        ]
    except Exception as e:
        # Log error for debugging
        import logging
        logging.getLogger(__name__).warning(f"Failed to fetch vision models: {e}")

    context = {
        "request": request,
        "user": user,
        "csrf_token": csrf_token,
        "templates": extract_templates,
        "recent_jobs": recent_jobs,
        "tab": tab,
        "vision_models": vision_models,
    }

    # For HTMX tab switching, return only the partial
    if request.headers.get("HX-Request") and request.headers.get("HX-Target") == "tab-content":
        template_name = f"pages/extract/_{tab}.html"
        return templates.TemplateResponse(template_name, context)

    return templates.TemplateResponse("pages/extract/index.html", context)


@router.post("/extract/process", response_class=HTMLResponse)
async def process_document(
    request: Request,
    template_id: str = Form(...),
    file: UploadFile = File(...),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Process a document using the extract service."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # Import extract service from modular location
    from app.modules.extract.services.extract_service import ExtractService

    extract_service = ExtractService()

    try:
        # Convert user to dict format expected by service
        current_user = {
            "id": str(user.id),
            "email": user.email,
            "is_superuser": user.is_superuser,
        }

        # Process the document
        result = await extract_service.process_document(
            db=db,
            file=file,
            template_id=template_id,
            context=None,
            current_user=current_user,
            api_key=None,
            config=None,
        )

        return templates.TemplateResponse(
            "pages/extract/_result.html",
            {
                "request": request,
                "user": user,
                "csrf_token": get_csrf_token(request),
                "result": result.model_dump(),  # Convert Pydantic model to dict for template
                "success": True,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "pages/extract/_result.html",
            {
                "request": request,
                "user": user,
                "csrf_token": get_csrf_token(request),
                "error": str(e),
                "success": False,
            },
        )


@router.get("/extract/templates", response_class=HTMLResponse)
async def list_templates(
    request: Request,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Get list of extract templates."""
    templates_result = await db.execute(
        select(ExtractTemplate).where(
            ExtractTemplate.is_active == True
        ).order_by(ExtractTemplate.id)
    )
    extract_templates = templates_result.scalars().all()

    return templates.TemplateResponse(
        "pages/extract/_templates_list.html",
        {
            "request": request,
            "user": user,
            "csrf_token": get_csrf_token(request),
            "templates": extract_templates,
        },
    )


@router.post("/extract/templates", response_class=HTMLResponse)
async def create_template(
    request: Request,
    template_id: str = Form(...),
    description: str = Form(""),
    system_prompt: str = Form(...),
    user_prompt: str = Form(...),
    output_schema: str = Form(""),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Create a new extract template."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # Parse output schema if provided
    schema = None
    if output_schema:
        try:
            schema = json.loads(output_schema)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON schema")

    template = ExtractTemplate(
        id=template_id,
        description=description,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=schema,
        is_default=False,
        is_active=True,
    )
    db.add(template)
    await db.commit()
    await db.refresh(template)

    # Return updated template list
    return await list_templates(request, user, db)


@router.put("/extract/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    default_model: str = Form(...),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Update extract module settings."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # TODO: Store settings in user preferences or module config
    # For now, just acknowledge the setting was received

    return templates.TemplateResponse(
        "partials/toast.html",
        {
            "request": request,
            "type": "success",
            "message": f"Default model set to {default_model}",
        },
    )


@router.post("/extract/wizard/analyze", response_class=HTMLResponse)
async def wizard_analyze(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    csrf_token: str = Form(...),
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Analyze a document and generate template suggestion."""
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    import logging
    logger = logging.getLogger(__name__)

    try:
        # Import services
        from app.modules.extract.services.document_processor import DocumentProcessor
        from app.modules.extract.services.template_wizard import TemplateWizardService

        # Process the uploaded file to get base64 image
        processor = DocumentProcessor()
        images_b64 = await processor.process_file(file)

        if not images_b64:
            return templates.TemplateResponse(
                "pages/extract/_wizard_result.html",
                {
                    "request": request,
                    "user": user,
                    "csrf_token": get_csrf_token(request),
                    "error": "Could not process the uploaded file",
                    "success": False,
                },
            )

        # Use first page for analysis
        image_b64 = images_b64[0]

        # Analyze with wizard service
        wizard = TemplateWizardService()
        result = await wizard.analyze_document(
            image_b64=image_b64,
            model_name=model,
            db=db,
            user_id=user.id,
        )

        logger.info(f"Wizard analysis complete: {result.get('document_type')}")

        # Fetch vision models for the form
        vision_models = []
        try:
            all_models = await llm_service.get_models()
            vision_models = [
                {"value": m.id, "label": m.id}
                for m in all_models
                if "vision" in m.capabilities
            ]
        except Exception:
            pass

        return templates.TemplateResponse(
            "pages/extract/_wizard_result.html",
            {
                "request": request,
                "user": user,
                "csrf_token": get_csrf_token(request),
                "result": result,
                "vision_models": vision_models,
                "selected_model": model,
                "success": True,
            },
        )

    except Exception as e:
        logger.exception(f"Wizard analysis failed: {e}")
        return templates.TemplateResponse(
            "pages/extract/_wizard_result.html",
            {
                "request": request,
                "user": user,
                "csrf_token": get_csrf_token(request),
                "error": str(e),
                "success": False,
            },
        )


@router.delete("/extract/templates/{template_id}", response_class=HTMLResponse)
async def delete_template(
    request: Request,
    template_id: str,
    user: User = Depends(get_current_user_from_session),
    db: AsyncSession = Depends(get_db),
):
    """Delete an extract template (soft delete by setting is_active=False)."""
    result = await db.execute(
        select(ExtractTemplate).where(
            ExtractTemplate.id == template_id,
            ExtractTemplate.is_default == False,  # Can't delete default templates
        )
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    await db.delete(template)
    await db.commit()

    # Return updated template list
    return await list_templates(request, user, db)
