"""
Template management service.

All templates are user-editable with no distinction between
built-in and custom templates.
"""

import logging
from datetime import datetime
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.extract_template import ExtractTemplate

from ..exceptions import TemplateExistsError, TemplateNotFoundError
from ..schemas import TemplateCreate, TemplateUpdate
from .defaults import DEFAULT_TEMPLATES

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages Extraction templates.

    Key design decisions:
    - All templates stored in database (not in code)
    - Default templates seeded on first run
    - All templates are user-editable
    - Reset function restores defaults
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    @classmethod
    async def seed_defaults(cls, db: AsyncSession):
        """
        Seed default templates on module initialization.

        Only creates templates that don't already exist,
        preserving user modifications.
        """
        for template_data in DEFAULT_TEMPLATES:
            existing = await db.get(ExtractTemplate, template_data["id"])
            if not existing:
                template = ExtractTemplate(**template_data)
                db.add(template)
                logger.info("Seeded default template: %s", template_data["id"])

        await db.commit()

    async def get_template(self, template_id: str) -> ExtractTemplate:
        """
        Get template by ID.

        Raises:
            TemplateNotFoundError: If template doesn't exist or is inactive
        """
        template = await self.db.get(ExtractTemplate, template_id)

        if not template or not template.is_active:
            raise TemplateNotFoundError(f"Template '{template_id}' not found")

        return template

    async def list_templates(self) -> List[ExtractTemplate]:
        """List all active templates."""
        stmt = (
            select(ExtractTemplate)
            .where(ExtractTemplate.is_active == True)  # noqa: E712
            .order_by(ExtractTemplate.id)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_template(self, data: TemplateCreate) -> ExtractTemplate:
        """
        Create a new template.

        Raises:
            TemplateExistsError: If template ID already exists
        """
        existing = await self.db.get(ExtractTemplate, data.id)
        if existing:
            raise TemplateExistsError(f"Template '{data.id}' already exists")

        template = ExtractTemplate(
            id=data.id,
            description=data.description,
            system_prompt=data.system_prompt,
            user_prompt=data.user_prompt,
            output_schema=data.output_schema,
            is_default=False,  # User-created templates are not defaults
        )

        self.db.add(template)
        await self.db.commit()
        await self.db.refresh(template)

        logger.info("Created template: %s", template.id)
        return template

    async def update_template(
        self,
        template_id: str,
        data: TemplateUpdate,
    ) -> ExtractTemplate:
        """
        Update any template (including defaults).

        Users can modify any template. The is_default flag is preserved
        so reset_defaults() can restore original content.
        """
        template = await self.db.get(ExtractTemplate, template_id)
        if not template:
            raise TemplateNotFoundError(f"Template '{template_id}' not found")

        # Update fields
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(template, field, value)

        template.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(template)

        logger.info("Updated template: %s", template.id)
        return template

    async def delete_template(self, template_id: str):
        """
        Delete any template.

        Users can delete any template, including defaults.
        reset_defaults() can restore deleted default templates.
        """
        template = await self.db.get(ExtractTemplate, template_id)
        if not template:
            raise TemplateNotFoundError(f"Template '{template_id}' not found")

        await self.db.delete(template)
        await self.db.commit()

        logger.info("Deleted template: %s", template_id)

    async def reset_defaults(self):
        """
        Reset all default templates to original state.

        - Updates existing default templates to original content
        - Recreates deleted default templates
        - Does not affect user-created templates
        """
        for template_data in DEFAULT_TEMPLATES:
            existing = await self.db.get(ExtractTemplate, template_data["id"])

            if existing:
                # Update to original values
                for field, value in template_data.items():
                    setattr(existing, field, value)
                existing.updated_at = datetime.utcnow()
                logger.info("Reset template: %s", template_data["id"])
            else:
                # Recreate deleted template
                template = ExtractTemplate(**template_data)
                self.db.add(template)
                logger.info("Restored template: %s", template_data["id"])

        await self.db.commit()
