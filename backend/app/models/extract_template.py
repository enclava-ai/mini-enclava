"""Extraction template model."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.database import Base


class ExtractTemplate(Base):
    """
    Extraction template for Extract processing.

    Templates define the prompts sent to the vision model and the expected
    output schema. All templates are user-editable.

    Attributes:
        id: Unique template identifier (e.g., "detailed_invoice") - serves as both ID and display name
        description: Template description/purpose
        system_prompt: System message for the vision model
        user_prompt: User message template (supports placeholders like {company_name})
        output_schema: Expected JSON schema for validation (optional)
        context_schema: Defines context variables this template accepts (optional)
        is_default: True for templates that shipped with the module
        is_active: Soft delete flag
    """

    __tablename__ = "extract_templates"

    id = Column(String(100), primary_key=True)
    description = Column(Text, nullable=True)

    system_prompt = Column(Text, nullable=False)
    user_prompt = Column(Text, nullable=False)
    output_schema = Column(JSONB, nullable=True)
    context_schema = Column(JSONB, nullable=True)

    is_default = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ExtractTemplate {self.id}>"
