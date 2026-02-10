"""Extraction result model."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base
from app.db.types import GUID, JSONB


class ExtractResult(Base):
    """
    Extraction result from Extract processing.

    A job may have multiple results if reprocessing occurred. The final
    result is marked with is_final=True.

    Attributes:
        id: Unique result identifier
        job_id: Parent job
        attempt_number: Which attempt this was (1, 2, 3...)
        raw_response: Raw model response (for debugging)
        parsed_data: Parsed JSON data
        validation_errors: List of validation errors
        validation_warnings: List of validation warnings
        is_final: Whether this is the final result for the job
    """

    __tablename__ = "extract_results"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    job_id = Column(
        GUID,
        ForeignKey("extract_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    attempt_number = Column(Integer, default=1, nullable=False)
    raw_response = Column(Text, nullable=True)
    parsed_data = Column(JSONB, nullable=True)

    validation_errors = Column(JSONB, default=list, nullable=False)
    validation_warnings = Column(JSONB, default=list, nullable=False)

    is_final = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    job = relationship("ExtractJob", back_populates="results")

    def __repr__(self):
        return f"<ExtractResult {self.id}: attempt {self.attempt_number}>"
