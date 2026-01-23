"""Custom exceptions for Extract."""


class ExtractError(Exception):
    """Base exception for Extract."""

    pass


class InvalidFileError(ExtractError):
    """Raised when uploaded file is invalid."""

    pass


class FileTooLargeError(ExtractError):
    """Raised when uploaded file exceeds size limit."""

    pass


class ProcessingError(ExtractError):
    """Raised when document processing fails."""

    pass


class BudgetExceededError(ExtractError):
    """Raised when budget check fails."""

    pass


class TemplateNotFoundError(ExtractError):
    """Raised when template does not exist."""

    pass


class TemplateExistsError(ExtractError):
    """Raised when trying to create duplicate template."""

    pass
