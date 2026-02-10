"""Jinja2 Template Configuration for Web Frontend"""

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from jinja2 import Environment

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Create Jinja2Templates instance
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def setup_template_globals(env: Environment) -> None:
    """Add global functions and variables to Jinja2 environment."""
    # Add any global functions here
    env.globals["len"] = len
    env.globals["enumerate"] = enumerate
    env.globals["range"] = range
    env.globals["str"] = str
    env.globals["int"] = int
    env.globals["float"] = float
    env.globals["bool"] = bool
    env.globals["dict"] = dict
    env.globals["list"] = list


def setup_template_filters(env: Environment) -> None:
    """Add custom filters to Jinja2 environment."""

    def format_currency(value: float, currency: str = "$") -> str:
        """Format a number as currency."""
        if value is None:
            return f"{currency}0.00"
        return f"{currency}{value:,.2f}"

    def format_number(value: float | int, decimals: int = 0) -> str:
        """Format a number with thousands separator."""
        if value is None:
            return "0"
        if decimals:
            return f"{value:,.{decimals}f}"
        return f"{value:,}"

    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format a number as percentage."""
        if value is None:
            return "0%"
        return f"{value:.{decimals}f}%"

    def truncate(text: str, length: int = 50, suffix: str = "...") -> str:
        """Truncate text to specified length."""
        if not text:
            return ""
        if len(text) <= length:
            return text
        return text[:length].rsplit(" ", 1)[0] + suffix

    def time_ago(dt) -> str:
        """Format datetime as relative time (e.g., '2 hours ago')."""
        from datetime import datetime, timezone

        if dt is None:
            return "Never"

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        diff = now - dt

        seconds = diff.total_seconds()
        if seconds < 60:
            return "Just now"
        minutes = seconds / 60
        if minutes < 60:
            return f"{int(minutes)} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes / 60
        if hours < 24:
            return f"{int(hours)} hour{'s' if hours != 1 else ''} ago"
        days = hours / 24
        if days < 30:
            return f"{int(days)} day{'s' if days != 1 else ''} ago"
        months = days / 30
        if months < 12:
            return f"{int(months)} month{'s' if months != 1 else ''} ago"
        years = months / 12
        return f"{int(years)} year{'s' if years != 1 else ''} ago"

    def mask_key(key: str, visible_chars: int = 8) -> str:
        """Mask an API key showing only first/last chars."""
        if not key:
            return ""
        if len(key) <= visible_chars * 2:
            return key
        return f"{key[:visible_chars]}...{key[-visible_chars:]}"

    env.filters["currency"] = format_currency
    env.filters["number"] = format_number
    env.filters["percentage"] = format_percentage
    env.filters["truncate"] = truncate
    env.filters["time_ago"] = time_ago
    env.filters["mask_key"] = mask_key


# Initialize template environment
setup_template_globals(templates.env)
setup_template_filters(templates.env)


def render_template(
    request: Request, template_name: str, context: dict[str, Any] | None = None
):
    """
    Render a Jinja2 template with common context.

    Args:
        request: The FastAPI request object
        template_name: Name of the template file
        context: Additional context variables

    Returns:
        TemplateResponse
    """
    ctx = {
        "request": request,
        **(context or {}),
    }
    return templates.TemplateResponse(template_name, ctx)
