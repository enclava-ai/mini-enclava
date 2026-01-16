"""
Internal LLM API endpoints - for frontend use with JWT authentication
"""

import logging
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.core.security import get_current_user
from app.services.llm.service import llm_service
from app.services.llm.models import ChatRequest, ChatMessage as LLMChatMessage
from app.services.llm.exceptions import (
    LLMError,
    ProviderError,
    SecurityError,
    ValidationError,
)
from app.api.v1.llm import get_cached_models  # Reuse the caching logic
from app.services.usage_recording import UsageRecordingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List available LLM models for authenticated users
    """
    try:
        models = await get_cached_models()
        return {"data": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models",
        )


@router.get("/providers/status")
async def get_provider_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get status of all LLM providers for authenticated users
    """
    try:
        provider_status = await llm_service.get_provider_status()
        return {
            "object": "provider_status",
            "data": {
                name: {
                    "provider": status.provider,
                    "status": status.status,
                    "latency_ms": status.latency_ms,
                    "success_rate": status.success_rate,
                    "last_check": status.last_check.isoformat(),
                    "error_message": status.error_message,
                    "models_available": status.models_available,
                }
                for name, status in provider_status.items()
            },
        }
    except Exception as e:
        logger.error(f"Failed to get provider status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provider status",
        )


@router.get("/health")
async def health_check(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get LLM service health status for authenticated users
    """
    try:
        health = await llm_service.health_check()
        return {
            "status": health["status"],
            "providers": health.get("providers", {}),
            "timestamp": health.get("timestamp"),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed",
        )


@router.get("/metrics")
async def get_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get LLM service metrics for authenticated users
    """
    try:
        metrics = await llm_service.get_metrics()
        return {"object": "metrics", "data": metrics}
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics",
        )


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions"""

    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create chat completion for authenticated frontend users.
    This endpoint is for playground and internal use only, using JWT authentication.
    """
    start_time = time.time()
    request_id = uuid4()
    user_id = current_user.get("id", current_user.get("sub"))
    usage_service = UsageRecordingService(db)

    try:
        # Convert request to LLM service format
        # For internal use, we use a special api_key_id of 0 to indicate JWT auth (Playground)
        chat_request = ChatRequest(
            model=request.model,
            messages=[
                LLMChatMessage(role=msg["role"], content=msg["content"])
                for msg in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream,
            user_id=str(user_id),
            api_key_id=None,  # None = Playground/internal (JWT auth)
        )

        # Log the request for debugging
        logger.info(
            f"Internal chat completion request from user {user_id}: model={request.model}"
        )

        # Process the request through the LLM service
        response = await llm_service.create_chat_completion(chat_request)

        # Extract usage data
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        latency_ms = int((time.time() - start_time) * 1000)

        # Record usage to usage_records table (source of truth for billing)
        await usage_service.record_request(
            request_id=request_id,
            user_id=user_id,
            api_key_id=None,  # None with no chatbot_id = Playground (JWT auth)
            provider_id=getattr(response, "provider", "privatemode"),
            provider_model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            endpoint="/api-internal/v1/llm/chat/completions",
            method="POST",
            is_streaming=request.stream or False,
            message_count=len(request.messages),
            latency_ms=latency_ms,
            status="success",
        )
        await db.commit()

        # Format the response to match OpenAI's structure
        formatted_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        return formatted_response

    except ValidationError as e:
        logger.error(f"Validation error in chat completion: {e}")
        # Record error
        latency_ms = int((time.time() - start_time) * 1000)
        await usage_service.record_error(
            user_id=user_id,
            api_key_id=None,  # None = Playground/internal (JWT auth)
            provider_id="privatemode",
            model=request.model,
            endpoint="/api-internal/v1/llm/chat/completions",
            error=e,
            latency_ms=latency_ms,
            request_id=request_id,
            message_count=len(request.messages),
            is_streaming=request.stream or False,
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except LLMError as e:
        logger.error(f"LLM service error: {e}")
        # Record error
        latency_ms = int((time.time() - start_time) * 1000)
        await usage_service.record_error(
            user_id=user_id,
            api_key_id=None,  # None = Playground/internal (JWT auth)
            provider_id="privatemode",
            model=request.model,
            endpoint="/api-internal/v1/llm/chat/completions",
            error=e,
            latency_ms=latency_ms,
            request_id=request_id,
            message_count=len(request.messages),
            is_streaming=request.stream or False,
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        # Record error
        latency_ms = int((time.time() - start_time) * 1000)
        try:
            await usage_service.record_error(
                user_id=user_id,
                api_key_id=None,  # None = Playground/internal (JWT auth)
                provider_id="privatemode",
                model=request.model,
                endpoint="/api-internal/v1/llm/chat/completions",
                error=e,
                latency_ms=latency_ms,
                request_id=request_id,
                message_count=len(request.messages),
                is_streaming=request.stream or False,
            )
            await db.commit()
        except Exception as record_error:
            logger.error(f"Failed to record usage error: {record_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion",
        )
