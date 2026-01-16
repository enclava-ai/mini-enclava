"""
LLM Service

Main service that coordinates providers, security, resilience, and metrics.
Replaces LiteLLM client functionality with direct provider integration.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime

from .models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ProviderStatus,
    LLMMetrics,
)
from .config import config_manager, ProviderConfig
from ...core.config import settings

from .resilience import ResilienceManagerFactory

# from .metrics import metrics_collector
from .providers import BaseLLMProvider, PrivateModeProvider
from .exceptions import (
    LLMError,
    ProviderError,
    SecurityError,
    ConfigurationError,
    ValidationError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


class LLMService:
    """Main LLM service coordinating all components"""

    def __init__(self):
        """Initialize LLM service"""
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._initialized = False
        self._startup_time: Optional[datetime] = None

        logger.info("LLM Service initialized")

    async def initialize(self):
        """Initialize service and providers"""
        if self._initialized:
            logger.warning("LLM Service already initialized")
            return

        start_time = time.time()
        self._startup_time = datetime.utcnow()

        try:
            # Get configuration
            config = config_manager.get_config()
            logger.info(
                f"Initializing LLM service with {len(config.providers)} configured providers"
            )

            # Initialize enabled providers
            enabled_providers = config_manager.get_enabled_providers()
            if not enabled_providers:
                raise ConfigurationError("No enabled providers found")

            for provider_name in enabled_providers:
                await self._initialize_provider(provider_name)

            # Verify we have at least one working provider
            if not self._providers:
                raise ConfigurationError("No providers successfully initialized")

            # Verify default provider is available
            default_provider = config.default_provider
            if default_provider not in self._providers:
                available_providers = list(self._providers.keys())
                logger.warning(
                    f"Default provider '{default_provider}' not available, using '{available_providers[0]}'"
                )
                config.default_provider = available_providers[0]

            # Initialize attestation monitoring
            await self._initialize_attestation()

            self._initialized = True
            initialization_time = (time.time() - start_time) * 1000

            logger.info(
                f"LLM Service initialized successfully in {initialization_time:.2f}ms"
            )
            logger.info(f"Available providers: {list(self._providers.keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise ConfigurationError(f"LLM service initialization failed: {e}")

    async def _initialize_provider(self, provider_name: str):
        """Initialize a specific provider"""
        try:
            provider_config = config_manager.get_provider_config(provider_name)
            if not provider_config or not provider_config.enabled:
                logger.warning(f"Provider '{provider_name}' not enabled, skipping")
                return

            # Get API key
            api_key = config_manager.get_api_key(provider_name)
            if not api_key:
                logger.error(f"No API key found for provider '{provider_name}'")
                return

            # Create provider instance
            provider = self._create_provider(provider_config, api_key)

            # Initialize provider
            await provider.initialize()

            # Test provider health
            health_status = await provider.health_check()
            if health_status.status == "unavailable":
                logger.error(
                    f"Provider '{provider_name}' failed health check: {health_status.error_message}"
                )
                return

            # Register provider
            self._providers[provider_name] = provider
            logger.info(
                f"Provider '{provider_name}' initialized successfully (status: {health_status.status})"
            )

            # Fetch and update models dynamically
            await self._refresh_provider_models(provider_name, provider)

        except Exception as e:
            logger.error(f"Failed to initialize provider '{provider_name}': {e}")

    def _create_provider(self, config: ProviderConfig, api_key: str) -> BaseLLMProvider:
        """Create provider instance based on configuration"""
        if config.name == "privatemode":
            return PrivateModeProvider(config, api_key)
        elif config.name == "redpill":
            from .providers.redpill import RedPillProvider
            return RedPillProvider(config, api_key)
        else:
            raise ConfigurationError(f"Unknown provider type: {config.name}")

    async def _refresh_provider_models(
        self, provider_name: str, provider: BaseLLMProvider
    ):
        """Fetch and update models dynamically from provider"""
        try:
            # Get models from provider
            models = await provider.get_models()
            model_ids = [model.id for model in models]

            # Update configuration
            await config_manager.refresh_provider_models(provider_name, model_ids)

            logger.info(
                f"Refreshed {len(model_ids)} models for provider '{provider_name}': {model_ids}"
            )

        except Exception as e:
            logger.error(
                f"Failed to refresh models for provider '{provider_name}': {e}"
            )

    async def _initialize_attestation(self):
        """Initialize attestation monitoring for all providers."""
        # Import here to avoid circular dependency
        from .attestation.scheduler import attestation_scheduler
        from .attestation.privatemode import PrivateModeAttestationVerifier
        from .attestation.redpill import RedPillAttestationVerifier

        if "privatemode" in self._providers:
            attestation_scheduler.register_provider(
                "privatemode",
                PrivateModeAttestationVerifier(
                    proxy_url=settings.PRIVATEMODE_PROXY_URL,
                    api_key=settings.PRIVATEMODE_API_KEY
                ),
                test_model=None  # Proxy health check only
            )
            logger.info("Registered PrivateMode provider for attestation monitoring")

        if "redpill" in self._providers:
            redpill_api_key = getattr(settings, "REDPILL_API_KEY", None)
            redpill_base_url = getattr(settings, "REDPILL_BASE_URL", "https://api.redpill.ai/v1")
            redpill_test_model = getattr(settings, "REDPILL_TEST_MODEL", "phala/deepseek-chat-v3-0324")

            if redpill_api_key:
                attestation_scheduler.register_provider(
                    "redpill",
                    RedPillAttestationVerifier(
                        api_base=redpill_base_url,
                        api_key=redpill_api_key
                    ),
                    test_model=redpill_test_model
                )
                logger.info("Registered RedPill provider for attestation monitoring")
            else:
                logger.warning("RedPill provider enabled but no API key found, skipping attestation registration")

        # Start periodic verification
        await attestation_scheduler.start()
        logger.info("Started attestation scheduler")

        # Run initial verification for all registered providers
        for provider_id in attestation_scheduler._verifiers:
            try:
                await attestation_scheduler.verify_now(provider_id)
                logger.info(f"Completed initial attestation verification for {provider_id}")
            except Exception as e:
                logger.error(f"Initial attestation verification failed for {provider_id}: {e}")

    async def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Create chat completion with security and resilience"""
        if not self._initialized:
            await self.initialize()

        # Validate request
        if not request.messages:
            raise ValidationError("Messages cannot be empty", field="messages")

        risk_score = 0.0

        # Get provider for model
        provider_name = self._get_provider_for_model(request.model)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ProviderError(
                f"No available provider for model '{request.model}'",
                provider=provider_name,
            )

        # Execute with resilience
        resilience_manager = ResilienceManagerFactory.get_manager(provider_name)
        start_time = time.time()

        try:
            response = await resilience_manager.execute(
                provider.create_chat_completion,
                request,
                retryable_exceptions=(ProviderError, TimeoutError),
                non_retryable_exceptions=(ValidationError,),
            )

            # Record successful request - metrics disabled
            total_latency = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            # Record failed request - metrics disabled
            total_latency = (time.time() - start_time) * 1000
            error_code = getattr(e, "error_code", e.__class__.__name__)

            logger.exception(
                "Chat completion failed for provider %s (model=%s, latency=%.2fms, error=%s)",
                provider_name,
                request.model,
                total_latency,
                error_code,
            )
            raise

    async def create_chat_completion_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming chat completion"""
        if not self._initialized:
            await self.initialize()

        # Security validation disabled - always allow streaming requests
        risk_score = 0.0

        # Get provider
        provider_name = self._get_provider_for_model(request.model)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ProviderError(
                f"No available provider for model '{request.model}'",
                provider=provider_name,
            )

        # Execute streaming with resilience
        resilience_manager = ResilienceManagerFactory.get_manager(provider_name)

        try:
            async for chunk in await resilience_manager.execute(
                provider.create_chat_completion_stream,
                request,
                retryable_exceptions=(ProviderError, TimeoutError),
                non_retryable_exceptions=(ValidationError,),
            ):
                yield chunk

        except Exception as e:
            # Record streaming failure - metrics disabled
            error_code = getattr(e, "error_code", e.__class__.__name__)
            logger.exception(
                "Streaming chat completion failed for provider %s (model=%s, error=%s)",
                provider_name,
                request.model,
                error_code,
            )
            raise

    async def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings with security and resilience"""
        if not self._initialized:
            await self.initialize()

        # Security validation disabled - always allow embedding requests
        risk_score = 0.0

        # Get provider
        provider_name = self._get_provider_for_model(request.model)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ProviderError(
                f"No available provider for model '{request.model}'",
                provider=provider_name,
            )

        # Execute with resilience
        resilience_manager = ResilienceManagerFactory.get_manager(provider_name)
        start_time = time.time()

        try:
            response = await resilience_manager.execute(
                provider.create_embedding,
                request,
                retryable_exceptions=(ProviderError, TimeoutError),
                non_retryable_exceptions=(ValidationError,),
            )

            # Record successful request - metrics disabled
            total_latency = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            # Record failed request - metrics disabled
            total_latency = (time.time() - start_time) * 1000
            error_code = getattr(e, "error_code", e.__class__.__name__)
            logger.exception(
                "Embedding request failed for provider %s (model=%s, latency=%.2fms, error=%s)",
                provider_name,
                request.model,
                total_latency,
                error_code,
            )
            raise

    async def get_models(self, provider_name: Optional[str] = None) -> List[ModelInfo]:
        """Get available models from all or specific provider"""
        if not self._initialized:
            await self.initialize()

        models = []

        if provider_name:
            # Get models from specific provider
            provider = self._providers.get(provider_name)
            if provider:
                try:
                    provider_models = await provider.get_models()
                    models.extend(provider_models)
                except Exception as e:
                    logger.error(f"Failed to get models from {provider_name}: {e}")
        else:
            # Get models from all providers
            for name, provider in self._providers.items():
                try:
                    provider_models = await provider.get_models()
                    models.extend(provider_models)
                except Exception as e:
                    logger.error(f"Failed to get models from {name}: {e}")

        return models

    async def get_provider_status(self) -> Dict[str, ProviderStatus]:
        """Get health status of all providers"""
        if not self._initialized:
            await self.initialize()

        status_dict = {}

        for name, provider in self._providers.items():
            try:
                status = await provider.health_check()
                status_dict[name] = status
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                status_dict[name] = ProviderStatus(
                    provider=name,
                    status="unavailable",
                    last_check=datetime.utcnow(),
                    error_message=str(e),
                    models_available=[],
                )

        return status_dict

    def get_metrics(self) -> LLMMetrics:
        """Get service metrics - metrics disabled"""
        # return metrics_collector.get_metrics()
        return LLMMetrics(
            total_requests=0, success_rate=0.0, avg_latency_ms=0, error_rates={}
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary - metrics disabled"""
        # metrics_health = metrics_collector.get_health_summary()
        resilience_health = ResilienceManagerFactory.get_all_health_status()

        return {
            "service_status": "healthy" if self._initialized else "initializing",
            "startup_time": self._startup_time.isoformat()
            if self._startup_time
            else None,
            "provider_count": len(self._providers),
            "active_providers": list(self._providers.keys()),
            "metrics": {"status": "disabled"},
            "resilience": resilience_health,
        }

    def get_provider_for_model(self, model: str) -> str:
        """
        Get provider name for a model (public API).

        Args:
            model: Model name

        Returns:
            Provider name string
        """
        return self._get_provider_for_model(model)

    def _get_provider_for_model(self, model: str) -> str:
        """Get provider name for a model with health checks"""
        # Import here to avoid circular dependency
        from .attestation.scheduler import attestation_scheduler

        # Check model routing first
        provider_name = config_manager.get_provider_for_model(model)
        if provider_name and provider_name in self._providers:
            # Verify provider is healthy
            if attestation_scheduler.is_healthy(provider_name):
                return provider_name

        # Fall back to healthy providers that support the model
        for name, provider in self._providers.items():
            if provider.supports_model(model) and attestation_scheduler.is_healthy(name):
                return name

        # Use default provider as last resort (if healthy)
        config = config_manager.get_config()
        if config.default_provider in self._providers:
            if attestation_scheduler.is_healthy(config.default_provider):
                return config.default_provider

        # If no healthy providers, try any provider (degraded mode)
        logger.warning(f"No healthy providers available for model '{model}', trying degraded mode")
        for name, provider in self._providers.items():
            if provider.supports_model(model):
                return name

        # If nothing else works, use first available provider
        if self._providers:
            return list(self._providers.keys())[0]

        raise ProviderError(f"No provider found for model '{model}'", provider="none")

    async def get_providers_health(self) -> List[Dict[str, Any]]:
        """Get health status of all providers with attestation details."""
        # Import here to avoid circular dependency
        from .attestation.scheduler import attestation_scheduler

        result = []
        for provider_id, provider in self._providers.items():
            health = attestation_scheduler.get_health(provider_id)

            provider_health = {
                "provider_id": provider_id,
                "display_name": getattr(provider, 'display_name', provider_id.capitalize()),
                "healthy": health.healthy if health else False,
                "last_check_at": health.last_check.timestamp.isoformat() if health and health.last_check else None,
                "last_healthy_at": health.last_healthy_at.isoformat() if health and health.last_healthy_at else None,
                "error": health.error if health else None,
                "attestation_details": self._get_attestation_details(health) if health else None,
            }

            result.append(provider_health)

        return result

    def _get_attestation_details(self, health) -> Optional[Dict[str, Any]]:
        """Extract attestation details from last check."""
        if not health or not health.last_check:
            return None

        check = health.last_check
        return {
            "intel_tdx_verified": check.intel_tdx_verified,
            "gpu_attestation_verified": check.gpu_attestation_verified,
            "nonce_binding_verified": check.nonce_binding_verified,
            "signing_address": check.signing_address,
        }

    async def cleanup(self):
        """Cleanup service resources"""
        logger.info("Cleaning up LLM service")

        # Stop attestation scheduler
        try:
            from .attestation.scheduler import attestation_scheduler
            await attestation_scheduler.stop()
            logger.info("Stopped attestation scheduler")
        except Exception as e:
            logger.error(f"Error stopping attestation scheduler: {e}")

        # Cleanup providers
        for name, provider in self._providers.items():
            try:
                await provider.cleanup()
                logger.debug(f"Cleaned up provider: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up provider {name}: {e}")

        self._providers.clear()
        self._initialized = False
        logger.info("LLM service cleanup completed")


# Global LLM service instance
llm_service = LLMService()
