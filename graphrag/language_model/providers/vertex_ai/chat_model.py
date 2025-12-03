# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Vertex AI Chat Model with ADC support."""

import json
import logging
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.response.base import ModelResponse

logger = logging.getLogger(__name__)


class VertexAIChatModel:
    """Vertex AI Chat Model using native SDK with ADC authentication."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Vertex AI Chat Model.

        Args:
            name: Model instance name
            config: Language model configuration
            vertex_project: GCP Project ID (overrides config)
            vertex_location: GCP Location (overrides config)
            **kwargs: Additional arguments
        """
        logger.error("=" * 80)
        logger.error("VERTEX AI CHAT MODEL INITIALIZATION STARTING")
        logger.error(f"Name: {name}")
        logger.error(f"Model: {config.model}")
        logger.error(f"Project: {vertex_project}")
        logger.error(f"Location: {vertex_location}")
        logger.error("=" * 80)
        
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            logger.error("✅ Successfully imported vertexai modules")
        except ImportError as e:
            msg = f"❌ CRITICAL: google-cloud-aiplatform package is required for Vertex AI models. Install it with: pip install google-cloud-aiplatform. Error: {e}"
            logger.error(msg)
            raise ImportError(msg) from e

        self.name = name
        self.config = config

        # Get project and location from kwargs or config
        # Priority: kwargs > config attributes > environment
        self.project = vertex_project or getattr(config, "vertex_project", None)
        self.location = vertex_location or getattr(config, "vertex_location", None)
        
        # Get api_base from config if provided (for custom endpoints)
        self.api_endpoint = config.api_base if config.api_base else None
        
        # Configure proxy if provided in config
        if config.proxy:
            import os
            logger.error(f"⚙️ Configuring proxy: {config.proxy}")
            os.environ["HTTPS_PROXY"] = config.proxy
            os.environ["HTTP_PROXY"] = config.proxy
            # Don't proxy localhost/internal services
            os.environ["NO_PROXY"] = "127.0.0.1,localhost"
            
            # CRITICAL: Force REST instead of gRPC when using proxy
            # gRPC/HTTP2 doesn't work well with most corporate proxies
            os.environ["GOOGLE_API_USE_REST_CLIENT"] = "true"
            logger.error("⚙️ Forcing REST client (not gRPC) for proxy compatibility")

        try:
            # Initialize Vertex AI with ADC
            # If project/location are None, vertexai.init() will use environment defaults
            # If api_endpoint is provided, use it to override the default endpoint
            logger.error(
                f"Attempting vertexai.init() with project={self.project}, location={self.location}, api_endpoint={self.api_endpoint}"
            )
            vertexai.init(
                project=self.project, 
                location=self.location,
                api_endpoint=self.api_endpoint
            )
            logger.error("✅ vertexai.init() succeeded")

            # Extract model name from config
            model_name = config.model or "gemini-pro"
            logger.error(f"Loading model: {model_name}")
            self.model = GenerativeModel(model_name)
            logger.error(f"✅ Vertex AI Chat Model initialized successfully: {model_name}")
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ VERTEX AI INITIALIZATION FAILED")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e!s}")
            logger.error(f"Project: {self.project}")
            logger.error(f"Location: {self.location}")
            logger.error(f"Model: {config.model}")
            logger.error("=" * 80)
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> "ModelResponse":
        """
        Async chat completion.

        Args:
            prompt: The prompt text
            history: Optional conversation history
            **kwargs: Additional model parameters

        Returns:
            ModelResponse object
        """
        # Import here to avoid circular imports
        from graphrag.language_model.response.base import (
            BaseModelOutput,
            BaseModelResponse,
        )

        # Build chat history if provided
        chat = self.model.start_chat(history=[])

        # Generate content
        generation_config = self._build_generation_config(kwargs)
        response = await chat.send_message_async(
            prompt, generation_config=generation_config
        )

        content = response.text

        # Handle JSON parsing if requested
        parsed_response = None
        if kwargs.get("json"):
            try:
                parsed_dict = json.loads(content)
                if "json_model" in kwargs and isinstance(kwargs["json_model"], type):
                    if issubclass(kwargs["json_model"], BaseModel):
                        parsed_response = kwargs["json_model"](**parsed_dict)
                else:
                    parsed_response = parsed_dict  # type: ignore
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from Vertex AI")

        return BaseModelResponse(
            output=BaseModelOutput(content=content),
            parsed_response=parsed_response,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat completion.

        Args:
            prompt: The prompt text
            history: Optional conversation history
            **kwargs: Additional model parameters

        Yields:
            Response text chunks
        """
        chat = self.model.start_chat(history=[])
        generation_config = self._build_generation_config(kwargs)

        response = await chat.send_message_async(
            prompt, generation_config=generation_config, stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> "ModelResponse":
        """
        Synchronous chat completion.

        Args:
            prompt: The prompt text
            history: Optional conversation history
            **kwargs: Additional model parameters

        Returns:
            ModelResponse object
        """
        from graphrag.language_model.response.base import (
            BaseModelOutput,
            BaseModelResponse,
        )

        chat = self.model.start_chat(history=[])
        generation_config = self._build_generation_config(kwargs)

        response = chat.send_message(prompt, generation_config=generation_config)
        content = response.text

        # Handle JSON parsing if requested
        parsed_response = None
        if kwargs.get("json"):
            try:
                parsed_dict = json.loads(content)
                if "json_model" in kwargs and isinstance(kwargs["json_model"], type):
                    if issubclass(kwargs["json_model"], BaseModel):
                        parsed_response = kwargs["json_model"](**parsed_dict)
                else:
                    parsed_response = parsed_dict  # type: ignore
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from Vertex AI")

        return BaseModelResponse(
            output=BaseModelOutput(content=content),
            parsed_response=parsed_response,
        )

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """
        Synchronous streaming chat completion.

        Args:
            prompt: The prompt text
            history: Optional conversation history
            **kwargs: Additional model parameters

        Yields:
            Response text chunks
        """
        chat = self.model.start_chat(history=[])
        generation_config = self._build_generation_config(kwargs)

        response = chat.send_message(
            prompt, generation_config=generation_config, stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _build_generation_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Build generation config from kwargs and model config.

        Args:
            kwargs: Keyword arguments from chat call

        Returns:
            Generation config dictionary
        """
        config = {}

        # Map common parameters
        if self.config.temperature is not None:
            config["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            config["max_output_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            config["top_p"] = self.config.top_p

        # Override with kwargs if provided
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]

        # Handle JSON mode
        if kwargs.get("json"):
            config["response_mime_type"] = "application/json"

        return config

