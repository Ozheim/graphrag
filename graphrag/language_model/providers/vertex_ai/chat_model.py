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
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:
            msg = "google-cloud-aiplatform package is required for Vertex AI models. Install it with: pip install google-cloud-aiplatform"
            raise ImportError(msg) from e

        self.name = name
        self.config = config

        # Get project and location from kwargs or config
        # Priority: kwargs > config attributes > environment
        self.project = vertex_project or getattr(config, "vertex_project", None)
        self.location = vertex_location or getattr(config, "vertex_location", None)

        # Initialize Vertex AI with ADC
        # If project/location are None, vertexai.init() will use environment defaults
        logger.info(
            f"Initializing Vertex AI for model {name} with project={self.project}, location={self.location}"
        )
        vertexai.init(project=self.project, location=self.location)

        # Extract model name from config
        model_name = config.model or "gemini-pro"
        self.model = GenerativeModel(model_name)

        logger.info(f"Vertex AI Chat Model initialized: {model_name}")

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

