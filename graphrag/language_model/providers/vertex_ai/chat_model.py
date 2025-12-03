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
        print(f"!!! VERTEX AI CHAT MODEL __INIT__ CALLED !!! name={name}, config.type={getattr(config, 'type', 'NO_TYPE')}")
        logger.error(f"!!! VERTEX AI CHAT MODEL __INIT__ CALLED !!! name={name}")
        
        print("!!! Step 1: Importing VertexAIRestClient...")
        from graphrag.language_model.providers.vertex_ai.rest_client import (
            VertexAIRestClient,
        )
        print("!!! Step 1: OK")

        print("!!! Step 2: Setting name and config...")
        self.name = name
        self.config = config
        print("!!! Step 2: OK")

        # Get project and location from kwargs or config
        print("!!! Step 3: Getting project and location...")
        self.project = vertex_project or getattr(config, "vertex_project", None)
        self.location = vertex_location or getattr(config, "vertex_location", None)
        model_name = config.model or "gemini-pro"
        print(f"!!! Step 3: OK - project={self.project}, location={self.location}, model={model_name}")
        
        # Use REST client for proxy compatibility
        print("!!! Step 4: Creating REST client...")
        try:
            self.rest_client = VertexAIRestClient(
                project=self.project,
                location=self.location,
                model=model_name,
                api_endpoint=config.api_base,
                proxy=config.proxy,
            )
            print(f"!!! Step 4: REST client created successfully")
            logger.info(f"Vertex AI Chat Model ready (REST): {model_name}")
        except Exception as e:
            print(f"!!! Step 4: FAILED - {type(e).__name__}: {e!s}")
            logger.error(f"Vertex AI init failed: {type(e).__name__}: {e!s}")
            raise
        
        print("!!! __INIT__ COMPLETED SUCCESSFULLY")

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> "ModelResponse":
        """
        Async chat completion using REST API.

        Args:
            prompt: The prompt text
            history: Optional conversation history
            **kwargs: Additional model parameters

        Returns:
            ModelResponse object
        """
        print(f"!!! ACHAT CALLED !!! prompt length: {len(prompt)}")
        
        from graphrag.language_model.response.base import (
            BaseModelOutput,
            BaseModelResponse,
        )

        logger.info(f"[CHAT_MODEL] achat called with prompt length: {len(prompt)}")
        # Use sync method (REST API is sync anyway)
        result = self.chat(prompt, history, **kwargs)
        logger.info(f"[CHAT_MODEL] Response content length: {len(result.output.content)}")
        return result

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
        # REST API doesn't support streaming, return full response
        response = self.chat(prompt, history, **kwargs)
        yield response.output.content

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> "ModelResponse":
        """
        Synchronous chat completion using REST API.

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

        logger.warning("[CHAT_MODEL] Starting chat request")
        
        # Call REST API
        config_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        config_kwargs.update(kwargs)
        
        logger.warning(f"[CHAT_MODEL] Calling REST client with config: {config_kwargs}")
        
        try:
            response = self.rest_client.generate_content(prompt, **config_kwargs)
            logger.warning(f"[CHAT_MODEL] REST response received: {str(response)[:200]}")
        except Exception as e:
            logger.error(f"[CHAT_MODEL] REST client failed: {type(e).__name__}: {e!s}")
            raise
        
        # Extract text from response
        content = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        logger.warning(f"[CHAT_MODEL] Extracted content length: {len(content)}")

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
        # REST API doesn't support streaming, return full response
        response = self.chat(prompt, history, **kwargs)
        yield response.output.content

