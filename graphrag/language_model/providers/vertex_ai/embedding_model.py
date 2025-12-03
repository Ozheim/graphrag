# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Vertex AI Embedding Model with ADC support."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig

logger = logging.getLogger(__name__)


class VertexAIEmbeddingModel:
    """Vertex AI Embedding Model using native SDK with ADC authentication."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Vertex AI Embedding Model.

        Args:
            name: Model instance name
            config: Language model configuration
            vertex_project: GCP Project ID (overrides config)
            vertex_location: GCP Location (overrides config)
            **kwargs: Additional arguments
        """
        from graphrag.language_model.providers.vertex_ai.rest_client import (
            VertexAIRestClient,
        )

        self.name = name
        self.config = config

        # Get project and location from kwargs or config
        self.project = vertex_project or getattr(config, "vertex_project", None)
        self.location = vertex_location or getattr(config, "vertex_location", None)
        model_name = config.model or "textembedding-gecko@003"
        
        # Use REST client for proxy compatibility
        self.rest_client = VertexAIRestClient(
            project=self.project,
            location=self.location,
            model=model_name,
            api_endpoint=config.api_base,
            proxy=config.proxy,
        )
        logger.info(f"Vertex AI Embedding Model ready (REST): {model_name}")

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """
        Async batch embedding generation.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        # Vertex AI SDK doesn't have native async support for embeddings
        # Fall back to sync implementation
        return self.embed_batch(text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Async single text embedding.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding vector
        """
        return self.embed(text, **kwargs)

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Batch embedding generation using REST API.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        try:
            # Vertex AI supports batch embedding with up to 250 texts
            batch_size = 250
            all_embeddings = []

            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                embeddings = self.rest_client.get_embeddings(batch)
                all_embeddings.extend(embeddings)

            return all_embeddings
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error generating batch embeddings: {e!s}")
            raise

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Single text embedding using REST API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding vector
        """
        try:
            embeddings = self.rest_client.get_embeddings([text])
            return embeddings[0] if embeddings else []
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error generating embedding: {e!s}")
            raise

