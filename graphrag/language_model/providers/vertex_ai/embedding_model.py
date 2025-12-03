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
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError as e:
            msg = "google-cloud-aiplatform package is required for Vertex AI models. Install it with: pip install google-cloud-aiplatform"
            raise ImportError(msg) from e

        self.name = name
        self.config = config

        # Get project and location from kwargs or config
        self.project = vertex_project or getattr(config, "vertex_project", None)
        self.location = vertex_location or getattr(config, "vertex_location", None)
        
        # Get api_base from config if provided (for custom endpoints)
        self.api_endpoint = config.api_base if config.api_base else None
        
        # Configure proxy if provided in config
        if config.proxy:
            import os
            logger.info(f"⚙️ Configuring proxy: {config.proxy}")
            os.environ["HTTPS_PROXY"] = config.proxy
            os.environ["HTTP_PROXY"] = config.proxy
            # Don't proxy localhost/internal services
            os.environ["NO_PROXY"] = "127.0.0.1,localhost"
            
            # CRITICAL: Force REST instead of gRPC when using proxy
            # gRPC/HTTP2 doesn't work well with most corporate proxies
            os.environ["GOOGLE_API_USE_REST_CLIENT"] = "true"
            logger.info("⚙️ Forcing REST client (not gRPC) for proxy compatibility")

        # Initialize Vertex AI with ADC
        logger.info(
            f"Initializing Vertex AI Embeddings for {name} with project={self.project}, location={self.location}, api_endpoint={self.api_endpoint}"
        )
        vertexai.init(
            project=self.project, 
            location=self.location,
            api_endpoint=self.api_endpoint
        )

        # Get model name from config
        model_name = config.model or "textembedding-gecko@003"
        self.model = TextEmbeddingModel.from_pretrained(model_name)

        logger.info(f"Vertex AI Embedding Model initialized: {model_name}")

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
        Batch embedding generation.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        try:
            # Vertex AI supports batch embedding with up to 250 texts
            # Split into batches if needed
            batch_size = 250
            all_embeddings = []

            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                embeddings = self.model.get_embeddings(batch)
                all_embeddings.extend([emb.values for emb in embeddings])

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Single text embedding.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding vector
        """
        try:
            embeddings = self.model.get_embeddings([text])
            return embeddings[0].values if embeddings else []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

