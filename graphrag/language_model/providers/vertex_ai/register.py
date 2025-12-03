# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Register Vertex AI providers with the ModelFactory."""

import logging

from graphrag.language_model.factory import ModelFactory
from graphrag.language_model.providers.vertex_ai.chat_model import VertexAIChatModel
from graphrag.language_model.providers.vertex_ai.embedding_model import (
    VertexAIEmbeddingModel,
)

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_vertex_ai_providers() -> None:
    """
    Register Vertex AI Chat and Embedding models with the ModelFactory.

    This function should be called early in the application lifecycle,
    before configuration validation, to ensure the custom providers
    are available when GraphRAG loads the settings.yaml.

    The providers are registered with the following type names:
    - vertex-chat-adc: Chat model with ADC authentication
    - vertex-embedding-adc: Embedding model with ADC authentication
    """
    global _REGISTERED
    if _REGISTERED:
        return

    print("!!! REGISTERING VERTEX AI PROVIDERS !!!")
    logger.error("!!! REGISTERING VERTEX AI PROVIDERS !!!")
    logger.info("Registering Vertex AI providers with ModelFactory")

    # Register Chat Model
    ModelFactory.register_chat(
        "vertex-chat-adc", lambda **kwargs: VertexAIChatModel(**kwargs)
    )

    # Register Embedding Model
    ModelFactory.register_embedding(
        "vertex-embedding-adc", lambda **kwargs: VertexAIEmbeddingModel(**kwargs)
    )

    _REGISTERED = True
    logger.info(
        "Vertex AI providers registered: vertex-chat-adc, vertex-embedding-adc"
    )

