# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Vertex AI model providers with ADC support."""

from graphrag.language_model.providers.vertex_ai.chat_model import VertexAIChatModel
from graphrag.language_model.providers.vertex_ai.embedding_model import (
    VertexAIEmbeddingModel,
)

__all__ = ["VertexAIChatModel", "VertexAIEmbeddingModel"]

