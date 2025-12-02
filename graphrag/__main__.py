# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The GraphRAG package."""

# Register custom Vertex AI providers before CLI runs
# This allows them to bypass Pydantic validation for ADC authentication
from graphrag.language_model.providers.vertex_ai.register import (
    register_vertex_ai_providers,
)

register_vertex_ai_providers()

from graphrag.cli.main import app

app(prog_name="graphrag")
