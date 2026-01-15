import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

# Importations du SDK Google
import vertexai
from google.cloud.aiplatform import types
from vertexai.language_models import TextEmbeddingModel as VertexEmbeddingClient
from vertexai.generative_models import GenerativeModel as VertexChatClient
from vertexai.generative_models import Content, Part, FinishReason

# Importations des protocoles GraphRAG
from graphrag.language_model.protocol import (
    ChatModel,
    ChatResult,
    ChatApiError,
    TextEmbeddingModel,
    TextEmbeddingService,
)

log = logging.getLogger(__name__)

class VertexAdcChatModel(ChatModel):
    """Implémentation du ChatModel pour Vertex AI (Gemini) utilisant ADC."""

    def __init__(self, vertex_project: str, vertex_location: str, **kwargs):
        self.project = vertex_project
        self.location = vertex_location
        self.model_name = kwargs.get("model", "gemini-2.5-flash")

        # Initialisation du client Vertex AI (ADC est géré ici)
        vertexai.init(project=self.project, location=self.location)
        self.client = VertexChatClient(model_name=self.model_name)
        log.info(f"VertexAdcChatModel initialisé pour project={self.project}, location={self.location}")

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Génère une complétion de chat en utilisant Vertex AI."""
        
        # Le SDK Vertex AI attend une liste de Content objects
        # On traduit le format GraphRAG (list[Dict[str, str]]) vers le format Vertex AI
        contents: List[Content] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model" # Vertex utilise 'model' au lieu de 'assistant'
            parts = [Part.from_text(msg["content"])]
            contents.append(Content(role=role, parts=parts))

        # Extraction des paramètres pour l'appel (timeout n'est pas utilisé directement ici)
        generation_config = types.GenerateContentConfig(
            max_output_tokens=model_parameters.get("max_tokens"),
            temperature=model_parameters.get("temperature"),
            top_p=model_parameters.get("top_p"),
        )
        
        try:
            # L'appel à l'API Vertex AI
            response = await asyncio.to_thread(
                self.client.generate_content,
                contents=contents,
                config=generation_config,
            )

            if not response.candidates:
                raise ChatApiError("Vertex AI returned an empty response.")
            
            # Formater la réponse dans le format GraphRAG
            text_response = response.candidates[0].content.parts[0].text
            
            return ChatResult(
                output=text_response,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                },
            )

        except Exception as e:
            # En cas d'erreur réseau, d'API ou d'authentification ADC
            log.error(f"Vertex AI Chat API Error: {e}")
            raise ChatApiError(f"Vertex AI API call failed: {e}")

class VertexAdcEmbeddingModel(TextEmbeddingModel):
    """Implémentation du TextEmbeddingModel pour Vertex AI."""

    def __init__(self, vertex_project: str, vertex_location: str, **kwargs):
        self.project = vertex_project
        self.location = vertex_location
        self.model_name = kwargs.get("model", "text-embedding-004")
        
        vertexai.init(project=self.project, location=self.location)
        self.client = VertexEmbeddingClient.from_pretrained(self.model_name)
        log.info(f"VertexAdcEmbeddingModel initialisé pour project={self.project}")

    async def generate_embeddings(self, texts: List[str]) -> TextEmbeddingService:
        """Génère des embeddings en utilisant Vertex AI."""
        try:
            # Vertex AI supporte le batching, mais nous l'appelons dans un thread pour l'asynchronisme
            embeddings = await asyncio.to_thread(
                self.client.embed_text,
                texts=texts,
            )
            
            # Formater le résultat dans le format GraphRAG
            vectors = [list(e.values) for e in embeddings]

            return TextEmbeddingService(vectors=vectors)

        except Exception as e:
            log.error(f"Vertex AI Embedding API Error: {e}")
            raise ChatApiError(f"Vertex AI Embedding API call failed: {e}")