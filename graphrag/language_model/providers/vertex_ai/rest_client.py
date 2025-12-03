# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Vertex AI REST client for proxy compatibility."""

import json
import logging
from typing import Any

import requests
from google.auth import default
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)


class VertexAIRestClient:
    """Vertex AI REST client that works with corporate proxies."""

    def __init__(
        self,
        project: str,
        location: str,
        model: str,
        api_endpoint: str | None = None,
        proxy: str | None = None,
    ):
        """Initialize Vertex AI REST client."""
        print(f"!!! VertexAIRestClient __init__: project={project}, location={location}, model={model}")
        
        self.project = project
        self.location = location
        self.model = model
        self.proxy = proxy
        
        # Build API endpoint
        if api_endpoint:
            self.base_url = api_endpoint
        else:
            self.base_url = f"https://{location}-aiplatform.googleapis.com"
        
        print(f"!!! VertexAIRestClient: base_url={self.base_url}")
        
        # Get credentials for authentication
        print("!!! VertexAIRestClient: Calling google.auth.default()...")
        try:
            self.credentials, _ = default()
            print(f"!!! VertexAIRestClient: Credentials obtained successfully")
        except Exception as e:
            print(f"!!! VertexAIRestClient: google.auth.default() FAILED: {type(e).__name__}: {e!s}")
            raise
        
        # Configure session with proxy
        print("!!! VertexAIRestClient: Configuring session with proxy...")
        self.session = requests.Session()
        if self.proxy:
            self.session.proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            logger.info(f"REST client configured with proxy: {self.proxy}")
        
        print("!!! VertexAIRestClient: Initialization complete")

    def _get_access_token(self) -> str:
        """Get fresh access token."""
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token

    def generate_content(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """
        Generate content using Vertex AI REST API.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response dict from Vertex AI
        """
        try:
            url = f"{self.base_url}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
            logger.info(f"[REST_CLIENT] REST API URL: {url}")
            
            logger.info("[REST_CLIENT] Getting access token...")
            token = self._get_access_token()
            logger.info(f"[REST_CLIENT] Token obtained: {token[:20]}...")
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            
            # Build request body
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {},
            }
            
            # Map GraphRAG parameters to Vertex AI
            if "temperature" in kwargs:
                body["generationConfig"]["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                body["generationConfig"]["maxOutputTokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs:
                body["generationConfig"]["topP"] = kwargs["top_p"]
            
            # Handle JSON mode
            if kwargs.get("json"):
                body["generationConfig"]["responseMimeType"] = "application/json"
            
            print(f"!!! REST_CLIENT: Making POST request (proxy: {self.proxy})")
            print(f"!!! REST_CLIENT: Prompt length: {len(prompt)} chars")
            print(f"!!! REST_CLIENT: Body structure: {list(body.keys())}")
            
            # Make request
            response = self.session.post(url, headers=headers, json=body, timeout=180)
            
            print(f"!!! REST_CLIENT: Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[REST_CLIENT] HTTP Error {response.status_code}: {response.text[:500]}")
            
            response.raise_for_status()
            result = response.json()
            logger.info(f"[REST_CLIENT] Response JSON structure: {list(result.keys())}")
            if "candidates" in result:
                logger.info(f"[REST_CLIENT] Number of candidates: {len(result['candidates'])}")
            logger.info("[REST_CLIENT] Response received successfully")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[REST_CLIENT] HTTP Request failed: {type(e).__name__}: {e!s}")
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"[REST_CLIENT] Unexpected error in generate_content: {type(e).__name__}: {e!s}")
            raise

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings using Vertex AI REST API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"[REST_CLIENT] get_embeddings called with {len(texts)} texts")
        
        url = f"{self.base_url}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:predict"
        logger.info(f"[REST_CLIENT] Embedding URL: {url}")
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        
        body = {"instances": [{"content": text} for text in texts]}
        
        logger.info("[REST_CLIENT] Sending embedding request...")
        response = self.session.post(url, headers=headers, json=body, timeout=180)
        logger.info(f"[REST_CLIENT] Embedding response status: {response.status_code}")
        
        response.raise_for_status()
        
        result = response.json()
        embeddings = [pred["embeddings"]["values"] for pred in result.get("predictions", [])]
        logger.info(f"[REST_CLIENT] Received {len(embeddings)} embeddings")
        return embeddings

