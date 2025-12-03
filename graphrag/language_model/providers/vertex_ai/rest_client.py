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
        self.project = project
        self.location = location
        self.model = model
        self.proxy = proxy
        
        # Build API endpoint
        if api_endpoint:
            self.base_url = api_endpoint
        else:
            self.base_url = f"https://{location}-aiplatform.googleapis.com"
        
        # Get credentials for authentication
        self.credentials, _ = default()
        
        # Configure session with proxy
        self.session = requests.Session()
        if self.proxy:
            self.session.proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            logger.info(f"REST client configured with proxy: {self.proxy}")

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
            logger.info(f"REST API URL: {url}")
            
            logger.info("Getting access token...")
            token = self._get_access_token()
            logger.info(f"Token obtained: {token[:20]}...")
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            
            # Build request body
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt[:100]}]}],
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
            
            logger.info(f"Making POST request (proxy: {self.proxy})...")
            logger.info(f"Request body: {json.dumps(body, indent=2)[:500]}")
            
            # Make request
            response = self.session.post(url, headers=headers, json=body, timeout=180)
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text[:500]}")
            
            response.raise_for_status()
            result = response.json()
            logger.info("Response received successfully")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed: {type(e).__name__}: {e!s}")
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error in generate_content: {type(e).__name__}: {e!s}")
            raise

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings using Vertex AI REST API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:predict"
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        
        body = {"instances": [{"content": text} for text in texts]}
        
        response = self.session.post(url, headers=headers, json=body, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        return [pred["embeddings"]["values"] for pred in result.get("predictions", [])]

