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
        # print(f"!!! VertexAIRestClient __init__: project={project}, location={location}, model={model}")
        
        self.project = project
        self.location = location
        self.model = model
        self.proxy = proxy
        
        # Build API endpoint
        if api_endpoint:
            self.base_url = api_endpoint
        else:
            self.base_url = f"https://{location}-aiplatform.googleapis.com"
        
        # print(f"!!! VertexAIRestClient: base_url={self.base_url}")
        
        # Get credentials for authentication
        # print("!!! VertexAIRestClient: Calling google.auth.default()...")
        try:
            self.credentials, _ = default()
            # print(f"!!! VertexAIRestClient: Credentials obtained successfully")
        except Exception as e:
            # print(f"!!! VertexAIRestClient: google.auth.default() FAILED: {type(e).__name__}: {e!s}")
            logger.error(f"ADC authentication failed: {type(e).__name__}: {e!s}")
            raise
        
        # Configure session with proxy
        # print("!!! VertexAIRestClient: Configuring session with proxy...")
        self.session = requests.Session()
        if self.proxy:
            self.session.proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            logger.info(f"REST client configured with proxy: {self.proxy}")
        
        # print("!!! VertexAIRestClient: Initialization complete")

    def _get_access_token(self) -> str:
        """Get fresh access token."""
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token

    def generate_content(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """
        Generate content using Vertex AI REST API with FULL DIAGNOSTICS.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response dict from Vertex AI
        """
        try:
            url = f"{self.base_url}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
            
            # We don't log the URL every time to reduce noise, unless needed
            # logger.info(f"[REST_CLIENT] REST API URL: {url}")
            
            token = self._get_access_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            
            # ---------------------------------------------------------
            # 1. BUILD REQUEST BODY WITH SAFETY SETTINGS
            # ---------------------------------------------------------
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {},
                # CRITICAL: Disable Safety Filters to prevent empty responses on industrial text
                "safetySettings": [
                    { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
                    { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
                    { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
                    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" }
                ]
            }
            
            # Map GraphRAG parameters to Vertex AI
            if "temperature" in kwargs:
                body["generationConfig"]["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                body["generationConfig"]["maxOutputTokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs:
                body["generationConfig"]["topP"] = kwargs["top_p"]
            
            # Handle JSON mode if requested by GraphRAG settings
            if kwargs.get("json"):
                body["generationConfig"]["responseMimeType"] = "application/json"
            
            # Log the request structure to confirm safetySettings are sent
            # print(f"!!! REST_CLIENT: Making POST request. Prompt len: {len(prompt)}")
            # print(f"!!! REST_CLIENT: Body keys: {list(body.keys())}")
            
            # ---------------------------------------------------------
            # 2. SEND REQUEST
            # ---------------------------------------------------------
            # Timeout increased to 180s for large chunks
            response = self.session.post(url, headers=headers, json=body, timeout=180)
            
            # print(f"!!! REST_CLIENT: Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[REST_CLIENT] HTTP Error {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
            
            result = response.json()

            # ---------------------------------------------------------
            # 3. EXPLICIT DIAGNOSTIC LOGGING (commented out for clean logs)
            # This block prints the exact reason why Google stopped generating
            # ---------------------------------------------------------
            # print("\n" + "="*50)
            # print("!!! GOOGLE API RESPONSE DIAGNOSTIC !!!")
            
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                # 3a. Check Finish Reason (STOP, SAFETY, RECITATION, etc.)
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                # print(f"!!! FINISH REASON: {finish_reason}")
                
                # 3b. Check Safety Ratings (did we get close to a block?)
                # if "safetyRatings" in candidate:
                #     print("!!! SAFETY SCORES:")
                #     for rating in candidate["safetyRatings"]:
                #         category = rating.get("category", "UNKNOWN").replace("HARM_CATEGORY_", "")
                #         probability = rating.get("probability", "UNKNOWN")
                #         blocked = rating.get("blocked", False)
                #         print(f"   - {category}: {probability} (Blocked: {blocked})")

                # 3c. Check Content Presence
                # if "content" in candidate and "parts" in candidate["content"]:
                #     print("!!! CONTENT STATUS: Content received successfully.")
                # else:
                #     print("!!! CONTENT STATUS: [EMPTY] - The model returned no text!")

                # 3d. Urgent Alerts - Only log critical failures
                if finish_reason == "SAFETY":
                    logger.error("CRITICAL: Request blocked by Safety Filters")
                elif finish_reason == "RECITATION":
                    logger.error("CRITICAL: Request blocked by Copyright/Recitation")
                elif finish_reason == "MAX_TOKENS":
                    logger.warning("Response truncated (max_tokens reached)")

            else:
                logger.error(f"No 'candidates' in response: {json.dumps(result, indent=2)}")
                
            # print("="*50 + "\n")
            # ---------------------------------------------------------
            
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
        # Reduce log noise for embeddings
        # logger.info(f"[REST_CLIENT] get_embeddings called with {len(texts)} texts")
        
        url = f"{self.base_url}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:predict"
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        
        body = {"instances": [{"content": text} for text in texts]}
        
        # logger.info("[REST_CLIENT] Sending embedding request...")
        response = self.session.post(url, headers=headers, json=body, timeout=180)
        
        if response.status_code != 200:
            logger.error(f"[REST_CLIENT] Embedding Error {response.status_code}: {response.text[:200]}")
        
        response.raise_for_status()
        
        result = response.json()
        embeddings = [pred["embeddings"]["values"] for pred in result.get("predictions", [])]
        # logger.info(f"[REST_CLIENT] Received {len(embeddings)} embeddings")
        return embeddings