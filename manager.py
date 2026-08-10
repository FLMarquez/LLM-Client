# // manager.py
import asyncio
import os
import random
from typing import Optional

import httpx

from schemas import ChatMessage, ModelConfig, ModelResponse
from base import BaseLLMClient
from openai_client import OpenAIClient
from anthropic_client import AnthropicClient
from gemini_client import GeminiClient


class AsyncLLMManager:
    PROVIDERS = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ):
        self.provider_name = provider or os.getenv("DEFAULT_PROVIDER", "openai")
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Proveedor no soportado: {self.provider_name}")
        self.client: BaseLLMClient = self.PROVIDERS[self.provider_name]()
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _status_code(self, exc: Exception) -> Optional[int]:
        # httpx (Gemini): exc.response.status_code
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        # SDKs de OpenAI/Anthropic: sus excepciones (APIStatusError y
        # subclases como RateLimitError) exponen .status_code directo.
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        return None

    def _is_rate_limited(self, exc: Exception) -> bool:
        return self._status_code(exc) == 429

    def _retry_after_seconds(self, exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    return None
        return None

    async def generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await self.client.generate(messages, config)
            except Exception as exc:
                if not self._is_rate_limited(exc) or attempt >= self.max_retries:
                    return ModelResponse(
                        text="",
                        provider=self.provider_name,
                        model="unknown",
                        error=str(exc),
                        success=False,
                    )
                delay = self._retry_after_seconds(exc)
                if delay is None:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"[AsyncLLMManager] 429 en {self.provider_name}, "
                    f"reintentando en {delay:.1f}s (intento {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ):
        attempt = 0
        while True:
            try:
                async for token in self.client.stream_generate(messages, config):
                    yield token
                return
            except Exception as exc:
                if not self._is_rate_limited(exc) or attempt >= self.max_retries:
                    yield f"[Error: {exc}]"
                    return
                delay = self._retry_after_seconds(exc)
                if delay is None:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"[AsyncLLMManager] 429 en streaming ({self.provider_name}), "
                    f"reintentando en {delay:.1f}s (intento {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1