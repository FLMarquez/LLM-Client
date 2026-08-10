# //  openai_client.py
import os
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from schemas import ChatMessage, ModelConfig, ModelResponse
from base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no está configurada")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> ModelResponse:
        # Sin try/except: ver nota en gemini_client.py
        cfg = config or ModelConfig()
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            stop=cfg.stop,
        )
        return ModelResponse(
            text=response.choices[0].message.content,
            provider="openai",
            model=response.model,
            usage=response.usage.model_dump() if response.usage else None,
            success=True,
        )

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or ModelConfig()
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            stop=cfg.stop,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content