# // anthropic_client.py
import os
from typing import Optional, AsyncGenerator
from anthropic import AsyncAnthropic
from schemas import ChatMessage, ModelConfig, ModelResponse
from base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    def _to_anthropic_messages(self, messages: list[ChatMessage]):
        system = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        return system, chat_messages

    async def generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> ModelResponse:
        # Sin try/except: ver nota en gemini_client.py
        cfg = config or ModelConfig()
        system, chat_msgs = self._to_anthropic_messages(messages)
        response = await self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=chat_msgs,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            stop_sequences=cfg.stop,
        )
        return ModelResponse(
            text=response.content[0].text,
            provider="anthropic",
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
        system, chat_msgs = self._to_anthropic_messages(messages)
        async with self.client.messages.stream(
            model=self.model_name,
            system=system,
            messages=chat_msgs,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            stop_sequences=cfg.stop,
        ) as stream:
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text