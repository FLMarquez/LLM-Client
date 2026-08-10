# // base.py

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from schemas import ChatMessage, ModelConfig, ModelResponse

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> ModelResponse:
        """Genera una respuesta completa (no streaming)."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Genera un stream de tokens (generador asíncrono)."""
        pass