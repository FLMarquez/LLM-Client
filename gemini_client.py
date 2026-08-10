import os
import httpx
import json
from typing import Optional, AsyncGenerator
from schemas import ChatMessage, ModelConfig, ModelResponse
from base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada")
        # Pineado a 2.5-flash a propósito: soporta apagar el thinking del
        # todo con thinkingBudget: 0. El alias "gemini-flash-latest" hoy
        # resuelve a la serie 3.x, que usa thinkingLevel en vez de
        # thinkingBudget y NO permite apagar el thinking completamente
        # (siempre gasta algo de maxOutputTokens pensando).
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent"
        self.client = httpx.AsyncClient(timeout=60.0)

    def _build_payload(self, messages: list[ChatMessage], config: ModelConfig, stream: bool = False):
        contents = []
        for msg in messages:
            role = "user" if msg.role in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
                "topP": config.top_p,
                "stopSequences": config.stop or [],
            }
        }
        # thinkingBudget: 0 apaga el razonamiento interno en modelos 2.5,
        # dejando todo el maxOutputTokens para la respuesta final.
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        return payload

    @staticmethod
    def _extract_text(candidate: dict) -> str:
        # Concatena todas las partes de texto que NO sean 'thought'
        # (por si algún modelo no respeta thinkingBudget: 0)
        parts = candidate.get("content", {}).get("parts", [])
        return "".join(
            p.get("text", "") for p in parts if p.get("text") and not p.get("thought")
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> ModelResponse:
        # Sin try/except: las excepciones (incluido 429) suben hasta
        # AsyncLLMManager, que decide si reintenta o arma el error final.
        cfg = config or ModelConfig()
        payload = self._build_payload(messages, cfg, stream=False)
        url = f"{self.base_url}?key={self.api_key}"
        response = await self.client.post(url, json=payload)
        if response.status_code >= 400:
            print(f"[GeminiClient] Respuesta de error ({response.status_code}): {response.text}")
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [{}])
        text = self._extract_text(candidates[0]) if candidates else ""
        return ModelResponse(
            text=text,
            provider="gemini",
            model=self.model_name,
            usage=None,
            success=True,
        )

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or ModelConfig()
        payload = self._build_payload(messages, cfg, stream=True)
        url = f"{self.stream_url}?key={self.api_key}&alt=sse"
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if part.get("text") and not part.get("thought"):
                                    yield part["text"]
                    except json.JSONDecodeError:
                        continue