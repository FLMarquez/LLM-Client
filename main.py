# // main.py

import asyncio
import os
from dotenv import load_dotenv
from schemas import ChatMessage, ModelConfig
from manager import AsyncLLMManager

load_dotenv()

async def main():
    # Elegir proveedor (se puede cambiar a "anthropic" o "gemini")
    provider = os.getenv("DEFAULT_PROVIDER", "openai")
    manager = AsyncLLMManager(provider=provider)

    messages = [
        ChatMessage(role="user", content="¿Qué es la entropía?")
    ]
    config = ModelConfig(temperature=0.7, max_tokens=300)

    print(f"=== Prueba con {provider} (modo normal) ===")
    response = await manager.generate(messages, config)
    if response.success:
        print("Respuesta:", response.text)
        if response.usage:
            print("Uso:", response.usage)
    else:
        print("Error:", response.error)

    print("\n=== Prueba con streaming ===")
    print("Streaming:")
    async for token in manager.stream_generate(messages, config):
        print(token, end="", flush=True)
    print("\n--- Fin del stream ---")

if __name__ == "__main__":
    asyncio.run(main())