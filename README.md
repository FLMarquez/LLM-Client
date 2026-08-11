# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI**, **Anthropic** y **Gemini** (Google).  
Este proyecto implementa una interfaz común que permite cambiar de proveedor de modelos de lenguaje sin modificar la lógica de negocio. Todas las llamadas son asíncronas (`async/await`) y se soporta streaming de tokens.

---

## Requisitos

- **Python 3.12** (o superior)
- **pip** (gestor de paquetes)
- Claves API de al menos uno de los proveedores soportados:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Anthropic](https://console.anthropic.com/account/keys)
  - [Google Gemini](https://aistudio.google.com/apikey)

---

## Instalación

1. **Clona el repositorio** (o descarga los archivos).
2. **Crea y activa un entorno virtual** (recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate        # En Linux/Mac
   # o en Windows:
   venv\Scripts\activate

   pip install -r requirements.txt


<!-- Es necesario crear .env con lo siguiente

# OpenAI
OPENAI_API_KEY=tu api key

# Anthropic
ANTHROPIC_API_KEY=tu api key

# Google Gemini
GEMINI_API_KEY=tu api key

# Selección del proveedor por defecto: openai, anthropic, gemini
DEFAULT_PROVIDER=gemini -->