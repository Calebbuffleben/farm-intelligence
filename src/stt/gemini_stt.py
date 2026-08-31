"""STT batch via Gemini (áudio inline → transcrição em PT-BR).

Áudio de WhatsApp é OGG/Opus, curto (limite prático de 16MB da Cloud API),
então o envio inline resolve — sem Files API no ano 1.
"""

import logging

from google import genai
from google.genai import types

from ..config.settings import Settings

logger = logging.getLogger(__name__)

_PROMPT = (
    "Transcreva este áudio de WhatsApp em português do Brasil. "
    "É uma conversa comercial do setor agrícola (nomes de fazendas, culturas, "
    "insumos e defensivos são esperados). Retorne SOMENTE o texto transcrito, "
    "sem comentários e sem marcações."
)


class GeminiStt:
    def __init__(self, settings: Settings):
        self._client = None
        self._model = settings.extractor_model
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY ausente — STT em fail-open")
            return
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        if not self._client:
            logger.warning("STT skip — sem GEMINI_API_KEY")
            return ""
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=audio, mime_type=mime_type or "audio/ogg"),
                _PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("STT retornou transcrição vazia")
        return text
