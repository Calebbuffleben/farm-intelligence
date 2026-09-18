"""STT batch via Gemini (áudio inline → transcrição em PT-BR).

Áudio de WhatsApp é OGG/Opus; a Evolution converte PTT para audio/mp4.
Sem Files API no ano 1 — envio inline.
"""

import logging
from typing import Optional

from google import genai
from google.genai import errors, types

from ..config.settings import Settings
from ..extractor.gemini_client import _uses_thinking, model_chain
from .audio_convert import audio_kind, to_stt_audio

logger = logging.getLogger(__name__)

_PROMPT = (
    "Transcreva este áudio de WhatsApp em português do Brasil. "
    "É uma conversa comercial do setor agrícola (nomes de fazendas, culturas, "
    "insumos e defensivos são esperados). Retorne SOMENTE o texto transcrito, "
    "sem comentários e sem marcações."
)

_MIME_ALIASES = {
    "audio/ogg": "audio/ogg",
    "audio/opus": "audio/ogg",
    "application/ogg": "audio/ogg",
    "audio/webm": "audio/webm",
    "audio/mp4": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
    "audio/mpeg": "audio/mp3",
    "audio/mp3": "audio/mp3",
    "audio/wav": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/aac": "audio/aac",
    "application/octet-stream": "audio/ogg",
}


def stt_mime(mime_type: Optional[str]) -> str:
    raw = (mime_type or "audio/ogg").split(";")[0].strip().lower()
    if raw in _MIME_ALIASES:
        return _MIME_ALIASES[raw]
    return raw if raw.startswith("audio/") else "audio/ogg"


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
        if not audio:
            raise RuntimeError("STT sem bytes de áudio")
        audio, mime = to_stt_audio(audio, stt_mime(mime_type))
        logger.info(
            "STT request bytes=%d mime=%s kind=%s",
            len(audio),
            mime,
            audio_kind(audio) or "unknown",
        )
        prompt = (
            types.Part.from_text(text=_PROMPT)
            if hasattr(types.Part, "from_text")
            else _PROMPT
        )
        last_error: Optional[BaseException] = None
        for model in model_chain(self._model):
            try:
                kwargs = {"temperature": 0.0}
                thinking = getattr(types, "ThinkingConfig", None)
                if thinking is not None and _uses_thinking(model):
                    kwargs["thinking_config"] = thinking(thinking_budget=0)
                response = self._client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=audio, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(**kwargs),
                )
                text = (response.text or "").strip()
                if text:
                    if model != self._model:
                        logger.warning("STT ok no fallback model=%s", model)
                        self._model = model
                    return text
                last_error = RuntimeError(f"STT vazio model={model}")
                logger.warning("STT vazio model=%s mime=%s — próximo modelo", model, mime)
            except (errors.ClientError, errors.ServerError) as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code not in {404, 429, 503}:
                    logger.exception("STT falhou model=%s", model)
                    raise
                logger.warning("STT %s model=%s — próximo modelo", code, model)
            except Exception:
                logger.exception("STT falhou model=%s", model)
                raise
        raise last_error or RuntimeError("STT retornou transcrição vazia")
