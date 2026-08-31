"""Copilot curto do RTV: JSON-mode Gemini depois do STT. Não é Live / overlay."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..config.settings import Settings

logger = logging.getLogger(__name__)

CoachTone = Literal["neutro", "alerta", "oportunidade"]

SYSTEM_INSTRUCTION = (
    "Você é um copilot de uma frase para o RTV agrícola, depois da transcrição. "
    "Não invente fazenda, preço, desconto nem compromisso. Não é coaching ao vivo. "
    "Responda só o JSON pedido."
)


class CoachNote(BaseModel):
    note: str = Field(
        description="Uma frase em PT-BR para o RTV, sem inventar dado",
        max_length=280,
    )
    tone: CoachTone = "neutro"


class CopilotCoach:
    def __init__(self, settings: Settings):
        self._key = settings.gemini_api_key
        self._model = settings.extractor_model
        self._max_tokens = settings.coach_max_output_tokens
        self._client = None
        if self._key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._key)
            except Exception as err:
                logger.warning("copilot client skip: %s", err)

    def note(self, transcript: str) -> Optional[CoachNote]:
        if not transcript.strip():
            return None
        if not self._client:
            return None
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self._model,
                contents=(
                    "Transcrição da mensagem do produtor:\n"
                    f"{transcript.strip()[:4000]}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=CoachNote,
                    max_output_tokens=self._max_tokens,
                    temperature=0.2,
                ),
            )
            parsed = response.parsed
            if isinstance(parsed, CoachNote):
                return parsed
            return CoachNote.model_validate_json(response.text)
        except Exception as err:
            logger.warning("copilot skip: %s", err)
            return None
