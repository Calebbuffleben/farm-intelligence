"""Chamada ao Gemini em JSON mode com schema estruturado (pydantic)."""

import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from ..config.settings import Settings
from .prompts import SYSTEM_INSTRUCTION, build_user_prompt
from .schema import ExtractionResult

logger = logging.getLogger(__name__)


class FactExtractor:
    def __init__(self, settings: Settings):
        self._client = None
        self._model = settings.extractor_model
        self._max_tokens = settings.extractor_max_output_tokens
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY ausente — extrator em fail-open")
            return
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def extract(
        self,
        carteira: Dict[str, Any],
        previous_summaries: List[str],
        messages: List[Dict[str, Any]],
        target_index: Optional[int] = None,
        human_links: Optional[List[Dict[str, Any]]] = None,
        session_retrieve: Optional[List[Dict[str, Any]]] = None,
        sales_policy: Optional[Dict[str, Any]] = None,
        previous_brief: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        if not self._client:
            return ExtractionResult(session_summary="", facts=[], unknowns=[])
        prompt = build_user_prompt(
            carteira,
            previous_summaries,
            messages,
            target_index=target_index,
            human_links=human_links,
            session_retrieve=session_retrieve,
            sales_policy=sales_policy,
            previous_brief=previous_brief,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                    max_output_tokens=self._max_tokens,
                    temperature=0.1,
                ),
            )
        except Exception:
            logger.exception("Gemini generate_content falhou model=%s", self._model)
            raise
        parsed = response.parsed
        if isinstance(parsed, ExtractionResult):
            return parsed
        text = getattr(response, "text", None) or ""
        if not text.strip():
            logger.error("Gemini devolveu vazio model=%s", self._model)
            return ExtractionResult(session_summary="", facts=[], unknowns=[])
        try:
            return ExtractionResult.model_validate_json(text)
        except Exception:
            logger.exception("Gemini JSON inválido model=%s chars=%d", self._model, len(text))
            raise
