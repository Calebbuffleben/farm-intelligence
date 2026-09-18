"""Chamada ao Gemini em JSON mode com schema estruturado (pydantic)."""

import logging
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors, types

from ..config.settings import Settings
from .prompts import SYSTEM_INSTRUCTION, build_user_prompt
from .schema import ExtractionResult

logger = logging.getLogger(__name__)
# gemini-flash-latest aponta para o preview da vez (3.8) — mesmo pool 503.
# 2.5 / 2.0 são estáveis e costumam ter cota quando o preview satura.
FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)
FALLBACK_MODEL = FALLBACK_MODELS[0]
# 404 = modelo inexistente; 429/503 = cota ou sobrecarga temporária.
_RETRYABLE_CODES = {404, 429, 503}
_BACKOFF_S = (0.0, 1.5, 3.0, 5.0)


def model_chain(primary: str) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for model in (primary, *FALLBACK_MODELS):
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


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
        repair_feedback: Optional[List[str]] = None,
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
            repair_feedback=repair_feedback,
        )
        config = self._content_config()
        response = self._generate(prompt, config)
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

    def _content_config(self) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": ExtractionResult,
            "max_output_tokens": self._max_tokens,
            "temperature": 0.1,
        }
        # JSON mode + schema pydantic não deve virar loop de function calling.
        afc = getattr(types, "AutomaticFunctionCallingConfig", None)
        if afc is not None:
            kwargs["automatic_function_calling"] = afc(disable=True)
        return types.GenerateContentConfig(**kwargs)

    def _generate(self, prompt: str, config: types.GenerateContentConfig):
        models = model_chain(self._model)
        last_error: Optional[BaseException] = None
        for index, model in enumerate(models):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                if model != self._model:
                    logger.warning(
                        "Gemini ok no fallback model=%s (primário=%s)",
                        model,
                        self._model,
                    )
                    self._model = model
                return response
            except (errors.ClientError, errors.ServerError) as exc:
                last_error = exc
                retryable = getattr(exc, "code", None) in _RETRYABLE_CODES
                has_next = index < len(models) - 1
                if not retryable or not has_next:
                    logger.exception("Gemini generate_content falhou model=%s", model)
                    raise
                wait = _BACKOFF_S[min(index + 1, len(_BACKOFF_S) - 1)]
                logger.warning(
                    "Gemini %s model=%s — tentando %s em %.1fs",
                    getattr(exc, "code", "?"),
                    model,
                    models[index + 1],
                    wait,
                )
                if wait > 0:
                    self._sleep(wait)
            except Exception:
                logger.exception("Gemini generate_content falhou model=%s", model)
                raise
        assert last_error is not None
        raise last_error

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)
