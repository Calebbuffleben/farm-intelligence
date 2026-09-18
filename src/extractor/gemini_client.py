"""Chamada ao Gemini em JSON mode com schema estruturado (pydantic)."""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors, types

from ..config.settings import Settings
from .prompts import SYSTEM_INSTRUCTION, build_user_prompt
from .schema import ExtractionResult
from ..stt.audio_convert import to_stt_audio

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
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def model_chain(primary: str) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for model in (primary, *FALLBACK_MODELS):
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def _uses_thinking(model: str) -> bool:
    name = model.lower()
    return "2.5" in name or name.startswith("gemini-3")


class _UnusableResponse(Exception):
    """HTTP 200 com JSON truncado ou vazio — tenta o próximo modelo."""


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
        audio: Optional[bytes] = None,
        audio_mime: Optional[str] = None,
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
            audio_target=bool(audio),
        )
        return self._generate(prompt, audio=audio, audio_mime=audio_mime)

    def _content_config(self, model: str) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": ExtractionResult,
            "max_output_tokens": self._max_tokens,
            "temperature": 0.1,
        }
        afc = getattr(types, "AutomaticFunctionCallingConfig", None)
        if afc is not None:
            kwargs["automatic_function_calling"] = afc(disable=True)
        # 2.5 Flash gasta o teto em "thinking" e corta o JSON no meio da string.
        thinking = getattr(types, "ThinkingConfig", None)
        if thinking is not None and _uses_thinking(model):
            kwargs["thinking_config"] = thinking(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _generate(
        self,
        prompt: str,
        audio: Optional[bytes] = None,
        audio_mime: Optional[str] = None,
    ) -> ExtractionResult:
        contents = self._contents(prompt, audio, audio_mime)
        models = model_chain(self._model)
        last_error: Optional[BaseException] = None
        for index, model in enumerate(models):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=self._content_config(model),
                )
                parsed = self._parse_response(response, model)
            except _UnusableResponse as exc:
                last_error = exc
                if index >= len(models) - 1:
                    raise
                logger.warning("%s — tentando %s", exc, models[index + 1])
                continue
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
                continue
            except Exception:
                logger.exception("Gemini generate_content falhou model=%s", model)
                raise
            if model != self._model:
                logger.warning(
                    "Gemini ok no fallback model=%s (primário=%s)",
                    model,
                    self._model,
                )
                self._model = model
            return parsed
        assert last_error is not None
        raise last_error

    @staticmethod
    def _contents(
        prompt: str,
        audio: Optional[bytes],
        audio_mime: Optional[str],
    ) -> Any:
        if not audio:
            return prompt
        converted, mime = to_stt_audio(audio, audio_mime or "audio/ogg")
        logger.info("extract com áudio bytes=%d mime=%s", len(converted), mime)
        prompt_part = (
            types.Part.from_text(text=prompt)
            if hasattr(types.Part, "from_text")
            else prompt
        )
        return [types.Part.from_bytes(data=converted, mime_type=mime), prompt_part]

    def _parse_response(self, response: Any, model: str) -> ExtractionResult:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ExtractionResult):
            return parsed
        text = _json_text(getattr(response, "text", None) or "")
        finish = _finish_reason(response)
        if not text:
            logger.error(
                "Gemini devolveu vazio model=%s finish=%s",
                model,
                finish,
            )
            raise _UnusableResponse(f"resposta vazia model={model} finish={finish}")
        try:
            return ExtractionResult.model_validate_json(text)
        except Exception as exc:
            logger.warning(
                "Gemini JSON inválido model=%s chars=%d finish=%s",
                model,
                len(text),
                finish,
            )
            raise _UnusableResponse(
                f"JSON inválido model={model} chars={len(text)} finish={finish}"
            ) from exc

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)


def _json_text(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def _finish_reason(response: Any) -> Optional[str]:
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        return str(reason) if reason is not None else None
    except Exception:
        return None
