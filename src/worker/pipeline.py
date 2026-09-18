"""Esteira de análise de UMA mensagem (Fase 3).

  contexto (backend) -> extração-delta (texto ou áudio no Gemini) -> análise

Áudio entra no mesmo extrator do texto: os bytes vão na chamada, sem depender
de object storage nem de uma transcrição isolada. Sem bytes do canal, não ACK.
"""

import logging
from typing import Any, Dict, List, Optional

from ..backend.client import BackendClient
from ..config.settings import Settings
from ..extractor.coach import CopilotCoach
from ..extractor.gemini_client import FactExtractor
from ..extractor.schema import ExtractionResult
from .payload import deal_quality_issues, is_unusable_brief, to_analysis_payload

logger = logging.getLogger(__name__)


class MessagePipeline:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend = BackendClient(settings)
        self._extractor = FactExtractor(settings)
        self._coach = CopilotCoach(settings)

    def process(self, message_id: str) -> None:
        ctx = self._backend.get_context(message_id)
        message = ctx["message"]

        if message["direction"] != "IN":
            logger.debug("mensagem OUT ignorada: %s", message_id)
            return

        if ctx.get("analysisAllowed") is False:
            logger.info("análise bloqueada por consentimento: %s", message_id)
            return

        audio: Optional[bytes] = None
        audio_mime: Optional[str] = None
        transcript: Optional[str] = message.get("transcript")

        if message["type"] == "AUDIO":
            audio, fetched_mime = self._backend.get_message_media(message_id)
            asset = message.get("mediaAsset") or {}
            audio_mime = asset.get("contentType") or fetched_mime
            if not audio:
                raise RuntimeError(f"áudio sem bytes message={message_id}")
            logger.info(
                "áudio na extração message=%s bytes=%d mime=%s",
                message_id,
                len(audio),
                audio_mime,
            )
        else:
            text = transcript or message.get("body") or ""
            if not text.strip():
                logger.info("mensagem sem texto analisável: %s", message_id)
                return

        messages, target_index = self._build_session_messages(
            ctx, message_id, transcript
        )
        carteira = self._build_carteira(ctx)
        summaries = [
            s["summary"] for s in ctx.get("previousSummaries", []) if s.get("summary")
        ]
        previous_brief = ctx.get("previousBrief") or None
        if is_unusable_brief(previous_brief):
            previous_brief = None

        extract_kw: Dict[str, Any] = {
            "target_index": target_index,
            "human_links": ctx.get("humanLinks") or [],
            "session_retrieve": ctx.get("previousSummaries") or [],
            "sales_policy": ctx.get("salesPolicy") or None,
            "previous_brief": previous_brief,
            "audio": audio,
            "audio_mime": audio_mime,
        }

        try:
            result = self._extractor.extract(
                carteira, summaries, messages, **extract_kw
            )
            quality_issues = deal_quality_issues(
                result.deal, ctx.get("salesPolicy") or None
            )
            if quality_issues:
                logger.warning(
                    "deal rejeitado message=%s issues=%s — tentando reparação",
                    message_id,
                    ", ".join(quality_issues),
                )
                result = self._extractor.extract(
                    carteira,
                    summaries,
                    messages,
                    **extract_kw,
                    repair_feedback=quality_issues,
                )
        except Exception:
            if message["type"] == "AUDIO":
                # Áudio não pode virar Card vazio. Sem ACK o Redis reentrega.
                raise
            logger.exception(
                "extração falhou message=%s — grava Card de Bordo mínimo",
                message_id,
            )
            result = ExtractionResult(session_summary="", facts=[], unknowns=[])

        if message["type"] == "AUDIO" and not _audio_was_analyzed(result):
            raise RuntimeError(
                f"Gemini não analisou o áudio message={message_id}"
            )

        if (result.transcript or "").strip():
            transcript = result.transcript.strip()

        coach_note: Optional[str] = None
        coach_tone: Optional[str] = None
        if message["type"] == "AUDIO" and transcript:
            coach = self._coach.note(transcript)
            if coach:
                coach_note = coach.note
                coach_tone = coach.tone

        source_text = (
            transcript or message.get("body") or result.session_summary or ""
        )
        payload = to_analysis_payload(
            ctx,
            result,
            transcript,
            self._settings.resolver_min_confidence,
            coach_note=coach_note,
            coach_tone=coach_tone,
            source_text=source_text,
            source_sent_at=str(message.get("sentAt") or ""),
        )
        self._backend.post_analysis(message_id, payload)
        logger.info(
            "análise publicada message=%s facts=%d unknowns=%d audio=%s transcript=%s",
            message_id,
            len(payload["facts"]),
            len(payload["unknowns"]),
            bool(audio),
            bool(transcript),
        )

    def _build_session_messages(
        self,
        ctx: Dict[str, Any],
        message_id: str,
        fresh_transcript: Optional[str],
    ) -> tuple[List[Dict[str, Any]], int]:
        messages: List[Dict[str, Any]] = []
        target_index = 0
        for i, m in enumerate(ctx.get("sessionMessages", [])):
            text = _display_text(m)
            if m["id"] == message_id:
                target_index = i
                if fresh_transcript:
                    text = fresh_transcript
            messages.append(
                {
                    "index": i,
                    "direction": m["direction"],
                    "text": text,
                    "ts": m.get("sentAt", ""),
                }
            )
        return messages, target_index

    @staticmethod
    def _build_carteira(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "producer": ctx.get("producer"),
            "farms": [
                {
                    "id": farm["id"],
                    "name": farm["name"],
                    "region": farm.get("region"),
                    "areaHa": farm.get("areaHa"),
                    "crops": [
                        {"crop": cs["crop"], "season": cs["seasonLabel"]}
                        for cs in farm.get("cropSeasons", [])
                    ],
                    "openFacts": farm.get("openFacts") or [],
                    "lastFactAt": farm.get("lastFactAt"),
                }
                for farm in ctx.get("farms", [])
            ],
        }


def _display_text(m: Dict[str, Any]) -> str:
    """Texto que o Gemini lê: transcrição, body, ou legenda da mídia."""
    spoken = (m.get("transcript") or "").strip()
    if spoken:
        return spoken
    body = (m.get("body") or "").strip()
    kind = (m.get("type") or "").upper()
    if kind == "IMAGE":
        return f"[imagem] {body}" if body else "[imagem]"
    if kind == "DOCUMENT":
        return f"[documento] {body}" if body else "[documento]"
    if kind == "AUDIO":
        return "[áudio]"
    if kind and kind not in {"TEXT", ""}:
        return f"[{kind.lower()}] {body}" if body else f"[{kind.lower()}]"
    return body or "[texto]"


def _audio_was_analyzed(result: ExtractionResult) -> bool:
    """Card de Bordo de voz só vale se o Gemini leu o recado."""
    if result.facts:
        return True
    if (result.transcript or "").strip() and result.deal:
        summary = (result.deal.context_summary or "").strip()
        if summary:
            return True
    return not deal_quality_issues(result.deal)
