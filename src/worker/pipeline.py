"""Esteira de análise de UMA mensagem (Fase 3).

  contexto (backend) -> STT se áudio -> copilot curto (fail-open) -> extração-delta -> análise

Regras aplicadas aqui (espelham a Fase 0):
- fato com farm_id abaixo de resolver_min_confidence NUNCA vira fato com
  fazenda: o vínculo é rebaixado para a fila unknown;
- extração incremental: só a mensagem alvo gera fatos; a sessão inteira e os
  resumos anteriores entram como contexto (resposta à "conversa infinita").
"""

import logging
from typing import Any, Dict, List, Optional

from ..backend.client import BackendClient
from ..config.settings import Settings
from ..extractor.coach import CopilotCoach
from ..extractor.gemini_client import FactExtractor
from ..extractor.schema import ExtractionResult
from ..stt.gemini_stt import GeminiStt
from .payload import deal_quality_issues, is_unusable_brief, to_analysis_payload

logger = logging.getLogger(__name__)


class MessagePipeline:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend = BackendClient(settings)
        self._extractor = FactExtractor(settings)
        self._stt = GeminiStt(settings)
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

        transcript: Optional[str] = message.get("transcript")
        if message["type"] == "AUDIO" and not transcript:
            # Bytes vêm do backend: S3 se existir, senão download ao vivo
            # no canal. Sem ACK se a busca falhar — Redis reentrega.
            audio, content_type = self._backend.get_message_media(message_id)
            asset = message.get("mediaAsset") or {}
            transcript = self._stt.transcribe(
                audio, asset.get("contentType") or content_type
            )
            if not (transcript or "").strip():
                if not self._settings.gemini_api_key:
                    logger.warning(
                        "STT fail-open sem chave — pulando áudio %s", message_id
                    )
                    return
                raise RuntimeError(f"STT vazio message={message_id}")
            logger.info("STT ok message=%s chars=%d", message_id, len(transcript))

        text = transcript or message.get("body") or ""
        if not text.strip():
            logger.info("mensagem sem texto analisável: %s", message_id)
            return

        coach_note: Optional[str] = None
        coach_tone: Optional[str] = None
        if message["type"] == "AUDIO" and transcript and transcript.strip():
            coach = self._coach.note(transcript)
            if coach:
                coach_note = coach.note
                coach_tone = coach.tone

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

        try:
            result = self._extractor.extract(
                carteira,
                summaries,
                messages,
                target_index=target_index,
                human_links=ctx.get("humanLinks") or [],
                session_retrieve=ctx.get("previousSummaries") or [],
                sales_policy=ctx.get("salesPolicy") or None,
                previous_brief=previous_brief,
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
                    target_index=target_index,
                    human_links=ctx.get("humanLinks") or [],
                    session_retrieve=ctx.get("previousSummaries") or [],
                    sales_policy=ctx.get("salesPolicy") or None,
                    previous_brief=previous_brief,
                    repair_feedback=quality_issues,
                )
        except Exception:
            logger.exception(
                "extração falhou message=%s — grava Card de Bordo mínimo",
                message_id,
            )
            result = ExtractionResult(session_summary="", facts=[], unknowns=[])
        payload = to_analysis_payload(
            ctx,
            result,
            transcript,
            self._settings.resolver_min_confidence,
            coach_note=coach_note,
            coach_tone=coach_tone,
            source_text=text,
            source_sent_at=str(message.get("sentAt") or ""),
        )
        self._backend.post_analysis(message_id, payload)
        logger.info(
            "análise publicada message=%s facts=%d unknowns=%d",
            message_id,
            len(payload["facts"]),
            len(payload["unknowns"]),
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
            text = m.get("transcript") or m.get("body") or f"[{m['type'].lower()}]"
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
