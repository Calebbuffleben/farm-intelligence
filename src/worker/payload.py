"""Monta o payload POST /internal/messages/:id/analysis.

Módulo puro (sem Gemini) para a regra: confiança < limiar não grava farm_id.
"""

import logging
from typing import Any, Dict, List, Optional

from ..extractor.schema import (
    FACT_SUBTYPES,
    NEXT_ACTION_KINDS,
    DealBriefOut,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

DEAL_STAGES = {"SONDAGEM", "NEGOCIACAO", "FECHAMENTO", "POS_VENDA", "SEM_NEGOCIO"}
DEAL_LEVELS = {"BAIXA", "MEDIA", "ALTA"}


def fallback_deal(text: str = "") -> Dict[str, Any]:
    """Card de Bordo mínimo — a UI não pode ficar presa em 'pending' sem DealBrief."""
    snippet = " ".join((text or "").split())[:180]
    return {
        "stage": "SONDAGEM",
        "stageConfidence": 0.3,
        "contextSummary": snippet
        or "Mensagem do produtor recebida; a extração automática não fechou o brief.",
        "intent": "MEDIA",
        "urgency": "MEDIA",
        "painPoint": None,
        "nextAction": "Releia a conversa e confirme o próximo passo com o produtor.",
        "nextActionKind": "aguardar",
        "nextActionDueHint": None,
        "blockerSubtype": None,
        "products": [],
    }


def to_deal_payload(deal: Optional[DealBriefOut]) -> Optional[Dict[str, Any]]:
    """Saneia o bloco `deal`: vocabulário fechado cai no default, textos truncados.

    Sem deal do Gemini → None; o caller grava fallback_deal para o Card aparecer.
    """
    if deal is None:
        return None
    stage = deal.stage if deal.stage in DEAL_STAGES else "SEM_NEGOCIO"
    kind = deal.next_action_kind if deal.next_action_kind in NEXT_ACTION_KINDS else "aguardar"
    blocker = deal.blocker_subtype if deal.blocker_subtype in FACT_SUBTYPES else None
    if deal.blocker_subtype and blocker is None:
        logger.warning("blocker_subtype fora do vocabulário: %s", deal.blocker_subtype)
    products = [p.strip()[:80] for p in deal.products if isinstance(p, str) and p.strip()]
    context = (deal.context_summary or "").strip()[:600]
    action = (deal.next_action or "").strip()[:400]
    if not context or not action:
        return None
    return {
        "stage": stage,
        "stageConfidence": max(0.0, min(1.0, float(deal.stage_confidence))),
        "contextSummary": context,
        "intent": deal.intent if deal.intent in DEAL_LEVELS else "MEDIA",
        "urgency": deal.urgency if deal.urgency in DEAL_LEVELS else "MEDIA",
        "painPoint": deal.pain_point.strip()[:400] if deal.pain_point else None,
        "nextAction": action,
        "nextActionKind": kind,
        "nextActionDueHint": deal.next_action_due_hint[:80] if deal.next_action_due_hint else None,
        "blockerSubtype": blocker,
        "products": products[:10],
    }


def to_analysis_payload(
    ctx: Dict[str, Any],
    result: ExtractionResult,
    transcript: Optional[str],
    min_confidence: float,
    coach_note: Optional[str] = None,
    coach_tone: Optional[str] = None,
) -> Dict[str, Any]:
    valid_farm_ids = {farm["id"] for farm in ctx.get("farms", [])}
    crop_season_by_farm: Dict[str, Dict[str, str]] = {
        farm["id"]: {cs["crop"]: cs["id"] for cs in farm.get("cropSeasons", [])}
        for farm in ctx.get("farms", [])
    }

    links: Dict[str, Dict[str, Any]] = {}
    facts: List[Dict[str, Any]] = []
    unknowns: List[Dict[str, Any]] = []

    for fact in result.facts:
        farm_id = fact.farm_id if fact.farm_id in valid_farm_ids else None
        if fact.farm_id and farm_id is None:
            logger.warning("farm_id inventado descartado: %s", fact.farm_id)

        if farm_id and fact.confidence < min_confidence:
            unknowns.append(
                {
                    "spanText": fact.evidence_span or fact.headline,
                    "candidates": [
                        {"farmId": farm_id, "confidence": fact.confidence}
                    ],
                }
            )
            farm_id = None

        crop_season_id = None
        if farm_id and fact.crop:
            crop_season_id = crop_season_by_farm.get(farm_id, {}).get(
                fact.crop.lower()
            )

        if farm_id:
            current = links.get(farm_id)
            if current is None or fact.confidence > current["confidence"]:
                link: Dict[str, Any] = {
                    "farmId": farm_id,
                    "confidence": fact.confidence,
                }
                if crop_season_id:
                    link["cropSeasonId"] = crop_season_id
                if fact.evidence_span:
                    link["spanText"] = fact.evidence_span[:1000]
                links[farm_id] = link

        facts.append(
            {
                "kind": fact.kind,
                "subtype": fact.subtype if fact.subtype in FACT_SUBTYPES else "outro",
                "severity": fact.severity,
                "confidence": fact.confidence,
                **({"farmId": farm_id} if farm_id else {}),
                **({"cropSeasonId": crop_season_id} if crop_season_id else {}),
                **({"productKey": fact.product[:120]} if fact.product else {}),
                "headline": fact.headline[:300],
                **({"moneyHint": fact.money_hint} if fact.money_hint else {}),
                **({"dueHintText": fact.due_hint_text} if fact.due_hint_text else {}),
                **(
                    {"evidenceSpan": fact.evidence_span[:1000]}
                    if fact.evidence_span
                    else {}
                ),
            }
        )

    for unknown in result.unknowns:
        unknowns.append(
            {
                "spanText": unknown.span_text[:1000],
                "candidates": [
                    {"farmId": c.farm_id, "confidence": c.confidence}
                    for c in unknown.candidates
                    if c.farm_id in valid_farm_ids
                ],
            }
        )

    payload: Dict[str, Any] = {
        "links": list(links.values()),
        "facts": facts,
        "unknowns": unknowns,
        "sessionSummary": result.session_summary[:2000],
        "coachNote": coach_note[:500] if coach_note else None,
    }
    if transcript:
        payload["transcript"] = transcript
    if coach_tone:
        payload["coachTone"] = coach_tone
    deal = to_deal_payload(result.deal)
    payload["deal"] = deal or fallback_deal(transcript or result.session_summary)
    return payload
