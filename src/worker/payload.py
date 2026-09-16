"""Monta o payload POST /internal/messages/:id/analysis.

Módulo puro (sem Gemini) para a regra: confiança < limiar não grava farm_id.
"""

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..extractor.schema import (
    FACT_SUBTYPES,
    NEXT_ACTION_KINDS,
    DealBriefOut,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

DEAL_STAGES = {"SONDAGEM", "NEGOCIACAO", "FECHAMENTO", "POS_VENDA", "SEM_NEGOCIO"}
DEAL_LEVELS = {"BAIXA", "MEDIA", "ALTA"}
DEAL_OWNERS = {"RTV", "MANAGER"}
ANALYSIS_QUALITIES = {"COMPLETE", "PARTIAL", "STALE"}
BUSINESS_TZ = ZoneInfo("America/Sao_Paulo")
GENERIC_PHRASES = (
    "releia a conversa",
    "confirme o próximo passo",
    "entre em contato",
    "mensagem recebida",
    "extração automática não fechou",
)


def _plain(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


def resolve_due_at(hint: Optional[str], sent_at: Optional[str]) -> Optional[str]:
    """Resolve apenas prazos PT-BR inequívocos, ancorados no horário da mensagem."""
    if not hint or not sent_at:
        return None
    try:
        anchor = datetime.fromisoformat(sent_at.replace("Z", "+00:00")).astimezone(BUSINESS_TZ)
    except (TypeError, ValueError):
        return None
    normalized = _plain(hint)
    target_date = None
    if re.search(r"\bamanha\b", normalized):
        target_date = anchor.date() + timedelta(days=1)
    elif re.search(r"\bhoje\b", normalized):
        target_date = anchor.date()
    else:
        weekdays = {
            "segunda": 0,
            "terca": 1,
            "quarta": 2,
            "quinta": 3,
            "sexta": 4,
            "sabado": 5,
            "domingo": 6,
        }
        for label, weekday in weekdays.items():
            if re.search(rf"\b{label}(?:-feira)?\b", normalized):
                target_date = anchor.date() + timedelta(days=(weekday - anchor.weekday()) % 7)
                break
    if target_date is None:
        match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", normalized)
        if match:
            year = int(match.group(3)) if match.group(3) else anchor.year
            if year < 100:
                year += 2000
            try:
                target_date = anchor.date().replace(
                    year=year, month=int(match.group(2)), day=int(match.group(1))
                )
            except ValueError:
                return None
    if target_date is None:
        return None
    # Fim do expediente local evita marcar como vencido no começo do próprio dia.
    return datetime.combine(
        target_date,
        datetime.min.time().replace(hour=18),
        tzinfo=BUSINESS_TZ,
    ).isoformat()


def valid_iso_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except (TypeError, ValueError):
        return None


def deal_quality_issues(
    deal: Optional[DealBriefOut],
    sales_policy: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if deal is None:
        return ["bloco deal ausente"]
    issues: List[str] = []
    context = (deal.context_summary or "").strip()
    action = (deal.next_action or "").strip()
    position = (deal.producer_position or "").strip()
    reason = (deal.next_action_reason or "").strip()
    combined = _plain(f"{context} {action}")
    if not context:
        issues.append("context_summary vazio")
    if not position:
        issues.append("producer_position vazio")
    if not action or len(action.split()) < 5:
        issues.append("next_action sem ação e objeto concretos")
    if not reason:
        issues.append("next_action_reason vazio")
    if any(phrase in combined for phrase in GENERIC_PHRASES):
        issues.append("texto operacional genérico")
    if deal.next_action_owner not in DEAL_OWNERS:
        issues.append("next_action_owner inválido")
    if deal.next_action_kind == "escalar_gestor" and deal.next_action_owner != "MANAGER":
        issues.append("escalada deve ter o gerente como responsável")
    if deal.next_action_owner == "MANAGER" and not deal.manager_guidance:
        issues.append("ação do gerente sem manager_guidance")
    if deal.analysis_quality != "COMPLETE":
        issues.append("analysis_quality reservado ao fail-open")
    if deal.next_action_kind == "aguardar":
        if not deal.next_action_due_hint or "se " not in _plain(action):
            issues.append("aguardar sem prazo e gatilho de saída")
    if deal.next_action_due_at and not valid_iso_datetime(deal.next_action_due_at):
        issues.append("next_action_due_at não é ISO-8601 válido")
    authority = (sales_policy or {}).get("discountAuthorityPct")
    if isinstance(authority, (int, float)):
        action_percentages = [
            float(value.replace(",", "."))
            for value in re.findall(
                r"(\d+(?:[.,]\d+)?)\s*%", deal.next_action
            )
        ]
        if any(value > float(authority) for value in action_percentages):
            if deal.next_action_owner != "MANAGER":
                issues.append("desconto acima da alçada sem escalada ao gerente")
        reply_percentages = [
            float(value.replace(",", "."))
            for value in re.findall(
                r"(\d+(?:[.,]\d+)?)\s*%", deal.suggested_reply or ""
            )
        ]
        if any(value > float(authority) for value in reply_percentages):
            issues.append("resposta sugerida promete desconto acima da alçada")
    return issues


def _partial_action(facts: List[Any], snippet: str) -> tuple[str, str, Optional[str]]:
    if facts:
        fact = facts[0]
        subject = fact.product or fact.headline
        if fact.subtype == "preco":
            return (
                f"Confirme a condição comercial solicitada para {subject} dentro da alçada da revenda.",
                "A condição comercial é a objeção identificada na última mensagem.",
                "escalar_gestor" if fact.severity == "CRITICAL" else "proposta",
            )
        if fact.subtype == "logistica":
            return (
                f"Confirme disponibilidade e janela de entrega para {subject}.",
                "A venda depende da viabilidade logística levantada pelo produtor.",
                "logistica",
            )
        if fact.kind == "FOLLOWUP":
            return (
                f"Cumpra o compromisso registrado: {fact.headline}.",
                "Há um follow-up explícito que precisa ser concluído para manter o avanço.",
                "followup",
            )
        return (
            f"Responda objetivamente ao ponto identificado: {fact.headline}.",
            "Esse é o sinal comercial mais concreto extraído da última mensagem.",
            "followup",
        )
    return (
        f"Responda ao ponto específico da última mensagem: “{snippet}”.",
        "A análise detalhada falhou, mas a mensagem do produtor exige uma resposta contextual.",
        "followup",
    )


def fallback_deal(
    text: str = "",
    facts: Optional[List[Any]] = None,
    previous_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fail-open fundamentado: preserva o brief anterior ou usa a mensagem real."""
    if previous_brief:
        preserved = dict(previous_brief)
        preserved.pop("updatedAt", None)
        preserved.setdefault("stageConfidence", 0.3)
        preserved.setdefault("producerPosition", preserved.get("contextSummary", ""))
        preserved.setdefault("dealChange", None)
        preserved.setdefault(
            "nextActionReason", "Orientação preservada porque a nova análise não foi concluída."
        )
        preserved.setdefault("nextActionOwner", "RTV")
        preserved.setdefault("nextActionDueHint", None)
        preserved.setdefault("nextActionDueAt", None)
        preserved.setdefault("suggestedReply", None)
        preserved.setdefault("managerGuidance", None)
        preserved["analysisQuality"] = "STALE"
        return preserved
    snippet = " ".join((text or "").split())[:180]
    if not snippet:
        snippet = "conteúdo não textual recebido"
    action, reason, kind = _partial_action(facts or [], snippet)
    first_fact = (facts or [None])[0]
    return {
        "stage": "SONDAGEM",
        "stageConfidence": 0.3,
        "contextSummary": f'O produtor trouxe este ponto: “{snippet}”. A classificação detalhada requer revisão.',
        "producerPosition": f'Última manifestação do produtor: “{snippet}”.',
        "dealChange": None,
        "intent": "MEDIA",
        "urgency": "MEDIA",
        "painPoint": first_fact.headline if first_fact else None,
        "nextAction": action,
        "nextActionReason": reason,
        "nextActionOwner": (
            "MANAGER" if kind == "escalar_gestor" else "RTV"
        ),
        "nextActionKind": kind,
        "nextActionDueHint": None,
        "nextActionDueAt": None,
        "suggestedReply": None,
        "managerGuidance": (
            "Revise a condição solicitada e defina a alçada antes da resposta."
            if kind == "escalar_gestor"
            else None
        ),
        "analysisQuality": "PARTIAL",
        "blockerSubtype": None,
        "products": [],
    }


def to_deal_payload(
    deal: Optional[DealBriefOut],
    source_sent_at: Optional[str] = None,
    sales_policy: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Saneia o bloco `deal`: vocabulário fechado cai no default, textos truncados.

    Sem deal do Gemini → None; o caller grava fallback_deal para o Card aparecer.
    """
    if deal is None:
        return None
    stage = deal.stage if deal.stage in DEAL_STAGES else "SEM_NEGOCIO"
    kind = deal.next_action_kind if deal.next_action_kind in NEXT_ACTION_KINDS else None
    blocker = deal.blocker_subtype if deal.blocker_subtype in FACT_SUBTYPES else None
    if deal.blocker_subtype and blocker is None:
        logger.warning("blocker_subtype fora do vocabulário: %s", deal.blocker_subtype)
    products = [p.strip()[:80] for p in deal.products if isinstance(p, str) and p.strip()]
    context = (deal.context_summary or "").strip()[:600]
    action = (deal.next_action or "").strip()[:400]
    if not context or not action or kind is None or deal_quality_issues(deal, sales_policy):
        return None
    due_at = valid_iso_datetime(deal.next_action_due_at) or resolve_due_at(
        deal.next_action_due_hint, source_sent_at
    )
    return {
        "stage": stage,
        "stageConfidence": max(0.0, min(1.0, float(deal.stage_confidence))),
        "contextSummary": context,
        "producerPosition": deal.producer_position.strip()[:600],
        "dealChange": deal.deal_change.strip()[:500] if deal.deal_change else None,
        "intent": deal.intent if deal.intent in DEAL_LEVELS else "MEDIA",
        "urgency": deal.urgency if deal.urgency in DEAL_LEVELS else "MEDIA",
        "painPoint": deal.pain_point.strip()[:400] if deal.pain_point else None,
        "nextAction": action,
        "nextActionReason": deal.next_action_reason.strip()[:400],
        "nextActionOwner": deal.next_action_owner,
        "nextActionKind": kind,
        "nextActionDueHint": deal.next_action_due_hint[:80] if deal.next_action_due_hint else None,
        "nextActionDueAt": due_at,
        "suggestedReply": deal.suggested_reply.strip()[:1000] if deal.suggested_reply else None,
        "managerGuidance": (
            deal.manager_guidance.strip()[:600] if deal.manager_guidance else None
        ),
        "analysisQuality": deal.analysis_quality,
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
    source_text: Optional[str] = None,
    source_sent_at: Optional[str] = None,
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
                    {"dueAt": fact_due_at}
                    if (
                        fact_due_at := valid_iso_datetime(fact.due_at)
                        or resolve_due_at(fact.due_hint_text, source_sent_at)
                    )
                    else {}
                ),
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
    deal = to_deal_payload(
        result.deal, source_sent_at, ctx.get("salesPolicy") or None
    )
    payload["deal"] = deal or fallback_deal(
        source_text or transcript or result.session_summary,
        result.facts,
        ctx.get("previousBrief"),
    )
    return payload
