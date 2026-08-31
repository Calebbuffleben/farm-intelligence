"""Monta o payload POST /internal/messages/:id/analysis.

Módulo puro (sem Gemini) para a regra: confiança < limiar não grava farm_id.
"""

import logging
from typing import Any, Dict, List, Optional

from ..extractor.schema import FACT_SUBTYPES, ExtractionResult

logger = logging.getLogger(__name__)


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
                links[farm_id] = {
                    "farmId": farm_id,
                    "cropSeasonId": crop_season_id,
                    "spanText": fact.evidence_span,
                    "confidence": fact.confidence,
                }

        facts.append(
            {
                "kind": fact.kind,
                "subtype": fact.subtype if fact.subtype in FACT_SUBTYPES else "outro",
                "severity": fact.severity,
                "confidence": fact.confidence,
                **({"farmId": farm_id} if farm_id else {}),
                **({"cropSeasonId": crop_season_id} if crop_season_id else {}),
                **({"productKey": fact.product} if fact.product else {}),
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
    return payload
