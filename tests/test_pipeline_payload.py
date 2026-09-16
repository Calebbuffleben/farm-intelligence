"""Regra de ouro: confiança abaixo do limiar não vira farm_id no fato."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.schema import (
    DealBriefOut,
    ExtractedFact,
    ExtractionResult,
    UnknownSpan,
)
from src.worker.payload import to_analysis_payload

CTX = {
    "farms": [
        {
            "id": "farm-1",
            "name": "Chapadão",
            "cropSeasons": [{"id": "cs-1", "crop": "soja", "seasonLabel": "2026/27"}],
        }
    ]
}


def test_low_confidence_farm_is_unknown_not_fact_link():
    result = ExtractionResult(
        session_summary="pedido de galões",
        facts=[
            ExtractedFact(
                kind="OPORTUNIDADE",
                subtype="volume_pedido",
                severity="INFO",
                confidence=0.4,
                farm_id="farm-1",
                crop="soja",
                headline="50 galões para o Chapadão",
                evidence_message_index=0,
                evidence_span="manda 50 galões",
            )
        ],
    )
    payload = to_analysis_payload(CTX, result, None, 0.7)
    assert payload["facts"][0].get("farmId") is None
    assert payload["links"] == []
    assert payload["unknowns"][0]["candidates"][0]["farmId"] == "farm-1"


def test_high_confidence_keeps_farm_and_crop_season():
    result = ExtractionResult(
        session_summary="ok",
        facts=[
            ExtractedFact(
                kind="OBJECAO",
                subtype="preco",
                severity="WARNING",
                confidence=0.91,
                farm_id="farm-1",
                crop="soja",
                headline="Pediu 5% de desconto",
                evidence_message_index=0,
            )
        ],
    )
    payload = to_analysis_payload(CTX, result, None, 0.7)
    assert payload["facts"][0]["farmId"] == "farm-1"
    assert payload["facts"][0]["cropSeasonId"] == "cs-1"
    assert payload["links"][0]["farmId"] == "farm-1"


def test_invented_farm_id_is_dropped():
    result = ExtractionResult(
        session_summary="ok",
        facts=[
            ExtractedFact(
                kind="RISCO",
                subtype="outro",
                confidence=0.99,
                farm_id="farm-inventada",
                headline="risco",
                evidence_message_index=0,
            )
        ],
        unknowns=[
            UnknownSpan(
                span_text="área de cima",
                evidence_message_index=0,
                reason="ambigua",
                candidates=[],
            )
        ],
    )
    payload = to_analysis_payload(CTX, result, "transcrição", 0.7)
    assert "farmId" not in payload["facts"][0]
    assert payload["unknowns"][0]["spanText"] == "área de cima"
    assert payload["transcript"] == "transcrição"


def test_coach_fields_pass_through():
    result = ExtractionResult(session_summary="ok", facts=[])
    payload = to_analysis_payload(
        CTX,
        result,
        "fala",
        0.7,
        coach_note="Produtor hesitou no preço.",
        coach_tone="alerta",
    )
    assert payload["coachNote"] == "Produtor hesitou no preço."
    assert payload["coachTone"] == "alerta"


def test_coach_omitted_when_absent():
    result = ExtractionResult(session_summary="ok", facts=[])
    payload = to_analysis_payload(CTX, result, None, 0.7)
    assert payload["coachNote"] is None
    assert "coachTone" not in payload


def test_valid_deal_passes_through():
    result = ExtractionResult(
        session_summary="ok",
        facts=[],
        deal=DealBriefOut(
            stage="NEGOCIACAO",
            stage_confidence=0.8,
            context_summary="Produtor quer fechar o defensivo X antes do plantio.",
            intent="ALTA",
            urgency="ALTA",
            pain_point="Concorrente parcelou em mais vezes.",
            next_action="Ofereça prazo de safra e confirme entrega para sexta.",
            next_action_kind="proposta",
            blocker_subtype="preco",
            products=["Defensivo X"],
        ),
    )
    payload = to_analysis_payload(CTX, result, None, 0.7)
    deal = payload["deal"]
    assert deal["stage"] == "NEGOCIACAO"
    assert deal["intent"] == "ALTA"
    assert deal["nextActionKind"] == "proposta"
    assert deal["blockerSubtype"] == "preco"
    assert deal["products"] == ["Defensivo X"]


def test_deal_out_of_vocabulary_falls_to_defaults():
    result = ExtractionResult(
        session_summary="ok",
        facts=[],
        deal=DealBriefOut(
            stage="negociacao",
            context_summary="Pediu preço.",
            intent="media",
            urgency="baixa",
            next_action="Mandar tabela.",
            next_action_kind="mandar_tabela_inventado",
            blocker_subtype="subtipo_inventado",
        ),
    )
    payload = to_analysis_payload(CTX, result, None, 0.7)
    deal = payload["deal"]
    assert deal["stage"] == "SEM_NEGOCIO"
    assert deal["intent"] == "MEDIA"
    assert deal["urgency"] == "MEDIA"
    assert deal["nextActionKind"] == "aguardar"
    assert deal["blockerSubtype"] is None
    assert deal["painPoint"] is None


def test_deal_fallback_when_absent():
    result = ExtractionResult(session_summary="ok", facts=[])
    payload = to_analysis_payload(CTX, result, "Pediu herbicida do Chapadão.", 0.7)
    deal = payload["deal"]
    assert deal["stage"] == "SONDAGEM"
    assert "Chapadão" in deal["contextSummary"]
    assert deal["nextActionKind"] == "aguardar"


if __name__ == "__main__":
    test_low_confidence_farm_is_unknown_not_fact_link()
    test_high_confidence_keeps_farm_and_crop_season()
    test_invented_farm_id_is_dropped()
    test_coach_fields_pass_through()
    test_coach_omitted_when_absent()
    test_valid_deal_passes_through()
    test_deal_out_of_vocabulary_falls_to_defaults()
    test_deal_fallback_when_absent()
    print("pipeline payload ok")
