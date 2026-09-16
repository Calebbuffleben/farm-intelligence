"""Regra de ouro: confiança abaixo do limiar não vira farm_id no fato."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.schema import (
    DealBriefOut,
    ExtractedFact,
    ExtractionResult,
    UnknownSpan,
)
from src.worker.payload import (
    deal_quality_issues,
    is_unusable_brief,
    resolve_due_at,
    to_analysis_payload,
)

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
            producer_position="Quer comprar o defensivo X se o prazo atender a safra.",
            intent="ALTA",
            urgency="ALTA",
            pain_point="Concorrente parcelou em mais vezes.",
            next_action="Ofereça prazo de safra e confirme entrega para sexta.",
            next_action_reason="Prazo e entrega são as condições declaradas para fechar.",
            next_action_owner="RTV",
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
            producer_position="Está comparando o preço antes de decidir.",
            intent="media",
            urgency="baixa",
            next_action="Mandar tabela.",
            next_action_reason="O preço foi solicitado pelo produtor.",
            next_action_owner="RTV",
            next_action_kind="mandar_tabela_inventado",
            blocker_subtype="subtipo_inventado",
        ),
    )
    payload = to_analysis_payload(CTX, result, None, 0.7)
    deal = payload["deal"]
    assert deal["analysisQuality"] == "PARTIAL"
    assert deal["nextActionKind"] == "followup"
    assert "Pediu preço" not in deal["nextAction"]


def test_deal_fallback_when_absent():
    result = ExtractionResult(session_summary="ok", facts=[])
    payload = to_analysis_payload(CTX, result, "Pediu herbicida do Chapadão.", 0.7)
    deal = payload["deal"]
    assert deal["stage"] == "SONDAGEM"
    assert "Chapadão" in deal["contextSummary"]
    assert deal["nextActionKind"] == "followup"
    assert deal["analysisQuality"] == "PARTIAL"
    assert "Releia a conversa" not in deal["nextAction"]


def test_text_body_is_used_by_contextual_fallback():
    result = ExtractionResult(session_summary="", facts=[])
    payload = to_analysis_payload(
        CTX,
        result,
        None,
        0.7,
        source_text="Preciso do herbicida X para a soja do Chapadão.",
    )
    assert "herbicida X" in payload["deal"]["producerPosition"]
    assert "última mensagem" in payload["deal"]["nextAction"]


def test_fallback_turns_price_delivery_and_window_into_action():
    result = ExtractionResult(session_summary="", facts=[])
    payload = to_analysis_payload(
        CTX,
        result,
        None,
        0.7,
        source_text=(
            "Preciso fechar o herbicida da soja do Chapadão essa semana. "
            "A janela abre sexta, quero 50 galões, mas preciso confirmar preço e entrega."
        ),
    )
    deal = payload["deal"]
    assert deal["analysisQuality"] == "PARTIAL"
    assert deal["nextActionKind"] == "proposta"
    assert "preço" in deal["nextAction"]
    assert "disponibilidade" in deal["nextAction"]
    assert "janela" in deal["nextAction"]
    assert "Responda ao ponto específico" not in deal["nextAction"]


def test_previous_brief_is_preserved_as_stale():
    previous = {
        "stage": "NEGOCIACAO",
        "stageConfidence": 0.9,
        "contextSummary": "Negocia o produto X.",
        "producerPosition": "Aceita o produto; discute prazo.",
        "intent": "ALTA",
        "urgency": "MEDIA",
        "painPoint": "Prazo curto.",
        "nextAction": "Enviar a condição aprovada para o produto X.",
        "nextActionReason": "O prazo é a única objeção aberta.",
        "nextActionOwner": "RTV",
        "nextActionKind": "proposta",
        "products": ["Produto X"],
    }
    ctx = {**CTX, "previousBrief": previous}
    payload = to_analysis_payload(ctx, ExtractionResult(session_summary=""), None, 0.7)
    assert payload["deal"]["stage"] == "NEGOCIACAO"
    assert payload["deal"]["nextAction"] == previous["nextAction"]
    assert payload["deal"]["analysisQuality"] == "STALE"


def test_generic_previous_brief_is_replaced_by_partial():
    previous = {
        "stage": "SONDAGEM",
        "contextSummary": "Mensagem do produtor recebida; a extração automática não fechou o brief.",
        "nextAction": "Releia a conversa e confirme o próximo passo com o produtor.",
        "nextActionKind": "aguardar",
        "nextActionOwner": "RTV",
        "analysisQuality": "PARTIAL",
    }
    assert is_unusable_brief(previous)
    ctx = {**CTX, "previousBrief": previous}
    payload = to_analysis_payload(
        ctx,
        ExtractionResult(session_summary=""),
        None,
        0.7,
        source_text="Preciso de 50 galões do herbicida X para o Chapadão.",
    )
    deal = payload["deal"]
    assert deal["analysisQuality"] == "PARTIAL"
    assert "Releia a conversa" not in deal["nextAction"]
    assert "extração automática não fechou" not in deal["contextSummary"]
    assert "herbicida X" in deal["producerPosition"]


def test_relative_deadline_is_resolved_from_message_timestamp():
    due = resolve_due_at("até amanhã", "2026-09-16T12:00:00Z")
    assert due is not None
    assert due.startswith("2026-09-17T18:00:00")


def test_generic_wait_is_rejected():
    deal = DealBriefOut(
        stage="SONDAGEM",
        context_summary="Mensagem recebida.",
        producer_position="Ainda não decidiu.",
        intent="MEDIA",
        urgency="MEDIA",
        next_action="Aguarde o produtor.",
        next_action_reason="É preciso esperar.",
        next_action_owner="RTV",
        next_action_kind="aguardar",
    )
    issues = deal_quality_issues(deal)
    assert "texto operacional genérico" in issues
    assert "aguardar sem prazo e gatilho de saída" in issues


def test_discount_above_authority_requires_manager():
    deal = DealBriefOut(
        stage="NEGOCIACAO",
        context_summary="Produtor condicionou a compra ao desconto.",
        producer_position="Pediu 8% para fechar o produto X.",
        intent="ALTA",
        urgency="ALTA",
        next_action="Ofereça 8% no produto X para fechar hoje.",
        next_action_reason="O desconto é a condição declarada.",
        next_action_owner="RTV",
        next_action_kind="proposta",
    )
    issues = deal_quality_issues(deal, {"discountAuthorityPct": 5})
    assert "desconto acima da alçada sem escalada ao gerente" in issues


def test_actionable_conversation_fixture_covers_quality_gate_scenarios():
    fixture_path = Path(__file__).parent / "fixtures" / "actionable_deals.json"
    scenarios = json.loads(fixture_path.read_text(encoding="utf-8"))
    names = {item["scenario"] for item in scenarios}
    assert names == {
        "preco",
        "logistica",
        "concorrente",
        "fechamento",
        "pos_venda",
        "aguardar_com_gatilho",
    }
    for item in scenarios:
        assert item["messages"]
        assert item["expected"]["mustMention"]


if __name__ == "__main__":
    test_low_confidence_farm_is_unknown_not_fact_link()
    test_high_confidence_keeps_farm_and_crop_season()
    test_invented_farm_id_is_dropped()
    test_coach_fields_pass_through()
    test_coach_omitted_when_absent()
    test_valid_deal_passes_through()
    test_deal_out_of_vocabulary_falls_to_defaults()
    test_deal_fallback_when_absent()
    test_text_body_is_used_by_contextual_fallback()
    test_previous_brief_is_preserved_as_stale()
    test_generic_previous_brief_is_replaced_by_partial()
    test_relative_deadline_is_resolved_from_message_timestamp()
    test_generic_wait_is_rejected()
    test_discount_above_authority_requires_manager()
    test_actionable_conversation_fixture_covers_quality_gate_scenarios()
    print("pipeline payload ok")
