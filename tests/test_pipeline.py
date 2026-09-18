"""Áudio vai ao extrator Gemini mesmo sem MediaAsset / STT isolado."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.schema import DealBriefOut, ExtractionResult
from src.worker.pipeline import MessagePipeline


def _pipeline() -> MessagePipeline:
    p = MessagePipeline.__new__(MessagePipeline)
    p._settings = MagicMock(gemini_api_key="k", resolver_min_confidence=0.7)
    p._backend = MagicMock()
    p._extractor = MagicMock()
    p._coach = MagicMock()
    p._coach.note.return_value = None
    p._extractor.extract.return_value = ExtractionResult(
        session_summary="ok",
        facts=[],
        unknowns=[],
        transcript="o preco do herbicida esta alto",
        deal=DealBriefOut(
            stage="NEGOCIACAO",
            context_summary="Produtor reclamou do preço do herbicida.",
            producer_position="Acha o herbicida caro.",
            intent="MEDIA",
            urgency="MEDIA",
            next_action="Envie hoje a condição permitida do herbicida e peça o volume.",
            next_action_kind="proposta",
        ),
    )
    return p


def _audio_ctx(*, asset=None):
    return {
        "message": {
            "id": "m1",
            "direction": "IN",
            "type": "AUDIO",
            "transcript": None,
            "body": None,
            "sentAt": "2026-09-18T00:00:00Z",
            "mediaAsset": asset,
        },
        "analysisAllowed": True,
        "sessionMessages": [
            {"id": "m1", "direction": "IN", "type": "AUDIO", "sentAt": ""}
        ],
        "previousSummaries": [],
        "humanLinks": [],
        "salesPolicy": None,
        "previousBrief": None,
        "producer": None,
        "farms": [],
    }


def test_audio_goes_to_extractor_without_stt():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx()
    p._backend.get_message_media.return_value = (b"ogg-bytes", "audio/ogg")

    p.process("m1")

    p._backend.get_message_media.assert_called_once_with("m1")
    kwargs = p._extractor.extract.call_args.kwargs
    assert kwargs["audio"] == b"ogg-bytes"
    assert kwargs["audio_mime"] == "audio/ogg"
    payload = p._backend.post_analysis.call_args[0][1]
    assert payload["transcript"] == "o preco do herbicida esta alto"
    assert payload["deal"]["stage"] == "NEGOCIACAO"


def test_audio_prefers_asset_content_type():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx(
        asset={"id": "a1", "contentType": "audio/mp4"}
    )
    p._backend.get_message_media.return_value = (b"mp4", "application/octet-stream")

    p.process("m1")

    assert p._extractor.extract.call_args.kwargs["audio_mime"] == "audio/mp4"


def test_audio_media_fetch_failure_does_not_publish():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx()
    p._backend.get_message_media.side_effect = RuntimeError("canal fora")
    try:
        p.process("m1")
        raise AssertionError("expected raise")
    except RuntimeError as exc:
        assert "canal fora" in str(exc)
    p._backend.post_analysis.assert_not_called()
    p._extractor.extract.assert_not_called()


def test_audio_empty_extract_does_not_ack():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx()
    p._backend.get_message_media.return_value = (b"ogg-bytes", "audio/ogg")
    p._extractor.extract.return_value = ExtractionResult(
        session_summary="", facts=[], unknowns=[]
    )
    try:
        p.process("m1")
        raise AssertionError("expected raise")
    except RuntimeError as exc:
        assert "não analisou o áudio" in str(exc)
    p._backend.post_analysis.assert_not_called()


def test_image_caption_goes_to_extractor_without_media_fetch():
    p = _pipeline()
    p._backend.get_context.return_value = {
        "message": {
            "id": "m1",
            "direction": "IN",
            "type": "IMAGE",
            "transcript": None,
            "body": "ferrugem na soja do chapadao",
            "sentAt": "2026-09-18T00:00:00Z",
            "mediaAsset": None,
        },
        "analysisAllowed": True,
        "sessionMessages": [
            {
                "id": "m1",
                "direction": "IN",
                "type": "IMAGE",
                "body": "ferrugem na soja do chapadao",
                "sentAt": "",
            }
        ],
        "previousSummaries": [],
        "humanLinks": [],
        "salesPolicy": None,
        "previousBrief": None,
        "producer": None,
        "farms": [],
    }
    p._extractor.extract.return_value = ExtractionResult(
        session_summary="ok", facts=[], unknowns=[]
    )
    p.process("m1")
    p._backend.get_message_media.assert_not_called()
    session = p._extractor.extract.call_args.args[2]
    assert session[0]["text"] == "[imagem] ferrugem na soja do chapadao"
    p._backend.post_analysis.assert_called_once()


if __name__ == "__main__":
    test_audio_goes_to_extractor_without_stt()
    test_audio_prefers_asset_content_type()
    test_audio_media_fetch_failure_does_not_publish()
    test_audio_empty_extract_does_not_ack()
    test_image_caption_goes_to_extractor_without_media_fetch()
    print("pipeline ok")
