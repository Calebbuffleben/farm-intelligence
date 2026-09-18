"""Áudio segue para STT mesmo sem MediaAsset no object storage."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.schema import ExtractionResult
from src.worker.pipeline import MessagePipeline


def _pipeline() -> MessagePipeline:
    p = MessagePipeline.__new__(MessagePipeline)
    p._settings = MagicMock(gemini_api_key="k", resolver_min_confidence=0.7)
    p._backend = MagicMock()
    p._stt = MagicMock()
    p._extractor = MagicMock()
    p._coach = MagicMock()
    p._coach.note.return_value = None
    p._extractor.extract.return_value = ExtractionResult(
        session_summary="ok", facts=[], unknowns=[]
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


def test_audio_without_asset_fetches_by_message_id():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx()
    p._backend.get_message_media.return_value = (b"ogg-bytes", "audio/ogg")
    p._stt.transcribe.return_value = "o preco do herbicida esta alto"

    p.process("m1")

    p._backend.get_message_media.assert_called_once_with("m1")
    p._backend.get_media.assert_not_called()
    p._stt.transcribe.assert_called_once_with(b"ogg-bytes", "audio/ogg")
    p._backend.post_analysis.assert_called_once()
    payload = p._backend.post_analysis.call_args[0][1]
    assert payload["transcript"] == "o preco do herbicida esta alto"


def test_audio_prefers_asset_content_type():
    p = _pipeline()
    p._backend.get_context.return_value = _audio_ctx(
        asset={"id": "a1", "contentType": "audio/mp4"}
    )
    p._backend.get_message_media.return_value = (b"mp4", "application/octet-stream")
    p._stt.transcribe.return_value = "ok"

    p.process("m1")

    p._stt.transcribe.assert_called_once_with(b"mp4", "audio/mp4")


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


if __name__ == "__main__":
    test_audio_without_asset_fetches_by_message_id()
    test_audio_prefers_asset_content_type()
    test_audio_media_fetch_failure_does_not_publish()
    print("pipeline ok")
