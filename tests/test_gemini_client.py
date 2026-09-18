"""Fallback de modelo: 503/429/404 não podem ACK um Card vazio."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import errors

from src.config.settings import Settings
from src.extractor.gemini_client import (
    FALLBACK_MODELS,
    FactExtractor,
    model_chain,
)
from src.extractor.schema import ExtractionResult

PRIMARY = "gemini-3.8-flash"


def _extractor(model: str = PRIMARY):
    with patch("src.extractor.gemini_client.genai.Client") as cls:
        mock = MagicMock()
        cls.return_value = mock
        ext = FactExtractor(Settings(gemini_api_key="k", extractor_model=model))
        ext._client = mock
        ext._sleep = lambda _seconds: None
        return ext, mock


def _ok_response(summary: str = "resumo"):
    return SimpleNamespace(
        parsed=ExtractionResult(session_summary=summary, facts=[], unknowns=[]),
        text="",
    )


def _server_error(code: int = 503):
    return errors.ServerError(
        code,
        {"error": {"message": "high demand", "status": "UNAVAILABLE"}},
        None,
    )


def _messages():
    return [{"index": 0, "direction": "IN", "text": "preciso de cotação", "ts": ""}]


def test_model_chain_dedupes_primary():
    chain = model_chain("gemini-2.5-flash")
    assert chain[0] == "gemini-2.5-flash"
    assert chain == list(dict.fromkeys(chain))


def test_503_retries_stable_flash():
    ext, client = _extractor()
    client.models.generate_content.side_effect = [
        _server_error(503),
        _ok_response(),
    ]
    result = ext.extract({}, [], _messages())
    assert result.session_summary == "resumo"
    assert ext._model == FALLBACK_MODELS[0]
    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models == [PRIMARY, FALLBACK_MODELS[0]]


def test_503_walks_full_chain_then_raises():
    ext, client = _extractor()
    chain = model_chain(PRIMARY)
    client.models.generate_content.side_effect = [_server_error(503)] * len(chain)
    try:
        ext.extract({}, [], _messages())
    except errors.ServerError as exc:
        assert exc.code == 503
    else:
        raise AssertionError("esperava ServerError")
    assert client.models.generate_content.call_count == len(chain)
    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models == chain


def test_429_retries_fallback_model():
    ext, client = _extractor()
    client.models.generate_content.side_effect = [
        _server_error(429),
        _ok_response("ok"),
    ]
    result = ext.extract({}, [], _messages())
    assert result.session_summary == "ok"


def test_404_retries_fallback_model():
    ext, client = _extractor()
    not_found = errors.ClientError(
        404,
        {"error": {"message": "not found", "status": "NOT_FOUND"}},
        None,
    )
    client.models.generate_content.side_effect = [not_found, _ok_response("ok")]
    result = ext.extract({}, [], _messages())
    assert result.session_summary == "ok"
    assert ext._model == FALLBACK_MODELS[0]


def test_400_does_not_fallback():
    ext, client = _extractor()
    bad = errors.ClientError(
        400,
        {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}},
        None,
    )
    client.models.generate_content.side_effect = bad
    try:
        ext.extract({}, [], _messages())
    except errors.ClientError as exc:
        assert exc.code == 400
    else:
        raise AssertionError("esperava ClientError")
    assert client.models.generate_content.call_count == 1


def test_truncated_json_tries_next_model():
    ext, client = _extractor("gemini-2.5-flash")
    truncated = SimpleNamespace(
        parsed=None,
        text='{\n  "session_summary": "o produtor pediu cotação esta semana, mas está',
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    client.models.generate_content.side_effect = [truncated, _ok_response("ok")]
    result = ext.extract({}, [], _messages())
    assert result.session_summary == "ok"
    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models[0] == "gemini-2.5-flash"
    assert models[1] == "gemini-2.5-flash-lite"


def test_extract_sends_audio_in_same_call():
    ext, client = _extractor()
    client.models.generate_content.return_value = _ok_response()
    wav = b"RIFF" + b"\x00" * 12
    ext.extract({}, [], _messages(), audio=wav, audio_mime="audio/wav")
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 2
    prompt = contents[1]
    text = getattr(prompt, "text", None) or str(prompt)
    assert "ÁUDIO DA MENSAGEM ALVO" in text


if __name__ == "__main__":
    test_model_chain_dedupes_primary()
    test_503_retries_stable_flash()
    test_503_walks_full_chain_then_raises()
    test_429_retries_fallback_model()
    test_404_retries_fallback_model()
    test_400_does_not_fallback()
    test_truncated_json_tries_next_model()
    test_extract_sends_audio_in_same_call()
    print("gemini_client ok")
