"""Fallback de modelo: 503/429/404 não podem ACK um Card vazio."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import errors

from src.config.settings import Settings
from src.extractor.gemini_client import FALLBACK_MODEL, FactExtractor
from src.extractor.schema import ExtractionResult


def _extractor(model: str = "gemini-3.8-flash"):
    with patch("src.extractor.gemini_client.genai.Client") as cls:
        mock = MagicMock()
        cls.return_value = mock
        ext = FactExtractor(Settings(gemini_api_key="k", extractor_model=model))
        ext._client = mock
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


def test_503_retries_fallback_model():
    ext, client = _extractor()
    client.models.generate_content.side_effect = [_server_error(503), _ok_response()]
    result = ext.extract({}, [], _messages())
    assert result.session_summary == "resumo"
    assert ext._model == FALLBACK_MODEL
    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models == ["gemini-3.8-flash", FALLBACK_MODEL]


def test_429_retries_fallback_model():
    ext, client = _extractor()
    client.models.generate_content.side_effect = [_server_error(429), _ok_response("ok")]
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
    assert ext._model == FALLBACK_MODEL


def test_both_models_503_raises():
    ext, client = _extractor()
    client.models.generate_content.side_effect = [
        _server_error(503),
        _server_error(503),
    ]
    try:
        ext.extract({}, [], _messages())
    except errors.ServerError as exc:
        assert exc.code == 503
    else:
        raise AssertionError("esperava ServerError")
    assert client.models.generate_content.call_count == 2


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


if __name__ == "__main__":
    test_503_retries_fallback_model()
    test_429_retries_fallback_model()
    test_404_retries_fallback_model()
    test_both_models_503_raises()
    test_400_does_not_fallback()
    print("gemini_client ok")
