"""STT: mime WhatsApp e fail-open sem chave."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.stt.gemini_stt import GeminiStt, stt_mime


def test_stt_mime_strips_opus_codec():
    assert stt_mime("audio/ogg; codecs=opus") == "audio/ogg"
    assert stt_mime("audio/mp4") == "audio/mp4"
    assert stt_mime("audio/mpeg") == "audio/mp3"
    assert stt_mime("application/octet-stream") == "audio/ogg"


def test_no_api_key_returns_empty():
    stt = GeminiStt(Settings(gemini_api_key=None))
    assert stt.transcribe(b"x", "audio/ogg") == ""


if __name__ == "__main__":
    test_stt_mime_strips_opus_codec()
    test_no_api_key_returns_empty()
    print("stt ok")
