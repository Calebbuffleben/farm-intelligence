"""STT: mime WhatsApp e fail-open sem chave."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.stt.gemini_stt import GeminiStt, stt_mime
from src.stt.audio_convert import audio_kind, needs_pcm_convert, to_stt_audio


def test_stt_mime_strips_opus_codec():
    assert stt_mime("audio/ogg; codecs=opus") == "audio/ogg"
    assert stt_mime("audio/mp4") == "audio/mp4"
    assert stt_mime("audio/mpeg") == "audio/mp3"
    assert stt_mime("application/octet-stream") == "audio/ogg"


def test_no_api_key_returns_empty():
    stt = GeminiStt(Settings(gemini_api_key=None))
    assert stt.transcribe(b"x", "audio/ogg") == ""


def test_opus_needs_convert():
    assert needs_pcm_convert("audio/ogg; codecs=opus") is True
    assert needs_pcm_convert("audio/mp4") is False
    assert needs_pcm_convert("audio/wav") is False


def test_ogg_magic():
    assert audio_kind(b"OggS....") == "ogg"
    assert audio_kind(b"RIFF....WAVE") == "wav"


def test_to_stt_passthrough_wav():
    wav = b"RIFF" + b"\x00" * 8
    out, mime = to_stt_audio(wav, "audio/wav")
    assert out == wav
    assert mime == "audio/wav"


if __name__ == "__main__":
    test_stt_mime_strips_opus_codec()
    test_no_api_key_returns_empty()
    test_opus_needs_convert()
    test_ogg_magic()
    test_to_stt_passthrough_wav()
    print("stt ok")
