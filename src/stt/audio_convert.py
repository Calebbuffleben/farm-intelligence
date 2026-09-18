"""Converte PTT WhatsApp (OGG/Opus) para WAV 16 kHz — o Gemini não transcreve Opus."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_PASSTHROUGH = {
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/aac",
    "audio/flac",
}


def needs_pcm_convert(mime: str) -> bool:
    base = (mime or "").split(";")[0].strip().lower()
    return base not in _PASSTHROUGH


def to_stt_audio(audio: bytes, mime: str) -> tuple[bytes, str]:
    """Devolve (bytes, mime) prontos para Part.from_bytes.

    OGG/Opus, webm e octet-stream passam pelo ffmpeg. Sem ffmpeg, devolve o
    original e o Gemini costuma responder vazio — o caller trata.
    """
    raw_mime = (mime or "audio/ogg").split(";")[0].strip().lower() or "audio/ogg"
    if not audio:
        return audio, raw_mime
    if not needs_pcm_convert(raw_mime):
        return audio, raw_mime
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning(
            "ffmpeg ausente — STT no mime original %s (%d bytes); PTT Opus tende a vir vazio",
            raw_mime,
            len(audio),
        )
        return audio, raw_mime
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            input=audio,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg não rodou: %s — mime original %s", exc, raw_mime)
        return audio, raw_mime
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
        logger.warning("ffmpeg falhou mime=%s: %s", raw_mime, err)
        return audio, raw_mime
    logger.info(
        "STT convert %s→wav in=%d out=%d",
        raw_mime,
        len(audio),
        len(proc.stdout),
    )
    return proc.stdout, "audio/wav"


def audio_kind(audio: bytes) -> Optional[str]:
    if audio[:4] == b"OggS":
        return "ogg"
    if audio[:4] == b"fLaC":
        return "flac"
    if audio[:4] == b"RIFF":
        return "wav"
    if audio[4:8] == b"ftyp":
        return "mp4"
    if audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb":
        return "mp3"
    return None
