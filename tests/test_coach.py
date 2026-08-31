"""Copilot pós-STT: sem Gemini não bloqueia a esteira."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.extractor.coach import CoachNote, CopilotCoach


def test_no_api_key_skips_note():
    coach = CopilotCoach(Settings(gemini_api_key=None))
    assert coach.note("o produtor pediu 5% de desconto") is None


def test_blank_transcript_skips_note():
    coach = CopilotCoach(Settings(gemini_api_key=None))
    assert coach.note("   ") is None


def test_coach_note_schema_roundtrip():
    parsed = CoachNote.model_validate(
        {"note": "Hesitou no preço; não invente desconto.", "tone": "alerta"}
    )
    assert parsed.tone == "alerta"
    assert "preço" in parsed.note


if __name__ == "__main__":
    test_no_api_key_skips_note()
    test_blank_transcript_skips_note()
    test_coach_note_schema_roundtrip()
    print("coach ok")
