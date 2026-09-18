"""Quem fala chega rotulado no prompt do Gemini."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.prompts import build_user_prompt, speaker_label


def test_speaker_label():
    assert speaker_label("IN") == "PRODUTOR (cliente)"
    assert speaker_label("OUT") == "RTV (vendedor da revenda)"
    assert speaker_label("") == "DESCONHECIDO"


def test_prompt_labels_producer_and_rtv():
    text = build_user_prompt(
        {"producer": {"name": "João"}, "farms": []},
        [],
        [
            {"index": 0, "direction": "IN", "text": "o preço está alto", "ts": "t0"},
            {"index": 1, "direction": "OUT", "text": "posso parcelar em 3", "ts": "t1"},
        ],
        target_index=0,
    )
    assert "PRODUTOR (cliente): o preço está alto" in text
    assert "RTV (vendedor da revenda): posso parcelar em 3" in text
    assert "NUNCA trate fala do RTV" not in text  # isso está no system, não no user
    assert "extraia fatos SOMENTE da mensagem [0]" in text


def test_audio_target_names_producer_speaker():
    text = build_user_prompt(
        {"farms": []},
        [],
        [{"index": 0, "direction": "IN", "text": "[áudio]", "ts": ""}],
        target_index=0,
        audio_target=True,
    )
    assert "falado por PRODUTOR (cliente)" in text
    assert "Se o falante é PRODUTOR" in text


if __name__ == "__main__":
    test_speaker_label()
    test_prompt_labels_producer_and_rtv()
    test_audio_target_names_producer_speaker()
    print("prompts speaker ok")
