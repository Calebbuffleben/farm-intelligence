"""Fase 0 — converte export .txt do WhatsApp em conversas.jsonl.

Uso:
  python fase0_parse_whatsapp_txt.py --input export1.txt export2.txt \
      --rtv-name "João" --output conversas.jsonl

Formato aceito (export padrão Android/iOS pt-BR):
  DD/MM/YYYY HH:MM - Nome: mensagem
  [DD/MM/YYYY, HH:MM:SS] Nome: mensagem

Áudios/mídia aparecem como "<Arquivo de mídia oculto>" — viram type=AUDIO_PLACEHOLDER.
Para a validação da Fase 0, transcreva manualmente os áudios relevantes e substitua
o placeholder pelo texto entre [AUDIO: ...] antes de rodar o validador.
"""

import argparse
import json
import re
import sys
from pathlib import Path

LINE_PATTERNS = [
    # 12/03/2026 14:05 - João: mensagem
    re.compile(
        r"^(?P<date>\d{2}/\d{2}/\d{4})[,]?\s+(?P<time>\d{2}:\d{2})(?::\d{2})?\s+-\s+(?P<name>[^:]+):\s?(?P<text>.*)$"
    ),
    # [12/03/2026, 14:05:33] João: mensagem
    re.compile(
        r"^\[(?P<date>\d{2}/\d{2}/\d{4}),?\s+(?P<time>\d{2}:\d{2})(?::\d{2})?\]\s+(?P<name>[^:]+):\s?(?P<text>.*)$"
    ),
]

MEDIA_MARKERS = (
    "<Arquivo de mídia oculto>",
    "<Media omitted>",
    "áudio ocultado",
)


def parse_file(path: Path, rtv_name: str) -> dict:
    messages = []
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        matched = None
        for pat in LINE_PATTERNS:
            m = pat.match(raw)
            if m:
                matched = m
                break
        if matched:
            if current:
                messages.append(current)
            name = matched.group("name").strip()
            text = matched.group("text").strip()
            direction = "OUT" if rtv_name.lower() in name.lower() else "IN"
            msg_type = "TEXT"
            if any(marker in text for marker in MEDIA_MARKERS):
                msg_type = "AUDIO_PLACEHOLDER"
            current = {
                "ts": f"{matched.group('date')} {matched.group('time')}",
                "direction": direction,
                "type": msg_type,
                "text": text,
            }
        elif current:
            # continuação de mensagem multi-linha
            current["text"] += "\n" + raw.strip()
    if current:
        messages.append(current)
    for i, m in enumerate(messages):
        m["index"] = i
    return {
        "conversation_id": path.stem,
        "source_file": path.name,
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--rtv-name", required=True, help="Nome do RTV como aparece no export")
    ap.add_argument("--output", default="conversas.jsonl")
    args = ap.parse_args()

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        for input_path in args.input:
            convo = parse_file(Path(input_path), args.rtv_name)
            placeholders = sum(
                1 for m in convo["messages"] if m["type"] == "AUDIO_PLACEHOLDER"
            )
            print(
                f"{input_path}: {len(convo['messages'])} mensagens, "
                f"{placeholders} áudios sem transcrição"
            )
            fh.write(json.dumps(convo, ensure_ascii=False) + "\n")
    print(f"gravado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
