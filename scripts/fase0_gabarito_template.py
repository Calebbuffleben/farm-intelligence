"""Fase 0 — gera o template de gabarito para anotação manual.

Uso:
  python fase0_gabarito_template.py --conversas conversas.jsonl --output gabarito.jsonl

Depois, para cada conversa, preencha `facts` manualmente com o que um analista
humano extrairia (o "ouro"). Campos por fato:
  kind: OBJECAO | RISCO | OPORTUNIDADE | FOLLOWUP | CONCORRENTE
  subtype: preco | credito_prazo | confianca_produto | logistica | janela_climatica |
           concorrente_generico | concorrente_marca | assistencia_tecnica | volume_pedido | outro
  farm_id: id da fazenda na carteira (ou null se nem humano resolve)
  evidence_message_index: índice da mensagem que sustenta o fato
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversas", required=True)
    ap.add_argument("--output", default="gabarito.jsonl")
    args = ap.parse_args()

    out = Path(args.output)
    with Path(args.conversas).open(encoding="utf-8") as fin, out.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            convo = json.loads(line)
            template = {
                "conversation_id": convo["conversation_id"],
                "producer_id": "PREENCHER",
                "facts": [
                    {
                        "kind": "PREENCHER",
                        "subtype": "PREENCHER",
                        "farm_id": None,
                        "evidence_message_index": -1,
                        "nota": "descreva o fato em uma frase",
                    }
                ],
            }
            fout.write(json.dumps(template, ensure_ascii=False) + "\n")
    print(f"template gravado: {out} — preencha manualmente antes de validar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
