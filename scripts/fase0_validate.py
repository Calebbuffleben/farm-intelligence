"""Fase 0 — validação offline do extrator contra gabarito humano (KILL GATE).

Uso:
  export GEMINI_API_KEY=...
  python fase0_validate.py --carteira carteira.json --conversas conversas.jsonl \
      --gabarito gabarito.jsonl --report report.json

Entradas:
  carteira.json    {"producers": [{"id","name","phones":[...],
                     "farms":[{"id","name","region","crops":[{"crop","season"}]}]}]}
  conversas.jsonl  1 conversa/linha: {"conversation_id","producer_id",
                     "messages":[{"index","ts","direction","type","text"}]}
                   (producer_id opcional se o gabarito tiver)
  gabarito.jsonl   1 conversa/linha: {"conversation_id","producer_id",
                     "facts":[{"kind","subtype","farm_id","evidence_message_index"}]}

Métricas e KILL GATE (documentado no plano):
  - resolucao_correta  >= 0.70  (fatos previstos com farm_id igual ao gabarito,
                                 sobre fatos do gabarito com farm_id)
  - resolucao_errada   <  0.10  (farm_id previsto != gabarito e != null;
                                 "não sei" (null/unknown) NÃO conta como erro)
  - recall_fatos       >= 0.60  (fatos do gabarito capturados por kind)
  Se falhar com prompt v2/v3, o produto na forma zero-input não se sustenta —
  reavaliar antes da Fase 2 (decisão de negócio, não técnica).

Matching fato previsto x gabarito: mesmo kind + mesma farm_id (null casa com null).
Subtype é reportado, mas não reprova o gate no v1.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings  # noqa: E402
from src.extractor.gemini_client import FactExtractor  # noqa: E402

KILL_GATE = {
    "resolucao_correta_min": 0.70,
    "resolucao_errada_max": 0.10,
    "recall_fatos_min": 0.60,
}


def load_jsonl(path: str) -> list:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def carteira_for_producer(carteira: dict, producer_id: str) -> dict:
    for producer in carteira["producers"]:
        if producer["id"] == producer_id:
            return {
                "producer": {"id": producer["id"], "name": producer["name"]},
                "farms": producer["farms"],
            }
    raise KeyError(f"producer_id {producer_id} não está na carteira")


def match_facts(predicted: list, expected: list) -> dict:
    """Matching guloso por (kind, farm_id). Retorna contadores."""
    remaining = list(expected)
    matched = 0
    for p in predicted:
        for e in remaining:
            if p["kind"] == e["kind"] and (p.get("farm_id") or None) == (
                e.get("farm_id") or None
            ):
                matched += 1
                remaining.remove(e)
                break
    return {
        "matched": matched,
        "missed": len(remaining),
        "extra": len(predicted) - matched,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carteira", required=True)
    ap.add_argument("--conversas", required=True)
    ap.add_argument("--gabarito", required=True)
    ap.add_argument("--report", default="fase0_report.json")
    args = ap.parse_args()

    carteira = json.loads(Path(args.carteira).read_text(encoding="utf-8"))
    conversas = {c["conversation_id"]: c for c in load_jsonl(args.conversas)}
    gabaritos = load_jsonl(args.gabarito)

    settings = Settings.from_env()
    extractor = FactExtractor(settings)

    # Resolução (só sobre fatos do gabarito com farm_id definida)
    res_total = res_correct = res_wrong = res_unknown = 0
    # Recall/precision de fatos por kind
    kind_stats: dict = defaultdict(lambda: {"matched": 0, "missed": 0, "extra": 0})
    per_conversation = []

    for gab in gabaritos:
        cid = gab["conversation_id"]
        convo = conversas.get(cid)
        if convo is None:
            print(f"AVISO: conversa {cid} do gabarito não está em --conversas; pulando")
            continue
        producer_id = gab.get("producer_id") or convo.get("producer_id")
        ctx = carteira_for_producer(carteira, producer_id)

        messages = [
            m for m in convo["messages"] if m["type"] in ("TEXT", "AUDIO_TRANSCRIBED")
        ]
        placeholders = len(convo["messages"]) - len(messages)
        if placeholders:
            print(f"{cid}: {placeholders} áudios sem transcrição ignorados")

        result = extractor.extract(ctx, [], messages)
        predicted = [f.model_dump() for f in result.facts]
        expected = gab["facts"]

        # resolução
        for e in expected:
            if not e.get("farm_id"):
                continue
            res_total += 1
            same_kind = [p for p in predicted if p["kind"] == e["kind"]]
            hit = next(
                (p for p in same_kind if p.get("farm_id") == e["farm_id"]), None
            )
            if hit:
                res_correct += 1
            elif any(p.get("farm_id") for p in same_kind):
                res_wrong += 1
            else:
                res_unknown += 1

        # fatos por kind
        by_kind_pred = defaultdict(list)
        for p in predicted:
            by_kind_pred[p["kind"]].append(p)
        by_kind_exp = defaultdict(list)
        for e in expected:
            by_kind_exp[e["kind"]].append(e)
        for kind in set(by_kind_pred) | set(by_kind_exp):
            m = match_facts(by_kind_pred.get(kind, []), by_kind_exp.get(kind, []))
            for k, v in m.items():
                kind_stats[kind][k] += v

        per_conversation.append(
            {
                "conversation_id": cid,
                "predicted": predicted,
                "unknowns": [u.model_dump() for u in result.unknowns],
                "expected": expected,
            }
        )
        print(f"{cid}: {len(predicted)} fatos previstos, {len(expected)} no gabarito")

    total_expected = sum(s["matched"] + s["missed"] for s in kind_stats.values())
    total_matched = sum(s["matched"] for s in kind_stats.values())
    recall = total_matched / total_expected if total_expected else 0.0
    res_correct_rate = res_correct / res_total if res_total else 0.0
    res_wrong_rate = res_wrong / res_total if res_total else 0.0

    gate = {
        "resolucao_correta": {
            "valor": round(res_correct_rate, 3),
            "minimo": KILL_GATE["resolucao_correta_min"],
            "passou": res_correct_rate >= KILL_GATE["resolucao_correta_min"],
        },
        "resolucao_errada": {
            "valor": round(res_wrong_rate, 3),
            "maximo": KILL_GATE["resolucao_errada_max"],
            "passou": res_wrong_rate < KILL_GATE["resolucao_errada_max"],
        },
        "recall_fatos": {
            "valor": round(recall, 3),
            "minimo": KILL_GATE["recall_fatos_min"],
            "passou": recall >= KILL_GATE["recall_fatos_min"],
        },
    }
    passed = all(g["passou"] for g in gate.values())

    report = {
        "kill_gate": gate,
        "kill_gate_passou": passed,
        "resolucao": {
            "total": res_total,
            "correta": res_correct,
            "errada": res_wrong,
            "nao_resolvida": res_unknown,
        },
        "fatos_por_kind": {k: dict(v) for k, v in kind_stats.items()},
        "conversas": per_conversation,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== KILL GATE =====")
    for name, g in gate.items():
        status = "PASSOU" if g["passou"] else "FALHOU"
        print(f"  {name}: {g['valor']} ({status})")
    print(f"\nVeredito: {'PASSOU — seguir para Fase 2' if passed else 'FALHOU — iterar prompt ou reavaliar produto'}")
    print(f"Relatório completo: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
