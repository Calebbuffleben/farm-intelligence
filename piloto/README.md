# Dados do piloto — Fase 0

O kill gate do extrator **só vale com conversas reais**. Não há dataset sintético
que substitua isso: o produto quebra em anáfora, áudio com vento e duas fazendas
na mesma mensagem.

Coloque os arquivos neste diretório (eles estão no `.gitignore` — não versionar PII).

## O que enviar (mínimo para o gate)

1. **10–20 exports `.txt` do WhatsApp** (Android/iOS: conversa → exportar sem mídia
   ou com mídia; o parser lê o texto). Preferível: RTV ↔ produtor, não grupo interno.
2. **Carteira** daqueles produtores: nome, telefones, fazendas (nome + região),
   culturas/safras. Planilha ou JSON no formato de `carteira.example.json`.
3. **Nome do RTV** como aparece no export (ex.: `João`).

Casos que o conjunto **precisa ter** (senão o gate mente):
- conversa longa (semanas/meses na mesma thread)
- um áudio (ou texto) citando **2+ fazendas**
- referência implícita ("aquele produto", "a área de cima")

Áudios no export viram placeholder. Transcreva os relevantes à mão no JSONL
depois do parser (`type: AUDIO_TRANSCRIBED`).

## Como o RTV exporta (WhatsApp)

No celular: conversa → ⋮ → Mais → Exportar conversa → Sem mídia (mais leve).
Manda o `.txt` (ou o zip) para quem for anotar o gabarito.

## Depois que os arquivos estiverem aqui

```bash
cd farm/intelligence/scripts
python fase0_parse_whatsapp_txt.py --input ../piloto/exports/*.txt \
  --rtv-name "João" --output ../piloto/conversas.jsonl
python fase0_gabarito_template.py --conversas ../piloto/conversas.jsonl \
  --output ../piloto/gabarito.jsonl
# preencher gabarito.jsonl + carteira.json
export GEMINI_API_KEY=...
python fase0_validate.py --carteira ../piloto/carteira.json \
  --conversas ../piloto/conversas.jsonl --gabarito ../piloto/gabarito.jsonl \
  --report ../piloto/fase0_report.json
```
