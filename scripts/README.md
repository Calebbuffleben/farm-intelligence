# Fase 0 — validação offline do extrator (kill gate)

Antes de escrever qualquer linha do canal WABA, provar que o Gemini extrai fatos
comerciais e resolve fazendas em conversas REAIS de WhatsApp do agro. Se falhar
aqui, o produto zero-input não se sustenta — parar e reavaliar.

## Passo a passo

1. **Conseguir dados reais**: 10–20 conversas exportadas do WhatsApp (RTV ↔ produtor)
   de uma revenda parceira, com a carteira (produtores, fazendas, culturas).
   Casos obrigatórios no conjunto: conversa longa (meses), áudio citando 2+ fazendas,
   referências implícitas ("aquele produto", "a área de cima").

2. **Converter export → JSONL**:
   ```bash
   python fase0_parse_whatsapp_txt.py --input export*.txt --rtv-name "João" --output conversas.jsonl
   ```
   Transcrever manualmente os áudios relevantes: substituir `type: AUDIO_PLACEHOLDER`
   por `type: AUDIO_TRANSCRIBED` e colocar a transcrição em `text`.

3. **Montar a carteira** (`carteira.json`):
   ```json
   {"producers": [{"id": "prod-1", "name": "Marcos Silva", "phones": ["+5566..."],
     "farms": [{"id": "farm-1", "name": "Fazenda Chapadão", "region": "Sorriso-MT",
       "crops": [{"crop": "soja", "season": "2026/27"}]}]}]}
   ```

4. **Gerar e preencher o gabarito** (anotação humana — o "ouro"):
   ```bash
   python fase0_gabarito_template.py --conversas conversas.jsonl --output gabarito.jsonl
   ```

5. **Rodar a validação**:
   ```bash
   export GEMINI_API_KEY=...
   pip install -r ../requirements.txt
   python fase0_validate.py --carteira carteira.json --conversas conversas.jsonl \
       --gabarito gabarito.jsonl --report fase0_report.json
   ```

## Kill gate

| Métrica | Limite | Significado |
|---|---|---|
| `resolucao_correta` | ≥ 70% | fatos com a fazenda certa |
| `resolucao_errada` | < 10% | fazenda ERRADA é pior que "não sei" — não sei vai para a fila unknown |
| `recall_fatos` | ≥ 60% | fatos do gabarito capturados (por kind) |

- Falhou? Iterar o prompt (`src/extractor/prompts.py`) até v3.
- Falhou no v3? Decisão de negócio: o zero-input não fecha — reavaliar antes da Fase 2.
- Exit code do script: 0 = passou, 1 = falhou (usável em CI).
