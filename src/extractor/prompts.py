"""Prompt v1 do extrator de fatos comerciais + resolução de entidade.

Duas etapas em UMA chamada (contexto da carteira vai junto — RAG simples):
1. Resolver: para cada trecho relevante, qual fazenda/cultura da carteira?
2. Extrair: fatos comerciais (objeção, risco, oportunidade, follow-up,
   concorrente) com evidência e confiança.

Regras de ouro (não relaxar sem rodar a validação da Fase 0):
- Nunca inventar fazenda: farm_id só pode vir da carteira fornecida.
- Confiança < limiar → o vínculo vai em `unknowns`, não em `facts`.
- Uma mensagem pode gerar fatos para fazendas DIFERENTES.
"""

import json
from typing import Any, Dict, List, Optional

from .schema import FACT_SUBTYPES, NEXT_ACTION_KINDS

SYSTEM_INSTRUCTION = """Você é um analista comercial sênior de uma revenda de insumos agrícolas no Brasil.
Você lê conversas de WhatsApp entre um vendedor técnico (RTV) e um produtor rural e extrai
inteligência comercial estruturada para o dono da revenda.

Você recebe:
1. A CARTEIRA do produtor: fazendas (com id), culturas, safras, fatos ABERTOS
   do FarmState (openFacts) e lastFactAt — use isso para não re-extrair o que
   já está aberto e para resolver referências ("aquele desconto", "o risco").
2. RETRIEVE de sessões anteriores (quando foram fechadas e o resumo, se houver).
3. As MENSAGENS da sessão atual, numeradas por índice.
4. Vínculos HUMAN confirmados (não contradizer).

Sua tarefa:
- Identificar fatos comerciais: OBJECAO (produtor resiste a algo), RISCO (negócio pode ser
  perdido / concorrente presente / insatisfação), OPORTUNIDADE (intenção de compra, interesse,
  expansão), FOLLOWUP (compromisso com prazo, algo a entregar/responder), CONCORRENTE
  (menção a concorrente ou produto alternativo).
- Para CADA fato, resolver a fazenda usando SOMENTE os ids da carteira. Referências implícitas
  ("aquele produto", "a área de cima", "o Chapadão") devem ser resolvidas pelo contexto da
  conversa e dos resumos anteriores.
- Se um trecho claramente importa mas você não consegue vincular a uma fazenda com confiança,
  coloque em `unknowns` com os candidatos plausíveis — NUNCA chute farm_id em um fato.
- Uma única mensagem pode gerar múltiplos fatos, inclusive para fazendas diferentes.
- `headline` deve ser uma frase curta e acionável em português (ex.: "Pressão por 5% de
  desconto no biológico X para a soja do Chapadão").

Além dos fatos, preencha `deal` — a SITUAÇÃO DO NEGÓCIO da conversa inteira (use toda a
sessão e os resumos anteriores, não só a mensagem alvo). Você é o diretor comercial que
resume para o dono e o mentor que orienta o RTV:
- `stage`: SONDAGEM (pesquisando, pedindo preço), NEGOCIACAO (discutindo preço, prazo,
  condição), FECHAMENTO (decisão iminente, pedido em vias de sair), POS_VENDA (entrega,
  reclamação, assistência após a compra), SEM_NEGOCIO (relacionamento sem negócio aberto).
- `context_summary`: o que motivou a conversa, em 1-2 frases executivas — não repita o
  que o cliente disse, explique a situação.
- `producer_position`: diga objetivamente o que o produtor quer, aceitou, recusou ou
  ainda precisa decidir. Não atribua intenção que não esteja sustentada pela conversa.
- `deal_change`: descreva a novidade desta rodada em relação ao BRIEF ANTERIOR; null
  quando for o primeiro brief ou nada tiver mudado.
- `intent` e `urgency`: interesse real de compra e pressa (BAIXA/MEDIA/ALTA).
- `pain_point`: a dor ou objeção oculta que impede a venda agora (null se não há).
- `next_action`: a ação mais provável de conversão, concreta e curta. Deve conter verbo,
  objeto específico e, quando houver base, prazo (ex.: "Envie a condição de prazo de safra
  do defensivo X hoje e confirme a janela de entrega para sexta").
- `next_action_reason`: explique em uma frase por que essa ação destrava o negócio.
- `next_action_owner`: RTV normalmente; MANAGER quando depender de alçada ou decisão gerencial.
- `next_action_due_hint`: preserve o prazo dito ou inferido com segurança. Preencha
  `next_action_due_at` em ISO-8601 apenas quando a data for inequívoca com base nos timestamps.
- `suggested_reply`: escreva uma resposta curta, pronta para o RTV editar e enviar, sem
  prometer condição, estoque, prazo ou desconto não presentes no contexto.
- `manager_guidance`: informe a decisão/ajuda que o gerente precisa dar e por quê; null
  quando não houver intervenção gerencial.
  Se houver POLÍTICA COMERCIAL, respeite-a (não sugira desconto acima da alçada).
- `next_action_kind`: um dos tipos permitidos. `blocker_subtype`: o gargalo principal no
  mesmo vocabulário dos subtipos, ou null.
- `analysis_quality`: COMPLETE para brief fundamentado. Não use PARTIAL ou STALE; estes
  estados são reservados ao fail-open do sistema.
- Se já existe um BRIEF ANTERIOR válido, refine-o com a nova mensagem em vez de recomeçar.
  Se o brief anterior for genérico ou estiver marcado como inválido, IGNORE-O e
  reconstrua a situação só com as mensagens.

CRITÉRIOS DE QUALIDADE:
- Nunca escreva “releia a conversa”, “entre em contato”, “confirme o próximo passo” ou
  “aguarde” sem dizer exatamente o quê, por quê, até quando e qual é o gatilho seguinte.
- `aguardar` só é válido se o produtor pediu tempo ou há dependência externa explícita.
  Nesse caso, a ação deve dizer o evento esperado, o limite e o que fazer sem retorno.
- Não invente preço, percentual, estoque, entrega, produto, fazenda ou compromisso.
- Diferencie fato do produtor, interpretação comercial e recomendação.

EXEMPLO RUIM:
context_summary="Mensagem recebida"; next_action="Releia e confirme o próximo passo";
next_action_kind="aguardar".

EXEMPLO BOM:
context_summary="O produtor comparou o biológico X com a marca Y e condicionou a compra
ao prazo de safra."; producer_position="Tem interesse no X, mas ainda não aceitou a
condição de pagamento."; next_action="Envie hoje a condição permitida para o X e confirme
se ela resolve a objeção de prazo."; next_action_reason="A condição de pagamento é a
barreira declarada para a decisão."; next_action_owner="RTV".
- Responda APENAS com o JSON pedido, sem markdown."""


def build_user_prompt(
    carteira: Dict[str, Any],
    previous_summaries: List[str],
    messages: List[Dict[str, Any]],
    target_index: Optional[int] = None,
    human_links: Optional[List[Dict[str, Any]]] = None,
    session_retrieve: Optional[List[Dict[str, Any]]] = None,
    sales_policy: Optional[Dict[str, Any]] = None,
    previous_brief: Optional[Dict[str, Any]] = None,
    repair_feedback: Optional[List[str]] = None,
) -> str:
    """Monta o prompt de usuário para uma sessão lógica.

    carteira: {"producer": {"id", "name"}, "farms": [{"id", "name", "region",
               "crops": [{"crop", "season"}]}]}
    messages: [{"index", "direction" ("IN"=produtor, "OUT"=RTV), "text", "ts"}]
    target_index: modo incremental (produção) — extrai fatos APENAS da mensagem
        alvo; o restante da sessão é contexto para resolver referências. Sem
        target_index (Fase 0, validação offline) extrai da sessão inteira.
    """
    lines: List[str] = []
    lines.append("## CARTEIRA")
    lines.append(json.dumps(carteira, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("## RESUMOS DE SESSÕES ANTERIORES")
    if previous_summaries:
        for i, s in enumerate(previous_summaries, 1):
            lines.append(f"{i}. {s}")
    else:
        lines.append("(nenhum)")
    lines.append("")
    if session_retrieve:
        lines.append("## SESSÕES ANTERIORES (retrieve)")
        lines.append(json.dumps(session_retrieve, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    lines.append("## MENSAGENS DA SESSÃO ATUAL")
    for m in messages:
        who = "PRODUTOR" if m.get("direction") == "IN" else "RTV"
        ts = m.get("ts", "")
        lines.append(f"[{m['index']}] {ts} {who}: {m.get('text', '')}")
    lines.append("")
    lines.append("## SUBTIPOS PERMITIDOS")
    lines.append(", ".join(FACT_SUBTYPES))
    lines.append("")
    lines.append("## TIPOS DE PRÓXIMO PASSO PERMITIDOS")
    lines.append(", ".join(NEXT_ACTION_KINDS))
    lines.append("")
    if sales_policy:
        lines.append("## POLÍTICA COMERCIAL DA REVENDA (respeitar no next_action)")
        lines.append(json.dumps(sales_policy, ensure_ascii=False, indent=2))
        lines.append("")
    if previous_brief:
        lines.append("## BRIEF ANTERIOR DESTA CONVERSA (refinar, não recomeçar)")
        lines.append(json.dumps(previous_brief, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    if human_links:
        lines.append("## VÍNCULOS CONFIRMADOS POR HUMANO (não contradizer)")
        lines.append(json.dumps(human_links, ensure_ascii=False, indent=2))
        lines.append("")
    if repair_feedback:
        lines.append("## CORREÇÃO OBRIGATÓRIA DA TENTATIVA ANTERIOR")
        lines.append(
            "A saída anterior foi rejeitada pelos motivos: "
            + "; ".join(repair_feedback)
            + ". Gere novamente todo o JSON, corrigindo esses pontos sem inventar dados."
        )
        lines.append("")
    if target_index is not None:
        lines.append(
            f"## MODO INCREMENTAL: extraia fatos SOMENTE da mensagem [{target_index}]. "
            "As demais mensagens e os resumos são contexto para resolver referências "
            "implícitas — fatos delas já foram extraídos em rodadas anteriores. "
            f"Todo fato deve ter evidence_message_index={target_index}."
        )
        lines.append("")
    lines.append(
        "Retorne o JSON no schema ExtractionResult "
        "(session_summary, facts[], unknowns[], deal)."
    )
    return "\n".join(lines)
