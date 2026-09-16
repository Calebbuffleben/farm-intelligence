"""Schema de saída do extrator (contrato JSON com o Gemini).

Espelha o modelo Prisma do backend: cada fato carrega o PRÓPRIO farm_id
(matriz de resolução — um áudio pode gerar fatos para 2+ fazendas).
Vínculo abaixo da confiança mínima NUNCA vira fato: vai para `unknowns`.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

FactKind = Literal["OBJECAO", "RISCO", "OPORTUNIDADE", "FOLLOWUP", "CONCORRENTE"]
FactSeverity = Literal["INFO", "WARNING", "CRITICAL"]

# Taxonomia agro fechada do ano 1 — o dashboard agrega por estes valores.
FACT_SUBTYPES = [
    "preco",
    "credito_prazo",
    "confianca_produto",
    "logistica",
    "janela_climatica",
    "concorrente_generico",
    "concorrente_marca",
    "assistencia_tecnica",
    "volume_pedido",
    "outro",
]


class ExtractedFact(BaseModel):
    kind: FactKind
    subtype: str = Field(description=f"Um de: {', '.join(FACT_SUBTYPES)}")
    severity: FactSeverity = "INFO"
    confidence: float = Field(ge=0.0, le=1.0)
    farm_id: Optional[str] = Field(
        default=None, description="ID da fazenda na carteira; null se não resolvido"
    )
    crop: Optional[str] = None
    product: Optional[str] = Field(default=None, description="Produto/insumo citado")
    headline: str = Field(description="Frase curta para o card do dashboard, em PT-BR")
    money_hint: Optional[str] = Field(
        default=None, description='Pista de dinheiro no texto, ex.: "50 galões", "5% desconto"'
    )
    due_hint_text: Optional[str] = Field(
        default=None, description='Pista de prazo, ex.: "amanhã à tarde", "depois da chuva"'
    )
    due_at: Optional[str] = Field(
        default=None,
        description="Prazo ISO-8601 apenas quando puder ser resolvido com segurança",
    )
    evidence_message_index: int = Field(
        description="Índice da mensagem que evidencia o fato"
    )
    evidence_span: Optional[str] = Field(
        default=None, description="Trecho literal da mensagem que sustenta o fato"
    )


class UnknownCandidate(BaseModel):
    farm_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class UnknownSpan(BaseModel):
    span_text: str
    evidence_message_index: int
    reason: str
    candidates: List[UnknownCandidate] = Field(default_factory=list)


DealStage = Literal["SONDAGEM", "NEGOCIACAO", "FECHAMENTO", "POS_VENDA", "SEM_NEGOCIO"]
DealLevel = Literal["BAIXA", "MEDIA", "ALTA"]
NextActionOwner = Literal["RTV", "MANAGER"]
AnalysisQuality = Literal["COMPLETE", "PARTIAL", "STALE"]

# Próximo passo em vocabulário fechado — o dashboard agrupa por isto.
NEXT_ACTION_KINDS = [
    "proposta",
    "followup",
    "logistica",
    "ligar",
    "escalar_gestor",
    "aguardar",
    "pos_venda",
]


class DealBriefOut(BaseModel):
    """Situação do negócio da conversa inteira (não só da mensagem alvo).

    Vira DealBrief (1 por Conversation) no backend. Resumo executivo em três
    pilares + GPS da venda (estágio e próximo passo).
    """

    # str (não Literal): o Gemini inventa valor e o Pydantic derrubava o
    # ExtractionResult inteiro — fatos e deal iam embora juntos. Saneia em payload.py.
    stage: str = Field(description="SONDAGEM | NEGOCIACAO | FECHAMENTO | POS_VENDA | SEM_NEGOCIO")
    stage_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    context_summary: str = Field(
        description="Contexto central: o que motivou a conversa (1-2 frases, PT-BR)"
    )
    producer_position: Optional[str] = Field(
        default=None,
        description="O que o produtor quer, aceitou, recusou ou ainda precisa decidir",
    )
    deal_change: Optional[str] = Field(
        default=None, description="O que mudou desde o brief anterior; null na primeira análise"
    )
    intent: str = Field(description="BAIXA | MEDIA | ALTA")
    urgency: str = Field(description="BAIXA | MEDIA | ALTA")
    pain_point: Optional[str] = Field(
        default=None,
        description="Dor / objeção oculta: o que impede a venda agora",
    )
    next_action: str = Field(
        description="Próximo passo sugerido ao RTV, acionável, 1 frase"
    )
    next_action_reason: Optional[str] = Field(
        default=None,
        description="Por que esta ação aumenta a chance de avanço",
    )
    next_action_owner: str = Field(description="RTV | MANAGER", default="RTV")
    next_action_kind: str = Field(
        description=f"Um de: {', '.join(NEXT_ACTION_KINDS)}"
    )
    next_action_due_hint: Optional[str] = Field(
        default=None, description='Prazo do próximo passo, ex.: "até sexta"'
    )
    next_action_due_at: Optional[str] = Field(
        default=None,
        description="Prazo ISO-8601 apenas quando puder ser resolvido com segurança",
    )
    suggested_reply: Optional[str] = Field(
        default=None,
        description="Resposta curta e editável para o RTV; nunca é enviada automaticamente",
    )
    manager_guidance: Optional[str] = Field(
        default=None,
        description="Decisão ou ajuda gerencial necessária; null se não houver intervenção",
    )
    analysis_quality: str = Field(
        default="COMPLETE", description="COMPLETE | PARTIAL | STALE"
    )
    blocker_subtype: Optional[str] = Field(
        default=None,
        description=f"Gargalo principal, um de: {', '.join(FACT_SUBTYPES)}; null se não há",
    )
    products: List[str] = Field(
        default_factory=list, description="Produtos/insumos em jogo"
    )


class ExtractionResult(BaseModel):
    """Saída completa de uma rodada do extrator sobre uma sessão lógica."""

    session_summary: str = Field(
        description="Resumo curto (2-3 frases) da sessão para RAG futuro"
    )
    facts: List[ExtractedFact] = Field(default_factory=list)
    unknowns: List[UnknownSpan] = Field(default_factory=list)
    deal: Optional[DealBriefOut] = Field(
        default=None, description="Situação do negócio da conversa"
    )
