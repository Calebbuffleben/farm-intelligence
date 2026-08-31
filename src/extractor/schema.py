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


class ExtractionResult(BaseModel):
    """Saída completa de uma rodada do extrator sobre uma sessão lógica."""

    session_summary: str = Field(
        description="Resumo curto (2-3 frases) da sessão para RAG futuro"
    )
    facts: List[ExtractedFact] = Field(default_factory=list)
    unknowns: List[UnknownSpan] = Field(default_factory=list)
