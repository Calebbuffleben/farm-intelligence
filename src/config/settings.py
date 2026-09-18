"""Configuração do worker de inteligência do Farm (batch, sem realtime)."""

import os
from dataclasses import dataclass
from typing import Optional

_DEFAULT_REDIS = "redis://localhost:6379/0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def redis_url_from_env() -> str:
    """Mesmo Redis do backend. REDIS_PRIVATE_URL primeiro: XREADGROUP BLOCK
    pelo proxy público da Railway (REDIS_URL / *.proxy.rlwy.net) estoura
    timeout depois de o servidor já ter colocado a mensagem no PEL."""
    return (
        (os.getenv("REDIS_PRIVATE_URL") or "").strip()
        or (os.getenv("REDIS_URL") or "").strip()
        or _DEFAULT_REDIS
    )


@dataclass
class Settings:
    # Fila de trabalho (backend publica message_ready após mídia no storage)
    redis_url: str = _DEFAULT_REDIS
    work_stream: str = "farm:messages:ready"
    consumer_group: str = "intelligence"

    # Gemini — extração de fatos + resolução de entidade (JSON mode)
    gemini_api_key: Optional[str] = None
    # 3.8-flash (preview) voltou 503 de sobrecarga em produção; 2.5 é estável.
    extractor_model: str = "gemini-2.5-flash"
    extractor_max_output_tokens: int = 8192
    # Copilot pós-STT: uma frase. Teto baixo de propósito (não é extração).
    coach_max_output_tokens: int = 256
    # Abaixo deste valor o vínculo NÃO vira fato — vai para a fila unknown.
    resolver_min_confidence: float = 0.7

    # API interna do backend (shared secret no header x-internal-token).
    # HTTP em vez de gRPC: volume é batch/baixo — o gRPC do Meet existia por
    # causa de streaming de áudio, que não se aplica aqui.
    backend_http_base_url: str = "http://localhost:8080"
    backend_internal_token: Optional[str] = None

    # STT batch (Fase 3)
    stt_provider: str = "gemini"  # gemini | assemblyai

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=redis_url_from_env(),
            work_stream=os.getenv("FARM_WORK_STREAM", cls.work_stream),
            consumer_group=os.getenv("FARM_CONSUMER_GROUP", cls.consumer_group),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            extractor_model=os.getenv("FARM_EXTRACTOR_MODEL", cls.extractor_model),
            extractor_max_output_tokens=int(
                os.getenv("FARM_EXTRACTOR_MAX_TOKENS", str(cls.extractor_max_output_tokens))
            ),
            coach_max_output_tokens=int(
                os.getenv("FARM_COACH_MAX_TOKENS", str(cls.coach_max_output_tokens))
            ),
            resolver_min_confidence=float(
                os.getenv("FARM_RESOLVER_MIN_CONFIDENCE", str(cls.resolver_min_confidence))
            ),
            backend_http_base_url=os.getenv(
                "BACKEND_HTTP_BASE_URL", cls.backend_http_base_url
            ),
            backend_internal_token=os.getenv("INTERNAL_API_TOKEN"),
            stt_provider=os.getenv("FARM_STT_PROVIDER", cls.stt_provider),
        )
