"""Worker batch do Farm — consome mensagens prontas e roda a esteira.

Esteira por mensagem (Fase 3):
  farm:messages:ready (Redis stream, publicado pelo backend: texto na
  ingestão, mídia após o MediaWorker subir para o storage)
    -> GET /internal/messages/{id}/context
    -> STT (Gemini) se áudio sem transcrição
    -> extração-delta (carteira + resumos + sessão como contexto)
    -> POST /internal/messages/{id}/analysis

Falha em uma mensagem: loga e faz ACK (a mensagem fica no banco; reprocesso
manual é possível re-publicando no stream — sem dead-letter no ano 1).
"""

import logging
import signal
import sys
import time

import redis
from dotenv import load_dotenv

from ..config.settings import Settings
from .pipeline import MessagePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("farm.worker")

_running = True


def _stop(signum, _frame):
    global _running
    logger.info("sinal %s recebido — encerrando", signum)
    _running = False


def main() -> int:
    load_dotenv()
    settings = Settings.from_env()
    pipeline = MessagePipeline(settings)
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        client.xgroup_create(
            settings.work_stream, settings.consumer_group, id="0", mkstream=True
        )
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    if not settings.gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY ausente — worker sobe em fail-open "
            "(STT, extrator e copilot devolvem vazio; não invento chave)"
        )
    logger.info(
        "worker pronto | stream=%s group=%s backend=%s gemini=%s",
        settings.work_stream,
        settings.consumer_group,
        settings.backend_http_base_url,
        "ok" if settings.gemini_api_key else "fail-open",
    )

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while _running:
        entries = client.xreadgroup(
            settings.consumer_group,
            "worker-1",
            {settings.work_stream: ">"},
            count=10,
            block=5000,
        )
        if not entries:
            continue
        for _stream, records in entries:
            for record_id, fields in records:
                message_id = fields.get("messageId")
                try:
                    if message_id:
                        pipeline.process(message_id)
                    else:
                        logger.warning("registro sem messageId: %s", fields)
                except Exception:
                    logger.exception("esteira falhou message=%s", message_id)
                client.xack(settings.work_stream, settings.consumer_group, record_id)
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
