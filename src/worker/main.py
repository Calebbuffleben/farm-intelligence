"""Worker batch do Farm — consome mensagens prontas e roda a esteira.

Esteira por mensagem (Fase 3):
  farm:messages:ready (Redis stream, publicado pelo backend: texto na
  ingestão, mídia após o MediaWorker subir para o storage)
    -> GET /internal/messages/{id}/context
    -> STT (Gemini) se áudio sem transcrição
    -> extração-delta (carteira + resumos + sessão como contexto)
    -> POST /internal/messages/{id}/analysis

Falha em uma mensagem: loga e NÃO dá ACK — o próximo loop relê o PEL.
Restart do container (Railway) reusa o consumidor `worker-1`; sem ler o
PEL (`0`) as mensagens entregues antes do kill ficam presas para sempre.

XREADGROUP BLOCK pelo proxy público: o Redis entrega, o socket estoura,
retry_on_timeout manda outro `>` e a mensagem fica invisível. Por isso
lê PEL (`0`) e novos (`>`) no mesmo loop, sem retry de timeout.
"""

import logging
import signal
import sys
import time
from urllib.parse import urlparse

import redis
from dotenv import load_dotenv

from ..config.settings import Settings
from .pipeline import MessagePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("farm.worker")

CONSUMER = "worker-1"
HEARTBEAT_S = 15
_running = True


def _stop(signum, _frame):
    global _running
    logger.info("sinal %s recebido — encerrando", signum)
    _running = False


def _handle_records(client, settings: Settings, pipeline: MessagePipeline, records) -> None:
    for record_id, fields in records:
        message_id = fields.get("messageId") if isinstance(fields, dict) else None
        try:
            if not message_id:
                logger.warning("registro sem messageId: %s", fields)
                client.xack(settings.work_stream, settings.consumer_group, record_id)
                continue
            logger.info(
                "processando message=%s redis_id=%s keys=%s",
                message_id,
                record_id,
                sorted(fields.keys()) if isinstance(fields, dict) else type(fields).__name__,
            )
            pipeline.process(message_id)
            client.xack(settings.work_stream, settings.consumer_group, record_id)
            logger.info("ack message=%s", message_id)
        except Exception:
            logger.exception("esteira falhou message=%s — sem ack, tenta de novo", message_id)


def _xpending_count(client, settings: Settings):
    try:
        info = client.xpending(settings.work_stream, settings.consumer_group)
        if isinstance(info, dict):
            return info.get("pending", info)
        return info
    except Exception:
        return "?"


def main() -> int:
    load_dotenv()
    settings = Settings.from_env()
    pipeline = MessagePipeline(settings)
    # block 1s, socket_timeout 10s. retry_on_timeout=False: retentar
    # XREADGROUP BLOCK depois do timeout deixa a mensagem no PEL.
    block_ms = 1000
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=10,
        retry_on_timeout=False,
        health_check_interval=30,
    )
    try:
        client.xgroup_create(
            settings.work_stream, settings.consumer_group, id="0", mkstream=True
        )
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    parsed = urlparse(settings.redis_url)
    redis_host = f"{parsed.hostname}:{parsed.port or 6379}{parsed.path or ''}"
    try:
        xlen = client.xlen(settings.work_stream)
    except Exception:
        xlen = -1
    if not settings.gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY ausente — worker sobe em fail-open "
            "(STT, extrator e copilot devolvem vazio; não invento chave)"
        )
    logger.info(
        "worker pronto | stream=%s group=%s redis=%s xlen=%s xpending=%s backend=%s gemini=%s",
        settings.work_stream,
        settings.consumer_group,
        redis_host,
        xlen,
        _xpending_count(client, settings),
        settings.backend_http_base_url,
        "ok" if settings.gemini_api_key else "fail-open",
    )

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        _next, claimed, *_rest = client.xautoclaim(
            settings.work_stream,
            settings.consumer_group,
            CONSUMER,
            min_idle_time=0,
            start_id="0-0",
            count=50,
        )
        if claimed:
            logger.info("pel recuperado n=%d", len(claimed))
            _handle_records(client, settings, pipeline, claimed)
    except Exception:
        logger.exception("xautoclaim falhou — segue no loop")

    idle_since = time.monotonic()
    while _running:
        saw_work = False
        try:
            pending = client.xreadgroup(
                settings.consumer_group,
                CONSUMER,
                {settings.work_stream: "0"},
                count=10,
            )
            if pending:
                saw_work = True
                for _stream, records in pending:
                    _handle_records(client, settings, pipeline, records)
            entries = client.xreadgroup(
                settings.consumer_group,
                CONSUMER,
                {settings.work_stream: ">"},
                count=10,
                block=block_ms,
            )
            if entries:
                saw_work = True
                for _stream, records in entries:
                    _handle_records(client, settings, pipeline, records)
        except redis.TimeoutError:
            logger.warning("redis timeout no XREADGROUP")
            continue
        except redis.ConnectionError:
            logger.exception("redis connection — espera 2s")
            time.sleep(2)
            continue
        if saw_work:
            idle_since = time.monotonic()
            continue
        if time.monotonic() - idle_since >= HEARTBEAT_S:
            try:
                xlen = client.xlen(settings.work_stream)
            except Exception:
                xlen = -1
            logger.info(
                "aguardando fila xlen=%s xpending=%s redis=%s",
                xlen,
                _xpending_count(client, settings),
                redis_host,
            )
            idle_since = time.monotonic()
    return 0


if __name__ == "__main__":
    sys.exit(main())
