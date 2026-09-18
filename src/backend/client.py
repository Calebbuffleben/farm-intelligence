"""Cliente da API interna do farm-backend (shared secret x-internal-token).

Endpoints (todos fora da borda pública):
  GET  /internal/messages/{id}/context   -> contexto completo para o extrator
  GET  /internal/messages/{id}/media     -> bytes da mídia (STT; S3 ou canal)
  GET  /internal/media/{assetId}         -> bytes da mídia (legado, via S3)
  POST /internal/messages/{id}/analysis  -> transcript + links + fatos + unknowns
"""

import logging
from typing import Any, Dict

import requests

from ..config.settings import Settings

logger = logging.getLogger(__name__)

_TIMEOUT_S = 60


class BackendClient:
    def __init__(self, settings: Settings):
        if not settings.backend_internal_token:
            raise RuntimeError("INTERNAL_API_TOKEN não configurado")
        self._base = settings.backend_http_base_url.rstrip("/")
        self._headers = {"x-internal-token": settings.backend_internal_token}

    def get_context(self, message_id: str) -> Dict[str, Any]:
        resp = requests.get(
            f"{self._base}/internal/messages/{message_id}/context",
            headers=self._headers,
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()

    def get_message_media(self, message_id: str) -> tuple[bytes, str]:
        resp = requests.get(
            f"{self._base}/internal/messages/{message_id}/media",
            headers=self._headers,
            timeout=90,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type

    def get_media(self, asset_id: str) -> tuple[bytes, str]:
        resp = requests.get(
            f"{self._base}/internal/media/{asset_id}",
            headers=self._headers,
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type

    def post_analysis(self, message_id: str, payload: Dict[str, Any]) -> None:
        resp = requests.post(
            f"{self._base}/internal/messages/{message_id}/analysis",
            headers=self._headers,
            json=payload,
            timeout=_TIMEOUT_S,
        )
        if resp.status_code >= 400:
            logger.error("analysis rejeitada (%s): %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
