"""REDIS_PRIVATE_URL vence REDIS_URL — XREADGROUP BLOCK não passa no proxy."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import redis_url_from_env


def _restore(key: str, previous):
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def test_prefers_private_over_public():
    prev_priv = os.environ.get("REDIS_PRIVATE_URL")
    prev_pub = os.environ.get("REDIS_URL")
    os.environ["REDIS_PRIVATE_URL"] = "redis://internal:6379/0"
    os.environ["REDIS_URL"] = "redis://hopper.proxy.rlwy.net:1234"
    try:
        assert redis_url_from_env() == "redis://internal:6379/0"
    finally:
        _restore("REDIS_PRIVATE_URL", prev_priv)
        _restore("REDIS_URL", prev_pub)


def test_falls_back_to_public():
    prev_priv = os.environ.get("REDIS_PRIVATE_URL")
    prev_pub = os.environ.get("REDIS_URL")
    os.environ.pop("REDIS_PRIVATE_URL", None)
    os.environ["REDIS_URL"] = "redis://hopper.proxy.rlwy.net:1234"
    try:
        assert redis_url_from_env() == "redis://hopper.proxy.rlwy.net:1234"
    finally:
        _restore("REDIS_PRIVATE_URL", prev_priv)
        _restore("REDIS_URL", prev_pub)
