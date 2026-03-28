"""
Session service for managing game state persistence in Redis.

Provides async Redis operations for saving, loading, and deleting
game sessions keyed by client-generated session IDs, as well as
webcam session management for opponent bluff-data connections.
"""

import json
import logging
import secrets
import string
import time
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from backend.config import get_settings

_redis_client: Optional[aioredis.Redis] = None
_last_metrics_redis_error_log_at: float = 0.0
_METRICS_REDIS_ERROR_LOG_INTERVAL_SEC = 15.0


logger = logging.getLogger(__name__)


async def get_redis() -> aioredis.Redis:
    """Get or create the async Redis client (singleton)."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection pool. Call on app shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def _reset_redis_client() -> None:
    """Drop the cached Redis client so the next request reconnects cleanly."""
    global _redis_client
    client = _redis_client
    _redis_client = None
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            pass


def _should_log_metrics_redis_error() -> bool:
    global _last_metrics_redis_error_log_at
    now = time.monotonic()
    if now - _last_metrics_redis_error_log_at < _METRICS_REDIS_ERROR_LOG_INTERVAL_SEC:
        return False
    _last_metrics_redis_error_log_at = now
    return True


def _session_key(session_id: str) -> str:
    """Build the Redis key for a session."""
    return f"poker:session:{session_id}"


async def save_session(session_id: str, data: dict[str, Any]) -> None:
    """
    Save session data to Redis with TTL.

    Args:
        session_id: Client-generated UUID.
        data: Full session snapshot (JSON-serialisable dict).
    """
    settings = get_settings()
    client = await get_redis()
    await client.set(
        _session_key(session_id),
        json.dumps(data),
        ex=settings.session_ttl_seconds,
    )


async def load_session(session_id: str) -> Optional[dict[str, Any]]:
    """
    Load session data from Redis.

    Returns:
        The session dict or None if not found / expired.
    """
    client = await get_redis()
    raw = await client.get(_session_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)


async def delete_session(session_id: str) -> None:
    """Delete a session from Redis."""
    client = await get_redis()
    await client.delete(_session_key(session_id))


async def session_exists(session_id: str) -> bool:
    """Check whether a session exists in Redis."""
    client = await get_redis()
    return bool(await client.exists(_session_key(session_id)))


# ---------------------------------------------------------------------------
# Webcam session management
# ---------------------------------------------------------------------------

_CODE_LENGTH = 6
_CODE_ALPHABET = string.ascii_uppercase + string.digits
_CODE_TTL_SECONDS = 3600  # 1 hour


def _webcam_code_key(code: str) -> str:
    """Redis key that maps a join code to its session_id."""
    return f"poker:webcam:code:{code.upper()}"


def _webcam_session_key(session_id: str) -> str:
    """Redis key for the webcam session data tied to a game session."""
    return f"poker:webcam:session:{session_id}"


def _webcam_metrics_key(cv_session_id: str) -> str:
    """Redis key for the latest CV metrics of a webcam session."""
    return f"poker:webcam:metrics:{cv_session_id}"


async def save_webcam_metrics(cv_session_id: str, metrics_json: str) -> None:
    """Save the latest CV analysis metrics for a webcam session (short TTL)."""
    try:
        client = await get_redis()
        await client.set(_webcam_metrics_key(cv_session_id), metrics_json, ex=10)
    except (RedisConnectionError, RedisTimeoutError):
        await _reset_redis_client()
        if _should_log_metrics_redis_error():
            logger.warning(
                "Redis unavailable while saving webcam metrics; metrics caching will retry automatically",
                exc_info=True,
            )


def _generate_code() -> str:
    """Generate a random 6-character alphanumeric code."""
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


async def generate_webcam_code(session_id: str) -> str:
    """
    Create a short join code that maps to *session_id*.

    The code is stored in Redis with a 1-hour TTL and can be used by
    opponents to discover the session they should connect to.
    """
    client = await get_redis()
    code = _generate_code()

    # Retry if the code already exists (extremely unlikely with 36^6 space)
    for _ in range(5):
        if not await client.exists(_webcam_code_key(code)):
            break
        code = _generate_code()

    payload = json.dumps({"session_id": session_id})
    await client.set(_webcam_code_key(code), payload, ex=_CODE_TTL_SECONDS)

    # Ensure the webcam session key exists (idempotent)
    settings = get_settings()
    wkey = _webcam_session_key(session_id)
    if not await client.exists(wkey):
        await client.set(wkey, json.dumps({"opponents": {}}), ex=settings.session_ttl_seconds)

    return code


async def lookup_webcam_code(code: str) -> Optional[str]:
    """
    Return the session_id associated with *code*, or ``None`` if the
    code is invalid or expired.
    """
    client = await get_redis()
    raw = await client.get(_webcam_code_key(code.upper()))
    if raw is None:
        return None
    data = json.loads(raw)
    return data.get("session_id")


async def join_webcam_session(
    session_id: str, player_position: int
) -> str:
    """
    Register an opponent at *player_position* in the webcam session.

    Returns a unique ``cv_session_id`` that the opponent should use when
    streaming frames to the CV pipeline.

    Raises ``ValueError`` if the position is already taken.
    """
    client = await get_redis()
    settings = get_settings()
    wkey = _webcam_session_key(session_id)

    raw = await client.get(wkey)
    if raw is None:
        data: dict[str, Any] = {"opponents": {}}
    else:
        data = json.loads(raw)

    pos_str = str(player_position)
    if pos_str in data["opponents"] and data["opponents"][pos_str].get("connected"):
        raise ValueError(f"Player position {player_position} is already connected")

    cv_session_id = f"{session_id}__p{player_position}"
    data["opponents"][pos_str] = {
        "connected": True,
        "cv_session_id": cv_session_id,
    }

    await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)
    return cv_session_id


async def disconnect_webcam(session_id: str, player_position: int) -> None:
    """Mark an opponent at *player_position* as disconnected."""
    client = await get_redis()
    settings = get_settings()
    wkey = _webcam_session_key(session_id)

    raw = await client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    pos_str = str(player_position)
    if pos_str in data["opponents"]:
        data["opponents"][pos_str]["connected"] = False
        await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)


async def reconnect_webcam(session_id: str, player_position: int) -> None:
    """Mark a previously disconnected opponent at *player_position* as connected again."""
    client = await get_redis()
    settings = get_settings()
    wkey = _webcam_session_key(session_id)

    raw = await client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    pos_str = str(player_position)
    if pos_str in data["opponents"]:
        data["opponents"][pos_str]["connected"] = True
        await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)


async def get_webcam_status(session_id: str) -> dict[str, Any]:
    """
    Return the current webcam connection status for all opponents in
    the given session.

    Returns a dict like::

        {
            "opponents": {
                "2": {"connected": true, "cv_session_id": "..."},
                ...
            }
        }
    """
    client = await get_redis()
    wkey = _webcam_session_key(session_id)
    raw = await client.get(wkey)
    if raw is None:
        return {"opponents": {}}
    
    data = json.loads(raw)
    
    # Fetch metrics for all connected opponents
    for pos_str, opp in data.get("opponents", {}).items():
        if opp.get("connected") and "cv_session_id" in opp:
            mkey = _webcam_metrics_key(opp["cv_session_id"])
            mraw = await client.get(mkey)
            if mraw is not None:
                opp["metrics"] = json.loads(mraw)
                
    return data
