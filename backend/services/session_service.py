"""
Session service for managing game state persistence in Redis.

Provides async Redis operations for saving, loading, and deleting
game sessions keyed by client-generated session IDs, as well as
webcam session management for opponent bluff-data connections.
"""

from collections import deque
from dataclasses import dataclass
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
_METRIC_HISTORY_KEEP_MS = 20 * 60_000
_METRIC_HISTORY_MAX_SAMPLES = 2_048


logger = logging.getLogger(__name__)


@dataclass
class WebcamMetricSample:
    timestamp_ms: int
    bluff_risk: float
    bluff_delta: float


_webcam_metric_history: dict[str, deque[WebcamMetricSample]] = {}


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
    previous_raw = await client.get(_session_key(session_id))
    previous_data = json.loads(previous_raw) if previous_raw is not None else None
    await client.set(
        _session_key(session_id),
        json.dumps(data),
        ex=settings.session_ttl_seconds,
    )
    await _sync_webcam_session_positions(
        session_id,
        previous_session_data=previous_data,
        next_session_data=data,
        client=client,
        ttl_seconds=settings.session_ttl_seconds,
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
    await delete_webcam_session(session_id, client=client)
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


def _default_webcam_session_data() -> dict[str, Any]:
    """Return the default Redis payload for a webcam session."""
    return {"code": None, "opponents": {}}


def _extract_table_context_from_session_data(session_data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Extract current table metadata for webcam join/seat labeling."""
    if not isinstance(session_data, dict):
        return {"tableSize": None, "botPosition": None}

    table_size_raw = session_data.get("tableSize")
    hand = session_data.get("hand") if isinstance(session_data.get("hand"), dict) else {}
    bot_position_raw = hand.get("botPosition")

    table_size = int(table_size_raw) if isinstance(table_size_raw, int) and 2 <= table_size_raw <= 6 else None
    bot_position = int(bot_position_raw) if isinstance(bot_position_raw, int) and 0 <= bot_position_raw <= 5 else None

    if table_size is not None and bot_position is not None and bot_position >= table_size:
        bot_position = None

    return {
        "tableSize": table_size,
        "botPosition": bot_position,
    }


async def _sync_webcam_session_positions(
    session_id: str,
    *,
    previous_session_data: Optional[dict[str, Any]],
    next_session_data: Optional[dict[str, Any]],
    client: aioredis.Redis,
    ttl_seconds: int,
) -> None:
    """Rotate webcam seat assignments when the saved game state rotates positions."""
    wkey = _webcam_session_key(session_id)
    raw = await client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    opponents = data.get("opponents")
    if not isinstance(opponents, dict):
        return

    previous_context = _extract_table_context_from_session_data(previous_session_data)
    next_context = _extract_table_context_from_session_data(next_session_data)
    previous_table_size = previous_context["tableSize"]
    next_table_size = next_context["tableSize"]
    previous_bot_position = previous_context["botPosition"]
    next_bot_position = next_context["botPosition"]

    rotated_opponents = opponents
    if (
        previous_table_size is not None
        and next_table_size is not None
        and previous_table_size == next_table_size
        and previous_bot_position is not None
        and next_bot_position is not None
    ):
        shift = (int(next_bot_position) - int(previous_bot_position)) % int(next_table_size)
        if shift != 0:
            rotated_opponents = {}
            for position, opponent in opponents.items():
                try:
                    seat_index = int(position)
                except (TypeError, ValueError):
                    rotated_opponents[position] = opponent
                    continue

                if 0 <= seat_index < int(next_table_size):
                    rotated_opponents[str((seat_index + shift) % int(next_table_size))] = opponent
                else:
                    rotated_opponents[position] = opponent

    data["opponents"] = rotated_opponents
    await client.set(wkey, json.dumps(data), ex=ttl_seconds)


def _resolve_opponent_position_key(
    webcam_session_data: dict[str, Any],
    *,
    player_position: Optional[int] = None,
    cv_session_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve the current opponent key by seat index or stable cv_session_id."""
    opponents = webcam_session_data.get("opponents")
    if not isinstance(opponents, dict):
        return None

    if player_position is not None:
        pos_str = str(player_position)
        if pos_str in opponents:
            return pos_str

    if cv_session_id:
        for position, opponent in opponents.items():
            if isinstance(opponent, dict) and opponent.get("cv_session_id") == cv_session_id:
                return position

    return None


def append_webcam_metric_sample(
    cv_session_id: str,
    metrics: dict[str, Any],
    *,
    timestamp_ms: Optional[int] = None,
) -> None:
    """Append a bluff-metric sample to the in-memory rolling history."""
    try:
        bluff_risk = float(metrics.get("bluffRisk", 0.0))
        bluff_delta = float(metrics.get("bluffDelta", 0.0))
    except (TypeError, ValueError):
        return

    ts = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000.0)
    history = _webcam_metric_history.setdefault(
        cv_session_id,
        deque(maxlen=_METRIC_HISTORY_MAX_SAMPLES),
    )
    history.append(
        WebcamMetricSample(
            timestamp_ms=ts,
            bluff_risk=bluff_risk,
            bluff_delta=bluff_delta,
        )
    )

    min_ts = ts - _METRIC_HISTORY_KEEP_MS
    while history and history[0].timestamp_ms < min_ts:
        history.popleft()


def summarize_webcam_metric_window(
    cv_session_id: str,
    *,
    started_at_ms: int,
    ended_at_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Summarize bluff metrics for a webcam stream over a time window."""
    end_ms = int(ended_at_ms if ended_at_ms is not None else time.time() * 1000.0)
    start_ms = int(min(started_at_ms, end_ms))
    history = _webcam_metric_history.get(cv_session_id)
    if not history:
        return {
            "started_at_ms": start_ms,
            "ended_at_ms": end_ms,
            "sample_count": 0,
            "avg_bluff_delta": None,
            "max_bluff_delta": None,
            "avg_bluff_risk": None,
            "max_bluff_risk": None,
        }

    samples = [
        sample
        for sample in history
        if start_ms <= sample.timestamp_ms <= end_ms
    ]
    if not samples:
        return {
            "started_at_ms": start_ms,
            "ended_at_ms": end_ms,
            "sample_count": 0,
            "avg_bluff_delta": None,
            "max_bluff_delta": None,
            "avg_bluff_risk": None,
            "max_bluff_risk": None,
        }

    sample_count = len(samples)
    avg_bluff_delta = sum(sample.bluff_delta for sample in samples) / float(sample_count)
    max_bluff_delta = max(sample.bluff_delta for sample in samples)
    avg_bluff_risk = sum(sample.bluff_risk for sample in samples) / float(sample_count)
    max_bluff_risk = max(sample.bluff_risk for sample in samples)
    return {
        "started_at_ms": start_ms,
        "ended_at_ms": end_ms,
        "sample_count": sample_count,
        "avg_bluff_delta": avg_bluff_delta,
        "max_bluff_delta": max_bluff_delta,
        "avg_bluff_risk": avg_bluff_risk,
        "max_bluff_risk": max_bluff_risk,
    }


def clear_webcam_metric_history(cv_session_id: Optional[str] = None) -> None:
    """Clear in-memory webcam metric history for one stream or all streams."""
    if cv_session_id is None:
        _webcam_metric_history.clear()
        return
    _webcam_metric_history.pop(cv_session_id, None)


async def save_webcam_metrics(cv_session_id: str, metrics_json: str) -> None:
    """Save the latest CV analysis metrics for a webcam session (short TTL)."""
    try:
        parsed_metrics = json.loads(metrics_json)
    except Exception:
        parsed_metrics = None
    if isinstance(parsed_metrics, dict):
        append_webcam_metric_sample(cv_session_id, parsed_metrics)

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
    settings = get_settings()
    wkey = _webcam_session_key(session_id)
    raw = await client.get(wkey)
    if raw is None:
        data = _default_webcam_session_data()
    else:
        data = json.loads(raw)
        data.setdefault("code", None)
        data.setdefault("opponents", {})

    existing_code = data.get("code")
    if isinstance(existing_code, str) and existing_code:
        payload = json.dumps({"session_id": session_id})
        await client.set(_webcam_code_key(existing_code), payload, ex=_CODE_TTL_SECONDS)
        await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)
        return existing_code

    code = _generate_code()

    # Retry if the code already exists (extremely unlikely with 36^6 space)
    for _ in range(5):
        if not await client.exists(_webcam_code_key(code)):
            break
        code = _generate_code()

    payload = json.dumps({"session_id": session_id})
    await client.set(_webcam_code_key(code), payload, ex=_CODE_TTL_SECONDS)
    data["code"] = code
    await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)

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
    session_id: str,
    player_position: int,
    player_name: Optional[str] = None,
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
    table_context = _extract_table_context_from_session_data(await load_session(session_id))
    table_size = table_context["tableSize"]
    bot_position = table_context["botPosition"]

    raw = await client.get(wkey)
    if raw is None:
        data: dict[str, Any] = {"opponents": {}}
    else:
        data = json.loads(raw)

    if table_size is not None and not (0 <= player_position < table_size):
        raise ValueError(f"Seat {player_position + 1} is not part of this table")
    if bot_position is not None and player_position == bot_position:
        raise ValueError(f"Seat {player_position + 1} belongs to the bot")

    pos_str = str(player_position)
    if pos_str in data["opponents"] and data["opponents"][pos_str].get("connected"):
        raise ValueError(f"Seat {player_position + 1} is already connected")

    cv_session_id = f"{session_id}__p{player_position}"
    resolved_player_name = (player_name or "").strip() or f"Player {player_position + 1}"
    data["opponents"][pos_str] = {
        "connected": True,
        "cv_session_id": cv_session_id,
        "player_name": resolved_player_name,
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
    pos_str = _resolve_opponent_position_key(data, player_position=player_position)
    if pos_str in data["opponents"]:
        data["opponents"][pos_str]["connected"] = False
        await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)


async def disconnect_webcam_by_cv_session(session_id: str, cv_session_id: str) -> None:
    """Mark an opponent as disconnected using the stable cv_session_id."""
    client = await get_redis()
    settings = get_settings()
    wkey = _webcam_session_key(session_id)

    raw = await client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    pos_str = _resolve_opponent_position_key(data, cv_session_id=cv_session_id)
    if pos_str is not None and pos_str in data["opponents"]:
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
    pos_str = _resolve_opponent_position_key(data, player_position=player_position)
    if pos_str in data["opponents"]:
        data["opponents"][pos_str]["connected"] = True
        await client.set(wkey, json.dumps(data), ex=settings.session_ttl_seconds)


async def reconnect_webcam_by_cv_session(session_id: str, cv_session_id: str) -> None:
    """Mark a previously disconnected opponent as connected using cv_session_id."""
    client = await get_redis()
    settings = get_settings()
    wkey = _webcam_session_key(session_id)

    raw = await client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    pos_str = _resolve_opponent_position_key(data, cv_session_id=cv_session_id)
    if pos_str is not None and pos_str in data["opponents"]:
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
    table_context = _extract_table_context_from_session_data(await load_session(session_id))
    if raw is None:
        return {"sessionActive": False, "opponents": {}, **table_context}

    data = json.loads(raw)
    opponents = data.get("opponents", {})
    result = {"sessionActive": True, "opponents": opponents, **table_context}

    # Fetch metrics for all connected opponents
    for pos_str, opp in opponents.items():
        if opp.get("connected") and "cv_session_id" in opp:
            mkey = _webcam_metrics_key(opp["cv_session_id"])
            mraw = await client.get(mkey)
            if mraw is not None:
                opp["metrics"] = json.loads(mraw)

    return result


async def delete_webcam_session(
    session_id: str, *, client: Optional[aioredis.Redis] = None
) -> None:
    """Delete the webcam join code, session state, and cached metrics."""
    redis_client = client or await get_redis()
    wkey = _webcam_session_key(session_id)
    raw = await redis_client.get(wkey)
    if raw is None:
        return

    data = json.loads(raw)
    code = data.get("code")
    if isinstance(code, str) and code:
        await redis_client.delete(_webcam_code_key(code))

    for opponent in data.get("opponents", {}).values():
        cv_session_id = opponent.get("cv_session_id")
        if isinstance(cv_session_id, str) and cv_session_id:
            await redis_client.delete(_webcam_metrics_key(cv_session_id))
            clear_webcam_metric_history(cv_session_id)

    await redis_client.delete(wkey)
