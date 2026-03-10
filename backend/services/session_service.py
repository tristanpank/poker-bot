"""
Session service for managing game state persistence in Redis.

Provides async Redis operations for saving, loading, and deleting
game sessions keyed by client-generated session IDs.
"""

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.config import get_settings

_redis_client: Optional[aioredis.Redis] = None


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
