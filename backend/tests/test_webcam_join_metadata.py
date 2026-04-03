"""Focused tests for webcam join metadata and player naming."""

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if "redis" not in sys.modules:
    redis_module = types.ModuleType("redis")
    redis_asyncio_module = types.ModuleType("redis.asyncio")
    redis_exceptions_module = types.ModuleType("redis.exceptions")

    class _DummyRedis:  # pragma: no cover - test bootstrap only
        pass

    class _DummyRedisConnectionError(Exception):
        pass

    class _DummyRedisTimeoutError(Exception):
        pass

    redis_asyncio_module.Redis = _DummyRedis
    redis_asyncio_module.from_url = lambda *args, **kwargs: _DummyRedis()
    redis_exceptions_module.ConnectionError = _DummyRedisConnectionError
    redis_exceptions_module.TimeoutError = _DummyRedisTimeoutError
    redis_module.asyncio = redis_asyncio_module
    redis_module.exceptions = redis_exceptions_module

    sys.modules["redis"] = redis_module
    sys.modules["redis.asyncio"] = redis_asyncio_module
    sys.modules["redis.exceptions"] = redis_exceptions_module

from backend.routers.session import router as session_router
from backend.routers.webcam import router as webcam_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(session_router)
    app.include_router(webcam_router)
    return app


@pytest.fixture
def mock_redis():
    store: dict[str, tuple[str, int | None]] = {}

    async def fake_get(key: str):
        entry = store.get(key)
        return entry[0] if entry else None

    async def fake_set(key: str, value: str, ex: int | None = None):
        store[key] = (value, ex)

    async def fake_exists(key: str):
        return 1 if key in store else 0

    async def fake_delete(key: str):
        store.pop(key, None)

    mock = AsyncMock()
    mock.get = AsyncMock(side_effect=fake_get)
    mock.set = AsyncMock(side_effect=fake_set)
    mock.exists = AsyncMock(side_effect=fake_exists)
    mock.delete = AsyncMock(side_effect=fake_delete)

    with patch("backend.services.session_service.get_redis", return_value=mock):
        yield mock


def test_status_by_code_includes_table_context(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-1",
            "data": {
                "tableSize": 6,
                "hand": {"botPosition": 3},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-1"},
    )
    code = gen_resp.json()["code"]
    status_resp = client.get(f"/session/webcam/status-by-code/{code}")

    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["tableSize"] == 6
    assert data["botPosition"] == 3


def test_status_by_code_prefers_saved_bot_seat_metadata(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-seat",
            "data": {
                "tableSize": 6,
                "botSeat": 5,
                "hand": {"botPosition": 1},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-seat"},
    )
    code = gen_resp.json()["code"]
    status_resp = client.get(f"/session/webcam/status-by-code/{code}")

    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["tableSize"] == 6
    assert data["botPosition"] == 5


def test_status_includes_manual_seats_without_blocking_join_metadata(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-manual",
            "data": {
                "tableSize": 6,
                "botSeat": 2,
                "manualSeats": [1, 4],
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-manual"},
    )
    code = gen_resp.json()["code"]
    status_resp = client.get(f"/session/webcam/status-by-code/{code}")

    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["botPosition"] == 2
    assert data["manualSeats"] == [1, 4]
    assert data["code"] == code


def test_join_uses_real_seat_index_and_persists_player_name(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-2",
            "data": {
                "tableSize": 6,
                "hand": {"botPosition": 3},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-2"},
    )
    code = gen_resp.json()["code"]

    join_resp = client.post(
        "/session/webcam/join",
        json={"code": code, "player_position": 0, "player_name": "Alice"},
    )
    status_resp = client.get("/session/webcam/status/metadata-session-2")

    assert join_resp.status_code == 200
    assert join_resp.json()["player_name"] == "Alice"
    assert status_resp.status_code == 200
    assert status_resp.json()["opponents"]["0"]["player_name"] == "Alice"


def test_join_defaults_player_name_when_blank(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-3",
            "data": {
                "tableSize": 6,
                "hand": {"botPosition": 2},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-3"},
    )
    code = gen_resp.json()["code"]

    join_resp = client.post(
        "/session/webcam/join",
        json={"code": code, "player_position": 4, "player_name": "   "},
    )

    assert join_resp.status_code == 200
    assert join_resp.json()["player_name"] == "Player 5"


def test_manual_host_seat_does_not_block_later_webcam_join(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-manual-join",
            "data": {
                "tableSize": 6,
                "botSeat": 2,
                "manualSeats": [4],
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-manual-join"},
    )
    code = gen_resp.json()["code"]

    join_resp = client.post(
        "/session/webcam/join",
        json={"code": code, "player_position": 4, "player_name": "Late Join"},
    )

    assert join_resp.status_code == 200
    assert join_resp.json()["player_name"] == "Late Join"


def test_webcam_positions_rotate_with_next_hand_and_keep_names(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-4",
            "data": {
                "tableSize": 6,
                "hand": {"botPosition": 4},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-4"},
    )
    code = gen_resp.json()["code"]

    join_resp = client.post(
        "/session/webcam/join",
        json={"code": code, "player_position": 1, "player_name": "Alice"},
    )
    cv_session_id = join_resp.json()["cv_session_id"]

    client.put(
        "/session/metadata-session-4",
        json={
            "tableSize": 6,
            "hand": {"botPosition": 3},
        },
    )

    status_resp = client.get("/session/webcam/status/metadata-session-4")

    assert status_resp.status_code == 200
    opponents = status_resp.json()["opponents"]
    assert "0" in opponents
    assert opponents["0"]["connected"] is True
    assert opponents["0"]["player_name"] == "Alice"
    assert opponents["0"]["cv_session_id"] == cv_session_id


def test_disconnect_and_reconnect_support_cv_session_id(app: FastAPI, mock_redis):
    client = TestClient(app)
    client.post(
        "/session",
        json={
            "session_id": "metadata-session-5",
            "data": {
                "tableSize": 6,
                "hand": {"botPosition": 2},
            },
        },
    )
    gen_resp = client.post(
        "/session/webcam/generate-code",
        json={"session_id": "metadata-session-5"},
    )
    code = gen_resp.json()["code"]

    join_resp = client.post(
        "/session/webcam/join",
        json={"code": code, "player_position": 4, "player_name": "Bob"},
    )
    cv_session_id = join_resp.json()["cv_session_id"]

    disconnect_resp = client.post(
        "/session/webcam/disconnect",
        json={"session_id": "metadata-session-5", "cv_session_id": cv_session_id},
    )
    reconnect_resp = client.post(
        "/session/webcam/reconnect",
        json={"session_id": "metadata-session-5", "cv_session_id": cv_session_id},
    )
    status_resp = client.get("/session/webcam/status/metadata-session-5")

    assert disconnect_resp.status_code == 200
    assert reconnect_resp.status_code == 200
    assert status_resp.status_code == 200
    assert status_resp.json()["opponents"]["4"]["connected"] is True
