"""Tests for webcam session management endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.fixture
def mock_redis():
    """Provide a dict-backed fake Redis for testing without a running server."""
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


@pytest.mark.asyncio
async def test_generate_code(mock_redis):
    """POST /session/webcam/generate-code returns a 6-char alphanumeric code."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/session/webcam/generate-code",
            json={"session_id": "test-session-1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "code" in data
    assert len(data["code"]) == 6
    assert data["code"].isalnum()


@pytest.mark.asyncio
async def test_join_with_valid_code(mock_redis):
    """POST /session/webcam/join succeeds with a valid code and returns session info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Generate code first
        gen_resp = await ac.post(
            "/session/webcam/generate-code",
            json={"session_id": "test-session-2"},
        )
        code = gen_resp.json()["code"]

        # Join with the code
        join_resp = await ac.post(
            "/session/webcam/join",
            json={"code": code, "player_position": 2},
        )
    assert join_resp.status_code == 200
    data = join_resp.json()
    assert data["session_id"] == "test-session-2"
    assert "cv_session_id" in data
    assert "p2" in data["cv_session_id"]


@pytest.mark.asyncio
async def test_join_invalid_code_returns_404(mock_redis):
    """POST /session/webcam/join returns 404 for invalid/expired codes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/session/webcam/join",
            json={"code": "ZZZZZZ", "player_position": 1},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_join_duplicate_position_returns_409(mock_redis):
    """POST /session/webcam/join returns 409 when position is already connected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        gen_resp = await ac.post(
            "/session/webcam/generate-code",
            json={"session_id": "test-session-3"},
        )
        code = gen_resp.json()["code"]

        # First join succeeds
        resp1 = await ac.post(
            "/session/webcam/join",
            json={"code": code, "player_position": 3},
        )
        assert resp1.status_code == 200

        # Duplicate join fails
        resp2 = await ac.post(
            "/session/webcam/join",
            json={"code": code, "player_position": 3},
        )
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_status_shows_connected_opponents(mock_redis):
    """GET /session/webcam/status/{session_id} lists connected opponents."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        gen_resp = await ac.post(
            "/session/webcam/generate-code",
            json={"session_id": "test-session-4"},
        )
        code = gen_resp.json()["code"]

        await ac.post("/session/webcam/join", json={"code": code, "player_position": 1})
        await ac.post("/session/webcam/join", json={"code": code, "player_position": 4})

        status_resp = await ac.get("/session/webcam/status/test-session-4")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["opponents"]["1"]["connected"] is True
    assert data["opponents"]["4"]["connected"] is True


@pytest.mark.asyncio
async def test_status_empty_session(mock_redis):
    """GET /session/webcam/status returns empty opponents for unknown session."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/session/webcam/status/nonexistent")
    assert resp.status_code == 200
    assert resp.json() == {"opponents": {}}


@pytest.mark.asyncio
async def test_disconnect_marks_opponent_disconnected(mock_redis):
    """POST /session/webcam/disconnect sets connected=false for the opponent."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        gen_resp = await ac.post(
            "/session/webcam/generate-code",
            json={"session_id": "test-session-5"},
        )
        code = gen_resp.json()["code"]

        await ac.post("/session/webcam/join", json={"code": code, "player_position": 2})

        # Disconnect
        disc_resp = await ac.post(
            "/session/webcam/disconnect",
            json={"session_id": "test-session-5", "player_position": 2},
        )
        assert disc_resp.status_code == 200

        # Verify status
        status_resp = await ac.get("/session/webcam/status/test-session-5")
    data = status_resp.json()
    assert data["opponents"]["2"]["connected"] is False
