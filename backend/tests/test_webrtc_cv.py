"""Tests for WebRTC signaling endpoints."""

import pytest
from aiortc import RTCPeerConnection


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


@pytest.mark.asyncio
async def test_webrtc_offer_returns_answer() -> None:
    """A valid SDP offer should be answered with an SDP answer."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    # Build a minimal offer with a video transceiver and DataChannel.
    sender = RTCPeerConnection()
    sender.addTransceiver("video", direction="sendonly")
    sender.createDataChannel("metadata")

    offer = await sender.createOffer()
    await sender.setLocalDescription(offer)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/cv/webrtc/offer",
            json={
                "sdp": sender.localDescription.sdp,
                "type": sender.localDescription.type,
                "sessionId": "test-webrtc-session",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "sdp" in data
    assert data["type"] == "answer"
    assert len(data["sdp"]) > 0

    await sender.close()


@pytest.mark.asyncio
async def test_webrtc_offer_invalid_type_returns_422() -> None:
    """Offering a type other than 'offer' should be rejected by schema validation."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/cv/webrtc/offer",
            json={
                "sdp": "v=0\r\n",
                "type": "answer",  # invalid – must be "offer"
                "sessionId": "test-session",
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_webrtc_offer_missing_sdp_returns_422() -> None:
    """Omitting the required 'sdp' field should be rejected by schema validation."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/cv/webrtc/offer",
            json={
                # sdp is intentionally omitted
                "type": "offer",
                "sessionId": "test-session",
            },
        )
    assert response.status_code == 422
