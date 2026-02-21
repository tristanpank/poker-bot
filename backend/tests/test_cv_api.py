"""Tests for backend CV analysis endpoints."""

import base64
from fastapi.testclient import TestClient

from backend.main import app


def _rgba_base64(width: int, height: int) -> str:
    # Simple synthetic pattern.
    raw = bytearray()
    for y in range(height):
        for x in range(width):
            raw.extend(
                [
                    (x * 17) % 255,  # R
                    (y * 23) % 255,  # G
                    ((x + y) * 11) % 255,  # B
                    255,  # A
                ]
            )
    return base64.b64encode(bytes(raw)).decode("ascii")


def _rgba_bytes(width: int, height: int) -> bytes:
    raw = bytearray()
    for y in range(height):
        for x in range(width):
            raw.extend(
                [
                    (x * 17) % 255,
                    (y * 23) % 255,
                    ((x + y) * 11) % 255,
                    255,
                ]
            )
    return bytes(raw)


def test_cv_analyze_returns_metrics() -> None:
    client = TestClient(app)

    payload = {
        "sessionId": "cv-test-session",
        "timestamp": 1_000,
        "width": 8,
        "height": 8,
        "rgbaBase64": _rgba_base64(8, 8),
        "streamFps": 24.2,
    }

    response = client.post("/cv/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]

    assert "bluffRisk" in metrics
    assert "analysisFps" in metrics
    assert "streamFps" in metrics
    assert metrics["streamFps"] == 24.2


def test_cv_analyze_reuses_session_state() -> None:
    client = TestClient(app)
    payload = {
        "sessionId": "cv-test-session-2",
        "timestamp": 10_000,
        "width": 8,
        "height": 8,
        "rgbaBase64": _rgba_base64(8, 8),
        "streamFps": 30.0,
    }

    first = client.post("/cv/analyze", json=payload)
    assert first.status_code == 200

    payload["timestamp"] = 10_200
    second = client.post("/cv/analyze", json=payload)
    assert second.status_code == 200
    second_metrics = second.json()["metrics"]

    assert second_metrics["analysisFps"] > 0


def test_cv_analyze_invalid_length_returns_400() -> None:
    client = TestClient(app)
    payload = {
        "sessionId": "cv-bad",
        "timestamp": 1_000,
        "width": 8,
        "height": 8,
        "rgbaBase64": base64.b64encode(b"bad").decode("ascii"),
        "streamFps": 0.0,
    }

    response = client.post("/cv/analyze", json=payload)
    assert response.status_code == 400


def test_cv_analyze_raw_returns_metrics() -> None:
    client = TestClient(app)

    response = client.post(
        "/cv/analyze-raw",
        params={
            "sessionId": "cv-raw-session",
            "timestamp": 5_000,
            "width": 8,
            "height": 8,
            "streamFps": 27.7,
        },
        content=_rgba_bytes(8, 8),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "metrics" in data
    assert data["metrics"]["streamFps"] == 27.7


def test_cv_analyze_raw_invalid_length_returns_400() -> None:
    client = TestClient(app)

    response = client.post(
        "/cv/analyze-raw",
        params={
            "sessionId": "cv-raw-bad",
            "timestamp": 5_000,
            "width": 8,
            "height": 8,
            "streamFps": 10.0,
        },
        content=b"bad",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 400


def test_cv_session_delete_returns_ok() -> None:
    client = TestClient(app)

    response = client.request(
        "DELETE",
        "/cv/session",
        json={"sessionId": "to-clear"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_cv_webrtc_offer_returns_answer() -> None:
    """A valid SDP offer should be answered with an SDP answer."""
    import asyncio

    from aiortc import RTCPeerConnection

    async def _make_offer() -> dict:
        pc = RTCPeerConnection()
        pc.createDataChannel("frames")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await pc.close()
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    offer = asyncio.run(_make_offer())

    client = TestClient(app)
    response = client.post(
        "/cv/webrtc-offer",
        json={"sdp": offer["sdp"], "type": offer["type"], "sessionId": "wrtc-test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"
    assert "sdp" in data
    assert len(data["sdp"]) > 0


def test_cv_webrtc_offer_rejects_invalid_type() -> None:
    """Offering a non-offer SDP type should return a validation error."""
    client = TestClient(app)
    response = client.post(
        "/cv/webrtc-offer",
        json={"sdp": "v=0\r\n", "type": "answer", "sessionId": "wrtc-bad"},
    )
    assert response.status_code == 422
