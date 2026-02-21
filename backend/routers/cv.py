"""
CV API router for backend frame analysis.
"""

import asyncio
import json
import struct

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.models.schemas import (
    CvAnalyzeRequest,
    CvAnalyzeResponse,
    CvSessionClearRequest,
    WebRtcAnswerResponse,
    WebRtcOfferRequest,
)
from backend.services.cv_service import CvService, get_cv_service


router = APIRouter(prefix="/cv", tags=["cv"])

# Active WebRTC peer connections (kept alive for the duration of the session).
_webrtc_pcs: set[object] = set()


@router.post("/analyze", response_model=CvAnalyzeResponse)
async def analyze_frame(
    request: CvAnalyzeRequest,
    cv_service: CvService = Depends(get_cv_service),
) -> CvAnalyzeResponse:
    """Analyze a frame and return bluff/stress proxy metrics."""
    try:
        metrics = cv_service.analyze(request)
        return CvAnalyzeResponse(metrics=metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"CV analysis error: {str(exc)}") from exc


@router.post("/analyze-raw", response_model=CvAnalyzeResponse)
async def analyze_raw_frame(
    request: Request,
    session_id: str = Query(..., alias="sessionId"),
    timestamp: int = Query(..., ge=0),
    width: int = Query(..., ge=1, le=2048),
    height: int = Query(..., ge=1, le=2048),
    stream_fps: float = Query(..., alias="streamFps", ge=0),
    cv_service: CvService = Depends(get_cv_service),
) -> CvAnalyzeResponse:
    """Analyze a raw RGBA frame body and return bluff/stress proxy metrics."""
    try:
        raw_body = await request.body()
        metrics = cv_service.analyze_raw(
            session_id=session_id,
            timestamp=timestamp,
            width=width,
            height=height,
            stream_fps=stream_fps,
            rgba_bytes=raw_body,
        )
        return CvAnalyzeResponse(metrics=metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"CV analysis error: {str(exc)}") from exc


@router.delete("/session")
async def clear_session(
    request: CvSessionClearRequest,
    cv_service: CvService = Depends(get_cv_service),
) -> dict[str, bool]:
    """Clear session analysis state."""
    cv_service.clear_session(request.session_id)
    return {"ok": True}


@router.post("/webrtc-offer", response_model=WebRtcAnswerResponse)
async def webrtc_offer(
    request: WebRtcOfferRequest,
    cv_service: CvService = Depends(get_cv_service),
) -> WebRtcAnswerResponse:
    """Accept a WebRTC SDP offer and return an answer.

    The frontend opens a DataChannel named ``frames`` over this connection and
    sends binary frame messages.  Each message has the layout::

        [uint32LE header_len][UTF-8 JSON header][raw RGBA bytes]

    The JSON header contains ``sessionId``, ``timestamp`` (ms), ``width``,
    ``height``, and ``streamFps``.  The backend analyses each frame via the
    usual CV pipeline and replies with the JSON-serialised ``CvAnalyzeResponse``
    on the same DataChannel.
    """
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=501,
            detail="WebRTC ingest is not available (aiortc is not installed).",
        ) from exc

    pc = RTCPeerConnection()
    _webrtc_pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("closed", "failed"):
            _webrtc_pcs.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-untyped-def]
        session_id = request.session_id

        @channel.on("message")
        async def on_message(message: bytes | str) -> None:
            if not isinstance(message, bytes) or len(message) < 4:
                return
            header_len = struct.unpack_from("<I", message, 0)[0]
            if len(message) < 4 + header_len:
                return
            try:
                header = json.loads(message[4 : 4 + header_len].decode("utf-8"))
                rgba_bytes = message[4 + header_len :]
                metrics = cv_service.analyze_raw(
                    session_id=header.get("sessionId", session_id),
                    timestamp=int(header["timestamp"]),
                    width=int(header["width"]),
                    height=int(header["height"]),
                    stream_fps=float(header["streamFps"]),
                    rgba_bytes=rgba_bytes,
                )
                response_json = CvAnalyzeResponse(metrics=metrics).model_dump_json(
                    by_alias=True
                )
                channel.send(response_json)
            except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
                pass  # Silently drop malformed frames.

    sdp_offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    await pc.setRemoteDescription(sdp_offer)
    answer = await pc.createAnswer()

    gathering_complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering() -> None:
        if pc.iceGatheringState == "complete":
            gathering_complete.set()

    await pc.setLocalDescription(answer)

    # Wait for ICE gathering so the returned SDP contains all candidates.
    if pc.iceGatheringState != "complete":
        try:
            await asyncio.wait_for(gathering_complete.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Return the partial SDP; the client can still connect with the
            # candidates gathered so far.
            pass

    return WebRtcAnswerResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
    )
