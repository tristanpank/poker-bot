"""WebRTC ingest service for real-time video frame processing.

Manages per-session RTCPeerConnections, receives video frames and DataChannel
metadata from the frontend, and feeds decoded frames into the CV pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

from backend.services.cv_service import CvService

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Per-frame metadata received via DataChannel from the frontend."""

    session_id: str
    frame_id: int
    capture_ts: int  # Unix epoch ms – as close to capture time as possible
    stream_fps: float = 0.0
    crop_width: int = 0
    crop_height: int = 0


class _SessionState:
    """State for a single active WebRTC ingest session."""

    def __init__(self) -> None:
        self.pc: Optional[RTCPeerConnection] = None
        self.datachannel = None  # Optional[RTCDataChannel]
        self.latest_meta: Optional[FrameMetadata] = None


class WebRtcIngestService:
    """
    Handles WebRTC peer connections for video ingest.

    For each session:
    - Accepts a SDP offer and returns a SDP answer.
    - Receives video frames from the remote track.
    - Receives per-frame metadata via a DataChannel.
    - Feeds decoded frames + metadata timestamps to CvService.
    - Sends CV metrics back to the frontend via the DataChannel.
    """

    def __init__(self, cv_service: CvService) -> None:
        self._cv_service = cv_service
        self._sessions: dict[str, _SessionState] = {}

    async def handle_offer(
        self, sdp: str, offer_type: str, session_id: str
    ) -> tuple[str, str]:
        """
        Process a WebRTC SDP offer and return the SDP answer.

        Returns:
            Tuple of (answer_sdp, answer_type).

        Raises:
            Exception: If the offer SDP is invalid or negotiation fails.
        """
        await self._cleanup_session(session_id)

        state = _SessionState()
        self._sessions[session_id] = state

        pc = RTCPeerConnection()
        state.pc = pc

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            state.datachannel = channel
            logger.debug(
                "DataChannel '%s' opened for session %s", channel.label, session_id
            )

            @channel.on("message")
            def on_message(message: str | bytes) -> None:
                if isinstance(message, bytes):
                    message = message.decode()
                try:
                    data = json.loads(message)
                    state.latest_meta = FrameMetadata(
                        session_id=data.get("sessionId", session_id),
                        frame_id=int(data.get("frameId", 0)),
                        capture_ts=int(data.get("captureTs", 0)),
                        stream_fps=float(data.get("streamFps", 0.0)),
                        crop_width=int(data.get("cropWidth", 0)),
                        crop_height=int(data.get("cropHeight", 0)),
                    )
                except Exception:
                    logger.debug(
                        "Failed to parse DataChannel metadata for session %s",
                        session_id,
                        exc_info=True,
                    )

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "video":
                asyncio.ensure_future(
                    self._process_video_track(track, session_id, state)
                )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.debug(
                "Connection state for session %s: %s",
                session_id,
                pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._cleanup_session(session_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc.localDescription.sdp, pc.localDescription.type

    async def _process_video_track(
        self, track, session_id: str, state: _SessionState
    ) -> None:
        """Receive decoded video frames and feed them into the CV pipeline."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break

            try:
                # Convert av.VideoFrame → RGB numpy array → RGBA bytes
                img = frame.to_ndarray(format="rgb24")
                h, w = img.shape[:2]
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                rgba = np.concatenate([img, alpha], axis=2)

                # Use client capture timestamp for accurate heart-rate estimation.
                meta = state.latest_meta
                if meta is not None and meta.capture_ts > 0:
                    timestamp = meta.capture_ts
                    stream_fps = meta.stream_fps
                else:
                    timestamp = int(time.time() * 1000)
                    stream_fps = 0.0

                metrics = self._cv_service.analyze_raw(
                    session_id=session_id,
                    timestamp=timestamp,
                    width=w,
                    height=h,
                    stream_fps=stream_fps,
                    rgba_bytes=rgba.tobytes(),
                )

                # Send CV metrics back to the frontend via the DataChannel.
                dc = state.datachannel
                if dc is not None and dc.readyState == "open":
                    dc.send(metrics.model_dump_json(by_alias=True))

                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                logger.exception(
                    "Error processing video frame for session %s (%d/%d)",
                    session_id,
                    consecutive_errors,
                    max_consecutive_errors,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "Too many consecutive errors for session %s; closing connection",
                        session_id,
                    )
                    await self._cleanup_session(session_id)
                    break

    async def _cleanup_session(self, session_id: str) -> None:
        """Close and remove a session's peer connection."""
        state = self._sessions.pop(session_id, None)
        if state is not None and state.pc is not None:
            try:
                await state.pc.close()
            except Exception:
                pass


_webrtc_service: Optional[WebRtcIngestService] = None


def get_webrtc_service() -> WebRtcIngestService:
    """Get singleton WebRTC ingest service instance."""
    global _webrtc_service
    if _webrtc_service is None:
        from backend.services.cv_service import get_cv_service

        _webrtc_service = WebRtcIngestService(get_cv_service())
    return _webrtc_service
