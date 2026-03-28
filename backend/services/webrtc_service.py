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
from backend.services.session_service import save_webcam_metrics

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


@dataclass
class PendingFrame:
    """Latest decoded frame waiting to be analyzed."""

    frame: object
    meta: Optional[FrameMetadata]


class _SessionState:
    """State for a single active WebRTC ingest session."""

    def __init__(self) -> None:
        self.pc: Optional[RTCPeerConnection] = None
        self.datachannel = None  # Optional[RTCDataChannel]
        self.latest_meta: Optional[FrameMetadata] = None
        self.frames_received: int = 0
        self.metadata_messages: int = 0
        self.last_status_log_at: float = 0.0
        self.first_frame_logged: bool = False
        self.last_timestamp_used: int = 0
        self.timestamp_corrections: int = 0
        self.last_correction_log_count: int = 0
        self.last_frame_recv_monotonic_ms: Optional[float] = None
        self.inferred_stream_fps: float = 0.0
        self.last_frame_width: int = 0
        self.last_frame_height: int = 0
        self.last_analysis_fps: float = 0.0
        self.last_stream_fps: float = 0.0
        self.last_signal_quality: str = "poor"
        self.last_bluff_risk: float = 0.0
        self.connection_state: str = "new"
        self.pending_frame: Optional[PendingFrame] = None
        self.pending_frame_event = asyncio.Event()
        self.reader_task: Optional[asyncio.Task[None]] = None
        self.analysis_task: Optional[asyncio.Task[None]] = None


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

    def list_active_sessions(self) -> list[dict[str, object]]:
        """Return a debug snapshot of all active WebRTC ingest sessions."""
        sessions: list[dict[str, object]] = []
        for session_id, state in self._sessions.items():
            latest_meta = state.latest_meta
            sessions.append(
                {
                    "sessionId": session_id,
                    "connectionState": state.connection_state,
                    "framesReceived": state.frames_received,
                    "metadataMessages": state.metadata_messages,
                    "analysisFps": round(state.last_analysis_fps, 1),
                    "streamFps": round(state.last_stream_fps, 1),
                    "inferredStreamFps": round(state.inferred_stream_fps, 1),
                    "signalQuality": state.last_signal_quality,
                    "bluffRisk": round(state.last_bluff_risk, 1),
                    "frameWidth": state.last_frame_width,
                    "frameHeight": state.last_frame_height,
                    "latestFrameId": latest_meta.frame_id if latest_meta is not None else None,
                    "captureWidth": latest_meta.crop_width if latest_meta is not None else 0,
                    "captureHeight": latest_meta.crop_height if latest_meta is not None else 0,
                }
            )

        sessions.sort(key=lambda session: str(session["sessionId"]))
        return sessions

    @staticmethod
    def _is_newer_metadata(
        incoming: FrameMetadata, current: Optional[FrameMetadata]
    ) -> bool:
        if current is None:
            return True
        if incoming.capture_ts > current.capture_ts:
            return True
        if incoming.capture_ts == current.capture_ts and incoming.frame_id > current.frame_id:
            return True
        return False

    @staticmethod
    def _next_monotonic_timestamp(
        candidate_ts: int, stream_fps: float, state: _SessionState
    ) -> int:
        if candidate_ts > state.last_timestamp_used:
            state.last_timestamp_used = candidate_ts
            return candidate_ts

        step_ms = max(1, int(round(1000.0 / stream_fps))) if stream_fps > 0 else 1
        corrected = state.last_timestamp_used + step_ms
        state.last_timestamp_used = corrected
        state.timestamp_corrections += 1
        return corrected

    @staticmethod
    def _update_inferred_stream_fps(state: _SessionState) -> None:
        recv_now_ms = time.perf_counter() * 1000.0
        previous = state.last_frame_recv_monotonic_ms
        state.last_frame_recv_monotonic_ms = recv_now_ms

        if previous is None:
            return

        dt_ms = recv_now_ms - previous
        if dt_ms <= 0:
            return

        instant_fps = 1000.0 / dt_ms
        if not np.isfinite(instant_fps):
            return

        instant_fps = max(0.0, min(instant_fps, 120.0))
        if state.inferred_stream_fps <= 0:
            state.inferred_stream_fps = instant_fps
        else:
            state.inferred_stream_fps = state.inferred_stream_fps * 0.85 + instant_fps * 0.15

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
        logger.info("Starting WebRTC CV ingest session %s", session_id)

        pc = RTCPeerConnection()
        state.pc = pc

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            state.datachannel = channel
            logger.info(
                "DataChannel '%s' opened for session %s", channel.label, session_id
            )

            @channel.on("message")
            def on_message(message: str | bytes) -> None:
                if isinstance(message, bytes):
                    message = message.decode()
                try:
                    data = json.loads(message)
                    state.metadata_messages += 1
                    incoming_meta = FrameMetadata(
                        session_id=data.get("sessionId", session_id),
                        frame_id=int(data.get("frameId", 0)),
                        capture_ts=int(data.get("captureTs", 0)),
                        stream_fps=float(data.get("streamFps", 0.0)),
                        crop_width=int(data.get("cropWidth", 0)),
                        crop_height=int(data.get("cropHeight", 0)),
                    )
                    if self._is_newer_metadata(incoming_meta, state.latest_meta):
                        state.latest_meta = incoming_meta
                    if state.metadata_messages == 1:
                        logger.info(
                            (
                                "Received first CV metadata for session %s "
                                "(frame_id=%d capture_ts=%d stream_fps=%.1f crop=%dx%d)"
                            ),
                            session_id,
                            incoming_meta.frame_id,
                            incoming_meta.capture_ts,
                            incoming_meta.stream_fps,
                            incoming_meta.crop_width,
                            incoming_meta.crop_height,
                        )
                    elif state.metadata_messages % 300 == 0:
                        logger.info(
                            "CV metadata heartbeat session=%s messages=%d latest_frame_id=%d",
                            session_id,
                            state.metadata_messages,
                            state.latest_meta.frame_id if state.latest_meta is not None else -1,
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
                logger.info("Attached video track for CV session %s", session_id)
                state.reader_task = asyncio.create_task(
                    self._read_video_track(track, session_id, state)
                )
                state.analysis_task = asyncio.create_task(
                    self._analyze_video_frames(session_id, state)
                )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state.connection_state = pc.connectionState
            logger.info(
                "Connection state for session %s: %s",
                session_id,
                pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._cleanup_session(session_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        asyncio.create_task(self._watch_session_startup(session_id, state))

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
                state.frames_received += 1
                state.last_frame_width = w
                state.last_frame_height = h
                self._update_inferred_stream_fps(state)
                if not state.first_frame_logged:
                    state.first_frame_logged = True
                    logger.info(
                        "Received first decoded video frame for CV session %s (%dx%d)",
                        session_id,
                        w,
                        h,
                    )

                # Use client capture timestamp for accurate heart-rate estimation.
                meta = state.latest_meta
                if meta is not None and meta.capture_ts > 0:
                    stream_fps = (
                        meta.stream_fps if meta.stream_fps > 0 else state.inferred_stream_fps
                    )
                    timestamp = self._next_monotonic_timestamp(
                        meta.capture_ts, stream_fps, state
                    )
                else:
                    stream_fps = state.inferred_stream_fps
                    timestamp = self._next_monotonic_timestamp(
                        int(time.time() * 1000), stream_fps, state
                    )

                should_log_correction = False
                if (
                    state.timestamp_corrections > 0
                    and state.timestamp_corrections != state.last_correction_log_count
                ):
                    if state.timestamp_corrections == 1:
                        should_log_correction = True
                    elif state.timestamp_corrections % 300 == 0:
                        should_log_correction = True

                if should_log_correction:
                    state.last_correction_log_count = state.timestamp_corrections
                    logger.warning(
                        (
                            "Corrected non-monotonic CV timestamps for session %s "
                            "(corrections=%d latest_frame_id=%s)"
                        ),
                        session_id,
                        state.timestamp_corrections,
                        meta.frame_id if meta is not None else "none",
                    )

                # Offload CPU-heavy CV analysis so multiple active webcams
                # do not block HTTP request handling on the asyncio loop.
                metrics = await asyncio.to_thread(
                    self._cv_service.analyze_raw,
                    session_id=session_id,
                    timestamp=timestamp,
                    width=w,
                    height=h,
                    stream_fps=stream_fps,
                    rgba_bytes=rgba.tobytes(),
                )
                state.last_analysis_fps = metrics.analysis_fps
                state.last_stream_fps = metrics.stream_fps
                state.last_signal_quality = metrics.signal_quality
                state.last_bluff_risk = metrics.bluff_risk

                now = time.time()
                if (
                    state.frames_received == 1
                    or now - state.last_status_log_at >= 15.0
                    or state.frames_received % 300 == 0
                ):
                    state.last_status_log_at = now
                    logger.info(
                        (
                            "WebRTC CV heartbeat session=%s frames=%d size=%dx%d "
                            "latest_frame_id=%s analysis_fps=%.1f stream_fps=%.1f "
                            "quality=%s bluff=%.1f"
                        ),
                        session_id,
                        state.frames_received,
                        w,
                        h,
                        meta.frame_id if meta is not None else "none",
                        metrics.analysis_fps,
                        metrics.stream_fps,
                        metrics.signal_quality,
                        metrics.bluff_risk,
                    )

                # Send CV metrics back to the frontend via the DataChannel.
                metrics_json = metrics.model_dump_json(by_alias=True)
                dc = state.datachannel
                if dc is not None and dc.readyState == "open":
                    dc.send(metrics_json)

                # Persist metrics to Redis for the Play page to poll
                asyncio.create_task(save_webcam_metrics(session_id, metrics_json))

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

    async def _read_video_track(
        self, track, session_id: str, state: _SessionState
    ) -> None:
        """Receive decoded video frames as quickly as possible."""
        while True:
            try:
                frame = await track.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                state.pending_frame_event.set()
                break

            state.frames_received += 1
            self._update_inferred_stream_fps(state)
            state.pending_frame = PendingFrame(frame=frame, meta=state.latest_meta)
            state.pending_frame_event.set()

    async def _analyze_video_frames(
        self, session_id: str, state: _SessionState
    ) -> None:
        """Analyze only the latest frame so receive FPS is not analysis-bound."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        while True:
            try:
                await state.pending_frame_event.wait()
                pending = state.pending_frame
                state.pending_frame = None
                state.pending_frame_event.clear()

                if pending is None:
                    if session_id not in self._sessions:
                        break
                    continue

                frame = pending.frame
                meta = pending.meta

                img = frame.to_ndarray(format="rgb24")
                h, w = img.shape[:2]
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                rgba = np.concatenate([img, alpha], axis=2)
                state.last_frame_width = w
                state.last_frame_height = h
                if not state.first_frame_logged:
                    state.first_frame_logged = True
                    logger.info(
                        "Received first decoded video frame for CV session %s (%dx%d)",
                        session_id,
                        w,
                        h,
                    )

                if meta is not None and meta.capture_ts > 0:
                    stream_fps = (
                        meta.stream_fps if meta.stream_fps > 0 else state.inferred_stream_fps
                    )
                    timestamp = self._next_monotonic_timestamp(
                        meta.capture_ts, stream_fps, state
                    )
                else:
                    stream_fps = state.inferred_stream_fps
                    timestamp = self._next_monotonic_timestamp(
                        int(time.time() * 1000), stream_fps, state
                    )

                should_log_correction = False
                if (
                    state.timestamp_corrections > 0
                    and state.timestamp_corrections != state.last_correction_log_count
                ):
                    if state.timestamp_corrections == 1:
                        should_log_correction = True
                    elif state.timestamp_corrections % 300 == 0:
                        should_log_correction = True

                if should_log_correction:
                    state.last_correction_log_count = state.timestamp_corrections
                    logger.warning(
                        (
                            "Corrected non-monotonic CV timestamps for session %s "
                            "(corrections=%d latest_frame_id=%s)"
                        ),
                        session_id,
                        state.timestamp_corrections,
                        meta.frame_id if meta is not None else "none",
                    )

                metrics = await asyncio.to_thread(
                    self._cv_service.analyze_raw,
                    session_id=session_id,
                    timestamp=timestamp,
                    width=w,
                    height=h,
                    stream_fps=stream_fps,
                    rgba_bytes=rgba.tobytes(),
                )
                state.last_analysis_fps = metrics.analysis_fps
                state.last_stream_fps = metrics.stream_fps
                state.last_signal_quality = metrics.signal_quality
                state.last_bluff_risk = metrics.bluff_risk

                now = time.time()
                if (
                    state.frames_received == 1
                    or now - state.last_status_log_at >= 15.0
                    or state.frames_received % 300 == 0
                ):
                    state.last_status_log_at = now
                    logger.info(
                        (
                            "WebRTC CV heartbeat session=%s frames=%d size=%dx%d "
                            "latest_frame_id=%s analysis_fps=%.1f stream_fps=%.1f "
                            "quality=%s bluff=%.1f"
                        ),
                        session_id,
                        state.frames_received,
                        w,
                        h,
                        meta.frame_id if meta is not None else "none",
                        metrics.analysis_fps,
                        metrics.stream_fps,
                        metrics.signal_quality,
                        metrics.bluff_risk,
                    )

                metrics_json = metrics.model_dump_json(by_alias=True)
                dc = state.datachannel
                if dc is not None and dc.readyState == "open":
                    dc.send(metrics_json)

                asyncio.create_task(save_webcam_metrics(session_id, metrics_json))
                consecutive_errors = 0

            except asyncio.CancelledError:
                raise
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

    async def _watch_session_startup(
        self, session_id: str, state: _SessionState
    ) -> None:
        """Log startup gaps so we can tell which WebRTC stage is failing."""
        await asyncio.sleep(5)
        active = self._sessions.get(session_id)
        if active is not state:
            return

        if state.datachannel is None:
            logger.warning(
                "CV session %s has no open DataChannel 5s after offer negotiation",
                session_id,
            )

        if state.metadata_messages == 0:
            logger.warning(
                "CV session %s has not received metadata messages 5s after startup",
                session_id,
            )

        if state.frames_received == 0:
            logger.warning(
                "CV session %s has not received decoded video frames 5s after startup",
                session_id,
            )

    async def _cleanup_session(self, session_id: str) -> None:
        """Close and remove a session's peer connection."""
        state = self._sessions.pop(session_id, None)
        if state is not None:
            logger.info(
                "Closing WebRTC CV session %s after %d frames and %d metadata messages",
                session_id,
                state.frames_received,
                state.metadata_messages,
            )
            state.pending_frame = None
            state.pending_frame_event.set()
            tasks = [state.reader_task, state.analysis_task]
            for task in tasks:
                if task is not None:
                    task.cancel()
            for task in tasks:
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
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
