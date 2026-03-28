"""
Computer vision analysis service.

This service keeps the API contract stable for the frontend, but sources its
behavioral metrics from the upstream `theali29/Lie-Detector` detector adapter.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from backend.models.schemas import CvAnalyzeRequest, CvMetrics
from backend.services.theali29_lie_detector import TheAli29LieDetector


SESSION_TTL_MS = 5 * 60_000
BASELINE_WINDOW_MS = 5 * 60_000
BASELINE_MIN_MS = 20_000
SESSION_LOG_INTERVAL_MS = 15_000
SESSION_LOG_FRAME_INTERVAL = 300

logger = logging.getLogger(__name__)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _round1(value: float) -> float:
    return round(value, 1)


@dataclass
class BaselineBucket:
    second: int
    stress_sum: float = 0.0
    bluff_sum: float = 0.0
    count: int = 0


@dataclass
class CvSessionState:
    detector: TheAli29LieDetector = field(default_factory=TheAli29LieDetector)
    baseline_buckets: list[BaselineBucket] = field(default_factory=list)
    last_seen_at: int = 0
    frames_analyzed: int = 0
    last_log_at: int = 0
    last_analysis_monotonic_ms: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class CvService:
    """Backend CV analyzer with in-memory per-session state."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._sessions: dict[str, CvSessionState] = {}

    def clear_session(self, session_id: str) -> None:
        with self._registry_lock:
            removed = self._sessions.pop(session_id, None)

        if removed is not None:
            removed.detector.close()
            logger.info(
                "Cleared CV session %s after %d analyzed frames",
                session_id,
                removed.frames_analyzed,
            )

    def analyze(self, request: CvAnalyzeRequest) -> CvMetrics:
        raw_bytes = self._decode_base64_payload(request.rgba_base64)
        return self.analyze_raw(
            session_id=request.session_id,
            timestamp=int(request.timestamp),
            width=request.width,
            height=request.height,
            stream_fps=float(request.stream_fps),
            rgba_bytes=raw_bytes,
        )

    def analyze_raw(
        self,
        session_id: str,
        timestamp: int,
        width: int,
        height: int,
        stream_fps: float,
        rgba_bytes: bytes,
    ) -> CvMetrics:
        rgba = self._decode_raw_rgba_bytes(rgba_bytes, width, height)
        frame_rgb = rgba.reshape(height, width, 4)[:, :, :3].copy()

        stale_sessions: list[tuple[str, CvSessionState]] = []
        with self._registry_lock:
            stale_sessions = self._pop_stale_sessions(now_ms=timestamp)
            session = self._sessions.get(session_id)
            if session is None:
                session = CvSessionState(last_seen_at=timestamp)
                self._sessions[session_id] = session
                logger.info(
                    "Created CV session %s using upstream source %s for %dx%d stream",
                    session_id,
                    "theali29/Lie-Detector",
                    width,
                    height,
                )
            session.last_seen_at = timestamp

        for stale_session_id, stale_session in stale_sessions:
            stale_session.detector.close()
            logger.info(
                "Expired CV session %s after %d analyzed frames and %d ms idle",
                stale_session_id,
                stale_session.frames_analyzed,
                max(0, timestamp - stale_session.last_seen_at),
            )

        with session.lock:
            raw_result = session.detector.analyze(
                frame_rgb=frame_rgb,
                stream_fps=stream_fps,
                timestamp_ms=timestamp,
            )
            metrics = self._build_metrics(
                raw_result=raw_result,
                timestamp=timestamp,
                stream_fps=stream_fps,
                session=session,
            )
            session.frames_analyzed += 1
            self._maybe_log_session_metrics(
                session_id=session_id,
                width=width,
                height=height,
                session=session,
                metrics=metrics,
                timestamp=timestamp,
            )
            return metrics

    def _pop_stale_sessions(self, now_ms: int) -> list[tuple[str, CvSessionState]]:
        stale: list[tuple[str, CvSessionState]] = []
        for session_id, state in list(self._sessions.items()):
            if now_ms - state.last_seen_at > SESSION_TTL_MS:
                stale.append((session_id, self._sessions.pop(session_id)))
        return stale

    def _build_metrics(
        self,
        *,
        raw_result,
        timestamp: int,
        stream_fps: float,
        session: CvSessionState,
    ) -> CvMetrics:
        baseline_stress, baseline_bluff, baseline_progress = self._update_baseline(
            session=session,
            timestamp=timestamp,
            stress=raw_result.stress,
            bluff_risk=raw_result.bluff_risk,
        )
        stress_delta = raw_result.stress - baseline_stress
        bluff_delta = raw_result.bluff_risk - baseline_bluff

        analysis_fps = 0.0
        analysis_now_ms = time.perf_counter() * 1000.0
        if session.last_analysis_monotonic_ms is not None:
            dt = analysis_now_ms - session.last_analysis_monotonic_ms
            if dt > 0:
                analysis_fps = 1000.0 / dt
        session.last_analysis_monotonic_ms = analysis_now_ms

        quality_score = _clamp(
            0.5 * raw_result.pulse_confidence
            + 0.35 * raw_result.skin_coverage
            + 15.0 * raw_result.calibration_progress,
            0.0,
            100.0,
        )

        baseline_strength = baseline_progress
        if (
            session.baseline_buckets
            and (session.baseline_buckets[-1].second - session.baseline_buckets[0].second) * 1000
            < BASELINE_MIN_MS
        ):
            baseline_strength = 0.0

        return CvMetrics(
            brightness=raw_result.brightness,
            motion=raw_result.motion,
            edge_density=raw_result.edge_density,
            activity_zone=raw_result.activity_zone,
            pulse_bpm=raw_result.pulse_bpm,
            pulse_confidence=raw_result.pulse_confidence,
            skin_coverage=raw_result.skin_coverage,
            stress=raw_result.stress,
            emotion=raw_result.emotion,
            bluff_risk=raw_result.bluff_risk,
            bluff_level=self._derive_bluff_level(
                risk=raw_result.bluff_risk,
                bluff_delta=bluff_delta,
                stress_delta=stress_delta,
                baseline_strength=baseline_strength,
            ),
            baseline_progress=_round1(baseline_progress * 100.0),
            baseline_stress=_round1(_clamp(baseline_stress, 0.0, 100.0)),
            baseline_bluff=_round1(_clamp(baseline_bluff, 0.0, 100.0)),
            bluff_delta=_round1(_clamp(bluff_delta, -100.0, 100.0)),
            signal_quality=self._derive_signal_quality(quality_score),
            analysis_fps=_round1(analysis_fps),
            stream_fps=_round1(_clamp(stream_fps, 0.0, 120.0)),
            updated_at=datetime.fromtimestamp(timestamp / 1000).strftime("%H:%M:%S"),
            analysis_source=raw_result.analysis_source,
        )

    def _decode_base64_payload(self, rgba_base64: str) -> bytes:
        try:
            return base64.b64decode(rgba_base64, validate=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid base64 frame payload.") from exc

    def _decode_raw_rgba_bytes(self, raw: bytes, width: int, height: int) -> np.ndarray:
        expected_size = width * height * 4
        if len(raw) != expected_size:
            raise ValueError(
                f"Invalid RGBA payload length: expected {expected_size}, got {len(raw)}."
            )
        return np.frombuffer(raw, dtype=np.uint8)

    def _update_baseline(
        self,
        *,
        session: CvSessionState,
        timestamp: int,
        stress: float,
        bluff_risk: float,
    ) -> tuple[float, float, float]:
        second = timestamp // 1000
        if session.baseline_buckets and session.baseline_buckets[-1].second == second:
            bucket = session.baseline_buckets[-1]
            bucket.stress_sum += stress
            bucket.bluff_sum += bluff_risk
            bucket.count += 1
        else:
            session.baseline_buckets.append(
                BaselineBucket(
                    second=second,
                    stress_sum=stress,
                    bluff_sum=bluff_risk,
                    count=1,
                )
            )

        min_second = second - (BASELINE_WINDOW_MS // 1000)
        drop_count = 0
        for bucket in session.baseline_buckets:
            if bucket.second < min_second:
                drop_count += 1
            else:
                break
        if drop_count:
            session.baseline_buckets = session.baseline_buckets[drop_count:]

        if not session.baseline_buckets:
            return stress, bluff_risk, 0.0

        total_count = sum(bucket.count for bucket in session.baseline_buckets)
        if total_count <= 0:
            return stress, bluff_risk, 0.0

        total_stress = sum(bucket.stress_sum for bucket in session.baseline_buckets)
        total_bluff = sum(bucket.bluff_sum for bucket in session.baseline_buckets)
        baseline_stress = total_stress / total_count
        baseline_bluff = total_bluff / total_count

        window_span_sec = max(
            0,
            session.baseline_buckets[-1].second - session.baseline_buckets[0].second,
        )
        progress = _clamp(window_span_sec / (BASELINE_WINDOW_MS / 1000.0), 0.0, 1.0)
        return baseline_stress, baseline_bluff, progress

    def _derive_bluff_level(
        self,
        *,
        risk: float,
        bluff_delta: float,
        stress_delta: float,
        baseline_strength: float,
    ) -> str:
        if baseline_strength >= 0.2 and (risk >= 64 or bluff_delta >= 12 or stress_delta >= 10):
            return "elevated"
        if risk >= 38 or bluff_delta >= 6:
            return "watch"
        return "low"

    def _derive_signal_quality(self, quality_score: float) -> str:
        if quality_score >= 65:
            return "good"
        if quality_score >= 35:
            return "fair"
        return "poor"

    def _maybe_log_session_metrics(
        self,
        *,
        session_id: str,
        width: int,
        height: int,
        session: CvSessionState,
        metrics: CvMetrics,
        timestamp: int,
    ) -> None:
        should_log = session.frames_analyzed == 0
        if not should_log and timestamp - session.last_log_at >= SESSION_LOG_INTERVAL_MS:
            should_log = True
        if not should_log and session.frames_analyzed % SESSION_LOG_FRAME_INTERVAL == 0:
            should_log = True
        if not should_log:
            return

        session.last_log_at = timestamp
        logger.info(
            (
                "CV heartbeat session=%s source=%s frames=%d frame=%dx%d analysis_fps=%.1f "
                "stream_fps=%.1f quality=%s pulse_bpm=%s pulse_conf=%.1f "
                "stress=%.1f bluff=%.1f baseline=%.1f%%"
            ),
            session_id,
            metrics.analysis_source,
            session.frames_analyzed,
            width,
            height,
            metrics.analysis_fps,
            metrics.stream_fps,
            metrics.signal_quality,
            "none" if metrics.pulse_bpm is None else f"{metrics.pulse_bpm:.1f}",
            metrics.pulse_confidence,
            metrics.stress,
            metrics.bluff_risk,
            metrics.baseline_progress,
        )


_cv_service: Optional[CvService] = None


def get_cv_service() -> CvService:
    global _cv_service
    if _cv_service is None:
        _cv_service = CvService()
    return _cv_service
