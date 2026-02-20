"""
Computer vision analysis service for bluff/stress proxy metrics.

Processes frame payloads from the frontend and maintains per-session
state for temporal signals such as POS pulse estimation.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

import numpy as np

from backend.models.schemas import CvAnalyzeRequest, CvMetrics


PIXEL_STRIDE = 1  # Full-resolution pixel sampling (no spatial skipping).
PULSE_WIN_MS = 60_000
PULSE_KEEP_MS = 75_000
SESSION_TTL_MS = 5 * 60_000
BASELINE_WINDOW_MS = 5 * 60_000
BASELINE_MIN_MS = 20_000
ROI_SEARCH_RADIUS_X = 8
ROI_SEARCH_RADIUS_Y = 6
ROI_SEARCH_STEP = 2
ActivityZone = Literal["none", "left", "center", "right"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _round1(value: float) -> float:
    return round(value, 1)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()

    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _normalize(signal: np.ndarray) -> np.ndarray:
    centered = signal - float(signal.mean())
    scale = float(np.std(centered, ddof=1)) if signal.size > 1 else 0.0
    if scale <= 1e-9:
        return centered
    return centered / scale


@dataclass
class PulseSample:
    timestamp: int
    r: float
    g: float
    b: float


@dataclass
class BaselineBucket:
    second: int
    stress_sum: float = 0.0
    bluff_sum: float = 0.0
    count: int = 0


@dataclass
class CvSessionState:
    prev_luma: Optional[np.ndarray] = None
    pulse_samples: list[PulseSample] = field(default_factory=list)
    smooth_pulse: Optional[float] = None
    pulse_baseline: Optional[float] = None
    prev_stress: Optional[float] = None
    last_analysis_ts: Optional[int] = None
    roi_center_x: Optional[float] = None
    roi_center_y: Optional[float] = None
    roi_width: Optional[float] = None
    roi_height: Optional[float] = None
    roi_stability: float = 0.0
    baseline_buckets: list[BaselineBucket] = field(default_factory=list)
    last_seen_at: int = 0


class CvService:
    """Backend CV analyzer with in-memory per-session state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, CvSessionState] = {}
        self._hann_cache: dict[int, np.ndarray] = {}

    def clear_session(self, session_id: str) -> None:
        """Delete a session's analysis state."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def analyze(self, request: CvAnalyzeRequest) -> CvMetrics:
        """
        Run CV analysis and return computed metrics.

        Raises:
            ValueError: If the frame payload is malformed.
        """
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
        """
        Run CV analysis directly from raw RGBA bytes.

        Raises:
            ValueError: If frame payload is malformed.
        """
        rgba = self._decode_raw_rgba_bytes(rgba_bytes, width, height)
        frame = rgba.reshape(height, width, 4)

        with self._lock:
            self._cleanup_sessions(timestamp)
            session = self._sessions.get(session_id)
            if session is None:
                session = CvSessionState(last_seen_at=timestamp)
                self._sessions[session_id] = session

            session.last_seen_at = timestamp
            metrics = self._compute_metrics(
                frame=frame,
                width=width,
                height=height,
                timestamp=timestamp,
                stream_fps=stream_fps,
                session=session,
            )

        return metrics

    def _cleanup_sessions(self, now_ms: int) -> None:
        stale_ids = [
            session_id
            for session_id, state in self._sessions.items()
            if now_ms - state.last_seen_at > SESSION_TTL_MS
        ]
        for session_id in stale_ids:
            self._sessions.pop(session_id, None)

    def _hann_window(self, n: int) -> np.ndarray:
        window = self._hann_cache.get(n)
        if window is None:
            window = np.hanning(n)
            self._hann_cache[n] = window
        return window

    def _welch_psd(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray] | None:
        n = signal.size
        if n < 48:
            return None

        seg_len = min(n, max(64, int(fs * 6.0)))
        step = max(1, seg_len // 2)
        if seg_len <= 2:
            return None

        window = self._hann_window(seg_len)
        window_power = float(np.sum(window * window)) + 1e-9

        acc_power: Optional[np.ndarray] = None
        segments = 0

        for start in range(0, n - seg_len + 1, step):
            segment = signal[start : start + seg_len]
            segment = segment - float(segment.mean())
            spectrum = np.fft.rfft(segment * window)
            power = (np.abs(spectrum) ** 2) / window_power
            if acc_power is None:
                acc_power = power
            else:
                acc_power += power
            segments += 1

        if acc_power is None or segments == 0:
            return None

        avg_power = acc_power / float(segments)
        freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
        return freqs, avg_power

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
        session: CvSessionState,
        timestamp: int,
        stress: float,
        bluff_risk: float,
    ) -> tuple[float, float, float]:
        second = timestamp // 1000
        if (
            session.baseline_buckets
            and session.baseline_buckets[-1].second == second
        ):
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
        while session.baseline_buckets and session.baseline_buckets[0].second < min_second:
            session.baseline_buckets.pop(0)

        if not session.baseline_buckets:
            return stress, bluff_risk, 0.0

        total_count = 0
        total_stress = 0.0
        total_bluff = 0.0
        for bucket in session.baseline_buckets:
            total_count += bucket.count
            total_stress += bucket.stress_sum
            total_bluff += bucket.bluff_sum

        if total_count <= 0:
            return stress, bluff_risk, 0.0

        baseline_stress = total_stress / total_count
        baseline_bluff = total_bluff / total_count
        window_span_sec = max(
            0,
            session.baseline_buckets[-1].second - session.baseline_buckets[0].second,
        )
        progress = _clamp(window_span_sec / (BASELINE_WINDOW_MS / 1000.0), 0.0, 1.0)
        return baseline_stress, baseline_bluff, progress

    def _compute_metrics(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        timestamp: int,
        stream_fps: float,
        session: CvSessionState,
    ) -> CvMetrics:
        luma = self._compute_luma(frame)
        base = self._analyze_base_signals(
            luma=luma,
            width=width,
            height=height,
            prev_luma=session.prev_luma,
        )
        session.prev_luma = luma

        skin = self._sample_skin_signal(frame, width, height, session)
        session.pulse_samples.append(
            PulseSample(timestamp=timestamp, r=skin["r"], g=skin["g"], b=skin["b"])
        )
        session.pulse_samples = [
            sample
            for sample in session.pulse_samples
            if timestamp - sample.timestamp <= PULSE_KEEP_MS
        ]

        pulse_bpm, pulse_confidence = self._estimate_pos_pulse(session.pulse_samples)

        if pulse_bpm is not None:
            if session.smooth_pulse is None:
                smooth = pulse_bpm
            else:
                smooth = session.smooth_pulse * 0.92 + pulse_bpm * 0.08
            session.smooth_pulse = _round1(smooth)
            pulse_bpm = session.smooth_pulse

            if session.pulse_baseline is None and pulse_confidence > 35:
                session.pulse_baseline = smooth
            elif (
                session.pulse_baseline is not None
                and pulse_confidence > 25
                and base["motion"] < 35
            ):
                session.pulse_baseline = session.pulse_baseline * 0.992 + smooth * 0.008

        pulse_delta = (
            abs(pulse_bpm - session.pulse_baseline)
            if pulse_bpm is not None and session.pulse_baseline is not None
            else 0.0
        )
        pulse_stress = _clamp(pulse_delta * 2.3, 0.0, 100.0)

        quality_score = _clamp(
            0.45 * skin["coverage"]
            + 0.35 * pulse_confidence
            + 0.2 * (100 - min(100, abs(base["brightness"] - 55) * 1.8)),
            0.0,
            100.0,
        )

        if pulse_bpm is None and quality_score < 45:
            pulse_confidence = _round1(max(0.0, pulse_confidence - 12))

        stress = _round1(
            _clamp(
                (0.5 * base["motion"] + 0.5 * pulse_stress) * (0.65 + quality_score / 200),
                0.0,
                100.0,
            )
        )

        stress_trend = 0.0 if session.prev_stress is None else stress - session.prev_stress
        session.prev_stress = stress

        emotion = self._derive_emotion(stress, base["motion"], quality_score)
        stillness = 60.0 if base["motion"] < 20 else max(0.0, 35.0 - base["motion"])
        raw_bluff_risk = _round1(
            _clamp(
                0.5 * stress
                + 0.24 * pulse_stress
                + 0.18 * max(0.0, stress_trend * 3)
                + 0.08 * stillness,
                0.0,
                100.0,
            )
        )
        baseline_stress, baseline_bluff, baseline_progress = self._update_baseline(
            session=session,
            timestamp=timestamp,
            stress=stress,
            bluff_risk=raw_bluff_risk,
        )
        stress_delta = stress - baseline_stress
        bluff_delta = raw_bluff_risk - baseline_bluff
        baseline_strength = baseline_progress
        if (
            session.baseline_buckets
            and (session.baseline_buckets[-1].second - session.baseline_buckets[0].second) * 1000
            < BASELINE_MIN_MS
        ):
            baseline_strength = 0.0

        analysis_fps = 0.0
        if session.last_analysis_ts is not None:
            dt = timestamp - session.last_analysis_ts
            if dt > 0:
                analysis_fps = 1000.0 / dt
        if stream_fps > 0:
            analysis_fps = min(analysis_fps, stream_fps + 0.5)
        session.last_analysis_ts = timestamp

        return CvMetrics(
            brightness=base["brightness"],
            motion=base["motion"],
            edge_density=base["edge_density"],
            activity_zone=base["activity_zone"],
            pulse_bpm=pulse_bpm,
            pulse_confidence=_round1(pulse_confidence),
            skin_coverage=skin["coverage"],
            stress=stress,
            emotion=emotion,
            bluff_risk=raw_bluff_risk,
            bluff_level=self._derive_bluff_level(
                risk=raw_bluff_risk,
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
        )

    def _compute_luma(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.uint16)
        green = frame[:, :, 1].astype(np.uint16)
        blue = frame[:, :, 2].astype(np.uint16)
        return ((77 * red + 150 * green + 29 * blue) >> 8).astype(np.uint8)

    def _compute_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        rgb = frame[:, :, :3].astype(np.int16)
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]

        max_rgb = np.maximum(np.maximum(red, green), blue)
        min_rgb = np.minimum(np.minimum(red, green), blue)

        return (
            (red > 45)
            & (green > 30)
            & (blue > 20)
            & (red > green)
            & (red > blue)
            & ((max_rgb - min_rgb) > 15)
            & (np.abs(red - green) > 10)
        )

    def _rect_skin_sum(
        self,
        integral: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> int:
        total = int(integral[y1 - 1, x1 - 1])
        if x0 > 0:
            total -= int(integral[y1 - 1, x0 - 1])
        if y0 > 0:
            total -= int(integral[y0 - 1, x1 - 1])
        if x0 > 0 and y0 > 0:
            total += int(integral[y0 - 1, x0 - 1])
        return total

    def _stabilize_face_roi(
        self,
        skin_mask: np.ndarray,
        width: int,
        height: int,
        session: CvSessionState,
    ) -> tuple[int, int, int, int, float]:
        ys, xs = np.nonzero(skin_mask)
        min_detect_pixels = max(36, (width * height) // 300)
        has_detection = xs.size >= min_detect_pixels

        if has_detection:
            cand_cx = float(xs.mean())
            cand_cy = float(ys.mean())
            spread_x = float(np.std(xs))
            spread_y = float(np.std(ys))
            cand_w = float(_clamp(spread_x * 3.4, width * 0.24, width * 0.66))
            cand_h = float(_clamp(spread_y * 3.8, height * 0.30, height * 0.78))
        else:
            cand_cx = width * 0.5
            cand_cy = height * 0.34
            cand_w = width * 0.40
            cand_h = height * 0.52

        if session.roi_center_x is None:
            cx = cand_cx
            cy = cand_cy
            roi_w = cand_w
            roi_h = cand_h
            stability = 0.45 if has_detection else 0.0
        else:
            blend = 0.32 if has_detection else 0.08
            cx = session.roi_center_x * (1.0 - blend) + cand_cx * blend
            cy = session.roi_center_y * (1.0 - blend) + cand_cy * blend
            roi_w = session.roi_width * (1.0 - blend) + cand_w * blend
            roi_h = session.roi_height * (1.0 - blend) + cand_h * blend

            integral = (
                skin_mask.astype(np.uint8)
                .cumsum(axis=0, dtype=np.int32)
                .cumsum(axis=1, dtype=np.int32)
            )
            half_w = max(5, int(round(roi_w * 0.5)))
            half_h = max(5, int(round(roi_h * 0.5)))
            base_cx = int(round(cx))
            base_cy = int(round(cy))
            best_cx = base_cx
            best_cy = base_cy
            best_score = -1

            for dy in range(-ROI_SEARCH_RADIUS_Y, ROI_SEARCH_RADIUS_Y + 1, ROI_SEARCH_STEP):
                for dx in range(-ROI_SEARCH_RADIUS_X, ROI_SEARCH_RADIUS_X + 1, ROI_SEARCH_STEP):
                    probe_cx = base_cx + dx
                    probe_cy = base_cy + dy
                    x0 = probe_cx - half_w
                    y0 = probe_cy - half_h
                    x1 = probe_cx + half_w
                    y1 = probe_cy + half_h
                    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
                        continue
                    score = self._rect_skin_sum(integral, x0, y0, x1, y1)
                    if score > best_score:
                        best_score = score
                        best_cx = probe_cx
                        best_cy = probe_cy

            cx = cx * 0.7 + float(best_cx) * 0.3
            cy = cy * 0.7 + float(best_cy) * 0.3
            stability = _clamp(
                session.roi_stability * 0.72 + (0.28 if has_detection else 0.06),
                0.0,
                1.0,
            )

        roi_w = float(_clamp(roi_w, width * 0.22, width * 0.70))
        roi_h = float(_clamp(roi_h, height * 0.28, height * 0.82))
        half_w = roi_w * 0.5
        half_h = roi_h * 0.5
        cx = float(_clamp(cx, half_w + 1.0, width - half_w - 1.0))
        cy = float(_clamp(cy, half_h + 1.0, height - half_h - 1.0))

        x0 = int(round(cx - half_w))
        y0 = int(round(cy - half_h))
        x1 = int(round(cx + half_w))
        y1 = int(round(cy + half_h))
        x0 = max(0, min(width - 2, x0))
        y0 = max(0, min(height - 2, y0))
        x1 = max(x0 + 2, min(width, x1))
        y1 = max(y0 + 2, min(height, y1))

        session.roi_center_x = cx
        session.roi_center_y = cy
        session.roi_width = float(x1 - x0)
        session.roi_height = float(y1 - y0)
        session.roi_stability = stability

        return x0, y0, x1, y1, stability

    def _analyze_base_signals(
        self,
        luma: np.ndarray,
        width: int,
        height: int,
        prev_luma: Optional[np.ndarray],
    ) -> dict[str, float | str]:
        sample = luma[1:height:PIXEL_STRIDE, 1:width:PIXEL_STRIDE]
        if sample.size == 0:
            return {
                "brightness": 0.0,
                "motion": 0.0,
                "edge_density": 0.0,
                "activity_zone": "none",
            }

        brightness = _round1(float(sample.mean()) / 255.0 * 100.0)

        left = luma[1:height:PIXEL_STRIDE, 0:width:PIXEL_STRIDE]
        top = luma[0:height:PIXEL_STRIDE, 1:width:PIXEL_STRIDE]
        h = min(sample.shape[0], left.shape[0], top.shape[0])
        w = min(sample.shape[1], left.shape[1], top.shape[1])
        sample_c = sample[:h, :w].astype(np.int16)
        left_c = left[:h, :w].astype(np.int16)
        top_c = top[:h, :w].astype(np.int16)
        edge_mask = (np.abs(sample_c - left_c) + np.abs(sample_c - top_c)) > 54
        edge_density = _round1(float(edge_mask.sum()) / max(1, edge_mask.size) * 100.0)

        motion = 0.0
        activity_zone: ActivityZone = "none"

        if prev_luma is not None and prev_luma.shape == luma.shape:
            prev_sample = prev_luma[1:height:PIXEL_STRIDE, 1:width:PIXEL_STRIDE].astype(np.int16)
            curr_sample = sample.astype(np.int16)
            delta = np.abs(curr_sample - prev_sample)
            moving = delta > 14

            motion_sum = float(delta[moving].sum())
            motion = min(100.0, (motion_sum / (sample.size * 255.0)) * 300.0)
            motion = _round1(motion)

            if motion_sum > sample.size * 8:
                x_coords = np.arange(
                    1,
                    1 + sample.shape[1] * PIXEL_STRIDE,
                    PIXEL_STRIDE,
                    dtype=np.float32,
                )
                x_coords = x_coords[: sample.shape[1]]
                x_grid = np.broadcast_to(x_coords, sample.shape)
                motion_x = float((delta * moving * x_grid).sum())
                center_x = motion_x / motion_sum if motion_sum > 0 else width / 2

                if center_x < width / 3:
                    activity_zone = "left"
                elif center_x > (2 * width) / 3:
                    activity_zone = "right"
                else:
                    activity_zone = "center"

        return {
            "brightness": brightness,
            "motion": motion,
            "edge_density": edge_density,
            "activity_zone": activity_zone,
        }

    def _sample_skin_signal(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        session: CvSessionState,
    ) -> dict[str, float]:
        skin_mask = self._compute_skin_mask(frame)
        roi_x0, roi_y0, roi_x1, roi_y1, roi_stability = self._stabilize_face_roi(
            skin_mask=skin_mask,
            width=width,
            height=height,
            session=session,
        )

        rgb = frame[:, :, :3].astype(np.int16)
        roi_w = max(1, roi_x1 - roi_x0)
        roi_h = max(1, roi_y1 - roi_y0)

        patches = [
            (
                roi_x0 + int(roi_w * 0.12),
                roi_y0 + int(roi_h * 0.40),
                roi_x0 + int(roi_w * 0.40),
                roi_y0 + int(roi_h * 0.76),
            ),
            (
                roi_x0 + int(roi_w * 0.60),
                roi_y0 + int(roi_h * 0.40),
                roi_x0 + int(roi_w * 0.88),
                roi_y0 + int(roi_h * 0.76),
            ),
            (
                roi_x0 + int(roi_w * 0.28),
                roi_y0 + int(roi_h * 0.14),
                roi_x0 + int(roi_w * 0.72),
                roi_y0 + int(roi_h * 0.34),
            ),
            (
                roi_x0 + int(roi_w * 0.38),
                roi_y0 + int(roi_h * 0.44),
                roi_x0 + int(roi_w * 0.62),
                roi_y0 + int(roi_h * 0.66),
            ),
        ]

        visited = 0
        accepted = 0
        accepted_r = 0.0
        accepted_g = 0.0
        accepted_b = 0.0
        fallback_r = 0.0
        fallback_g = 0.0
        fallback_b = 0.0

        for x0, y0, x1, y1 in patches:
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(x0 + 1, min(width, x1))
            y1 = max(y0 + 1, min(height, y1))

            region = rgb[y0:y1:PIXEL_STRIDE, x0:x1:PIXEL_STRIDE]
            if region.size == 0:
                continue
            region_mask = skin_mask[y0:y1:PIXEL_STRIDE, x0:x1:PIXEL_STRIDE]
            r = region[:, :, 0]
            g = region[:, :, 1]
            b = region[:, :, 2]

            count = region.shape[0] * region.shape[1]
            visited += int(count)
            fallback_r += float(r.sum())
            fallback_g += float(g.sum())
            fallback_b += float(b.sum())

            kept = int(region_mask.sum())
            accepted += kept
            if kept > 0:
                accepted_r += float(r[region_mask].sum())
                accepted_g += float(g[region_mask].sum())
                accepted_b += float(b[region_mask].sum())

        if visited == 0:
            return {"r": 0.0, "g": 0.0, "b": 0.0, "coverage": 0.0}

        if accepted > 0:
            rr = accepted_r / (accepted * 255.0)
            gg = accepted_g / (accepted * 255.0)
            bb = accepted_b / (accepted * 255.0)
        else:
            rr = fallback_r / (visited * 255.0)
            gg = fallback_g / (visited * 255.0)
            bb = fallback_b / (visited * 255.0)

        coverage = (accepted / max(1, visited)) * 100.0
        coverage *= 0.72 + 0.28 * roi_stability

        return {
            "r": rr,
            "g": gg,
            "b": bb,
            "coverage": _round1(_clamp(coverage, 0.0, 100.0)),
        }

    def _estimate_pos_pulse(self, samples: list[PulseSample]) -> tuple[Optional[float], float]:
        if len(samples) < 60:
            return None, 0.0

        latest = samples[-1].timestamp
        windowed = [sample for sample in samples if latest - sample.timestamp <= PULSE_WIN_MS]
        if len(windowed) < 60:
            return None, 0.0

        t_ms = np.array([sample.timestamp for sample in windowed], dtype=np.float64)
        r = np.array([sample.r for sample in windowed], dtype=np.float64)
        g = np.array([sample.g for sample in windowed], dtype=np.float64)
        b = np.array([sample.b for sample in windowed], dtype=np.float64)

        # Keep strictly increasing timestamps only.
        if t_ms.size < 2:
            return None, 0.0
        keep_idx = np.concatenate(([0], np.where(np.diff(t_ms) > 0)[0] + 1))
        t_ms = t_ms[keep_idx]
        r = r[keep_idx]
        g = g[keep_idx]
        b = b[keep_idx]
        if t_ms.size < 60:
            return None, 0.0

        dt_ms = np.diff(t_ms)
        if dt_ms.size == 0:
            return None, 0.0
        native_fs = 1000.0 / float(np.median(dt_ms))
        if not np.isfinite(native_fs) or native_fs < 8:
            return None, 0.0

        t_sec = (t_ms - t_ms[0]) / 1000.0
        duration_sec = float(t_sec[-1])
        if duration_sec < 8.0:
            return None, 0.0

        uniform_fs = float(_clamp(native_fs, 24.0, 45.0))
        t_uniform = np.arange(0.0, duration_sec, 1.0 / uniform_fs, dtype=np.float64)
        if t_uniform.size < 64:
            return None, 0.0

        r_u = np.interp(t_uniform, t_sec, r)
        g_u = np.interp(t_uniform, t_sec, g)
        b_u = np.interp(t_uniform, t_sec, b)

        trend_win = max(3, int(uniform_fs * 1.2))
        r_d = _normalize(r_u - _moving_average(r_u, trend_win))
        g_d = _normalize(g_u - _moving_average(g_u, trend_win))
        b_d = _normalize(b_u - _moving_average(b_u, trend_win))

        x = g_d - b_d
        y = -2.0 * r_d + g_d + b_d
        std_y = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        alpha = (
            float(np.std(x, ddof=1)) / std_y
            if std_y > 1e-6 and x.size > 1
            else 0.0
        )

        pulse = x + alpha * y
        pulse = pulse - _moving_average(pulse, max(3, int(uniform_fs * 0.8)))
        pulse = np.tanh(_normalize(pulse) * 1.4)

        welch = self._welch_psd(pulse, uniform_fs)
        if welch is None:
            return None, 0.0
        freqs, power = welch

        band_mask = (freqs >= 0.7) & (freqs <= 3.2)
        if not np.any(band_mask):
            return None, 0.0
        band_freqs = freqs[band_mask]
        band_power = power[band_mask]
        if band_power.size < 3:
            return None, 0.0

        peak_idx = int(np.argmax(band_power))
        peak_freq = float(band_freqs[peak_idx])
        peak_power = float(band_power[peak_idx])
        if peak_power <= 0:
            return None, 0.0

        sorted_power = np.sort(band_power)
        second_peak_power = float(sorted_power[-2]) if sorted_power.size > 1 else 0.0
        peak_ratio = peak_power / (second_peak_power + 1e-9)
        total_band_power = float(band_power.sum())
        local_mask = np.abs(band_freqs - peak_freq) <= 0.08
        concentration = float(band_power[local_mask].sum()) / (total_band_power + 1e-9)

        pulse_centered = pulse - float(pulse.mean())
        acf = np.correlate(pulse_centered, pulse_centered, mode="full")
        acf = acf[acf.size // 2 :]
        if acf.size < 3 or acf[0] <= 1e-9:
            return None, 0.0
        acf = acf / acf[0]

        min_lag = max(1, int(uniform_fs / 3.2))
        max_lag = min(acf.size - 1, int(uniform_fs / 0.7))
        if max_lag <= min_lag:
            return None, 0.0
        acf_band = acf[min_lag : max_lag + 1]
        acf_peak_lag = int(np.argmax(acf_band)) + min_lag
        acf_peak_value = float(acf[acf_peak_lag])
        acf_freq = float(uniform_fs / acf_peak_lag)

        spec_score = _clamp(
            0.6 * concentration + 0.4 * _clamp((peak_ratio - 1.0) / 2.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        acf_score = _clamp((acf_peak_value - 0.15) / 0.55, 0.0, 1.0)

        if abs(acf_freq - peak_freq) <= 0.22:
            best_hz = (0.7 * peak_freq) + (0.3 * acf_freq)
        elif acf_score > spec_score + 0.12:
            best_hz = acf_freq
        else:
            best_hz = peak_freq

        best_hz = float(_clamp(best_hz, 0.7, 3.2))
        confidence = _clamp(
            100.0 * (0.55 * spec_score + 0.35 * acf_score + 0.10 * _clamp(duration_sec / 20.0, 0.0, 1.0)),
            0.0,
            100.0,
        )

        bpm = _round1(best_hz * 60.0)
        return bpm, _round1(confidence)

    def _derive_signal_quality(self, score: float) -> str:
        if score >= 70:
            return "good"
        if score >= 45:
            return "fair"
        return "poor"

    def _derive_emotion(self, stress: float, motion: float, quality: float) -> str:
        if quality < 30:
            return "unknown"
        if stress < 24 and motion < 20:
            return "calm"
        if stress < 46:
            return "focused"
        if stress < 70:
            return "tense"
        return "agitated"

    def _derive_bluff_level(
        self,
        risk: float,
        bluff_delta: float = 0.0,
        stress_delta: float = 0.0,
        baseline_strength: float = 0.0,
    ) -> str:
        adjusted_delta = max(0.0, 0.68 * bluff_delta + 0.32 * stress_delta)

        if baseline_strength >= 0.2:
            elevated_thresh = 68.0
            watch_thresh = 42.0

            if adjusted_delta >= 28 and risk >= 46:
                return "elevated"
            if adjusted_delta >= 16 and risk >= 34:
                return "watch"

            elevated_thresh -= 5.0 * baseline_strength * _clamp((adjusted_delta - 8.0) / 20.0, 0.0, 1.0)
            watch_thresh -= 7.0 * baseline_strength * _clamp((adjusted_delta - 6.0) / 18.0, 0.0, 1.0)
        else:
            elevated_thresh = 68.0
            watch_thresh = 42.0

        if risk >= elevated_thresh:
            return "elevated"
        if risk >= watch_thresh:
            return "watch"
        return "low"


_cv_service: Optional[CvService] = None


def get_cv_service() -> CvService:
    """Get singleton CV service instance."""
    global _cv_service
    if _cv_service is None:
        _cv_service = CvService()
    return _cv_service
