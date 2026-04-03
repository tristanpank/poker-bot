"""
Adapter for the upstream detector from:
https://github.com/theali29/Lie-Detector

This module intentionally keeps the source attribution explicit. The facial
landmark, BPM, blink, gaze, hand-on-face, and lip-compression logic below is
adapted from the upstream project's `intercept.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - import-time dependency guard
    mp = None
    _MEDIAPIPE_IMPORT_ERROR = exc
    _MP_FACE_MESH = None
    _MP_HANDS = None
else:
    try:
        _MP_FACE_MESH = mp.solutions.face_mesh
        _MP_HANDS = mp.solutions.hands
    except AttributeError:
        try:
            from mediapipe.python.solutions import face_mesh as _MP_FACE_MESH
            from mediapipe.python.solutions import hands as _MP_HANDS
        except Exception as exc:  # pragma: no cover - defensive import fallback
            _MP_FACE_MESH = None
            _MP_HANDS = None
            _MEDIAPIPE_IMPORT_ERROR = exc
        else:
            _MEDIAPIPE_IMPORT_ERROR = None
    else:
        _MEDIAPIPE_IMPORT_ERROR = None


SOURCE_REPO = "theali29/Lie-Detector"
MAX_FRAMES = 120
RECENT_FRAMES = int(MAX_FRAMES / 10)
EYE_BLINK_HEIGHT = 0.15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = 0.35
TELL_MAX_TTL = 30
PULSE_WIN_MS = 15_000
PULSE_KEEP_MS = 20_000
PULSE_HOLD_MS = 4_000
ActivityZone = Literal["none", "left", "center", "right"]
EmotionState = Literal["unknown", "calm", "focused", "tense", "agitated"]

FACEMESH_FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
]

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
class _Tell:
    text: str
    ttl: int = TELL_MAX_TTL


@dataclass
class PulseSample:
    timestamp: int
    r: float
    g: float
    b: float


@dataclass
class UpstreamDetectionResult:
    analysis_source: str
    brightness: float
    motion: float
    edge_density: float
    activity_zone: ActivityZone
    pulse_bpm: Optional[float]
    pulse_confidence: float
    skin_coverage: float
    stress: float
    emotion: EmotionState
    bluff_risk: float
    active_tell_keys: list[str]
    active_tell_texts: list[str]
    calibration_progress: float


@dataclass
class TheAli29LieDetector:
    """
    Per-session adapter for the upstream detector.

    The upstream project stores state in module globals. We keep equivalent
    state per session so multiple sessions can run independently.
    """

    tells: dict[str, _Tell] = field(default_factory=dict)
    blinks: list[bool] = field(default_factory=list)
    hand_on_face: list[bool] = field(default_factory=list)
    pulse_samples: list[PulseSample] = field(default_factory=list)
    avg_bpms: list[float] = field(default_factory=list)
    gaze_values: list[float] = field(default_factory=list)
    frame_count: int = 0
    prev_gray: Optional[np.ndarray] = None
    smooth_pulse_bpm: Optional[float] = None
    smooth_pulse_confidence: float = 0.0
    last_pulse_lock_ms: Optional[int] = None
    smooth_bluff_risk: Optional[float] = None
    smooth_stress: Optional[float] = None
    _face_mesh: object | None = field(default=None, init=False, repr=False)
    _hands: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if mp is None or _MP_FACE_MESH is None or _MP_HANDS is None:
            raise RuntimeError(
                "mediapipe is required to use theali29/Lie-Detector metrics."
            ) from _MEDIAPIPE_IMPORT_ERROR

        self._face_mesh = _MP_FACE_MESH.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = _MP_HANDS.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.7,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            close = getattr(self._face_mesh, "close", None)
            if callable(close):
                close()
            self._face_mesh = None

        if self._hands is not None:
            close = getattr(self._hands, "close", None)
            if callable(close):
                close()
            self._hands = None

    def analyze(
        self,
        frame_rgb: np.ndarray,
        stream_fps: float,
        timestamp_ms: int,
    ) -> UpstreamDetectionResult:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        face_landmarks, hands_landmarks = self._find_face_and_hands(frame_rgb)
        brightness, motion, edge_density = self._compute_base_metrics(gray)
        activity_zone = self._activity_zone(face_landmarks)

        self._decrement_tells()

        pulse_bpm: Optional[float] = None
        face_coverage = 0.0
        pulse_signal_confidence = 0.0
        pulse_delta = 0.0
        blink_score = 0.0
        hand_score = 0.0
        gaze_score = 0.0
        lip_score = 0.0
        avg_gaze = 0.0
        lip_ratio = 1.0
        if face_landmarks is not None:
            face = face_landmarks.landmark
            face_coverage = _clamp(self._get_face_relative_area(face) * 1000.0, 0.0, 100.0)
            pulse_candidate_bpm, pulse_signal_confidence, pulse_patch_quality = (
                self._estimate_pulse_candidate(
                    frame_rgb=frame_rgb,
                    face=face,
                    stream_fps=stream_fps,
                    timestamp_ms=timestamp_ms,
                )
            )
            face_coverage = _clamp(
                0.72 * face_coverage + 0.28 * pulse_patch_quality,
                0.0,
                100.0,
            )
            pulse_bpm, pulse_confidence = self._stabilize_pulse(
                pulse_candidate_bpm=pulse_candidate_bpm,
                signal_confidence=pulse_signal_confidence,
                face_coverage=face_coverage,
                calibration_progress=_clamp((self.frame_count + 1) / float(MAX_FRAMES), 0.0, 1.0),
                timestamp_ms=timestamp_ms,
            )
            self.tells["avg_bpms"] = _Tell(
                text="BPM: ..." if pulse_bpm is None else f"BPM: {pulse_bpm:.1f}"
            )
            bpm_change, pulse_delta = self._update_bpm_baseline(
                pulse_bpm=pulse_bpm,
                pulse_confidence=pulse_confidence,
            )
            if bpm_change:
                self.tells["bpm_change"] = _Tell(text=bpm_change)

            self.blinks.append(self._is_blinking(face))
            if len(self.blinks) > MAX_FRAMES:
                self.blinks = self.blinks[-MAX_FRAMES:]
            blink_tell = self._get_blink_tell()
            blink_score = self._compute_blink_score()
            if blink_tell:
                self.tells["blinking"] = _Tell(text=blink_tell)

            recent_hand = self._check_hand_on_face(hands_landmarks, face)
            self.hand_on_face.append(recent_hand)
            if len(self.hand_on_face) > MAX_FRAMES:
                self.hand_on_face = self.hand_on_face[-MAX_FRAMES:]
            hand_score = self._compute_hand_score()
            if recent_hand:
                self.tells["hand"] = _Tell(text="Hand covering face")

            avg_gaze = self._get_avg_gaze(face)
            gaze_score = self._detect_gaze_change(avg_gaze)
            if gaze_score >= 65.0:
                self.tells["gaze"] = _Tell(text="Change in gaze")

            lip_ratio = self._get_lip_ratio(face)
            lip_score = self._compute_lip_score(lip_ratio)
            if lip_score >= 65.0:
                self.tells["lips"] = _Tell(text="Lip compression")
        else:
            pulse_bpm, pulse_confidence = self._stabilize_pulse(
                pulse_candidate_bpm=None,
                signal_confidence=0.0,
                face_coverage=0.0,
                calibration_progress=_clamp((self.frame_count + 1) / float(MAX_FRAMES), 0.0, 1.0),
                timestamp_ms=timestamp_ms,
            )

        self.frame_count += 1
        calibration_progress = _clamp(self.frame_count / float(MAX_FRAMES), 0.0, 1.0)
        active_tell_keys = [key for key in self.tells.keys() if key != "avg_bpms"]
        active_tell_texts = [self.tells[key].text for key in active_tell_keys]

        pulse_score = _clamp(
            (abs(pulse_delta) / 18.0) * 100.0 * (pulse_confidence / 100.0),
            0.0,
            100.0,
        )
        raw_bluff_risk = _clamp(
            (
                0.30 * pulse_score
                + 0.20 * blink_score
                + 0.20 * hand_score
                + 0.15 * gaze_score
                + 0.15 * lip_score
            )
            * (0.55 + 0.45 * calibration_progress),
            0.0,
            100.0,
        )
        if self.smooth_bluff_risk is None:
            bluff_risk = raw_bluff_risk
        else:
            bluff_risk = self.smooth_bluff_risk * 0.84 + raw_bluff_risk * 0.16
        self.smooth_bluff_risk = bluff_risk

        raw_stress = _clamp(
            0.42 * bluff_risk
            + 0.22 * motion
            + 0.18 * pulse_score
            + 0.10 * hand_score
            + 0.08 * max(0.0, face_coverage - 18.0),
            0.0,
            100.0,
        )
        if self.smooth_stress is None:
            stress = raw_stress
        else:
            stress = self.smooth_stress * 0.82 + raw_stress * 0.18
        self.smooth_stress = stress

        return UpstreamDetectionResult(
            analysis_source=SOURCE_REPO,
            brightness=brightness,
            motion=motion,
            edge_density=edge_density,
            activity_zone=activity_zone,
            pulse_bpm=None if pulse_bpm is None else _round1(pulse_bpm),
            pulse_confidence=_round1(pulse_confidence),
            skin_coverage=_round1(face_coverage),
            stress=_round1(stress),
            emotion=self._emotion_from_risk(face_landmarks is not None, bluff_risk, stress),
            bluff_risk=_round1(bluff_risk),
            active_tell_keys=active_tell_keys,
            active_tell_texts=active_tell_texts,
            calibration_progress=calibration_progress,
        )

    def _compute_base_metrics(self, gray: np.ndarray) -> tuple[float, float, float]:
        brightness = float(gray.mean()) / 255.0 * 100.0

        if self.prev_gray is None:
            motion = 0.0
        else:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion = _clamp(float(diff.mean()) / 255.0 * 400.0, 0.0, 100.0)

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size) * 100.0
        self.prev_gray = gray
        return _round1(brightness), _round1(motion), _round1(edge_density)

    def _activity_zone(self, face_landmarks: object | None) -> ActivityZone:
        if face_landmarks is None:
            return "none"

        face = face_landmarks.landmark
        center_x = (face[234].x + face[454].x) * 0.5
        if center_x < 0.34:
            return "left"
        if center_x > 0.66:
            return "right"
        return "center"

    def _find_face_and_hands(self, image_rgb: np.ndarray) -> tuple[object | None, object | None]:
        working = image_rgb.copy()
        working.flags.writeable = False
        faces = self._face_mesh.process(working)
        hands_landmarks = self._hands.process(working).multi_hand_landmarks

        face_landmarks = None
        if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
            face_landmarks = faces.multi_face_landmarks[0]

        return face_landmarks, hands_landmarks

    def _decrement_tells(self) -> None:
        for key, tell in list(self.tells.items()):
            tell.ttl -= 1
            if tell.ttl <= 0:
                del self.tells[key]

    def _face_box_pixels(
        self,
        image: np.ndarray,
        face: list[object],
    ) -> tuple[int, int, int, int] | None:
        height, width = image.shape[:2]
        xs = np.array([face[index].x for index in FACEMESH_FACE_OVAL], dtype=np.float64)
        ys = np.array([face[index].y for index in FACEMESH_FACE_OVAL], dtype=np.float64)
        if xs.size == 0 or ys.size == 0:
            return None

        x0 = int(np.floor(xs.min() * width))
        x1 = int(np.ceil(xs.max() * width))
        y0 = int(np.floor(ys.min() * height))
        y1 = int(np.ceil(ys.max() * height))
        x0 = max(0, min(width - 2, x0))
        y0 = max(0, min(height - 2, y0))
        x1 = max(x0 + 2, min(width, x1))
        y1 = max(y0 + 2, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _build_pulse_patches(
        self,
        image: np.ndarray,
        face: list[object],
    ) -> list[np.ndarray]:
        face_box = self._face_box_pixels(image, face)
        if face_box is None:
            return []

        x0, y0, x1, y1 = face_box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        patch_specs = [
            (0.10, 0.42, 0.34, 0.72),
            (0.66, 0.42, 0.90, 0.72),
            (0.30, 0.14, 0.70, 0.30),
        ]

        patches: list[np.ndarray] = []
        for left, top, right, bottom in patch_specs:
            px0 = x0 + int(round(width * left))
            py0 = y0 + int(round(height * top))
            px1 = x0 + int(round(width * right))
            py1 = y0 + int(round(height * bottom))
            px0 = max(0, min(image.shape[1] - 2, px0))
            py0 = max(0, min(image.shape[0] - 2, py0))
            px1 = max(px0 + 2, min(image.shape[1], px1))
            py1 = max(py0 + 2, min(image.shape[0], py1))
            patches.append(image[py0:py1, px0:px1])

        return patches

    def _sample_patch_rgb(
        self,
        region: np.ndarray,
    ) -> tuple[np.ndarray, float, float] | None:
        if region.size == 0:
            return None

        working = region
        if working.shape[0] * working.shape[1] > 4_096:
            working = working[::2, ::2]

        pixels = working.reshape(-1, 3).astype(np.float64) / 255.0
        if pixels.shape[0] < 64:
            return None

        luma = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
        chroma = np.max(pixels, axis=1) - np.min(pixels, axis=1)
        luma_lo, luma_hi = np.percentile(luma, [18.0, 88.0])
        chroma_lo, chroma_hi = np.percentile(chroma, [12.0, 92.0])
        keep = (
            (luma >= luma_lo)
            & (luma <= luma_hi)
            & (chroma >= chroma_lo)
            & (chroma <= chroma_hi)
        )
        if int(np.count_nonzero(keep)) < max(48, pixels.shape[0] // 8):
            return None

        filtered = pixels[keep]
        mean_rgb = filtered.mean(axis=0)
        coverage = float(filtered.shape[0]) / float(max(1, pixels.shape[0]))
        weight = float(filtered.shape[0])
        return mean_rgb, coverage, weight

    def _estimate_pulse_candidate(
        self,
        *,
        frame_rgb: np.ndarray,
        face: list[object],
        stream_fps: float,
        timestamp_ms: int,
    ) -> tuple[Optional[float], float, float]:
        patch_means: list[np.ndarray] = []
        patch_coverages: list[float] = []
        patch_weights: list[float] = []

        for patch in self._build_pulse_patches(frame_rgb, face):
            sampled = self._sample_patch_rgb(patch)
            if sampled is None:
                continue
            mean_rgb, coverage, weight = sampled
            patch_means.append(mean_rgb)
            patch_coverages.append(coverage)
            patch_weights.append(weight)

        patch_count_ratio = float(len(patch_means)) / 3.0
        if patch_means:
            patch_stack = np.asarray(patch_means, dtype=np.float64)
            weights = np.asarray(patch_weights, dtype=np.float64)
            rgb = np.average(patch_stack, axis=0, weights=weights)
            avg_coverage = float(np.average(np.asarray(patch_coverages), weights=weights))
            consistency = 1.0
            if patch_stack.shape[0] > 1:
                consistency = 1.0 - _clamp(
                    float(np.mean(np.std(patch_stack, axis=0))) / 0.05,
                    0.0,
                    1.0,
                )
            patch_quality = _clamp(
                100.0
                * (
                    0.45 * avg_coverage
                    + 0.35 * patch_count_ratio
                    + 0.20 * consistency
                ),
                0.0,
                100.0,
            )
            if len(patch_means) >= 2:
                self.pulse_samples.append(
                    PulseSample(
                        timestamp=timestamp_ms,
                        r=float(rgb[0]),
                        g=float(rgb[1]),
                        b=float(rgb[2]),
                    )
                )
        else:
            patch_quality = 0.0

        self.pulse_samples = [
            sample
            for sample in self.pulse_samples
            if timestamp_ms - sample.timestamp <= PULSE_KEEP_MS
        ]

        pulse_candidate_bpm, estimator_confidence = self._estimate_pos_pulse(
            samples=self.pulse_samples,
            default_fps=stream_fps,
        )

        window_duration_sec = 0.0
        if len(self.pulse_samples) >= 2:
            window_duration_sec = max(
                0.0,
                (self.pulse_samples[-1].timestamp - self.pulse_samples[0].timestamp) / 1000.0,
            )
        warmup_progress = _clamp((window_duration_sec - 1.5) / 4.5, 0.0, 1.0)
        signal_confidence = _clamp(
            max(
                estimator_confidence,
                patch_quality * (0.18 + 0.22 * warmup_progress),
            ),
            0.0,
            100.0,
        )
        return pulse_candidate_bpm, signal_confidence, patch_quality

    def _welch_psd(
        self,
        signal: np.ndarray,
        fs: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        n = signal.size
        if n < 48:
            return None

        seg_len = min(n, max(64, int(fs * 6.0)))
        step = max(1, seg_len // 2)
        if seg_len <= 2:
            return None

        window = np.hanning(seg_len).astype(np.float64)
        window_power = float(np.sum(window * window)) + 1e-9
        accumulated_power: np.ndarray | None = None
        segments = 0

        for start in range(0, n - seg_len + 1, step):
            segment = signal[start : start + seg_len]
            segment = segment - float(segment.mean())
            spectrum = np.fft.rfft(segment * window)
            power = (np.abs(spectrum) ** 2) / window_power
            if accumulated_power is None:
                accumulated_power = power
            else:
                accumulated_power += power
            segments += 1

        if accumulated_power is None or segments <= 0:
            return None

        return np.fft.rfftfreq(seg_len, d=1.0 / fs), accumulated_power / float(segments)

    def _estimate_pos_pulse(
        self,
        *,
        samples: list[PulseSample],
        default_fps: float,
    ) -> tuple[Optional[float], float]:
        if len(samples) < 48:
            return None, 0.0

        latest = samples[-1].timestamp
        windowed = [sample for sample in samples if latest - sample.timestamp <= PULSE_WIN_MS]
        if len(windowed) < 48:
            return None, 0.0

        timestamps = np.asarray([sample.timestamp for sample in windowed], dtype=np.float64)
        rgb = np.asarray([[sample.r, sample.g, sample.b] for sample in windowed], dtype=np.float64)

        keep = np.concatenate(([True], np.diff(timestamps) > 0.0))
        timestamps = timestamps[keep]
        rgb = rgb[keep]
        if timestamps.size < 48:
            return None, 0.0

        diffs_ms = np.diff(timestamps)
        diffs_ms = diffs_ms[diffs_ms > 0.0]
        if diffs_ms.size == 0:
            return None, 0.0

        native_fs = 1000.0 / float(np.median(diffs_ms))
        if not np.isfinite(native_fs):
            native_fs = default_fps if default_fps > 0.0 else 30.0
        native_fs = _clamp(native_fs, 12.0, 60.0)

        t_sec = (timestamps - timestamps[0]) / 1000.0
        duration_sec = float(t_sec[-1]) if t_sec.size > 0 else 0.0
        if duration_sec < 3.0:
            return None, 0.0

        uniform_fs = float(_clamp(native_fs, 20.0, 45.0))
        t_uniform = np.arange(0.0, duration_sec, 1.0 / uniform_fs, dtype=np.float64)
        if t_uniform.size < 64:
            return None, 0.0

        rgb_uniform = np.stack(
            [
                np.interp(t_uniform, t_sec, rgb[:, 0]),
                np.interp(t_uniform, t_sec, rgb[:, 1]),
                np.interp(t_uniform, t_sec, rgb[:, 2]),
            ],
            axis=1,
        )

        trend_win = max(3, int(uniform_fs * 1.2))
        detrended = np.empty_like(rgb_uniform)
        for channel in range(3):
            detrended[:, channel] = _normalize(
                rgb_uniform[:, channel] - _moving_average(rgb_uniform[:, channel], trend_win)
            )

        x = detrended[:, 1] - detrended[:, 2]
        y = -2.0 * detrended[:, 0] + detrended[:, 1] + detrended[:, 2]
        std_y = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        alpha = float(np.std(x, ddof=1)) / std_y if std_y > 1e-6 else 0.0

        pulse = x + alpha * y
        pulse = pulse - _moving_average(pulse, max(3, int(uniform_fs * 0.8)))
        pulse = np.tanh(_normalize(pulse) * 1.35)
        pulse_green = detrended[:, 1] - 0.50 * detrended[:, 0] - 0.15 * detrended[:, 2]
        pulse_green = pulse_green - _moving_average(
            pulse_green,
            max(3, int(uniform_fs * 0.8)),
        )
        pulse_green = np.tanh(_normalize(pulse_green) * 1.20)

        pos_candidate = self._estimate_frequency_from_signal(
            signal=pulse,
            fs=uniform_fs,
            duration_sec=duration_sec,
        )
        green_candidate = self._estimate_frequency_from_signal(
            signal=pulse_green,
            fs=uniform_fs,
            duration_sec=duration_sec,
        )

        if pos_candidate[0] is None and green_candidate[0] is None:
            return None, 0.0

        if pos_candidate[0] is not None and green_candidate[0] is not None:
            pos_bpm, pos_conf = pos_candidate
            green_bpm, green_conf = green_candidate
            if abs(pos_bpm - green_bpm) <= 6.0:
                total_conf = pos_conf + green_conf + 1e-9
                blended_bpm = (pos_bpm * pos_conf + green_bpm * green_conf) / total_conf
                boosted_conf = _clamp(max(pos_conf, green_conf) + 8.0, 0.0, 100.0)
                return _round1(blended_bpm), _round1(boosted_conf)
            return (pos_bpm, pos_conf) if pos_conf >= green_conf else (green_bpm, green_conf)

        return pos_candidate if pos_candidate[0] is not None else green_candidate

    def _estimate_frequency_from_signal(
        self,
        *,
        signal: np.ndarray,
        fs: float,
        duration_sec: float,
    ) -> tuple[Optional[float], float]:
        welch = self._welch_psd(signal, fs)
        if welch is None:
            return None, 0.0
        freqs, power = welch
        band = (freqs >= 0.7) & (freqs <= 3.2)
        if not np.any(band):
            return None, 0.0

        band_freqs = freqs[band]
        band_power = power[band]
        if band_power.size < 3:
            return None, 0.0

        peak_idx = int(np.argmax(band_power))
        peak_freq = float(band_freqs[peak_idx])
        peak_power = float(band_power[peak_idx])
        if peak_power <= 0.0:
            return None, 0.0

        total_band_power = float(band_power.sum()) + 1e-9
        local_mask = np.abs(band_freqs - peak_freq) <= 0.08
        local_power = float(band_power[local_mask].sum())
        concentration = local_power / total_band_power
        sorted_power = np.sort(band_power)
        second_peak_power = float(sorted_power[-2]) if sorted_power.size > 1 else 0.0
        peak_ratio = peak_power / (second_peak_power + 1e-9)

        signal_centered = signal - float(signal.mean())
        acf = np.correlate(signal_centered, signal_centered, mode="full")
        acf = acf[acf.size // 2 :]
        if acf.size < 3 or acf[0] <= 1e-9:
            return None, 0.0
        acf = acf / acf[0]

        min_lag = max(1, int(fs / 3.2))
        max_lag = min(acf.size - 1, int(fs / 0.7))
        if max_lag <= min_lag:
            return None, 0.0

        acf_band = acf[min_lag : max_lag + 1]
        acf_peak_lag = int(np.argmax(acf_band)) + min_lag
        acf_peak_value = float(acf[acf_peak_lag])
        acf_freq = float(fs / acf_peak_lag)

        spec_score = _clamp(
            0.58 * concentration + 0.42 * _clamp((peak_ratio - 1.0) / 2.2, 0.0, 1.0),
            0.0,
            1.0,
        )
        acf_score = _clamp((acf_peak_value - 0.12) / 0.52, 0.0, 1.0)

        if abs(acf_freq - peak_freq) <= 0.22:
            best_hz = 0.72 * peak_freq + 0.28 * acf_freq
        elif acf_score > spec_score + 0.10:
            best_hz = acf_freq
        else:
            best_hz = peak_freq

        best_hz = float(_clamp(best_hz, 0.7, 3.2))
        confidence = _clamp(
            100.0
            * (
                0.52 * spec_score
                + 0.33 * acf_score
                + 0.15 * _clamp(duration_sec / 10.0, 0.0, 1.0)
            ),
            0.0,
            100.0,
        )
        return _round1(best_hz * 60.0), _round1(confidence)

    def _update_bpm_baseline(
        self,
        *,
        pulse_bpm: Optional[float],
        pulse_confidence: float,
    ) -> tuple[str, float]:
        if pulse_bpm is None or pulse_confidence < 30.0:
            return "", 0.0

        history = [bpm for bpm in self.avg_bpms[-90:] if 45.0 <= bpm <= 160.0]
        baseline = float(np.median(history)) if len(history) >= 8 else None

        if pulse_confidence >= 45.0:
            self.avg_bpms.append(float(pulse_bpm))
            if len(self.avg_bpms) > 180:
                self.avg_bpms = self.avg_bpms[-180:]

        if baseline is None:
            return "", 0.0

        bpm_delta = float(pulse_bpm) - baseline
        if len(history) < 12:
            return "", bpm_delta
        if bpm_delta > SIGNIFICANT_BPM_CHANGE:
            return "Heart rate increasing", bpm_delta
        if bpm_delta < -SIGNIFICANT_BPM_CHANGE:
            return "Heart rate decreasing", bpm_delta
        return "", bpm_delta

    def _is_blinking(self, face: list[object]) -> bool:
        eye_right = [face[p] for p in [159, 145, 133, 33]]
        eye_right_ratio = self._aspect_ratio(*eye_right)
        eye_left = [face[p] for p in [386, 374, 362, 263]]
        eye_left_ratio = self._aspect_ratio(*eye_left)
        return ((eye_right_ratio + eye_left_ratio) / 2.0) < EYE_BLINK_HEIGHT

    def _get_blink_tell(self) -> Optional[str]:
        if len(self.blinks) < RECENT_FRAMES:
            return None

        recent_closed = float(sum(self.blinks[-RECENT_FRAMES:])) / float(RECENT_FRAMES)
        avg_closed = float(sum(self.blinks)) / float(max(1, len(self.blinks)))
        if recent_closed > (20.0 * avg_closed):
            return "Increased blinking"
        if avg_closed > (20.0 * recent_closed):
            return "Decreased blinking"
        return None

    def _check_hand_on_face(self, hands_landmarks: object | None, face: list[object]) -> bool:
        if not hands_landmarks:
            return False

        face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
        face_points = [[[p.x, p.y] for p in face_landmarks]]
        face_contours = np.array(face_points).astype(np.single)

        for hand_landmarks in hands_landmarks:
            hand = [(point.x, point.y) for point in hand_landmarks.landmark]
            for finger in [4, 8, 20]:
                overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
                if overlap != -1:
                    return True
        return False

    def _get_avg_gaze(self, face: list[object]) -> float:
        gaze_left = self._get_gaze(face, 476, 474, 263, 362)
        gaze_right = self._get_gaze(face, 471, 469, 33, 133)
        return round((gaze_left + gaze_right) / 2.0, 1)

    def _get_gaze(
        self,
        face: list[object],
        iris_left_side: int,
        iris_right_side: int,
        eye_left_corner: int,
        eye_right_corner: int,
    ) -> float:
        iris = (
            face[iris_left_side].x + face[iris_right_side].x,
            face[iris_left_side].y + face[iris_right_side].y,
        )
        eye_center = (
            face[eye_left_corner].x + face[eye_right_corner].x,
            face[eye_left_corner].y + face[eye_right_corner].y,
        )
        gaze_dist = dist.euclidean(iris, eye_center)
        eye_width = abs(face[eye_right_corner].x - face[eye_left_corner].x)
        if eye_width <= 1e-9:
            return 0.0
        gaze_relative = gaze_dist / eye_width
        if (eye_center[0] - iris[0]) < 0:
            gaze_relative *= -1.0
        return float(gaze_relative)

    def _detect_gaze_change(self, avg_gaze: float) -> float:
        self.gaze_values.append(avg_gaze)
        if len(self.gaze_values) > MAX_FRAMES:
            self.gaze_values = self.gaze_values[-MAX_FRAMES:]

        if len(self.gaze_values) < max(8, RECENT_FRAMES):
            return 0.0

        history = np.asarray(self.gaze_values, dtype=np.float64)
        center = float(np.median(history))
        spread = float(np.median(np.abs(history - center))) * 1.4826
        spread = max(spread, 0.05)
        z = abs(avg_gaze - center) / spread
        return _clamp((z - 0.8) / 2.2 * 100.0, 0.0, 100.0)

    def _get_lip_ratio(self, face: list[object]) -> float:
        return self._aspect_ratio(face[0], face[17], face[61], face[291])

    def _get_face_relative_area(self, face: list[object]) -> float:
        face_width = abs(max(face[454].x, 0.0) - max(face[234].x, 0.0))
        face_height = abs(max(face[152].y, 0.0) - max(face[10].y, 0.0))
        return float(face_width * face_height)

    def _aspect_ratio(self, top: object, bottom: object, right: object, left: object) -> float:
        height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
        width = dist.euclidean([right.x, right.y], [left.x, left.y])
        if width <= 1e-9:
            return 0.0
        return float(height / width)

    def _stabilize_pulse(
        self,
        *,
        pulse_candidate_bpm: Optional[float],
        signal_confidence: float,
        face_coverage: float,
        calibration_progress: float,
        timestamp_ms: int,
    ) -> tuple[Optional[float], float]:
        tracking_confidence = _clamp(
            0.60 * signal_confidence
            + 0.22 * face_coverage
            + 18.0 * calibration_progress,
            0.0,
            100.0,
        )

        if pulse_candidate_bpm is not None:
            if self.smooth_pulse_bpm is not None:
                max_step = 4.0 + 0.10 * signal_confidence
                pulse_candidate_bpm = _clamp(
                    pulse_candidate_bpm,
                    self.smooth_pulse_bpm - max_step,
                    self.smooth_pulse_bpm + max_step,
                )
            if self.smooth_pulse_bpm is None:
                self.smooth_pulse_bpm = pulse_candidate_bpm
            else:
                blend = 0.06 + 0.12 * (signal_confidence / 100.0)
                self.smooth_pulse_bpm = (
                    self.smooth_pulse_bpm * (1.0 - blend) + pulse_candidate_bpm * blend
                )

            if self.smooth_pulse_confidence <= 0.0:
                self.smooth_pulse_confidence = tracking_confidence
            else:
                self.smooth_pulse_confidence = (
                    self.smooth_pulse_confidence * 0.78 + tracking_confidence * 0.22
                )
            self.last_pulse_lock_ms = timestamp_ms
            return self.smooth_pulse_bpm, self.smooth_pulse_confidence

        hold_ms = None if self.last_pulse_lock_ms is None else timestamp_ms - self.last_pulse_lock_ms
        if (
            self.smooth_pulse_bpm is not None
            and hold_ms is not None
            and hold_ms <= PULSE_HOLD_MS
            and face_coverage >= 12.0
        ):
            decay = 1.0 - (hold_ms / float(PULSE_HOLD_MS))
            held_confidence = max(
                tracking_confidence * 0.55,
                self.smooth_pulse_confidence * max(0.24, decay),
            )
            self.smooth_pulse_confidence = held_confidence
            return self.smooth_pulse_bpm, held_confidence

        self.smooth_pulse_confidence = tracking_confidence * 0.55
        return None, self.smooth_pulse_confidence

    def _compute_blink_score(self) -> float:
        if len(self.blinks) < RECENT_FRAMES:
            return 0.0

        recent_closed = float(sum(self.blinks[-RECENT_FRAMES:])) / float(RECENT_FRAMES)
        avg_closed = float(sum(self.blinks)) / float(max(1, len(self.blinks)))
        baseline = max(avg_closed, 0.02)
        deviation = abs(recent_closed - avg_closed) / baseline
        return _clamp((deviation - 0.8) / 2.5 * 100.0, 0.0, 100.0)

    def _compute_hand_score(self) -> float:
        if not self.hand_on_face:
            return 0.0

        window = self.hand_on_face[-RECENT_FRAMES:]
        return _clamp(float(sum(window)) / float(max(1, len(window))) * 100.0, 0.0, 100.0)

    def _compute_lip_score(self, lip_ratio: float) -> float:
        return _clamp((0.95 - lip_ratio) / 0.45 * 100.0, 0.0, 100.0)

    def _emotion_from_risk(
        self,
        face_detected: bool,
        bluff_risk: float,
        stress: float,
    ) -> EmotionState:
        if not face_detected:
            return "unknown"
        if bluff_risk >= 70 or stress >= 75:
            return "agitated"
        if bluff_risk >= 38 or stress >= 42:
            return "tense"
        if bluff_risk >= 12:
            return "focused"
        return "calm"
