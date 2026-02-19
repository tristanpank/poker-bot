"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type ActivityZone = "none" | "left" | "center" | "right";
type EmotionState = "unknown" | "calm" | "focused" | "tense" | "agitated";
type BluffLevel = "low" | "watch" | "elevated";
type SignalQuality = "poor" | "fair" | "good";
type MetricTone = "neutral" | "good" | "warn" | "alert";

type VisionMetrics = {
  brightness: number;
  motion: number;
  edgeDensity: number;
  activityZone: ActivityZone;
  pulseBpm: number | null;
  pulseConfidence: number;
  skinCoverage: number;
  stress: number;
  emotion: EmotionState;
  bluffRisk: number;
  bluffLevel: BluffLevel;
  signalQuality: SignalQuality;
  analysisFps: number;
  streamFps: number;
  updatedAt: string;
};

type TeardownOptions = {
  resetMetrics?: boolean;
  updateState?: boolean;
};

type PulseSample = {
  t: number;
  r: number;
  g: number;
  b: number;
};

type BluffPoint = {
  t: number;
  value: number;
};

const W = 160;
const H = 90;
const STEP = 2;
const ANALYSIS_FRAME_MS = 40;
const PULSE_WIN_MS = 20_000;
const PULSE_KEEP_MS = 30_000;
const BLUFF_WINDOW_MS = 30_000;
const SEND_MAX_BITRATE = 3_500_000;
const SEND_MAX_FPS = 30;

const INITIAL: VisionMetrics = {
  brightness: 0,
  motion: 0,
  edgeDensity: 0,
  activityZone: "none",
  pulseBpm: null,
  pulseConfidence: 0,
  skinCoverage: 0,
  stress: 0,
  emotion: "unknown",
  bluffRisk: 0,
  bluffLevel: "low",
  signalQuality: "poor",
  analysisFps: 0,
  streamFps: 0,
  updatedAt: "--",
};

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
const r1 = (v: number) => Math.round(v * 10) / 10;

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i];
  return sum / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = avg(values);
  let s = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - mean;
    s += d * d;
  }
  return Math.sqrt(s / (values.length - 1));
}

function analyzeFrame(
  frame: ImageData,
  prevLuma: Uint8Array | null,
): {
  nextLuma: Uint8Array;
  brightness: number;
  motion: number;
  edgeDensity: number;
  activityZone: ActivityZone;
} {
  const { data } = frame;
  const nextLuma = new Uint8Array(W * H);

  let count = 0;
  let brightSum = 0;
  let motionSum = 0;
  let motionX = 0;
  let edgeCount = 0;

  for (let y = 1; y < H; y += STEP) {
    for (let x = 1; x < W; x += STEP) {
      const p = y * W + x;
      const i = p * 4;

      const luma = (77 * data[i] + 150 * data[i + 1] + 29 * data[i + 2]) >> 8;
      nextLuma[p] = luma;
      brightSum += luma;
      count += 1;

      if (prevLuma) {
        const d = Math.abs(luma - prevLuma[p]);
        if (d > 14) {
          motionSum += d;
          motionX += d * x;
        }
      }

      const li = i - 4;
      const ti = i - W * 4;
      const left = (77 * data[li] + 150 * data[li + 1] + 29 * data[li + 2]) >> 8;
      const top = (77 * data[ti] + 150 * data[ti + 1] + 29 * data[ti + 2]) >> 8;
      if (Math.abs(luma - left) + Math.abs(luma - top) > 54) edgeCount += 1;
    }
  }

  const brightness = count > 0 ? (brightSum / count / 255) * 100 : 0;
  const edgeDensity = count > 0 ? (edgeCount / count) * 100 : 0;
  const motion = prevLuma && count > 0 ? Math.min(100, (motionSum / (count * 255)) * 300) : 0;

  let activityZone: ActivityZone = "none";
  if (count > 0 && motionSum > count * 8) {
    const cx = motionX / motionSum;
    if (cx < W / 3) activityZone = "left";
    else if (cx > (2 * W) / 3) activityZone = "right";
    else activityZone = "center";
  }

  return {
    nextLuma,
    brightness: r1(brightness),
    motion: r1(motion),
    edgeDensity: r1(edgeDensity),
    activityZone,
  };
}

function sampleForeheadSignal(frame: ImageData): { r: number; g: number; b: number; coverage: number } {
  const { data } = frame;
  const x0 = Math.floor(W * 0.35);
  const x1 = Math.ceil(W * 0.65);
  const y0 = Math.floor(H * 0.16);
  const y1 = Math.ceil(H * 0.42);

  let visited = 0;
  let accepted = 0;
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let fr = 0;
  let fg = 0;
  let fb = 0;

  for (let y = y0; y < y1; y += STEP) {
    for (let x = x0; x < x1; x += STEP) {
      const i = (y * W + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      fr += r;
      fg += g;
      fb += b;
      visited += 1;

      const maxRgb = Math.max(r, g, b);
      const minRgb = Math.min(r, g, b);
      const isSkin =
        r > 45 && g > 30 && b > 20 && r > g && r > b && maxRgb - minRgb > 15 && Math.abs(r - g) > 10;

      if (isSkin) {
        sr += r;
        sg += g;
        sb += b;
        accepted += 1;
      }
    }
  }

  const denom = accepted > 0 ? accepted : Math.max(1, visited);
  const rr = (accepted > 0 ? sr : fr) / denom / 255;
  const gg = (accepted > 0 ? sg : fg) / denom / 255;
  const bb = (accepted > 0 ? sb : fb) / denom / 255;

  return {
    r: rr,
    g: gg,
    b: bb,
    coverage: r1(visited > 0 ? (accepted / visited) * 100 : 0),
  };
}
function estimatePosPulse(samples: PulseSample[]): { bpm: number | null; confidence: number } {
  if (samples.length < 45) return { bpm: null, confidence: 0 };

  const latest = samples[samples.length - 1].t;
  const windowed: PulseSample[] = [];
  for (let i = samples.length - 1; i >= 0; i -= 1) {
    if (latest - samples[i].t > PULSE_WIN_MS) break;
    windowed.unshift(samples[i]);
  }
  if (windowed.length < 45) return { bpm: null, confidence: 0 };

  const r = windowed.map((s) => s.r);
  const g = windowed.map((s) => s.g);
  const b = windowed.map((s) => s.b);

  const mr = avg(r);
  const mg = avg(g);
  const mb = avg(b);
  if (mr < 0.01 || mg < 0.01 || mb < 0.01) return { bpm: null, confidence: 0 };

  const x = new Array<number>(windowed.length);
  const y = new Array<number>(windowed.length);
  for (let i = 0; i < windowed.length; i += 1) {
    const rn = r[i] / mr - 1;
    const gn = g[i] / mg - 1;
    const bn = b[i] / mb - 1;
    x[i] = gn - bn;
    y[i] = -2 * rn + gn + bn;
  }

  const alpha = std(y) > 1e-6 ? std(x) / std(y) : 0;
  const pulse = x.map((v, i) => v + alpha * y[i]);
  const pulseMean = avg(pulse);
  for (let i = 0; i < pulse.length; i += 1) pulse[i] -= pulseMean;

  let dtSum = 0;
  for (let i = 1; i < windowed.length; i += 1) dtSum += windowed[i].t - windowed[i - 1].t;
  const fs = 1000 / (dtSum / Math.max(1, windowed.length - 1));
  if (!Number.isFinite(fs) || fs < 6) return { bpm: null, confidence: 0 };

  let bestHz = 0;
  let bestPower = 0;
  let totalPower = 0;

  for (let hz = 0.8; hz <= 2.9; hz += 0.025) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < pulse.length; i += 1) {
      const angle = (2 * Math.PI * hz * i) / fs;
      re += pulse[i] * Math.cos(angle);
      im -= pulse[i] * Math.sin(angle);
    }
    const power = re * re + im * im;
    totalPower += power;
    if (power > bestPower) {
      bestPower = power;
      bestHz = hz;
    }
  }

  if (bestHz <= 0 || bestPower <= 0) return { bpm: null, confidence: 0 };
  const dominance = bestPower / (totalPower + 1e-6);

  return {
    bpm: r1(bestHz * 60),
    confidence: r1(clamp(dominance * 280, 0, 100)),
  };
}

function deriveSignalQuality(score: number): SignalQuality {
  if (score >= 70) return "good";
  if (score >= 45) return "fair";
  return "poor";
}

function deriveEmotion(stress: number, motion: number, quality: number): EmotionState {
  if (quality < 30) return "unknown";
  if (stress < 24 && motion < 20) return "calm";
  if (stress < 46) return "focused";
  if (stress < 70) return "tense";
  return "agitated";
}

function deriveBluffLevel(risk: number): BluffLevel {
  if (risk >= 68) return "elevated";
  if (risk >= 42) return "watch";
  return "low";
}

function numberLabel(v: number | null, suffix = ""): string {
  return v === null ? "--" : `${v}${suffix}`;
}

function toneForQuality(q: SignalQuality): MetricTone {
  if (q === "good") return "good";
  if (q === "fair") return "warn";
  return "alert";
}

function toneForEmotion(e: EmotionState): MetricTone {
  if (e === "agitated") return "alert";
  if (e === "tense") return "warn";
  if (e === "unknown") return "neutral";
  return "good";
}

function toneForBluff(level: BluffLevel): MetricTone {
  if (level === "elevated") return "alert";
  if (level === "watch") return "warn";
  return "good";
}

function rankCodec(mimeType: string): number {
  const normalized = mimeType.toLowerCase();
  if (normalized === "video/h264") return 0;
  if (normalized === "video/vp9") return 1;
  if (normalized === "video/vp8") return 2;
  return 3;
}

function preferVideoCodecs(transceiver: RTCRtpTransceiver): void {
  if (typeof RTCRtpSender === "undefined" || !RTCRtpSender.getCapabilities) {
    return;
  }

  const caps = RTCRtpSender.getCapabilities("video");
  if (!caps?.codecs?.length) {
    return;
  }

  const sorted = [...caps.codecs].sort(
    (a, b) => rankCodec(a.mimeType) - rankCodec(b.mimeType),
  );

  try {
    transceiver.setCodecPreferences(sorted);
  } catch {
    // Keep browser defaults if codec preference setting is unsupported.
  }
}

async function tuneVideoSender(sender: RTCRtpSender): Promise<void> {
  try {
    const params = sender.getParameters();
    const encodings =
      params.encodings && params.encodings.length > 0 ? params.encodings : [{}];

    const tunedEncodings = encodings.map((encoding) => ({
      ...encoding,
      maxBitrate:
        typeof encoding.maxBitrate === "number"
          ? Math.max(encoding.maxBitrate, SEND_MAX_BITRATE)
          : SEND_MAX_BITRATE,
      maxFramerate:
        typeof encoding.maxFramerate === "number"
          ? Math.min(Math.max(encoding.maxFramerate, 24), SEND_MAX_FPS)
          : SEND_MAX_FPS,
      scaleResolutionDownBy: 1,
    }));

    await sender.setParameters({
      ...params,
      encodings: tunedEncodings,
    });
  } catch {
    // Sender parameter tuning is optional and browser-dependent.
  }
}

export default function Home() {
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const localStreamRef = useRef<MediaStream | null>(null);
  const senderPcRef = useRef<RTCPeerConnection | null>(null);
  const receiverPcRef = useRef<RTCPeerConnection | null>(null);

  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTsRef = useRef<number | null>(null);
  const lastSampleTsRef = useRef<number | null>(null);
  const lastDecodedTsRef = useRef<number | null>(null);
  const lastDecodedFramesRef = useRef<number | null>(null);
  const smoothStreamFpsRef = useRef(0);
  const prevLumaRef = useRef<Uint8Array | null>(null);
  const pulseSamplesRef = useRef<PulseSample[]>([]);
  const smoothPulseRef = useRef<number | null>(null);
  const pulseBaselineRef = useRef<number | null>(null);
  const prevStressRef = useRef<number | null>(null);
  const activeRef = useRef(false);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<VisionMetrics>(INITIAL);
  const [bluffHistory, setBluffHistory] = useState<BluffPoint[]>([]);

  const teardown = useCallback((opts: TeardownOptions = {}) => {
    const { resetMetrics = false, updateState = true } = opts;
    activeRef.current = false;

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    senderPcRef.current?.close();
    receiverPcRef.current?.close();
    senderPcRef.current = null;
    receiverPcRef.current = null;

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((t) => t.stop());
      localStreamRef.current = null;
    }

    if (localVideoRef.current) localVideoRef.current.srcObject = null;
    if (remoteVideoRef.current) {
      const s = remoteVideoRef.current.srcObject;
      if (s instanceof MediaStream) s.getTracks().forEach((t) => t.stop());
      remoteVideoRef.current.srcObject = null;
    }

    ctxRef.current = null;
    lastFrameTsRef.current = null;
    lastSampleTsRef.current = null;
    lastDecodedTsRef.current = null;
    lastDecodedFramesRef.current = null;
    smoothStreamFpsRef.current = 0;
    prevLumaRef.current = null;
    pulseSamplesRef.current = [];
    smoothPulseRef.current = null;
    pulseBaselineRef.current = null;
    prevStressRef.current = null;

    if (updateState) setIsStreaming(false);
    if (resetMetrics) {
      setMetrics(INITIAL);
      setBluffHistory([]);
    }
  }, []);
  const runAnalysis = useCallback(
    function loop(ts: number) {
      if (!activeRef.current) return;

      if (lastSampleTsRef.current === null || ts - lastSampleTsRef.current >= ANALYSIS_FRAME_MS) {
        lastSampleTsRef.current = ts;

        const video = remoteVideoRef.current;
        const canvas = canvasRef.current;

        if (video && canvas && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          let streamFps = smoothStreamFpsRef.current;
          if (typeof video.getVideoPlaybackQuality === "function") {
            const quality = video.getVideoPlaybackQuality();
            const decodedFrames = quality.totalVideoFrames;

            if (lastDecodedFramesRef.current !== null && lastDecodedTsRef.current !== null) {
              const frameDelta = decodedFrames - lastDecodedFramesRef.current;
              const dt = ts - lastDecodedTsRef.current;
              if (frameDelta >= 0 && dt > 0) {
                const instantStreamFps = (frameDelta * 1000) / dt;
                streamFps =
                  smoothStreamFpsRef.current === 0
                    ? instantStreamFps
                    : smoothStreamFpsRef.current * 0.75 + instantStreamFps * 0.25;
                smoothStreamFpsRef.current = streamFps;
              }
            }

            lastDecodedFramesRef.current = decodedFrames;
            lastDecodedTsRef.current = ts;
          }

          if (canvas.width !== W) canvas.width = W;
          if (canvas.height !== H) canvas.height = H;

          if (!ctxRef.current) {
            ctxRef.current = canvas.getContext("2d", { willReadFrequently: true });
          }

          const ctx = ctxRef.current;
          if (ctx) {
            ctx.drawImage(video, 0, 0, W, H);
            const frame = ctx.getImageData(0, 0, W, H);

            const base = analyzeFrame(frame, prevLumaRef.current);
            prevLumaRef.current = base.nextLuma;

            const skin = sampleForeheadSignal(frame);
            pulseSamplesRef.current.push({ t: ts, r: skin.r, g: skin.g, b: skin.b });
            while (pulseSamplesRef.current.length > 0 && ts - pulseSamplesRef.current[0].t > PULSE_KEEP_MS) {
              pulseSamplesRef.current.shift();
            }

            const pulse = estimatePosPulse(pulseSamplesRef.current);
            let pulseBpm = smoothPulseRef.current;
            let pulseConfidence = pulse.confidence;

            if (pulse.bpm !== null) {
              const smooth =
                smoothPulseRef.current === null
                  ? pulse.bpm
                  : smoothPulseRef.current * 0.92 + pulse.bpm * 0.08;
              smoothPulseRef.current = r1(smooth);
              pulseBpm = smoothPulseRef.current;

              if (pulseBaselineRef.current === null && pulse.confidence > 35) {
                pulseBaselineRef.current = smooth;
              } else if (pulseBaselineRef.current !== null && pulse.confidence > 25 && base.motion < 35) {
                pulseBaselineRef.current = pulseBaselineRef.current * 0.992 + smooth * 0.008;
              }
            }

            const pulseDelta =
              pulseBpm !== null && pulseBaselineRef.current !== null ? Math.abs(pulseBpm - pulseBaselineRef.current) : 0;
            const pulseStress = clamp(pulseDelta * 2.3, 0, 100);

            const qualityScore = clamp(
              0.45 * skin.coverage +
                0.35 * pulseConfidence +
                0.2 * (100 - Math.min(100, Math.abs(base.brightness - 55) * 1.8)),
              0,
              100,
            );

            if (pulse.bpm === null && qualityScore < 45) {
              pulseConfidence = r1(Math.max(0, pulseConfidence - 12));
            }

            const stress = r1(
              clamp(
                (0.5 * base.motion + 0.5 * pulseStress) * (0.65 + qualityScore / 200),
                0,
                100,
              ),
            );

            const prevStress = prevStressRef.current;
            const stressTrend = prevStress === null ? 0 : stress - prevStress;
            prevStressRef.current = stress;

            const emotion = deriveEmotion(stress, base.motion, qualityScore);
            const stillness = base.motion < 20 ? 60 : Math.max(0, 35 - base.motion);
            const bluffRisk = r1(
              clamp(
                0.5 * stress + 0.24 * pulseStress + 0.18 * Math.max(0, stressTrend * 3) + 0.08 * stillness,
                0,
                100,
              ),
            );

            setBluffHistory((previous) => {
              const next = [...previous, { t: ts, value: bluffRisk }];
              const minTs = ts - BLUFF_WINDOW_MS;
              let firstValidIndex = 0;

              while (firstValidIndex < next.length && next[firstValidIndex].t < minTs) {
                firstValidIndex += 1;
              }

              return firstValidIndex > 0 ? next.slice(firstValidIndex) : next;
            });

            let analysisFps = 0;
            if (lastFrameTsRef.current !== null) {
              const dt = ts - lastFrameTsRef.current;
              if (dt > 0) analysisFps = 1000 / dt;
            }
            lastFrameTsRef.current = ts;

            setMetrics({
              brightness: base.brightness,
              motion: base.motion,
              edgeDensity: base.edgeDensity,
              activityZone: base.activityZone,
              pulseBpm,
              pulseConfidence: r1(pulseConfidence),
              skinCoverage: skin.coverage,
              stress,
              emotion,
              bluffRisk,
              bluffLevel: deriveBluffLevel(bluffRisk),
              signalQuality: deriveSignalQuality(qualityScore),
              analysisFps: r1(analysisFps),
              streamFps: r1(streamFps),
              updatedAt: new Date().toLocaleTimeString(),
            });
          }
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    },
    [],
  );

  const startStream = useCallback(async () => {
    if (isStreaming) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support camera streaming.");
      return;
    }

    setError(null);
    teardown({ resetMetrics: true, updateState: false });

    try {
      const localStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: SEND_MAX_FPS, max: SEND_MAX_FPS },
          facingMode: "user",
        },
        audio: false,
      });

      localStreamRef.current = localStream;
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = localStream;
        void localVideoRef.current.play().catch(() => undefined);
      }

      const sender = new RTCPeerConnection();
      const receiver = new RTCPeerConnection();
      senderPcRef.current = sender;
      receiverPcRef.current = receiver;

      sender.onicecandidate = (e) => {
        if (e.candidate) void receiver.addIceCandidate(e.candidate).catch(() => undefined);
      };
      receiver.onicecandidate = (e) => {
        if (e.candidate) void sender.addIceCandidate(e.candidate).catch(() => undefined);
      };

      receiver.ontrack = (e) => {
        const [stream] = e.streams;
        if (stream && remoteVideoRef.current) {
          remoteVideoRef.current.srcObject = stream;
          void remoteVideoRef.current.play().catch(() => undefined);
        }
      };

      const [videoTrack] = localStream.getVideoTracks();
      if (!videoTrack) {
        throw new Error("No video track available from camera.");
      }

      try {
        videoTrack.contentHint = "detail";
        await videoTrack.applyConstraints({
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: SEND_MAX_FPS, max: SEND_MAX_FPS },
        });
      } catch {
        // Keep original constraints if camera does not support higher-quality values.
      }

      const transceiver = sender.addTransceiver(videoTrack, {
        direction: "sendonly",
        streams: [localStream],
      });
      preferVideoCodecs(transceiver);
      await tuneVideoSender(transceiver.sender);

      const offer = await sender.createOffer();
      await sender.setLocalDescription(offer);
      await receiver.setRemoteDescription(offer);
      const answer = await receiver.createAnswer();
      await receiver.setLocalDescription(answer);
      await sender.setRemoteDescription(answer);

      prevLumaRef.current = null;
      pulseSamplesRef.current = [];
      smoothPulseRef.current = null;
      pulseBaselineRef.current = null;
      prevStressRef.current = null;
      lastFrameTsRef.current = null;
      lastSampleTsRef.current = null;
      lastDecodedTsRef.current = null;
      lastDecodedFramesRef.current = null;
      smoothStreamFpsRef.current = 0;

      activeRef.current = true;
      setIsStreaming(true);
      rafRef.current = requestAnimationFrame(runAnalysis);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error while starting stream.";
      setError(`Unable to start camera stream: ${msg}`);
      teardown({ resetMetrics: true, updateState: true });
    }
  }, [isStreaming, runAnalysis, teardown]);

  const stopStream = useCallback(() => {
    teardown({ resetMetrics: true, updateState: true });
  }, [teardown]);

  useEffect(() => {
    return () => {
      teardown({ resetMetrics: false, updateState: false });
    };
  }, [teardown]);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <section className="flex flex-col gap-3 rounded-2xl border border-slate-700 bg-slate-900/60 p-6 shadow-xl shadow-slate-950/30">
          <h1 className="text-2xl font-semibold tracking-tight">WebRTC Bluff Signal + Lightweight POS CV</h1>
          <p className="max-w-3xl text-sm text-slate-300">
            Streams camera via WebRTC, estimates pulse with a POS-style skin signal,
            then combines pulse and motion into stress/emotion/bluff-pressure proxies.
          </p>
          <p className="max-w-3xl text-xs text-slate-400">
            Heuristic only: visual cues are not a reliable lie detector.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              type="button"
              onClick={startStream}
              disabled={isStreaming}
              className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              Start stream
            </button>

            <button
              type="button"
              onClick={stopStream}
              disabled={!isStreaming}
              className="rounded-md bg-rose-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              Stop stream
            </button>

            <span className={`rounded-md px-3 py-1 text-xs font-medium ${isStreaming ? "bg-emerald-500/20 text-emerald-300" : "bg-slate-700 text-slate-200"}`}>
              {isStreaming ? "Live" : "Offline"}
            </span>
          </div>

          {error && (
            <p className="rounded-md border border-rose-500/50 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {error}
            </p>
          )}
        </section>
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">Local camera</h2>
            <video
              ref={localVideoRef}
              className="aspect-video w-full rounded-lg border border-slate-700 bg-black object-cover"
              autoPlay
              muted
              playsInline
            />
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">WebRTC received stream</h2>
            <video
              ref={remoteVideoRef}
              className="aspect-video w-full rounded-lg border border-slate-700 bg-black object-cover"
              autoPlay
              muted
              playsInline
            />
          </div>
        </section>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/60 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-300">
            Real-time deception proxy metrics
          </h2>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="Pulse (POS)" value={numberLabel(metrics.pulseBpm, " BPM")} />
            <MetricCard label="Pulse confidence" value={`${metrics.pulseConfidence}%`} tone={metrics.pulseConfidence >= 55 ? "good" : "warn"} />
            <MetricCard label="Signal quality" value={metrics.signalQuality} tone={toneForQuality(metrics.signalQuality)} />
            <MetricCard label="Skin coverage" value={`${metrics.skinCoverage}%`} tone={metrics.skinCoverage >= 35 ? "good" : "warn"} />

            <MetricCard label="Stress index" value={`${metrics.stress}%`} tone={metrics.stress >= 68 ? "alert" : metrics.stress >= 42 ? "warn" : "good"} />
            <MetricCard label="Emotion state" value={metrics.emotion} tone={toneForEmotion(metrics.emotion)} />
            <MetricCard label="Bluff pressure" value={`${metrics.bluffRisk}%`} tone={toneForBluff(metrics.bluffLevel)} />
            <MetricCard label="Bluff level" value={metrics.bluffLevel} tone={toneForBluff(metrics.bluffLevel)} />

            <MetricCard label="Global motion" value={`${metrics.motion}%`} />
            <MetricCard label="Brightness" value={`${metrics.brightness}%`} />
            <MetricCard label="Edge density" value={`${metrics.edgeDensity}%`} />
            <MetricCard label="Activity zone" value={metrics.activityZone} />

            <MetricCard label="Analysis FPS" value={`${metrics.analysisFps}`} />
            <MetricCard label="Stream FPS" value={`${metrics.streamFps}`} />
            <MetricCard label="Last update" value={metrics.updatedAt} />
          </div>

          <BluffPressureChart points={bluffHistory} windowMs={BLUFF_WINDOW_MS} />

          <p className="mt-4 text-xs text-slate-400">
            Optimized for low compute cost and responsiveness, not forensic certainty.
          </p>

          <canvas ref={canvasRef} className="hidden" />
        </section>
      </main>
    </div>
  );
}

type BluffPressureChartProps = {
  points: BluffPoint[];
  windowMs: number;
};

function BluffPressureChart({ points, windowMs }: BluffPressureChartProps) {
  const width = 760;
  const height = 220;
  const padX = 36;
  const padY = 18;
  const plotWidth = width - padX * 2;
  const plotHeight = height - padY * 2;
  const bottomY = padY + plotHeight;
  const windowSeconds = Math.round(windowMs / 1000);
  const labelPrefix = `Bluff-pressure trend (last ${windowSeconds}s)`;

  if (points.length === 0) {
    return (
      <div className="mt-6 rounded-lg border border-slate-700 bg-slate-950/60 p-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-200">{labelPrefix}</p>
          <p className="text-xs text-slate-400">Collecting data...</p>
        </div>
        <div className="mt-3 h-44 rounded-md border border-slate-800/90 bg-slate-950/80" />
      </div>
    );
  }

  const latestTs = points[points.length - 1].t;
  const startTs = latestTs - windowMs;
  const visible = points.filter((point) => point.t >= startTs);
  const chartPoints =
    visible.length >= 2 ? visible : [{ t: startTs, value: visible[0].value }, ...visible];

  const toX = (t: number) => padX + ((t - startTs) / windowMs) * plotWidth;
  const toY = (v: number) => padY + (1 - clamp(v, 0, 100) / 100) * plotHeight;

  const lineSegments = chartPoints.map((point) => `${toX(point.t)} ${toY(point.value)}`);
  const linePath = `M ${lineSegments.join(" L ")}`;

  const firstX = toX(chartPoints[0].t);
  const lastX = toX(chartPoints[chartPoints.length - 1].t);
  const areaPath = `M ${firstX} ${bottomY} L ${lineSegments.join(" L ")} L ${lastX} ${bottomY} Z`;

  const latestValue = chartPoints[chartPoints.length - 1].value;
  const xTicks = [
    { x: padX, label: `-${windowSeconds}s` },
    { x: padX + plotWidth / 3, label: `-${Math.round((windowSeconds * 2) / 3)}s` },
    { x: padX + (2 * plotWidth) / 3, label: `-${Math.round(windowSeconds / 3)}s` },
    { x: padX + plotWidth, label: "now" },
  ];

  return (
    <div className="mt-6 rounded-lg border border-slate-700 bg-slate-950/60 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-slate-200">{labelPrefix}</p>
        <p className="text-xs text-rose-300">Current: {latestValue}%</p>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="mt-3 h-52 w-full rounded-md border border-slate-800/90 bg-slate-950/80">
        {[0, 25, 50, 75, 100].map((level) => {
          const y = toY(level);
          return (
            <g key={level}>
              <line x1={padX} y1={y} x2={padX + plotWidth} y2={y} stroke="rgb(51 65 85)" strokeWidth={1} />
              <text x={6} y={y + 4} fill="rgb(148 163 184)" fontSize="11">
                {level}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick) => (
          <g key={tick.label}>
            <line x1={tick.x} y1={padY} x2={tick.x} y2={bottomY} stroke="rgb(30 41 59)" strokeWidth={1} />
            <text x={tick.x} y={height - 4} textAnchor="middle" fill="rgb(148 163 184)" fontSize="11">
              {tick.label}
            </text>
          </g>
        ))}

        <path d={areaPath} fill="rgb(244 63 94 / 0.20)" />
        <path d={linePath} fill="none" stroke="rgb(251 113 133)" strokeWidth={3} strokeLinejoin="round" strokeLinecap="round" />

        <circle
          cx={lastX}
          cy={toY(latestValue)}
          r={4}
          fill="rgb(251 113 133)"
          stroke="rgb(15 23 42)"
          strokeWidth={2}
        />
      </svg>
    </div>
  );
}

type MetricCardProps = {
  label: string;
  value: string;
  tone?: MetricTone;
};

function MetricCard({ label, value, tone = "neutral" }: MetricCardProps) {
  const toneClassByType: Record<MetricTone, string> = {
    neutral: "border-slate-700 text-slate-100",
    good: "border-emerald-500/40 text-emerald-200",
    warn: "border-amber-500/40 text-amber-200",
    alert: "border-rose-500/40 text-rose-200",
  };

  return (
    <div className={`rounded-lg border bg-slate-950/60 p-3 ${toneClassByType[tone]}`}>
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="pt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
