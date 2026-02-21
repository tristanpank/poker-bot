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
  baselineProgress: number;
  baselineStress: number;
  baselineBluff: number;
  bluffDelta: number;
  signalQuality: SignalQuality;
  analysisFps: number;
  streamFps: number;
  updatedAt: string;
};

type TeardownOptions = {
  resetMetrics?: boolean;
  updateState?: boolean;
};

type BluffPoint = {
  t: number;
  value: number;
};

type CvAnalyzeResponse = {
  metrics: VisionMetrics;
};

type CaptureResult = {
  fps: number;
  width: number;
  height: number;
};

const W = 256;
const H = 144;
const CROP_FACTOR = 0.70; // Center crop: keep middle 70% of width and height.
const ANALYSIS_FRAME_MS = 16;
const BLUFF_WINDOW_MS = 30_000;
const SEND_MAX_BITRATE = 45_000_000;
const SEND_MAX_FPS = 60;
const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

const CAPTURE_FPS_TARGETS = [60, 59, 50, 45, 40, 30, 24];
const HIGH_FPS_CAPTURE_PROFILES = [
  { width: 7680, height: 4320 },
  { width: 4096, height: 2160 },
  { width: 3840, height: 2160 },
  { width: 3200, height: 1800 },
  { width: 2560, height: 1440 },
  { width: 1920, height: 1080 },
  { width: 1600, height: 900 },
  { width: 1280, height: 720 },
  { width: 960, height: 540 },
  { width: 640, height: 360 },
];

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
  baselineProgress: 0,
  baselineStress: 0,
  baselineBluff: 0,
  bluffDelta: 0,
  signalQuality: "poor",
  analysisFps: 0,
  streamFps: 0,
  updatedAt: "--",
};

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
const r1 = (v: number) => Math.round(v * 10) / 10;

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

function captureArea(result: CaptureResult): number {
  return result.width * result.height;
}

function shouldPreferCapture(candidate: CaptureResult, current: CaptureResult): boolean {
  if (candidate.fps > current.fps + 0.5) {
    return true;
  }
  if (candidate.fps + 0.5 < current.fps) {
    return false;
  }
  return captureArea(candidate) > captureArea(current);
}

function readCaptureResult(track: MediaStreamTrack): CaptureResult {
  const settings = track.getSettings();
  return {
    fps: settings.frameRate ?? 0,
    width: settings.width ?? 0,
    height: settings.height ?? 0,
  };
}

async function tuneCaptureTrack(track: MediaStreamTrack): Promise<CaptureResult> {
  let best = readCaptureResult(track);

  for (const fpsTarget of CAPTURE_FPS_TARGETS) {
    for (const profile of HIGH_FPS_CAPTURE_PROFILES) {
      try {
        await track.applyConstraints({
          width: { ideal: profile.width, max: profile.width },
          height: { ideal: profile.height, max: profile.height },
          frameRate: { ideal: fpsTarget, max: fpsTarget },
        });
      } catch {
        continue;
      }

      const candidate = readCaptureResult(track);
      if (shouldPreferCapture(candidate, best)) {
        best = candidate;
      }
    }
  }

  if (best.width > 0 && best.height > 0) {
    try {
      await track.applyConstraints({
        width: { ideal: best.width, max: best.width },
        height: { ideal: best.height, max: best.height },
        frameRate: {
          ideal: Math.max(24, Math.round(best.fps || 30)),
          max: Math.max(24, Math.round(best.fps || 30)),
        },
      });
      best = readCaptureResult(track);
    } catch {
      // Keep current settings if best-profile reapply fails.
    }
  }

  return best;
}

export default function Home() {
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const localStreamRef = useRef<MediaStream | null>(null);
  const senderPcRef = useRef<RTCPeerConnection | null>(null);
  const receiverPcRef = useRef<RTCPeerConnection | null>(null);
  const backendPcRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);

  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastSampleTsRef = useRef<number | null>(null);
  const lastDecodedTsRef = useRef<number | null>(null);
  const lastDecodedFramesRef = useRef<number | null>(null);
  const lastAnalyzedDecodedFramesRef = useRef<number | null>(null);
  const lastAnalyzedVideoTimeRef = useRef<number | null>(null);
  const smoothStreamFpsRef = useRef(0);
  const requestInFlightRef = useRef(false);
  const sessionIdRef = useRef(createSessionId());
  const activeRef = useRef(false);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<VisionMetrics>(INITIAL);
  const [bluffHistory, setBluffHistory] = useState<BluffPoint[]>([]);
  const [captureInfo, setCaptureInfo] = useState<string>("--");

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

    if (backendPcRef.current) {
      backendPcRef.current.close();
      backendPcRef.current = null;
    }
    dataChannelRef.current = null;

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
    lastSampleTsRef.current = null;
    lastDecodedTsRef.current = null;
    lastDecodedFramesRef.current = null;
    lastAnalyzedDecodedFramesRef.current = null;
    lastAnalyzedVideoTimeRef.current = null;
    smoothStreamFpsRef.current = 0;
    requestInFlightRef.current = false;

    const previousSessionId = sessionIdRef.current;
    sessionIdRef.current = createSessionId();
    void fetch(`${BACKEND_BASE_URL}/cv/session`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: previousSessionId }),
    }).catch(() => undefined);

    if (updateState) setIsStreaming(false);
    if (resetMetrics) {
      setMetrics(INITIAL);
      setBluffHistory([]);
      setCaptureInfo("--");
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
          let decodedFrames: number | null = null;
          if (typeof video.getVideoPlaybackQuality === "function") {
            const quality = video.getVideoPlaybackQuality();
            decodedFrames = quality.totalVideoFrames;

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

          const hasNewDecodedFrame =
            decodedFrames !== null
              ? lastAnalyzedDecodedFramesRef.current === null ||
                decodedFrames > lastAnalyzedDecodedFramesRef.current
              : lastAnalyzedVideoTimeRef.current === null ||
                video.currentTime > lastAnalyzedVideoTimeRef.current + 0.0005;

          if (!hasNewDecodedFrame) {
            rafRef.current = requestAnimationFrame(loop);
            return;
          }

          if (canvas.width !== W) canvas.width = W;
          if (canvas.height !== H) canvas.height = H;

          if (!ctxRef.current) {
            ctxRef.current = canvas.getContext("2d", { willReadFrequently: true });
          }

          const ctx = ctxRef.current;
          if (ctx) {
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            const cropX = vw * ((1 - CROP_FACTOR) / 2);
            const cropY = vh * ((1 - CROP_FACTOR) / 2);
            ctx.drawImage(video, cropX, cropY, vw * CROP_FACTOR, vh * CROP_FACTOR, 0, 0, W, H);

            const dc = dataChannelRef.current;
            if (dc && dc.readyState === "open") {
              // WebRTC DataChannel transport: fire-and-forget, metrics arrive via dc.onmessage.
              const frame = ctx.getImageData(0, 0, W, H);
              const timestamp = Date.now();
              const headerJson = JSON.stringify({
                sessionId: sessionIdRef.current,
                timestamp,
                width: W,
                height: H,
                streamFps: r1(streamFps),
              });
              const headerBytes = new TextEncoder().encode(headerJson);
              const lenBuf = new ArrayBuffer(4);
              new DataView(lenBuf).setUint32(0, headerBytes.length, true);
              const payload = new Uint8Array(4 + headerBytes.length + frame.data.length);
              payload.set(new Uint8Array(lenBuf), 0);
              payload.set(headerBytes, 4);
              payload.set(frame.data, 4 + headerBytes.length);
              if (decodedFrames !== null) {
                lastAnalyzedDecodedFramesRef.current = decodedFrames;
              }
              lastAnalyzedVideoTimeRef.current = video.currentTime;
              try {
                dc.send(payload.buffer);
              } catch {
                // DataChannel send failed; next tick will retry or fall back to HTTP POST.
              }
            } else if (!requestInFlightRef.current) {
              // HTTP POST fallback.
              const frame = ctx.getImageData(0, 0, W, H);
              const timestamp = Date.now();
              const params = new URLSearchParams({
                sessionId: sessionIdRef.current,
                timestamp: String(timestamp),
                width: String(W),
                height: String(H),
                streamFps: String(r1(streamFps)),
              });

              requestInFlightRef.current = true;
              if (decodedFrames !== null) {
                lastAnalyzedDecodedFramesRef.current = decodedFrames;
              }
              lastAnalyzedVideoTimeRef.current = video.currentTime;

              void fetch(`${BACKEND_BASE_URL}/cv/analyze-raw?${params.toString()}`, {
                method: "POST",
                headers: { "Content-Type": "application/octet-stream" },
                body: frame.data,
              })
                .then(async (response) => {
                  if (!response.ok) {
                    throw new Error(`Backend analysis failed (${response.status})`);
                  }
                  return (await response.json()) as CvAnalyzeResponse;
                })
                .then((result) => {
                  if (!activeRef.current) {
                    return;
                  }

                  setError(null);
                  setMetrics(result.metrics);
                  setBluffHistory((previous) => {
                    const now = performance.now();
                    const next = [...previous, { t: now, value: result.metrics.bluffRisk }];
                    const minTs = now - BLUFF_WINDOW_MS;
                    let firstValidIndex = 0;

                    while (firstValidIndex < next.length && next[firstValidIndex].t < minTs) {
                      firstValidIndex += 1;
                    }

                    return firstValidIndex > 0 ? next.slice(firstValidIndex) : next;
                  });
                })
                .catch((caughtError) => {
                  const message =
                    caughtError instanceof Error
                      ? caughtError.message
                      : "Unknown backend analysis error.";
                  setError(message);
                })
                .finally(() => {
                  requestInFlightRef.current = false;
                });
            }
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
          width: { ideal: 3840, min: 640, max: 7680 },
          height: { ideal: 2160, min: 360, max: 4320 },
          frameRate: { ideal: SEND_MAX_FPS, max: SEND_MAX_FPS },
          facingMode: { ideal: "user" },
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
        videoTrack.contentHint = "motion";
      } catch {
        // Keep default content hint where unsupported.
      }

      const capture = await tuneCaptureTrack(videoTrack);
      const displayFps = r1(capture.fps);
      setCaptureInfo(
        displayFps > 0 && capture.width > 0 && capture.height > 0
          ? `${displayFps} fps @ ${capture.width}x${capture.height}`
          : "--",
      );

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

      lastSampleTsRef.current = null;
      lastDecodedTsRef.current = null;
      lastDecodedFramesRef.current = null;
      lastAnalyzedDecodedFramesRef.current = null;
      lastAnalyzedVideoTimeRef.current = null;
      smoothStreamFpsRef.current = 0;
      requestInFlightRef.current = false;
      sessionIdRef.current = createSessionId();

      activeRef.current = true;
      setIsStreaming(true);
      rafRef.current = requestAnimationFrame(runAnalysis);

      // Attempt to set up a WebRTC DataChannel to the backend for efficient
      // frame delivery.  Failures are non-fatal; HTTP POST remains the fallback.
      try {
        const backendPc = new RTCPeerConnection({
          iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });
        backendPcRef.current = backendPc;

        const dc = backendPc.createDataChannel("frames", {
          // Unordered, unreliable: drop stale frames rather than buffer them.
          // For real-time analysis it is always better to process the freshest
          // frame available than to deliver an old one reliably.
          ordered: false,
          maxRetransmits: 0,
        });
        dataChannelRef.current = dc;

        dc.onmessage = (event) => {
          if (!activeRef.current) return;
          try {
            const result = JSON.parse(event.data as string) as CvAnalyzeResponse;
            setError(null);
            setMetrics(result.metrics);
            setBluffHistory((previous) => {
              const now = performance.now();
              const next = [...previous, { t: now, value: result.metrics.bluffRisk }];
              const minTs = now - BLUFF_WINDOW_MS;
              let firstValidIndex = 0;
              while (firstValidIndex < next.length && next[firstValidIndex].t < minTs) {
                firstValidIndex += 1;
              }
              return firstValidIndex > 0 ? next.slice(firstValidIndex) : next;
            });
          } catch {
            // Ignore malformed responses.
          }
        };

        const offer = await backendPc.createOffer();
        await backendPc.setLocalDescription(offer);

        const sigResp = await fetch(`${BACKEND_BASE_URL}/cv/webrtc-offer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: offer.sdp,
            type: offer.type,
            sessionId: sessionIdRef.current,
          }),
        });

        if (sigResp.ok) {
          const answer = (await sigResp.json()) as { sdp: string; type: string };
          await backendPc.setRemoteDescription(new RTCSessionDescription(answer));
        }
      } catch {
        // WebRTC DataChannel setup failed; HTTP POST will be used for all frames.
      }
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
            Streams camera via WebRTC, sends compact frame data to a backend CV pipeline,
            and receives real-time stress/emotion/bluff-pressure metrics.
          </p>
          <p className="max-w-3xl text-xs text-slate-400">
            Heuristic only: visual cues are not a reliable lie detector.
          </p>
          <p className="max-w-3xl text-xs text-slate-400">Capture negotiated (max-data profile): {captureInfo}</p>

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
            <MetricCard label="Baseline stress (5m)" value={`${metrics.baselineStress}%`} />
            <MetricCard label="Emotion state" value={metrics.emotion} tone={toneForEmotion(metrics.emotion)} />
            <MetricCard label="Bluff pressure" value={`${metrics.bluffRisk}%`} tone={toneForBluff(metrics.bluffLevel)} />
            <MetricCard label="Bluff level" value={metrics.bluffLevel} tone={toneForBluff(metrics.bluffLevel)} />
            <MetricCard label="Baseline bluff (5m)" value={`${metrics.baselineBluff}%`} />
            <MetricCard
              label="Bluff delta vs baseline"
              value={`${metrics.bluffDelta > 0 ? "+" : ""}${metrics.bluffDelta}%`}
              tone={metrics.bluffDelta >= 16 ? "alert" : metrics.bluffDelta >= 8 ? "warn" : "good"}
            />
            <MetricCard
              label="Baseline window fill"
              value={`${metrics.baselineProgress}%`}
              tone={metrics.baselineProgress >= 85 ? "good" : metrics.baselineProgress >= 40 ? "warn" : "neutral"}
            />

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
