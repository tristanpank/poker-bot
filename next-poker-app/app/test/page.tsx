"use client";

import { useCallback, useRef, useState } from "react";

import {
  BackendStreamStatus,
  BluffPoint,
  BluffLevel,
  EmotionState,
  SignalQuality,
  createCvStreamSessionId,
  useCvWebRtcStream,
} from "../lib/useCvWebRtcStream";

type MetricTone = "neutral" | "good" | "warn" | "alert";

const BLUFF_WINDOW_MS = 30_000;
const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

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

function resolutionLabel(width: number, height: number): string {
  if (width <= 0 || height <= 0) {
    return "--";
  }

  return `${width}x${height}`;
}

function scaleLabel(session: BackendStreamStatus): string {
  const deliveredPixels = session.frameWidth * session.frameHeight;
  const capturedPixels = session.captureWidth * session.captureHeight;
  if (deliveredPixels <= 0 || capturedPixels <= 0) {
    return "--";
  }

  const ratio = deliveredPixels / capturedPixels;
  return `${Math.round(ratio * 100)}%`;
}

export default function TestPage() {
  const [currentSessionId, setCurrentSessionId] = useState(() =>
    createCvStreamSessionId(),
  );
  const sessionIdRef = useRef(currentSessionId);
  const {
    videoRef: localVideoRef,
    isStreaming,
    error,
    metrics,
    bluffHistory,
    captureInfo,
    backendStatus,
    currentBackendStream,
    startStream,
    stopStream,
  } = useCvWebRtcStream({
    backendBaseUrl: BACKEND_BASE_URL,
    clearBackendSessionOnStop: true,
  });

  const start = useCallback(async () => {
    await startStream({ sessionId: sessionIdRef.current });
  }, [startStream]);

  const stop = useCallback(async () => {
    await stopStream();
    sessionIdRef.current = createCvStreamSessionId();
    setCurrentSessionId(sessionIdRef.current);
  }, [stopStream]);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <section className="flex flex-col gap-3 rounded-2xl border border-slate-700 bg-slate-900/60 p-6 shadow-xl shadow-slate-950/30">
          <h1 className="text-2xl font-semibold tracking-tight">
            WebRTC Bluff Signal + Lightweight POS CV
          </h1>
          <p className="max-w-3xl text-sm text-slate-300">
            This page uses the exact same WebRTC CV client logic as the real join
            page, but shows the full metrics dashboard for debugging.
          </p>
          <p className="max-w-3xl text-xs text-slate-400">
            Heuristic only: visual cues are not a reliable lie detector.
          </p>
          <p className="max-w-3xl text-xs text-slate-400">
            Capture negotiated (shared client): {captureInfo}
          </p>
          <p className="max-w-3xl text-xs text-slate-400">
            Backend sees {backendStatus.activeStreamCount} active stream
            {backendStatus.activeStreamCount === 1 ? "" : "s"} right now.
            {currentBackendStream
              ? ` This test page is receiving ${currentBackendStream.analysisFps.toFixed(1)} analysis FPS from the backend while the backend is processing ${currentBackendStream.streamFps.toFixed(1)} stream FPS for this session. Capture is ${resolutionLabel(currentBackendStream.captureWidth, currentBackendStream.captureHeight)} and backend-delivered frames are ${resolutionLabel(currentBackendStream.frameWidth, currentBackendStream.frameHeight)}.`
              : " This page will appear in the backend stream table once its WebRTC session is connected."}
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              type="button"
              onClick={start}
              disabled={isStreaming}
              className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              Start stream
            </button>

            <button
              type="button"
              onClick={stop}
              disabled={!isStreaming}
              className="rounded-md bg-rose-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              Stop stream
            </button>

            <span
              className={`rounded-md px-3 py-1 text-xs font-medium ${
                isStreaming
                  ? "bg-emerald-500/20 text-emerald-300"
                  : "bg-slate-700 text-slate-200"
              }`}
            >
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
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
              Shared local camera feed
            </h2>
            <video
              ref={localVideoRef}
              className="aspect-video w-full rounded-lg border border-slate-700 bg-black object-cover"
              autoPlay
              muted
              playsInline
            />
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
              Real-page parity
            </h2>
            <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-slate-700 bg-slate-950 text-center text-sm text-slate-400">
              The same streaming client powers both the test page and the real
              join page.
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/60 p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
                Backend stream load
              </h2>
              <p className="mt-1 text-xs text-slate-400">
                This shows every active WebRTC CV stream the backend is handling,
                so you can compare this page against the rest of the system load.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-slate-300">
              Active streams: {backendStatus.activeStreamCount}
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full text-left text-xs text-slate-300">
              <thead className="text-slate-500">
                <tr className="border-b border-slate-800">
                  <th className="px-3 py-2 font-medium">Session</th>
                  <th className="px-3 py-2 font-medium">State</th>
                  <th className="px-3 py-2 font-medium">Analysis FPS</th>
                  <th className="px-3 py-2 font-medium">Stream FPS</th>
                  <th className="px-3 py-2 font-medium">Inferred FPS</th>
                  <th className="px-3 py-2 font-medium">Frames</th>
                  <th className="px-3 py-2 font-medium">Delivered</th>
                  <th className="px-3 py-2 font-medium">Capture</th>
                  <th className="px-3 py-2 font-medium">Scale</th>
                  <th className="px-3 py-2 font-medium">Signal</th>
                </tr>
              </thead>
              <tbody>
                {backendStatus.sessions.length === 0 ? (
                  <tr>
                    <td
                      className="px-3 py-4 text-slate-500"
                      colSpan={10}
                    >
                      No active backend CV streams.
                    </td>
                  </tr>
                ) : (
                  backendStatus.sessions.map((session) => {
                    const isCurrent = session.sessionId === currentSessionId;
                    return (
                      <tr
                        key={session.sessionId}
                        className={`border-b border-slate-900/80 ${
                          isCurrent ? "bg-emerald-500/10" : ""
                        }`}
                      >
                        <td className="px-3 py-2 font-mono text-[11px] text-slate-200">
                          {isCurrent ? "This page" : session.sessionId.slice(0, 8)}
                        </td>
                        <td className="px-3 py-2">{session.connectionState}</td>
                        <td className="px-3 py-2">{session.analysisFps.toFixed(1)}</td>
                        <td className="px-3 py-2">{session.streamFps.toFixed(1)}</td>
                        <td className="px-3 py-2">
                          {session.inferredStreamFps.toFixed(1)}
                        </td>
                        <td className="px-3 py-2">{session.framesReceived}</td>
                        <td className="px-3 py-2">
                          {resolutionLabel(session.frameWidth, session.frameHeight)}
                        </td>
                        <td className="px-3 py-2">
                          {resolutionLabel(
                            session.captureWidth,
                            session.captureHeight,
                          )}
                        </td>
                        <td className="px-3 py-2">{scaleLabel(session)}</td>
                        <td className="px-3 py-2">{session.signalQuality}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/60 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-300">
            Real-time deception proxy metrics
          </h2>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Pulse (POS)"
              value={numberLabel(metrics.pulseBpm, " BPM")}
            />
            <MetricCard
              label="Pulse confidence"
              value={`${metrics.pulseConfidence}%`}
              tone={metrics.pulseConfidence >= 55 ? "good" : "warn"}
            />
            <MetricCard
              label="Signal quality"
              value={metrics.signalQuality}
              tone={toneForQuality(metrics.signalQuality)}
            />
            <MetricCard
              label="Skin coverage"
              value={`${metrics.skinCoverage}%`}
              tone={metrics.skinCoverage >= 35 ? "good" : "warn"}
            />

            <MetricCard
              label="Stress index"
              value={`${metrics.stress}%`}
              tone={
                metrics.stress >= 68
                  ? "alert"
                  : metrics.stress >= 42
                    ? "warn"
                    : "good"
              }
            />
            <MetricCard
              label="Baseline stress (5m)"
              value={`${metrics.baselineStress}%`}
            />
            <MetricCard
              label="Emotion state"
              value={metrics.emotion}
              tone={toneForEmotion(metrics.emotion)}
            />
            <MetricCard
              label="Bluff pressure"
              value={`${metrics.bluffRisk}%`}
              tone={toneForBluff(metrics.bluffLevel)}
            />
            <MetricCard
              label="Bluff level"
              value={metrics.bluffLevel}
              tone={toneForBluff(metrics.bluffLevel)}
            />
            <MetricCard
              label="Baseline bluff (5m)"
              value={`${metrics.baselineBluff}%`}
            />
            <MetricCard
              label="Bluff delta vs baseline"
              value={`${metrics.bluffDelta > 0 ? "+" : ""}${metrics.bluffDelta}%`}
              tone={
                metrics.bluffDelta >= 16
                  ? "alert"
                  : metrics.bluffDelta >= 8
                    ? "warn"
                    : "good"
              }
            />
            <MetricCard
              label="Baseline window fill"
              value={`${metrics.baselineProgress}%`}
              tone={
                metrics.baselineProgress >= 85
                  ? "good"
                  : metrics.baselineProgress >= 40
                    ? "warn"
                    : "neutral"
              }
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
            Optimized for low compute cost and responsiveness, not forensic
            certainty.
          </p>
        </section>
      </main>
    </div>
  );
}

type BluffPressureChartProps = {
  points: BluffPoint[];
  windowMs: number;
};

function BluffPressureChart({
  points,
  windowMs,
}: BluffPressureChartProps) {
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

  const lineSegments = chartPoints.map(
    (point) => `${toX(point.t)} ${toY(point.value)}`,
  );
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

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="mt-3 h-52 w-full rounded-md border border-slate-800/90 bg-slate-950/80"
      >
        {[0, 25, 50, 75, 100].map((level) => {
          const y = toY(level);
          return (
            <g key={level}>
              <line
                x1={padX}
                y1={y}
                x2={padX + plotWidth}
                y2={y}
                stroke="rgb(51 65 85)"
                strokeWidth={1}
              />
              <text x={6} y={y + 4} fill="rgb(148 163 184)" fontSize="11">
                {level}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick) => (
          <g key={tick.label}>
            <line
              x1={tick.x}
              y1={padY}
              x2={tick.x}
              y2={bottomY}
              stroke="rgb(30 41 59)"
              strokeWidth={1}
            />
            <text
              x={tick.x}
              y={height - 4}
              textAnchor="middle"
              fill="rgb(148 163 184)"
              fontSize="11"
            >
              {tick.label}
            </text>
          </g>
        ))}

        <path d={areaPath} fill="rgb(244 63 94 / 0.20)" />
        <path
          d={linePath}
          fill="none"
          stroke="rgb(251 113 133)"
          strokeWidth={3}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

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
