'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type ActivityZone = 'none' | 'left' | 'center' | 'right';
export type EmotionState = 'unknown' | 'calm' | 'focused' | 'tense' | 'agitated';
export type BluffLevel = 'low' | 'watch' | 'elevated';
export type SignalQuality = 'poor' | 'fair' | 'good';
export type VisionMetrics = {
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

export type BluffPoint = {
  t: number;
  value: number;
};

type CaptureResult = {
  fps: number;
  width: number;
  height: number;
};

type StartStreamArgs = {
  sessionId: string;
};

type UseCvWebRtcStreamOptions = {
  backendBaseUrl: string;
  clearBackendSessionOnStop?: boolean;
};

type StreamPhase =
  | 'idle'
  | 'requesting_camera'
  | 'tuning_camera'
  | 'negotiating_webrtc'
  | 'waiting_for_backend'
  | 'connecting'
  | 'live';

const ANALYSIS_FRAME_MS = 16;
const BLUFF_WINDOW_MS = 30_000;
const SEND_MAX_BITRATE = 12_000_000;
const SEND_MAX_FPS = 30;
const CAPTURE_FPS_TARGETS = [30, 24];
const HIGH_FPS_CAPTURE_PROFILES = [
  { width: 1280, height: 720 },
  { width: 960, height: 540 },
  { width: 640, height: 360 },
];

export const INITIAL_VISION_METRICS: VisionMetrics = {
  brightness: 0,
  motion: 0,
  edgeDensity: 0,
  activityZone: 'none',
  pulseBpm: null,
  pulseConfidence: 0,
  skinCoverage: 0,
  stress: 0,
  emotion: 'unknown',
  bluffRisk: 0,
  bluffLevel: 'low',
  baselineProgress: 0,
  baselineStress: 0,
  baselineBluff: 0,
  bluffDelta: 0,
  signalQuality: 'poor',
  analysisFps: 0,
  streamFps: 0,
  updatedAt: '--',
};

export function createCvStreamSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function waitForIceGatheringComplete(
  pc: RTCPeerConnection,
  timeoutMs = 500
): Promise<'complete' | 'timeout'> {
  if (pc.iceGatheringState === 'complete') {
    return Promise.resolve('complete');
  }

  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => {
      pc.removeEventListener('icegatheringstatechange', handleStateChange);
      resolve('timeout');
    }, timeoutMs);

    const handleStateChange = () => {
      if (pc.iceGatheringState === 'complete') {
        window.clearTimeout(timeout);
        pc.removeEventListener('icegatheringstatechange', handleStateChange);
        resolve('complete');
      }
    };

    pc.addEventListener('icegatheringstatechange', handleStateChange);
  });
}

const r1 = (v: number) => Math.round(v * 10) / 10;

function rankCodec(mimeType: string): number {
  const normalized = mimeType.toLowerCase();
  if (normalized === 'video/h264') return 0;
  if (normalized === 'video/vp9') return 1;
  if (normalized === 'video/vp8') return 2;
  return 3;
}

function preferVideoCodecs(transceiver: RTCRtpTransceiver): void {
  if (typeof RTCRtpSender === 'undefined' || !RTCRtpSender.getCapabilities) {
    return;
  }

  const caps = RTCRtpSender.getCapabilities('video');
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
        typeof encoding.maxBitrate === 'number'
          ? Math.max(encoding.maxBitrate, SEND_MAX_BITRATE)
          : SEND_MAX_BITRATE,
      maxFramerate:
        typeof encoding.maxFramerate === 'number'
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

function shouldPreferCapture(
  candidate: CaptureResult,
  current: CaptureResult,
): boolean {
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

  if (best.fps >= 24 && best.width >= 640 && best.height >= 360) {
    return best;
  }

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
        if (best.fps >= 24 && best.width >= 640 && best.height >= 360) {
          return best;
        }
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

export function useCvWebRtcStream({
  backendBaseUrl,
  clearBackendSessionOnStop = false,
}: UseCvWebRtcStreamOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const metaIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCallbackRef = useRef<number | null>(null);
  const frameIdRef = useRef(0);
  const activeRef = useRef(false);
  const activeSessionIdRef = useRef<string | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<VisionMetrics>(INITIAL_VISION_METRICS);
  const [bluffHistory, setBluffHistory] = useState<BluffPoint[]>([]);
  const [captureInfo, setCaptureInfo] = useState('--');
  const [phase, setPhase] = useState<StreamPhase>('idle');

  const teardown = useCallback(async (resetMetrics: boolean) => {
    activeRef.current = false;

    if (metaIntervalRef.current !== null) {
      clearInterval(metaIntervalRef.current);
      metaIntervalRef.current = null;
    }

    if (frameCallbackRef.current !== null && videoRef.current) {
      videoRef.current.cancelVideoFrameCallback?.(frameCallbackRef.current);
      frameCallbackRef.current = null;
    }

    dcRef.current?.close();
    dcRef.current = null;
    pcRef.current?.close();
    pcRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const endedSessionId = activeSessionIdRef.current;
    activeSessionIdRef.current = null;
    frameIdRef.current = 0;
    setIsStreaming(false);
    setPhase('idle');

    if (clearBackendSessionOnStop && endedSessionId) {
      await fetch(`${backendBaseUrl}/cv/session`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: endedSessionId }),
      }).catch(() => undefined);
    }

    if (resetMetrics) {
      setMetrics(INITIAL_VISION_METRICS);
      setBluffHistory([]);
      setCaptureInfo('--');
    }
  }, [backendBaseUrl, clearBackendSessionOnStop]);

  const stopStream = useCallback(async () => {
    await teardown(true);
  }, [teardown]);

  useEffect(() => {
    return () => {
      void teardown(false);
    };
  }, [teardown]);

  const startStream = useCallback(async ({ sessionId }: StartStreamArgs) => {
    if (isStreaming) {
      return true;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not support camera streaming.');
      return false;
    }

    setError(null);
    await teardown(true);

    try {
      setPhase('requesting_camera');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640, max: 1920 },
          height: { ideal: 720, min: 360, max: 1080 },
          frameRate: { ideal: SEND_MAX_FPS, max: SEND_MAX_FPS },
          facingMode: { ideal: 'user' },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        void videoRef.current.play().catch(() => undefined);
      }

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
      pcRef.current = pc;

      pc.onconnectionstatechange = () => {
        console.info('CV WebRTC connection state:', pc.connectionState);
      };

      pc.oniceconnectionstatechange = () => {
        console.info('CV WebRTC ICE state:', pc.iceConnectionState);
      };

      const [videoTrack] = stream.getVideoTracks();
      if (!videoTrack) {
        throw new Error('No video track available from camera.');
      }

      try {
        videoTrack.contentHint = 'motion';
      } catch {
        // Keep default content hint where unsupported.
      }

      setPhase('tuning_camera');
      const capture = await tuneCaptureTrack(videoTrack);
      const displayFps = r1(capture.fps);
      setCaptureInfo(
        displayFps > 0 && capture.width > 0 && capture.height > 0
          ? `${displayFps} fps @ ${capture.width}x${capture.height}`
          : '--',
      );

      const transceiver = pc.addTransceiver(videoTrack, {
        direction: 'sendonly',
        streams: [stream],
      });
      preferVideoCodecs(transceiver);
      await tuneVideoSender(transceiver.sender);

      const dc = pc.createDataChannel('metadata');
      dcRef.current = dc;

      dc.onmessage = (event) => {
        try {
          const result = JSON.parse(String(event.data)) as VisionMetrics;
          if (!activeRef.current) {
            return;
          }

          setPhase('live');
          setMetrics(result);
          setBluffHistory((previous) => {
            const now = performance.now();
            const next = [...previous, { t: now, value: result.bluffRisk }];
            const minTs = now - BLUFF_WINDOW_MS;
            let firstValidIndex = 0;

            while (
              firstValidIndex < next.length &&
              next[firstValidIndex].t < minTs
            ) {
              firstValidIndex += 1;
            }

            return firstValidIndex > 0 ? next.slice(firstValidIndex) : next;
          });
        } catch {
          setError('Failed to parse backend CV metrics.');
        }
      };

      dc.onopen = () => {
        setPhase('connecting');
        const videoEl = videoRef.current;
        if (!videoEl) {
          return;
        }

        const sendMetadata = (captureTs: number, presentedFrames?: number) => {
          if (!activeRef.current || dc.readyState !== 'open') {
            return;
          }

          const trackSettings = videoTrack.getSettings();
          const streamFps =
            typeof trackSettings.frameRate === 'number' ? r1(trackSettings.frameRate) : 0;

          dc.send(JSON.stringify({
            sessionId,
            frameId: presentedFrames ?? frameIdRef.current++,
            captureTs: Math.round(captureTs),
            streamFps,
            cropWidth: trackSettings.width ?? 0,
            cropHeight: trackSettings.height ?? 0,
          }));
        };

        if (typeof videoEl.requestVideoFrameCallback === 'function') {
          const loop = (_now: number, metadata: VideoFrameCallbackMetadata) => {
            sendMetadata(
              performance.timeOrigin + metadata.expectedDisplayTime,
              metadata.presentedFrames,
            );
            frameCallbackRef.current = videoEl.requestVideoFrameCallback(loop);
          };

          frameCallbackRef.current = videoEl.requestVideoFrameCallback(loop);
          return;
        }

        metaIntervalRef.current = setInterval(() => {
          sendMetadata(Date.now());
        }, ANALYSIS_FRAME_MS);
      };

      setPhase('negotiating_webrtc');
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const iceGatheringResult = await waitForIceGatheringComplete(pc);
      console.info('CV WebRTC ICE gathering result:', iceGatheringResult);

      setPhase('waiting_for_backend');
      const response = await fetch(`${backendBaseUrl}/cv/webrtc/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp ?? offer.sdp,
          type: pc.localDescription?.type ?? offer.type,
          sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`WebRTC negotiation failed (${response.status})`);
      }

      const answer = (await response.json()) as RTCSessionDescriptionInit;
      await pc.setRemoteDescription(new RTCSessionDescription(answer));

      activeSessionIdRef.current = sessionId;
      activeRef.current = true;
      setPhase('connecting');
      setIsStreaming(true);
      return true;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unknown error while starting stream.';
      setError(`Unable to start camera stream: ${message}`);
      await teardown(true);
      return false;
    }
  }, [backendBaseUrl, isStreaming, teardown]);

  return {
    videoRef,
    isStreaming,
    error,
    metrics,
    bluffHistory,
    captureInfo,
    phase,
    startStream,
    stopStream,
  };
}
