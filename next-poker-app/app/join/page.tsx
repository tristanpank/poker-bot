'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

type JoinState = 'idle' | 'joined' | 'streaming';

export default function JoinPage() {
  const [code, setCode] = useState('');
  const [playerPosition, setPlayerPosition] = useState(1);
  const [joinState, setJoinState] = useState<JoinState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [cvSessionId, setCvSessionId] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const metaIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeRef = useRef(false);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      activeRef.current = false;
      if (metaIntervalRef.current) clearInterval(metaIntervalRef.current);
      if (dcRef.current) dcRef.current.close();
      if (pcRef.current) pcRef.current.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const handleJoin = useCallback(async () => {
    setError(null);
    if (!code.trim()) {
      setError('Please enter a join code.');
      return;
    }

    try {
      const res = await fetch(`${BACKEND}/session/webcam/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code.trim().toUpperCase(), player_position: playerPosition }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `Failed to join (${res.status})`);
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setCvSessionId(data.cv_session_id);
      setJoinState('joined');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to join session.');
    }
  }, [code, playerPosition]);

  const startWebcam = useCallback(async () => {
    if (!cvSessionId || !sessionId) return;
    setError(null);

    if (joinState === 'joined') {
      await fetch(`${BACKEND}/session/webcam/reconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, player_position: playerPosition }),
      }).catch(() => undefined);
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 15 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        void videoRef.current.play().catch(() => undefined);
      }

      activeRef.current = true;
      setJoinState('streaming');

      // Setup WebRTC
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });
      pcRef.current = pc;

      stream.getTracks().forEach((track) => pc.addTrack(track, stream));

      const dc = pc.createDataChannel('metadata');
      dcRef.current = dc;

      dc.onopen = () => {
        let frameId = 0;
        metaIntervalRef.current = setInterval(() => {
          if (dc.readyState === 'open' && activeRef.current) {
            dc.send(JSON.stringify({
              sessionId: cvSessionId,
              frameId: frameId++,
              captureTs: Date.now(),
              streamFps: 15
            }));
          }
        }, 100);
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const res = await fetch(`${BACKEND}/cv/webrtc/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          session_id: cvSessionId
        })
      });

      if (!res.ok) {
        throw new Error('Failed to negotiate WebRTC connection');
      }

      const answer = await res.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
      setError(
        err instanceof Error ? `Camera error: ${err.message}` : 'Unable to access camera.'
      );
    }
  }, [cvSessionId, sessionId]);

  const stopWebcam = useCallback(async () => {
    activeRef.current = false;
    if (metaIntervalRef.current) {
      clearInterval(metaIntervalRef.current);
      metaIntervalRef.current = null;
    }
    if (dcRef.current) {
      dcRef.current.close();
      dcRef.current = null;
    }
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;

    // Notify backend of disconnect
    if (sessionId) {
      await fetch(`${BACKEND}/session/webcam/disconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, player_position: playerPosition }),
      }).catch(() => undefined);
    }

    setJoinState('joined');
  }, [sessionId, playerPosition]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-4 py-10 text-slate-100 flex items-start justify-center"
      style={{ fontFamily: "'Outfit', sans-serif" }}>
      <main className="w-full max-w-lg flex flex-col gap-6">
        {/* Header */}
        <section className="text-center">
          <div className="inline-flex items-center gap-2 mb-3">
            <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-medium uppercase tracking-widest text-emerald-400">
              Opponent Webcam
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
            Join Poker Session
          </h1>
          <p className="mt-2 text-sm text-slate-400 max-w-md mx-auto">
            Enter the code shared by the bot operator to connect your webcam for bluff analysis.
          </p>
        </section>

        {/* Join form */}
        {joinState === 'idle' && (
          <section className="rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm p-6 shadow-2xl shadow-black/20 flex flex-col gap-5">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1.5 block">
                Session Code
              </label>
              <input
                type="text"
                maxLength={6}
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
                placeholder="ABC123"
                className="w-full rounded-xl border border-slate-600/50 bg-slate-800/80 px-4 py-3 text-center text-2xl font-mono font-bold tracking-[0.3em] text-white placeholder:text-slate-600 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 transition-all"
              />
            </div>

            <div>
              <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1.5 block">
                Your Position (relative to bot)
              </label>
              <div className="grid grid-cols-5 gap-2">
                {[1, 2, 3, 4, 5].map((pos) => (
                  <button
                    key={pos}
                    onClick={() => setPlayerPosition(pos)}
                    className={`rounded-xl py-3 text-sm font-semibold transition-all ${playerPosition === pos
                        ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/25'
                        : 'bg-slate-800/80 text-slate-300 border border-slate-600/30 hover:bg-slate-700/80 hover:border-slate-500/40'
                      }`}
                  >
                    P{pos}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleJoin}
              disabled={code.trim().length < 6}
              className="mt-1 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-6 py-3.5 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition-all hover:from-emerald-400 hover:to-emerald-500 hover:shadow-emerald-500/30 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
            >
              Join Session
            </button>

            {error && (
              <p className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 text-center">
                {error}
              </p>
            )}
          </section>
        )}

        {/* Camera preview / streaming state */}
        {joinState !== 'idle' && (
          <section className="rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm p-6 shadow-2xl shadow-black/20 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400 font-medium">
                  Connected as <span className="text-emerald-400 font-semibold">Player {playerPosition}</span>
                </p>
                <p className="text-xs text-slate-500 mt-0.5">Session: {sessionId?.slice(0, 8)}…</p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${joinState === 'streaming'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'bg-slate-700/50 text-slate-400 border border-slate-600/30'
                  }`}
              >
                {joinState === 'streaming' ? '● Live' : 'Ready'}
              </span>
            </div>

            <div className="rounded-xl overflow-hidden border border-slate-700/50 bg-black aspect-video">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />
            </div>

            <div className="flex gap-3">
              {joinState === 'joined' && (
                <button
                  onClick={startWebcam}
                  className="flex-1 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition-all hover:from-emerald-400 hover:to-emerald-500"
                >
                  Start Webcam
                </button>
              )}
              {joinState === 'streaming' && (
                <button
                  onClick={stopWebcam}
                  className="flex-1 rounded-xl bg-gradient-to-r from-rose-500 to-rose-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-rose-500/20 transition-all hover:from-rose-400 hover:to-rose-500"
                >
                  Stop Webcam
                </button>
              )}
            </div>

            {error && (
              <p className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 text-center">
                {error}
              </p>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
