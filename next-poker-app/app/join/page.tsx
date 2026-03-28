'use client';

import { useCallback, useState } from 'react';
import { useEffect } from 'react';

import { useCvWebRtcStream } from '../lib/useCvWebRtcStream';

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const JOIN_PAGE_STATE_KEY = 'poker.join.page.state';
const SESSION_STATUS_POLL_INTERVAL_MS = 3000;

type JoinState = 'idle' | 'joined' | 'streaming';
type PersistedJoinPageState = {
  code: string;
  playerPosition: number;
  sessionId: string | null;
  cvSessionId: string | null;
  joinState: JoinState;
};

type WebcamSessionStatus = {
  sessionActive: boolean;
  session_id?: string;
  opponents: Record<
    string,
    {
      connected: boolean;
      cv_session_id: string;
    }
  >;
};

export default function JoinPage() {
  const [code, setCode] = useState('');
  const [playerPosition, setPlayerPosition] = useState(1);
  const [joinState, setJoinState] = useState<JoinState>('idle');
  const [joinError, setJoinError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [cvSessionId, setCvSessionId] = useState<string | null>(null);
  const [isJoining, setIsJoining] = useState(false);
  const [isStartingWebcam, setIsStartingWebcam] = useState(false);
  const [isStoppingWebcam, setIsStoppingWebcam] = useState(false);
  const [availableSeats, setAvailableSeats] = useState<boolean[]>([true, true, true, true, true]);
  const [isLoadingSeatStatus, setIsLoadingSeatStatus] = useState(false);

  const {
    videoRef,
    isStreaming,
    error: streamError,
    captureInfo,
    phase,
    startStream,
    stopStream,
  } = useCvWebRtcStream({
    backendBaseUrl: BACKEND,
  });

  const error = joinError ?? streamError;
  const trimmedCode = code.trim().toUpperCase();
  const hasCompleteCode = trimmedCode.length === 6;
  const isSeatTaken = useCallback((position: number) => !availableSeats[position - 1], [availableSeats]);
  const hasAnyAvailableSeat = availableSeats.some(Boolean);

  const notifyDisconnect = useCallback(
    (targetSessionId: string, targetPlayerPosition: number, keepalive = false) => {
      const body = JSON.stringify({
        session_id: targetSessionId,
        player_position: targetPlayerPosition,
      });

      if (keepalive && typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
        const payload = new Blob([body], { type: 'application/json' });
        navigator.sendBeacon(`${BACKEND}/session/webcam/disconnect`, payload);
        return;
      }

      void fetch(`${BACKEND}/session/webcam/disconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        keepalive,
      }).catch(() => undefined);
    },
    [],
  );

  const resetToIdle = useCallback(
    async (message: string) => {
      await stopStream();
      setJoinError(message);
      setSessionId(null);
      setCvSessionId(null);
      setJoinState('idle');
      if (typeof window !== 'undefined') {
        window.sessionStorage.removeItem(JOIN_PAGE_STATE_KEY);
      }
    },
    [stopStream],
  );

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      const raw = window.sessionStorage.getItem(JOIN_PAGE_STATE_KEY);
      if (!raw) {
        return;
      }

      const saved = JSON.parse(raw) as PersistedJoinPageState;
      setCode(saved.code ?? '');
      setPlayerPosition(saved.playerPosition ?? 1);
      setSessionId(saved.sessionId ?? null);
      setCvSessionId(saved.cvSessionId ?? null);

      if (saved.sessionId && saved.cvSessionId && saved.joinState !== 'idle') {
        notifyDisconnect(saved.sessionId, saved.playerPosition ?? 1);
        // A browser refresh destroys the active camera/WebRTC objects, so we
        // restore the page into the joined state and let the user restart the
        // webcam without re-joining the table position.
        setJoinState('joined');
      }
    } catch {
      window.sessionStorage.removeItem(JOIN_PAGE_STATE_KEY);
    }
  }, [notifyDisconnect]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    if (!sessionId || !cvSessionId || joinState === 'idle') {
      window.sessionStorage.removeItem(JOIN_PAGE_STATE_KEY);
      return;
    }

    const nextState: PersistedJoinPageState = {
      code,
      playerPosition,
      sessionId,
      cvSessionId,
      joinState: isStreaming ? 'streaming' : 'joined',
    };
    window.sessionStorage.setItem(JOIN_PAGE_STATE_KEY, JSON.stringify(nextState));
  }, [code, cvSessionId, isStreaming, joinState, playerPosition, sessionId]);

  useEffect(() => {
    if (!sessionId || joinState === 'idle') {
      return;
    }

    const pollStatus = async () => {
      try {
        const res = await fetch(`${BACKEND}/session/webcam/status/${sessionId}`);
        if (!res.ok) {
          return;
        }

        const data = (await res.json()) as WebcamSessionStatus;
        if (!data.sessionActive) {
          await resetToIdle('The host ended this game. Rejoin with the new table code.');
        }
      } catch {
        // Ignore transient polling failures and keep the current UI state.
      }
    };

    void pollStatus();
    const pollId = window.setInterval(() => {
      void pollStatus();
    }, SESSION_STATUS_POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(pollId);
    };
  }, [joinState, resetToIdle, sessionId]);

  useEffect(() => {
    if (!sessionId || joinState === 'idle') {
      return;
    }

    const handlePageHide = () => {
      notifyDisconnect(sessionId, playerPosition, true);
    };

    window.addEventListener('pagehide', handlePageHide);
    window.addEventListener('beforeunload', handlePageHide);

    return () => {
      window.removeEventListener('pagehide', handlePageHide);
      window.removeEventListener('beforeunload', handlePageHide);
    };
  }, [joinState, notifyDisconnect, playerPosition, sessionId]);

  useEffect(() => {
    if (joinState !== 'idle' || !hasCompleteCode) {
      setAvailableSeats([true, true, true, true, true]);
      setIsLoadingSeatStatus(false);
      return;
    }

    let cancelled = false;

    const loadSeatStatus = async () => {
      setIsLoadingSeatStatus(true);
      try {
        const res = await fetch(`${BACKEND}/session/webcam/status-by-code/${trimmedCode}`);
        if (!res.ok) {
          if (!cancelled) {
            setAvailableSeats([true, true, true, true, true]);
          }
          return;
        }

        const data = (await res.json()) as WebcamSessionStatus;
        const nextAvailableSeats = [true, true, true, true, true];
        for (const [position, opponent] of Object.entries(data.opponents)) {
          const seatIndex = Number(position) - 1;
          if (
            Number.isInteger(seatIndex) &&
            seatIndex >= 0 &&
            seatIndex < nextAvailableSeats.length &&
            opponent.connected
          ) {
            nextAvailableSeats[seatIndex] = false;
          }
        }

        if (!cancelled) {
          setAvailableSeats(nextAvailableSeats);
        }
      } catch {
        if (!cancelled) {
          setAvailableSeats([true, true, true, true, true]);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSeatStatus(false);
        }
      }
    };

    void loadSeatStatus();
    const pollId = window.setInterval(() => {
      void loadSeatStatus();
    }, SESSION_STATUS_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
    };
  }, [hasCompleteCode, joinState, trimmedCode]);

  useEffect(() => {
    if (joinState !== 'idle' || !isSeatTaken(playerPosition)) {
      return;
    }

    const nextAvailablePosition = availableSeats.findIndex(Boolean);
    if (nextAvailablePosition >= 0) {
      setPlayerPosition(nextAvailablePosition + 1);
    }
  }, [availableSeats, isSeatTaken, joinState, playerPosition]);

  const handleJoin = useCallback(async () => {
    setJoinError(null);
    if (!trimmedCode) {
      setJoinError('Please enter a join code.');
      return;
    }

    if (isSeatTaken(playerPosition)) {
      setJoinError(`Player ${playerPosition} is already taken. Choose an available seat.`);
      return;
    }

    try {
      setIsJoining(true);
      const res = await fetch(`${BACKEND}/session/webcam/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: trimmedCode, player_position: playerPosition }),
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
      setJoinError(err instanceof Error ? err.message : 'Failed to join session.');
    } finally {
      setIsJoining(false);
    }
  }, [isSeatTaken, playerPosition, trimmedCode]);

  const startWebcam = useCallback(async () => {
    if (!cvSessionId || !sessionId) return;
    setJoinError(null);

    try {
      setIsStartingWebcam(true);
      if (joinState === 'joined') {
        void fetch(`${BACKEND}/session/webcam/reconnect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, player_position: playerPosition }),
        }).catch(() => undefined);
      }

      const started = await startStream({ sessionId: cvSessionId });
      if (started) {
        setJoinState('streaming');
      }
    } finally {
      setIsStartingWebcam(false);
    }
  }, [cvSessionId, joinState, playerPosition, sessionId, startStream]);

  const stopWebcam = useCallback(async () => {
    try {
      setIsStoppingWebcam(true);
      await stopStream();

      if (sessionId) {
        notifyDisconnect(sessionId, playerPosition);
      }

      setJoinState('joined');
    } finally {
      setIsStoppingWebcam(false);
    }
  }, [notifyDisconnect, playerPosition, sessionId, stopStream]);

  const leaveSession = useCallback(async () => {
    await stopWebcam();
    setJoinError(null);
    setSessionId(null);
    setCvSessionId(null);
    setJoinState('idle');
    if (typeof window !== 'undefined') {
      window.sessionStorage.removeItem(JOIN_PAGE_STATE_KEY);
    }
  }, [stopWebcam]);

  const statusLabel = isJoining
    ? 'Joining...'
    : isStartingWebcam
      ? phase === 'requesting_camera'
        ? 'Requesting camera...'
        : phase === 'tuning_camera'
          ? 'Tuning camera...'
          : phase === 'negotiating_webrtc'
            ? 'Preparing WebRTC...'
            : phase === 'waiting_for_backend'
              ? 'Contacting backend...'
              : phase === 'connecting'
                ? 'Connecting stream...'
                : 'Starting webcam...'
      : isStoppingWebcam
        ? 'Stopping webcam...'
        : isStreaming
          ? phase === 'live'
            ? 'Live'
            : 'Connecting stream...'
          : 'Ready';

  const statusClass = isJoining || isStartingWebcam || isStoppingWebcam
    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
    : isStreaming && phase !== 'live'
      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
      : isStreaming
      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
      : 'bg-slate-700/50 text-slate-400 border border-slate-600/30';

  const isBusy = isJoining || isStartingWebcam || isStoppingWebcam;

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-4 py-10 text-slate-100 flex items-start justify-center"
      style={{ fontFamily: "'Outfit', sans-serif" }}
    >
      <main className="w-full max-w-lg flex flex-col gap-6">
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
          <p className="mt-2 text-xs text-slate-500 max-w-md mx-auto">
            Capture negotiated: {captureInfo}
          </p>
        </section>

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
                onChange={(e) => {
                  setCode(e.target.value.toUpperCase());
                  setJoinError(null);
                }}
                disabled={isJoining}
                placeholder="ABC123"
                className="w-full rounded-xl border border-slate-600/50 bg-slate-800/80 px-4 py-3 text-center text-2xl font-mono font-bold tracking-[0.3em] text-white placeholder:text-slate-600 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 transition-all"
              />
            </div>

            <div>
              <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1.5 block">
                Your Position (relative to bot)
              </label>
              <div className="grid grid-cols-5 gap-2">
                {[1, 2, 3, 4, 5].map((pos) => {
                  const taken = hasCompleteCode && isSeatTaken(pos);
                  return (
                    <button
                      key={pos}
                      onClick={() => setPlayerPosition(pos)}
                      disabled={isJoining || taken}
                      className={`rounded-xl py-3 text-sm font-semibold transition-all ${
                        taken
                          ? 'bg-slate-900/80 text-slate-500 border border-slate-700/40 cursor-not-allowed opacity-60'
                          : playerPosition === pos
                            ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/25'
                            : 'bg-slate-800/80 text-slate-300 border border-slate-600/30 hover:bg-slate-700/80 hover:border-slate-500/40'
                      }`}
                    >
                      P{pos}
                    </button>
                  );
                })}
              </div>
              {hasCompleteCode && (
                <p className="mt-2 text-xs text-slate-500">
                  {isLoadingSeatStatus
                    ? 'Checking available seats...'
                    : hasAnyAvailableSeat
                      ? 'Taken seats are greyed out.'
                      : 'All seats are currently taken.'}
                </p>
              )}
            </div>

            <button
              onClick={handleJoin}
              disabled={!hasCompleteCode || isJoining || isLoadingSeatStatus || isSeatTaken(playerPosition) || !hasAnyAvailableSeat}
              className="mt-1 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-6 py-3.5 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition-all hover:from-emerald-400 hover:to-emerald-500 hover:shadow-emerald-500/30 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {isJoining ? 'Joining...' : 'Join Session'}
            </button>

            {error && (
              <p className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 text-center">
                {error}
              </p>
            )}
          </section>
        )}

        {joinState !== 'idle' && (
          <section className="rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm p-6 shadow-2xl shadow-black/20 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400 font-medium">
                  Connected as <span className="text-emerald-400 font-semibold">Player {playerPosition}</span>
                </p>
                <p className="text-xs text-slate-500 mt-0.5">Session: {sessionId?.slice(0, 8)}...</p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${statusClass}`}
              >
                {statusLabel}
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
              {!isStreaming && (
                <button
                  onClick={startWebcam}
                  disabled={isBusy}
                  className="flex-1 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition-all hover:from-emerald-400 hover:to-emerald-500"
                >
                  {isStartingWebcam ? 'Starting Webcam...' : 'Start Webcam'}
                </button>
              )}
              {isStreaming && (
                <button
                  onClick={stopWebcam}
                  disabled={isBusy}
                  className="flex-1 rounded-xl bg-gradient-to-r from-rose-500 to-rose-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-rose-500/20 transition-all hover:from-rose-400 hover:to-rose-500"
                >
                  {isStoppingWebcam ? 'Stopping Webcam...' : 'Stop Webcam'}
                </button>
              )}
            </div>

            {!isStreaming && (
              <button
                onClick={leaveSession}
                disabled={isBusy}
                className="rounded-xl border border-slate-600/40 bg-slate-800/60 px-4 py-3 text-sm font-semibold text-slate-200 transition-all hover:bg-slate-700/70 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Leave Session
              </button>
            )}

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
