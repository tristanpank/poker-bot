'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { FULL_RING_SEAT_COUNT, getSeatLabel } from '../../lib/tablePositions';

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const POLL_INTERVAL_MS = 3000;

type OpponentStatus = {
  connected: boolean;
  cv_session_id: string;
  player_name?: string;
  metrics?: {
    bluffLevel: string;
    bluffRisk: number;
    emotion: string;
    stress: number;
    pulseBpm: number | null;
  };
};

type WebcamStatusData = {
  code?: string | null;
  tableSize?: number | null;
  botPosition?: number | null;
  opponents: Record<string, OpponentStatus>;
};

type WebcamStatusProps = {
  sessionId: string | null;
  tableSize: number;
  botSeat: number | null;
};

export default function WebcamStatus({ sessionId, tableSize, botSeat }: WebcamStatusProps) {
  const [code, setCode] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusData, setStatusData] = useState<WebcamStatusData>({ opponents: {} });
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const poll = async () => {
      try {
        const res = await fetch(`${BACKEND}/session/webcam/status/${sessionId}`);
        if (res.ok) {
          const data: WebcamStatusData = await res.json();
          setStatusData(data);
        }
      } catch {
        // Silently ignore polling errors.
      }
    };

    void poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [sessionId]);

  const generateCode = useCallback(async () => {
    if (!sessionId || botSeat === null) return;
    setIsGenerating(true);
    setError(null);

    try {
      const res = await fetch(`${BACKEND}/session/webcam/generate-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!res.ok) throw new Error('Failed to generate code');

      const data = await res.json();
      setCode(data.code);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate code');
    } finally {
      setIsGenerating(false);
    }
  }, [botSeat, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      setCode(null);
      setStatusData({ opponents: {} });
      setError(null);
      return;
    }

    if (botSeat !== null) {
      void generateCode();
    }
  }, [botSeat, generateCode, sessionId]);

  const copyCode = useCallback(async () => {
    const nextCode = code ?? statusData.code ?? null;
    if (!nextCode) return;
    try {
      await navigator.clipboard.writeText(nextCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Ignore clipboard failures.
    }
  }, [code, statusData.code]);

  const resolvedTableSize = tableSize || FULL_RING_SEAT_COUNT;
  const resolvedBotSeat = botSeat ?? (typeof statusData.botPosition === 'number' ? statusData.botPosition : null);
  const displayCode = code ?? statusData.code ?? null;
  const opponentSeats = Array.from({ length: resolvedTableSize }, (_, seat) => seat).filter(
    (seat) => seat !== resolvedBotSeat,
  );
  const connectedCount = Object.values(statusData.opponents).filter((opponent) => opponent.connected).length;

  return (
    <section className="rounded-2xl border border-slate-700/30 bg-slate-900/40 p-3 backdrop-blur-sm flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Opponent Webcams
        </h3>
        {displayCode && (
          <span className="text-[10px] text-slate-600">
            {connectedCount}/{opponentSeats.length} seated
          </span>
        )}
      </div>

      {!displayCode ? (
        <button
          onClick={generateCode}
          disabled={isGenerating || !sessionId || botSeat === null}
          className="w-full rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-blue-500/15 transition-all hover:bg-blue-400 disabled:opacity-40"
        >
          {isGenerating ? 'Loading Code...' : botSeat === null ? 'Choose Bot Seat First' : 'Generate Code'}
        </button>
      ) : (
        <>
          {resolvedBotSeat !== null && (
            <p className="text-[10px] text-slate-500">
              Bot seat: <span className="text-slate-300">{getSeatLabel(resolvedBotSeat)}</span>
            </p>
          )}

          <div className="flex items-center gap-1.5">
            <div className="flex-1 rounded-lg bg-slate-800/80 border border-slate-600/30 py-2 text-center">
              <span className="text-lg font-mono font-bold tracking-[0.25em] text-white">
                {displayCode}
              </span>
            </div>
            <button
              onClick={copyCode}
              className="rounded-lg bg-slate-800/80 border border-slate-600/30 px-3 py-2 text-xs text-slate-300 hover:text-white"
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>

          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
            {opponentSeats.map((seat) => {
              const opponent = statusData.opponents[String(seat)];
              const isConnected = opponent?.connected ?? false;
              const displayName = opponent?.player_name ?? `Player ${seat + 1}`;

              return (
                <div
                  key={seat}
                  className={`rounded-lg py-1.5 px-0.5 text-center text-[10px] font-semibold transition-all ${
                    isConnected
                      ? 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400'
                      : 'bg-slate-800/40 border border-slate-700/20 text-slate-500'
                  }`}
                >
                  <div className="flex min-h-[58px] flex-col items-center justify-center gap-0.5">
                    <div
                      className={`h-1.5 w-1.5 rounded-full ${
                        isConnected ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : 'bg-slate-600'
                      }`}
                    />
                    <span className="text-[11px] uppercase tracking-wide opacity-90">{getSeatLabel(seat)}</span>
                    <span className="max-w-full truncate px-1 text-[9px] text-slate-200">{displayName}</span>
                    {isConnected && opponent?.metrics && (
                      <div className="mt-1 flex flex-col items-center text-[9px] leading-tight opacity-90">
                        <span className="text-white font-bold uppercase tracking-wider">{opponent.metrics.bluffLevel}</span>
                        <span className={opponent.metrics.bluffRisk > 60 ? 'text-rose-400' : 'text-emerald-300'}>
                          {opponent.metrics.bluffRisk}% Risk
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
      {!displayCode && botSeat === null && (
        <p className="text-[10px] text-slate-500 text-center">
          Pick the bot&apos;s real table seat before sharing the join code.
        </p>
      )}
      {error && (
        <p className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300 text-center">
          {error}
        </p>
      )}
    </section>
  );
}
