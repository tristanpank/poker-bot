'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const POLL_INTERVAL_MS = 3000;

type OpponentStatus = {
  connected: boolean;
  cv_session_id: string;
  metrics?: {
    bluffLevel: string;
    bluffRisk: number;
    emotion: string;
    stress: number;
    pulseBpm: number | null;
  };
};

type WebcamStatusData = {
  opponents: Record<string, OpponentStatus>;
};

type WebcamStatusProps = {
  sessionId: string | null;
  tableSize: number;
};

export default function WebcamStatus({ sessionId, tableSize }: WebcamStatusProps) {
  const [code, setCode] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusData, setStatusData] = useState<WebcamStatusData>({ opponents: {} });
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll for opponent connection status
  useEffect(() => {
    if (!sessionId || !code) return;

    const poll = async () => {
      try {
        const res = await fetch(`${BACKEND}/session/webcam/status/${sessionId}`);
        if (res.ok) {
          const data: WebcamStatusData = await res.json();
          setStatusData(data);
        }
      } catch {
        // Silently ignore polling errors
      }
    };

    void poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [sessionId, code]);

  const generateCode = useCallback(async () => {
    if (!sessionId) return;
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
  }, [sessionId]);

  const copyCode = useCallback(async () => {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: select text
    }
  }, [code]);

  // Positions 1..tableSize-1 (bot is position 0)
  const opponentPositions = Array.from({ length: tableSize - 1 }, (_, i) => i + 1);

  const connectedCount = Object.values(statusData.opponents).filter((o) => o.connected).length;

  return (
    <section className="rounded-2xl border border-slate-700/30 bg-slate-900/40 backdrop-blur-sm p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Opponent Webcams
        </h3>
        {code && (
          <span className="text-[10px] text-slate-600">
            {connectedCount}/{tableSize - 1} connected
          </span>
        )}
      </div>

      {!code ? (
        <button
          onClick={generateCode}
          disabled={isGenerating || !sessionId}
          className="w-full rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-blue-500/15 transition-all hover:bg-blue-400 disabled:opacity-40"
        >
          {isGenerating ? 'Generating…' : 'Generate Code'}
        </button>
      ) : (
        <>
          {/* Code display */}
          <div className="flex items-center gap-1.5">
            <div className="flex-1 rounded-lg bg-slate-800/80 border border-slate-600/30 py-2 text-center">
              <span className="text-lg font-mono font-bold tracking-[0.25em] text-white">
                {code}
              </span>
            </div>
            <button
              onClick={copyCode}
              className="rounded-lg bg-slate-800/80 border border-slate-600/30 px-3 py-2 text-xs text-slate-300 hover:text-white"
            >
              {copied ? '✓' : '⎘'}
            </button>
          </div>

          {/* Player connection grid - more compact */}
          <div className="grid grid-cols-5 gap-1.5">
            {opponentPositions.map((pos) => {
              const opponent = statusData.opponents[String(pos)];
              const isConnected = opponent?.connected ?? false;

              return (
                <div
                  key={pos}
                  className={`rounded-lg py-1.5 px-0.5 text-center text-[10px] font-semibold transition-all ${isConnected
                    ? 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400'
                    : 'bg-slate-800/40 border border-slate-700/20 text-slate-500'
                    }`}
                >
                  <div className="flex flex-col items-center justify-center gap-0.5 min-h-[40px]">
                    <div className="flex items-center gap-1">
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : 'bg-slate-600'
                          }`}
                      />
                      <span>P{pos}</span>
                    </div>
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
      {error && (
        <p className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300 text-center">
          {error}
        </p>
      )}
    </section>
  );
}
