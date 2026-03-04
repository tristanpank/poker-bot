import React from 'react';

type SetupDetailsProps = {
    hasSession: boolean;
    sessionStacks: number[];
    sessionProfit: number;
    smallBlind: number;
    setSmallBlind: (amt: number) => void;
    bigBlind: number;
    setBigBlind: (amt: number) => void;
    buyIn: number;
    setBuyIn: (amt: number) => void;
    onBack: () => void;
    onStart: () => void;
    onEnd: () => void;
};

export default function SetupDetails({
    hasSession, sessionStacks, sessionProfit, smallBlind, setSmallBlind, bigBlind, setBigBlind, buyIn, setBuyIn, onBack, onStart, onEnd
}: SetupDetailsProps) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-2">
                    {hasSession ? 'Between Hands' : 'Game Details'}
                </h1>
                {hasSession && (
                    <div className="mt-2 flex flex-col gap-1">
                        <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Bot Stack</span>
                        <span className="text-3xl font-bold text-[var(--color-text-primary)]">{sessionStacks[0]}</span>
                        <span className={`text-sm font-semibold ${sessionProfit >= 0 ? 'text-[var(--color-accent)]' : 'text-[var(--color-danger)]'}`}>
                            {sessionProfit >= 0 ? '+' : ''}{sessionProfit} session
                        </span>
                    </div>
                )}
                {!hasSession && <p className="text-[var(--color-text-secondary)] text-xs tracking-[0.2em] uppercase font-semibold">Configure Your Session</p>}
            </div>

            {!hasSession && (
                <div className="w-full max-w-sm flex flex-col gap-5 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                    <div className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] flex flex-col gap-5">
                        <div className="flex justify-between items-center">
                            <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Small Blind</label>
                            <input type="number" value={smallBlind} onChange={e => setSmallBlind(Number(e.target.value) || 1)}
                                className="w-20 text-center font-bold bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-white text-sm" />
                        </div>
                        <div className="h-[1px] w-full bg-[var(--color-border-color)]" />
                        <div className="flex justify-between items-center">
                            <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Big Blind</label>
                            <input type="number" value={bigBlind} onChange={e => setBigBlind(Number(e.target.value) || 2)}
                                className="w-20 text-center font-bold bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-white text-sm" />
                        </div>
                        <div className="h-[1px] w-full bg-[var(--color-border-color)]" />
                        <div className="flex justify-between items-center">
                            <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Buy In (chips)</label>
                            <input type="number" value={buyIn} onChange={e => setBuyIn(Number(e.target.value) || 200)}
                                className="w-24 text-center font-bold bg-slate-800 border border-slate-600 rounded-lg px-2 py-2 text-white text-sm" />
                        </div>
                    </div>
                </div>
            )}

            <div className="w-full max-w-sm flex gap-3 mt-4 animate-slide-up" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
                {!hasSession && (
                    <button onClick={onBack}
                        className="flex-1 py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-border-color)] bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] active:scale-95 flex items-center justify-center gap-2">
                        <span className="text-lg">←</span> Back
                    </button>
                )}
                <button onClick={onStart}
                    className="flex-[2] py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-2">
                    {hasSession ? 'New Hand' : 'Start Session'} <span className="text-lg">→</span>
                </button>
                {hasSession && (
                    <button onClick={onEnd}
                        className="flex-1 py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-danger)]/30 bg-red-500/10 text-[var(--color-danger)] hover:bg-[var(--color-danger)] hover:text-white active:scale-95 flex items-center justify-center">
                        End
                    </button>
                )}
            </div>
        </div>
    );
}
