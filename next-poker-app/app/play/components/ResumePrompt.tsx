import React from 'react';

type SessionInfo = {
    tableSize: number;
    smallBlind: number;
    bigBlind: number;
    buyIn: number;
    phase: string;
    sessionProfit: number;
    botStack: number;
};

type ResumePromptProps = {
    sessionInfo: SessionInfo;
    onResume: () => void;
    onStartFresh: () => void;
    isLoading: boolean;
};

export default function ResumePrompt({ sessionInfo, onResume, onStartFresh, isLoading }: ResumePromptProps) {
    if (isLoading) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-6 p-6 min-h-[100dvh]">
                <div className="animate-pulse text-[var(--color-text-secondary)] text-sm uppercase tracking-[0.2em] font-semibold">
                    Checking for saved session…
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-10 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-3 flex items-center justify-center gap-3">
                    <span className="text-5xl text-[var(--color-accent)]">♠</span>
                    PokerBot
                </h1>
                <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.2em] uppercase font-semibold">Saved Session Found</p>
            </div>

            <div className="w-full max-w-sm animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                <div className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-6 shadow-[0_4px_24px_rgba(0,0,0,0.2)] flex flex-col gap-4">
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Table</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{sessionInfo.tableSize}-Max</span>
                    </div>
                    <div className="h-[1px] w-full bg-[var(--color-border-color)]" />
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Blinds</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{sessionInfo.smallBlind}/{sessionInfo.bigBlind}</span>
                    </div>
                    <div className="h-[1px] w-full bg-[var(--color-border-color)]" />
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Bot Stack</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{sessionInfo.botStack}</span>
                    </div>
                    <div className="h-[1px] w-full bg-[var(--color-border-color)]" />
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Session Profit</span>
                        <span className={`text-sm font-bold ${sessionInfo.sessionProfit >= 0 ? 'text-[var(--color-accent)]' : 'text-[var(--color-danger)]'}`}>
                            {sessionInfo.sessionProfit >= 0 ? '+' : ''}{sessionInfo.sessionProfit}
                        </span>
                    </div>
                </div>
            </div>

            <div className="w-full max-w-sm flex flex-col gap-3 animate-slide-up" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
                <button onClick={onResume}
                    className="w-full py-5 px-6 rounded-2xl font-bold text-lg tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-3">
                    Resume Game <span className="text-xl">→</span>
                </button>
                <button onClick={onStartFresh}
                    className="w-full py-4 px-6 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-border-color)] bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] active:scale-95 flex items-center justify-center gap-2">
                    Start Fresh
                </button>
            </div>
        </div>
    );
}
