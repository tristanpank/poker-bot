import React from 'react';

type Card = { rank: string; suit: string };
type OpponentReveal = {
    playerIndex: number;
    position: number;
    cards: Card[];
    mucked: boolean;
};
type ResolveResult = {
    result: 'won' | 'lost' | 'push';
    amount: number;
    delta: number;
};

type HandResultProps = {
    opponents: OpponentReveal[];
    currentOpponent: OpponentReveal | null;
    usedCards: Set<string>;
    pendingRank: string | null;
    setPendingRank: (rank: string | null) => void;
    onSelectCard: (rank: string, suit: string) => void;
    onMuckCurrent: () => void;
    onClearCurrent: () => void;
    canResolve: boolean;
    isResolving: boolean;
    resolveError: string | null;
    resolveResult: ResolveResult | null;
    onResolve: () => void;
    onContinue: () => void;
    onBack: () => void;
};

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const;
const SUITS = [
    { key: 's', sym: 'S' },
    { key: 'h', sym: 'H' },
    { key: 'd', sym: 'D' },
    { key: 'c', sym: 'C' },
] as const;

const suitSym = (s: string) => ({ s: 'S', h: 'H', d: 'D', c: 'C' }[s] ?? s);

export default function HandResult({
    opponents,
    currentOpponent,
    usedCards,
    pendingRank,
    setPendingRank,
    onSelectCard,
    onMuckCurrent,
    onClearCurrent,
    canResolve,
    isResolving,
    resolveError,
    resolveResult,
    onResolve,
    onContinue,
    onBack,
}: HandResultProps) {
    return (
        <div className="flex-1 flex flex-col p-6 min-h-[100dvh] gap-5">
            <div className="text-center">
                <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">Showdown</h1>
                <p className="text-[var(--color-text-secondary)] text-xs">Enter opponent hole cards in table order, or mark muck.</p>
            </div>

            <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                <p className="text-xs uppercase tracking-wider text-[var(--color-text-secondary)] mb-3">Opponents</p>
                <div className="flex flex-col gap-2">
                    {opponents.length === 0 && (
                        <p className="text-sm text-[var(--color-text-secondary)]">No active opponents. Resolve to collect the pot.</p>
                    )}
                    {opponents.map((opp) => (
                        <div key={opp.playerIndex} className={`rounded-xl border px-3 py-2 flex items-center justify-between ${currentOpponent?.playerIndex === opp.playerIndex ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border-color)] bg-slate-900/30'}`}>
                            <span className="text-sm font-semibold text-[var(--color-text-primary)]">Seat {opp.position}</span>
                            {opp.mucked ? (
                                <span className="text-xs font-bold uppercase text-amber-300">Mucked</span>
                            ) : (
                                <span className="text-xs text-[var(--color-text-secondary)]">
                                    {opp.cards.length === 0 ? 'Pending' : opp.cards.map((c) => `${c.rank}${suitSym(c.suit)}`).join(' ')}
                                </span>
                            )}
                        </div>
                    ))}
                </div>
            </section>

            {currentOpponent && !resolveResult && (
                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                    <p className="text-xs uppercase tracking-wider text-[var(--color-text-secondary)] mb-3">
                        Enter cards for Seat {currentOpponent.position}
                    </p>
                    {pendingRank ? (
                        <div className="flex flex-col gap-3">
                            <div className="grid grid-cols-4 gap-2">
                                {SUITS.map((s) => {
                                    const isUsed = usedCards.has(`${pendingRank}${s.key}`);
                                    return (
                                        <button
                                            key={s.key}
                                            disabled={isUsed}
                                            onClick={() => onSelectCard(pendingRank, s.key)}
                                            className="py-3 rounded-xl border border-slate-600 bg-slate-800 text-sm font-bold text-white hover:bg-slate-700 disabled:opacity-25 disabled:cursor-not-allowed"
                                        >
                                            {s.sym}
                                        </button>
                                    );
                                })}
                            </div>
                            <button
                                onClick={() => setPendingRank(null)}
                                className="text-xs text-[var(--color-text-secondary)] hover:text-white"
                            >
                                Back To Ranks
                            </button>
                        </div>
                    ) : (
                        <div className="grid grid-cols-7 gap-2">
                            {RANKS.map((rank) => {
                                const allUsed = SUITS.every((s) => usedCards.has(`${rank}${s.key}`));
                                return (
                                    <button
                                        key={rank}
                                        disabled={allUsed}
                                        onClick={() => setPendingRank(rank)}
                                        className="py-2 rounded-lg border border-slate-600 bg-slate-800 text-xs font-bold text-white hover:bg-slate-700 disabled:opacity-25 disabled:cursor-not-allowed"
                                    >
                                        {rank}
                                    </button>
                                );
                            })}
                        </div>
                    )}

                    <div className="mt-3 flex gap-2">
                        <button
                            onClick={onMuckCurrent}
                            className="flex-1 py-2 rounded-xl text-xs font-bold uppercase border border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20"
                        >
                            Muck
                        </button>
                        <button
                            onClick={onClearCurrent}
                            className="flex-1 py-2 rounded-xl text-xs font-bold uppercase border border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700"
                        >
                            Clear
                        </button>
                    </div>
                </section>
            )}

            {resolveError && (
                <p className="text-sm text-red-300">{resolveError}</p>
            )}

            {resolveResult && (
                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                    <p className="text-xs uppercase tracking-wider text-[var(--color-text-secondary)] mb-2">Result</p>
                    <p className="text-2xl font-bold text-[var(--color-text-primary)]">
                        {resolveResult.result.toUpperCase()} {resolveResult.amount}
                    </p>
                    <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                        Delta: {resolveResult.delta >= 0 ? '+' : ''}{resolveResult.delta}
                    </p>
                </section>
            )}

            <div className="mt-auto flex gap-3">
                <button
                    onClick={onBack}
                    className="flex-1 py-3 rounded-xl border border-slate-600 bg-slate-800 text-slate-200 text-sm font-semibold hover:bg-slate-700"
                >
                    Back To Hand
                </button>
                {!resolveResult ? (
                    <button
                        onClick={onResolve}
                        disabled={!canResolve || isResolving}
                        className="flex-1 py-3 rounded-xl bg-[var(--color-accent)] text-slate-950 text-sm font-bold hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                        {isResolving ? 'Resolving...' : 'Resolve Hand'}
                    </button>
                ) : (
                    <button
                        onClick={onContinue}
                        className="flex-1 py-3 rounded-xl bg-[var(--color-accent)] text-slate-950 text-sm font-bold hover:bg-emerald-400"
                    >
                        Next Hand
                    </button>
                )}
            </div>
        </div>
    );
}
