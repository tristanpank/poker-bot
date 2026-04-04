import React from 'react';

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const;
const SUITS = [
    { key: 's', sym: '\u2660', label: 'Spades' },
    { key: 'h', sym: '\u2665', label: 'Hearts' },
    { key: 'd', sym: '\u2666', label: 'Diamonds' },
    { key: 'c', sym: '\u2663', label: 'Clubs' },
] as const;

type CardSelectorProps = {
    pickingFor: 'hole' | 'community' | 'showdown' | null;
    holeCardsCount: number;
    communityCardsCount: number;
    usedCards: Set<string>;
    pendingRank: string | null;
    setPendingRank: (rank: string | null) => void;
    onSelectCard: (rank: string, suit: string) => void;
    onCancel: () => void;
    onConfirmCommunity?: () => void;
};

export default function CardSelector({
    pickingFor, holeCardsCount, communityCardsCount, usedCards,
    pendingRank, setPendingRank, onSelectCard, onCancel, onConfirmCommunity
}: CardSelectorProps) {
    if (!pickingFor) return null;

    return (
        <div className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-3 animate-fade-in no-select">
            <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-2">
                {pickingFor === 'hole'
                    ? `Select hole card ${holeCardsCount + 1} of 2`
                    : pickingFor === 'community'
                        ? `Select community card (${communityCardsCount + 1}${communityCardsCount < 3 ? '/3 flop' : communityCardsCount === 3 ? ' turn' : ' river'})`
                        : `Select showdown card ${holeCardsCount + 1} of 2`
                }
            </p>
            {pendingRank ? (
                <div className="flex flex-col gap-2 items-center">
                    <p className="text-xs text-[var(--color-text-primary)] font-semibold">Pick suit for {pendingRank}</p>
                    <div className="flex gap-2">
                        {SUITS.map(s => {
                            const isUsed = usedCards.has(`${pendingRank}${s.key}`);
                            return (
                                <button key={s.key} disabled={isUsed}
                                    onClick={() => onSelectCard(pendingRank, s.key)}
                                    className={`w-14 h-16 rounded-lg border-2 flex flex-col items-center justify-center gap-0.5 transition-all duration-200 active:scale-90
                    ${isUsed ? 'opacity-20 cursor-not-allowed border-slate-700 bg-slate-900'
                                            : s.key === 'h' || s.key === 'd'
                                                ? 'border-red-500/40 bg-red-500/10 text-red-400 hover:bg-red-500/20'
                                                : 'border-slate-500/40 bg-slate-800 text-slate-200 hover:bg-slate-700'}`}>
                                    <span className="text-xl">{s.sym}</span>
                                    <span className="text-[8px] uppercase tracking-wider">{s.label}</span>
                                </button>
                            );
                        })}
                    </div>
                    <button onClick={() => setPendingRank(null)} className="text-[10px] text-[var(--color-text-secondary)] hover:text-white mt-0.5">&larr; Back to ranks</button>
                </div>
            ) : pickingFor === 'community' && communityCardsCount >= 5 ? (
                <div className="py-6 text-center text-xs font-semibold text-emerald-400">
                    Maximum 5 community cards reached. Click Confirm to continue.
                </div>
            ) : (
                <div className="grid grid-cols-7 gap-1">
                    {RANKS.map(r => {
                        const allUsed = SUITS.every(s => usedCards.has(`${r}${s.key}`));
                        return (
                            <button key={r} disabled={allUsed}
                                onClick={() => setPendingRank(r)}
                                className={`py-2 rounded-lg font-bold text-xs transition-all duration-150 active:scale-95 
                  ${allUsed ? 'opacity-20 cursor-not-allowed bg-slate-900 text-slate-600'
                                        : 'bg-slate-800 text-white border border-slate-600 hover:bg-slate-700'}`}>
                                {r}
                            </button>
                        );
                    })}
                </div>
            )}
            <div className="flex justify-between items-center mt-2 border-t border-slate-700/50 pt-2">
                <button onClick={onCancel}
                    className="text-[10px] text-[var(--color-text-secondary)] hover:text-white px-2 py-1">
                    Cancel
                </button>
                {pickingFor === 'community' && onConfirmCommunity && (
                    <button
                        onClick={onConfirmCommunity}
                        disabled={
                            !(communityCardsCount === 3) &&
                            !(communityCardsCount === 4) &&
                            !(communityCardsCount === 5)
                        }
                        className="text-[10px] font-bold px-3 py-1.5 rounded-lg bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 disabled:opacity-30 transition-all">
                        Confirm
                    </button>
                )}
            </div>
        </div>
    );
}
