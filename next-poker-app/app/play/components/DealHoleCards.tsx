import React from 'react';

type Card = { rank: string; suit: string };
const suitSym = (s: string) => ({ s: '♠', h: '♥', d: '♦', c: '♣' }[s] ?? s);

type DealHoleCardsProps = {
    botPosition: number;
    holeCards: Card[];
    canUndo: boolean;
    onUndo: () => void;
    children?: React.ReactNode; // For injecting CardSelector
};

export default function DealHoleCards({ botPosition, holeCards, canUndo, onUndo, children }: DealHoleCardsProps) {
    return (
        <div className="flex-1 flex flex-col gap-6 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-2xl font-bold text-[var(--color-text-primary)] mb-1">Deal Hole Cards</h1>
                <p className="text-[var(--color-text-secondary)] text-xs">Bot is at seat {botPosition}</p>
            </div>
            <div className="flex justify-center gap-3">
                {[0, 1].map(i => holeCards[i] ? (
                    <div key={i} className={`card-mini suit-${holeCards[i].suit}`}>
                        <span className="card-rank">{holeCards[i].rank}</span>
                        <span className="card-suit">{suitSym(holeCards[i].suit)}</span>
                    </div>
                ) : (
                    <div key={i} className="card-mini card-placeholder">
                        <span className="text-lg">?</span>
                    </div>
                ))}
            </div>
            {children}
            <div className="flex justify-center">
                {canUndo && (
                    <button onClick={onUndo} className="px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all">
                        ↩ Undo
                    </button>
                )}
            </div>
        </div>
    );
}
