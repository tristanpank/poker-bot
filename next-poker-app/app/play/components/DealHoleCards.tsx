import React from 'react';
import { getTablePosition } from '../../lib/tablePositions';
import TableVisual, { TableSeatVisual } from './TableVisual';

type Card = { rank: string; suit: string };
type PlayerState = { position: number; stack: number; bet: number; hole_cards: Card[] | null; is_bot: boolean; is_active: boolean; has_acted: boolean };

const suitSym = (s: string) => ({ s: '\u2660', h: '\u2665', d: '\u2666', c: '\u2663' }[s] ?? s);

type DealHoleCardsProps = {
    botPosition: number;
    holeCards: Card[];
    players: PlayerState[];
    tableSeats: TableSeatVisual[];
    tableStatus: string;
    canUndo: boolean;
    onUndo: () => void;
    children?: React.ReactNode;
};

export default function DealHoleCards({
    botPosition,
    holeCards,
    players,
    tableSeats,
    tableStatus,
    canUndo,
    onUndo,
    children,
}: DealHoleCardsProps) {
    return (
        <div className="flex-1 flex flex-col gap-2 p-3">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Deal Hole Cards</h1>
                <p className="text-[var(--color-text-secondary)] text-[10px]">
                    {players.length >= 2 ? `Bot position: ${getTablePosition(botPosition, players.length)}` : 'Waiting for opponents'}
                </p>
            </div>

            <TableVisual
                seats={tableSeats}
                center={(
                    <div className="flex flex-col items-center gap-3">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-[var(--color-text-secondary)]">Preflop</p>
                        <div className="flex items-center justify-center -space-x-2 scale-105">
                            {[0, 1, 2, 3, 4].map((i) => (
                                <div key={i} className="card-mini card-placeholder scale-75 opacity-40">
                                    <span className="text-base">?</span>
                                </div>
                            ))}
                        </div>
                        <p className="text-xs font-semibold text-[var(--color-text-primary)]">{tableStatus}</p>
                    </div>
                )}
            />

            <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-3">
                <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-2">Bot&apos;s Hand</p>
                <div className="flex justify-center gap-2">
                    {[0, 1].map((i) => holeCards[i] ? (
                        <div key={i} className={`card-mini suit-${holeCards[i].suit} scale-90`}>
                            <span className="card-rank">{holeCards[i].rank}</span>
                            <span className="card-suit">{suitSym(holeCards[i].suit)}</span>
                        </div>
                    ) : (
                        <div key={i} className="card-mini card-placeholder scale-90">
                            <span className="text-lg">?</span>
                        </div>
                    ))}
                </div>
            </section>

            <div className="flex-1 flex flex-col min-h-0">
                {children}
            </div>

            <div className="flex justify-center mt-auto py-1">
                {canUndo && (
                    <button onClick={onUndo} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-600">
                        Undo
                    </button>
                )}
            </div>
        </div>
    );
}
