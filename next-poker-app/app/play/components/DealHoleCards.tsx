import React from 'react';

type Card = { rank: string; suit: string };
type PlayerState = { position: number; stack: number; bet: number; hole_cards: Card[] | null; is_bot: boolean; is_active: boolean; has_acted: boolean };

const suitSym = (s: string) => ({ s: '\u2660', h: '\u2665', d: '\u2666', c: '\u2663' }[s] ?? s);

const seatRoleByPlayers: Record<number, string[]> = {
    2: ['BTN/SB', 'BB'],
    3: ['BTN', 'SB', 'BB'],
    4: ['BTN', 'SB', 'BB', 'UTG'],
    5: ['BTN', 'SB', 'BB', 'UTG', 'CO'],
    6: ['BTN', 'SB', 'BB', 'UTG', 'HJ', 'CO'],
};

function getSeatRole(position: number, numPlayers: number): string {
    const roles = seatRoleByPlayers[numPlayers] ?? seatRoleByPlayers[6];
    return roles[position] ?? `Seat ${position + 1}`;
}

function getPlayerName(player: PlayerState, numPlayers: number): string {
    const role = getSeatRole(player.position, numPlayers);
    if (player.is_bot) return `Bot (${role})`;
    return `Player ${player.position + 1} (${role})`;
}

type DealHoleCardsProps = {
    botPosition: number;
    holeCards: Card[];
    players: PlayerState[];
    canUndo: boolean;
    onUndo: () => void;
    children?: React.ReactNode;
};

export default function DealHoleCards({ botPosition, holeCards, players, canUndo, onUndo, children }: DealHoleCardsProps) {
    return (
        <div className="flex-1 flex flex-col gap-6 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-2xl font-bold text-[var(--color-text-primary)] mb-1">Deal Hole Cards</h1>
                <p className="text-[var(--color-text-secondary)] text-xs">Bot is at seat {botPosition + 1}</p>
            </div>

            <div className="flex justify-center gap-3">
                {[0, 1].map((i) => holeCards[i] ? (
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

            <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                <div className="flex justify-between items-center mb-3">
                    <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Table</p>
                    <p className="text-xs text-[var(--color-text-secondary)]">Select bot cards for this seat</p>
                </div>
                <div className="flex flex-col gap-2">
                    {players.map((player, idx) => (
                        <div
                            key={idx}
                            className={`rounded-xl border px-3 py-2 ${player.is_bot ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border-color)] bg-slate-900/30'}`}
                        >
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-semibold text-[var(--color-text-primary)]">
                                    {getPlayerName(player, players.length)}
                                </span>
                                <span className="text-xs text-[var(--color-text-secondary)]">
                                    Seat {player.position + 1}
                                </span>
                            </div>
                            <div className="text-xs text-[var(--color-text-secondary)] mt-1">
                                Stack: <span className="text-[var(--color-text-primary)] font-semibold">{player.stack}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {children}

            <div className="flex justify-center">
                {canUndo && (
                    <button onClick={onUndo} className="px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all">
                        Undo
                    </button>
                )}
            </div>
        </div>
    );
}
