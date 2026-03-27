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
        <div className="flex-1 flex flex-col gap-2 p-3">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Deal Hole Cards</h1>
                <p className="text-[var(--color-text-secondary)] text-[10px]">Bot is at seat {botPosition + 1}</p>
            </div>

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

            <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                    <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold">Table</p>
                </div>
                <div className="flex flex-col gap-1.5">
                    {players.map((player, idx) => (
                        <div
                            key={idx}
                            className={`rounded-lg border px-3 py-1.5 ${player.is_bot ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border-color)] bg-slate-900/30'}`}
                        >
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-semibold text-[var(--color-text-primary)]">
                                    {getPlayerName(player, players.length)}
                                </span>
                                <span className="text-[10px] text-[var(--color-text-secondary)]">
                                    Seat {player.position + 1}
                                </span>
                            </div>
                            <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">
                                Stack: <span className="text-[var(--color-text-primary)] font-semibold">{player.stack}</span>
                            </div>
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
