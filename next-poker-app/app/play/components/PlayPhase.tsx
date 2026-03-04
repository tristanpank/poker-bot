import React from 'react';

type Card = { rank: string; suit: string };
type PlayerState = { position: number; stack: number; bet: number; hole_cards: Card[] | null; is_bot: boolean; is_active: boolean; has_acted: boolean; };
type BotResponse = { action: string; action_id: number; amount: number | null; equity: number; hand_strength_category: string; q_values: Record<string, number> | null };

const ACTION_COLORS: Record<string, string> = {
    FOLD: 'text-red-400', CALL: 'text-blue-400', RAISE_SMALL: 'text-emerald-400',
    RAISE_MEDIUM: 'text-emerald-300', RAISE_LARGE: 'text-yellow-400', ALL_IN: 'text-amber-300',
};

const suitSym = (s: string) => ({ s: '♠', h: '♥', d: '♦', c: '♣' }[s] ?? s);

function relativeLabel(pos: number, botPos: number, total: number): string {
    const diff = ((pos - botPos) % total + total) % total;
    if (diff === 0) return 'Bot';
    return `Player ${pos} — ${diff} seat${diff > 1 ? 's' : ''} left of Bot`;
}

// Helper to determine the badge (D, SB, BB) for a given seat position
function getPlayerBadge(position: number, numPlayers: number): string | null {
    if (numPlayers === 2) {
        if (position === 0) return 'SB/D';
        if (position === 1) return 'BB';
    } else {
        if (position === 0) return 'D';
        if (position === 1) return 'SB';
        if (position === 2) return 'BB';
    }
    return null;
}

type PlayPhaseProps = {
    pot: number;
    currentBet: number;
    botPosition: number;
    holeCards: Card[];
    communityCards: Card[];
    street: 'preflop' | 'flop' | 'turn' | 'river';
    players: PlayerState[];
    currentPlayerIdx: number;
    isLoading: boolean;
    botResponse: BotResponse | null;
    showQValues: boolean;
    setShowQValues: (show: boolean) => void;
    showRaiseInput: boolean;
    setShowRaiseInput: (show: boolean) => void;
    raiseInput: string;
    setRaiseInput: (val: string) => void;
    pickingFor: 'hole' | 'community' | null;
    onOpenCommunityPicker: () => void;
    onQueryBot: () => void;
    onRecordAction: (action: 'fold' | 'check_call' | 'raise', amount?: number) => void;
    onUndo: () => void;
    onEndHand: () => void;
    canUndo: boolean;
    undoLabel?: string;
    children?: React.ReactNode; // For injecting CardSelector
};

export default function PlayPhase({
    pot, currentBet, botPosition, holeCards, communityCards, street, players, currentPlayerIdx,
    isLoading, botResponse, showQValues, setShowQValues, showRaiseInput, setShowRaiseInput,
    raiseInput, setRaiseInput, pickingFor, onOpenCommunityPicker, onQueryBot, onRecordAction,
    onUndo, onEndHand, canUndo, undoLabel, children
}: PlayPhaseProps) {

    const currentPlayer = players[currentPlayerIdx];
    const isBotTurn = currentPlayer?.is_bot ?? false;
    const botPlayer = players.find(p => p.is_bot);

    return (
        <div className="flex-1 flex flex-col min-h-[100dvh]">
            {/* Header */}
            <header className="p-4 flex justify-between items-center border-b border-[var(--color-border-color)] bg-slate-950/30 sticky top-0 z-10">
                <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                        <span className="text-lg text-[var(--color-accent)]">♠</span>
                        <span className="text-lg font-bold text-[var(--color-text-primary)]">PokerBot</span>
                    </div>
                    {botPlayer && (
                        <div className="text-xs text-[var(--color-text-secondary)] font-semibold flex items-center gap-1">
                            Stack: <span className="text-[var(--color-text-primary)]">{botPlayer.stack}</span>
                        </div>
                    )}
                </div>
                <div className="flex flex-col items-end gap-0.5">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-[var(--color-text-secondary)]">Pot</span>
                        <span className="text-sm font-bold text-[var(--color-accent)]">{pot}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-[var(--color-text-secondary)]">Bet</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{currentBet}</span>
                    </div>
                </div>
            </header>

            <main className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto pb-36">
                {/* Bot's Cards */}
                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                    <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold mb-2">Bot&apos;s Hand (Seat {botPosition})</p>
                    <div className="flex gap-2">
                        {holeCards.map((c, i) => (
                            <div key={i} className={`card-mini suit-${c.suit}`}>
                                <span className="card-rank">{c.rank}</span>
                                <span className="card-suit">{suitSym(c.suit)}</span>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Community Cards */}
                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                    <div className="flex justify-between items-center mb-2">
                        <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">
                            Board — {street}
                        </p>
                        {((street === 'preflop' && communityCards.length === 0) ||
                            (street === 'flop' && communityCards.length === 3) ||
                            (street === 'turn' && communityCards.length === 4)) && (
                                <button onClick={onOpenCommunityPicker}
                                    className="text-xs font-semibold px-3 py-1 rounded-lg bg-[var(--color-accent)]/15 text-[var(--color-accent)] border border-[var(--color-accent)]/30 hover:bg-[var(--color-accent)]/25 active:scale-95 transition-all">
                                    + Deal {communityCards.length === 0 ? 'Flop' : communityCards.length === 3 ? 'Turn' : 'River'}
                                </button>
                            )}
                    </div>
                    <div className="flex gap-2">
                        {communityCards.length > 0 ? communityCards.map((c, i) => (
                            <div key={i} className={`card-mini suit-${c.suit}`}>
                                <span className="card-rank">{c.rank}</span>
                                <span className="card-suit">{suitSym(c.suit)}</span>
                            </div>
                        )) : (
                            <p className="text-sm text-slate-500 italic">No community cards yet</p>
                        )}
                    </div>
                </section>

                {/* Card Selector (if active) */}
                {children}

                {/* Bot Recommendation */}
                {isLoading && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-2xl p-6 flex items-center justify-center gap-3">
                        <div className="w-5 h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                        <span className="text-sm text-[var(--color-text-secondary)]">Thinking...</span>
                    </section>
                )}

                {botResponse && !isLoading && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-2xl p-5 animate-recommend">
                        <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold mb-3">Agent Recommendation</p>
                        <div className={`text-3xl font-bold uppercase tracking-wide mb-2 ${ACTION_COLORS[botResponse.action] ?? 'text-white'}`}
                            style={{ textShadow: '0 0 20px rgba(16,185,129,0.3)' }}>
                            {botResponse.action.replace('_', ' ')}
                        </div>
                        {botResponse.amount && (
                            <p className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">Amount: {botResponse.amount} chips</p>
                        )}
                        <div className="flex gap-4 text-sm">
                            <div>
                                <span className="text-[var(--color-text-secondary)]">Equity: </span>
                                <span className="font-semibold text-[var(--color-accent)]">{(botResponse.equity * 100).toFixed(1)}%</span>
                            </div>
                            <div>
                                <span className="text-[var(--color-text-secondary)]">Strength: </span>
                                <span className="font-semibold text-[var(--color-text-primary)]">{botResponse.hand_strength_category}</span>
                            </div>
                        </div>
                        {/* Equity bar */}
                        <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden mt-3">
                            <div className="h-full bg-[var(--color-accent)] shadow-[0_0_10px_var(--color-accent-glow)] rounded-full transition-all duration-500"
                                style={{ width: `${botResponse.equity * 100}%` }} />
                        </div>
                        {/* Q-values */}
                        {botResponse.q_values && (
                            <div className="mt-3">
                                <button onClick={() => setShowQValues(!showQValues)}
                                    className="text-xs text-[var(--color-text-secondary)] hover:text-white transition-colors">
                                    {showQValues ? '▼' : '▶'} Q-Values
                                </button>
                                {showQValues && (
                                    <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                                        {Object.entries(botResponse.q_values).sort(([, a], [, b]) => b - a).map(([action, val]) => (
                                            <div key={action} className={`flex justify-between px-2 py-1 rounded ${action === botResponse!.action ? 'bg-[var(--color-accent)]/15 text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}>
                                                <span>{action}</span>
                                                <span className="font-mono">{val.toFixed(1)}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </section>
                )}

                {/* Active Player Indicator + Opponent Actions */}
                {!pickingFor && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                        <div className="flex justify-between items-center mb-5">
                            <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">
                                {currentPlayerIdx === -1 ? 'Betting Round Complete' : (isBotTurn ? 'Bot\'s Turn' : `Acting: ${relativeLabel(currentPlayer?.position ?? 0, botPosition, players.length)}`)}
                            </p>
                            <div className="flex gap-2">
                                {players.map((p, i) => {
                                    const badge = getPlayerBadge(p.position, players.length);
                                    return (
                                        <div key={i} className="flex flex-col items-center gap-1">
                                            {badge && (
                                                <span className="text-[8px] font-bold text-[var(--color-text-secondary)] tracking-tighter uppercase">{badge}</span>
                                            )}
                                            {!badge && <span className="h-3" />} {/* Spacer to align items vertically */}
                                            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold border transition-all
                                              ${!p.is_active ? 'bg-slate-900 border-slate-700 text-slate-600 line-through'
                                                    : i === currentPlayerIdx ? 'bg-[var(--color-accent)]/20 border-[var(--color-accent)] text-[var(--color-accent)] shadow-[0_0_8px_var(--color-accent-glow)]'
                                                        : p.is_bot ? 'bg-blue-500/15 border-blue-500/40 text-blue-400'
                                                            : 'bg-slate-800 border-slate-600 text-slate-300'}`}>
                                                {p.is_bot ? 'B' : p.position}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {currentPlayerIdx !== -1 && (isBotTurn ? (
                            <button onClick={onQueryBot}
                                disabled={isLoading}
                                className="w-full py-3 rounded-xl font-bold text-sm uppercase tracking-wider transition-all duration-200 bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_4px_16px_rgba(16,185,129,0.3)]">
                                {isLoading ? 'Thinking...' : 'Get Bot Action'}
                            </button>
                        ) : (
                            <div className="flex flex-col gap-2">
                                <div className="grid grid-cols-3 gap-2">
                                    <button onClick={() => onRecordAction('fold')}
                                        className="py-3 rounded-xl font-semibold text-sm uppercase bg-red-500/10 text-[var(--color-danger)] border border-red-500/30 hover:bg-[var(--color-danger)] hover:text-white active:scale-95 transition-all">
                                        Fold
                                    </button>
                                    <button onClick={() => onRecordAction('check_call')}
                                        className="py-3 rounded-xl font-semibold text-sm uppercase bg-blue-500/10 text-[var(--color-info)] border border-blue-500/30 hover:bg-[var(--color-info)] hover:text-white active:scale-95 transition-all">
                                        {currentBet > (currentPlayer?.bet ?? 0) ? 'Call' : 'Check'}
                                    </button>
                                    <button onClick={() => setShowRaiseInput(!showRaiseInput)}
                                        className="py-3 rounded-xl font-semibold text-sm uppercase bg-emerald-500/10 text-[var(--color-accent)] border border-emerald-500/30 hover:bg-[var(--color-accent)] hover:text-slate-950 active:scale-95 transition-all">
                                        Raise
                                    </button>
                                </div>
                                {showRaiseInput && (
                                    <div className="flex gap-2 animate-fade-in">
                                        <input type="number" value={raiseInput} onChange={e => setRaiseInput(e.target.value)}
                                            placeholder="Total bet amount"
                                            className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-3 py-2 text-sm text-white font-semibold" />
                                        <button onClick={() => { if (raiseInput) onRecordAction('raise', Number(raiseInput)); }}
                                            disabled={!raiseInput}
                                            className="px-4 py-2 rounded-xl font-bold text-sm bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 transition-all disabled:opacity-40">
                                            Confirm
                                        </button>
                                    </div>
                                )}
                            </div>
                        ))}
                    </section>
                )}
            </main>

            {/* Bottom bar: Undo + End Hand */}
            <nav className="fixed bottom-0 w-full max-w-[480px] p-4 bg-gradient-to-t from-slate-950 from-60% to-transparent flex gap-3 z-20">
                {canUndo && (
                    <button onClick={onUndo}
                        className="flex-1 py-3 rounded-xl font-semibold text-sm bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all flex items-center justify-center gap-2">
                        ↩ Undo
                        {undoLabel && <span className="text-[10px] text-slate-500">({undoLabel})</span>}
                    </button>
                )}
                <button onClick={onEndHand}
                    className="flex-1 py-3 rounded-xl font-semibold text-sm uppercase bg-amber-500/15 text-amber-400 border border-amber-500/30 hover:bg-amber-500 hover:text-slate-950 active:scale-95 transition-all">
                    End Hand
                </button>
            </nav>
        </div>
    );
}
