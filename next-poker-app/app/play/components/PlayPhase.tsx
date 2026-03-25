import React from 'react';

type Card = { rank: string; suit: string };
type PlayerState = { position: number; stack: number; bet: number; hole_cards: Card[] | null; is_bot: boolean; is_active: boolean; has_acted: boolean };
type BotResponse = { action: string; action_id: number | null; amount: number | null };
type LegalActionState = {
    canFold: boolean;
    canCheck: boolean;
    canCall: boolean;
    canRaise: boolean;
    toCall: number;
    minRaiseTo: number | null;
    maxRaiseTo: number | null;
};
type ShowdownEntry = { playerIndex: number; position: number; cards: Card[]; mucked: boolean };
type ShowdownResult = { result: 'won' | 'lost' | 'push'; amount: number; delta: number };
type ResultFlash = { result: 'won' | 'lost' | 'push'; delta: number };

const ACTION_COLORS: Record<string, string> = {
    FOLD: 'text-red-400',
    CHECK: 'text-blue-300',
    CALL: 'text-blue-400',
    RAISE_SMALL: 'text-emerald-300',
    RAISE_LARGE: 'text-amber-300',
    AGGRO_SMALL: 'text-emerald-300',
    AGGRO_LARGE: 'text-amber-300',
    RAISE_33_POT: 'text-emerald-300',
    RAISE_66_POT: 'text-emerald-300',
    RAISE_POT: 'text-emerald-300',
    RAISE_133_POT: 'text-amber-300',
    ALL_IN: 'text-amber-300',
    fold: 'text-red-400',
    check: 'text-blue-300',
    call: 'text-blue-400',
    raise_amt: 'text-emerald-300',
};

const seatRoleByPlayers: Record<number, string[]> = {
    2: ['SB/BTN', 'BB'],
    3: ['SB', 'BB', 'BTN'],
    4: ['SB', 'BB', 'UTG', 'BTN'],
    5: ['SB', 'BB', 'UTG', 'CO', 'BTN'],
    6: ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'],
};

const suitSym = (s: string) => ({ s: '\u2660', h: '\u2665', d: '\u2666', c: '\u2663' }[s] ?? s);

function getSeatRole(position: number, numPlayers: number): string {
    const roles = seatRoleByPlayers[numPlayers] ?? seatRoleByPlayers[6];
    return roles[position] ?? `Seat ${position + 1}`;
}

function getPlayerName(player: PlayerState, numPlayers: number): string {
    const role = getSeatRole(player.position, numPlayers);
    if (player.is_bot) return `Bot (${role})`;
    return `Player ${player.position + 1} (${role})`;
}

function normalizeAction(action: string): string {
    if (action.startsWith('RAISE_') || action.startsWith('AGGRO_')) return 'raise_amt';
    return action.toLowerCase();
}

function displayAction(action: string): string {
    const normalized = normalizeAction(action);
    if (normalized === 'raise_amt') return 'RAISE';
    return normalized.toUpperCase();
}

type PlayPhaseProps = {
    pot: number;
    currentBet: number;
    bigBlind: number;
    botPosition: number;
    holeCards: Card[];
    communityCards: Card[];
    street: 'preflop' | 'flop' | 'turn' | 'river';
    players: PlayerState[];
    currentPlayerIdx: number;
    isLoading: boolean;
    botResponse: BotResponse | null;
    showRaiseInput: boolean;
    setShowRaiseInput: (show: boolean) => void;
    raiseInput: string;
    setRaiseInput: (val: string) => void;
    onOpenCommunityPicker: () => void;
    onQueryBot: () => void;
    onRecordAction: (action: 'fold' | 'check_call' | 'raise', amount?: number) => void;
    onUndo: () => void;
    canUndo: boolean;
    undoLabel?: string;
    legalActions: LegalActionState;
    showdownMode: boolean;
    showdownEntries: ShowdownEntry[];
    currentShowdownPlayerIndex: number | null;
    showdownCanResolve: boolean;
    isResolvingShowdown: boolean;
    showdownError: string | null;
    showdownResult: ShowdownResult | null;
    resultFlash: ResultFlash | null;
    onMuckShowdown: () => void;
    onClearShowdown: () => void;
    onResolveShowdown: () => void;
    children?: React.ReactNode;
};

export default function PlayPhase({
    pot,
    currentBet,
    bigBlind,
    botPosition,
    holeCards,
    communityCards,
    street,
    players,
    currentPlayerIdx,
    isLoading,
    botResponse,
    showRaiseInput,
    setShowRaiseInput,
    raiseInput,
    setRaiseInput,
    onOpenCommunityPicker,
    onQueryBot,
    onRecordAction,
    onUndo,
    canUndo,
    undoLabel,
    legalActions,
    showdownMode,
    showdownEntries,
    currentShowdownPlayerIndex,
    showdownCanResolve,
    isResolvingShowdown,
    showdownError,
    showdownResult,
    resultFlash,
    onMuckShowdown,
    onClearShowdown,
    onResolveShowdown,
    children,
}: PlayPhaseProps) {
    const currentPlayer = players[currentPlayerIdx];
    const isBotTurn = currentPlayer?.is_bot ?? false;
    const botPlayer = players.find((p) => p.is_bot);
    const raiseMin = legalActions.minRaiseTo ?? 0;
    const raiseMax = legalActions.maxRaiseTo ?? 0;
    const raiseValue = Number(raiseInput);
    const hasRaiseValue = Number.isFinite(raiseValue);
    const raiseIsLegal = legalActions.canRaise && hasRaiseValue && raiseValue >= raiseMin && raiseValue <= raiseMax;
    const canDealNextStreet = (
        !showdownMode
        && currentPlayerIdx === -1
        && (
            (street === 'preflop' && communityCards.length === 0)
            || (street === 'flop' && communityCards.length === 3)
            || (street === 'turn' && communityCards.length === 4)
        )
    );

    return (
        <div className="flex-1 flex flex-col">
            <header className="p-2 px-3 flex justify-between items-center border-b border-[var(--color-border-color)] bg-slate-950/30 sticky top-0 z-10">
                <div className="flex flex-col gap-0">
                    <div className="flex items-center gap-1.5">
                        <span className="text-base text-[var(--color-accent)]">{"\u2660"}</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">PokerBot</span>
                    </div>
                    {botPlayer && (
                        <div className="text-[10px] text-[var(--color-text-secondary)] font-semibold">
                            Stack: <span className="text-[var(--color-text-primary)]">{botPlayer.stack}</span>
                        </div>
                    )}
                </div>
                <div className="flex flex-col items-end gap-0">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-[var(--color-text-secondary)] font-medium">Pot</span>
                        <span className="text-sm font-bold text-[var(--color-accent)]">{pot}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-[var(--color-text-secondary)] font-medium">Bet</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{currentBet}</span>
                    </div>
                </div>
            </header>

            <main className="flex-1 p-2 flex flex-col gap-2 pb-2">
                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-2.5">
                    <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-1.5">Bot&apos;s Hand (Seat {botPosition + 1})</p>
                    <div className="flex gap-1.5">
                        {holeCards.map((c, i) => (
                            <div key={i} className={`card-mini suit-${c.suit} scale-90 origin-left`}>
                                <span className="card-rank">{c.rank}</span>
                                <span className="card-suit">{suitSym(c.suit)}</span>
                            </div>
                        ))}
                    </div>
                </section>

                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-2.5">
                    <div className="flex justify-between items-center mb-1.5">
                        <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold">
                            Board - {street}
                        </p>
                        {canDealNextStreet && (
                            <button
                                onClick={onOpenCommunityPicker}
                                className="text-[10px] font-bold px-2 py-1 rounded-lg bg-[var(--color-accent)]/15 text-[var(--color-accent)] border border-[var(--color-accent)]/30 hover:bg-[var(--color-accent)]/25 active:scale-95 transition-all"
                            >
                                + Deal {communityCards.length === 0 ? 'Flop' : communityCards.length === 3 ? 'Turn' : 'River'}
                            </button>
                        )}
                    </div>
                    <div className="flex gap-1.5">
                        {communityCards.length > 0 ? communityCards.map((c, i) => (
                            <div key={i} className={`card-mini suit-${c.suit} scale-90 origin-left`}>
                                <span className="card-rank">{c.rank}</span>
                                <span className="card-suit">{suitSym(c.suit)}</span>
                            </div>
                        )) : (
                            <p className="text-xs text-slate-500 italic">No community cards yet</p>
                        )}
                    </div>
                </section>

                {!showdownMode && children}

                <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-2.5">
                    <div className="flex justify-between items-center mb-2">
                        <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold">Table</p>
                    </div>
                    <div className="flex flex-col gap-1.5">
                        {players.map((player, idx) => (
                            <div
                                key={idx}
                                className={`rounded-lg border px-3 py-1.5 ${idx === currentPlayerIdx ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border-color)] bg-slate-900/30'} ${!player.is_active ? 'opacity-60' : ''}`}
                            >
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-semibold text-[var(--color-text-primary)]">
                                        {getPlayerName(player, players.length)}
                                    </span>
                                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                                        {player.is_active ? 'Active' : 'Folded'}
                                    </span>
                                </div>
                                <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">
                                    Stack: <span className="text-[var(--color-text-primary)] font-semibold">{player.stack}</span> | Bet: <span className="text-[var(--color-text-primary)] font-semibold">{player.bet}</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    {!showdownMode && (
                        <div className="mt-2">
                            <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-1.5">
                                {currentPlayerIdx === -1 ? 'Betting Round Complete' : (isBotTurn ? 'Bot Turn' : `Acting: ${currentPlayer ? getPlayerName(currentPlayer, players.length) : ''}`)}
                            </p>

                            {currentPlayerIdx !== -1 && (isBotTurn ? (
                                <button
                                    onClick={onQueryBot}
                                    disabled={isLoading}
                                    className="w-full py-2 rounded-lg font-bold text-xs uppercase tracking-wider bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isLoading ? 'Thinking...' : 'Get Bot Action'}
                                </button>
                            ) : (
                                <div className="flex flex-col gap-2">
                                    <div className="grid grid-cols-3 gap-1.5">
                                        <button
                                            onClick={() => onRecordAction('fold')}
                                            disabled={!legalActions.canFold}
                                            className="py-2 rounded-lg font-bold text-xs uppercase bg-red-500/10 text-[var(--color-danger)] border border-red-500/30 hover:bg-[var(--color-danger)] hover:text-white active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                        >
                                            Fold
                                        </button>
                                        <button
                                            onClick={() => onRecordAction('check_call')}
                                            disabled={!(legalActions.canCheck || legalActions.canCall)}
                                            className="py-2 rounded-lg font-bold text-xs uppercase bg-blue-500/10 text-[var(--color-info)] border border-blue-500/30 hover:bg-[var(--color-info)] hover:text-white active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                        >
                                            {legalActions.canCall ? 'Call' : 'Check'}
                                        </button>
                                        <button
                                            onClick={() => setShowRaiseInput(!showRaiseInput)}
                                            disabled={!legalActions.canRaise}
                                            className="py-2 rounded-lg font-bold text-xs uppercase bg-emerald-500/10 text-[var(--color-accent)] border border-emerald-500/30 hover:bg-[var(--color-accent)] hover:text-slate-950 active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                        >
                                            Raise
                                        </button>
                                    </div>
                                    {showRaiseInput && legalActions.canRaise && (
                                        <div className="flex flex-col gap-1.5 animate-fade-in mt-1">
                                            <div className="flex gap-1.5">
                                                <input
                                                    type="number"
                                                    value={raiseInput}
                                                    onChange={(e) => setRaiseInput(e.target.value)}
                                                    min={raiseMin}
                                                    max={raiseMax}
                                                    placeholder={`${raiseMin}-${raiseMax}`}
                                                    className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs text-white font-semibold"
                                                />
                                                <button
                                                    onClick={() => { if (raiseIsLegal) onRecordAction('raise', Math.trunc(raiseValue)); }}
                                                    disabled={!raiseIsLegal}
                                                    className="px-3 py-1.5 rounded-lg font-bold text-xs bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-40"
                                                >
                                                    Confirm
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </section>

                {isLoading && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-xl p-3 flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                        <span className="text-xs text-[var(--color-text-secondary)]">Thinking...</span>
                    </section>
                )}

                {botResponse && !isLoading && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-xl p-3 animate-recommend">
                        <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-1.5">Agent Recommendation</p>
                        <div className={`text-2xl font-bold uppercase tracking-wide ${ACTION_COLORS[botResponse.action] ?? ACTION_COLORS[normalizeAction(botResponse.action)] ?? 'text-white'}`}>
                            {displayAction(botResponse.action)}
                        </div>
                        {botResponse.amount !== null && botResponse.amount !== undefined && (
                            <p className="text-sm font-semibold text-[var(--color-text-primary)] mt-1">Amount: {botResponse.amount}</p>
                        )}
                    </section>
                )}

                {showdownMode && (
                    <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-xl p-3">
                        <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider font-bold mb-2">Opponent Reveals</p>
                        <div className="flex flex-col gap-1.5">
                            {showdownEntries.map((entry, idx) => {
                                const player = players[entry.playerIndex];
                                const name = player ? getPlayerName(player, players.length) : `Player ${entry.position + 1}`;
                                const isCurrent = entry.playerIndex === currentShowdownPlayerIndex;
                                return (
                                    <div
                                        key={entry.playerIndex}
                                        className={`rounded-lg border px-3 py-1.5 ${isCurrent ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border-color)] bg-slate-900/30'}`}
                                    >
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs font-semibold text-[var(--color-text-primary)]">{idx + 1}. {name}</span>
                                            <span className="text-[10px] text-[var(--color-text-secondary)]">
                                                {entry.mucked ? 'Mucked' : (entry.cards.length === 2 ? entry.cards.map((c) => `${c.rank}${suitSym(c.suit)}`).join(' ') : 'Pending')}
                                            </span>
                                        </div>

                                        {isCurrent && !entry.mucked && !showdownResult && (
                                            <div className="mt-1.5">
                                                {children}
                                                <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                                                    <button
                                                        onClick={onMuckShowdown}
                                                        className="py-1.5 rounded-lg text-xs font-bold uppercase border border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20"
                                                    >
                                                        Muck
                                                    </button>
                                                    <button
                                                        onClick={onClearShowdown}
                                                        className="py-1.5 rounded-lg text-xs font-bold uppercase border border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700"
                                                    >
                                                        Clear
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        {showdownResult && (
                            <div className="mt-2 rounded-lg border border-[var(--color-border-color)] bg-slate-900/40 px-3 py-1.5 text-center">
                                <p className="text-sm font-bold text-[var(--color-text-primary)]">
                                    {showdownResult.result.toUpperCase()} {showdownResult.amount}
                                </p>
                            </div>
                        )}

                        <div className="mt-2">
                            {!showdownResult ? (
                                <button
                                    onClick={onResolveShowdown}
                                    disabled={!showdownCanResolve || isResolvingShowdown}
                                    className="w-full py-2 rounded-lg font-bold text-xs uppercase tracking-wider bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 disabled:opacity-40"
                                >
                                    {isResolvingShowdown ? 'Resolving...' : 'Resolve Showdown'}
                                </button>
                            ) : (
                                <p className="text-[10px] text-[var(--color-text-secondary)] font-bold uppercase tracking-wider text-center py-1">
                                    Next hand...
                                </p>
                            )}
                        </div>
                    </section>
                )}
            </main>

            {canUndo && (
                <div className="p-2 bg-slate-950/50 backdrop-blur border-t border-slate-800">
                    <button
                        onClick={onUndo}
                        className="w-full py-2 rounded-lg font-bold text-[10px] bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all flex items-center justify-center gap-2"
                    >
                        Undo {undoLabel && `(${undoLabel})`}
                    </button>
                </div>
            )}

            {resultFlash && (
                <div
                    className={`fixed top-2 left-1/2 -translate-x-1/2 z-30 px-3 py-1 rounded-lg border shadow-lg ${resultFlash.result === 'won'
                            ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                            : resultFlash.result === 'lost'
                                ? 'border-red-400/60 bg-red-500/20 text-red-200'
                                : 'border-slate-400/60 bg-slate-700/30 text-slate-100'
                        }`}
                >
                    <p className="text-xs font-bold">
                        {resultFlash.delta >= 0 ? '+' : ''}{resultFlash.delta}
                    </p>
                </div>
            )}
        </div>
    );
}
