'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

// ── Constants & Types ────────────────────────────────────────────────────────

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const;
const SUITS = [
    { key: 's', sym: '♠', label: 'Spades' },
    { key: 'h', sym: '♥', label: 'Hearts' },
    { key: 'd', sym: '♦', label: 'Diamonds' },
    { key: 'c', sym: '♣', label: 'Clubs' },
] as const;
const ACTION_COLORS: Record<string, string> = {
    FOLD: 'text-red-400', CALL: 'text-blue-400', RAISE_SMALL: 'text-emerald-400',
    RAISE_MEDIUM: 'text-emerald-300', RAISE_LARGE: 'text-yellow-400', ALL_IN: 'text-amber-300',
};

type Card = { rank: string; suit: string };
type PlayerState = { position: number; stack: number; bet: number; hole_cards: Card[] | null; is_bot: boolean; is_active: boolean; has_acted: boolean; };
type BotResponse = { action: string; action_id: number; amount: number | null; equity: number; hand_strength_category: string; q_values: Record<string, number> | null };
type Phase = 'setup-size' | 'setup-details' | 'deal-position' | 'deal-hole' | 'play' | 'hand-result';

type HandState = {
    botPosition: number;
    holeCards: Card[];
    communityCards: Card[];
    players: PlayerState[];
    pot: number;
    currentBet: number;
    currentPlayerIdx: number; // index in players array of who's acting
    botResponse: BotResponse | null;
    street: 'preflop' | 'flop' | 'turn' | 'river';
    isLoading: boolean;
};

type HistoryEntry = { phase: Phase; hand: HandState; label: string };

const suitSym = (s: string) => ({ s: '♠', h: '♥', d: '♦', c: '♣' }[s] ?? s);

function initPlayers(count: number, stack: number, botPos: number): PlayerState[] {
    return Array.from({ length: count }, (_, i) => ({
        position: i, stack, bet: 0,
        hole_cards: i === botPos ? [] : null,
        is_bot: i === botPos, is_active: true, has_acted: false,
    }));
}

function nextActivePlayer(players: PlayerState[], afterIdx: number): number {
    const n = players.length;
    for (let offset = 1; offset < n; offset++) {
        const idx = (afterIdx + offset) % n;
        if (players[idx].is_active) return idx;
    }
    return -1; // only one player left
}

function firstActivePlayerFrom(players: PlayerState[], startPos: number): number {
    // Find the first active player starting from a given seat position (inclusive)
    const n = players.length;
    for (let offset = 0; offset < n; offset++) {
        const idx = (startPos + offset) % n;
        if (players[idx].is_active) return idx;
    }
    return startPos;
}

function relativeLabel(pos: number, botPos: number, total: number): string {
    const diff = ((pos - botPos) % total + total) % total;
    if (diff === 0) return 'Bot';
    return `Player ${pos} — ${diff} seat${diff > 1 ? 's' : ''} left of Bot`;
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function PlayPage() {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    // Session config
    const [tableSize, setTableSize] = useState(6);
    const [smallBlind, setSmallBlind] = useState(1);
    const [bigBlind, setBigBlind] = useState(2);
    const [buyIn, setBuyIn] = useState(200);
    const [sessionStacks, setSessionStacks] = useState<number[]>([]);
    const [sessionProfit, setSessionProfit] = useState(0);

    // Phase + hand state
    const [phase, setPhase] = useState<Phase>('setup-size');
    const [hand, setHand] = useState<HandState>({
        botPosition: 0, holeCards: [], communityCards: [], players: [],
        pot: 0, currentBet: 0, currentPlayerIdx: 0, botResponse: null,
        street: 'preflop', isLoading: false,
    });

    // Card selector
    const [pickingFor, setPickingFor] = useState<'hole' | 'community' | null>(null);
    const [pendingRank, setPendingRank] = useState<string | null>(null);

    // Undo history
    const [history, setHistory] = useState<HistoryEntry[]>([]);

    // Raise input
    const [raiseInput, setRaiseInput] = useState('');
    const [showRaiseInput, setShowRaiseInput] = useState(false);

    // Q-values toggle
    const [showQValues, setShowQValues] = useState(false);

    // Hand result state
    const [resultType, setResultType] = useState<'won' | 'lost' | null>(null);
    const [resultAmt, setResultAmt] = useState('');

    const usedCards = useCallback((): Set<string> => {
        const set = new Set<string>();
        hand.holeCards.forEach(c => set.add(`${c.rank}${c.suit}`));
        hand.communityCards.forEach(c => set.add(`${c.rank}${c.suit}`));
        return set;
    }, [hand.holeCards, hand.communityCards]);

    // ── Push state to history ──
    const pushHistory = useCallback((label: string) => {
        setHistory(prev => [...prev, { phase, hand: JSON.parse(JSON.stringify(hand)), label }]);
    }, [phase, hand]);

    // ── Undo last action ──
    const undo = useCallback(() => {
        setHistory(prev => {
            if (prev.length === 0) return prev;
            const last = prev[prev.length - 1];
            setPhase(last.phase);
            setHand(last.hand);
            setPickingFor(null);
            setPendingRank(null);
            setShowRaiseInput(false);
            return prev.slice(0, -1);
        });
    }, []);

    // ── API call ──
    const queryBot = useCallback(async (currentHand: HandState, bb: number) => {
        setHand(prev => ({ ...prev, isLoading: true }));
        try {
            const body = {
                community_cards: currentHand.communityCards.map(c => ({ rank: c.rank, suit: c.suit })),
                pot: currentHand.pot,
                players: currentHand.players.map(p => ({
                    position: p.position, stack: p.stack, bet: p.bet,
                    hole_cards: p.is_bot ? currentHand.holeCards.map(c => ({ rank: c.rank, suit: c.suit })) : null,
                    is_bot: p.is_bot, is_active: p.is_active,
                })),
                bot_position: currentHand.botPosition,
                current_bet: currentHand.currentBet,
                big_blind: bb,
                model_version: 'v19',
            };
            const res = await fetch(`${BACKEND}/poker/action`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(`API error: ${res.status}`);
            const data: BotResponse = await res.json();
            setHand(prev => {
                // Apply bot action to game state
                const players = prev.players.map(p => ({ ...p }));
                const botIdx = prev.currentPlayerIdx;
                let pot = prev.pot;
                let currentBet = prev.currentBet;

                players[botIdx] = { ...players[botIdx], has_acted: true };

                if (data.action === 'FOLD') {
                    players[botIdx].is_active = false;
                } else if (data.action === 'CALL') {
                    const callAmt = Math.min(currentBet - players[botIdx].bet, players[botIdx].stack);
                    players[botIdx].bet += callAmt;
                    players[botIdx].stack -= callAmt;
                    pot += callAmt;
                } else if (['RAISE_SMALL', 'RAISE_MEDIUM', 'RAISE_LARGE', 'ALL_IN'].includes(data.action) && data.amount) {
                    const raiseAmt = data.amount;
                    const additional = raiseAmt - players[botIdx].bet;

                    // Reset has_acted for all other active players
                    players.forEach((p, i) => {
                        if (i !== botIdx && p.is_active) p.has_acted = false;
                    });

                    players[botIdx].bet = raiseAmt;
                    players[botIdx].stack -= additional;
                    pot += additional;
                    currentBet = raiseAmt;
                }

                // Check if the betting round is complete
                const activePlayers = players.filter(p => p.is_active);
                const allActedAndMatched = activePlayers.every(p =>
                    p.has_acted && (p.bet === currentBet || p.stack === 0)
                );

                if (allActedAndMatched || activePlayers.length <= 1) {
                    return {
                        ...prev,
                        players, pot, currentBet,
                        botResponse: data,
                        isLoading: false,
                        currentPlayerIdx: -1, // End of round
                    };
                }

                const nextIdx = nextActivePlayer(players, botIdx);
                return {
                    ...prev,
                    players, pot, currentBet,
                    botResponse: data,
                    isLoading: false,
                    currentPlayerIdx: nextIdx !== -1 ? nextIdx : botIdx,
                };
            });
        } catch (err) {
            console.error('Bot query failed:', err);
            setHand(prev => ({ ...prev, isLoading: false }));
        }
    }, []);

    // ── Card selection handler ──
    const selectCard = useCallback((rank: string, suit: string) => {
        const card: Card = { rank, suit };
        pushHistory(`Select ${rank}${suitSym(suit)}`);

        if (pickingFor === 'hole') {
            const newHole = [...hand.holeCards, card];
            setHand(prev => ({ ...prev, holeCards: newHole }));
            if (newHole.length >= 2) {
                setPickingFor(null);
                setPhase('play');

                // Construct players for preflop state
                const players = hand.players.map(p => ({ ...p, bet: 0, has_acted: false }));
                const n = players.length;

                // Handle preflop blinds for tables with > 2 players
                if (n > 2) {
                    // SB is Seat 1, BB is Seat 2 (Dealer is Seat 0)
                    const sbIdx = firstActivePlayerFrom(players, 1);
                    const bbIdx = firstActivePlayerFrom(players, sbIdx + 1);

                    if (sbIdx !== -1) {
                        const sbAmt = Math.min(smallBlind, players[sbIdx].stack);
                        players[sbIdx].bet += sbAmt;
                        players[sbIdx].stack -= sbAmt;
                    }
                    if (bbIdx !== -1) {
                        const bbAmt = Math.min(bigBlind, players[bbIdx].stack);
                        players[bbIdx].bet += bbAmt;
                        players[bbIdx].stack -= bbAmt;
                    }
                } else if (n === 2) {
                    // Heads Up: Dealer (Seat 0) is SB, Seat 1 is BB
                    const sbIdx = 0;
                    const bbIdx = 1;

                    const sbAmt = Math.min(smallBlind, players[sbIdx].stack);
                    players[sbIdx].bet += sbAmt;
                    players[sbIdx].stack -= sbAmt;

                    const bbAmt = Math.min(bigBlind, players[bbIdx].stack);
                    players[bbIdx].bet += bbAmt;
                    players[bbIdx].stack -= bbAmt;
                }

                const newPot = Math.min(smallBlind, players[0]?.stack ?? smallBlind) + Math.min(bigBlind, players[1]?.stack ?? bigBlind);
                const currentBet = bigBlind;

                // First to act preflop: UTG (Seat 3, or Seat 0 for heads up)
                const utgIdx = n > 2 ? firstActivePlayerFrom(players, 3) : 0;

                setHand(prev => ({
                    ...prev,
                    holeCards: newHole,
                    pot: newPot,
                    currentBet: currentBet,
                    players,
                    currentPlayerIdx: utgIdx,
                    botResponse: null,
                }));
            }
        } else if (pickingFor === 'community') {
            const newComm = [...hand.communityCards, card];
            const street = newComm.length <= 3 ? 'flop' : newComm.length === 4 ? 'turn' : 'river';

            setHand(prev => ({ ...prev, communityCards: newComm, street }));
        }
        setPendingRank(null);
    }, [pickingFor, hand, pushHistory, smallBlind, bigBlind]);

    // ── Confirm Community Cards ──
    const confirmCommunityCards = useCallback(() => {
        pushHistory('Confirm dealt cards');
        setPickingFor(null);

        let street = 'flop';
        if (hand.communityCards.length === 4) street = 'turn';
        if (hand.communityCards.length === 5) street = 'river';

        // New street: reset bets to 0 and set first active player from seat 1 (postflop)
        const players = hand.players.map(p => ({ ...p, bet: 0, has_acted: false }));
        const firstToAct = firstActivePlayerFrom(players, 1);

        setHand(prev => ({
            ...prev,
            street: street as any,
            players,
            currentBet: 0,
            currentPlayerIdx: firstToAct,
            botResponse: null,
        }));
    }, [hand, pushHistory]);

    // ── Record opponent action ──
    const recordOpponentAction = useCallback((action: 'fold' | 'check_call' | 'raise', raiseAmt?: number) => {
        const playerIdx = hand.currentPlayerIdx;
        const player = hand.players[playerIdx];
        if (!player) return;

        pushHistory(`P${player.position} ${action}${raiseAmt ? ` ${raiseAmt}` : ''}`);

        const players = hand.players.map(p => ({ ...p }));
        let pot = hand.pot;
        let currentBet = hand.currentBet;

        if (action === 'fold') {
            players[playerIdx].is_active = false;
            players[playerIdx].has_acted = true;
        } else if (action === 'check_call') {
            const callAmt = Math.min(currentBet - players[playerIdx].bet, players[playerIdx].stack);
            players[playerIdx].bet += callAmt;
            players[playerIdx].stack -= callAmt;
            players[playerIdx].has_acted = true;
            pot += callAmt;
        } else if (action === 'raise' && raiseAmt) {
            const totalBet = raiseAmt;
            const additional = totalBet - players[playerIdx].bet;

            // Reset has_acted for all other active players
            players.forEach((p, i) => {
                if (i !== playerIdx && p.is_active) p.has_acted = false;
            });

            players[playerIdx].bet = totalBet;
            players[playerIdx].stack -= additional;
            players[playerIdx].has_acted = true;
            pot += additional;
            currentBet = totalBet;
        }

        // Check if the betting round is complete
        const activePlayersCount = players.filter(p => p.is_active).length;
        const allActedAndMatched = players.filter(p => p.is_active).every(p =>
            p.has_acted && (p.bet === currentBet || p.stack === 0)
        );

        if (allActedAndMatched || activePlayersCount <= 1) {
            setHand(prev => ({
                ...prev, players, pot, currentBet,
                currentPlayerIdx: -1, // End of round
            }));
        } else {
            // Find next active player in rotation
            const nextIdx = nextActivePlayer(players, playerIdx);
            setHand(prev => ({
                ...prev, players, pot, currentBet,
                currentPlayerIdx: nextIdx !== -1 ? nextIdx : playerIdx,
            }));
        }

        setShowRaiseInput(false);
        setRaiseInput('');
    }, [hand, pushHistory, queryBot, bigBlind]);

    // ── Start a new hand ──
    const startNewHand = useCallback(() => {
        setHistory([]);
        setPhase('deal-position');
        setHand({
            botPosition: 0, holeCards: [], communityCards: [],
            players: initPlayers(tableSize, sessionStacks[0] ?? buyIn, 0),
            pot: 0, currentBet: 0, currentPlayerIdx: 0, botResponse: null,
            street: 'preflop', isLoading: false,
        });
        setPickingFor(null);
        setPendingRank(null);
        setShowQValues(false);
        setResultType(null);
        setResultAmt('');
    }, [tableSize, sessionStacks, buyIn]);

    // ── Finish hand with result ──
    const finishHand = useCallback((won: boolean, amount: number) => {
        const delta = won ? amount : -amount;
        setSessionProfit(prev => prev + delta);
        setSessionStacks(prev => {
            const newStacks = [...prev];
            newStacks[0] = (newStacks[0] ?? buyIn) + delta;
            return newStacks;
        });
        setPhase('setup-details'); // go back to between-hands view
        setHistory([]);
    }, [buyIn]);

    if (!mounted) return null;

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Setup - Table Size
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'setup-size') {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-10 p-6 min-h-[100dvh]">
                <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                    <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-3 flex items-center justify-center gap-3">
                        <span className="text-5xl text-[var(--color-accent)]">♠</span>
                        PokerBot
                    </h1>
                    <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.2em] uppercase font-semibold">Select Table Size</p>
                </div>
                <div className="w-full max-w-sm grid grid-cols-2 gap-4 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                    {[2, 3, 4, 5, 6].map(size => (
                        <button key={size} onClick={() => setTableSize(size)}
                            className={`p-6 rounded-2xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-1 ${tableSize === size
                                ? 'bg-[var(--color-accent)]/15 border-[var(--color-accent)] shadow-[0_0_20px_var(--color-accent-glow)] text-[var(--color-accent)] scale-105'
                                : 'bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)] hover:border-[var(--color-text-secondary)]'}`}>
                            <span className="text-4xl font-bold">{size}</span>
                            <span className={`text-[10px] uppercase tracking-wider font-semibold ${tableSize === size ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}>Max Players</span>
                        </button>
                    ))}
                </div>
                <div className="w-full max-w-sm mt-6 animate-slide-up" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
                    <button onClick={() => setPhase('setup-details')}
                        className="w-full py-5 px-6 rounded-2xl font-bold text-lg tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-3">
                        Continue <span className="text-xl">→</span>
                    </button>
                </div>
            </div>
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Setup - Details
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'setup-details') {
        const hasSession = sessionStacks.length > 0;
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
                        <button onClick={() => setPhase('setup-size')}
                            className="flex-1 py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-border-color)] bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] active:scale-95 flex items-center justify-center gap-2">
                            <span className="text-lg">←</span> Back
                        </button>
                    )}
                    <button onClick={() => {
                        if (!hasSession) setSessionStacks(Array(tableSize).fill(buyIn));
                        startNewHand();
                    }}
                        className="flex-[2] py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-2">
                        {hasSession ? 'New Hand' : 'Start Session'} <span className="text-lg">→</span>
                    </button>
                    {hasSession && (
                        <button onClick={() => { setSessionStacks([]); setSessionProfit(0); setPhase('setup-size'); }}
                            className="flex-1 py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-danger)]/30 bg-red-500/10 text-[var(--color-danger)] hover:bg-[var(--color-danger)] hover:text-white active:scale-95 flex items-center justify-center">
                            End
                        </button>
                    )}
                </div>
            </div>
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Deal - Position
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'deal-position') {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
                <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                    <h1 className="text-3xl font-bold tracking-tight text-[var(--color-text-primary)] mb-2">Bot Position</h1>
                    <p className="text-[var(--color-text-secondary)] text-xs tracking-[0.2em] uppercase font-semibold">Select where the bot is sitting</p>
                </div>
                <div className="w-full max-w-sm grid grid-cols-3 gap-3 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                    {Array.from({ length: tableSize }, (_, i) => (
                        <button key={i} onClick={() => {
                            pushHistory('Set position');
                            const players = initPlayers(tableSize, sessionStacks[0] ?? buyIn, i);
                            setHand(prev => ({ ...prev, botPosition: i, players, currentPlayerIdx: 0 }));
                            setPhase('deal-hole');
                            setPickingFor('hole');
                        }}
                            className="p-5 rounded-2xl border-2 transition-all duration-200 flex flex-col items-center gap-1 bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-accent)]/15 hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] active:scale-95">
                            <span className="text-2xl font-bold">{i}</span>
                            <span className="text-[9px] uppercase tracking-wider text-[var(--color-text-secondary)]">Seat {i}</span>
                        </button>
                    ))}
                </div>
                {history.length > 0 && (
                    <button onClick={undo} className="mt-4 px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all">
                        ↩ Undo
                    </button>
                )}
            </div>
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SHARED: Card Selector Component
    // ═══════════════════════════════════════════════════════════════════════════
    const used = usedCards();
    const cardSelector = pickingFor ? (
        <div className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4 animate-fade-in">
            <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold mb-3">
                {pickingFor === 'hole'
                    ? `Select hole card ${hand.holeCards.length + 1} of 2`
                    : `Select community card (${hand.communityCards.length + 1}${hand.communityCards.length < 3 ? '/3 flop' : hand.communityCards.length === 3 ? ' turn' : ' river'})`
                }
            </p>
            {pendingRank ? (
                <div className="flex flex-col gap-3 items-center">
                    <p className="text-sm text-[var(--color-text-primary)] font-semibold">Pick suit for {pendingRank}</p>
                    <div className="flex gap-3">
                        {SUITS.map(s => {
                            const isUsed = used.has(`${pendingRank}${s.key}`);
                            return (
                                <button key={s.key} disabled={isUsed}
                                    onClick={() => selectCard(pendingRank, s.key)}
                                    className={`w-16 h-20 rounded-xl border-2 flex flex-col items-center justify-center gap-1 transition-all duration-200 active:scale-90
                    ${isUsed ? 'opacity-20 cursor-not-allowed border-slate-700 bg-slate-900'
                                            : s.key === 'h' || s.key === 'd'
                                                ? 'border-red-500/40 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:border-red-400'
                                                : 'border-slate-500/40 bg-slate-800 text-slate-200 hover:bg-slate-700 hover:border-slate-400'}`}>
                                    <span className="text-2xl">{s.sym}</span>
                                    <span className="text-[9px] uppercase tracking-wider">{s.label}</span>
                                </button>
                            );
                        })}
                    </div>
                    <button onClick={() => setPendingRank(null)} className="text-xs text-[var(--color-text-secondary)] hover:text-white mt-1">← Back to ranks</button>
                </div>
            ) : (
                <div className="grid grid-cols-7 gap-2">
                    {RANKS.map(r => {
                        const allUsed = SUITS.every(s => used.has(`${r}${s.key}`));
                        return (
                            <button key={r} disabled={allUsed}
                                onClick={() => setPendingRank(r)}
                                className={`py-3 rounded-xl font-bold text-sm transition-all duration-150 active:scale-90 
                  ${allUsed ? 'opacity-20 cursor-not-allowed bg-slate-900 text-slate-600'
                                        : 'bg-slate-800 text-white border border-slate-600 hover:bg-slate-700 hover:border-[var(--color-accent)]'}`}>
                                {r}
                            </button>
                        );
                    })}
                </div>
            )}
            <div className="flex justify-between items-center mt-3 border-t border-slate-700/50 pt-3">
                <button onClick={() => { setPickingFor(null); setPendingRank(null); }}
                    className="text-xs text-[var(--color-text-secondary)] hover:text-white px-3 py-2">
                    Cancel
                </button>
                {pickingFor === 'community' && (
                    <button
                        onClick={confirmCommunityCards}
                        disabled={
                            !(hand.communityCards.length === 3) &&
                            !(hand.communityCards.length === 4) &&
                            !(hand.communityCards.length === 5)
                        }
                        className="text-xs font-bold px-4 py-2 rounded-lg bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed transition-all">
                        Confirm & Start Round
                    </button>
                )}
            </div>
        </div>
    ) : null;

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Deal Hole Cards
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'deal-hole') {
        return (
            <div className="flex-1 flex flex-col gap-6 p-6 min-h-[100dvh]">
                <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                    <h1 className="text-2xl font-bold text-[var(--color-text-primary)] mb-1">Deal Hole Cards</h1>
                    <p className="text-[var(--color-text-secondary)] text-xs">Bot is at seat {hand.botPosition}</p>
                </div>
                <div className="flex justify-center gap-3">
                    {[0, 1].map(i => hand.holeCards[i] ? (
                        <div key={i} className={`card-mini suit-${hand.holeCards[i].suit}`}>
                            <span className="card-rank">{hand.holeCards[i].rank}</span>
                            <span className="card-suit">{suitSym(hand.holeCards[i].suit)}</span>
                        </div>
                    ) : (
                        <div key={i} className="card-mini card-placeholder">
                            <span className="text-lg">?</span>
                        </div>
                    ))}
                </div>
                {cardSelector}
                <div className="flex justify-center">
                    {history.length > 0 && (
                        <button onClick={undo} className="px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all">
                            ↩ Undo
                        </button>
                    )}
                </div>
            </div>
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Play Phase (main game view)
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'play') {
        const currentPlayer = hand.players[hand.currentPlayerIdx];
        const isBotTurn = currentPlayer?.is_bot ?? false;
        const activePlayers = hand.players.filter(p => p.is_active);

        return (
            <div className="flex-1 flex flex-col min-h-[100dvh]">
                {/* Header */}
                <header className="p-4 flex justify-between items-center border-b border-[var(--color-border-color)] bg-slate-950/30 sticky top-0 z-10">
                    <div className="flex items-center gap-2">
                        <span className="text-lg text-[var(--color-accent)]">♠</span>
                        <span className="text-lg font-bold text-[var(--color-text-primary)]">PokerBot</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-[var(--color-text-secondary)]">Pot</span>
                        <span className="text-sm font-bold text-[var(--color-accent)]">{hand.pot}</span>
                        <span className="text-xs text-slate-500 mx-1">|</span>
                        <span className="text-xs text-[var(--color-text-secondary)]">Bet</span>
                        <span className="text-sm font-bold text-[var(--color-text-primary)]">{hand.currentBet}</span>
                    </div>
                </header>

                <main className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto pb-36">
                    {/* Bot's Cards */}
                    <section className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-4">
                        <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold mb-2">Bot&apos;s Hand (Seat {hand.botPosition})</p>
                        <div className="flex gap-2">
                            {hand.holeCards.map((c, i) => (
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
                                Board — {hand.street}
                            </p>
                            {((hand.street === 'preflop' && hand.communityCards.length === 0) ||
                                (hand.street === 'flop' && hand.communityCards.length === 3) ||
                                (hand.street === 'turn' && hand.communityCards.length === 4)) && (
                                    <button onClick={() => { pushHistory('Open card picker'); setPickingFor('community'); }}
                                        className="text-xs font-semibold px-3 py-1 rounded-lg bg-[var(--color-accent)]/15 text-[var(--color-accent)] border border-[var(--color-accent)]/30 hover:bg-[var(--color-accent)]/25 active:scale-95 transition-all">
                                        + Deal {hand.communityCards.length === 0 ? 'Flop' : hand.communityCards.length === 3 ? 'Turn' : 'River'}
                                    </button>
                                )}
                        </div>
                        <div className="flex gap-2">
                            {hand.communityCards.length > 0 ? hand.communityCards.map((c, i) => (
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
                    {cardSelector}

                    {/* Bot Recommendation */}
                    {hand.isLoading && (
                        <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-2xl p-6 flex items-center justify-center gap-3">
                            <div className="w-5 h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                            <span className="text-sm text-[var(--color-text-secondary)]">Thinking...</span>
                        </section>
                    )}

                    {hand.botResponse && !hand.isLoading && (
                        <section className="bg-[var(--color-surface)] border border-[var(--color-accent)]/30 rounded-2xl p-5 animate-recommend">
                            <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold mb-3">Agent Recommendation</p>
                            <div className={`text-3xl font-bold uppercase tracking-wide mb-2 ${ACTION_COLORS[hand.botResponse.action] ?? 'text-white'}`}
                                style={{ textShadow: '0 0 20px rgba(16,185,129,0.3)' }}>
                                {hand.botResponse.action.replace('_', ' ')}
                            </div>
                            {hand.botResponse.amount && (
                                <p className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">Amount: {hand.botResponse.amount} chips</p>
                            )}
                            <div className="flex gap-4 text-sm">
                                <div>
                                    <span className="text-[var(--color-text-secondary)]">Equity: </span>
                                    <span className="font-semibold text-[var(--color-accent)]">{(hand.botResponse.equity * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                    <span className="text-[var(--color-text-secondary)]">Strength: </span>
                                    <span className="font-semibold text-[var(--color-text-primary)]">{hand.botResponse.hand_strength_category}</span>
                                </div>
                            </div>
                            {/* Equity bar */}
                            <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden mt-3">
                                <div className="h-full bg-[var(--color-accent)] shadow-[0_0_10px_var(--color-accent-glow)] rounded-full transition-all duration-500"
                                    style={{ width: `${hand.botResponse.equity * 100}%` }} />
                            </div>
                            {/* Q-values */}
                            {hand.botResponse.q_values && (
                                <div className="mt-3">
                                    <button onClick={() => setShowQValues(!showQValues)}
                                        className="text-xs text-[var(--color-text-secondary)] hover:text-white transition-colors">
                                        {showQValues ? '▼' : '▶'} Q-Values
                                    </button>
                                    {showQValues && (
                                        <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                                            {Object.entries(hand.botResponse.q_values).sort(([, a], [, b]) => b - a).map(([action, val]) => (
                                                <div key={action} className={`flex justify-between px-2 py-1 rounded ${action === hand.botResponse!.action ? 'bg-[var(--color-accent)]/15 text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}>
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
                            <div className="flex justify-between items-center mb-3">
                                <p className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">
                                    {hand.currentPlayerIdx === -1 ? 'Betting Round Complete' : (isBotTurn ? 'Bot\'s Turn' : `Acting: ${relativeLabel(currentPlayer?.position ?? 0, hand.botPosition, hand.players.length)}`)}
                                </p>
                                <div className="flex gap-1">
                                    {hand.players.map((p, i) => (
                                        <div key={i} className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border transition-all
                      ${!p.is_active ? 'bg-slate-900 border-slate-700 text-slate-600 line-through'
                                                : i === hand.currentPlayerIdx ? 'bg-[var(--color-accent)]/20 border-[var(--color-accent)] text-[var(--color-accent)] shadow-[0_0_8px_var(--color-accent-glow)]'
                                                    : p.is_bot ? 'bg-blue-500/15 border-blue-500/40 text-blue-400'
                                                        : 'bg-slate-800 border-slate-600 text-slate-300'}`}>
                                            {p.is_bot ? 'B' : p.position}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {hand.currentPlayerIdx !== -1 && (isBotTurn ? (
                                <button onClick={() => { pushHistory('Query bot'); queryBot(hand, bigBlind); }}
                                    disabled={hand.isLoading}
                                    className="w-full py-3 rounded-xl font-bold text-sm uppercase tracking-wider transition-all duration-200 bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_4px_16px_rgba(16,185,129,0.3)]">
                                    {hand.isLoading ? 'Thinking...' : 'Get Bot Action'}
                                </button>
                            ) : (
                                <div className="flex flex-col gap-2">
                                    <div className="grid grid-cols-3 gap-2">
                                        <button onClick={() => recordOpponentAction('fold')}
                                            className="py-3 rounded-xl font-semibold text-sm uppercase bg-red-500/10 text-[var(--color-danger)] border border-red-500/30 hover:bg-[var(--color-danger)] hover:text-white active:scale-95 transition-all">
                                            Fold
                                        </button>
                                        <button onClick={() => recordOpponentAction('check_call')}
                                            className="py-3 rounded-xl font-semibold text-sm uppercase bg-blue-500/10 text-[var(--color-info)] border border-blue-500/30 hover:bg-[var(--color-info)] hover:text-white active:scale-95 transition-all">
                                            {hand.currentBet > (currentPlayer?.bet ?? 0) ? 'Call' : 'Check'}
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
                                            <button onClick={() => { if (raiseInput) recordOpponentAction('raise', Number(raiseInput)); }}
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
                    {history.length > 0 && (
                        <button onClick={undo}
                            className="flex-1 py-3 rounded-xl font-semibold text-sm bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all flex items-center justify-center gap-2">
                            ↩ Undo
                            <span className="text-[10px] text-slate-500">({history[history.length - 1]?.label})</span>
                        </button>
                    )}
                    <button onClick={() => { pushHistory('End hand'); setPhase('hand-result'); }}
                        className="flex-1 py-3 rounded-xl font-semibold text-sm uppercase bg-amber-500/15 text-amber-400 border border-amber-500/30 hover:bg-amber-500 hover:text-slate-950 active:scale-95 transition-all">
                        End Hand
                    </button>
                </nav>
            </div>
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDER: Hand Result
    // ═══════════════════════════════════════════════════════════════════════════
    if (phase === 'hand-result') {

        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
                <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                    <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">Hand Result</h1>
                    <p className="text-[var(--color-text-secondary)] text-xs">How did the bot do?</p>
                </div>

                <div className="w-full max-w-sm flex gap-3">
                    <button onClick={() => setResultType('won')}
                        className={`flex-1 py-5 rounded-2xl font-bold text-lg uppercase transition-all duration-200 active:scale-95
              ${resultType === 'won' ? 'bg-[var(--color-accent)] text-slate-950 shadow-[0_0_20px_var(--color-accent-glow)]'
                                : 'bg-emerald-500/10 border border-emerald-500/30 text-[var(--color-accent)] hover:bg-emerald-500/20'}`}>
                        Won
                    </button>
                    <button onClick={() => setResultType('lost')}
                        className={`flex-1 py-5 rounded-2xl font-bold text-lg uppercase transition-all duration-200 active:scale-95
              ${resultType === 'lost' ? 'bg-[var(--color-danger)] text-white shadow-[0_0_20px_var(--color-danger-glow)]'
                                : 'bg-red-500/10 border border-red-500/30 text-[var(--color-danger)] hover:bg-red-500/20'}`}>
                        Lost
                    </button>
                </div>

                {resultType && (
                    <div className="w-full max-w-sm flex flex-col gap-4 animate-fade-in">
                        <div className="flex items-center gap-3">
                            <label className="text-sm text-[var(--color-text-secondary)] font-semibold whitespace-nowrap">
                                Amount {resultType === 'won' ? 'won' : 'lost'}:
                            </label>
                            <input type="number" value={resultAmt} onChange={e => setResultAmt(e.target.value)}
                                placeholder="Chips" autoFocus
                                className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-3 py-3 text-lg text-white font-bold text-center" />
                        </div>
                        <button onClick={() => { if (resultAmt) finishHand(resultType === 'won', Number(resultAmt)); }}
                            disabled={!resultAmt}
                            className="w-full py-4 rounded-2xl font-bold text-sm uppercase tracking-wider transition-all duration-200 bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_4px_16px_rgba(16,185,129,0.3)]">
                            Confirm & Next Hand
                        </button>
                    </div>
                )}

                <button onClick={undo}
                    className="text-sm text-[var(--color-text-secondary)] hover:text-white transition-colors">
                    ↩ Back to hand
                </button>
            </div>
        );
    }

    return null;
}
