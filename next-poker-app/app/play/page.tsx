'use client';

import { useState, useCallback, useEffect } from 'react';
import SetupSize from './components/SetupSize';
import SetupDetails from './components/SetupDetails';
import DealPosition from './components/DealPosition';
import DealHoleCards from './components/DealHoleCards';
import CardSelector from './components/CardSelector';
import PlayPhase from './components/PlayPhase';
import HandResult from './components/HandResult';

// ── Constants & Types ────────────────────────────────────────────────────────

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

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
            street: street as 'flop' | 'turn' | 'river',
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
    }, [hand, pushHistory]);

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
    // RENDER ROUTING
    // ═══════════════════════════════════════════════════════════════════════════

    if (phase === 'setup-size') {
        return <SetupSize tableSize={tableSize} setTableSize={setTableSize} onContinue={() => setPhase('setup-details')} />;
    }

    if (phase === 'setup-details') {
        return (
            <SetupDetails
                hasSession={sessionStacks.length > 0} sessionStacks={sessionStacks} sessionProfit={sessionProfit}
                smallBlind={smallBlind} setSmallBlind={setSmallBlind} bigBlind={bigBlind} setBigBlind={setBigBlind}
                buyIn={buyIn} setBuyIn={setBuyIn} onBack={() => setPhase('setup-size')}
                onStart={() => { if (sessionStacks.length === 0) setSessionStacks(Array(tableSize).fill(buyIn)); startNewHand(); }}
                onEnd={() => { setSessionStacks([]); setSessionProfit(0); setPhase('setup-size'); }}
            />
        );
    }

    if (phase === 'deal-position') {
        return (
            <DealPosition
                tableSize={tableSize}
                onSelectSeat={(i) => {
                    pushHistory('Set position');
                    const players = initPlayers(tableSize, sessionStacks[0] ?? buyIn, i);
                    setHand(prev => ({ ...prev, botPosition: i, players, currentPlayerIdx: 0 }));
                    setPhase('deal-hole');
                    setPickingFor('hole');
                }}
                canUndo={history.length > 0} onUndo={undo}
            />
        );
    }

    const commonCardSelectorProps = {
        pickingFor, holeCardsCount: hand.holeCards.length, communityCardsCount: hand.communityCards.length,
        usedCards: usedCards(), pendingRank, setPendingRank, onSelectCard: selectCard,
        onCancel: () => { setPickingFor(null); setPendingRank(null); },
        onConfirmCommunity: confirmCommunityCards,
    };

    if (phase === 'deal-hole') {
        return (
            <DealHoleCards botPosition={hand.botPosition} holeCards={hand.holeCards} canUndo={history.length > 0} onUndo={undo}>
                <CardSelector {...commonCardSelectorProps} />
            </DealHoleCards>
        );
    }

    if (phase === 'play') {
        return (
            <PlayPhase
                pot={hand.pot} currentBet={hand.currentBet} botPosition={hand.botPosition}
                holeCards={hand.holeCards} communityCards={hand.communityCards} street={hand.street}
                players={hand.players} currentPlayerIdx={hand.currentPlayerIdx} isLoading={hand.isLoading}
                botResponse={hand.botResponse} showQValues={showQValues} setShowQValues={setShowQValues}
                showRaiseInput={showRaiseInput} setShowRaiseInput={setShowRaiseInput} raiseInput={raiseInput}
                setRaiseInput={setRaiseInput} pickingFor={pickingFor} onOpenCommunityPicker={() => { pushHistory('Open card picker'); setPickingFor('community'); }}
                onQueryBot={() => queryBot(hand, bigBlind)} onRecordAction={recordOpponentAction}
                onUndo={undo} onEndHand={() => setPhase('hand-result')}
                canUndo={history.length > 0} undoLabel={history[history.length - 1]?.label}
            >
                <CardSelector {...commonCardSelectorProps} />
            </PlayPhase>
        );
    }

    if (phase === 'hand-result') {
        return (
            <HandResult
                resultType={resultType} setResultType={setResultType} resultAmt={resultAmt} setResultAmt={setResultAmt}
                onConfirm={finishHand} onUndo={undo}
            />
        );
    }

    return null;
}
