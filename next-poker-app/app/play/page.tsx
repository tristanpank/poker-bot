'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import SetupSize from './components/SetupSize';
import SetupDetails from './components/SetupDetails';
import DealPosition from './components/DealPosition';
import DealHoleCards from './components/DealHoleCards';
import CardSelector from './components/CardSelector';
import PlayPhase from './components/PlayPhase';

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const MODEL_VERSION = 'v21';

type Card = { rank: string; suit: string };
type PlayerState = {
    position: number;
    stack: number;
    bet: number;
    hole_cards: Card[] | null;
    is_bot: boolean;
    is_active: boolean;
    has_acted: boolean;
};
type BotResponse = {
    action: 'fold' | 'check' | 'call' | 'raise_amt';
    action_id: number | null;
    amount: number | null;
};
type Phase = 'setup-size' | 'setup-details' | 'deal-position' | 'deal-hole' | 'play';
type LegalActionState = {
    canFold: boolean;
    canCheck: boolean;
    canCall: boolean;
    canRaise: boolean;
    toCall: number;
    minRaiseTo: number | null;
    maxRaiseTo: number | null;
};
type HandState = {
    botPosition: number;
    holeCards: Card[];
    communityCards: Card[];
    players: PlayerState[];
    startingStacks: number[];
    pot: number;
    currentBet: number;
    currentPlayerIdx: number;
    botResponse: BotResponse | null;
    street: 'preflop' | 'flop' | 'turn' | 'river';
    isLoading: boolean;
};
type HistoryEntry = { phase: Phase; hand: HandState; label: string };
type ShowdownEntry = { playerIndex: number; position: number; cards: Card[]; mucked: boolean };

type BackendLegalActions = {
    actor_index: number;
    actions: Array<'fold' | 'check' | 'call' | 'raise_amt'>;
    to_call: number;
    min_raise_to: number | null;
    max_raise_to: number | null;
};
type BackendAppliedAction = {
    action: 'fold' | 'check' | 'call' | 'raise_amt';
    raise_amt: number | null;
    action_id?: number | null;
};
type BackendGameState = {
    session_id?: string | null;
    community_cards: Card[];
    pot: number;
    players: PlayerState[];
    bot_position: number;
    current_bet: number;
    big_blind: number;
    current_player_idx: number;
    model_version?: string | null;
};
type BackendStepRequest = {
    game_state: BackendGameState;
    actor: 'bot' | 'opponent';
    action?: 'fold' | 'check' | 'call' | 'raise_amt';
    raise_amt?: number;
    model_version?: string;
};
type BackendStepResponse = {
    game_state: BackendGameState;
    applied_action: BackendAppliedAction;
    legal_actions: BackendLegalActions;
    round_complete: boolean;
};
type BackendShowdownOpponent = {
    player_index: number;
    hole_cards: Card[] | null;
    mucked: boolean;
};
type BackendResolveRequest = {
    game_state: BackendGameState;
    starting_stacks: number[];
    opponents: BackendShowdownOpponent[];
};
type BackendResolveResponse = {
    result: 'won' | 'lost' | 'push';
    amount: number;
    delta: number;
    bot_payout: number;
    bot_contribution: number;
    pot: number;
    winner_indices: number[];
    next_game_state: BackendGameState;
};
type ShowdownResult = {
    result: 'won' | 'lost' | 'push';
    amount: number;
    delta: number;
};
type ResultFlash = {
    result: 'won' | 'lost' | 'push';
    delta: number;
};

const EMPTY_LEGAL_ACTIONS: LegalActionState = {
    canFold: false,
    canCheck: false,
    canCall: false,
    canRaise: false,
    toCall: 0,
    minRaiseTo: null,
    maxRaiseTo: null,
};

const EMPTY_HAND: HandState = {
    botPosition: 0,
    holeCards: [],
    communityCards: [],
    players: [],
    startingStacks: [],
    pot: 0,
    currentBet: 0,
    currentPlayerIdx: 0,
    botResponse: null,
    street: 'preflop',
    isLoading: false,
};

const suitSym = (s: string) => ({ s: '\u2660', h: '\u2665', d: '\u2666', c: '\u2663' }[s] ?? s);

function initPlayers(count: number, stack: number, botPos: number): PlayerState[] {
    return Array.from({ length: count }, (_, i) => ({
        position: i,
        stack,
        bet: 0,
        hole_cards: i === botPos ? [] : null,
        is_bot: i === botPos,
        is_active: true,
        has_acted: false,
    }));
}

function firstActivePlayerFrom(players: PlayerState[], startPos: number): number {
    const n = players.length;
    for (let offset = 0; offset < n; offset++) {
        const idx = (startPos + offset) % n;
        if (players[idx]?.is_active) return idx;
    }
    return startPos;
}

function getStreetFromBoardCount(count: number): 'preflop' | 'flop' | 'turn' | 'river' {
    if (count <= 0) return 'preflop';
    if (count <= 3) return 'flop';
    if (count === 4) return 'turn';
    return 'river';
}

function mapBackendLegalActions(legal: BackendLegalActions | null | undefined): LegalActionState {
    if (!legal) return EMPTY_LEGAL_ACTIONS;
    return {
        canFold: legal.actions.includes('fold'),
        canCheck: legal.actions.includes('check'),
        canCall: legal.actions.includes('call'),
        canRaise: legal.actions.includes('raise_amt'),
        toCall: Math.max(0, legal.to_call ?? 0),
        minRaiseTo: legal.min_raise_to ?? null,
        maxRaiseTo: legal.max_raise_to ?? null,
    };
}

function toBackendGameState(hand: HandState, bigBlind: number): BackendGameState {
    return {
        community_cards: hand.communityCards.map((c) => ({ rank: c.rank, suit: c.suit })),
        pot: hand.pot,
        players: hand.players.map((p) => ({
            position: p.position,
            stack: p.stack,
            bet: p.bet,
            hole_cards: p.is_bot ? hand.holeCards.map((c) => ({ rank: c.rank, suit: c.suit })) : null,
            is_bot: p.is_bot,
            is_active: p.is_active,
            has_acted: p.has_acted,
        })),
        bot_position: hand.botPosition,
        current_bet: hand.currentBet,
        big_blind: bigBlind,
        current_player_idx: hand.currentPlayerIdx,
    };
}

function mapBackendGameState(gameState: BackendGameState, prev: HandState): HandState {
    const botPlayer = gameState.players.find((p) => p.is_bot);
    const holeCards = botPlayer?.hole_cards ?? prev.holeCards;
    const communityCards = gameState.community_cards ?? [];
    return {
        ...prev,
        botPosition: gameState.bot_position,
        holeCards,
        communityCards,
        players: gameState.players,
        startingStacks: prev.startingStacks,
        pot: gameState.pot,
        currentBet: gameState.current_bet,
        currentPlayerIdx: gameState.current_player_idx,
        street: getStreetFromBoardCount(communityCards.length),
    };
}

function mapBackendStateToNextHand(gameState: BackendGameState): HandState {
    const players = gameState.players.map((player) => ({
        ...player,
        bet: 0,
        hole_cards: player.is_bot ? [] : null,
        has_acted: false,
        is_active: player.stack > 0,
    }));
    const botPlayer = players.find((player) => player.is_bot);

    return {
        ...EMPTY_HAND,
        botPosition: gameState.bot_position,
        players,
        startingStacks: players.map((player) => player.stack),
        currentPlayerIdx: players.findIndex((player) => player.is_active),
        holeCards: botPlayer?.hole_cards ?? [],
    };
}

function mapAppliedActionToBotResponse(applied: BackendAppliedAction): BotResponse {
    return {
        action: applied.action,
        action_id: applied.action_id ?? null,
        amount: applied.raise_amt ?? null,
    };
}

async function parseError(res: Response): Promise<string> {
    try {
        const data = await res.json();
        if (typeof data?.detail === 'string') return data.detail;
        return JSON.stringify(data);
    } catch {
        return await res.text();
    }
}

export default function PlayPage() {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    const [tableSize, setTableSize] = useState(6);
    const [smallBlind, setSmallBlind] = useState(1);
    const [bigBlind, setBigBlind] = useState(2);
    const [buyIn, setBuyIn] = useState(200);
    const [sessionStacks, setSessionStacks] = useState<number[]>([]);
    const [sessionProfit, setSessionProfit] = useState(0);

    const [phase, setPhase] = useState<Phase>('setup-size');
    const [hand, setHand] = useState<HandState>(EMPTY_HAND);

    const [pickingFor, setPickingFor] = useState<'hole' | 'community' | 'showdown' | null>(null);
    const [pendingRank, setPendingRank] = useState<string | null>(null);

    const [history, setHistory] = useState<HistoryEntry[]>([]);

    const [raiseInput, setRaiseInput] = useState('');
    const [showRaiseInput, setShowRaiseInput] = useState(false);

    const [legalActions, setLegalActions] = useState<LegalActionState>(EMPTY_LEGAL_ACTIONS);

    const [showdownEntries, setShowdownEntries] = useState<ShowdownEntry[]>([]);
    const [showdownResult, setShowdownResult] = useState<ShowdownResult | null>(null);
    const [showdownError, setShowdownError] = useState<string | null>(null);
    const [isResolvingShowdown, setIsResolvingShowdown] = useState(false);
    const [isShowdownMode, setIsShowdownMode] = useState(false);
    const [resultFlash, setResultFlash] = useState<ResultFlash | null>(null);

    const legalRequestSeq = useRef(0);
    const nextHandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const autoResolveSingleLeftRef = useRef(false);

    useEffect(() => () => {
        if (nextHandTimerRef.current) {
            clearTimeout(nextHandTimerRef.current);
            nextHandTimerRef.current = null;
        }
    }, []);

    const usedCards = useCallback((): Set<string> => {
        const set = new Set<string>();
        hand.holeCards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        hand.communityCards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        showdownEntries.forEach((entry) => {
            entry.cards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        });
        return set;
    }, [hand.holeCards, hand.communityCards, showdownEntries]);

    const pushHistory = useCallback((label: string) => {
        setHistory((prev) => [...prev, { phase, hand: JSON.parse(JSON.stringify(hand)), label }]);
    }, [phase, hand]);

    const undo = useCallback(() => {
        setHistory((prev) => {
            if (prev.length === 0) return prev;
            const last = prev[prev.length - 1];
            if (nextHandTimerRef.current) {
                clearTimeout(nextHandTimerRef.current);
                nextHandTimerRef.current = null;
            }
            autoResolveSingleLeftRef.current = false;
            setPhase(last.phase);
            setHand(last.hand);
            setPickingFor(null);
            setPendingRank(null);
            setShowRaiseInput(false);
            setShowdownError(null);
            setShowdownEntries([]);
            setShowdownResult(null);
            setIsShowdownMode(false);
            setResultFlash(null);
            return prev.slice(0, -1);
        });
    }, []);

    const fetchLegalActions = useCallback(async (currentHand: HandState, silent = false) => {
        if (
            currentHand.currentPlayerIdx < 0 ||
            currentHand.players.length === 0 ||
            !currentHand.players[currentHand.currentPlayerIdx] ||
            !currentHand.players[currentHand.currentPlayerIdx].is_active
        ) {
            setLegalActions(EMPTY_LEGAL_ACTIONS);
            return;
        }

        const seq = ++legalRequestSeq.current;
        try {
            const res = await fetch(`${BACKEND}/poker/legal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(toBackendGameState(currentHand, bigBlind)),
            });
            if (!res.ok) {
                const detail = await parseError(res);
                throw new Error(`API error: ${res.status} ${detail}`);
            }
            const data: BackendLegalActions = await res.json();
            if (seq === legalRequestSeq.current) {
                setLegalActions(mapBackendLegalActions(data));
            }
        } catch (err) {
            console.error('Legal actions query failed:', err);
            if (!silent && seq === legalRequestSeq.current) {
                setLegalActions(EMPTY_LEGAL_ACTIONS);
            }
        }
    }, [bigBlind]);

    const stepAction = useCallback(async (
        currentHand: HandState,
        request: Omit<BackendStepRequest, 'game_state'>,
    ): Promise<BackendStepResponse> => {
        const payload: BackendStepRequest = {
            ...request,
            game_state: toBackendGameState(currentHand, bigBlind),
        };
        const res = await fetch(`${BACKEND}/poker/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const detail = await parseError(res);
            throw new Error(`API error: ${res.status} ${detail}`);
        }
        return await res.json();
    }, [bigBlind]);

    const queryBot = useCallback(async (currentHand: HandState) => {
        const actorIdx = currentHand.currentPlayerIdx;
        const actor = currentHand.players[actorIdx];
        if (!actor || !actor.is_bot || !actor.is_active) return;

        setHand((prev) => ({ ...prev, isLoading: true }));
        try {
            const data = await stepAction(currentHand, { actor: 'bot', model_version: MODEL_VERSION });
            setHand((prev) => ({
                ...mapBackendGameState(data.game_state, prev),
                isLoading: false,
                botResponse: mapAppliedActionToBotResponse(data.applied_action),
            }));
            setLegalActions(mapBackendLegalActions(data.legal_actions));
        } catch (err) {
            console.error('Bot step failed:', err);
            setHand((prev) => ({ ...prev, isLoading: false }));
            await fetchLegalActions(currentHand, true);
        }
    }, [fetchLegalActions, stepAction]);

    const selectCard = useCallback((rank: string, suit: string) => {
        const card: Card = { rank, suit };
        pushHistory(`Select ${rank}${suitSym(suit)}`);

        if (pickingFor === 'hole') {
            const newHole = [...hand.holeCards, card];
            setHand((prev) => ({ ...prev, holeCards: newHole }));
            if (newHole.length >= 2) {
                setPickingFor(null);
                setPhase('play');

                const players = hand.players.map((p) => ({ ...p, bet: 0, has_acted: false }));
                const n = players.length;
                let postedPot = 0;
                let bbIdx = n > 1 ? 1 : 0;

                const postBlind = (idx: number, amount: number) => {
                    if (idx < 0 || idx >= n) return;
                    const posted = Math.min(amount, players[idx].stack);
                    players[idx].bet += posted;
                    players[idx].stack -= posted;
                    postedPot += posted;
                };

                if (n > 2) {
                    const sbIdx = firstActivePlayerFrom(players, 1);
                    bbIdx = firstActivePlayerFrom(players, (sbIdx + 1) % n);
                    postBlind(sbIdx, smallBlind);
                    postBlind(bbIdx, bigBlind);
                } else if (n === 2) {
                    const sbIdx = 0;
                    bbIdx = 1;
                    postBlind(sbIdx, smallBlind);
                    postBlind(bbIdx, bigBlind);
                }

                const currentBet = Math.max(0, players[bbIdx]?.bet ?? 0);
                const firstToAct = n > 1 ? firstActivePlayerFrom(players, (bbIdx + 1) % n) : 0;

                setHand((prev) => ({
                    ...prev,
                    holeCards: newHole,
                    pot: postedPot,
                    currentBet,
                    players,
                    currentPlayerIdx: firstToAct,
                    botResponse: null,
                }));
                setIsShowdownMode(false);
                setShowdownEntries([]);
                setShowdownResult(null);
                setShowdownError(null);
                setLegalActions(EMPTY_LEGAL_ACTIONS);
            }
        } else if (pickingFor === 'community') {
            const newComm = [...hand.communityCards, card];
            const street = getStreetFromBoardCount(newComm.length);
            setHand((prev) => ({ ...prev, communityCards: newComm, street }));
        }

        setPendingRank(null);
    }, [bigBlind, hand, pickingFor, pushHistory, smallBlind]);

    const confirmCommunityCards = useCallback(() => {
        pushHistory('Confirm dealt cards');
        setPickingFor(null);
        setShowRaiseInput(false);
        setRaiseInput('');

        const street = getStreetFromBoardCount(hand.communityCards.length);
        const players = hand.players.map((p) => ({ ...p, bet: 0, has_acted: false }));
        const firstToAct = firstActivePlayerFrom(players, 1);

        setHand((prev) => ({
            ...prev,
            street,
            players,
            currentBet: 0,
            currentPlayerIdx: firstToAct,
            botResponse: null,
        }));
        setIsShowdownMode(false);
        setShowdownEntries([]);
        setShowdownResult(null);
        setShowdownError(null);
        setLegalActions(EMPTY_LEGAL_ACTIONS);
    }, [hand.communityCards.length, hand.players, pushHistory]);

    const recordOpponentAction = useCallback(async (action: 'fold' | 'check_call' | 'raise', raiseAmt?: number) => {
        const playerIdx = hand.currentPlayerIdx;
        const player = hand.players[playerIdx];
        if (!player || player.is_bot || !player.is_active) return;

        let backendAction: 'fold' | 'check' | 'call' | 'raise_amt';
        let backendRaiseAmt: number | undefined;

        if (action === 'fold') {
            backendAction = 'fold';
        } else if (action === 'check_call') {
            backendAction = legalActions.canCall ? 'call' : 'check';
        } else {
            const parsed = Number(raiseAmt);
            if (!Number.isFinite(parsed)) return;
            backendAction = 'raise_amt';
            backendRaiseAmt = Math.trunc(parsed);
        }

        pushHistory(`P${player.position} ${backendAction}${backendRaiseAmt ? ` ${backendRaiseAmt}` : ''}`);
        setHand((prev) => ({ ...prev, isLoading: true }));

        try {
            const data = await stepAction(hand, {
                actor: 'opponent',
                action: backendAction,
                raise_amt: backendRaiseAmt,
            });
            setHand((prev) => ({
                ...mapBackendGameState(data.game_state, prev),
                isLoading: false,
            }));
            setLegalActions(mapBackendLegalActions(data.legal_actions));
        } catch (err) {
            console.error('Opponent step failed:', err);
            setHand((prev) => ({ ...prev, isLoading: false }));
            await fetchLegalActions(hand, true);
        } finally {
            setShowRaiseInput(false);
            setRaiseInput('');
        }
    }, [fetchLegalActions, hand, legalActions.canCall, pushHistory, stepAction]);

    const beginShowdown = useCallback(() => {
        const entries = hand.players
            .map((player, index) => ({ player, index }))
            .filter(({ player }) => player.is_active && !player.is_bot)
            .map(({ player, index }) => ({
                playerIndex: index,
                position: player.position,
                cards: [],
                mucked: false,
            }));
        setShowdownEntries(entries);
        setPendingRank(null);
        setShowdownResult(null);
        setShowdownError(null);
        setIsShowdownMode(true);
        setPickingFor(entries.length > 0 ? 'showdown' : null);
        setResultFlash(null);
    }, [hand.players]);

    const currentShowdownEntry = showdownEntries.find((entry) => !entry.mucked && entry.cards.length < 2) ?? null;

    const showdownUsedCards = useCallback(() => {
        const set = new Set<string>();
        hand.holeCards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        hand.communityCards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        showdownEntries.forEach((entry) => {
            entry.cards.forEach((c) => set.add(`${c.rank}${c.suit}`));
        });
        return set;
    }, [hand.communityCards, hand.holeCards, showdownEntries]);

    const selectShowdownCard = useCallback((rank: string, suit: string) => {
        if (!currentShowdownEntry) return;
        const cardId = `${rank}${suit}`;
        if (showdownUsedCards().has(cardId)) return;

        setShowdownEntries((prev) => prev.map((entry) => {
            if (entry.playerIndex !== currentShowdownEntry.playerIndex) return entry;
            if (entry.mucked || entry.cards.length >= 2) return entry;
            return { ...entry, cards: [...entry.cards, { rank, suit }] };
        }));
        setPendingRank(null);
        setShowdownError(null);
    }, [currentShowdownEntry, showdownUsedCards]);

    const muckCurrentShowdownPlayer = useCallback(() => {
        if (!currentShowdownEntry) return;
        setShowdownEntries((prev) => prev.map((entry) => (
            entry.playerIndex === currentShowdownEntry.playerIndex
                ? { ...entry, cards: [], mucked: true }
                : entry
        )));
        setPendingRank(null);
        setShowdownError(null);
    }, [currentShowdownEntry]);

    const clearCurrentShowdownPlayer = useCallback(() => {
        if (!currentShowdownEntry) return;
        setShowdownEntries((prev) => prev.map((entry) => (
            entry.playerIndex === currentShowdownEntry.playerIndex
                ? { ...entry, cards: [], mucked: false }
                : entry
        )));
        setPendingRank(null);
        setShowdownError(null);
    }, [currentShowdownEntry]);

    const applyResolvedHand = useCallback((data: BackendResolveResponse, showShowdownResult: boolean) => {
        if (showShowdownResult) {
            setShowdownResult({ result: data.result, amount: data.amount, delta: data.delta });
        } else {
            setShowdownResult(null);
        }
        setResultFlash({ result: data.result, delta: data.delta });

        if (nextHandTimerRef.current) {
            clearTimeout(nextHandTimerRef.current);
        }
        nextHandTimerRef.current = setTimeout(() => {
            const nextHand = mapBackendStateToNextHand(data.next_game_state);
            const nextBotStack = nextHand.players.find((player) => player.is_bot)?.stack ?? 0;

            setSessionProfit((prev) => prev + data.delta);
            setSessionStacks((prev) => {
                const next = [...prev];
                next[0] = nextBotStack;
                return next.length > 0 ? next : [nextBotStack];
            });

            setHistory([]);
            setHand(nextHand);
            setPhase('deal-hole');
            setPickingFor('hole');
            setPendingRank(null);
            setShowRaiseInput(false);
            setRaiseInput('');
            setShowdownEntries([]);
            setShowdownResult(null);
            setShowdownError(null);
            setIsResolvingShowdown(false);
            setIsShowdownMode(false);
            setLegalActions(EMPTY_LEGAL_ACTIONS);
            setResultFlash(null);
            autoResolveSingleLeftRef.current = false;
            nextHandTimerRef.current = null;
        }, 1000);
    }, []);

    const resolveHand = useCallback(async () => {
        if (isResolvingShowdown) return;
        const canResolve = showdownEntries.every((entry) => entry.mucked || entry.cards.length === 2);
        if (!canResolve) {
            setShowdownError('Enter both cards or muck for each active opponent.');
            return;
        }

        setIsResolvingShowdown(true);
        setShowdownError(null);
        try {
            const payload: BackendResolveRequest = {
                game_state: toBackendGameState(hand, bigBlind),
                starting_stacks: hand.startingStacks,
                opponents: showdownEntries.map((entry) => ({
                    player_index: entry.playerIndex,
                    hole_cards: entry.mucked ? null : entry.cards,
                    mucked: entry.mucked,
                })),
            };
            const res = await fetch(`${BACKEND}/poker/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const detail = await parseError(res);
                throw new Error(`API error: ${res.status} ${detail}`);
            }
            const data: BackendResolveResponse = await res.json();
            applyResolvedHand(data, true);
        } catch (err) {
            console.error('Hand resolve failed:', err);
            setShowdownError(err instanceof Error ? err.message : 'Failed to resolve hand');
        } finally {
            setIsResolvingShowdown(false);
        }
    }, [applyResolvedHand, bigBlind, hand, isResolvingShowdown, showdownEntries]);

    const autoResolveSingleLeft = useCallback(async () => {
        if (isResolvingShowdown) return;
        setIsResolvingShowdown(true);
        setShowdownError(null);
        try {
            const payload: BackendResolveRequest = {
                game_state: toBackendGameState(hand, bigBlind),
                starting_stacks: hand.startingStacks,
                opponents: [],
            };
            const res = await fetch(`${BACKEND}/poker/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const detail = await parseError(res);
                throw new Error(`API error: ${res.status} ${detail}`);
            }
            const data: BackendResolveResponse = await res.json();
            applyResolvedHand(data, false);
        } catch (err) {
            console.error('Auto resolve failed:', err);
            setShowdownError(err instanceof Error ? err.message : 'Failed to auto resolve hand');
            autoResolveSingleLeftRef.current = false;
        } finally {
            setIsResolvingShowdown(false);
        }
    }, [applyResolvedHand, bigBlind, hand, isResolvingShowdown]);

    const startNewHand = useCallback(() => {
        const baseStack = sessionStacks[0] ?? buyIn;
        if (nextHandTimerRef.current) {
            clearTimeout(nextHandTimerRef.current);
            nextHandTimerRef.current = null;
        }
        autoResolveSingleLeftRef.current = false;
        setHistory([]);
        setPhase('deal-position');
        setHand({
            ...EMPTY_HAND,
            players: initPlayers(tableSize, baseStack, 0),
            startingStacks: Array(tableSize).fill(baseStack),
        });
        setPickingFor(null);
        setPendingRank(null);
        setShowRaiseInput(false);
        setRaiseInput('');
        setShowdownEntries([]);
        setShowdownResult(null);
        setShowdownError(null);
        setIsResolvingShowdown(false);
        setIsShowdownMode(false);
        setLegalActions(EMPTY_LEGAL_ACTIONS);
        setResultFlash(null);
    }, [buyIn, sessionStacks, tableSize]);

    useEffect(() => {
        if (phase !== 'play' || hand.isLoading || pickingFor !== null) return;
        void fetchLegalActions(hand, true);
    }, [fetchLegalActions, hand, phase, pickingFor]);

    useEffect(() => {
        if (phase !== 'play') {
            autoResolveSingleLeftRef.current = false;
            return;
        }
        if (isShowdownMode) return;
        if (pickingFor !== null) return;
        if (hand.currentPlayerIdx !== -1) return;

        const activePlayers = hand.players.filter((player) => player.is_active).length;
        if (activePlayers !== 1) return;
        if (autoResolveSingleLeftRef.current) return;

        autoResolveSingleLeftRef.current = true;
        void autoResolveSingleLeft();
    }, [autoResolveSingleLeft, hand.currentPlayerIdx, hand.players, isShowdownMode, phase, pickingFor]);

    useEffect(() => {
        if (phase !== 'play') return;
        if (isShowdownMode) return;
        if (pickingFor !== null) return;
        if (hand.players.filter((player) => player.is_active).length <= 1) return;
        if (hand.street !== 'river') return;
        if (hand.communityCards.length !== 5) return;
        if (hand.currentPlayerIdx !== -1) return;
        beginShowdown();
    }, [beginShowdown, hand.communityCards.length, hand.currentPlayerIdx, hand.players, hand.street, isShowdownMode, phase, pickingFor]);

    useEffect(() => {
        if (!isShowdownMode) return;
        setPendingRank(null);
        if (currentShowdownEntry) {
            setPickingFor('showdown');
        } else {
            setPickingFor(null);
        }
    }, [currentShowdownEntry, isShowdownMode]);

    const showdownCanResolve = showdownEntries.every((entry) => entry.mucked || entry.cards.length === 2);

    if (!mounted) return null;

    if (phase === 'setup-size') {
        return (
            <SetupSize
                tableSize={tableSize}
                setTableSize={setTableSize}
                onContinue={() => setPhase('setup-details')}
            />
        );
    }

    if (phase === 'setup-details') {
        return (
            <SetupDetails
                hasSession={sessionStacks.length > 0}
                sessionStacks={sessionStacks}
                sessionProfit={sessionProfit}
                smallBlind={smallBlind}
                setSmallBlind={setSmallBlind}
                bigBlind={bigBlind}
                setBigBlind={setBigBlind}
                buyIn={buyIn}
                setBuyIn={setBuyIn}
                onBack={() => setPhase('setup-size')}
                onStart={() => {
                    if (sessionStacks.length === 0) {
                        setSessionStacks(Array(tableSize).fill(buyIn));
                    }
                    startNewHand();
                }}
                onEnd={() => {
                    setSessionStacks([]);
                    setSessionProfit(0);
                    setPhase('setup-size');
                }}
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
                    setHand((prev) => ({
                        ...prev,
                        botPosition: i,
                        players,
                        startingStacks: players.map((p) => p.stack),
                        currentPlayerIdx: 0,
                    }));
                    setPhase('deal-hole');
                    setPickingFor('hole');
                }}
                canUndo={history.length > 0}
                onUndo={undo}
            />
        );
    }

    const commonCardSelectorProps = {
        pickingFor,
        holeCardsCount: pickingFor === 'showdown' ? (currentShowdownEntry?.cards.length ?? 0) : hand.holeCards.length,
        communityCardsCount: hand.communityCards.length,
        usedCards: usedCards(),
        pendingRank,
        setPendingRank,
        onSelectCard: (rank: string, suit: string) => {
            if (pickingFor === 'showdown') {
                selectShowdownCard(rank, suit);
            } else {
                selectCard(rank, suit);
            }
        },
        onCancel: () => {
            if (pickingFor !== 'showdown') {
                setPickingFor(null);
            }
            setPendingRank(null);
        },
        onConfirmCommunity: confirmCommunityCards,
    };

    if (phase === 'deal-hole') {
        return (
            <DealHoleCards
                botPosition={hand.botPosition}
                holeCards={hand.holeCards}
                players={hand.players}
                canUndo={history.length > 0}
                onUndo={undo}
            >
                <CardSelector {...commonCardSelectorProps} />
            </DealHoleCards>
        );
    }

    if (phase === 'play') {
        const canOpenCommunityPicker = (
            hand.currentPlayerIdx === -1
            && (
                (hand.street === 'preflop' && hand.communityCards.length === 0)
                || (hand.street === 'flop' && hand.communityCards.length === 3)
                || (hand.street === 'turn' && hand.communityCards.length === 4)
            )
        );
        return (
            <PlayPhase
                pot={hand.pot}
                currentBet={hand.currentBet}
                bigBlind={bigBlind}
                botPosition={hand.botPosition}
                holeCards={hand.holeCards}
                communityCards={hand.communityCards}
                street={hand.street}
                players={hand.players}
                currentPlayerIdx={hand.currentPlayerIdx}
                isLoading={hand.isLoading}
                botResponse={hand.botResponse}
                showRaiseInput={showRaiseInput}
                setShowRaiseInput={setShowRaiseInput}
                raiseInput={raiseInput}
                setRaiseInput={setRaiseInput}
                onOpenCommunityPicker={() => {
                    if (!canOpenCommunityPicker) return;
                    pushHistory('Open card picker');
                    setPickingFor('community');
                }}
                onQueryBot={() => queryBot(hand)}
                onRecordAction={recordOpponentAction}
                onUndo={undo}
                canUndo={history.length > 0}
                undoLabel={history[history.length - 1]?.label}
                legalActions={legalActions}
                showdownMode={isShowdownMode}
                showdownEntries={showdownEntries}
                currentShowdownPlayerIndex={currentShowdownEntry?.playerIndex ?? null}
                showdownCanResolve={showdownCanResolve}
                isResolvingShowdown={isResolvingShowdown}
                showdownError={showdownError}
                showdownResult={showdownResult}
                resultFlash={resultFlash}
                onMuckShowdown={muckCurrentShowdownPlayer}
                onClearShowdown={clearCurrentShowdownPlayer}
                onResolveShowdown={resolveHand}
            >
                <CardSelector {...commonCardSelectorProps} />
            </PlayPhase>
        );
    }

    return null;
}
