from __future__ import annotations

import copy
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from pokerkit import Automation, Card, NoLimitTexasHoldem

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
MODELS_DIR = os.path.join(SRC_ROOT, "models")
for _path in (FEATURES_DIR, MODELS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from poker_state_v24 import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_COUNT_V21,
    ACTION_FOLD,
    ACTIVE_PLAYERS_SLICE_V24,
    ALL_RAISE_ACTIONS,
    BOARD_TEXTURE_SLICE_V24,
    DRAW_FLAGS_SLICE_V24,
    FACING_BET_BUCKET_SLICE_V24,
    HAND_CLASS_SLICE_V24,
    IDX_IN_POSITION_FLAG_V24,
    IDX_LAST_AGGRESSOR_FLAG_V24,
    NON_ALL_IN_RAISE_ACTIONS,
    PLAYERS_BEHIND_SLICE_V24,
    POSITION_SLICE_V24,
    POT_BUCKET_SLICE_V24,
    PREPERCENTILE_SLICE_V24,
    RAISE_COUNT_SLICE_V24,
    SPR_BUCKET_SLICE_V24,
    STREET_SLICE_V24,
    TO_CALL_BUCKET_SLICE_V24,
    abstract_raise_target,
    build_legal_action_mask,
    encode_info_state,
    estimate_preflop_strength,
    flatten_cards_list,
    street_from_board_len,
)
from poker_model_v24 import PokerDeepCFRNet, masked_policy
from preflop_blueprint_v24 import canonical_preflop_hand_key
from tabular_policy_v24 import (
    TabularNode,
    TabularPolicySnapshot,
    average_policy,
    deserialize_node_store,
    freeze_policy_snapshot,
    normalize_masked_policy,
    regret_matching,
    serialize_node_store,
    uniform_legal_policy,
)

SYNTHETIC_OPPONENT_STYLES = ("nit", "overfolder", "overcaller", "over3better", "station", "maniac")
RUNTIME_POLICY_DEFAULTS_V24 = {
    "algorithm_name": "tabular_mccfr_6max",
    "evaluation_mode": "heuristics",
    "eval_hero_seat": 0,
    "checkpoint_pool": (),
    "synthetic_opponent_style": "",
    "current_iteration": 0,
    "parallel_rollouts": False,
    "max_checkpoint_pool": 32,
}
POSTFLOP_CONDITION_STREET_KEYS = ("flop", "turn", "river")
POSTFLOP_CONDITION_RATE_COUNT_KEYS = {
    "check_when_legal": ("check_when_legal_hits", "check_when_legal_opportunities"),
    "bet_raise_when_checked_to": ("bet_raise_when_checked_to_hits", "bet_raise_when_checked_to_opportunities"),
    "aggressive_when_checked_to": ("aggressive_when_checked_to_hits", "aggressive_when_checked_to_opportunities"),
    "fold_when_facing_bet": ("fold_when_facing_bet_hits", "fold_when_facing_bet_opportunities"),
    "call_when_facing_bet": ("call_when_facing_bet_hits", "call_when_facing_bet_opportunities"),
    "raise_when_facing_bet": ("raise_when_facing_bet_hits", "raise_when_facing_bet_opportunities"),
}
PREFLOP_STAT_KEYS = (
    "hands_played",
    "preflop_opportunities",
    "vpip_counts",
    "pfr_counts",
    "three_bet_counts",
    "faced_open_opportunities",
    "fold_vs_open_counts",
    "call_vs_open_counts",
    "faced_three_bet_opportunities",
    "fold_vs_three_bet_counts",
    "squeeze_opportunities",
    "squeeze_counts",
    "faced_cbet_flop_opportunities",
    "fold_vs_cbet_flop_counts",
    "faced_cbet_turn_opportunities",
    "fold_vs_cbet_turn_counts",
    "aggression_opportunities",
    "aggression_counts",
    "fold_preflop_counts",
)
_WORKER_BASE_SNAPSHOT_SIGNATURE: Optional[str] = None
_WORKER_BASE_NODE_STORE: Dict[str, TabularNode] = {}


def _new_preflop_stats(num_players: int) -> Dict[str, List[int]]:
    stats = {key: [0] * max(1, int(num_players)) for key in PREFLOP_STAT_KEYS}
    stats["hands_seen_flags"] = [0] * max(1, int(num_players))
    return stats


def _new_action_histogram() -> np.ndarray:
    return np.zeros(ACTION_COUNT_V21, dtype=np.int64)


def _new_postflop_conditioned_counts() -> Dict[str, Dict[str, int]]:
    return {
        street_key: {
            count_key: 0
            for hit_key, opp_key in POSTFLOP_CONDITION_RATE_COUNT_KEYS.values()
            for count_key in (hit_key, opp_key)
        }
        for street_key in POSTFLOP_CONDITION_STREET_KEYS
    }


def build_runtime_policy_config(config: Optional[object] = None) -> SimpleNamespace:
    if config is None:
        payload: Dict[str, object] = {}
    elif isinstance(config, dict):
        payload = dict(config)
    else:
        payload = {key: value for key, value in vars(config).items() if not str(key).startswith("_")}
    merged = dict(RUNTIME_POLICY_DEFAULTS_V24)
    merged.update(payload)
    merged["parallel_rollouts"] = False
    return SimpleNamespace(**merged)


@dataclass
class HandContext:
    starting_stacks: List[int]
    big_blind: int
    small_blind: int
    in_hand: List[bool]
    contributions: List[float]
    hole_cards_by_seat: List[List[Card]] = field(default_factory=list)
    deck_order: List[Card] = field(default_factory=list)
    remaining_deck: List[Card] = field(default_factory=list)
    dealt_burn_cards: List[Card] = field(default_factory=list)
    dealt_board_cards: List[Card] = field(default_factory=list)
    action_history: List[tuple[int, int, bool]] = field(default_factory=list)
    current_street: int = 0
    street_raise_count: int = 0
    preflop_raise_count: int = 0
    preflop_call_count: int = 0
    preflop_opened: bool = False
    preflop_last_raiser: Optional[int] = None
    last_aggressor: Optional[int] = None
    last_action_was_all_in: bool = False
    cbet_flop_initiator: Optional[int] = None
    cbet_turn_initiator: Optional[int] = None
    total_actions: int = 0
    preflop_actions: int = 0
    flop_seen: bool = False
    turn_seen: bool = False
    river_seen: bool = False


@dataclass
class TraversalResult:
    utility_bb: float = 0.0
    unclipped_utility_bb: float = 0.0
    traverser_seat: int = 0
    monitor_sampled: bool = False
    traverser_decisions: int = 0
    action_counts: np.ndarray = field(default_factory=_new_action_histogram)
    preflop_action_counts: np.ndarray = field(default_factory=_new_action_histogram)
    postflop_action_counts: np.ndarray = field(default_factory=_new_action_histogram)
    postflop_conditioned_counts: Dict[str, Dict[str, int]] = field(default_factory=_new_postflop_conditioned_counts)
    invalid_state_count: int = 0
    invalid_action_count: int = 0
    vpip: bool = False
    pfr: bool = False
    three_bet: bool = False
    preflop_jam: bool = False
    flop_seen: bool = False
    total_actions: int = 0
    preflop_actions: int = 0
    blueprint_decisions: int = 0
    preflop_decisions: int = 0
    preflop_stats: Dict[str, List[int]] = field(default_factory=dict)
    perf_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class TraversalBatchResult:
    results: List[TraversalResult] = field(default_factory=list)
    node_deltas: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class HandResult:
    hero_profit_bb: float
    hero_seat: int
    action_counts: np.ndarray
    preflop_action_counts: np.ndarray
    postflop_action_counts: np.ndarray
    postflop_conditioned_counts: Dict[str, Dict[str, int]]
    illegal_action_count: int
    win: bool
    vpip: bool
    pfr: bool
    three_bet: bool
    preflop_jam: bool
    flop_seen: bool
    turn_seen: bool
    river_seen: bool
    showdown_seen: bool
    showdown_won: bool
    total_actions: int
    preflop_actions: int
    postflop_actions: int
    blueprint_decisions: int
    preflop_decisions: int
    cbet_flop_opportunity: bool
    cbet_flop_taken: bool
    fold_vs_cbet_flop_opportunity: bool
    fold_vs_cbet_flop: bool
    cbet_turn_opportunity: bool
    cbet_turn_taken: bool
    fold_vs_cbet_turn_opportunity: bool
    fold_vs_cbet_turn: bool
    rfi_opportunity: bool
    rfi_attempt: bool
    hero_hand_key: Optional[str] = None
    preflop_stats: Dict[str, List[int]] = field(default_factory=dict)


def _new_perf_breakdown() -> Dict[str, float]:
    return {
        "state_init_time": 0.0,
        "chance_time": 0.0,
        "traverser_state_time": 0.0,
        "opponent_state_time": 0.0,
        "regret_infer_time": 0.0,
        "strategy_infer_time": 0.0,
        "branch_clone_time": 0.0,
        "apply_time": 0.0,
    }


def _record_action_history(hand_ctx: HandContext, actor: int, action_id: int, is_legacy: bool) -> None:
    hand_ctx.action_history.append((int(actor), int(action_id), bool(is_legacy)))
    hand_ctx.total_actions += 1
    if int(hand_ctx.current_street) == 0:
        hand_ctx.preflop_actions += 1


def _sample_stacks(rng: random.Random, config) -> List[int]:
    bb = int(getattr(config, "big_blind", 10))
    return [int(max(90.0, min(110.0, rng.gauss(100.0, 6.0))) * bb) for _ in range(int(getattr(config, "num_players", 6)))]


def _shuffled_deck(rng: random.Random) -> List[Card]:
    specs = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"]
    rng.shuffle(specs)
    return [list(Card.parse(spec))[0] for spec in specs]


def _create_state_and_context(rng: random.Random, config):
    stacks = _sample_stacks(rng, config)
    state = NoLimitTexasHoldem.create_state(
        automations=(
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
            Automation.CARD_BURNING,
        ),
        ante_trimming_status=True,
        raw_antes={-1: 0},
        raw_blinds_or_straddles=(int(getattr(config, "small_blind", 5)), int(getattr(config, "big_blind", 10))),
        min_bet=int(getattr(config, "big_blind", 10)),
        raw_starting_stacks=stacks,
        player_count=int(getattr(config, "num_players", 6)),
    )
    deck = _shuffled_deck(rng)
    deck_order = list(deck)
    state.deck_cards = deque(deck_order)
    player_count = int(getattr(config, "num_players", 6))
    hole_cards_by_seat: List[List[Card]] = [[] for _ in range(player_count)]
    while state.can_deal_hole():
        seat = int(getattr(state, "hole_dealee_index", 0))
        card = deck.pop(0)
        state.deal_hole(cards=(card,), player_index=seat)
        hole_cards_by_seat[seat].append(card)
    hand_ctx = HandContext(
        starting_stacks=list(stacks),
        big_blind=int(getattr(config, "big_blind", 10)),
        small_blind=int(getattr(config, "small_blind", 5)),
        in_hand=[True] * int(getattr(config, "num_players", 6)),
        contributions=[float(start - stack) for start, stack in zip(stacks, state.stacks)],
        hole_cards_by_seat=hole_cards_by_seat,
        deck_order=deck_order,
        remaining_deck=list(deck),
    )
    return state, hand_ctx


def _record_board_progress(state, hand_ctx: HandContext, previous_board_len: int) -> None:
    board_cards = flatten_cards_list(state.board_cards)
    hand_ctx.current_street = street_from_board_len(len(board_cards))
    hand_ctx.street_raise_count = 0
    hand_ctx.flop_seen = hand_ctx.flop_seen or hand_ctx.current_street >= 1
    hand_ctx.turn_seen = hand_ctx.turn_seen or hand_ctx.current_street >= 2
    hand_ctx.river_seen = hand_ctx.river_seen or hand_ctx.current_street >= 3
    if hand_ctx.current_street == 1:
        hand_ctx.cbet_flop_initiator = None
    elif hand_ctx.current_street == 2:
        hand_ctx.cbet_turn_initiator = None
    if len(board_cards) > int(previous_board_len):
        hand_ctx.dealt_board_cards.extend(board_cards[int(previous_board_len) :])


def _record_burn_progress(state, hand_ctx: HandContext) -> None:
    burn_cards = list(getattr(state, "burn_cards", ()))
    recorded = len(getattr(hand_ctx, "dealt_burn_cards", ()))
    if len(burn_cards) > recorded:
        hand_ctx.dealt_burn_cards.extend(burn_cards[recorded:])


def _burn_pending_card(state, hand_ctx: HandContext, target_burn_cards: Optional[Sequence[Card]] = None) -> bool:
    if not bool(getattr(state, "card_burning_status", False)):
        return False
    burn_idx = len(getattr(state, "burn_cards", ()))
    burn_card = None if target_burn_cards is None or burn_idx >= len(target_burn_cards) else target_burn_cards[burn_idx]
    if burn_card is None:
        state.burn_card()
        hand_ctx.remaining_deck = list(getattr(state, "deck_cards", ()))
    else:
        state.burn_card(card=burn_card)
    _record_burn_progress(state, hand_ctx)
    return True


def _advance_chance_nodes(state, hand_ctx: HandContext) -> None:
    while state.status:
        _record_burn_progress(state, hand_ctx)
        if _burn_pending_card(state, hand_ctx):
            continue
        if not state.can_deal_board():
            break
        previous = len(flatten_cards_list(state.board_cards))
        state.deal_board()
        hand_ctx.remaining_deck = list(getattr(state, "deck_cards", ()))
        _record_board_progress(state, hand_ctx, previous)


def _safe_utility_bb(state, traverser: int, hand_ctx: HandContext) -> float:
    return (float(state.stacks[int(traverser)]) - float(hand_ctx.starting_stacks[int(traverser)])) / float(hand_ctx.big_blind)


def _sample_action(probs: np.ndarray, rng: random.Random) -> int:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    total = float(arr.sum())
    if total <= 0.0:
        return int(np.argmax(arr))
    draw = rng.random() * total
    running = 0.0
    for idx, value in enumerate(arr):
        running += float(value)
        if draw <= running:
            return int(idx)
    return int(np.argmax(arr))


def _legal_actions_from_mask(legal_mask: np.ndarray) -> List[int]:
    return [int(i) for i, value in enumerate(np.asarray(legal_mask, dtype=np.float32).reshape(-1)) if float(value) > 0.5]


def _is_raise_action(action_id: int) -> bool:
    return int(action_id) in ALL_RAISE_ACTIONS


def _small_raise_action(legal_mask: np.ndarray) -> Optional[int]:
    for action_id in NON_ALL_IN_RAISE_ACTIONS:
        if 0 <= int(action_id) < len(legal_mask) and float(legal_mask[int(action_id)]) > 0.5:
            return int(action_id)
    return None


def _large_raise_action(legal_mask: np.ndarray) -> Optional[int]:
    for action_id in reversed(NON_ALL_IN_RAISE_ACTIONS):
        if 0 <= int(action_id) < len(legal_mask) and float(legal_mask[int(action_id)]) > 0.5:
            return int(action_id)
    return None


def _hand_summary(state, actor: int, hand_ctx: HandContext) -> Dict[str, float]:
    hole_cards = flatten_cards_list(state.hole_cards[actor])
    to_call_bb = float(max(state.bets) - state.bets[actor]) / float(max(1, hand_ctx.big_blind))
    pot_bb = float(sum(p.amount for p in getattr(state, "pots", [])) + sum(state.bets)) / float(max(1, hand_ctx.big_blind))
    vec = encode_info_state(state, actor, hand_ctx)
    hand_class = int(np.argmax(vec[HAND_CLASS_SLICE_V24])) if float(vec[HAND_CLASS_SLICE_V24].sum()) > 0 else 0
    return {
        "street": float(getattr(hand_ctx, "current_street", 0)),
        "preflop_strength": estimate_preflop_strength(hole_cards, num_opponents=max(1, sum(1 for value in hand_ctx.in_hand if value) - 1)),
        "hand_class": float(hand_class),
        "pot_odds": float(to_call_bb / max(to_call_bb + pot_bb, 1e-6)),
        "facing_bet": 1.0 if to_call_bb > 1e-6 else 0.0,
    }


def _safe_prior_policy(state, actor: int, hand_ctx: HandContext, legal_mask: np.ndarray) -> np.ndarray:
    summary = _hand_summary(state, actor, hand_ctx)
    pre = float(summary["preflop_strength"])
    cls = int(summary["hand_class"])
    pot_odds = float(summary["pot_odds"])
    facing = bool(summary["facing_bet"] > 0.5)
    policy = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
    small_raise = _small_raise_action(legal_mask)
    large_raise = _large_raise_action(legal_mask)
    if int(summary["street"]) == 0:
        if facing:
            if pre >= 0.82:
                if large_raise is not None:
                    policy[int(large_raise)] += 0.30
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.70
            elif pre >= 0.48 or pot_odds <= 0.24:
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.68
                if float(legal_mask[ACTION_FOLD]) > 0.5:
                    policy[ACTION_FOLD] += 0.32
            else:
                if float(legal_mask[ACTION_FOLD]) > 0.5:
                    policy[ACTION_FOLD] += 0.85
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.15
        else:
            if pre >= 0.75 and small_raise is not None:
                policy[int(small_raise)] += 0.70
                if large_raise is not None:
                    policy[int(large_raise)] += 0.15
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.15
            elif pre >= 0.56 and small_raise is not None:
                policy[int(small_raise)] += 0.68
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.32
            else:
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.92
                if small_raise is not None:
                    policy[int(small_raise)] += 0.08
    else:
        if facing:
            if cls >= 5:
                if large_raise is not None:
                    policy[int(large_raise)] += 0.25
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.75
            elif cls >= 3:
                if small_raise is not None:
                    policy[int(small_raise)] += 0.10
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.65
                if float(legal_mask[ACTION_FOLD]) > 0.5:
                    policy[ACTION_FOLD] += 0.25
            elif cls == 1 and pot_odds <= 0.26:
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.60
                if float(legal_mask[ACTION_FOLD]) > 0.5:
                    policy[ACTION_FOLD] += 0.40
            else:
                if float(legal_mask[ACTION_FOLD]) > 0.5:
                    policy[ACTION_FOLD] += 0.80
                if float(legal_mask[ACTION_CALL]) > 0.5:
                    policy[ACTION_CALL] += 0.20
        else:
            if cls >= 5:
                if large_raise is not None:
                    policy[int(large_raise)] += 0.40
                if small_raise is not None:
                    policy[int(small_raise)] += 0.30
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.30
            elif cls >= 2:
                if small_raise is not None:
                    policy[int(small_raise)] += 0.18
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.82
            else:
                if float(legal_mask[ACTION_CHECK]) > 0.5:
                    policy[ACTION_CHECK] += 0.92
                if small_raise is not None:
                    policy[int(small_raise)] += 0.08
    return normalize_masked_policy(policy, legal_mask)


def _heuristic_action(state, actor: int, hand_ctx: HandContext, rng: random.Random) -> int:
    legal_mask = build_legal_action_mask(state, actor, hand_ctx)
    return _sample_action(_safe_prior_policy(state, actor, hand_ctx, legal_mask), rng)


def _synthetic_opponent_action(style: str, state, actor: int, hand_ctx: HandContext, rng: random.Random, legal_mask: Optional[np.ndarray] = None) -> int:
    legal_mask = build_legal_action_mask(state, actor, hand_ctx) if legal_mask is None else legal_mask
    policy = _safe_prior_policy(state, actor, hand_ctx, legal_mask)
    summary = _hand_summary(state, actor, hand_ctx)
    style = str(style or "nit").lower()
    if style == "maniac":
        if _small_raise_action(legal_mask) is not None:
            policy[int(_small_raise_action(legal_mask))] += 0.30
        if _large_raise_action(legal_mask) is not None:
            policy[int(_large_raise_action(legal_mask))] += 0.15
        policy[ACTION_FOLD] *= 0.35
    elif style == "overfolder" and summary["facing_bet"] > 0.5:
        policy[ACTION_FOLD] += 0.35
        policy[ACTION_CALL] *= 0.50
    elif style == "station" and summary["facing_bet"] > 0.5:
        policy[ACTION_CALL] += 0.25
        policy[ACTION_FOLD] *= 0.55
    elif style == "over3better" and int(summary["street"]) == 0 and _small_raise_action(legal_mask) is not None:
        policy[int(_small_raise_action(legal_mask))] += 0.22
    elif style == "overcaller" and summary["facing_bet"] > 0.5:
        policy[ACTION_CALL] += 0.20
    return _sample_action(normalize_masked_policy(policy, legal_mask), rng)


def _record_preflop_action_stats(preflop_stats: Dict[str, List[int]], actor: int, hand_ctx, to_call: float, prior_preflop_raises: int, action_id: int) -> None:
    seat = int(actor)
    if seat >= len(preflop_stats.get("preflop_opportunities", [])):
        return
    if preflop_stats["hands_seen_flags"][seat] == 0:
        preflop_stats["hands_seen_flags"][seat] = 1
        preflop_stats["hands_played"][seat] += 1
    preflop_stats["preflop_opportunities"][seat] += 1
    preflop_stats["aggression_opportunities"][seat] += 1
    facing = float(to_call) > 1e-6
    if _is_raise_action(action_id):
        preflop_stats["aggression_counts"][seat] += 1
        preflop_stats["pfr_counts"][seat] += 1
        if prior_preflop_raises == 1 and facing:
            preflop_stats["three_bet_counts"][seat] += 1
    if action_id == ACTION_CALL or _is_raise_action(action_id):
        preflop_stats["vpip_counts"][seat] += 1
    if action_id == ACTION_FOLD:
        preflop_stats["fold_preflop_counts"][seat] += 1
    if prior_preflop_raises == 1 and facing:
        preflop_stats["faced_open_opportunities"][seat] += 1
        if action_id == ACTION_FOLD:
            preflop_stats["fold_vs_open_counts"][seat] += 1
        elif action_id == ACTION_CALL:
            preflop_stats["call_vs_open_counts"][seat] += 1
    if prior_preflop_raises >= 2 and facing:
        preflop_stats["faced_three_bet_opportunities"][seat] += 1
        if action_id == ACTION_FOLD:
            preflop_stats["fold_vs_three_bet_counts"][seat] += 1
    if prior_preflop_raises == 1 and int(getattr(hand_ctx, "preflop_call_count", 0)) >= 1 and facing:
        preflop_stats["squeeze_opportunities"][seat] += 1
        if _is_raise_action(action_id):
            preflop_stats["squeeze_counts"][seat] += 1


def _record_postflop_action_stats(preflop_stats: Dict[str, List[int]], actor: int, hand_ctx, to_call: float, action_id: int) -> None:
    seat = int(actor)
    if seat < len(preflop_stats.get("aggression_opportunities", [])):
        preflop_stats["aggression_opportunities"][seat] += 1
        if _is_raise_action(action_id):
            preflop_stats["aggression_counts"][seat] += 1
    if int(getattr(hand_ctx, "current_street", 0)) == 1:
        pre = getattr(hand_ctx, "preflop_last_raiser", None)
        if int(getattr(hand_ctx, "street_raise_count", 0)) == 0 and pre is not None and int(pre) == seat and float(to_call) <= 1e-6 and _is_raise_action(action_id) and getattr(hand_ctx, "cbet_flop_initiator", None) is None:
            hand_ctx.cbet_flop_initiator = seat
        cbetter = getattr(hand_ctx, "cbet_flop_initiator", None)
        if cbetter is not None and int(cbetter) != seat and float(to_call) > 1e-6:
            preflop_stats["faced_cbet_flop_opportunities"][seat] += 1
            if action_id == ACTION_FOLD:
                preflop_stats["fold_vs_cbet_flop_counts"][seat] += 1
    if int(getattr(hand_ctx, "current_street", 0)) == 2:
        pre = getattr(hand_ctx, "preflop_last_raiser", None)
        if int(getattr(hand_ctx, "street_raise_count", 0)) == 0 and pre is not None and int(pre) == seat and float(to_call) <= 1e-6 and _is_raise_action(action_id) and getattr(hand_ctx, "cbet_turn_initiator", None) is None and getattr(hand_ctx, "cbet_flop_initiator", None) == seat:
            hand_ctx.cbet_turn_initiator = seat
        cbetter = getattr(hand_ctx, "cbet_turn_initiator", None)
        if cbetter is not None and int(cbetter) != seat and float(to_call) > 1e-6:
            preflop_stats["faced_cbet_turn_opportunities"][seat] += 1
            if action_id == ACTION_FOLD:
                preflop_stats["fold_vs_cbet_turn_counts"][seat] += 1


def _record_postflop_conditioned_decision(conditioned_counts: Dict[str, Dict[str, int]], hand_ctx, legal_mask: np.ndarray, to_call: float, action_id: int) -> None:
    street = int(getattr(hand_ctx, "current_street", 0))
    if street < 1 or street > 3:
        return
    counts = conditioned_counts[POSTFLOP_CONDITION_STREET_KEYS[street - 1]]
    facing = float(to_call) > 1e-6
    can_check = ACTION_CHECK < len(legal_mask) and float(legal_mask[ACTION_CHECK]) > 0.5
    can_call = ACTION_CALL < len(legal_mask) and float(legal_mask[ACTION_CALL]) > 0.5
    can_fold = ACTION_FOLD < len(legal_mask) and float(legal_mask[ACTION_FOLD]) > 0.5
    can_raise = any(float(legal_mask[action_id]) > 0.5 for action_id in ALL_RAISE_ACTIONS if action_id < len(legal_mask))
    if can_check:
        counts["check_when_legal_opportunities"] += 1
        if int(action_id) == ACTION_CHECK:
            counts["check_when_legal_hits"] += 1
    if can_check and not facing and can_raise:
        counts["bet_raise_when_checked_to_opportunities"] += 1
        counts["aggressive_when_checked_to_opportunities"] += 1
        if _is_raise_action(action_id):
            counts["bet_raise_when_checked_to_hits"] += 1
            counts["aggressive_when_checked_to_hits"] += 1
    if facing and can_fold:
        counts["fold_when_facing_bet_opportunities"] += 1
        if int(action_id) == ACTION_FOLD:
            counts["fold_when_facing_bet_hits"] += 1
    if facing and can_call:
        counts["call_when_facing_bet_opportunities"] += 1
        if int(action_id) == ACTION_CALL:
            counts["call_when_facing_bet_hits"] += 1
    if facing and can_raise:
        counts["raise_when_facing_bet_opportunities"] += 1
        if _is_raise_action(action_id):
            counts["raise_when_facing_bet_hits"] += 1


def _raise_target_bounds(state) -> tuple[int, int]:
    min_raise = getattr(state, "min_completion_betting_or_raising_to_amount", None)
    max_raise = getattr(state, "max_completion_betting_or_raising_to_amount", None)
    if min_raise is None:
        min_raise = getattr(state, "min_completion_betting_or_raising_to", 0)
    if max_raise is None:
        max_raise = getattr(state, "max_completion_betting_or_raising_to", 0)
    return int(min_raise or 0), int(max_raise or 0)


def apply_abstract_action(state, actor: int, action_id: int, hand_ctx: HandContext, record_history: bool = True) -> bool:
    actor = int(actor)
    requested = int(action_id)
    legal_mask = build_legal_action_mask(state, actor, hand_ctx)
    valid = 0 <= requested < ACTION_COUNT_V21 and float(legal_mask[requested]) > 0.5
    before_stack = float(state.stacks[actor])
    to_call = max(state.bets) - state.bets[actor]
    applied_raise = False
    applied_action = requested
    hand_ctx.last_action_was_all_in = False
    try:
        if requested == ACTION_FOLD and state.can_fold():
            state.fold()
            hand_ctx.in_hand[actor] = False
        elif requested in (ACTION_CHECK, ACTION_CALL) and state.can_check_or_call():
            state.check_or_call()
            if hand_ctx.current_street == 0 and float(to_call) > 1e-6 and requested == ACTION_CALL:
                hand_ctx.preflop_call_count += 1
        elif requested in ALL_RAISE_ACTIONS and state.can_complete_bet_or_raise_to():
            _, max_raise = _raise_target_bounds(state)
            target = abstract_raise_target(state, requested, hand_ctx)
            if target is None:
                raise ValueError("raise target unavailable")
            target = min(int(target), int(max_raise))
            state.complete_bet_or_raise_to(int(target))
            applied_raise = True
            hand_ctx.last_action_was_all_in = bool(int(target) >= int(max_raise))
        elif state.can_check_or_call():
            valid = False
            applied_action = ACTION_CALL if float(to_call) > 1e-6 else ACTION_CHECK
            state.check_or_call()
        elif state.can_fold():
            valid = False
            applied_action = ACTION_FOLD
            state.fold()
            hand_ctx.in_hand[actor] = False
    except Exception:
        valid = False
        if state.can_check_or_call():
            applied_action = ACTION_CALL if float(to_call) > 1e-6 else ACTION_CHECK
            state.check_or_call()
        elif state.can_fold():
            applied_action = ACTION_FOLD
            state.fold()
            hand_ctx.in_hand[actor] = False
    hand_ctx.contributions[actor] += max(0.0, before_stack - float(state.stacks[actor]))
    if applied_raise:
        hand_ctx.last_aggressor = actor
        hand_ctx.street_raise_count += 1
        if hand_ctx.current_street == 0:
            hand_ctx.preflop_raise_count += 1
            hand_ctx.preflop_opened = True
            hand_ctx.preflop_last_raiser = actor
    if record_history:
        _record_action_history(hand_ctx, actor, applied_action, False)
    return bool(valid)


def _fresh_replay_state_and_context(source_ctx: HandContext):
    state = NoLimitTexasHoldem.create_state(
        automations=(
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        ante_trimming_status=True,
        raw_antes={-1: 0},
        raw_blinds_or_straddles=(int(source_ctx.small_blind), int(source_ctx.big_blind)),
        min_bet=int(source_ctx.big_blind),
        raw_starting_stacks=list(source_ctx.starting_stacks),
        player_count=len(source_ctx.starting_stacks),
    )
    state.deck_cards = deque(source_ctx.deck_order)
    dealt_counts = [0] * len(source_ctx.hole_cards_by_seat)
    while state.can_deal_hole():
        seat = int(getattr(state, "hole_dealee_index", 0))
        card_index = dealt_counts[seat]
        card = source_ctx.hole_cards_by_seat[seat][card_index]
        state.deal_hole(cards=(card,), player_index=seat)
        dealt_counts[seat] += 1
    ctx = HandContext(
        starting_stacks=list(source_ctx.starting_stacks),
        big_blind=int(source_ctx.big_blind),
        small_blind=int(source_ctx.small_blind),
        in_hand=[True] * len(source_ctx.starting_stacks),
        contributions=[float(start - stack) for start, stack in zip(source_ctx.starting_stacks, state.stacks)],
        hole_cards_by_seat=[list(cards) for cards in source_ctx.hole_cards_by_seat],
        deck_order=list(source_ctx.deck_order),
        remaining_deck=list(source_ctx.remaining_deck),
    )
    return state, ctx


def _replay_pending_board_cards(state, hand_ctx: HandContext, target_board_cards: Sequence[Card], target_burn_cards: Optional[Sequence[Card]] = None) -> None:
    target_flat = list(flatten_cards_list(target_board_cards))
    target_burns = list(target_burn_cards or ())
    while state.status:
        if _burn_pending_card(state, hand_ctx, target_burns):
            continue
        if not state.can_deal_board():
            break
        current_flat = list(flatten_cards_list(state.board_cards))
        if len(current_flat) >= len(target_flat):
            break
        count = 3 if len(current_flat) == 0 else 1
        next_cards = tuple(target_flat[len(current_flat) : len(current_flat) + count])
        if not next_cards:
            break
        state.deal_board(cards=next_cards)
        _record_board_progress(state, hand_ctx, len(current_flat))


def _clone_branch_after_action(state, actor: int, action_id: int, hand_ctx: HandContext, perf_dict: Dict[str, float]):
    if int(getattr(hand_ctx, "current_street", 0)) >= 1:
        start = time.perf_counter()
        branch_state, branch_ctx = _fresh_replay_state_and_context(hand_ctx)
        for hist_actor, hist_action, _ in list(getattr(hand_ctx, "action_history", [])):
            apply_abstract_action(branch_state, int(hist_actor), int(hist_action), branch_ctx, record_history=True)
            _replay_pending_board_cards(branch_state, branch_ctx, hand_ctx.dealt_board_cards, hand_ctx.dealt_burn_cards)
        perf_dict["branch_clone_time"] += time.perf_counter() - start
    else:
        start = time.perf_counter()
        branch_state = copy.deepcopy(state)
        branch_ctx = copy.deepcopy(hand_ctx)
        perf_dict["branch_clone_time"] += time.perf_counter() - start
    start = time.perf_counter()
    apply_abstract_action(branch_state, actor, action_id, branch_ctx)
    perf_dict["apply_time"] += time.perf_counter() - start
    return branch_state, branch_ctx


def _slice_bucket(vec: np.ndarray, slc: slice) -> int:
    bucket = np.asarray(vec[slc], dtype=np.float32)
    if bucket.size <= 0 or float(bucket.sum()) <= 1e-8:
        return 0
    return int(np.argmax(bucket))


def _binary_signature(vec: np.ndarray, slc: slice) -> str:
    arr = np.asarray(vec[slc], dtype=np.float32)
    return "".join("1" if float(value) > 0.5 else "0" for value in arr)


def infoset_key_from_vector(vec: np.ndarray) -> str:
    street_bucket = _slice_bucket(vec, STREET_SLICE_V24)
    position_bucket = _slice_bucket(vec, POSITION_SLICE_V24)
    active_bucket = _slice_bucket(vec, ACTIVE_PLAYERS_SLICE_V24)
    behind_bucket = _slice_bucket(vec, PLAYERS_BEHIND_SLICE_V24)
    pot_bucket = _slice_bucket(vec, POT_BUCKET_SLICE_V24)
    spr_bucket = _slice_bucket(vec, SPR_BUCKET_SLICE_V24)
    to_call_bucket = _slice_bucket(vec, TO_CALL_BUCKET_SLICE_V24)
    facing_bucket = _slice_bucket(vec, FACING_BET_BUCKET_SLICE_V24)
    raise_bucket = _slice_bucket(vec, RAISE_COUNT_SLICE_V24)
    pre_bucket = _slice_bucket(vec, PREPERCENTILE_SLICE_V24)
    hand_bucket = _slice_bucket(vec, HAND_CLASS_SLICE_V24)
    board_bits = _binary_signature(vec, BOARD_TEXTURE_SLICE_V24)
    draw_bits = _binary_signature(vec, DRAW_FLAGS_SLICE_V24)
    last_flag = int(float(vec[IDX_LAST_AGGRESSOR_FLAG_V24]) > 0.5)
    ip_flag = int(float(vec[IDX_IN_POSITION_FLAG_V24]) > 0.5)
    if street_bucket == 0:
        street_component = f"pre{pre_bucket}"
    else:
        street_component = f"post{hand_bucket}"
    return (
        f"s{street_bucket}|p{position_bucket}|a{active_bucket}|b{behind_bucket}|"
        f"pot{pot_bucket}|spr{spr_bucket}|tc{to_call_bucket}|fb{facing_bucket}|"
        f"r{raise_bucket}|{street_component}|bd{board_bits}|dr{draw_bits}|"
        f"la{last_flag}|ip{ip_flag}"
    )


def build_infoset_key(state, actor: int, hand_ctx: HandContext) -> tuple[str, np.ndarray, np.ndarray]:
    state_vec, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
    return infoset_key_from_vector(state_vec), state_vec, legal_mask


def _clone_tabular_node(node: TabularNode) -> TabularNode:
    return TabularNode(
        legal_mask=np.asarray(node.legal_mask, dtype=np.float32).copy(),
        regret_sum=np.asarray(node.regret_sum, dtype=np.float32).copy(),
        strategy_sum=np.asarray(node.strategy_sum, dtype=np.float32).copy(),
        visits=int(node.visits),
    )


def _apply_node_delta_payload(node_store: Dict[str, TabularNode], delta_payload: Dict[str, Dict[str, object]]) -> None:
    if not isinstance(node_store, dict) or not isinstance(delta_payload, dict):
        return
    for infoset_key, payload in delta_payload.items():
        if not isinstance(payload, dict):
            continue
        legal_mask = np.asarray(payload.get("legal_mask", np.zeros(ACTION_COUNT_V21, dtype=np.float32)), dtype=np.float32).reshape(-1)
        node = node_store.get(str(infoset_key))
        if node is None:
            node = TabularNode.new(legal_mask)
            node_store[str(infoset_key)] = node
        else:
            node.merge_legal_mask(legal_mask)
        regret_delta = np.asarray(payload.get("regret_sum_delta", np.zeros_like(node.regret_sum)), dtype=np.float32).reshape(-1)
        strategy_delta = np.asarray(payload.get("strategy_sum_delta", np.zeros_like(node.strategy_sum)), dtype=np.float32).reshape(-1)
        if regret_delta.shape == node.regret_sum.shape:
            node.regret_sum += regret_delta
        if strategy_delta.shape == node.strategy_sum.shape:
            node.strategy_sum += strategy_delta
        node.visits += int(payload.get("visit_delta", 0))


def _lookup_live_node(config, infoset_key: str, legal_mask: np.ndarray, create: bool) -> Optional[TabularNode]:
    node_store = getattr(config, "live_node_store", None)
    if not isinstance(node_store, dict):
        return None
    base_store = getattr(config, "base_node_store", node_store)
    overlay_mode = isinstance(base_store, dict) and base_store is not node_store
    node = node_store.get(infoset_key)
    if not overlay_mode:
        if node is None:
            if not create:
                return None
            node = TabularNode.new(legal_mask)
            node_store[infoset_key] = node
        else:
            node.merge_legal_mask(legal_mask)
        return node

    if node is not None:
        node.merge_legal_mask(legal_mask)
        return node
    base_node = base_store.get(infoset_key) if isinstance(base_store, dict) else None
    if base_node is None:
        if not create:
            return None
        node = TabularNode.new(legal_mask)
        node_store[infoset_key] = node
        return node
    if not create:
        return base_node
    node = _clone_tabular_node(base_node)
    node.merge_legal_mask(legal_mask)
    node_store[infoset_key] = node
    return node


def _mark_touched_infoset(config, infoset_key: str) -> None:
    touched = getattr(config, "local_touched_infosets", None)
    if isinstance(touched, set):
        touched.add(str(infoset_key))


def _live_policy(state, actor: int, hand_ctx: HandContext, config, *, use_average: bool) -> tuple[str, np.ndarray, np.ndarray]:
    infoset_key, _, legal_mask = build_infoset_key(state, actor, hand_ctx)
    node = _lookup_live_node(config, infoset_key, legal_mask, create=False)
    if node is None:
        return infoset_key, uniform_legal_policy(legal_mask), legal_mask
    if use_average:
        policy = average_policy(node.strategy_sum, legal_mask, regret_sum=node.regret_sum)
    else:
        policy = regret_matching(node.regret_sum, legal_mask)
    return infoset_key, policy, legal_mask


def _snapshot_policy(snapshot: Optional[TabularPolicySnapshot], state, actor: int, hand_ctx: HandContext) -> tuple[str, np.ndarray, np.ndarray]:
    infoset_key, _, legal_mask = build_infoset_key(state, actor, hand_ctx)
    if snapshot is None or not isinstance(snapshot, TabularPolicySnapshot):
        return infoset_key, uniform_legal_policy(legal_mask), legal_mask
    entry = snapshot.policy_table.get(infoset_key)
    if entry is None:
        return infoset_key, uniform_legal_policy(legal_mask), legal_mask
    if float(np.asarray(entry.average_policy, dtype=np.float32).sum()) > 1e-8:
        return infoset_key, normalize_masked_policy(entry.average_policy, legal_mask), legal_mask
    if float(np.asarray(entry.current_policy, dtype=np.float32).sum()) > 1e-8:
        return infoset_key, normalize_masked_policy(entry.current_policy, legal_mask), legal_mask
    return infoset_key, uniform_legal_policy(legal_mask), legal_mask


def _aligned_state_vector_for_model(model: Optional[PokerDeepCFRNet], state_vec: np.ndarray) -> np.ndarray:
    if model is None:
        return np.asarray(state_vec, dtype=np.float32)
    expected_dim = int(getattr(model, "state_dim", 0) or 0)
    vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if expected_dim <= 0 or vec.shape[0] == expected_dim:
        return vec
    if vec.shape[0] > expected_dim:
        return vec[:expected_dim]
    aligned = np.zeros(expected_dim, dtype=np.float32)
    aligned[: vec.shape[0]] = vec
    return aligned


def _checkpoint_snapshot_by_seat(config, rng: random.Random, hero_seat: int) -> Dict[int, Optional[TabularPolicySnapshot]]:
    pool = list(getattr(config, "checkpoint_pool", ()) or ())
    if not pool:
        return {}
    candidates = list(pool[:-1]) if len(pool) > 1 else list(pool)
    seat_map: Dict[int, Optional[TabularPolicySnapshot]] = {}
    player_count = int(getattr(config, "num_players", 6))
    for seat in range(player_count):
        if seat == int(hero_seat):
            continue
        seat_map[seat] = candidates[rng.randrange(len(candidates))]
    return seat_map


def _should_prune_action(node: TabularNode, action_id: int, config) -> bool:
    current_iteration = int(getattr(config, "current_iteration", 0))
    prune_after = int(getattr(config, "prune_after_iteration", 200))
    floor = float(getattr(config, "negative_regret_floor", -300_000_000.0))
    return current_iteration >= prune_after and float(node.regret_sum[int(action_id)]) <= floor


def _external_sampling_traversal(state, traverser: int, hand_ctx: HandContext, config, rng: random.Random, result: Optional[TraversalResult], perf: Dict[str, float]) -> float:
    while state.status:
        start = time.perf_counter()
        _advance_chance_nodes(state, hand_ctx)
        perf["chance_time"] += time.perf_counter() - start
        if not state.status:
            break
        actor = state.actor_index
        if actor is None:
            break
        infoset_key, _, legal_mask = build_infoset_key(state, actor, hand_ctx)
        legal_actions = _legal_actions_from_mask(legal_mask)
        if not legal_actions:
            break
        if int(actor) == int(traverser):
            start = time.perf_counter()
            node = _lookup_live_node(config, infoset_key, legal_mask, create=True)
            perf["traverser_state_time"] += time.perf_counter() - start
            if node is None:
                return _safe_utility_bb(state, traverser, hand_ctx)
            sigma = regret_matching(node.regret_sum, legal_mask)
            if result is not None:
                result.traverser_decisions += 1
            branch_actions = [action for action in legal_actions if not _should_prune_action(node, action, config)]
            if not branch_actions:
                branch_actions = list(legal_actions)
            action_values = np.full(ACTION_COUNT_V21, 0.0, dtype=np.float32)
            node_value = 0.0
            for action_id in branch_actions:
                branch_state, branch_ctx = _clone_branch_after_action(state, actor, action_id, hand_ctx, perf)
                child_rng = random.Random(rng.randrange(2**30))
                action_value = _external_sampling_traversal(branch_state, traverser, branch_ctx, config, child_rng, None, perf)
                action_values[int(action_id)] = float(action_value)
                node_value += float(sigma[int(action_id)]) * float(action_value)
            for action_id in branch_actions:
                node.regret_sum[int(action_id)] += float(action_values[int(action_id)] - node_value)
            node.visits += 1
            _mark_touched_infoset(config, infoset_key)
            return float(node_value)
        start = time.perf_counter()
        node = _lookup_live_node(config, infoset_key, legal_mask, create=True)
        perf["opponent_state_time"] += time.perf_counter() - start
        if node is None:
            chosen_action = _sample_action(uniform_legal_policy(legal_mask), rng)
        else:
            sigma = regret_matching(node.regret_sum, legal_mask)
            node.strategy_sum += sigma
            node.visits += 1
            _mark_touched_infoset(config, infoset_key)
            chosen_action = _sample_action(sigma, rng)
        start = time.perf_counter()
        apply_abstract_action(state, actor, chosen_action, hand_ctx)
        perf["apply_time"] += time.perf_counter() - start
    return _safe_utility_bb(state, traverser, hand_ctx)


def _record_action_metrics(result: TraversalResult, actor: int, action_id: int, hand_ctx: HandContext, legal_mask: np.ndarray, to_call: float) -> None:
    result.action_counts[int(action_id)] += 1
    if hand_ctx.current_street == 0:
        result.preflop_decisions += 1
        result.preflop_action_counts[int(action_id)] += 1
        prior = int(hand_ctx.preflop_raise_count)
        _record_preflop_action_stats(result.preflop_stats, actor, hand_ctx, to_call, prior, int(action_id))
        result.vpip = result.vpip or int(action_id) == ACTION_CALL or _is_raise_action(int(action_id))
        result.pfr = result.pfr or _is_raise_action(int(action_id))
        result.three_bet = result.three_bet or (_is_raise_action(int(action_id)) and prior >= 1)
    else:
        result.postflop_action_counts[int(action_id)] += 1
        _record_postflop_action_stats(result.preflop_stats, actor, hand_ctx, to_call, int(action_id))
        _record_postflop_conditioned_decision(result.postflop_conditioned_counts, hand_ctx, legal_mask, to_call, int(action_id))


def _sample_training_rollout(state, traverser: int, hand_ctx: HandContext, config, rng: random.Random, result: TraversalResult, perf: Dict[str, float]) -> float:
    while state.status:
        start = time.perf_counter()
        _advance_chance_nodes(state, hand_ctx)
        perf["chance_time"] += time.perf_counter() - start
        if not state.status:
            break
        actor = state.actor_index
        if actor is None:
            break
        _, policy, legal_mask = _live_policy(state, actor, hand_ctx, config, use_average=False)
        chosen_action = _sample_action(policy, rng)
        to_call = float(max(state.bets) - state.bets[actor])
        if int(actor) == int(traverser):
            _record_action_metrics(result, actor, chosen_action, hand_ctx, legal_mask, to_call)
        start = time.perf_counter()
        valid = apply_abstract_action(state, actor, chosen_action, hand_ctx)
        perf["apply_time"] += time.perf_counter() - start
        if int(actor) == int(traverser) and hand_ctx.current_street == 0 and hand_ctx.last_action_was_all_in:
            result.preflop_jam = True
        if not valid:
            result.invalid_action_count += 1
    return _safe_utility_bb(state, traverser, hand_ctx)


def _choose_policy_action(snapshot: Optional[TabularPolicySnapshot], state, actor: int, hand_ctx: HandContext, rng: random.Random, config, seat_snapshot_map: Optional[Dict[int, Optional[TabularPolicySnapshot]]] = None, return_details: bool = False):
    mode = str(getattr(config, "evaluation_mode", "heuristics") or "heuristics").lower()
    if mode == "heuristics" and int(actor) != int(getattr(config, "eval_hero_seat", 0)):
        legal_mask = build_legal_action_mask(state, actor, hand_ctx)
        chosen_action = _heuristic_action(state, actor, hand_ctx, rng)
        if not return_details:
            return chosen_action
        return chosen_action, {"legal_mask": legal_mask, "policy": _safe_prior_policy(state, actor, hand_ctx, legal_mask)}
    if (mode == "synthetic" or mode in SYNTHETIC_OPPONENT_STYLES) and int(actor) != int(getattr(config, "eval_hero_seat", 0)):
        legal_mask = build_legal_action_mask(state, actor, hand_ctx)
        style = str(getattr(config, "synthetic_opponent_style", "nit") or "nit") if mode == "synthetic" else mode
        chosen_action = _synthetic_opponent_action(style, state, actor, hand_ctx, rng, legal_mask=legal_mask)
        if not return_details:
            return chosen_action
        return chosen_action, {"legal_mask": legal_mask, "policy": _safe_prior_policy(state, actor, hand_ctx, legal_mask)}
    seat_snapshot = snapshot
    if mode == "checkpoints" and seat_snapshot_map is not None and int(actor) in seat_snapshot_map:
        seat_snapshot = seat_snapshot_map.get(int(actor))
    _, policy, legal_mask = _snapshot_policy(seat_snapshot, state, actor, hand_ctx)
    chosen_action = _sample_action(policy, rng)
    if not return_details:
        return chosen_action
    return chosen_action, {"legal_mask": legal_mask, "policy": policy}


def _policy_action_for_snapshot(
    snapshot,
    state,
    actor: int,
    hand_ctx: HandContext,
    rng: random.Random,
    opponent_profile=None,
    opponent_profiles_by_seat=None,
    config=None,
    return_details: bool = False,
):
    del opponent_profile
    if isinstance(snapshot, PokerDeepCFRNet):
        state_vec, legal_mask = encode_info_state(
            state,
            actor,
            hand_ctx,
            return_legal_mask=True,
            opponent_profiles_by_seat=opponent_profiles_by_seat,
        )
        state_vec = _aligned_state_vector_for_model(snapshot, state_vec)
        with torch.no_grad():
            logits = (
                snapshot.forward_strategy(torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        policy = masked_policy(logits, legal_mask)
        chosen_action = _sample_action(policy, rng)
        if not return_details:
            return chosen_action
        return chosen_action, {
            "state_vec": state_vec,
            "legal_mask": legal_mask,
            "policy": policy,
            "guidance": {},
        }

    if config is None:
        runtime_config = build_runtime_policy_config({"evaluation_mode": "self_play"})
    elif isinstance(config, dict):
        runtime_payload = dict(config)
        runtime_payload["evaluation_mode"] = "self_play"
        runtime_config = build_runtime_policy_config(runtime_payload)
    else:
        runtime_payload = {key: value for key, value in vars(config).items() if not str(key).startswith("_")}
        runtime_payload["evaluation_mode"] = "self_play"
        runtime_config = build_runtime_policy_config(runtime_payload)
    return _choose_policy_action(
        snapshot if isinstance(snapshot, TabularPolicySnapshot) else None,
        state,
        actor,
        hand_ctx,
        rng,
        runtime_config,
        return_details=return_details,
    )


def run_traversal(hand_seed: int, traverser_seat: int, actor_snapshot: Optional[TabularPolicySnapshot], opponent_snapshot: Optional[TabularPolicySnapshot], config) -> TraversalResult:
    del actor_snapshot, opponent_snapshot
    rng = random.Random(int(hand_seed))
    perf = _new_perf_breakdown()
    start = time.perf_counter()
    state, hand_ctx = _create_state_and_context(rng, config)
    perf["state_init_time"] += time.perf_counter() - start
    result = TraversalResult(traverser_seat=int(traverser_seat))
    result.preflop_stats = _new_preflop_stats(getattr(config, "num_players", 6))
    _external_sampling_traversal(state, int(traverser_seat), hand_ctx, config, rng, result, perf)

    monitor_interval = max(1, int(getattr(config, "training_monitor_interval_traversals", 8)))
    if int(hand_seed) % monitor_interval == 0:
        metrics_rng = random.Random(int(hand_seed) ^ 0x5F3759DF)
        metrics_state, metrics_ctx = _create_state_and_context(metrics_rng, config)
        sampled_utility = _sample_training_rollout(metrics_state, int(traverser_seat), metrics_ctx, config, metrics_rng, result, perf)
        result.utility_bb = float(sampled_utility)
        result.unclipped_utility_bb = float(sampled_utility)
        result.flop_seen = bool(getattr(metrics_ctx, "flop_seen", False))
        result.total_actions = int(getattr(metrics_ctx, "total_actions", 0))
        result.preflop_actions = int(getattr(metrics_ctx, "preflop_actions", 0))
        result.monitor_sampled = True
    result.perf_breakdown = perf
    return result


def run_policy_hand(hand_seed: int, actor_snapshot: Optional[TabularPolicySnapshot], config) -> HandResult:
    rng = random.Random(int(hand_seed))
    state, hand_ctx = _create_state_and_context(rng, config)
    hero_seat = int(getattr(config, "eval_hero_seat", 0)) % int(getattr(config, "num_players", 6))
    action_counts = _new_action_histogram()
    preflop_action_counts = _new_action_histogram()
    postflop_action_counts = _new_action_histogram()
    conditioned_counts = _new_postflop_conditioned_counts()
    preflop_stats = _new_preflop_stats(getattr(config, "num_players", 6))
    illegal_action_count = 0
    vpip = pfr = three_bet = preflop_jam = False
    rfi_opportunity = rfi_attempt = False
    cbet_flop_opportunity = cbet_flop_taken = fold_vs_cbet_flop_opportunity = fold_vs_cbet_flop = False
    cbet_turn_opportunity = cbet_turn_taken = fold_vs_cbet_turn_opportunity = fold_vs_cbet_turn = False
    hero_hand_key = None
    seat_checkpoint_map = _checkpoint_snapshot_by_seat(config, rng, hero_seat)
    if int(hero_seat) < len(state.hole_cards):
        hole = flatten_cards_list(state.hole_cards[hero_seat])
        hero_hand_key = canonical_preflop_hand_key(hole) if hole else None
    while state.status:
        _advance_chance_nodes(state, hand_ctx)
        if not state.status:
            break
        actor = state.actor_index
        if actor is None:
            break
        chosen_action, details = _choose_policy_action(
            actor_snapshot,
            state,
            actor,
            hand_ctx,
            rng,
            config,
            seat_snapshot_map=seat_checkpoint_map,
            return_details=True,
        )
        to_call = float(max(state.bets) - state.bets[actor])
        if int(actor) == hero_seat:
            action_counts[int(chosen_action)] += 1
            if hand_ctx.current_street == 0:
                preflop_action_counts[int(chosen_action)] += 1
                prior = int(hand_ctx.preflop_raise_count)
                _record_preflop_action_stats(preflop_stats, actor, hand_ctx, to_call, prior, int(chosen_action))
                if prior == 0:
                    rfi_opportunity = True
                    rfi_attempt = rfi_attempt or _is_raise_action(int(chosen_action))
                vpip = vpip or int(chosen_action) == ACTION_CALL or _is_raise_action(int(chosen_action))
                pfr = pfr or _is_raise_action(int(chosen_action))
                three_bet = three_bet or (_is_raise_action(int(chosen_action)) and prior >= 1)
            else:
                postflop_action_counts[int(chosen_action)] += 1
                _record_postflop_action_stats(preflop_stats, actor, hand_ctx, to_call, int(chosen_action))
                _record_postflop_conditioned_decision(conditioned_counts, hand_ctx, details["legal_mask"], to_call, int(chosen_action))
            current_street = int(getattr(hand_ctx, "current_street", 0))
            preflop_last_raiser = getattr(hand_ctx, "preflop_last_raiser", None)
            if current_street == 1:
                if int(getattr(hand_ctx, "street_raise_count", 0)) == 0 and preflop_last_raiser is not None and int(preflop_last_raiser) == actor and float(to_call) <= 1e-6 and getattr(hand_ctx, "cbet_flop_initiator", None) is None:
                    cbet_flop_opportunity = True
                    cbet_flop_taken = cbet_flop_taken or _is_raise_action(int(chosen_action))
                cbetter = getattr(hand_ctx, "cbet_flop_initiator", None)
                if cbetter is not None and int(cbetter) != actor and float(to_call) > 1e-6:
                    fold_vs_cbet_flop_opportunity = True
                    fold_vs_cbet_flop = fold_vs_cbet_flop or int(chosen_action) == ACTION_FOLD
            elif current_street == 2:
                if int(getattr(hand_ctx, "street_raise_count", 0)) == 0 and preflop_last_raiser is not None and int(preflop_last_raiser) == actor and getattr(hand_ctx, "cbet_flop_initiator", None) == actor and float(to_call) <= 1e-6 and getattr(hand_ctx, "cbet_turn_initiator", None) is None:
                    cbet_turn_opportunity = True
                    cbet_turn_taken = cbet_turn_taken or _is_raise_action(int(chosen_action))
                cbetter = getattr(hand_ctx, "cbet_turn_initiator", None)
                if cbetter is not None and int(cbetter) != actor and float(to_call) > 1e-6:
                    fold_vs_cbet_turn_opportunity = True
                    fold_vs_cbet_turn = fold_vs_cbet_turn or int(chosen_action) == ACTION_FOLD
        else:
            if hand_ctx.current_street == 0:
                _record_preflop_action_stats(preflop_stats, actor, hand_ctx, to_call, int(hand_ctx.preflop_raise_count), int(chosen_action))
            else:
                _record_postflop_action_stats(preflop_stats, actor, hand_ctx, to_call, int(chosen_action))
        valid = apply_abstract_action(state, actor, int(chosen_action), hand_ctx)
        if int(actor) == hero_seat and hand_ctx.current_street == 0 and hand_ctx.last_action_was_all_in:
            preflop_jam = True
        if not valid:
            illegal_action_count += 1
    hero_profit_bb = (float(state.stacks[hero_seat]) - float(hand_ctx.starting_stacks[hero_seat])) / float(hand_ctx.big_blind)
    total_actions = int(getattr(hand_ctx, "total_actions", 0))
    preflop_actions = int(getattr(hand_ctx, "preflop_actions", 0))
    showdown_seen = int(sum(1 for flag in getattr(hand_ctx, "in_hand", []) if flag)) > 1
    return HandResult(
        hero_profit_bb=hero_profit_bb,
        hero_seat=hero_seat,
        action_counts=action_counts,
        preflop_action_counts=preflop_action_counts,
        postflop_action_counts=postflop_action_counts,
        postflop_conditioned_counts=conditioned_counts,
        illegal_action_count=illegal_action_count,
        win=hero_profit_bb > 0.0,
        vpip=bool(vpip),
        pfr=bool(pfr),
        three_bet=bool(three_bet),
        preflop_jam=bool(preflop_jam),
        flop_seen=bool(hand_ctx.flop_seen),
        turn_seen=bool(hand_ctx.turn_seen),
        river_seen=bool(hand_ctx.river_seen),
        showdown_seen=bool(showdown_seen),
        showdown_won=bool(showdown_seen and hero_profit_bb > 0.0),
        total_actions=total_actions,
        preflop_actions=preflop_actions,
        postflop_actions=max(0, total_actions - preflop_actions),
        blueprint_decisions=0,
        preflop_decisions=int(preflop_action_counts.sum()),
        cbet_flop_opportunity=bool(cbet_flop_opportunity),
        cbet_flop_taken=bool(cbet_flop_taken),
        fold_vs_cbet_flop_opportunity=bool(fold_vs_cbet_flop_opportunity),
        fold_vs_cbet_flop=bool(fold_vs_cbet_flop),
        cbet_turn_opportunity=bool(cbet_turn_opportunity),
        cbet_turn_taken=bool(cbet_turn_taken),
        fold_vs_cbet_turn_opportunity=bool(fold_vs_cbet_turn_opportunity),
        fold_vs_cbet_turn=bool(fold_vs_cbet_turn),
        rfi_opportunity=bool(rfi_opportunity),
        rfi_attempt=bool(rfi_attempt),
        hero_hand_key=hero_hand_key,
        preflop_stats=preflop_stats,
    )


def _serialize_node_delta_payload(
    base_store: Dict[str, TabularNode],
    updated_store: Dict[str, TabularNode],
    touched_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    payload: Dict[str, Dict[str, object]] = {}
    candidate_keys = set(str(key) for key in touched_keys) if touched_keys is not None else (set(base_store.keys()) | set(updated_store.keys()))
    for key in candidate_keys:
        updated = updated_store.get(key)
        if updated is None:
            continue
        base = base_store.get(key)
        legal_mask = np.asarray(updated.legal_mask, dtype=np.float32).copy()
        if base is None:
            regret_delta = np.asarray(updated.regret_sum, dtype=np.float32).copy()
            strategy_delta = np.asarray(updated.strategy_sum, dtype=np.float32).copy()
            visit_delta = int(updated.visits)
        else:
            regret_delta = np.asarray(updated.regret_sum - np.asarray(base.regret_sum, dtype=np.float32), dtype=np.float32).copy()
            strategy_delta = np.asarray(updated.strategy_sum - np.asarray(base.strategy_sum, dtype=np.float32), dtype=np.float32).copy()
            visit_delta = int(updated.visits - int(base.visits))
            legal_mask = np.maximum(legal_mask, np.asarray(base.legal_mask, dtype=np.float32))
        if visit_delta == 0 and not np.any(regret_delta) and not np.any(strategy_delta):
            continue
        payload[str(key)] = {
            "legal_mask": legal_mask.astype(np.float32),
            "regret_sum_delta": regret_delta.astype(np.float32),
            "strategy_sum_delta": strategy_delta.astype(np.float32),
            "visit_delta": int(visit_delta),
        }
    return payload


def run_traversal_batch_mp(hand_seeds: List[int], traverser_seats: List[int], actor_state_dict, advantage_state_dict, config_dict: Dict[str, object], snapshot_signature: str) -> TraversalBatchResult:
    del advantage_state_dict
    global _WORKER_BASE_SNAPSHOT_SIGNATURE, _WORKER_BASE_NODE_STORE
    config = build_runtime_policy_config(config_dict)
    if isinstance(actor_state_dict, dict) and "__snapshot_mode__" in actor_state_dict:
        mode = str(actor_state_dict.get("__snapshot_mode__", "")).strip().lower()
        if mode == "full":
            _WORKER_BASE_NODE_STORE = deserialize_node_store(actor_state_dict.get("node_store", {}))
            _WORKER_BASE_SNAPSHOT_SIGNATURE = str(snapshot_signature)
        elif mode == "delta":
            base_signature = str(actor_state_dict.get("base_signature", ""))
            if str(_WORKER_BASE_SNAPSHOT_SIGNATURE) != base_signature or not isinstance(_WORKER_BASE_NODE_STORE, dict):
                raise RuntimeError(f"Worker snapshot cache miss for signature '{snapshot_signature}'.")
            _apply_node_delta_payload(_WORKER_BASE_NODE_STORE, actor_state_dict.get("delta_payload", {}))
            _WORKER_BASE_SNAPSHOT_SIGNATURE = str(snapshot_signature)
        else:
            raise RuntimeError(f"Unsupported worker snapshot mode '{mode}'.")
    elif isinstance(actor_state_dict, dict):
        _WORKER_BASE_NODE_STORE = deserialize_node_store(actor_state_dict)
        _WORKER_BASE_SNAPSHOT_SIGNATURE = str(snapshot_signature)
    elif str(snapshot_signature) != str(_WORKER_BASE_SNAPSHOT_SIGNATURE) or not isinstance(_WORKER_BASE_NODE_STORE, dict):
        raise RuntimeError(f"Worker snapshot cache miss for signature '{snapshot_signature}'.")
    config.base_node_store = _WORKER_BASE_NODE_STORE
    config.live_node_store = {}
    config.local_touched_infosets = set()
    results = [run_traversal(seed, seat, None, None, config) for seed, seat in zip(hand_seeds, traverser_seats)]
    return TraversalBatchResult(
        results=results,
        node_deltas=_serialize_node_delta_payload(
            config.base_node_store,
            config.live_node_store,
            touched_keys=getattr(config, "local_touched_infosets", None),
        ),
    )


__all__ = [
    "SYNTHETIC_OPPONENT_STYLES",
    "HandContext",
    "TraversalResult",
    "TraversalBatchResult",
    "HandResult",
    "TabularNode",
    "TabularPolicySnapshot",
    "_policy_action_for_snapshot",
    "apply_abstract_action",
    "build_infoset_key",
    "build_runtime_policy_config",
    "freeze_policy_snapshot",
    "infoset_key_from_vector",
    "run_policy_hand",
    "run_traversal",
    "run_traversal_batch_mp",
    "serialize_node_store",
]
