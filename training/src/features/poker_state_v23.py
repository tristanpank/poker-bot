import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from pokerkit import Card

BASE_STATE_DIM_V23 = 98
ACTION_COUNT_V21 = 5
VALIDATE_INFO_STATE_V21 = os.getenv("POKER_V21_VALIDATE_INFO_STATE", "0").strip() in {"1", "true", "True"}

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_HALF_POT = 3
ACTION_RAISE_POT_OR_ALL_IN = 4

ACTION_NAMES_V21 = [
    "Fold",
    "Check",
    "Call",
    "Raise 1/2 Pot",
    "Raise Pot/All-In",
]

POSITION_NAMES_V21 = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
RANK_ORDER = "23456789TJQKA"
SUIT_ORDER = "cdhs"
BROADWAY_RANKS = set("TJQKA")
RELATIVE_OPPONENT_SLOTS_V23 = 5
PER_OPPONENT_PROFILE_FEATURE_NAMES_V23 = [
    "vpip_pct",
    "pfr_pct",
    "three_bet_pct",
    "fold_to_open_pct",
    "fold_to_three_bet_pct",
    "call_open_pct",
    "squeeze_pct",
    "fold_to_cbet_flop_pct",
    "fold_to_cbet_turn_pct",
    "aggression_freq_pct",
    "hands_confidence_norm",
]
OPPONENT_PROFILE_PER_SLOT_DIM_V23 = len(PER_OPPONENT_PROFILE_FEATURE_NAMES_V23)


def _relative_opponent_feature_names() -> List[str]:
    names: List[str] = []
    for slot in range(1, RELATIVE_OPPONENT_SLOTS_V23 + 1):
        prefix = f"rel_opp_{slot}_"
        for stat_name in PER_OPPONENT_PROFILE_FEATURE_NAMES_V23:
            names.append(f"{prefix}{stat_name}")
    return names


OPPONENT_PROFILE_FEATURE_NAMES_V23 = _relative_opponent_feature_names()
OPPONENT_PROFILE_DIM_V23 = len(OPPONENT_PROFILE_FEATURE_NAMES_V23)
OPPONENT_PROFILE_DEFAULT_SLOT_V23 = tuple(0.0 for _ in range(OPPONENT_PROFILE_PER_SLOT_DIM_V23))
OPPONENT_PROFILE_DEFAULT_V23 = tuple(0.0 for _ in range(OPPONENT_PROFILE_DIM_V23))
STATE_DIM_V21 = BASE_STATE_DIM_V23 + OPPONENT_PROFILE_DIM_V23

FEATURE_NAMES_V21 = [
    "high_private_2",
    "high_private_3",
    "high_private_4",
    "high_private_5",
    "high_private_6",
    "high_private_7",
    "high_private_8",
    "high_private_9",
    "high_private_T",
    "high_private_J",
    "high_private_Q",
    "high_private_K",
    "high_private_A",
    "low_private_2",
    "low_private_3",
    "low_private_4",
    "low_private_5",
    "low_private_6",
    "low_private_7",
    "low_private_8",
    "low_private_9",
    "low_private_T",
    "low_private_J",
    "low_private_Q",
    "low_private_K",
    "low_private_A",
    "is_suited",
    "is_pocket_pair",
    "gap_connector",
    "gap_one",
    "gap_two",
    "gap_three_plus",
    "board_rank_2",
    "board_rank_3",
    "board_rank_4",
    "board_rank_5",
    "board_rank_6",
    "board_rank_7",
    "board_rank_8",
    "board_rank_9",
    "board_rank_T",
    "board_rank_J",
    "board_rank_Q",
    "board_rank_K",
    "board_rank_A",
    "board_suit_c",
    "board_suit_d",
    "board_suit_h",
    "board_suit_s",
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "board_paired",
    "board_trips_plus",
    "board_monotone",
    "board_two_tone",
    "board_connected",
    "preflop_strength_scalar",
    "hero_flush_draw",
    "current_hand_strength_scalar",
    "pos_sb",
    "pos_bb",
    "pos_utg",
    "pos_mp",
    "pos_co",
    "pos_btn",
    "active_players_norm",
    "players_before_norm",
    "players_after_norm",
    "pot_size_norm",
    "to_call_norm",
    "min_raise_to_norm",
    "hero_stack_norm",
    "effective_stack_norm",
    "spr_norm",
    "pot_odds",
    "hero_commitment",
    "street_raise_count_norm",
    "last_aggressive_size_norm",
    "hero_total_contribution_norm",
    "can_fold",
    "can_check",
    "can_call",
    "can_raise_half_pot",
    "can_raise_pot_or_all_in",
    "rel_active_self",
    "rel_active_1",
    "rel_active_2",
    "rel_active_3",
    "rel_active_4",
    "rel_active_5",
    "preflop_unopened",
    "preflop_single_raised",
    "preflop_three_bet_plus",
    "hero_is_last_aggressor",
    "high_hole_rank_scalar",
    "low_hole_rank_scalar",
    *OPPONENT_PROFILE_FEATURE_NAMES_V23,
]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _coerce_opponent_profile_slot(slot_profile: Optional[Sequence[float] | Dict[str, float]]) -> np.ndarray:
    slot = np.zeros(OPPONENT_PROFILE_PER_SLOT_DIM_V23, dtype=np.float32)
    if slot_profile is None:
        return slot

    if isinstance(slot_profile, dict):
        for idx, name in enumerate(PER_OPPONENT_PROFILE_FEATURE_NAMES_V23):
            slot[idx] = _clamp01(float(slot_profile.get(name, 0.0)))
        return slot

    values: Sequence[float] = slot_profile
    if not isinstance(values, (list, tuple, np.ndarray)):
        return slot
    for idx in range(min(OPPONENT_PROFILE_PER_SLOT_DIM_V23, len(values))):
        slot[idx] = _clamp01(float(values[idx]))
    return slot


def _coerce_opponent_profile(opponent_profile: Optional[Sequence[float] | Dict[str, float]]) -> np.ndarray:
    profile = np.zeros(OPPONENT_PROFILE_DIM_V23, dtype=np.float32)
    if opponent_profile is None:
        return profile

    if isinstance(opponent_profile, dict):
        has_relative_keys = any(name in opponent_profile for name in OPPONENT_PROFILE_FEATURE_NAMES_V23)
        if has_relative_keys:
            for idx, name in enumerate(OPPONENT_PROFILE_FEATURE_NAMES_V23):
                profile[idx] = _clamp01(float(opponent_profile.get(name, 0.0)))
            return profile
        slot = _coerce_opponent_profile_slot(opponent_profile)
        for slot_idx in range(RELATIVE_OPPONENT_SLOTS_V23):
            start = slot_idx * OPPONENT_PROFILE_PER_SLOT_DIM_V23
            end = start + OPPONENT_PROFILE_PER_SLOT_DIM_V23
            profile[start:end] = slot
        return profile

    values: Sequence[float] = opponent_profile
    if not isinstance(values, (list, tuple, np.ndarray)):
        return profile
    length = len(values)
    if length >= OPPONENT_PROFILE_DIM_V23:
        for idx in range(OPPONENT_PROFILE_DIM_V23):
            profile[idx] = _clamp01(float(values[idx]))
        return profile
    if length == OPPONENT_PROFILE_PER_SLOT_DIM_V23:
        slot = _coerce_opponent_profile_slot(values)
        for slot_idx in range(RELATIVE_OPPONENT_SLOTS_V23):
            start = slot_idx * OPPONENT_PROFILE_PER_SLOT_DIM_V23
            end = start + OPPONENT_PROFILE_PER_SLOT_DIM_V23
            profile[start:end] = slot
        return profile
    for idx in range(min(OPPONENT_PROFILE_DIM_V23, length)):
        profile[idx] = _clamp01(float(values[idx]))
    return profile


def _coerce_opponent_profiles_by_seat(
    opponent_profiles_by_seat: Optional[Dict[int, Sequence[float] | Dict[str, float]]],
    hero_seat: int,
    player_count: int,
) -> np.ndarray:
    profile = np.zeros(OPPONENT_PROFILE_DIM_V23, dtype=np.float32)
    if not isinstance(opponent_profiles_by_seat, dict) or player_count <= 0:
        return profile

    write_idx = 0
    for rel_offset in range(1, RELATIVE_OPPONENT_SLOTS_V23 + 1):
        seat = (int(hero_seat) + rel_offset) % int(player_count)
        slot = _coerce_opponent_profile_slot(opponent_profiles_by_seat.get(seat))
        end_idx = write_idx + OPPONENT_PROFILE_PER_SLOT_DIM_V23
        profile[write_idx:end_idx] = slot
        write_idx = end_idx
    return profile

def flatten_cards_list(items) -> List[Card]:
    if isinstance(items, Card):
        return [items]
    if not items:
        return []
    if isinstance(items, (list, tuple)) and items and isinstance(items[0], Card):
        return list(items)

    out: List[Card] = []
    stack = [items]
    while stack:
        current = stack.pop()
        if isinstance(current, Card):
            out.append(current)
            continue
        if isinstance(current, (list, tuple)):
            for value in reversed(current):
                stack.append(value)
    return out


def street_from_board_len(board_len: int) -> int:
    if board_len <= 0:
        return 0
    if board_len == 3:
        return 1
    if board_len == 4:
        return 2
    return 3


def _rank_index(card: Card) -> int:
    rank = getattr(card.rank, "value", card.rank)
    return RANK_ORDER.index(rank)


def _suit_index(card: Card) -> int:
    suit = getattr(card.suit, "value", card.suit)
    return SUIT_ORDER.index(suit)


def _normalize(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return float(max(0.0, min(cap, value)) / cap)


def estimate_preflop_strength(hole_cards: List[Card], num_opponents: int = 1) -> float:
    hole_cards = flatten_cards_list(hole_cards)
    if len(hole_cards) != 2:
        return 0.35

    r1 = _rank_index(hole_cards[0])
    r2 = _rank_index(hole_cards[1])
    hi = max(r1, r2) / 12.0
    lo = min(r1, r2) / 12.0
    pair = 1.0 if r1 == r2 else 0.0
    suited = 1.0 if _suit_index(hole_cards[0]) == _suit_index(hole_cards[1]) else 0.0
    gap = abs(r1 - r2)
    connected = 1.0 if gap == 1 else 0.0
    one_gap = 1.0 if gap == 2 else 0.0
    broadway = 1.0 if max(r1, r2) >= 8 and min(r1, r2) >= 7 else 0.0

    # Deterministic scalar (0..1): favors high cards, pairs, suitedness and connectivity.
    base = (
        0.08
        + (0.40 * hi)
        + (0.16 * lo)
        + (0.22 * pair)
        + (0.08 * suited)
        + (0.04 * connected)
        + (0.02 * one_gap)
        + (0.03 * broadway)
    )
    opponent_penalty = max(0, num_opponents - 1) * 0.035
    return float(max(0.0, min(0.99, base - opponent_penalty)))


def _gap_bucket(hole_cards: List[Card]) -> np.ndarray:
    bucket = np.zeros(4, dtype=np.float32)
    if len(hole_cards) != 2:
        bucket[3] = 1.0
        return bucket
    ranks = sorted(_rank_index(card) for card in hole_cards)
    low, high = ranks
    if high == 12 and low == 0:
        gap = 1
    else:
        gap = high - low
    if gap == 1:
        bucket[0] = 1.0
    elif gap == 2:
        bucket[1] = 1.0
    elif gap == 3:
        bucket[2] = 1.0
    else:
        bucket[3] = 1.0
    return bucket


def _has_flush_draw(cards: List[Card]) -> float:
    suit_counts = [0, 0, 0, 0]
    for card in cards:
        suit_counts[_suit_index(card)] += 1
    return 1.0 if max(suit_counts, default=0) >= 4 else 0.0


def _has_straight_draw(cards: List[Card]) -> float:
    if len(cards) < 4:
        return 0.0
    ranks = {_rank_index(card) for card in cards}
    if 12 in ranks:
        ranks.add(-1)
    ordered = sorted(ranks)
    for start_idx in range(len(ordered)):
        window = ordered[start_idx : start_idx + 4]
        if len(window) == 4 and window[-1] - window[0] <= 4:
            return 1.0
    return 0.0


def _active_flags(state, hand_ctx) -> List[bool]:
    if hand_ctx is not None and hasattr(hand_ctx, "in_hand"):
        return list(hand_ctx.in_hand)
    statuses = getattr(state, "statuses", None)
    if statuses is not None:
        return [bool(flag) for flag in statuses]
    return [True] * len(getattr(state, "stacks", []))


def _legal_actions(state) -> List[int]:
    legal: List[int] = []
    to_call = max(state.bets) - state.bets[state.actor_index]
    if to_call > 0:
        if state.can_fold():
            legal.append(ACTION_FOLD)
        if state.can_check_or_call():
            legal.append(ACTION_CALL)
    else:
        if state.can_check_or_call():
            legal.append(ACTION_CHECK)
    if state.can_complete_bet_or_raise_to():
        legal.append(ACTION_RAISE_HALF_POT)
        legal.append(ACTION_RAISE_POT_OR_ALL_IN)
    return legal


def _get_raise_bounds(state) -> tuple:
    min_raise = getattr(state, "min_completion_betting_or_raising_to_amount", None)
    max_raise = getattr(state, "max_completion_betting_or_raising_to_amount", None)
    if min_raise is None:
        min_raise = getattr(state, "min_completion_betting_or_raising_to", 0)
    if max_raise is None:
        max_raise = getattr(state, "max_completion_betting_or_raising_to", 0)
    return float(min_raise or 0.0), float(max_raise or 0.0)


def build_legal_action_mask(state, hero_seat: int, hand_ctx) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
    actor = getattr(state, "actor_index", None)
    if actor is None or actor != hero_seat:
        return mask
    for action in _legal_actions(state):
        mask[action] = 1.0
    return mask


def _board_rank_hist(board: List[Card]) -> np.ndarray:
    hist = np.zeros(13, dtype=np.float32)
    if not board:
        return hist
    for card in board:
        hist[_rank_index(card)] += 1.0
    hist /= float(len(board))
    return hist


def _board_suit_hist(board: List[Card]) -> np.ndarray:
    hist = np.zeros(4, dtype=np.float32)
    if not board:
        return hist
    for card in board:
        hist[_suit_index(card)] += 1.0
    hist /= float(len(board))
    return hist


def _board_connected(board: List[Card]) -> float:
    if len(board) < 3:
        return 0.0
    ranks = sorted(set(_rank_index(card) for card in board))
    if 12 in ranks:
        ranks = sorted(set(ranks + [-1]))
    for start in range(len(ranks)):
        window = ranks[start : start + 3]
        if len(window) == 3 and window[-1] - window[0] <= 4:
            return 1.0
    return 0.0


def _current_hand_strength_scalar(hole_cards: List[Card], board_cards: List[Card]) -> float:
    hole_cards = flatten_cards_list(hole_cards)
    board_cards = flatten_cards_list(board_cards)
    if len(hole_cards) != 2:
        return 0.0
    if len(board_cards) < 3:
        active_opponents = 1
        return estimate_preflop_strength(hole_cards, num_opponents=active_opponents)

    cards = hole_cards + board_cards
    rank_counts = [0] * 13
    suit_counts = [0] * 4
    unique_ranks = set()
    for card in cards:
        rank_idx = _rank_index(card)
        suit_idx = _suit_index(card)
        rank_counts[rank_idx] += 1
        suit_counts[suit_idx] += 1
        unique_ranks.add(rank_idx)

    max_rank = max(rank_counts)
    pair_count = 0
    has_trips = False
    has_quads = False
    for count in rank_counts:
        if count >= 2:
            pair_count += 1
        if count >= 3:
            has_trips = True
        if count >= 4:
            has_quads = True
    has_flush = max(suit_counts) >= 5

    if 12 in unique_ranks:
        unique_ranks.add(-1)
    ordered = sorted(unique_ranks)
    has_straight = False
    for idx in range(max(0, len(ordered) - 4)):
        window = ordered[idx : idx + 5]
        if len(window) == 5 and window[-1] - window[0] == 4:
            has_straight = True
            break

    if has_flush and has_straight:
        return 0.995
    if has_quads:
        return 0.96
    if has_trips and pair_count >= 2:
        return 0.90
    if has_flush:
        return 0.82
    if has_straight:
        return 0.74
    if has_trips:
        return 0.64
    if pair_count >= 2:
        return 0.50
    if max_rank >= 2:
        return 0.34
    return 0.18


def encode_info_state(
    state,
    hero_seat: int,
    hand_ctx,
    return_legal_mask: bool = False,
    opponent_profile: Optional[Sequence[float] | Dict[str, float]] = None,
    opponent_profiles_by_seat: Optional[Dict[int, Sequence[float] | Dict[str, float]]] = None,
):
    hole_cards = flatten_cards_list(state.hole_cards[hero_seat])
    board_cards = flatten_cards_list(state.board_cards)
    active_flags = _active_flags(state, hand_ctx)
    player_count = len(state.stacks)

    features = np.zeros(STATE_DIM_V21, dtype=np.float32)

    if len(hole_cards) == 2:
        rank_indices = sorted((_rank_index(card) for card in hole_cards))
        low_rank = rank_indices[0]
        high_rank = rank_indices[1]
        features[high_rank] = 1.0
        features[13 + low_rank] = 1.0
        features[26] = 1.0 if _suit_index(hole_cards[0]) == _suit_index(hole_cards[1]) else 0.0
        features[27] = 1.0 if low_rank == high_rank else 0.0
        features[28:32] = _gap_bucket(hole_cards)
        # Explicit scalar rank signal so the model can directly distinguish A > K > Q ...
        features[96] = float(high_rank / 12.0)
        features[97] = float(low_rank / 12.0)

    board_len = len(board_cards)
    street_idx = street_from_board_len(board_len)
    features[49 + street_idx] = 1.0

    board_rank_counts = np.zeros(13, dtype=np.int32)
    board_suit_counts = np.zeros(4, dtype=np.int32)
    for card in board_cards:
        board_rank_counts[_rank_index(card)] += 1
        board_suit_counts[_suit_index(card)] += 1
    if board_len > 0:
        inv_len = 1.0 / float(board_len)
        features[32:45] = board_rank_counts.astype(np.float32) * inv_len
        features[45:49] = board_suit_counts.astype(np.float32) * inv_len
    features[53] = 1.0 if np.any(board_rank_counts >= 2) else 0.0
    features[54] = 1.0 if np.any(board_rank_counts >= 3) else 0.0
    features[55] = 1.0 if board_len >= 3 and np.any(board_suit_counts == board_len) else 0.0
    features[56] = 1.0 if board_len >= 3 and np.count_nonzero(board_suit_counts) == 2 else 0.0
    features[57] = _board_connected(board_cards)
    active_opponents = max(1, sum(1 for idx, flag in enumerate(active_flags) if flag and idx != hero_seat))
    features[58] = estimate_preflop_strength(hole_cards, num_opponents=active_opponents)
    features[59] = _has_flush_draw(hole_cards + board_cards)
    features[60] = _current_hand_strength_scalar(hole_cards, board_cards)

    features[61 + hero_seat] = 1.0

    active_count = sum(1 for flag in active_flags if flag)
    players_before = sum(1 for idx in range(hero_seat) if active_flags[idx])
    players_after = sum(1 for idx in range(hero_seat + 1, player_count) if active_flags[idx])
    features[67] = _normalize(float(active_count), 6.0)
    features[68] = _normalize(float(players_before), 5.0)
    features[69] = _normalize(float(players_after), 5.0)

    total_pot = float(sum(pot.amount for pot in getattr(state, "pots", [])) + sum(state.bets))
    current_bet = float(max(state.bets))
    hero_bet = float(state.bets[hero_seat])
    to_call = max(0.0, current_bet - hero_bet)
    hero_stack = float(state.stacks[hero_seat])

    opponent_stacks = [
        float(state.stacks[idx])
        for idx in range(player_count)
        if idx != hero_seat and active_flags[idx]
    ]
    effective_stack = min([hero_stack] + opponent_stacks) if opponent_stacks else hero_stack
    big_blind = float(getattr(hand_ctx, "big_blind", 10.0))
    min_raise_to = 0.0
    if hero_seat == getattr(state, "actor_index", None) and state.can_complete_bet_or_raise_to():
        min_raise_to, _ = _get_raise_bounds(state)

    features[70] = _normalize(total_pot / big_blind, 200.0)
    features[71] = _normalize(to_call / big_blind, 50.0)
    features[72] = _normalize(min_raise_to / big_blind, 200.0)
    features[73] = _normalize(hero_stack / big_blind, 200.0)
    features[74] = _normalize(effective_stack / big_blind, 200.0)
    spr = hero_stack / max(total_pot, big_blind)
    features[75] = _normalize(spr, 20.0)
    features[76] = float(to_call / max(total_pot + to_call, 1.0))

    contributions = getattr(hand_ctx, "contributions", [0.0] * player_count)
    hero_contrib = float(contributions[hero_seat]) if hero_seat < len(contributions) else hero_bet
    features[77] = float(hero_contrib / max(hero_contrib + hero_stack, 1.0))
    features[78] = _normalize(float(getattr(hand_ctx, "street_raise_count", 0)), 4.0)
    features[79] = _normalize(float(getattr(hand_ctx, "last_aggressive_size_bb", 0.0)), 20.0)
    features[80] = _normalize(hero_contrib / big_blind, 200.0)

    legal_mask = build_legal_action_mask(state, hero_seat, hand_ctx)
    features[81:86] = legal_mask

    rel_flags = np.zeros(6, dtype=np.float32)
    for offset in range(6):
        seat = (hero_seat + offset) % player_count
        rel_flags[offset] = 1.0 if active_flags[seat] else 0.0
    features[86:92] = rel_flags

    preflop_raise_count = int(getattr(hand_ctx, "preflop_raise_count", 0))
    features[92] = 1.0 if street_idx == 0 and preflop_raise_count == 0 else 0.0
    features[93] = 1.0 if street_idx == 0 and preflop_raise_count == 1 else 0.0
    features[94] = 1.0 if street_idx == 0 and preflop_raise_count >= 2 else 0.0
    features[95] = 1.0 if getattr(hand_ctx, "last_aggressor", None) == hero_seat else 0.0
    if opponent_profiles_by_seat is not None:
        features[BASE_STATE_DIM_V23:] = _coerce_opponent_profiles_by_seat(
            opponent_profiles_by_seat,
            hero_seat,
            player_count,
        )
    else:
        features[BASE_STATE_DIM_V23:] = _coerce_opponent_profile(opponent_profile)

    if VALIDATE_INFO_STATE_V21:
        validate_info_state(features, legal_mask)
    if return_legal_mask:
        return features, legal_mask
    return features


def validate_info_state(vec: np.ndarray, legal_mask: np.ndarray) -> None:
    if vec.shape != (STATE_DIM_V21,):
        raise ValueError(f"Expected state vector of shape {(STATE_DIM_V21,)}, got {vec.shape}")
    if legal_mask.shape != (ACTION_COUNT_V21,):
        raise ValueError(f"Expected legal mask of shape {(ACTION_COUNT_V21,)}, got {legal_mask.shape}")
    if not np.isfinite(vec).all():
        raise ValueError("State vector contains non-finite values")
    if not np.isfinite(legal_mask).all():
        raise ValueError("Legal mask contains non-finite values")
    if (vec < -1e-6).any() or (vec > 1.000001).any():
        raise ValueError("State vector contains values outside the normalized range [0, 1]")
    if (legal_mask < -1e-6).any() or (legal_mask > 1.000001).any():
        raise ValueError("Legal mask contains values outside [0, 1]")


def debug_feature_map(vec: np.ndarray) -> Dict[str, float]:
    validate_info_state(vec, np.zeros(ACTION_COUNT_V21, dtype=np.float32))
    return {name: float(vec[idx]) for idx, name in enumerate(FEATURE_NAMES_V21)}
