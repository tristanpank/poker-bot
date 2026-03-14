import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from pokerkit import Card

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preflop_blueprint_v24 import canonical_preflop_hand_key, combo_percentile_for_hand_key

BASE_STATE_DIM_V24 = 80
ACTION_COUNT_V24 = 5
ACTION_COUNT_V21 = ACTION_COUNT_V24
VALIDATE_INFO_STATE_V24 = os.getenv("POKER_V24_VALIDATE_INFO_STATE", "0").strip() in {"1", "true", "True"}
VALIDATE_INFO_STATE_V21 = VALIDATE_INFO_STATE_V24

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_AGGRO_SMALL = 3
ACTION_AGGRO_LARGE = 4

ACTION_RAISE_33_POT = ACTION_AGGRO_SMALL
ACTION_RAISE_66_POT = ACTION_AGGRO_SMALL
ACTION_RAISE_POT = ACTION_AGGRO_LARGE
ACTION_RAISE_133_POT = ACTION_AGGRO_LARGE
ACTION_ALL_IN = ACTION_AGGRO_LARGE
ACTION_RAISE_POT_OR_ALL_IN = ACTION_AGGRO_LARGE

ACTION_NAMES_V24 = ["Fold", "Check", "Call", "Aggro Small", "Aggro Large"]
ACTION_NAMES_V21 = ACTION_NAMES_V24
NON_ALL_IN_RAISE_ACTIONS = (ACTION_AGGRO_SMALL, ACTION_AGGRO_LARGE)
ALL_RAISE_ACTIONS = NON_ALL_IN_RAISE_ACTIONS
SMALL_RAISE_ACTIONS = (ACTION_AGGRO_SMALL,)
MEDIUM_RAISE_ACTIONS = (ACTION_AGGRO_LARGE,)
LARGE_RAISE_ACTIONS = (ACTION_AGGRO_LARGE,)

DEEP_STACK_PREFLOP_ALL_IN_MAX_EFFECTIVE_BB = 40.0
PREFLOP_OPEN_RAISE_TO_BB = {
    ACTION_AGGRO_SMALL: 2.25,
    ACTION_AGGRO_LARGE: 2.50,
}
POSTFLOP_BET_POT_MULTIPLIERS = {
    ACTION_AGGRO_SMALL: 0.50,
    ACTION_AGGRO_LARGE: 1.00,
}
FACING_BET_RAISE_TO_MULTIPLIERS = {
    ACTION_AGGRO_SMALL: 2.50,
    ACTION_AGGRO_LARGE: 3.50,
}

POSITION_NAMES_V21 = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
RANK_ORDER = "23456789TJQKA"
SUIT_ORDER = "cdhs"
POSTFLOP_POSITION_RANK = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

PER_OPPONENT_PROFILE_FEATURE_NAMES_V24 = [
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
OPPONENT_PROFILE_PER_SLOT_DIM_V24 = len(PER_OPPONENT_PROFILE_FEATURE_NAMES_V24)
OPPONENT_PROFILE_FEATURE_NAMES_V24 = [f"opp_{name}" for name in PER_OPPONENT_PROFILE_FEATURE_NAMES_V24]
OPPONENT_PROFILE_DIM_V24 = len(OPPONENT_PROFILE_FEATURE_NAMES_V24)
OPPONENT_PROFILE_DEFAULT_SLOT_V24 = tuple(0.0 for _ in range(OPPONENT_PROFILE_PER_SLOT_DIM_V24))
OPPONENT_PROFILE_DEFAULT_V24 = tuple(0.0 for _ in range(OPPONENT_PROFILE_DIM_V24))

STATE_DIM_V24 = BASE_STATE_DIM_V24 + OPPONENT_PROFILE_DIM_V24
STATE_DIM_V21 = STATE_DIM_V24
PUBLIC_BELIEF_STATE_DIM_V24 = STATE_DIM_V24
PRIVATE_INFO_STATE_INDICES_V24 = tuple()

STREET_SLICE_V24 = slice(0, 4)
POSITION_SLICE_V24 = slice(4, 10)
ACTIVE_PLAYERS_SLICE_V24 = slice(10, 16)
PLAYERS_BEHIND_SLICE_V24 = slice(16, 22)
POT_BUCKET_SLICE_V24 = slice(22, 28)
SPR_BUCKET_SLICE_V24 = slice(28, 34)
TO_CALL_BUCKET_SLICE_V24 = slice(34, 40)
FACING_BET_BUCKET_SLICE_V24 = slice(40, 45)
RAISE_COUNT_SLICE_V24 = slice(45, 49)
PREPERCENTILE_SLICE_V24 = slice(49, 57)
HAND_CLASS_SLICE_V24 = slice(57, 65)
BOARD_TEXTURE_SLICE_V24 = slice(65, 69)
DRAW_FLAGS_SLICE_V24 = slice(69, 71)
SCALAR_SLICE_V24 = slice(71, 80)
OPPONENT_PROFILE_SLICE_V24 = slice(BASE_STATE_DIM_V24, STATE_DIM_V24)

IDX_LAST_AGGRESSOR_FLAG_V24 = 71
IDX_IN_POSITION_FLAG_V24 = 72
IDX_POT_ODDS_V24 = 73
IDX_HERO_COMMITMENT_V24 = 74
IDX_EFFECTIVE_STACK_V24 = 75
IDX_ACTIVE_PLAYERS_NORM_V24 = 76
IDX_TO_CALL_NORM_V24 = 77
IDX_POT_SIZE_NORM_V24 = 78
IDX_FACING_BET_FLAG_V24 = 79

HAND_CLASS_NAMES_V24 = (
    "air",
    "draw",
    "pair",
    "two_pair",
    "trips",
    "straight",
    "flush",
    "full_house_plus",
)

FEATURE_NAMES_V24 = (
    [f"street_{name}" for name in ("preflop", "flop", "turn", "river")]
    + [f"pos_{name.lower()}" for name in POSITION_NAMES_V21]
    + [f"active_players_{count}" for count in range(1, 7)]
    + [f"players_behind_{count}" for count in range(6)]
    + [f"pot_bucket_{idx}" for idx in range(POT_BUCKET_SLICE_V24.stop - POT_BUCKET_SLICE_V24.start)]
    + [f"spr_bucket_{idx}" for idx in range(SPR_BUCKET_SLICE_V24.stop - SPR_BUCKET_SLICE_V24.start)]
    + [f"to_call_bucket_{idx}" for idx in range(TO_CALL_BUCKET_SLICE_V24.stop - TO_CALL_BUCKET_SLICE_V24.start)]
    + [f"facing_bucket_{idx}" for idx in range(FACING_BET_BUCKET_SLICE_V24.stop - FACING_BET_BUCKET_SLICE_V24.start)]
    + [f"raise_count_{idx}" for idx in range(RAISE_COUNT_SLICE_V24.stop - RAISE_COUNT_SLICE_V24.start)]
    + [f"preflop_percentile_{idx}" for idx in range(PREPERCENTILE_SLICE_V24.stop - PREPERCENTILE_SLICE_V24.start)]
    + [f"hand_class_{name}" for name in HAND_CLASS_NAMES_V24]
    + ["board_paired", "board_monotone", "board_two_tone", "board_connected"]
    + ["flush_draw", "straight_draw"]
    + [
        "hero_is_last_aggressor",
        "hero_in_position",
        "pot_odds",
        "hero_commitment",
        "effective_stack_norm",
        "active_players_norm",
        "to_call_norm",
        "pot_size_norm",
        "facing_bet_flag",
    ]
    + OPPONENT_PROFILE_FEATURE_NAMES_V24
)
FEATURE_NAMES_V21 = FEATURE_NAMES_V24
FEATURE_INDEX_V24 = {name: idx for idx, name in enumerate(FEATURE_NAMES_V24)}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _rank_index(card: Card) -> int:
    rank = str(getattr(card.rank, "value", card.rank))
    return RANK_ORDER.index(rank)


def _suit_index(card: Card) -> int:
    suit = str(getattr(card.suit, "value", card.suit))
    return SUIT_ORDER.index(suit)


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


def _active_flags(state, hand_ctx) -> List[bool]:
    if hand_ctx is not None and hasattr(hand_ctx, "in_hand"):
        return list(hand_ctx.in_hand)
    statuses = getattr(state, "statuses", None)
    if statuses is not None:
        return [bool(flag) for flag in statuses]
    return [True] * len(getattr(state, "stacks", []))


def _effective_stack_bb(state, actor: int, hand_ctx) -> float:
    active_flags = _active_flags(state, hand_ctx)
    hero_stack = float(state.stacks[int(actor)])
    opponent_stacks = [
        float(state.stacks[idx])
        for idx, is_active in enumerate(active_flags)
        if idx != int(actor) and is_active
    ]
    effective_stack = min([hero_stack] + opponent_stacks) if opponent_stacks else hero_stack
    big_blind = float(max(1.0, getattr(hand_ctx, "big_blind", 10.0)))
    return float(effective_stack / big_blind)


def _all_in_allowed(state, hand_ctx) -> bool:
    actor = getattr(state, "actor_index", None)
    if actor is None:
        return False
    if int(getattr(hand_ctx, "current_street", 0)) != 0:
        return True
    effective_stack_bb = _effective_stack_bb(state, int(actor), hand_ctx)
    return effective_stack_bb <= DEEP_STACK_PREFLOP_ALL_IN_MAX_EFFECTIVE_BB or int(
        getattr(hand_ctx, "preflop_raise_count", 0)
    ) >= 2


def _legal_actions(state, hand_ctx=None) -> List[int]:
    legal: List[int] = []
    actor = getattr(state, "actor_index", None)
    if actor is None:
        return legal
    to_call = max(state.bets) - state.bets[actor]
    if to_call > 0:
        if state.can_fold():
            legal.append(ACTION_FOLD)
        if state.can_check_or_call():
            legal.append(ACTION_CALL)
    else:
        if state.can_check_or_call():
            legal.append(ACTION_CHECK)
    if state.can_complete_bet_or_raise_to():
        legal.extend(_legal_raise_actions(state, hand_ctx))
    return legal


def _get_raise_bounds(state) -> tuple[float, float]:
    min_raise = getattr(state, "min_completion_betting_or_raising_to_amount", None)
    max_raise = getattr(state, "max_completion_betting_or_raising_to_amount", None)
    if min_raise is None:
        min_raise = getattr(state, "min_completion_betting_or_raising_to", 0)
    if max_raise is None:
        max_raise = getattr(state, "max_completion_betting_or_raising_to", 0)
    return float(min_raise or 0.0), float(max_raise or 0.0)


def _is_in_position(actor_seat: int, aggressor_seat: Optional[int]) -> bool:
    if aggressor_seat is None:
        return False
    return int(POSTFLOP_POSITION_RANK.get(int(actor_seat), 0)) > int(
        POSTFLOP_POSITION_RANK.get(int(aggressor_seat), 0)
    )


def _estimated_preflop_limpers(hand_ctx) -> int:
    return max(0, int(getattr(hand_ctx, "preflop_call_count", 0)))


def abstract_raise_target(state, action_id: int, hand_ctx=None) -> Optional[int]:
    if not state.can_complete_bet_or_raise_to():
        return None
    actor = getattr(state, "actor_index", None)
    if actor is None:
        return None
    if hand_ctx is None:
        hand_ctx = type(
            "RaiseContext",
            (),
            {
                "current_street": street_from_board_len(len(flatten_cards_list(state.board_cards))),
                "preflop_raise_count": 0,
                "preflop_call_count": 0,
                "preflop_last_raiser": None,
                "big_blind": 10.0,
            },
        )()
    min_raise, max_raise = _get_raise_bounds(state)
    if max_raise <= 0.0 or max_raise < min_raise:
        return None

    current_bet = float(state.bets[actor])
    highest_bet = float(max(state.bets))
    to_call = max(0.0, highest_bet - current_bet)
    pot = float(sum(pot_item.amount for pot_item in getattr(state, "pots", [])) + sum(state.bets))
    big_blind = float(max(1.0, getattr(hand_ctx, "big_blind", 10.0)))
    street = int(getattr(hand_ctx, "current_street", 0))
    preflop_raises = int(getattr(hand_ctx, "preflop_raise_count", 0))

    if street == 0 and preflop_raises == 0:
        base_size_bb = PREFLOP_OPEN_RAISE_TO_BB.get(int(action_id))
        if base_size_bb is None:
            return None
        limper_bonus = min(1.0, 0.5 * float(_estimated_preflop_limpers(hand_ctx)))
        target = (base_size_bb + limper_bonus) * big_blind
    elif street == 0:
        aggressor = getattr(hand_ctx, "preflop_last_raiser", None)
        in_position = _is_in_position(int(actor), aggressor)
        effective_stack_bb = _effective_stack_bb(state, int(actor), hand_ctx)
        if int(action_id) == ACTION_AGGRO_LARGE and (
            preflop_raises >= 2 or effective_stack_bb <= DEEP_STACK_PREFLOP_ALL_IN_MAX_EFFECTIVE_BB
        ):
            target = float(max_raise)
        else:
            multiplier = 3.5 if in_position else 4.5
            if int(action_id) == ACTION_AGGRO_LARGE:
                multiplier += 0.75
            target = highest_bet * multiplier
    elif to_call <= 1e-6:
        multiplier = POSTFLOP_BET_POT_MULTIPLIERS.get(int(action_id))
        if multiplier is None:
            return None
        target = current_bet + (multiplier * max(pot, big_blind))
    else:
        multiplier = FACING_BET_RAISE_TO_MULTIPLIERS.get(int(action_id))
        if multiplier is None:
            return None
        target = highest_bet * multiplier

    target = max(min_raise, min(max_raise, float(target)))
    if int(action_id) == ACTION_AGGRO_LARGE and street == 0:
        effective_stack_bb = _effective_stack_bb(state, int(actor), hand_ctx)
        if effective_stack_bb <= DEEP_STACK_PREFLOP_ALL_IN_MAX_EFFECTIVE_BB and _all_in_allowed(state, hand_ctx):
            target = float(max_raise)
    return int(round(target))


def _legal_raise_actions(state, hand_ctx=None) -> List[int]:
    if not state.can_complete_bet_or_raise_to():
        return []
    _, max_raise = _get_raise_bounds(state)
    if int(round(max_raise)) <= 0:
        return []
    actions: List[int] = []
    seen_targets = set()
    for action_id in (ACTION_AGGRO_SMALL, ACTION_AGGRO_LARGE):
        target = abstract_raise_target(state, action_id, hand_ctx)
        if target is None or target in seen_targets:
            continue
        seen_targets.add(target)
        actions.append(action_id)
    return actions


def summarize_legal_action_mask(legal_mask: Sequence[float]) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    summary = np.zeros(5, dtype=np.float32)
    for idx in range(min(summary.shape[0], mask.shape[0])):
        summary[idx] = 1.0 if float(mask[idx]) > 0.5 else 0.0
    return summary


def build_legal_action_mask(state, hero_seat: int, hand_ctx) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
    actor = getattr(state, "actor_index", None)
    if actor is None or int(actor) != int(hero_seat):
        return mask
    for action in _legal_actions(state, hand_ctx):
        mask[int(action)] = 1.0
    return mask


def _bucket_index(value: float, cutoffs: Sequence[float]) -> int:
    numeric = float(value)
    for idx, cutoff in enumerate(cutoffs):
        if numeric < float(cutoff):
            return idx
    return len(cutoffs)


def _set_bucket(vec: np.ndarray, bucket_slice: slice, value: float, cutoffs: Sequence[float]) -> None:
    width = bucket_slice.stop - bucket_slice.start
    if width <= 0:
        return
    idx = min(width - 1, _bucket_index(value, cutoffs))
    vec[bucket_slice.start + idx] = 1.0


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
    for idx in range(len(ordered)):
        window = ordered[idx : idx + 4]
        if len(window) == 4 and window[-1] - window[0] <= 4:
            return 1.0
    return 0.0


def _board_connected(board_cards: List[Card]) -> float:
    if len(board_cards) < 3:
        return 0.0
    ranks = sorted(set(_rank_index(card) for card in board_cards))
    if 12 in ranks:
        ranks = sorted(set(ranks + [-1]))
    for idx in range(len(ranks)):
        window = ranks[idx : idx + 3]
        if len(window) == 3 and window[-1] - window[0] <= 4:
            return 1.0
    return 0.0


def _best_hand_class(hole_cards: List[Card], board_cards: List[Card]) -> tuple[int, float]:
    cards = hole_cards + board_cards
    if len(hole_cards) != 2:
        return 0, 0.0
    if len(board_cards) < 3:
        return 0, max(0.05, 1.0 - combo_percentile_for_hand_key(canonical_preflop_hand_key(hole_cards)))

    rank_counts = [0] * 13
    suit_counts = [0] * 4
    unique_ranks = set()
    for card in cards:
        rank_idx = _rank_index(card)
        suit_idx = _suit_index(card)
        rank_counts[rank_idx] += 1
        suit_counts[suit_idx] += 1
        unique_ranks.add(rank_idx)

    pair_count = 0
    trips = False
    quads = False
    for count in rank_counts:
        if count >= 2:
            pair_count += 1
        if count >= 3:
            trips = True
        if count >= 4:
            quads = True
    flush = max(suit_counts) >= 5

    if 12 in unique_ranks:
        unique_ranks.add(-1)
    ordered = sorted(unique_ranks)
    straight = False
    for idx in range(max(0, len(ordered) - 4)):
        window = ordered[idx : idx + 5]
        if len(window) == 5 and window[-1] - window[0] == 4:
            straight = True
            break

    if quads or (trips and pair_count >= 2):
        return 7, 0.96
    if flush:
        return 6, 0.86
    if straight:
        return 5, 0.78
    if trips:
        return 4, 0.66
    if pair_count >= 2:
        return 3, 0.56
    if max(rank_counts) >= 2:
        return 2, 0.44
    if _has_flush_draw(cards) > 0.5 or _has_straight_draw(cards) > 0.5:
        return 1, 0.34
    return 0, 0.18


def estimate_preflop_strength(hole_cards: List[Card], num_opponents: int = 1) -> float:
    hole_cards = flatten_cards_list(hole_cards)
    if len(hole_cards) != 2:
        return 0.35
    percentile = combo_percentile_for_hand_key(canonical_preflop_hand_key(hole_cards))
    strength = max(0.02, 1.0 - float(percentile))
    opponent_penalty = max(0, int(num_opponents) - 1) * 0.03
    return float(max(0.02, min(0.98, strength - opponent_penalty)))


def _coerce_opponent_profile(opponent_profile: Optional[Sequence[float] | Dict[str, float]]) -> np.ndarray:
    profile = np.zeros(OPPONENT_PROFILE_DIM_V24, dtype=np.float32)
    if opponent_profile is None:
        return profile
    if isinstance(opponent_profile, dict):
        for idx, name in enumerate(OPPONENT_PROFILE_FEATURE_NAMES_V24):
            profile[idx] = _clamp01(float(opponent_profile.get(name, 0.0)))
        return profile
    if not isinstance(opponent_profile, (list, tuple, np.ndarray)):
        return profile
    for idx in range(min(len(profile), len(opponent_profile))):
        profile[idx] = _clamp01(float(opponent_profile[idx]))
    return profile


def _coerce_opponent_profiles_by_seat(
    opponent_profiles_by_seat: Optional[Dict[int, Sequence[float] | Dict[str, float]]],
    hero_seat: int,
    player_count: int,
    active_flags: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    if not isinstance(opponent_profiles_by_seat, dict):
        return np.zeros(OPPONENT_PROFILE_DIM_V24, dtype=np.float32)
    aggregate = np.zeros(OPPONENT_PROFILE_DIM_V24, dtype=np.float32)
    total = 0.0
    for rel_offset in range(1, int(player_count)):
        seat = (int(hero_seat) + rel_offset) % int(player_count)
        if active_flags is not None and seat < len(active_flags) and not bool(active_flags[seat]):
            continue
        slot = _coerce_opponent_profile(opponent_profiles_by_seat.get(seat))
        weight = max(0.05, float(slot[-1])) if len(slot) > 0 else 1.0
        aggregate += weight * slot
        total += weight
    if total <= 1e-8:
        return aggregate
    return (aggregate / total).astype(np.float32)


def encode_info_state(
    state,
    hero_seat: int,
    hand_ctx,
    return_legal_mask: bool = False,
    opponent_profile: Optional[Sequence[float] | Dict[str, float]] = None,
    opponent_profiles_by_seat: Optional[Dict[int, Sequence[float] | Dict[str, float]]] = None,
):
    hero_seat = int(hero_seat)
    hole_cards = flatten_cards_list(state.hole_cards[hero_seat])
    board_cards = flatten_cards_list(state.board_cards)
    active_flags = _active_flags(state, hand_ctx)
    player_count = len(state.stacks)
    legal_mask = build_legal_action_mask(state, hero_seat, hand_ctx)
    vec = np.zeros(STATE_DIM_V24, dtype=np.float32)

    street_idx = street_from_board_len(len(board_cards))
    vec[STREET_SLICE_V24.start + street_idx] = 1.0
    vec[POSITION_SLICE_V24.start + (hero_seat % len(POSITION_NAMES_V21))] = 1.0

    active_players = sum(1 for flag in active_flags if flag)
    players_behind = sum(1 for seat in range(player_count) if seat != hero_seat and active_flags[seat] and seat > hero_seat)
    _set_bucket(vec, ACTIVE_PLAYERS_SLICE_V24, float(active_players) - 1.0, [1, 2, 3, 4, 5])
    _set_bucket(vec, PLAYERS_BEHIND_SLICE_V24, float(players_behind), [1, 2, 3, 4, 5])

    total_pot = float(sum(pot.amount for pot in getattr(state, "pots", [])) + sum(state.bets))
    current_bet = float(max(state.bets))
    hero_bet = float(state.bets[hero_seat])
    to_call = max(0.0, current_bet - hero_bet)
    hero_stack = float(state.stacks[hero_seat])
    effective_stack_bb = _effective_stack_bb(state, hero_seat, hand_ctx)
    big_blind = float(max(1.0, getattr(hand_ctx, "big_blind", 10.0)))
    spr = hero_stack / max(total_pot, big_blind)
    facing_bet_size = (to_call / max(total_pot, big_blind)) if to_call > 1e-6 else 0.0

    _set_bucket(vec, POT_BUCKET_SLICE_V24, total_pot / big_blind, [4, 8, 16, 32, 64])
    _set_bucket(vec, SPR_BUCKET_SLICE_V24, spr, [1, 2.5, 5, 8, 12])
    _set_bucket(vec, TO_CALL_BUCKET_SLICE_V24, to_call / big_blind, [0.5, 1.5, 4, 8, 16])
    _set_bucket(vec, FACING_BET_BUCKET_SLICE_V24, facing_bet_size, [1e-6, 0.2, 0.45, 0.8])
    _set_bucket(
        vec,
        RAISE_COUNT_SLICE_V24,
        float(max(0, int(getattr(hand_ctx, "street_raise_count", 0)))),
        [1, 2, 3],
    )

    preflop_percentile = combo_percentile_for_hand_key(canonical_preflop_hand_key(hole_cards)) if len(hole_cards) == 2 else 1.0
    _set_bucket(vec, PREPERCENTILE_SLICE_V24, preflop_percentile, [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92])

    hand_class_idx, hand_strength = _best_hand_class(hole_cards, board_cards)
    vec[HAND_CLASS_SLICE_V24.start + hand_class_idx] = 1.0

    board_rank_counts = np.zeros(13, dtype=np.int32)
    board_suit_counts = np.zeros(4, dtype=np.int32)
    for card in board_cards:
        board_rank_counts[_rank_index(card)] += 1
        board_suit_counts[_suit_index(card)] += 1
    vec[BOARD_TEXTURE_SLICE_V24.start] = 1.0 if np.any(board_rank_counts >= 2) else 0.0
    vec[BOARD_TEXTURE_SLICE_V24.start + 1] = 1.0 if len(board_cards) >= 3 and np.any(board_suit_counts == len(board_cards)) else 0.0
    vec[BOARD_TEXTURE_SLICE_V24.start + 2] = 1.0 if len(board_cards) >= 3 and np.count_nonzero(board_suit_counts) == 2 else 0.0
    vec[BOARD_TEXTURE_SLICE_V24.start + 3] = _board_connected(board_cards)
    vec[DRAW_FLAGS_SLICE_V24.start] = _has_flush_draw(hole_cards + board_cards)
    vec[DRAW_FLAGS_SLICE_V24.start + 1] = _has_straight_draw(hole_cards + board_cards)

    vec[IDX_LAST_AGGRESSOR_FLAG_V24] = 1.0 if getattr(hand_ctx, "last_aggressor", None) == hero_seat else 0.0
    vec[IDX_IN_POSITION_FLAG_V24] = 1.0 if _is_in_position(hero_seat, getattr(hand_ctx, "last_aggressor", None)) else 0.0
    vec[IDX_POT_ODDS_V24] = float(to_call / max(total_pot + to_call, 1.0))
    hero_contrib = float(
        getattr(hand_ctx, "contributions", [hero_bet] * max(1, player_count))[hero_seat]
        if hero_seat < len(getattr(hand_ctx, "contributions", []))
        else hero_bet
    )
    vec[IDX_HERO_COMMITMENT_V24] = float(hero_contrib / max(hero_contrib + hero_stack, 1.0))
    vec[IDX_EFFECTIVE_STACK_V24] = _clamp01(effective_stack_bb / 120.0)
    vec[IDX_ACTIVE_PLAYERS_NORM_V24] = _clamp01(active_players / 6.0)
    vec[IDX_TO_CALL_NORM_V24] = _clamp01((to_call / big_blind) / 16.0)
    vec[IDX_POT_SIZE_NORM_V24] = _clamp01((total_pot / big_blind) / 64.0)
    vec[IDX_FACING_BET_FLAG_V24] = 1.0 if to_call > 1e-6 else 0.0

    if opponent_profiles_by_seat is not None:
        vec[OPPONENT_PROFILE_SLICE_V24] = _coerce_opponent_profiles_by_seat(
            opponent_profiles_by_seat,
            hero_seat,
            player_count,
            active_flags=active_flags,
        )
    else:
        vec[OPPONENT_PROFILE_SLICE_V24] = _coerce_opponent_profile(opponent_profile)

    if VALIDATE_INFO_STATE_V21:
        validate_info_state(vec, legal_mask)
    if return_legal_mask:
        return vec, legal_mask
    return vec


def encode_public_belief_state(
    state,
    hero_seat: int,
    hand_ctx,
    return_legal_mask: bool = False,
    opponent_profile: Optional[Sequence[float] | Dict[str, float]] = None,
    opponent_profiles_by_seat: Optional[Dict[int, Sequence[float] | Dict[str, float]]] = None,
    info_state: Optional[np.ndarray] = None,
    legal_mask: Optional[np.ndarray] = None,
):
    if info_state is None or (return_legal_mask and legal_mask is None):
        info_state, inferred_legal = encode_info_state(
            state,
            hero_seat,
            hand_ctx,
            return_legal_mask=True,
            opponent_profile=opponent_profile,
            opponent_profiles_by_seat=opponent_profiles_by_seat,
        )
        if legal_mask is None:
            legal_mask = inferred_legal
    vec = np.asarray(info_state, dtype=np.float32).reshape(-1)
    if return_legal_mask:
        return vec, np.asarray(legal_mask, dtype=np.float32)
    return vec


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
        raise ValueError("State vector contains values outside [0, 1]")
    if (legal_mask < -1e-6).any() or (legal_mask > 1.000001).any():
        raise ValueError("Legal mask contains values outside [0, 1]")


def debug_feature_map(vec: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape != (STATE_DIM_V21,):
        raise ValueError(f"Expected vector of shape {(STATE_DIM_V21,)}, got {arr.shape}")
    return {name: float(arr[idx]) for idx, name in enumerate(FEATURE_NAMES_V21)}
