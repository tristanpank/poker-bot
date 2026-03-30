from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from pokerkit import Card

from position_abstraction import canonical_late_position_index

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_SMALL = 3
ACTION_RAISE_MEDIUM = 4
ACTION_RAISE_LARGE = 5
ACTION_ALL_IN = 6
ACTION_RAISE_33_POT = ACTION_RAISE_SMALL
ACTION_RAISE_66_POT = ACTION_RAISE_MEDIUM
ACTION_RAISE_POT = ACTION_RAISE_LARGE

NON_ALL_IN_RAISE_ACTIONS = (ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE)
ALL_RAISE_ACTIONS = NON_ALL_IN_RAISE_ACTIONS
POSITION_NAMES_V25 = ("SB", "BB", "UTG", "MP", "CO", "BTN")

BUILTIN_PREFLOP_BLUEPRINT_NAME = "builtin_6max_100bb_v1"
SUPPORTED_PREFLOP_BLUEPRINTS = frozenset({BUILTIN_PREFLOP_BLUEPRINT_NAME})
RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"
RANK_VALUE = {rank: 14 - idx for idx, rank in enumerate(RANK_ORDER_HIGH_TO_LOW)}
POSTFLOP_POSITION_RANK = {
    0: 0,  # SB
    1: 1,  # BB
    2: 2,  # UTG
    3: 3,  # MP
    4: 4,  # CO
    5: 5,  # BTN
}

ACTION_AGGRO_SMALL = ACTION_RAISE_SMALL
ACTION_AGGRO_LARGE = ACTION_RAISE_LARGE
POSITION_NAMES_V24 = POSITION_NAMES_V25


def _generate_canonical_hand_keys() -> tuple[str, ...]:
    keys = []
    for i, high_rank in enumerate(RANK_ORDER_HIGH_TO_LOW):
        for low_rank in RANK_ORDER_HIGH_TO_LOW[i:]:
            if high_rank == low_rank:
                keys.append(f"{high_rank}{low_rank}")
            else:
                keys.append(f"{high_rank}{low_rank}s")
                keys.append(f"{high_rank}{low_rank}o")
    return tuple(keys)


ALL_CANONICAL_HAND_KEYS = _generate_canonical_hand_keys()


def _combo_count_for_key(hand_key: str) -> int:
    if len(hand_key) == 2:
        return 6
    return 4 if hand_key.endswith("s") else 12


def _hand_strength_score(hand_key: str) -> float:
    if len(hand_key) == 2:
        pair_rank = RANK_VALUE.get(hand_key[0], 2)
        return float(85.0 + (pair_rank * 6.0) + (3.0 if pair_rank >= 11 else 0.0) + (3.0 if pair_rank <= 6 else 0.0))

    high_rank = RANK_VALUE.get(hand_key[0], 2)
    low_rank = RANK_VALUE.get(hand_key[1], 2)
    suited = hand_key.endswith("s")
    gap = high_rank - low_rank
    connector_bonus = {
        1: 9.0,
        2: 6.0,
        3: 3.0,
        4: 0.0,
    }.get(gap, -3.0)
    broadway = high_rank >= 10 and low_rank >= 10
    ace_wheel = hand_key[0] == "A" and hand_key[1] in {"5", "4", "3", "2"}

    score = (high_rank * 6.2) + (low_rank * 2.9) + connector_bonus
    if suited:
        score += 4.5
    if broadway:
        score += 6.0
    if ace_wheel and suited:
        score += 4.5
    if high_rank == 14:
        score += 3.5
    if high_rank <= 9 and low_rank <= 6 and not suited:
        score -= 3.0
    return float(score)


HAND_STRENGTH_ORDER = tuple(
    hand_key
    for _, hand_key in sorted(
        ((_hand_strength_score(hand_key), hand_key) for hand_key in ALL_CANONICAL_HAND_KEYS),
        reverse=True,
    )
)
HAND_STRENGTH_RANK = {hand_key: idx for idx, hand_key in enumerate(HAND_STRENGTH_ORDER)}


def _combo_percentile_lookup() -> dict[str, float]:
    total_combos = float(sum(_combo_count_for_key(hand_key) for hand_key in HAND_STRENGTH_ORDER))
    cumulative = 0.0
    result: dict[str, float] = {}
    for hand_key in HAND_STRENGTH_ORDER:
        cumulative += float(_combo_count_for_key(hand_key))
        result[hand_key] = float(cumulative / total_combos)
    return result


HAND_COMBO_PERCENTILE = _combo_percentile_lookup()


def canonical_preflop_hand_key(hole_cards: Sequence[Card]) -> str:
    cards = []
    for card in hole_cards:
        if card is None:
            continue
        if hasattr(card, "rank") and hasattr(card, "suit"):
            cards.append(card)
            continue
        try:
            for nested in card:
                if nested is not None and hasattr(nested, "rank") and hasattr(nested, "suit"):
                    cards.append(nested)
        except TypeError:
            continue
    if len(cards) != 2:
        return "72o"

    def _rank_index(card: Card) -> int:
        rank = getattr(card.rank, "value", card.rank)
        try:
            return RANK_ORDER_HIGH_TO_LOW.index(str(rank))
        except ValueError:
            return len(RANK_ORDER_HIGH_TO_LOW) - 1

    ordered = sorted(cards, key=_rank_index)
    c1, c2 = ordered
    rank1 = str(getattr(c1.rank, "value", c1.rank))
    rank2 = str(getattr(c2.rank, "value", c2.rank))
    if rank1 == rank2:
        return f"{rank1}{rank2}"
    suited = str(getattr(c1.suit, "value", c1.suit)) == str(getattr(c2.suit, "value", c2.suit))
    return f"{rank1}{rank2}{'s' if suited else 'o'}"


def combo_percentile_for_hand_key(hand_key: str) -> float:
    return float(HAND_COMBO_PERCENTILE.get(str(hand_key or "72o"), 1.0))


def preflop_stack_bucket(effective_stack_bb: float) -> str:
    eff_stack = float(max(0.0, effective_stack_bb))
    if eff_stack <= 20.0:
        return "short"
    if eff_stack <= 50.0:
        return "medium"
    return "deep"


def _is_in_position(actor_seat: int, aggressor_seat: Optional[int]) -> bool:
    if aggressor_seat is None:
        return False
    return int(POSTFLOP_POSITION_RANK.get(int(actor_seat), 0)) > int(POSTFLOP_POSITION_RANK.get(int(aggressor_seat), 0))


def _normalize_masked_policy(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    probs = np.asarray(policy, dtype=np.float32).reshape(-1)
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if probs.shape != mask.shape:
        size = min(probs.shape[0], mask.shape[0])
        trimmed = np.zeros_like(mask, dtype=np.float32)
        trimmed[:size] = probs[:size]
        probs = trimmed
    probs = probs * np.where(mask > 0.5, 1.0, 0.0)
    total = float(probs.sum())
    if total <= 1e-8:
        fallback = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        denom = float(fallback.sum())
        if denom <= 1e-8:
            if probs.shape[0] <= 0:
                return np.zeros(0, dtype=np.float32)
            return np.full(probs.shape[0], 1.0 / float(probs.shape[0]), dtype=np.float32)
        return (fallback / denom).astype(np.float32)
    return (probs / total).astype(np.float32)


def _fallback_action(legal_mask: np.ndarray, to_call_bb: float) -> int:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if float(to_call_bb) <= 1e-6 and len(mask) > ACTION_CHECK and float(mask[ACTION_CHECK]) > 0.5:
        return ACTION_CHECK
    if len(mask) > ACTION_CALL and float(mask[ACTION_CALL]) > 0.5:
        return ACTION_CALL
    if len(mask) > ACTION_CHECK and float(mask[ACTION_CHECK]) > 0.5:
        return ACTION_CHECK
    if len(mask) > ACTION_FOLD and float(mask[ACTION_FOLD]) > 0.5:
        return ACTION_FOLD
    for idx, is_legal in enumerate(mask):
        if float(is_legal) > 0.5:
            return int(idx)
    return ACTION_CHECK


def _first_legal_raise(legal_mask: np.ndarray, preferred_actions: Sequence[int]) -> Optional[int]:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    for action_id in preferred_actions:
        if 0 <= int(action_id) < mask.shape[0] and float(mask[int(action_id)]) > 0.5:
            return int(action_id)
    for action_id in NON_ALL_IN_RAISE_ACTIONS:
        if 0 <= int(action_id) < mask.shape[0] and float(mask[int(action_id)]) > 0.5:
            return int(action_id)
    return None


def _apply_mix(policy: np.ndarray, primary_action: Optional[int], secondary_action: Optional[int], primary_weight: float) -> np.ndarray:
    if primary_action is None and secondary_action is None:
        return policy
    if primary_action is not None:
        policy[int(primary_action)] += float(max(0.0, min(1.0, primary_weight)))
    if secondary_action is not None:
        policy[int(secondary_action)] += float(max(0.0, min(1.0, 1.0 - primary_weight)))
    return policy


def _mix_weight(percentile: float, threshold: float, width: float = 0.015) -> float:
    distance = abs(float(percentile) - float(threshold))
    if distance >= float(width):
        return 1.0
    return float(max(0.55, min(0.85, 0.85 - (distance / max(width, 1e-6)) * 0.30)))


def _open_raise_percent(actor_seat: int, stack_bucket: str) -> float:
    base = {
        0: 0.38,  # SB
        2: 0.16,  # UTG
        3: 0.20,  # MP
        4: 0.29,  # CO
        5: 0.46,  # BTN
    }.get(int(actor_seat), 0.0)
    if stack_bucket == "short":
        base += 0.02
    elif stack_bucket == "medium":
        base += 0.01
    return float(max(0.0, min(0.70, base)))


def _open_complete_percent(actor_seat: int, stack_bucket: str) -> float:
    if int(actor_seat) != 0:
        return 0.0
    base = 0.62
    if stack_bucket == "short":
        base = 0.55
    elif stack_bucket == "medium":
        base = 0.58
    return float(base)


def _vs_rfi_thresholds(actor_seat: int, aggressor_seat: Optional[int], stack_bucket: str) -> tuple[float, float]:
    opener = int(aggressor_seat if aggressor_seat is not None else 2)
    opener_looseness = {
        0: 0.16,
        1: 0.00,
        2: 0.00,
        3: 0.03,
        4: 0.08,
        5: 0.14,
    }.get(opener, 0.03)
    in_position = _is_in_position(actor_seat, aggressor_seat)
    if int(actor_seat) == 1:  # BB
        call_pct = 0.18 + opener_looseness + (0.16 if opener in {0, 5} else 0.02)
        raise_pct = 0.05 + (opener_looseness * 0.55) + (0.03 if opener in {4, 5, 0} else 0.0)
    elif int(actor_seat) == 0:  # SB
        call_pct = 0.04 + opener_looseness + (0.03 if opener == 5 else 0.0)
        raise_pct = 0.06 + (opener_looseness * 0.65) + (0.02 if opener in {4, 5} else 0.0)
    else:
        call_pct = 0.08 + opener_looseness + (0.04 if in_position else -0.01)
        raise_pct = 0.04 + (opener_looseness * 0.45) + (0.01 if in_position else 0.02)

    if stack_bucket == "short":
        call_pct -= 0.05
        raise_pct += 0.03
    elif stack_bucket == "medium":
        call_pct -= 0.02
        raise_pct += 0.015

    raise_pct = float(max(0.02, min(0.20, raise_pct)))
    call_pct = float(max(raise_pct + 0.01, min(0.55, call_pct)))
    return raise_pct, call_pct


def _vs_three_bet_thresholds(actor_seat: int, aggressor_seat: Optional[int], stack_bucket: str) -> tuple[float, float]:
    in_position = _is_in_position(actor_seat, aggressor_seat)
    four_bet_pct = 0.020 + (0.004 if not in_position else 0.0)
    call_pct = 0.040 + (0.020 if in_position else 0.0)
    if stack_bucket == "short":
        four_bet_pct += 0.025
        call_pct -= 0.020
    elif stack_bucket == "medium":
        four_bet_pct += 0.012
        call_pct -= 0.010
    four_bet_pct = float(max(0.015, min(0.08, four_bet_pct)))
    call_pct = float(max(four_bet_pct + 0.005, min(0.12, call_pct)))
    return four_bet_pct, call_pct


def _vs_four_bet_thresholds(stack_bucket: str) -> tuple[float, float]:
    if stack_bucket == "short":
        return 0.035, 0.045
    if stack_bucket == "medium":
        return 0.018, 0.030
    return 0.010, 0.018


@lru_cache(maxsize=None)
def _build_chart(
    stack_bucket: str,
    spot_name: str,
    actor_seat: int,
    aggressor_seat: int,
    in_position: bool,
) -> dict[str, tuple[str, str]]:
    chart: dict[str, tuple[str, str]] = {}
    for hand_key in ALL_CANONICAL_HAND_KEYS:
        percentile = combo_percentile_for_hand_key(hand_key)
        primary = "fold"
        secondary = "fold"

        if spot_name == "rfi":
            raise_pct = _open_raise_percent(actor_seat, stack_bucket)
            if actor_seat == 0:
                complete_pct = _open_complete_percent(actor_seat, stack_bucket)
                if percentile <= raise_pct:
                    primary, secondary = ("open_raise_big", "open_raise") if percentile <= 0.08 else ("open_raise", "complete")
                elif percentile <= complete_pct:
                    primary, secondary = "complete", "fold"
            else:
                if percentile <= raise_pct:
                    primary, secondary = ("open_raise_big", "open_raise") if percentile <= 0.06 else ("open_raise", "fold")
        elif spot_name == "bb_vs_sb_complete":
            raise_pct = 0.36 if stack_bucket == "deep" else (0.40 if stack_bucket == "medium" else 0.44)
            if percentile <= raise_pct:
                primary, secondary = ("isolate_big", "isolate") if percentile <= 0.10 else ("isolate", "check")
            else:
                primary, secondary = "check", "check"
        elif spot_name == "vs_rfi":
            raise_pct, call_pct = _vs_rfi_thresholds(actor_seat, aggressor_seat, stack_bucket)
            if percentile <= raise_pct:
                primary, secondary = ("jam" if stack_bucket == "short" and percentile <= min(0.05, raise_pct * 0.75) else ("three_bet_ip" if in_position else "three_bet_oop")), "call"
            elif percentile <= call_pct:
                primary, secondary = "call", "fold"
        elif spot_name == "vs_3bet":
            raise_pct, call_pct = _vs_three_bet_thresholds(actor_seat, aggressor_seat, stack_bucket)
            if percentile <= raise_pct:
                if stack_bucket == "short" and percentile <= min(0.045, raise_pct):
                    primary, secondary = "jam", "four_bet"
                else:
                    primary, secondary = "four_bet", "call"
            elif percentile <= call_pct:
                primary, secondary = "call", "fold"
        elif spot_name == "vs_4bet_plus":
            jam_pct, call_pct = _vs_four_bet_thresholds(stack_bucket)
            if percentile <= jam_pct:
                primary, secondary = "jam", "call"
            elif percentile <= call_pct:
                primary, secondary = "call", "fold"

        chart[hand_key] = (primary, secondary)
    return chart


def _semantic_action_to_policy(
    semantic_action: str,
    legal_mask: np.ndarray,
    to_call_bb: float,
    in_position: bool,
) -> np.ndarray:
    policy = np.zeros_like(np.asarray(legal_mask, dtype=np.float32).reshape(-1), dtype=np.float32)
    preferred_raise: Optional[int] = None

    if semantic_action == "fold":
        if policy.shape[0] > ACTION_FOLD:
            policy[ACTION_FOLD] = 1.0
        return policy
    if semantic_action == "check":
        if policy.shape[0] > ACTION_CHECK:
            policy[ACTION_CHECK] = 1.0
        return policy
    if semantic_action in {"call", "complete"}:
        if policy.shape[0] > ACTION_CALL:
            policy[ACTION_CALL] = 1.0
        if semantic_action == "complete" and policy.shape[0] > ACTION_CHECK:
            policy[ACTION_CHECK] += 0.20
        return policy

    if semantic_action == "open_raise":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_MEDIUM, ACTION_RAISE_SMALL))
    elif semantic_action == "open_raise_big":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_LARGE, ACTION_RAISE_MEDIUM))
    elif semantic_action == "isolate":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_MEDIUM, ACTION_RAISE_SMALL))
    elif semantic_action == "isolate_big":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_LARGE, ACTION_RAISE_MEDIUM))
    elif semantic_action == "three_bet_ip":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE, ACTION_RAISE_SMALL))
    elif semantic_action == "three_bet_oop":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_LARGE, ACTION_RAISE_MEDIUM))
    elif semantic_action == "four_bet":
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_LARGE, ACTION_RAISE_MEDIUM))
    elif semantic_action == "jam":
        if policy.shape[0] > ACTION_ALL_IN:
            policy[ACTION_ALL_IN] = 1.0
        if float(policy.sum()) > 0.0:
            return policy
        preferred_raise = _first_legal_raise(legal_mask, (ACTION_RAISE_LARGE, ACTION_RAISE_MEDIUM))

    if preferred_raise is not None:
        policy[int(preferred_raise)] = 1.0
        return policy

    fallback = _fallback_action(np.asarray(legal_mask, dtype=np.float32), to_call_bb)
    policy[int(fallback)] = 1.0
    return policy


def preflop_blueprint_policy(
    hole_cards: Sequence[Card],
    actor_seat: int,
    legal_mask: Sequence[float],
    effective_stack_bb: float,
    to_call_bb: float,
    preflop_raise_count: int,
    preflop_call_count: int,
    aggressor_seat: Optional[int],
    player_count: int = 6,
    blueprint_name: str = BUILTIN_PREFLOP_BLUEPRINT_NAME,
) -> tuple[np.ndarray, dict[str, object]]:
    if str(blueprint_name or BUILTIN_PREFLOP_BLUEPRINT_NAME) not in SUPPORTED_PREFLOP_BLUEPRINTS:
        return np.zeros(len(np.asarray(legal_mask, dtype=np.float32).reshape(-1)), dtype=np.float32), {
            "covered": False,
            "reason": "unsupported_blueprint",
        }

    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    hand_key = canonical_preflop_hand_key(hole_cards)
    combo_pct = combo_percentile_for_hand_key(hand_key)
    stack_bucket = preflop_stack_bucket(effective_stack_bb)
    spot_name = "rfi"
    canonical_actor_seat = canonical_late_position_index(player_count, actor_seat)
    canonical_aggressor_seat = (
        canonical_late_position_index(player_count, aggressor_seat)
        if aggressor_seat is not None
        else None
    )
    in_position = _is_in_position(canonical_actor_seat, canonical_aggressor_seat)

    if int(preflop_raise_count) <= 0:
        if int(actor_seat) == 1 and float(to_call_bb) <= 1e-6 and int(preflop_call_count) > 0:
            spot_name = "bb_vs_sb_complete"
        else:
            spot_name = "rfi"
    elif int(preflop_raise_count) == 1:
        spot_name = "vs_rfi"
    elif int(preflop_raise_count) == 2:
        spot_name = "vs_3bet"
    else:
        spot_name = "vs_4bet_plus"

    chart = _build_chart(
        stack_bucket,
        spot_name,
        int(canonical_actor_seat),
        int(canonical_aggressor_seat) if canonical_aggressor_seat is not None else -1,
        bool(in_position),
    )
    primary_semantic, secondary_semantic = chart.get(hand_key, ("fold", "fold"))

    primary_policy = _semantic_action_to_policy(primary_semantic, mask, to_call_bb, in_position)
    secondary_policy = _semantic_action_to_policy(secondary_semantic, mask, to_call_bb, in_position)
    primary_weight = 1.0

    if spot_name == "rfi":
        threshold = _open_raise_percent(canonical_actor_seat, stack_bucket)
        if int(actor_seat) == 0 and primary_semantic == "complete":
            threshold = _open_complete_percent(canonical_actor_seat, stack_bucket)
        primary_weight = _mix_weight(combo_pct, threshold)
    elif spot_name == "bb_vs_sb_complete":
        threshold = 0.36 if stack_bucket == "deep" else (0.40 if stack_bucket == "medium" else 0.44)
        primary_weight = _mix_weight(combo_pct, threshold)
    elif spot_name == "vs_rfi":
        threshold, call_threshold = _vs_rfi_thresholds(
            canonical_actor_seat,
            canonical_aggressor_seat,
            stack_bucket,
        )
        primary_weight = _mix_weight(combo_pct, threshold if primary_semantic.startswith("three_bet") or primary_semantic == "jam" else call_threshold)
    elif spot_name == "vs_3bet":
        threshold, call_threshold = _vs_three_bet_thresholds(
            canonical_actor_seat,
            canonical_aggressor_seat,
            stack_bucket,
        )
        primary_weight = _mix_weight(combo_pct, threshold if primary_semantic in {"four_bet", "jam"} else call_threshold)
    elif spot_name == "vs_4bet_plus":
        threshold, call_threshold = _vs_four_bet_thresholds(stack_bucket)
        primary_weight = _mix_weight(combo_pct, threshold if primary_semantic == "jam" else call_threshold)

    policy = np.zeros_like(mask, dtype=np.float32)
    primary_action = int(np.argmax(primary_policy)) if float(primary_policy.sum()) > 0.0 else None
    secondary_action = int(np.argmax(secondary_policy)) if float(secondary_policy.sum()) > 0.0 else None
    policy = _apply_mix(policy, primary_action, secondary_action, primary_weight)
    policy = _normalize_masked_policy(policy, mask)
    return policy, {
        "covered": True,
        "hand_key": hand_key,
        "combo_percentile": combo_pct,
        "stack_bucket": stack_bucket,
        "spot": spot_name,
        "in_position": bool(in_position),
        "primary": primary_semantic,
        "secondary": secondary_semantic,
    }
