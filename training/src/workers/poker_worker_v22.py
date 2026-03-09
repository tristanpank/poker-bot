import gc
import copy
import os
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from pokerkit import Automation, Card, NoLimitTexasHoldem, StandardHighHand

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
MODELS_DIR = os.path.join(SRC_ROOT, "models")
for path in (FEATURES_DIR, MODELS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from poker_model_v22 import PokerDeepCFRNet, masked_policy, regret_matching
from poker_state_v22 import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_COUNT_V21,
    ACTION_FOLD,
    ACTION_RAISE_HALF_POT,
    ACTION_RAISE_POT_OR_ALL_IN,
    ACTION_NAMES_V21,
    POSITION_NAMES_V21,
    build_legal_action_mask,
    debug_feature_map,
    encode_info_state,
    estimate_preflop_strength,
    flatten_cards_list,
)

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

_MP_MODEL_CACHE: Dict[str, object] = {
    "signature": None,
    "actor": None,
    "opponent": None,
}
@dataclass
class HandContext:
    starting_stacks: List[int]
    big_blind: int
    small_blind: int
    in_hand: List[bool]
    contributions: List[float]
    current_street: int = 0
    street_raise_count: int = 0
    preflop_raise_count: int = 0
    preflop_opened: bool = False
    last_aggressor: Optional[int] = None
    last_aggressive_size_bb: float = 0.0


@dataclass
class TraversalResult:
    advantage_samples: List[tuple] = field(default_factory=list)
    strategy_samples: List[tuple] = field(default_factory=list)
    utility_bb: float = 0.0
    unclipped_utility_bb: float = 0.0
    traverser_seat: int = 0
    traverser_decisions: int = 0
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_COUNT_V21, dtype=np.int64))
    invalid_state_count: int = 0
    invalid_action_count: int = 0
    vpip: bool = False
    pfr: bool = False
    three_bet: bool = False
    perf_breakdown: Dict[str, float] = field(default_factory=dict)
    debug_state: Optional[Dict[str, float]] = None


@dataclass
class HandResult:
    hero_profit_bb: float
    hero_seat: int
    action_counts: np.ndarray
    illegal_action_count: int
    win: bool
    vpip: bool
    pfr: bool
    three_bet: bool
    rfi_opportunity: bool
    rfi_attempt: bool
    hero_hand_key: Optional[str] = None


def _street_from_board(state) -> int:
    board_len = len(flatten_cards_list(state.board_cards))
    if board_len <= 0:
        return 0
    if board_len == 3:
        return 1
    if board_len == 4:
        return 2
    return 3


def _sample_stacks(rng: random.Random, config) -> List[int]:
    stacks: List[int] = []
    for _ in range(config.num_players):
        bb_stack = max(85.0, min(115.0, rng.gauss(100.0, 8.0)))
        stacks.append(int(bb_stack * config.big_blind))
    return stacks


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
        raw_blinds_or_straddles=(config.small_blind, config.big_blind),
        min_bet=config.big_blind,
        raw_starting_stacks=stacks,
        player_count=config.num_players,
    )
    while state.can_deal_hole():
        state.deal_hole()
    contributions = [float(start - stack) for start, stack in zip(stacks, state.stacks)]
    hand_ctx = HandContext(
        starting_stacks=list(stacks),
        big_blind=config.big_blind,
        small_blind=config.small_blind,
        in_hand=[True] * config.num_players,
        contributions=contributions,
    )
    return state, hand_ctx


def _advance_chance_nodes(state, hand_ctx: HandContext) -> None:
    while state.status and state.can_deal_board():
        state.deal_board()
        hand_ctx.current_street = _street_from_board(state)
        hand_ctx.street_raise_count = 0


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


def _safe_utility_bb(state, traverser: int, hand_ctx: HandContext) -> float:
    final_stack = float(state.stacks[traverser])
    start_stack = float(hand_ctx.starting_stacks[traverser])
    return (final_stack - start_stack) / float(hand_ctx.big_blind)


def _inference(model: Optional[PokerDeepCFRNet], state_vec: np.ndarray, head: str) -> np.ndarray:
    if model is None:
        return np.zeros(ACTION_COUNT_V21, dtype=np.float32)
    with torch.inference_mode():
        if state_vec.dtype != np.float32:
            state_vec = state_vec.astype(np.float32, copy=False)
        tensor = torch.from_numpy(state_vec).unsqueeze(0)
        if head == "regret":
            logits = model.forward_regret(tensor)
        else:
            logits = model.forward_strategy(tensor)
        return logits.squeeze(0).cpu().numpy().astype(np.float32)


def _sample_action(probs: np.ndarray, rng: random.Random) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        return int(np.argmax(probs))
    draw = rng.random() * total
    cumulative = 0.0
    for idx, value in enumerate(probs):
        cumulative += float(value)
        if draw <= cumulative:
            return idx
    return int(np.argmax(probs))


def _best_hand_strength_scalar(hole_cards, board_cards) -> float:
    hole_cards = flatten_cards_list(hole_cards)
    board_cards = flatten_cards_list(board_cards)
    if len(hole_cards) != 2:
        return 0.25
    if len(board_cards) < 3:
        return estimate_preflop_strength(hole_cards)

    cards = hole_cards + board_cards
    if len(cards) < 5:
        return estimate_preflop_strength(hole_cards)

    try:
        best_hand = max(
            StandardHighHand("".join(f"{card.rank.value}{card.suit.value}" for card in combo))
            for combo in combinations(cards, 5)
        )
        label = best_hand.entry.label.name
    except Exception:
        return estimate_preflop_strength(hole_cards)

    category_scale = {
        "HIGH_CARD": 0.20,
        "ONE_PAIR": 0.35,
        "TWO_PAIR": 0.50,
        "THREE_OF_A_KIND": 0.62,
        "STRAIGHT": 0.72,
        "FLUSH": 0.80,
        "FULL_HOUSE": 0.88,
        "FOUR_OF_A_KIND": 0.95,
        "STRAIGHT_FLUSH": 0.99,
    }
    return float(category_scale.get(label, 0.30))


def _canonical_preflop_key(hole_cards: List[Card]) -> str:
    hole_cards = flatten_cards_list(hole_cards)
    if len(hole_cards) != 2:
        return "72o"
    rank_order = "23456789TJQKA"

    def _rank_idx(card: Card) -> int:
        rank = getattr(card.rank, "value", card.rank)
        return rank_order.index(rank)

    cards = sorted(hole_cards, key=_rank_idx, reverse=True)
    c1, c2 = cards
    r1 = getattr(c1.rank, "value", c1.rank)
    r2 = getattr(c2.rank, "value", c2.rank)
    if r1 == r2:
        return f"{r1}{r2}"
    suited = getattr(c1.suit, "value", c1.suit) == getattr(c2.suit, "value", c2.suit)
    return f"{r1}{r2}{'s' if suited else 'o'}"


def _raise_target_bounds(state) -> tuple:
    min_raise = getattr(state, "min_completion_betting_or_raising_to_amount", None)
    max_raise = getattr(state, "max_completion_betting_or_raising_to_amount", None)
    if min_raise is None:
        min_raise = getattr(state, "min_completion_betting_or_raising_to", 0)
    if max_raise is None:
        max_raise = getattr(state, "max_completion_betting_or_raising_to", 0)
    return int(min_raise or 0), int(max_raise or 0)


def _heuristic_action(state, actor: int, hand_ctx: HandContext, rng: random.Random) -> int:
    state_vec, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
    legal_actions = [idx for idx, value in enumerate(legal_mask) if value > 0.5]
    if not legal_actions:
        return ACTION_CHECK

    hole_cards = flatten_cards_list(state.hole_cards[actor])
    board_cards = flatten_cards_list(state.board_cards)
    strength = _best_hand_strength_scalar(hole_cards, board_cards)
    to_call = (max(state.bets) - state.bets[actor]) / float(hand_ctx.big_blind)
    pot_odds = float(state_vec[76])
    flush_draw = bool(state_vec[59] > 0.5)
    straight_draw = bool(state_vec[60] > 0.5)

    if strength >= 0.85:
        if legal_mask[ACTION_RAISE_POT_OR_ALL_IN] > 0.5:
            return ACTION_RAISE_POT_OR_ALL_IN
        if legal_mask[ACTION_RAISE_HALF_POT] > 0.5:
            return ACTION_RAISE_HALF_POT
        if legal_mask[ACTION_CALL] > 0.5:
            return ACTION_CALL
        return ACTION_CHECK

    if strength >= max(0.45, pot_odds + 0.08):
        if to_call <= 0.0 and strength >= 0.62 and legal_mask[ACTION_RAISE_HALF_POT] > 0.5 and rng.random() < 0.55:
            return ACTION_RAISE_HALF_POT
        if legal_mask[ACTION_CALL] > 0.5:
            return ACTION_CALL
        if legal_mask[ACTION_CHECK] > 0.5:
            return ACTION_CHECK

    if (flush_draw or straight_draw) and pot_odds <= 0.30:
        if legal_mask[ACTION_CALL] > 0.5:
            return ACTION_CALL
        if legal_mask[ACTION_CHECK] > 0.5:
            return ACTION_CHECK

    if to_call <= 0.0 and legal_mask[ACTION_RAISE_HALF_POT] > 0.5 and rng.random() < 0.08:
        return ACTION_RAISE_HALF_POT

    if legal_mask[ACTION_CHECK] > 0.5:
        return ACTION_CHECK
    if legal_mask[ACTION_FOLD] > 0.5 and strength < pot_odds:
        return ACTION_FOLD
    if legal_mask[ACTION_CALL] > 0.5:
        return ACTION_CALL
    return legal_actions[0]


def apply_abstract_action(state, actor: int, action_id: int, hand_ctx: HandContext) -> bool:
    actor = int(actor)
    valid = True
    requested = int(action_id)
    legal_mask = build_legal_action_mask(state, actor, hand_ctx)
    if requested < 0 or requested >= ACTION_COUNT_V21 or legal_mask[requested] <= 0.5:
        valid = False

    before_stack = float(state.stacks[actor])
    before_bet = float(state.bets[actor])
    to_call = max(state.bets) - state.bets[actor]
    pot = float(sum(pot_item.amount for pot_item in getattr(state, "pots", [])) + sum(state.bets))

    applied_raise = False
    try:
        if requested == ACTION_FOLD and state.can_fold():
            state.fold()
            hand_ctx.in_hand[actor] = False
        elif requested in (ACTION_CHECK, ACTION_CALL) and state.can_check_or_call():
            state.check_or_call()
        elif requested in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN) and state.can_complete_bet_or_raise_to():
            min_raise, max_raise = _raise_target_bounds(state)
            if requested == ACTION_RAISE_HALF_POT:
                target = max(min_raise, int(state.bets[actor] + to_call + (0.5 * pot)))
            else:
                pot_target = getattr(state, "pot_completion_betting_or_raising_to_amount", None)
                if pot_target is None:
                    target = max(min_raise, int(state.bets[actor] + to_call + pot))
                else:
                    target = max(min_raise, int(pot_target))
            target = min(target, max_raise)
            state.complete_bet_or_raise_to(int(target))
            applied_raise = True
        elif state.can_check_or_call():
            valid = False
            state.check_or_call()
        elif state.can_fold():
            valid = False
            state.fold()
            hand_ctx.in_hand[actor] = False
    except Exception:
        valid = False
        if state.can_check_or_call():
            state.check_or_call()
        elif state.can_fold():
            state.fold()
            hand_ctx.in_hand[actor] = False

    invested = max(0.0, before_stack - float(state.stacks[actor]))
    hand_ctx.contributions[actor] += invested

    if applied_raise:
        hand_ctx.last_aggressor = actor
        hand_ctx.street_raise_count += 1
        if hand_ctx.current_street == 0:
            hand_ctx.preflop_raise_count += 1
            hand_ctx.preflop_opened = True
        raise_delta_bb = max(0.0, (float(state.bets[actor]) - before_bet) / float(hand_ctx.big_blind))
        hand_ctx.last_aggressive_size_bb = raise_delta_bb

    return valid


def _simulate_from_state(
    state,
    traverser: int,
    hand_ctx: HandContext,
    actor_snapshot: Optional[PokerDeepCFRNet],
    opponent_snapshot: Optional[PokerDeepCFRNet],
    config,
    rng: random.Random,
    traverser_depth: int,
    record: bool,
    result: Optional[TraversalResult],
    perf: Dict[str, float],
) -> float:
    while state.status:
        chance_start = time.perf_counter()
        _advance_chance_nodes(state, hand_ctx)
        perf["chance_time"] += time.perf_counter() - chance_start
        if not state.status:
            break

        actor = state.actor_index
        if actor is None:
            break

        if actor == traverser:
            try:
                state_start = time.perf_counter()
                state_vec, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
                perf["traverser_state_time"] += time.perf_counter() - state_start
            except Exception:
                if record and result is not None:
                    result.invalid_state_count += 1
                return _safe_utility_bb(state, traverser, hand_ctx)

            if record and result is not None:
                result.traverser_decisions += 1
                if result.debug_state is None:
                    result.debug_state = debug_feature_map(state_vec)

            infer_start = time.perf_counter()
            regret_logits = _inference(actor_snapshot, state_vec, head="regret")
            perf["regret_infer_time"] += time.perf_counter() - infer_start
            sigma = regret_matching(regret_logits, legal_mask)

            if traverser_depth < config.full_branch_depth:
                action_values = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
                legal_actions = [action_id for action_id, is_legal in enumerate(legal_mask) if is_legal > 0.5]
                branch_actions = list(legal_actions)
                max_branch_actions = int(getattr(config, "max_branch_actions", 0))
                if max_branch_actions > 0 and len(branch_actions) > max_branch_actions:
                    ordered = np.argsort(sigma[branch_actions])[::-1]
                    branch_actions = [branch_actions[idx] for idx in ordered[:max_branch_actions]]
                branch_mask = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
                for action_id in branch_actions:
                    branch_mask[action_id] = 1.0
                if branch_mask.sum() > 0:
                    branch_sigma = sigma * branch_mask
                    branch_sigma /= max(float(branch_sigma.sum()), 1e-8)
                else:
                    branch_sigma = sigma
                    branch_mask = legal_mask.copy()
                for action_id in branch_actions:
                    branch_rng = random.Random()
                    branch_rng.setstate(rng.getstate())
                    action_values[action_id] = clone_and_rollout_branch(
                        state,
                        traverser,
                        action_id,
                        hand_ctx,
                        actor_snapshot,
                        opponent_snapshot,
                        config,
                        branch_rng,
                        traverser_depth + 1,
                        perf,
                    )
                if record and result is not None:
                    node_value = float(np.dot(branch_sigma, action_values))
                    regrets = (action_values - node_value) * branch_mask
                    result.advantage_samples.append((state_vec, legal_mask.copy(), regrets.astype(np.float32), 1.0))
                    result.strategy_samples.append((state_vec, legal_mask.copy(), sigma.astype(np.float32), 1.0))
            elif record and result is not None:
                result.strategy_samples.append((state_vec, legal_mask.copy(), sigma.astype(np.float32), 1.0))

            chosen_action = _sample_action(sigma, rng)
            if record and result is not None:
                result.action_counts[chosen_action] += 1
                if hand_ctx.current_street == 0:
                    prior_preflop_raises = int(hand_ctx.preflop_raise_count)
                    if chosen_action in (ACTION_CALL, ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        result.vpip = True
                    if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        result.pfr = True
                        if prior_preflop_raises >= 1:
                            result.three_bet = True

            apply_start = time.perf_counter()
            is_valid = apply_abstract_action(state, actor, chosen_action, hand_ctx)
            perf["apply_time"] += time.perf_counter() - apply_start
            if record and result is not None and not is_valid:
                result.invalid_action_count += 1
            traverser_depth += 1
            continue

        try:
            state_start = time.perf_counter()
            opponent_state, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
            perf["opponent_state_time"] += time.perf_counter() - state_start
        except Exception:
            if record and result is not None:
                result.invalid_state_count += 1
            return _safe_utility_bb(state, traverser, hand_ctx)

        policy_model = opponent_snapshot if opponent_snapshot is not None else actor_snapshot
        infer_start = time.perf_counter()
        policy_logits = _inference(policy_model, opponent_state, head="strategy")
        perf["strategy_infer_time"] += time.perf_counter() - infer_start
        probs = masked_policy(policy_logits, legal_mask)
        chosen_action = _sample_action(probs, rng)
        apply_start = time.perf_counter()
        is_valid = apply_abstract_action(state, actor, chosen_action, hand_ctx)
        perf["apply_time"] += time.perf_counter() - apply_start
        if record and result is not None and not is_valid:
            result.invalid_action_count += 1

    return _safe_utility_bb(state, traverser, hand_ctx)


def clone_and_rollout_branch(
    state,
    traverser: int,
    action_id: int,
    hand_ctx: HandContext,
    actor_snapshot: Optional[PokerDeepCFRNet],
    opponent_snapshot: Optional[PokerDeepCFRNet],
    config,
    rng: Optional[random.Random] = None,
    traverser_depth: int = 1,
    perf: Optional[Dict[str, float]] = None,
) -> float:
    perf_dict = perf if perf is not None else _new_perf_breakdown()
    clone_start = time.perf_counter()
    branch_state = copy.deepcopy(state)
    branch_ctx = copy.deepcopy(hand_ctx)
    perf_dict["branch_clone_time"] += time.perf_counter() - clone_start
    branch_rng = rng if rng is not None else random.Random()
    apply_start = time.perf_counter()
    apply_abstract_action(branch_state, traverser, action_id, branch_ctx)
    perf_dict["apply_time"] += time.perf_counter() - apply_start
    return _simulate_from_state(
        branch_state,
        traverser,
        branch_ctx,
        actor_snapshot,
        opponent_snapshot,
        config,
        branch_rng,
        traverser_depth,
        False,
        None,
        perf_dict,
    )


def run_traversal(
    hand_seed: int,
    traverser_seat: int,
    actor_snapshot: Optional[PokerDeepCFRNet],
    opponent_snapshot: Optional[PokerDeepCFRNet],
    config,
) -> TraversalResult:
    rng = random.Random(int(hand_seed))
    perf = _new_perf_breakdown()
    init_start = time.perf_counter()
    state, hand_ctx = _create_state_and_context(rng, config)
    perf["state_init_time"] += time.perf_counter() - init_start
    result = TraversalResult(traverser_seat=int(traverser_seat))
    clipped_utility = _simulate_from_state(
        state,
        int(traverser_seat),
        hand_ctx,
        actor_snapshot,
        opponent_snapshot,
        config,
        rng,
        0,
        True,
        result,
        perf,
    )
    result.utility_bb = clipped_utility
    result.unclipped_utility_bb = _safe_utility_bb(state, int(traverser_seat), hand_ctx)
    result.perf_breakdown = perf
    return result


def _policy_action_for_snapshot(
    snapshot: Optional[PokerDeepCFRNet],
    state,
    actor: int,
    hand_ctx: HandContext,
    rng: random.Random,
) -> int:
    state_vec, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
    logits = _inference(snapshot, state_vec, head="strategy")
    probs = masked_policy(logits, legal_mask)
    return _sample_action(probs, rng)


def run_policy_hand(hand_seed: int, actor_snapshot: Optional[PokerDeepCFRNet], config) -> HandResult:
    rng = random.Random(int(hand_seed))
    state, hand_ctx = _create_state_and_context(rng, config)
    hero_seat = int(getattr(config, "eval_hero_seat", 0)) % config.num_players
    opponent_mode = getattr(config, "evaluation_mode", "heuristics")
    checkpoint_pool = list(getattr(config, "checkpoint_pool", []))

    seat_models: Dict[int, Optional[PokerDeepCFRNet]] = {}
    if opponent_mode == "checkpoints" and checkpoint_pool:
        opponent_seats = [seat for seat in range(config.num_players) if seat != hero_seat]
        for seat in opponent_seats:
            seat_models[seat] = rng.choice(checkpoint_pool)

    action_counts = np.zeros(ACTION_COUNT_V21, dtype=np.int64)
    illegal_action_count = 0
    vpip = False
    pfr = False
    three_bet = False
    rfi_opportunity = False
    rfi_attempt = False
    hero_preflop_seen = False
    hero_hand_key = _canonical_preflop_key(flatten_cards_list(state.hole_cards[hero_seat]))

    while state.status:
        _advance_chance_nodes(state, hand_ctx)
        if not state.status:
            break

        actor = state.actor_index
        if actor is None:
            break

        if actor == hero_seat:
            preflop_action = hand_ctx.current_street == 0
            prior_preflop_raises = int(hand_ctx.preflop_raise_count)
            chosen_action = _policy_action_for_snapshot(actor_snapshot, state, actor, hand_ctx, rng)
            action_counts[chosen_action] += 1
            if preflop_action and not hero_preflop_seen:
                hero_preflop_seen = True
                if prior_preflop_raises == 0:
                    rfi_opportunity = True
                    if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        rfi_attempt = True
            if preflop_action:
                if chosen_action in (ACTION_CALL, ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                    vpip = True
                if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                    pfr = True
                    if prior_preflop_raises >= 1:
                        three_bet = True
        else:
            if opponent_mode == "checkpoints" and seat_models:
                chosen_action = _policy_action_for_snapshot(seat_models.get(actor), state, actor, hand_ctx, rng)
            else:
                chosen_action = _heuristic_action(state, actor, hand_ctx, rng)

        is_valid = apply_abstract_action(state, actor, chosen_action, hand_ctx)
        if not is_valid:
            illegal_action_count += 1

    hero_profit_bb = (float(state.stacks[hero_seat]) - float(hand_ctx.starting_stacks[hero_seat])) / float(hand_ctx.big_blind)
    return HandResult(
        hero_profit_bb=hero_profit_bb,
        hero_seat=hero_seat,
        action_counts=action_counts,
        illegal_action_count=illegal_action_count,
        win=hero_profit_bb > 0.0,
        vpip=vpip,
        pfr=pfr,
        three_bet=three_bet,
        rfi_opportunity=rfi_opportunity,
        rfi_attempt=rfi_attempt,
        hero_hand_key=hero_hand_key,
    )


def _load_model_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
) -> PokerDeepCFRNet:
    inferred_state_dim = int(state_dim)
    inferred_hidden_dim = int(hidden_dim)
    inferred_action_dim = int(action_dim)

    input_weight = None
    regret_out_weight = None
    for key, tensor in state_dict.items():
        name = str(key)
        if input_weight is None and name.endswith("input_layer.weight"):
            input_weight = tensor
        if regret_out_weight is None and name.endswith("regret_head.2.weight"):
            regret_out_weight = tensor
    if input_weight is not None and getattr(input_weight, "ndim", 0) == 2:
        inferred_hidden_dim = int(input_weight.shape[0])
        inferred_state_dim = int(input_weight.shape[1])
    if regret_out_weight is not None and getattr(regret_out_weight, "ndim", 0) == 2:
        inferred_action_dim = int(regret_out_weight.shape[0])

    model = PokerDeepCFRNet(
        state_dim=inferred_state_dim,
        hidden_dim=inferred_hidden_dim,
        action_dim=inferred_action_dim,
        init_weights=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def run_traversal_batch_mp(
    hand_seeds: List[int],
    traverser_seats: List[int],
    actor_state_dict: Dict[str, torch.Tensor],
    opponent_state_dict: Optional[Dict[str, torch.Tensor]],
    config_dict: Dict[str, object],
    snapshot_signature: str,
) -> List[TraversalResult]:
    cache_signature = _MP_MODEL_CACHE.get("signature")
    if cache_signature != snapshot_signature:
        _MP_MODEL_CACHE["actor"] = None
        _MP_MODEL_CACHE["opponent"] = None
        gc.collect()
        state_dim = int(config_dict.get("state_dim", 98))
        hidden_dim = int(config_dict.get("hidden_dim", 256))
        action_dim = int(config_dict.get("action_count", ACTION_COUNT_V21))
        actor_model = _load_model_from_state_dict(actor_state_dict, state_dim, hidden_dim, action_dim)
        if opponent_state_dict is None:
            opponent_model = actor_model
        else:
            opponent_model = _load_model_from_state_dict(opponent_state_dict, state_dim, hidden_dim, action_dim)
        _MP_MODEL_CACHE["signature"] = snapshot_signature
        _MP_MODEL_CACHE["actor"] = actor_model
        _MP_MODEL_CACHE["opponent"] = opponent_model

    actor_model = _MP_MODEL_CACHE["actor"]
    opponent_model = _MP_MODEL_CACHE["opponent"]
    config_ns = SimpleNamespace(**config_dict)
    results: List[TraversalResult] = []
    for seed, traverser_seat in zip(hand_seeds, traverser_seats):
        results.append(run_traversal(seed, traverser_seat, actor_model, opponent_model, config_ns))
    return results
