import gc
import copy
import os
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from multiprocessing.connection import Client
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

from poker_model_v23 import PokerDeepCFRNet, load_compatible_state_dict, masked_policy, regret_matching
from poker_state_v23 import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_COUNT_V21,
    ACTION_FOLD,
    ACTION_RAISE_HALF_POT,
    ACTION_RAISE_POT_OR_ALL_IN,
    ACTION_NAMES_V21,
    OPPONENT_PROFILE_DEFAULT_V23,
    POSITION_NAMES_V21,
    STATE_DIM_V21,
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
    "actor_signature": None,
    "opponent_signature": None,
    "actor": None,
    "opponent": None,
}
_MP_GPU_CLIENT_CACHE: Dict[str, object] = {
    "enabled": False,
    "endpoint": None,
    "authkey": None,
    "conn": None,
}


@dataclass
class RemoteModelRef:
    gpu_service_key: str
    state_dim: int = STATE_DIM_V21
    hidden_dim: int = 0
    action_dim: int = ACTION_COUNT_V21
OPP_PROFILE_IDX_VPIP = 0
OPP_PROFILE_IDX_PFR = 1
OPP_PROFILE_IDX_THREE_BET = 2
OPP_PROFILE_IDX_FOLD_TO_OPEN = 3
OPP_PROFILE_IDX_FOLD_TO_THREE_BET = 4
OPP_PROFILE_IDX_CALL_OPEN = 5
OPP_PROFILE_IDX_SQUEEZE = 6
OPP_PROFILE_IDX_FOLD_TO_CBET_FLOP = 7
OPP_PROFILE_IDX_FOLD_TO_CBET_TURN = 8
OPP_PROFILE_IDX_AGGRESSION = 9
OPP_PROFILE_IDX_CONFIDENCE = 10
SYNTHETIC_OPPONENT_STYLES = (
    "nit",
    "overfolder",
    "overcaller",
    "over3better",
    "station",
    "maniac",
)
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


def _new_preflop_stats(num_players: int) -> Dict[str, List[int]]:
    count = max(1, int(num_players))
    stats = {key: [0] * count for key in PREFLOP_STAT_KEYS}
    stats["hands_seen_flags"] = [0] * count
    return stats


def _record_preflop_action_stats(
    preflop_stats: Dict[str, List[int]],
    actor: int,
    hand_ctx,
    to_call: float,
    prior_preflop_raises: int,
    action_id: int,
) -> None:
    if not preflop_stats:
        return
    seat = int(actor)
    if seat < 0:
        return
    opportunities = preflop_stats.get("preflop_opportunities", [])
    if seat >= len(opportunities):
        return

    seen_flags = preflop_stats.get("hands_seen_flags", [])
    hands_played = preflop_stats.get("hands_played", [])
    if seat < len(seen_flags) and seen_flags[seat] == 0:
        seen_flags[seat] = 1
        if seat < len(hands_played):
            hands_played[seat] += 1

    opportunities[seat] += 1
    preflop_stats["aggression_opportunities"][seat] += 1
    facing_wager = float(to_call) > 1e-6
    if action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
        preflop_stats["aggression_counts"][seat] += 1
    if action_id in (ACTION_CALL, ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
        preflop_stats["vpip_counts"][seat] += 1
    if action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
        preflop_stats["pfr_counts"][seat] += 1
        if prior_preflop_raises == 1 and facing_wager:
            preflop_stats["three_bet_counts"][seat] += 1
    if action_id == ACTION_FOLD:
        preflop_stats["fold_preflop_counts"][seat] += 1

    faced_open = prior_preflop_raises == 1 and facing_wager
    if faced_open:
        preflop_stats["faced_open_opportunities"][seat] += 1
        if action_id == ACTION_FOLD:
            preflop_stats["fold_vs_open_counts"][seat] += 1
        elif action_id == ACTION_CALL:
            preflop_stats["call_vs_open_counts"][seat] += 1

    faced_three_bet = prior_preflop_raises >= 2 and facing_wager
    if faced_three_bet:
        preflop_stats["faced_three_bet_opportunities"][seat] += 1
        if action_id == ACTION_FOLD:
            preflop_stats["fold_vs_three_bet_counts"][seat] += 1

    squeeze_opp = prior_preflop_raises == 1 and int(getattr(hand_ctx, "preflop_call_count", 0)) >= 1 and facing_wager
    if squeeze_opp:
        preflop_stats["squeeze_opportunities"][seat] += 1
        if action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
            preflop_stats["squeeze_counts"][seat] += 1


def _record_postflop_action_stats(
    preflop_stats: Dict[str, List[int]],
    actor: int,
    hand_ctx,
    to_call: float,
    action_id: int,
) -> None:
    if not preflop_stats:
        return
    seat = int(actor)
    if seat < 0:
        return
    aggression_opportunities = preflop_stats.get("aggression_opportunities", [])
    aggression_counts = preflop_stats.get("aggression_counts", [])
    if seat < len(aggression_opportunities):
        aggression_opportunities[seat] += 1
        if action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN) and seat < len(aggression_counts):
            aggression_counts[seat] += 1

    if int(getattr(hand_ctx, "current_street", 0)) == 1:
        preflop_last_raiser = getattr(hand_ctx, "preflop_last_raiser", None)
        if (
            int(getattr(hand_ctx, "street_raise_count", 0)) == 0
            and preflop_last_raiser is not None
            and int(preflop_last_raiser) == seat
            and float(to_call) <= 1e-6
            and action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN)
            and getattr(hand_ctx, "cbet_flop_initiator", None) is None
        ):
            hand_ctx.cbet_flop_initiator = seat
        cbetter = getattr(hand_ctx, "cbet_flop_initiator", None)
        if cbetter is not None and int(cbetter) != seat and float(to_call) > 1e-6:
            faced = preflop_stats.get("faced_cbet_flop_opportunities", [])
            folds = preflop_stats.get("fold_vs_cbet_flop_counts", [])
            if seat < len(faced):
                faced[seat] += 1
                if action_id == ACTION_FOLD and seat < len(folds):
                    folds[seat] += 1
        return

    if int(getattr(hand_ctx, "current_street", 0)) == 2:
        preflop_last_raiser = getattr(hand_ctx, "preflop_last_raiser", None)
        if (
            int(getattr(hand_ctx, "street_raise_count", 0)) == 0
            and preflop_last_raiser is not None
            and int(preflop_last_raiser) == seat
            and float(to_call) <= 1e-6
            and action_id in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN)
            and getattr(hand_ctx, "cbet_turn_initiator", None) is None
            and getattr(hand_ctx, "cbet_flop_initiator", None) == seat
        ):
            hand_ctx.cbet_turn_initiator = seat
        cbetter = getattr(hand_ctx, "cbet_turn_initiator", None)
        if cbetter is not None and int(cbetter) != seat and float(to_call) > 1e-6:
            faced = preflop_stats.get("faced_cbet_turn_opportunities", [])
            folds = preflop_stats.get("fold_vs_cbet_turn_counts", [])
            if seat < len(faced):
                faced[seat] += 1
                if action_id == ACTION_FOLD and seat < len(folds):
                    folds[seat] += 1


def _aligned_state_vector_for_model(model: Optional[PokerDeepCFRNet], state_vec: np.ndarray) -> np.ndarray:
    if model is None:
        return state_vec
    expected_dim = int(getattr(model, "state_dim", 0) or 0)
    if expected_dim <= 0:
        return state_vec
    actual_dim = int(state_vec.shape[0])
    if actual_dim == expected_dim:
        return state_vec
    if actual_dim > expected_dim:
        return state_vec[:expected_dim]
    aligned = np.zeros(expected_dim, dtype=np.float32)
    aligned[:actual_dim] = state_vec
    return aligned


def _close_remote_inference_client() -> None:
    conn = _MP_GPU_CLIENT_CACHE.get("conn")
    if conn is not None:
        try:
            conn.send({"type": "close"})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
    _MP_GPU_CLIENT_CACHE["conn"] = None


def _configure_remote_inference(config_like) -> None:
    enabled = bool(getattr(config_like, "gpu_rollout_inference_enabled", False))
    endpoint = getattr(config_like, "gpu_inference_endpoint", None)
    authkey = getattr(config_like, "gpu_inference_authkey", None)
    if isinstance(config_like, dict):
        enabled = bool(config_like.get("gpu_rollout_inference_enabled", False))
        endpoint = config_like.get("gpu_inference_endpoint")
        authkey = config_like.get("gpu_inference_authkey")
    if isinstance(authkey, bytearray):
        authkey = bytes(authkey)
    cache_changed = endpoint != _MP_GPU_CLIENT_CACHE.get("endpoint") or authkey != _MP_GPU_CLIENT_CACHE.get("authkey")
    if cache_changed:
        _close_remote_inference_client()
    _MP_GPU_CLIENT_CACHE["enabled"] = enabled and endpoint is not None and authkey is not None
    _MP_GPU_CLIENT_CACHE["endpoint"] = endpoint
    _MP_GPU_CLIENT_CACHE["authkey"] = authkey
    if not _MP_GPU_CLIENT_CACHE["enabled"]:
        _close_remote_inference_client()


def _get_remote_inference_client():
    if not _MP_GPU_CLIENT_CACHE.get("enabled"):
        return None
    conn = _MP_GPU_CLIENT_CACHE.get("conn")
    if conn is not None:
        return conn
    endpoint = _MP_GPU_CLIENT_CACHE.get("endpoint")
    authkey = _MP_GPU_CLIENT_CACHE.get("authkey")
    if endpoint is None or authkey is None:
        return None
    conn = Client(endpoint, authkey=authkey)
    _MP_GPU_CLIENT_CACHE["conn"] = conn
    return conn


def _normalize_heads(heads) -> tuple[str, ...]:
    if heads is None:
        return ("regret", "strategy", "exploit")
    if isinstance(heads, str):
        heads = (heads,)
    normalized: List[str] = []
    for head in heads:
        name = str(head).strip().lower()
        if name in {"regret", "strategy", "exploit"} and name not in normalized:
            normalized.append(name)
    return tuple(normalized) or ("strategy",)


def _remote_forward(model_key: str, state_vec: np.ndarray, heads=None) -> Dict[str, np.ndarray]:
    conn = _get_remote_inference_client()
    if conn is None:
        raise RuntimeError("GPU rollout inference client is not configured.")
    request = {
        "type": "infer",
        "model_key": str(model_key),
        "state_vec": np.asarray(state_vec, dtype=np.float32),
        "heads": list(_normalize_heads(heads)),
    }
    try:
        conn.send(request)
        response = conn.recv()
    except Exception:
        _close_remote_inference_client()
        conn = _get_remote_inference_client()
        if conn is None:
            raise
        conn.send(request)
        response = conn.recv()
    if not isinstance(response, dict) or not bool(response.get("ok", False)):
        raise RuntimeError(str(response.get("error", "GPU rollout inference request failed.")))
    outputs = response.get("outputs", {})
    if not isinstance(outputs, dict):
        raise RuntimeError("GPU rollout inference response is missing outputs.")
    return {str(name): np.asarray(value, dtype=np.float32) for name, value in outputs.items()}


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
    preflop_call_count: int = 0
    preflop_opened: bool = False
    preflop_last_raiser: Optional[int] = None
    last_aggressor: Optional[int] = None
    last_aggressive_size_bb: float = 0.0
    cbet_flop_initiator: Optional[int] = None
    cbet_turn_initiator: Optional[int] = None


@dataclass
class TraversalResult:
    advantage_samples: List[tuple] = field(default_factory=list)
    strategy_samples: List[tuple] = field(default_factory=list)
    exploit_samples: List[tuple] = field(default_factory=list)
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
    preflop_stats: Dict[str, List[int]] = field(default_factory=dict)
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
    preflop_stats: Dict[str, List[int]] = field(default_factory=dict)


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
        if hand_ctx.current_street == 1:
            hand_ctx.cbet_flop_initiator = None
        elif hand_ctx.current_street == 2:
            hand_ctx.cbet_turn_initiator = None


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


def _forward_outputs(model: Optional[object], state_vec: np.ndarray, heads=None) -> Dict[str, np.ndarray]:
    selected_heads = _normalize_heads(heads)
    if model is None:
        zeros = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
        return {head: zeros.copy() for head in selected_heads}
    if state_vec.dtype != np.float32:
        state_vec = state_vec.astype(np.float32, copy=False)
    state_vec = _aligned_state_vector_for_model(model, state_vec)
    remote_key = getattr(model, "gpu_service_key", "")
    if remote_key and _MP_GPU_CLIENT_CACHE.get("enabled"):
        return _remote_forward(str(remote_key), state_vec, heads=selected_heads)
    with torch.inference_mode():
        tensor = torch.from_numpy(state_vec).unsqueeze(0)
        if selected_heads == ("regret",):
            outputs = {"regret": model.forward_regret(tensor)}
        elif selected_heads == ("strategy",):
            outputs = {"strategy": model.forward_strategy(tensor)}
        elif selected_heads == ("exploit",):
            outputs = {"exploit": model.forward_exploit(tensor)}
        else:
            trunk = model._forward_trunk(tensor)
            outputs = {}
            if "regret" in selected_heads:
                outputs["regret"] = model.regret_head(trunk)
            if "strategy" in selected_heads:
                outputs["strategy"] = model.strategy_head(trunk)
            if "exploit" in selected_heads:
                outputs["exploit"] = model.exploit_head(trunk)
        return {name: value.squeeze(0).cpu().numpy().astype(np.float32) for name, value in outputs.items()}


def _inference(model: Optional[object], state_vec: np.ndarray, head: str) -> np.ndarray:
    outputs = _forward_outputs(model, state_vec, heads=(head,))
    return np.asarray(outputs.get(head, np.zeros(ACTION_COUNT_V21, dtype=np.float32)), dtype=np.float32)


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


def _normalize_masked_policy(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    probs = np.asarray(policy, dtype=np.float32).reshape(-1)
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    probs = probs * np.where(mask > 0.5, 1.0, 0.0)
    total = float(probs.sum())
    if total <= 1e-9:
        fallback = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        denom = float(fallback.sum())
        if denom <= 1e-9:
            return np.full_like(mask, 1.0 / max(1, mask.shape[0]), dtype=np.float32)
        return (fallback / denom).astype(np.float32)
    return (probs / total).astype(np.float32)


def _first_legal_action(legal_mask: np.ndarray, *actions: int) -> int:
    for action_id in actions:
        if 0 <= int(action_id) < len(legal_mask) and float(legal_mask[int(action_id)]) > 0.5:
            return int(action_id)
    for action_id, is_legal in enumerate(legal_mask):
        if float(is_legal) > 0.5:
            return int(action_id)
    return ACTION_CHECK


def _aggressive_action(legal_mask: np.ndarray, prefer_large: bool = False) -> int:
    if prefer_large:
        return _first_legal_action(
            legal_mask,
            ACTION_RAISE_POT_OR_ALL_IN,
            ACTION_RAISE_HALF_POT,
            ACTION_CALL,
            ACTION_CHECK,
            ACTION_FOLD,
        )
    return _first_legal_action(
        legal_mask,
        ACTION_RAISE_HALF_POT,
        ACTION_RAISE_POT_OR_ALL_IN,
        ACTION_CALL,
        ACTION_CHECK,
        ACTION_FOLD,
    )


def _passive_action(legal_mask: np.ndarray, facing_wager: bool) -> int:
    if facing_wager:
        return _first_legal_action(
            legal_mask,
            ACTION_CALL,
            ACTION_CHECK,
            ACTION_FOLD,
            ACTION_RAISE_HALF_POT,
            ACTION_RAISE_POT_OR_ALL_IN,
        )
    return _first_legal_action(
        legal_mask,
        ACTION_CHECK,
        ACTION_CALL,
        ACTION_FOLD,
        ACTION_RAISE_HALF_POT,
        ACTION_RAISE_POT_OR_ALL_IN,
    )


def _fold_action(legal_mask: np.ndarray) -> int:
    return _first_legal_action(
        legal_mask,
        ACTION_FOLD,
        ACTION_CHECK,
        ACTION_CALL,
        ACTION_RAISE_HALF_POT,
        ACTION_RAISE_POT_OR_ALL_IN,
    )


def _synthetic_preflop_action(
    style: str,
    legal_mask: np.ndarray,
    hand_ctx,
    preflop_strength: float,
    pot_odds: float,
    to_call_bb: float,
    rng: random.Random,
) -> int:
    style = str(style or "nit").lower()
    facing_wager = float(to_call_bb) > 1e-6
    preflop_raises = int(getattr(hand_ctx, "preflop_raise_count", 0))

    if style == "nit":
        if preflop_raises == 0:
            if preflop_strength >= 0.80:
                return _aggressive_action(legal_mask)
            return _passive_action(legal_mask, facing_wager)
        if preflop_raises == 1:
            if preflop_strength >= 0.92 and rng.random() < 0.55:
                return _aggressive_action(legal_mask)
            if preflop_strength >= 0.84:
                return _passive_action(legal_mask, True)
            return _fold_action(legal_mask)
        if preflop_strength >= 0.95:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)

    if style == "overfolder":
        if preflop_raises == 0:
            if preflop_strength >= 0.48:
                return _aggressive_action(legal_mask)
            return _passive_action(legal_mask, facing_wager)
        if preflop_strength >= 0.90 and rng.random() < 0.35:
            return _aggressive_action(legal_mask)
        if preflop_strength >= 0.82:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)

    if style == "overcaller":
        if preflop_raises == 0:
            if preflop_strength >= 0.68:
                return _aggressive_action(legal_mask)
            if preflop_strength >= 0.32 and rng.random() < 0.60:
                return _passive_action(legal_mask, facing_wager)
            return _passive_action(legal_mask, facing_wager)
        if preflop_strength >= 0.84 and rng.random() < 0.20:
            return _aggressive_action(legal_mask)
        if preflop_strength >= 0.28 or pot_odds <= 0.32 or rng.random() < 0.30:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)

    if style == "over3better":
        if preflop_raises == 0:
            if preflop_strength >= 0.38 or rng.random() < 0.18:
                return _aggressive_action(legal_mask)
            if preflop_strength >= 0.24:
                return _passive_action(legal_mask, facing_wager)
            return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)
        if preflop_raises == 1:
            if preflop_strength >= 0.42 or rng.random() < 0.42:
                return _aggressive_action(legal_mask)
            if preflop_strength >= 0.22 or pot_odds <= 0.24:
                return _passive_action(legal_mask, True)
            return _fold_action(legal_mask)
        if preflop_strength >= 0.56 or rng.random() < 0.22:
            return _aggressive_action(legal_mask, prefer_large=True)
        if preflop_strength >= 0.30 or pot_odds <= 0.18:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)

    if style == "station":
        if preflop_raises == 0:
            if preflop_strength >= 0.72 and rng.random() < 0.45:
                return _aggressive_action(legal_mask)
            if preflop_strength >= 0.18 or rng.random() < 0.55:
                return _passive_action(legal_mask, facing_wager)
            return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)
        if preflop_strength >= 0.90 and rng.random() < 0.15:
            return _aggressive_action(legal_mask)
        if preflop_strength >= 0.16 or pot_odds <= 0.45 or rng.random() < 0.55:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)

    if preflop_raises == 0:
        if preflop_strength >= 0.28 or rng.random() < 0.75:
            return _aggressive_action(legal_mask, prefer_large=rng.random() < 0.35)
        return _passive_action(legal_mask, facing_wager)
    if preflop_raises == 1:
        if preflop_strength >= 0.36 or rng.random() < 0.62:
            return _aggressive_action(legal_mask, prefer_large=rng.random() < 0.45)
        if preflop_strength >= 0.18 or rng.random() < 0.30:
            return _passive_action(legal_mask, True)
        return _fold_action(legal_mask)
    if preflop_strength >= 0.46 or rng.random() < 0.35:
        return _aggressive_action(legal_mask, prefer_large=True)
    if pot_odds <= 0.32 or rng.random() < 0.30:
        return _passive_action(legal_mask, True)
    return _fold_action(legal_mask)


def _synthetic_postflop_action(
    style: str,
    legal_mask: np.ndarray,
    made_strength: float,
    pot_odds: float,
    to_call_bb: float,
    hand_ctx,
    rng: random.Random,
) -> int:
    style = str(style or "nit").lower()
    facing_wager = float(to_call_bb) > 1e-6
    street_raises = int(getattr(hand_ctx, "street_raise_count", 0))

    if style == "nit":
        if made_strength >= 0.84:
            return _aggressive_action(legal_mask, prefer_large=True)
        if made_strength >= 0.62 and not facing_wager:
            return _aggressive_action(legal_mask)
        if made_strength >= max(0.58, pot_odds + 0.08):
            return _passive_action(legal_mask, facing_wager)
        return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)

    if style == "overfolder":
        if made_strength >= 0.86 and rng.random() < 0.55:
            return _aggressive_action(legal_mask)
        if made_strength >= max(0.66, pot_odds + 0.16):
            return _passive_action(legal_mask, facing_wager)
        return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)

    if style == "overcaller":
        if made_strength >= 0.88 and rng.random() < 0.18:
            return _aggressive_action(legal_mask)
        if made_strength >= 0.34 or pot_odds <= 0.40 or rng.random() < 0.30:
            return _passive_action(legal_mask, facing_wager)
        return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)

    if style == "over3better":
        if made_strength >= 0.72:
            return _aggressive_action(legal_mask, prefer_large=street_raises > 0 or rng.random() < 0.30)
        if not facing_wager and made_strength >= 0.42:
            return _aggressive_action(legal_mask)
        if made_strength >= max(0.34, pot_odds + 0.02):
            return _passive_action(legal_mask, facing_wager)
        return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)

    if style == "station":
        if made_strength >= 0.92 and rng.random() < 0.12:
            return _aggressive_action(legal_mask)
        if made_strength >= 0.26 or pot_odds <= 0.52 or rng.random() < 0.35:
            return _passive_action(legal_mask, facing_wager)
        return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)

    if made_strength >= 0.58 or rng.random() < 0.45:
        return _aggressive_action(legal_mask, prefer_large=street_raises > 0 or rng.random() < 0.40)
    if made_strength >= 0.22 or pot_odds <= 0.35 or rng.random() < 0.28:
        return _passive_action(legal_mask, facing_wager)
    return _fold_action(legal_mask) if facing_wager else _passive_action(legal_mask, False)


def _synthetic_opponent_action(
    style: str,
    state,
    actor: int,
    hand_ctx,
    rng: random.Random,
    state_vec: Optional[np.ndarray] = None,
    legal_mask: Optional[np.ndarray] = None,
) -> int:
    if state_vec is None or legal_mask is None:
        state_vec, legal_mask = encode_info_state(state, actor, hand_ctx, return_legal_mask=True)
    hole_cards = flatten_cards_list(state.hole_cards[actor])
    board_cards = flatten_cards_list(state.board_cards)
    preflop_strength = estimate_preflop_strength(hole_cards)
    made_strength = _best_hand_strength_scalar(hole_cards, board_cards)
    pot_odds = float(state_vec[76]) if len(state_vec) > 76 else 0.0
    to_call_bb = (max(state.bets) - state.bets[actor]) / float(max(1, getattr(hand_ctx, "big_blind", 10)))

    if int(getattr(hand_ctx, "current_street", 0)) == 0:
        return _synthetic_preflop_action(
            style,
            legal_mask,
            hand_ctx,
            preflop_strength,
            pot_odds,
            float(to_call_bb),
            rng,
        )
    return _synthetic_postflop_action(
        style,
        legal_mask,
        made_strength,
        pot_odds,
        float(to_call_bb),
        hand_ctx,
        rng,
    )


def _extract_profile_value(opponent_profile, idx: int) -> float:
    if isinstance(opponent_profile, (list, tuple, np.ndarray)) and idx < len(opponent_profile):
        return float(max(0.0, min(1.0, float(opponent_profile[idx]))))
    return 0.0


def _aggregate_opponent_profile(
    opponent_profiles_by_seat,
    hero_seat: int,
    hand_ctx: HandContext,
    player_count: int,
) -> tuple[np.ndarray, float]:
    aggregate = np.zeros(OPP_PROFILE_IDX_CONFIDENCE + 1, dtype=np.float32)
    if not isinstance(opponent_profiles_by_seat, dict):
        return aggregate, 0.0

    total_weight = 0.0
    confidence_sum = 0.0
    active_flags = getattr(hand_ctx, "in_hand", [True] * max(1, int(player_count)))
    for seat in range(max(1, int(player_count))):
        if seat == int(hero_seat):
            continue
        if seat < len(active_flags) and not bool(active_flags[seat]):
            continue
        seat_profile = opponent_profiles_by_seat.get(seat)
        if not isinstance(seat_profile, (list, tuple, np.ndarray)):
            continue
        confidence = _extract_profile_value(seat_profile, OPP_PROFILE_IDX_CONFIDENCE)
        weight = max(0.01, confidence)
        total_weight += weight
        confidence_sum += confidence
        for idx in range(OPP_PROFILE_IDX_CONFIDENCE + 1):
            aggregate[idx] += float(_extract_profile_value(seat_profile, idx) * weight)

    if total_weight <= 1e-9:
        return aggregate, 0.0
    aggregate /= float(total_weight)
    mean_confidence = float(max(0.0, min(1.0, confidence_sum / max(1.0, total_weight))))
    return aggregate.astype(np.float32), mean_confidence


def _aggregate_selected_opponent_profile(
    opponent_profiles_by_seat,
    selected_seats: List[int],
) -> tuple[np.ndarray, float]:
    aggregate = np.zeros(OPP_PROFILE_IDX_CONFIDENCE + 1, dtype=np.float32)
    if not isinstance(opponent_profiles_by_seat, dict):
        return aggregate, 0.0

    total_weight = 0.0
    confidence_sum = 0.0
    for seat in selected_seats:
        seat_profile = opponent_profiles_by_seat.get(int(seat))
        if not isinstance(seat_profile, (list, tuple, np.ndarray)):
            continue
        confidence = _extract_profile_value(seat_profile, OPP_PROFILE_IDX_CONFIDENCE)
        weight = max(0.01, confidence)
        total_weight += weight
        confidence_sum += confidence
        for idx in range(OPP_PROFILE_IDX_CONFIDENCE + 1):
            aggregate[idx] += float(_extract_profile_value(seat_profile, idx) * weight)

    if total_weight <= 1e-9:
        return aggregate, 0.0
    aggregate /= float(total_weight)
    mean_confidence = float(max(0.0, min(1.0, confidence_sum / max(1.0, total_weight))))
    return aggregate.astype(np.float32), mean_confidence


def _new_exploit_policy_scores(legal_mask: np.ndarray, facing_wager: bool) -> np.ndarray:
    scores = np.full_like(legal_mask, 1e-3, dtype=np.float32)
    if facing_wager:
        scores[ACTION_FOLD] += 0.18
        scores[ACTION_CALL] += 0.42
        scores[ACTION_CHECK] += 0.06
    else:
        scores[ACTION_CHECK] += 0.44
        scores[ACTION_CALL] += 0.12
    if float(legal_mask[ACTION_RAISE_HALF_POT]) > 0.5:
        scores[ACTION_RAISE_HALF_POT] += 0.24
    if float(legal_mask[ACTION_RAISE_POT_OR_ALL_IN]) > 0.5:
        scores[ACTION_RAISE_POT_OR_ALL_IN] += 0.08
    return scores


def compute_exploit_guidance(
    state_vec: np.ndarray,
    legal_mask: np.ndarray,
    opponent_profile,
    opponent_profiles_by_seat,
    actor: int,
    hand_ctx: HandContext,
    player_count: int,
    config,
) -> Dict[str, object]:
    street = int(getattr(hand_ctx, "current_street", 0))
    preflop_raise_count = int(getattr(hand_ctx, "preflop_raise_count", 0))
    facing_wager = bool(float(legal_mask[ACTION_CALL]) > 0.5 and float(legal_mask[ACTION_CHECK]) <= 0.5)
    active_flags = getattr(hand_ctx, "in_hand", [True] * max(1, int(player_count)))
    selected_seats: List[int] = []
    if street == 0 and preflop_raise_count >= 1:
        opener = getattr(hand_ctx, "preflop_last_raiser", None)
        if opener is not None and int(opener) != int(actor):
            selected_seats = [int(opener)]
    elif street > 0 and facing_wager:
        aggressor = getattr(hand_ctx, "last_aggressor", None)
        if aggressor is not None and int(aggressor) != int(actor):
            selected_seats = [int(aggressor)]
    else:
        for rel_offset in range(1, max(1, int(player_count))):
            seat = (int(actor) + rel_offset) % int(player_count)
            if seat < len(active_flags) and bool(active_flags[seat]):
                selected_seats.append(int(seat))

    aggregated, aggregated_conf = _aggregate_selected_opponent_profile(opponent_profiles_by_seat, selected_seats)
    if aggregated_conf <= 1e-9:
        aggregated, aggregated_conf = _aggregate_opponent_profile(
            opponent_profiles_by_seat,
            hero_seat=int(actor),
            hand_ctx=hand_ctx,
            player_count=int(player_count),
        )
    if aggregated_conf <= 1e-9 and isinstance(opponent_profile, (list, tuple, np.ndarray)):
        for idx in range(min(OPP_PROFILE_IDX_CONFIDENCE + 1, len(opponent_profile))):
            aggregated[idx] = _extract_profile_value(opponent_profile, idx)
        aggregated_conf = _extract_profile_value(aggregated, OPP_PROFILE_IDX_CONFIDENCE)

    confidence = float(max(0.0, min(1.0, aggregated_conf)))
    vpip_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_VPIP)
    pfr_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_PFR)
    three_bet_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_THREE_BET)
    fold_to_open_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_FOLD_TO_OPEN)
    fold_to_three_bet_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_FOLD_TO_THREE_BET)
    call_open_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_CALL_OPEN)
    fold_to_cbet_flop_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_FOLD_TO_CBET_FLOP)
    fold_to_cbet_turn_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_FOLD_TO_CBET_TURN)
    aggression_rate = _extract_profile_value(aggregated, OPP_PROFILE_IDX_AGGRESSION)

    min_confidence = float(max(0.0, min(1.0, getattr(config, "exploit_min_confidence", 0.10))))
    if confidence < min_confidence:
        return {
            "teacher_policy": None,
            "blend_lambda": 0.0,
            "confidence": confidence,
            "leak_score": 0.0,
            "reasons": [],
            "aggregated_profile": aggregated.astype(np.float32),
        }

    if bool(getattr(config, "exploit_only_preflop_unopened", False)) and (street != 0 or preflop_raise_count != 0):
        return {
            "teacher_policy": None,
            "blend_lambda": 0.0,
            "confidence": confidence,
            "leak_score": 0.0,
            "reasons": [],
            "aggregated_profile": aggregated.astype(np.float32),
        }
    can_raise = bool(
        float(legal_mask[ACTION_RAISE_HALF_POT]) > 0.5 or float(legal_mask[ACTION_RAISE_POT_OR_ALL_IN]) > 0.5
    )
    preflop_strength = float(state_vec[58]) if len(state_vec) > 58 else 0.5
    hand_strength = float(state_vec[60]) if len(state_vec) > 60 else preflop_strength
    pot_odds = float(state_vec[76]) if len(state_vec) > 76 else 0.0
    hero_is_last_aggressor = bool(len(state_vec) > 95 and float(state_vec[95]) > 0.5)

    scores = _new_exploit_policy_scores(legal_mask, facing_wager)
    reasons: List[str] = []
    signal_terms: List[float] = []

    if street == 0:
        defend_vs_open_rate = max(0.0, min(1.0, call_open_rate + three_bet_rate))
        overfold_score = max(0.0, fold_to_open_rate - 0.58)
        underdefend_score = max(0.0, 0.38 - defend_vs_open_rate)
        tight_score = max(0.0, 0.21 - vpip_rate)

        if preflop_raise_count == 0 and can_raise:
            steal_signal = max(0.0, min(1.0, overfold_score * 1.6 + underdefend_score * 1.1 + tight_score * 0.8))
            if steal_signal > 1e-3:
                open_boost = float(0.95 + (0.65 * steal_signal))
                if preflop_strength >= 0.42:
                    scores[ACTION_RAISE_HALF_POT] += open_boost
                    if float(legal_mask[ACTION_RAISE_POT_OR_ALL_IN]) > 0.5 and preflop_strength >= 0.72:
                        scores[ACTION_RAISE_POT_OR_ALL_IN] += 0.18 * steal_signal
                else:
                    scores[ACTION_CHECK] += 0.22
                scores[ACTION_FOLD] *= 0.15
                scores[ACTION_CALL] *= 0.45
                signal_terms.append(steal_signal)
                reasons.append(
                    f"Steal wider: villains fold to opens {fold_to_open_rate:.0%} with {confidence:.0%} confidence."
                )

        if preflop_raise_count >= 1:
            if can_raise:
                punish_three_bet_signal = max(
                    0.0,
                    min(1.0, max(0.0, fold_to_three_bet_rate - 0.55) * 1.5 + max(0.0, pfr_rate - 0.22) * 0.8),
                )
                if punish_three_bet_signal > 1e-3 and preflop_strength >= 0.34:
                    scores[ACTION_RAISE_HALF_POT] += float(0.80 + 0.60 * punish_three_bet_signal)
                    if float(legal_mask[ACTION_RAISE_POT_OR_ALL_IN]) > 0.5 and preflop_strength >= 0.60:
                        scores[ACTION_RAISE_POT_OR_ALL_IN] += 0.20 * punish_three_bet_signal
                    signal_terms.append(punish_three_bet_signal)
                    reasons.append(
                        f"3-bet punish more: fold-to-3bet is {fold_to_three_bet_rate:.0%}."
                    )

            if facing_wager:
                mania_signal = max(
                    0.0,
                    min(1.0, max(0.0, three_bet_rate - 0.18) * 1.1 + max(0.0, aggression_rate - 0.58) * 0.9),
                )
                if mania_signal > 1e-3 and preflop_strength >= 0.38:
                    scores[ACTION_CALL] += float(0.50 + 0.40 * mania_signal)
                    scores[ACTION_FOLD] *= max(0.35, 1.0 - 0.55 * mania_signal)
                    signal_terms.append(mania_signal * 0.75)
                    reasons.append(
                        f"Defend lighter: villains 3-bet {three_bet_rate:.0%} and play aggressively."
                    )

                station_signal = max(0.0, min(1.0, max(0.0, call_open_rate - 0.46) * 1.1 + max(0.0, vpip_rate - 0.34) * 0.6))
                if station_signal > 1e-3 and preflop_strength < 0.28:
                    scores[ACTION_FOLD] += float(0.55 + 0.35 * station_signal)
                    scores[ACTION_CALL] *= 0.65
                    signal_terms.append(station_signal * 0.55)
                    reasons.append(
                        f"Tighten weak continues: villains call opens {call_open_rate:.0%}."
                    )
    else:
        if hero_is_last_aggressor and can_raise and not facing_wager and street in (1, 2):
            fold_to_cbet_rate = fold_to_cbet_flop_rate if street == 1 else fold_to_cbet_turn_rate
            cbet_signal = max(0.0, fold_to_cbet_rate - 0.56)
            if cbet_signal > 1e-3 and hand_strength < 0.78:
                scores[ACTION_RAISE_HALF_POT] += float(0.80 + 0.55 * cbet_signal)
                scores[ACTION_CHECK] *= max(0.30, 1.0 - 0.60 * cbet_signal)
                signal_terms.append(cbet_signal)
                street_label = "flop" if street == 1 else "turn"
                reasons.append(
                    f"Bluff c-bet more: villains fold to {street_label} c-bets {fold_to_cbet_rate:.0%}."
                )

        station_signal = max(0.0, min(1.0, max(0.0, call_open_rate - 0.45) * 0.9 + max(0.0, 0.42 - fold_to_cbet_flop_rate) * 0.9))
        if station_signal > 1e-3:
            if hand_strength < 0.50 and not facing_wager:
                scores[ACTION_CHECK] += float(0.55 + 0.35 * station_signal)
                scores[ACTION_RAISE_HALF_POT] *= 0.55
                signal_terms.append(station_signal * 0.7)
                reasons.append("Bluff less: villains are sticky and under-fold postflop.")
            elif hand_strength >= 0.62 and can_raise:
                scores[ACTION_RAISE_HALF_POT] += float(0.45 + 0.35 * station_signal)
                if float(legal_mask[ACTION_RAISE_POT_OR_ALL_IN]) > 0.5 and hand_strength >= 0.82:
                    scores[ACTION_RAISE_POT_OR_ALL_IN] += 0.22 * station_signal
                signal_terms.append(station_signal * 0.65)
                reasons.append("Value-bet harder: villains call too often.")

        if facing_wager:
            mania_signal = max(0.0, min(1.0, max(0.0, aggression_rate - 0.60) * 1.1 + max(0.0, three_bet_rate - 0.16) * 0.5))
            if mania_signal > 1e-3 and hand_strength >= max(0.42, pot_odds + 0.04):
                scores[ACTION_CALL] += float(0.60 + 0.35 * mania_signal)
                scores[ACTION_FOLD] *= max(0.35, 1.0 - 0.50 * mania_signal)
                signal_terms.append(mania_signal * 0.8)
                reasons.append("Call down lighter: villains over-apply pressure.")

            passive_value_signal = max(0.0, min(1.0, max(0.0, 0.42 - aggression_rate) * 0.9 + max(0.0, 0.26 - pfr_rate) * 0.5))
            if passive_value_signal > 1e-3 and hand_strength < max(0.56, pot_odds + 0.10):
                scores[ACTION_FOLD] += float(0.50 + 0.40 * passive_value_signal)
                scores[ACTION_CALL] *= 0.70
                signal_terms.append(passive_value_signal * 0.65)
                reasons.append("Respect passive aggression: range skews stronger.")

    leak_score = float(max(0.0, min(1.0, sum(signal_terms))))
    lambda_max = float(max(0.0, min(1.0, getattr(config, "exploit_prior_strength", 0.35))))
    blend_lambda = float(max(0.0, min(lambda_max, confidence * leak_score)))
    if blend_lambda <= 1e-4:
        return {
            "teacher_policy": None,
            "blend_lambda": 0.0,
            "confidence": confidence,
            "leak_score": leak_score,
            "reasons": reasons[:3],
            "aggregated_profile": aggregated.astype(np.float32),
        }

    scores = np.clip(scores, 1e-4, None)
    teacher_policy = _normalize_masked_policy(scores.astype(np.float32), legal_mask)
    return {
        "teacher_policy": teacher_policy,
        "blend_lambda": blend_lambda,
        "confidence": confidence,
        "leak_score": leak_score,
        "reasons": reasons[:3],
        "aggregated_profile": aggregated.astype(np.float32),
    }


def _exploit_prior_from_profile(
    state_vec: np.ndarray,
    opponent_profile,
    opponent_profiles_by_seat,
    actor: int,
    hand_ctx: HandContext,
    player_count: int,
    legal_mask: np.ndarray,
    config,
) -> tuple[Optional[np.ndarray], float, Dict[str, object]]:
    if not getattr(config, "exploit_prior_enabled", False):
        return None, 0.0, {
            "teacher_policy": None,
            "blend_lambda": 0.0,
            "confidence": 0.0,
            "leak_score": 0.0,
            "reasons": [],
            "aggregated_profile": np.zeros(OPP_PROFILE_IDX_CONFIDENCE + 1, dtype=np.float32),
        }

    guidance = compute_exploit_guidance(
        state_vec,
        legal_mask,
        opponent_profile,
        opponent_profiles_by_seat,
        actor,
        hand_ctx,
        player_count,
        config,
    )
    teacher_policy = guidance.get("teacher_policy")
    blend_lambda = float(guidance.get("blend_lambda", 0.0) or 0.0)
    return teacher_policy, blend_lambda, guidance


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
            if hand_ctx.current_street == 0 and float(to_call) > 1e-6 and requested == ACTION_CALL:
                hand_ctx.preflop_call_count += 1
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
            hand_ctx.preflop_last_raiser = actor
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
    traverser_opponent_profile = getattr(config, "opponent_profile", OPPONENT_PROFILE_DEFAULT_V23)
    traverser_profiles_by_seat = getattr(config, "opponent_profiles_by_seat", None)
    opponent_policy_mode = str(getattr(config, "opponent_policy_mode", "snapshot") or "snapshot").lower()
    synthetic_opponent_style = str(getattr(config, "synthetic_opponent_style", "") or "").lower()
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
                state_vec, legal_mask = encode_info_state(
                    state,
                    actor,
                    hand_ctx,
                    return_legal_mask=True,
                    opponent_profile=traverser_opponent_profile,
                    opponent_profiles_by_seat=traverser_profiles_by_seat,
                )
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
            actor_outputs = _forward_outputs(actor_snapshot, state_vec, heads=("regret", "exploit"))
            infer_elapsed = time.perf_counter() - infer_start
            perf["regret_infer_time"] += infer_elapsed * 0.5
            perf["strategy_infer_time"] += infer_elapsed * 0.5
            regret_logits = actor_outputs["regret"]
            exploit_logits = actor_outputs["exploit"]
            base_sigma = regret_matching(regret_logits, legal_mask)
            exploit_sigma = masked_policy(exploit_logits, legal_mask)
            exploit_prior, exploit_strength, _ = _exploit_prior_from_profile(
                state_vec,
                traverser_opponent_profile,
                traverser_profiles_by_seat,
                actor,
                hand_ctx,
                len(state.stacks),
                legal_mask,
                config,
            )
            teacher_mix = float(max(0.0, min(1.0, getattr(config, "exploit_teacher_mix", 0.65))))
            if exploit_prior is not None:
                exploit_sigma = _normalize_masked_policy(
                    ((1.0 - teacher_mix) * exploit_sigma) + (teacher_mix * exploit_prior),
                    legal_mask,
                )
            sigma = _normalize_masked_policy(
                ((1.0 - float(max(0.0, min(1.0, exploit_strength)))) * base_sigma)
                + (float(max(0.0, min(1.0, exploit_strength))) * exploit_sigma),
                legal_mask,
            )
            exploit_target = exploit_prior if exploit_prior is not None else base_sigma
            exploit_weight = float(
                max(
                    max(0.0, min(1.0, getattr(config, "exploit_safety_weight", 0.20))),
                    exploit_strength,
                )
            )

            if traverser_depth < config.full_branch_depth:
                action_values = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
                legal_actions = [action_id for action_id, is_legal in enumerate(legal_mask) if is_legal > 0.5]
                branch_actions = list(legal_actions)
                max_branch_actions = int(getattr(config, "max_branch_actions", 0))
                if max_branch_actions > 0 and len(branch_actions) > max_branch_actions:
                    ordered = np.argsort(base_sigma[branch_actions])[::-1]
                    branch_actions = [branch_actions[idx] for idx in ordered[:max_branch_actions]]
                branch_mask = np.zeros(ACTION_COUNT_V21, dtype=np.float32)
                for action_id in branch_actions:
                    branch_mask[action_id] = 1.0
                if branch_mask.sum() > 0:
                    branch_sigma = base_sigma * branch_mask
                    branch_sigma /= max(float(branch_sigma.sum()), 1e-8)
                else:
                    branch_sigma = base_sigma
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
                    result.strategy_samples.append((state_vec, legal_mask.copy(), base_sigma.astype(np.float32), 1.0))
                    result.exploit_samples.append(
                        (state_vec, legal_mask.copy(), exploit_target.astype(np.float32), exploit_weight)
                    )
            elif record and result is not None:
                result.strategy_samples.append((state_vec, legal_mask.copy(), base_sigma.astype(np.float32), 1.0))
                result.exploit_samples.append(
                    (state_vec, legal_mask.copy(), exploit_target.astype(np.float32), exploit_weight)
                )

            chosen_action = _sample_action(sigma, rng)
            if record and result is not None:
                result.action_counts[chosen_action] += 1
                to_call = float(max(state.bets) - state.bets[actor])
                if hand_ctx.current_street == 0:
                    prior_preflop_raises = int(hand_ctx.preflop_raise_count)
                    _record_preflop_action_stats(
                        result.preflop_stats,
                        actor,
                        hand_ctx,
                        to_call,
                        prior_preflop_raises,
                        chosen_action,
                    )
                    if chosen_action in (ACTION_CALL, ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        result.vpip = True
                    if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        result.pfr = True
                        if prior_preflop_raises >= 1:
                            result.three_bet = True
                else:
                    _record_postflop_action_stats(result.preflop_stats, actor, hand_ctx, to_call, chosen_action)

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

        if opponent_policy_mode == "synthetic":
            chosen_action = _synthetic_opponent_action(
                synthetic_opponent_style,
                state,
                actor,
                hand_ctx,
                rng,
                state_vec=opponent_state,
                legal_mask=legal_mask,
            )
        else:
            policy_model = opponent_snapshot if opponent_snapshot is not None else actor_snapshot
            infer_start = time.perf_counter()
            policy_logits = _inference(policy_model, opponent_state, head="strategy")
            perf["strategy_infer_time"] += time.perf_counter() - infer_start
            probs = masked_policy(policy_logits, legal_mask)
            chosen_action = _sample_action(probs, rng)
        if record and result is not None:
            to_call = float(max(state.bets) - state.bets[actor])
            if hand_ctx.current_street == 0:
                prior_preflop_raises = int(hand_ctx.preflop_raise_count)
                _record_preflop_action_stats(
                    result.preflop_stats,
                    actor,
                    hand_ctx,
                    to_call,
                    prior_preflop_raises,
                    chosen_action,
                )
            else:
                _record_postflop_action_stats(result.preflop_stats, actor, hand_ctx, to_call, chosen_action)
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
    _configure_remote_inference(config)
    rng = random.Random(int(hand_seed))
    perf = _new_perf_breakdown()
    init_start = time.perf_counter()
    state, hand_ctx = _create_state_and_context(rng, config)
    perf["state_init_time"] += time.perf_counter() - init_start
    result = TraversalResult(traverser_seat=int(traverser_seat))
    result.preflop_stats = _new_preflop_stats(getattr(config, "num_players", 6))
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
    opponent_profile=None,
    opponent_profiles_by_seat=None,
    config=None,
    return_details: bool = False,
):
    state_vec, legal_mask = encode_info_state(
        state,
        actor,
        hand_ctx,
        return_legal_mask=True,
        opponent_profile=opponent_profile,
        opponent_profiles_by_seat=opponent_profiles_by_seat,
    )
    need_exploit = bool(snapshot is not None and config is not None and getattr(config, "exploit_prior_enabled", False))
    outputs = _forward_outputs(snapshot, state_vec, heads=("strategy", "exploit") if need_exploit else ("strategy",))
    base_policy = masked_policy(outputs["strategy"], legal_mask)
    final_policy = base_policy
    guidance = {
        "teacher_policy": None,
        "blend_lambda": 0.0,
        "confidence": 0.0,
        "leak_score": 0.0,
        "reasons": [],
        "aggregated_profile": np.zeros(OPP_PROFILE_IDX_CONFIDENCE + 1, dtype=np.float32),
    }
    if need_exploit:
        exploit_policy = masked_policy(outputs["exploit"], legal_mask)
        teacher_policy, blend_lambda, guidance = _exploit_prior_from_profile(
            state_vec,
            opponent_profile,
            opponent_profiles_by_seat,
            actor,
            hand_ctx,
            len(state.stacks),
            legal_mask,
            config,
        )
        if teacher_policy is not None:
            teacher_mix = float(max(0.0, min(1.0, getattr(config, "exploit_teacher_mix", 0.65))))
            exploit_policy = _normalize_masked_policy(
                ((1.0 - teacher_mix) * exploit_policy) + (teacher_mix * teacher_policy),
                legal_mask,
            )
        final_policy = _normalize_masked_policy(
            ((1.0 - blend_lambda) * base_policy) + (blend_lambda * exploit_policy),
            legal_mask,
        )

    chosen_action = _sample_action(final_policy, rng)
    if return_details:
        details = {
            "state_vec": state_vec,
            "legal_mask": legal_mask,
            "base_policy": base_policy,
            "final_policy": final_policy,
            "guidance": guidance,
        }
        return chosen_action, details
    return chosen_action


def run_policy_hand(hand_seed: int, actor_snapshot: Optional[PokerDeepCFRNet], config) -> HandResult:
    _configure_remote_inference(config)
    rng = random.Random(int(hand_seed))
    state, hand_ctx = _create_state_and_context(rng, config)
    hero_seat = int(getattr(config, "eval_hero_seat", 0)) % config.num_players
    opponent_mode = getattr(config, "evaluation_mode", "heuristics")
    checkpoint_pool = list(getattr(config, "checkpoint_pool", []))
    traverser_opponent_profile = getattr(config, "opponent_profile", OPPONENT_PROFILE_DEFAULT_V23)
    traverser_profiles_by_seat = getattr(config, "opponent_profiles_by_seat", None)
    configured_synthetic_style = str(getattr(config, "synthetic_opponent_style", "") or "").lower()

    seat_models: Dict[int, Optional[PokerDeepCFRNet]] = {}
    if opponent_mode == "checkpoints" and checkpoint_pool:
        opponent_seats = [seat for seat in range(config.num_players) if seat != hero_seat]
        for seat in opponent_seats:
            seat_models[seat] = rng.choice(checkpoint_pool)
    synthetic_style = ""
    if opponent_mode == "synthetic":
        synthetic_style = configured_synthetic_style if configured_synthetic_style in SYNTHETIC_OPPONENT_STYLES else "nit"
    elif opponent_mode in SYNTHETIC_OPPONENT_STYLES:
        synthetic_style = str(opponent_mode)

    action_counts = np.zeros(ACTION_COUNT_V21, dtype=np.int64)
    illegal_action_count = 0
    vpip = False
    pfr = False
    three_bet = False
    rfi_opportunity = False
    rfi_attempt = False
    hero_preflop_seen = False
    hero_hand_key = _canonical_preflop_key(flatten_cards_list(state.hole_cards[hero_seat]))
    preflop_stats = _new_preflop_stats(getattr(config, "num_players", 6))

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
            chosen_action = _policy_action_for_snapshot(
                actor_snapshot,
                state,
                actor,
                hand_ctx,
                rng,
                opponent_profile=traverser_opponent_profile,
                opponent_profiles_by_seat=traverser_profiles_by_seat,
                config=config,
            )
            action_counts[chosen_action] += 1
            if preflop_action and not hero_preflop_seen:
                hero_preflop_seen = True
                if prior_preflop_raises == 0:
                    rfi_opportunity = True
                    if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                        rfi_attempt = True
            to_call = float(max(state.bets) - state.bets[actor])
            if preflop_action:
                _record_preflop_action_stats(
                    preflop_stats,
                    actor,
                    hand_ctx,
                    to_call,
                    prior_preflop_raises,
                    chosen_action,
                )
                if chosen_action in (ACTION_CALL, ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                    vpip = True
                if chosen_action in (ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN):
                    pfr = True
                    if prior_preflop_raises >= 1:
                        three_bet = True
            else:
                _record_postflop_action_stats(preflop_stats, actor, hand_ctx, to_call, chosen_action)
        else:
            if opponent_mode == "checkpoints" and seat_models:
                chosen_action = _policy_action_for_snapshot(
                    seat_models.get(actor),
                    state,
                    actor,
                    hand_ctx,
                    rng,
                    config=config,
                )
            elif synthetic_style:
                chosen_action = _synthetic_opponent_action(synthetic_style, state, actor, hand_ctx, rng)
            else:
                chosen_action = _heuristic_action(state, actor, hand_ctx, rng)
            to_call = float(max(state.bets) - state.bets[actor])
            if hand_ctx.current_street == 0:
                _record_preflop_action_stats(
                    preflop_stats,
                    actor,
                    hand_ctx,
                    to_call,
                    int(hand_ctx.preflop_raise_count),
                    chosen_action,
                )
            else:
                _record_postflop_action_stats(preflop_stats, actor, hand_ctx, to_call, chosen_action)

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
        preflop_stats=preflop_stats,
    )


def _load_model_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    quantize_for_cpu_inference: bool = False,
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
    load_compatible_state_dict(model, state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if quantize_for_cpu_inference:
        quantize_dynamic = None
        try:
            quantize_dynamic = torch.ao.quantization.quantize_dynamic
        except Exception:
            quantize_dynamic = getattr(torch.quantization, "quantize_dynamic", None)
        if quantize_dynamic is not None:
            try:
                quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                quantized.eval()
                setattr(quantized, "state_dim", inferred_state_dim)
                setattr(quantized, "hidden_dim", inferred_hidden_dim)
                setattr(quantized, "action_dim", inferred_action_dim)
                model = quantized
            except Exception:
                pass
    return model


def run_traversal_batch_mp(
    hand_seeds: List[int],
    traverser_seats: List[int],
    actor_state_dict: Optional[Dict[str, torch.Tensor]],
    opponent_state_dict: Optional[Dict[str, torch.Tensor]],
    config_dict: Dict[str, object],
    snapshot_signature: str,
) -> List[TraversalResult]:
    _configure_remote_inference(config_dict)
    state_dim = int(config_dict.get("state_dim", STATE_DIM_V21))
    hidden_dim = int(config_dict.get("hidden_dim", 256))
    action_dim = int(config_dict.get("action_count", ACTION_COUNT_V21))
    use_gpu_service = bool(config_dict.get("gpu_rollout_inference_enabled", False))
    remote_opponents = bool(config_dict.get("gpu_remote_opponents", False))
    opponent_policy_mode = str(config_dict.get("opponent_policy_mode", "snapshot") or "snapshot").lower()
    quantize_local_opponents = bool(config_dict.get("quantize_local_opponent_models", False))
    actor_signature = str(config_dict.get("actor_cache_signature", snapshot_signature))
    opponent_signature = str(config_dict.get("opponent_cache_signature", snapshot_signature))

    if _MP_MODEL_CACHE.get("actor_signature") != actor_signature:
        _MP_MODEL_CACHE["actor"] = None
        if use_gpu_service:
            actor_model = RemoteModelRef(
                gpu_service_key=str(config_dict.get("gpu_actor_model_key", "actor_snapshot")),
                state_dim=int(config_dict.get("gpu_actor_state_dim", state_dim)),
                hidden_dim=int(config_dict.get("gpu_actor_hidden_dim", hidden_dim)),
                action_dim=int(config_dict.get("gpu_actor_action_dim", action_dim)),
            )
        else:
            if actor_state_dict is None:
                raise ValueError("actor_state_dict is required when GPU rollout inference is disabled.")
            actor_model = _load_model_from_state_dict(actor_state_dict, state_dim, hidden_dim, action_dim)
        _MP_MODEL_CACHE["actor_signature"] = actor_signature
        _MP_MODEL_CACHE["actor"] = actor_model

    actor_model = _MP_MODEL_CACHE["actor"]

    if _MP_MODEL_CACHE.get("opponent_signature") != opponent_signature:
        _MP_MODEL_CACHE["opponent"] = None
        if use_gpu_service and remote_opponents:
            opponent_key = str(config_dict.get("gpu_opponent_model_key", getattr(actor_model, "gpu_service_key", "actor_snapshot")) or getattr(actor_model, "gpu_service_key", "actor_snapshot"))
            if opponent_key == getattr(actor_model, "gpu_service_key", ""):
                opponent_model = actor_model
            else:
                opponent_model = RemoteModelRef(
                    gpu_service_key=opponent_key,
                    state_dim=int(config_dict.get("gpu_opponent_state_dim", state_dim)),
                    hidden_dim=int(config_dict.get("gpu_opponent_hidden_dim", hidden_dim)),
                    action_dim=int(config_dict.get("gpu_opponent_action_dim", action_dim)),
                )
        elif opponent_policy_mode != "snapshot":
            opponent_model = None
        elif opponent_state_dict is None:
            if actor_state_dict is None:
                if isinstance(actor_model, RemoteModelRef):
                    raise ValueError("actor_state_dict is required for local opponent inference.")
            opponent_model = _load_model_from_state_dict(
                actor_state_dict,
                state_dim,
                hidden_dim,
                action_dim,
                quantize_for_cpu_inference=quantize_local_opponents,
            )
        else:
            opponent_model = _load_model_from_state_dict(
                opponent_state_dict,
                state_dim,
                hidden_dim,
                action_dim,
                quantize_for_cpu_inference=quantize_local_opponents,
            )
        _MP_MODEL_CACHE["opponent_signature"] = opponent_signature
        _MP_MODEL_CACHE["opponent"] = opponent_model

    _MP_MODEL_CACHE["signature"] = snapshot_signature

    opponent_model = _MP_MODEL_CACHE["opponent"]
    config_ns = SimpleNamespace(**config_dict)
    results: List[TraversalResult] = []
    for seed, traverser_seat in zip(hand_seeds, traverser_seats):
        results.append(run_traversal(seed, traverser_seat, actor_model, opponent_model, config_ns))
    return results
