from __future__ import annotations

import copy
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
FEATURES_DIR = SRC_ROOT / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

try:
    from poker_state_v26 import ACTION_COUNT_V26, ACTION_NAMES_V26, POSITION_NAMES_V26
    from poker_trainer_v26 import DeepCFRConfigV26, DeepCFRTrainerV26
except ImportError:  # pragma: no cover - package-style imports
    from ..features.poker_state_v26 import ACTION_COUNT_V26, ACTION_NAMES_V26, POSITION_NAMES_V26
    from .poker_trainer_v26 import DeepCFRConfigV26, DeepCFRTrainerV26


SYNTHETIC_OPPONENT_STYLES: tuple[str, ...] = ("nit", "overfolder", "overcaller", "over3better", "station", "maniac")
POSTFLOP_RATE_KEYS = (
    "flop_seen",
    "turn_seen",
    "river_seen",
    "showdown_seen",
    "showdown_won",
    "cbet_flop",
    "fold_vs_cbet_flop",
    "cbet_turn",
    "fold_vs_cbet_turn",
)
POSTFLOP_COUNT_KEYS = (
    "hands",
    "flop_seen",
    "turn_seen",
    "river_seen",
    "showdown_seen",
    "showdown_won",
    "cbet_flop_opportunity",
    "cbet_flop_taken",
    "fold_vs_cbet_flop_opportunity",
    "fold_vs_cbet_flop",
    "cbet_turn_opportunity",
    "cbet_turn_taken",
    "fold_vs_cbet_turn_opportunity",
    "fold_vs_cbet_turn",
)
POSTFLOP_RATE_COUNT_KEYS = {
    "flop_seen": ("flop_seen", "hands"),
    "turn_seen": ("turn_seen", "hands"),
    "river_seen": ("river_seen", "hands"),
    "showdown_seen": ("showdown_seen", "hands"),
    "showdown_won": ("showdown_won", "showdown_seen"),
    "cbet_flop": ("cbet_flop_taken", "cbet_flop_opportunity"),
    "fold_vs_cbet_flop": ("fold_vs_cbet_flop", "fold_vs_cbet_flop_opportunity"),
    "cbet_turn": ("cbet_turn_taken", "cbet_turn_opportunity"),
    "fold_vs_cbet_turn": ("fold_vs_cbet_turn", "fold_vs_cbet_turn_opportunity"),
}
POSTFLOP_CONDITION_STREET_KEYS = ("flop", "turn", "river")
POSTFLOP_CONDITION_METRIC_KEYS = (
    "check_when_legal",
    "bet_raise_when_checked_to",
    "aggressive_when_checked_to",
    "fold_when_facing_bet",
    "call_when_facing_bet",
    "raise_when_facing_bet",
)
GUI_CHECKPOINT_FORMAT = "deepcfr_v26_gui"
RAISE_EPSILON_V26 = 0.01
RANK_TEXT_BY_VALUE = {
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9",
    8: "T",
    9: "J",
    10: "Q",
    11: "K",
    12: "A",
}
RANK_INDEX_HIGH_TO_LOW = {rank: idx for idx, rank in enumerate("AKQJT98765432")}


@dataclass
class DeepCFRGuiConfigV26(DeepCFRConfigV26):
    num_players: int = 6
    small_blind: float = 1.0
    big_blind: float = 2.0
    stake: float = 200.0
    traversals_per_chunk: int = 400
    evaluation_mode: str = "checkpoints"
    training_monitor_mode: str = "phased"
    eval_hero_seat: int = 0
    checkpoint_interval: int = 100000
    averaging_window_traversals: int = 512
    utility_averaging_window_hands: int = 10000
    current_iteration: int = 0
    advantage_batch_size: int = 128
    strategy_batch_size: int = 128
    parallel_traversal_enabled: bool = False
    parallel_traversal_workers: int = 1
    parallel_traversal_min_traversals: int = 200
    phase1_iterations: int = 1000
    phase1_traversals_per_iteration: int = 200
    phase2_iterations: int = 2000
    phase2_traversals_per_iteration: int = 400
    phase3_iterations: int = 10000
    phase3_traversals_per_iteration: int = 400


@dataclass
class TrainingSnapshot:
    status: str
    traversals_completed: int
    traverser_decisions: int
    exploration_epsilon: float
    advantage_buffer_size: int
    strategy_buffer_size: int
    postflop_value_buffer_size: int
    regret_loss: float
    strategy_loss: float
    postflop_value_loss: float
    ema_regret_loss: float
    ema_strategy_loss: float
    ema_postflop_value_loss: float
    avg_utility_bb: float
    vpip: float
    pfr: float
    three_bet: float
    preflop_jam_rate: float
    flop_seen_rate: float
    avg_actions_per_hand: float
    avg_preflop_actions_per_hand: float
    blueprint_coverage_pct: float
    utility_window_count: int
    style_window_count: int
    position_window_size: int
    action_entropy: float
    invalid_state_count: int
    invalid_action_count: int
    hands_per_second: float
    learner_steps: int
    chunk_learner_steps: int
    chunk_regret_steps: int
    chunk_strategy_steps: int
    chunk_postflop_value_steps: int
    chunk_advantage_samples: int
    chunk_strategy_samples: int
    chunk_postflop_value_samples: int
    postflop_samples_per_traversal: float
    checkpoint_pool_size: int
    action_histogram: List[int]
    preflop_action_histogram: List[int]
    postflop_action_histogram: List[int]
    postflop_conditioned_rates_by_street: Dict[str, Dict[str, float]]
    postflop_conditioned_counts_by_street: Dict[str, Dict[str, Dict[str, int]]]
    position_avg_utility_bb: Dict[str, float]
    perf_breakdown_ms: Dict[str, float]
    infoset_count: int
    pruning_active: bool
    discount_active: bool
    last_discount_factor: float
    algorithm_name: str
    monitor_mode: str
    timestamp: float


@dataclass
class EvaluationReport:
    mode: str
    hands: int
    avg_profit_bb: float
    win_rate: float
    vpip: float
    rfi: float
    pfr: float
    three_bet: float
    preflop_jam_rate: float
    flop_seen_rate: float
    avg_actions_per_hand: float
    avg_preflop_actions_per_hand: float
    blueprint_coverage_pct: float
    illegal_action_count: int
    runtime_seconds: float
    action_histogram: List[int]
    preflop_action_histogram: List[int]
    postflop_action_histogram: List[int]
    postflop_conditioned_rates_by_street: Dict[str, Dict[str, float]]
    postflop_conditioned_counts_by_street: Dict[str, Dict[str, Dict[str, int]]]
    position_avg_profit_bb: Dict[str, float]
    vpip_by_position: Dict[str, float]
    rfi_by_position: Dict[str, float]
    pfr_by_position: Dict[str, float]
    three_bet_by_position: Dict[str, float]
    vpip_hand_grid: List[List[float]]
    rfi_hand_grid: List[List[float]]
    pfr_hand_grid: List[List[float]]
    three_bet_hand_grid: List[List[float]]
    vpip_hand_grid_by_position: Dict[str, List[List[float]]]
    rfi_hand_grid_by_position: Dict[str, List[List[float]]]
    pfr_hand_grid_by_position: Dict[str, List[List[float]]]
    three_bet_hand_grid_by_position: Dict[str, List[List[float]]]
    postflop_rates: Dict[str, float]
    postflop_counts: Dict[str, int]
    postflop_rates_by_position: Dict[str, Dict[str, float]]
    postflop_counts_by_position: Dict[str, Dict[str, int]]
    postflop_hand_grids: Dict[str, List[List[float]]]
    postflop_hand_grids_by_position: Dict[str, Dict[str, List[List[float]]]]
    postflop_profit_by_stage: Dict[str, float]


@dataclass
class ExploitSuiteReport:
    hands_per_mode: int
    leak_reports: Dict[str, EvaluationReport]
    robust_reports: Dict[str, EvaluationReport]
    avg_leak_profit_bb: float
    avg_robust_profit_bb: float
    avg_leak_win_rate: float
    avg_robust_win_rate: float
    suite_name: str = "eval_suite"


class _PerspectiveAgentWrapperV26:
    """Match the external trainer's fixed-opponent wrapper semantics."""

    def __init__(self, agent):
        self.agent = agent
        self.player_id = getattr(agent, "player_id", 0)

    def choose_action(self, state):
        return self.agent.choose_action(state)


def _zero_grid() -> List[List[float]]:
    return [[0.0 for _ in range(13)] for _ in range(13)]


def _postflop_rates(counts: Dict[str, int]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for metric, (num_key, den_key) in POSTFLOP_RATE_COUNT_KEYS.items():
        denominator = int(counts.get(den_key, 0))
        rates[metric] = float(int(counts.get(num_key, 0)) / denominator) if denominator > 0 else 0.0
    return rates


def _new_postflop_counts() -> Dict[str, int]:
    return {key: 0 for key in POSTFLOP_COUNT_KEYS}


def _new_conditioned_rates() -> Dict[str, Dict[str, float]]:
    return {
        street: {metric: 0.0 for metric in POSTFLOP_CONDITION_METRIC_KEYS}
        for street in POSTFLOP_CONDITION_STREET_KEYS
    }


def _new_conditioned_counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        street: {metric: {"hits": 0, "opportunities": 0} for metric in POSTFLOP_CONDITION_METRIC_KEYS}
        for street in POSTFLOP_CONDITION_STREET_KEYS
    }


def _position_name_for_seat(button: int, seat: int, num_players: int = 6) -> str:
    if num_players != 6:
        return POSITION_NAMES_V26[seat % len(POSITION_NAMES_V26)]
    offset = (seat - button) % num_players
    by_offset = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}
    return by_offset.get(offset, POSITION_NAMES_V26[seat % len(POSITION_NAMES_V26)])


def _action_bucket(action_enum_value: int) -> int:
    if action_enum_value == 0:
        return 0
    if action_enum_value in (1, 2):
        return 1
    return 2


def _is_raise_action(action_enum_value: int) -> bool:
    return int(action_enum_value) == 3


def _action_entropy_from_histogram(histogram: List[int]) -> float:
    total = float(sum(histogram))
    if total <= 0:
        return 0.0
    probs = np.array(histogram, dtype=np.float64) / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _safe_mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _weighted_mean(values, weights) -> float:
    if not values or not weights:
        return 0.0
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return 0.0
    weighted_total = sum(float(value) * float(weight) for value, weight in zip(values, weights))
    return float(weighted_total / total_weight)


def _trim_weighted_window(values: deque[float], weights: deque[int], max_total_weight: int) -> None:
    max_total = int(max(1, max_total_weight))
    current_total = int(sum(weights))
    while values and weights and current_total > max_total:
        overflow = current_total - max_total
        head_weight = int(weights[0])
        if head_weight <= overflow:
            values.popleft()
            weights.popleft()
            current_total -= head_weight
            continue
        weights[0] = int(head_weight - overflow)
        current_total -= overflow


def _weighted_position_mean(position_maps: deque[Dict[str, float]], weights: deque[int], names: List[str]) -> Dict[str, float]:
    if not position_maps or not weights:
        return {name: 0.0 for name in names}
    totals = {name: 0.0 for name in names}
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return totals
    for mapping, weight in zip(position_maps, weights):
        weight_value = float(weight)
        for name in names:
            totals[name] += float(mapping.get(name, 0.0)) * weight_value
    return {name: float(totals[name] / total_weight) for name in names}


_V26_WORKER_IMPORT_CACHE: dict[str, tuple] = {}
_V26_WORKER_OPPONENT_CACHE: dict[tuple[str, int], object] = {}


def _v26_worker_imports(repo_root: str):
    cached = _V26_WORKER_IMPORT_CACHE.get(repo_root)
    if cached is not None:
        return cached
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import pokers as pkrs
    from src.agents.random_agent import RandomAgent
    from src.core.deep_cfr import DeepCFRAgent
    from src.training.train import _cfr_traverse_with_opponents

    cached = (pkrs, RandomAgent, DeepCFRAgent, _cfr_traverse_with_opponents)
    _V26_WORKER_IMPORT_CACHE[repo_root] = cached
    return cached


def _v26_build_worker_agent(repo_root: str, agent_payload: dict, *, player_id: int, num_players: int):
    _pkrs, _random_agent_cls, deep_cfr_agent_cls, _fixed_opponent_traverse = _v26_worker_imports(repo_root)
    agent = deep_cfr_agent_cls(player_id=player_id, num_players=num_players, device="cpu")
    agent.iteration_count = int(agent_payload.get("iteration", 0))
    agent.advantage_net.load_state_dict(agent_payload["advantage_net"])
    agent.strategy_net.load_state_dict(agent_payload["strategy_net"])
    agent.advantage_net.eval()
    agent.strategy_net.eval()
    agent.min_bet_size = float(agent_payload.get("min_bet_size", 0.1))
    agent.max_bet_size = float(agent_payload.get("max_bet_size", 3.0))
    return agent


def _v26_load_worker_opponent(repo_root: str, spec: Optional[dict], *, seat: int, num_players: int):
    _pkrs, random_agent_cls, deep_cfr_agent_cls, _fixed_opponent_traverse = _v26_worker_imports(repo_root)
    if spec is None:
        return None
    kind = str(spec.get("kind", "")).strip().lower()
    if kind == "random":
        return random_agent_cls(seat)
    if kind == "checkpoint":
        checkpoint_path = str(spec.get("path", "")).strip()
        cache_key = (checkpoint_path, int(seat))
        cached = _V26_WORKER_OPPONENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        opponent = deep_cfr_agent_cls(player_id=seat, num_players=num_players, device="cpu")
        opponent.load_model(checkpoint_path)
        _V26_WORKER_OPPONENT_CACHE[cache_key] = opponent
        return opponent
    return None


def _v26_collect_traversal_samples(task: dict) -> dict:
    repo_root = str(task["repo_root"])
    pkrs, random_agent_cls, deep_cfr_agent_cls, fixed_opponent_traverse = _v26_worker_imports(repo_root)
    hero_seat = int(task["hero_seat"])
    num_players = int(task["num_players"])
    learning_agent = _v26_build_worker_agent(
        repo_root,
        dict(task["agent_payload"]),
        player_id=hero_seat,
        num_players=num_players,
    )
    mode = str(task["mode"]).strip().lower()
    seeds_and_buttons = list(task["seeds_and_buttons"])

    if mode == "random":
        opponents = [random_agent_cls(seat) for seat in range(num_players)]
        for seed_value, button_pos in seeds_and_buttons:
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=int(button_pos),
                sb=float(task["small_blind"]),
                bb=float(task["big_blind"]),
                stake=float(task["stake"]),
                seed=int(seed_value),
            )
            learning_agent.cfr_traverse(state, int(task["iteration"]), opponents)
    else:
        opponent_specs = list(task.get("opponent_specs") or [])
        opponents = [None] * num_players
        wrappers = [None] * num_players
        for seat in range(num_players):
            if seat == hero_seat:
                continue
            spec = opponent_specs[seat] if seat < len(opponent_specs) else None
            opponent = _v26_load_worker_opponent(repo_root, spec, seat=seat, num_players=num_players)
            opponents[seat] = opponent
            if opponent is not None:
                wrappers[seat] = _PerspectiveAgentWrapperV26(opponent)
        for seed_value, button_pos in seeds_and_buttons:
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=int(button_pos),
                sb=float(task["small_blind"]),
                bb=float(task["big_blind"]),
                stake=float(task["stake"]),
                seed=int(seed_value),
            )
            fixed_opponent_traverse(
                learning_agent,
                state,
                int(task["iteration"]),
                wrappers,
                verbose=False,
            )

    alpha = float(getattr(learning_agent.advantage_memory, "alpha", 0.6))
    inv_alpha = 1.0 / alpha if alpha > 0 else 1.0
    raw_priorities = [
        float(priority ** inv_alpha) for priority in list(getattr(learning_agent.advantage_memory, "priorities", []))
    ]
    return {
        "advantage_samples": list(zip(list(getattr(learning_agent.advantage_memory, "buffer", [])), raw_priorities)),
        "strategy_samples": list(getattr(learning_agent, "strategy_memory", [])),
    }


class _SyntheticHandTrackerV26:
    POSITION_RANK = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}

    def __init__(self, big_blind: float):
        self.big_blind = float(max(1.0, big_blind))
        self.reset()

    def reset(self) -> None:
        self.current_street = 0
        self.preflop_raise_count = 0
        self.preflop_call_count = 0
        self.preflop_last_raiser: Optional[int] = None
        self.last_aggressor: Optional[int] = None
        self._last_signature = None

    def observe_state(self, state) -> None:
        self.current_street = int(state.stage)
        record = getattr(state, "from_action", None)
        if record is None:
            return
        signature = (
            int(record.player),
            int(record.stage),
            int(record.action.action),
            round(float(getattr(record.action, "amount", 0.0) or 0.0), 6),
            round(float(state.pot), 6),
            int(state.current_player),
        )
        if signature == self._last_signature:
            return
        action_stage = int(record.stage)
        action_id = int(record.action.action)
        actor = int(record.player)
        if action_stage == 0:
            if action_id == 3:
                self.preflop_raise_count += 1
                self.preflop_last_raiser = actor
                self.last_aggressor = actor
            elif action_id == 2:
                self.preflop_call_count += 1
        elif action_id == 3:
            self.last_aggressor = actor
        self.current_street = int(state.stage)
        self._last_signature = signature

    def is_in_position(self, actor: int, aggressor: Optional[int], button: int) -> bool:
        if aggressor is None:
            return False
        actor_name = _position_name_for_seat(button, actor)
        aggressor_name = _position_name_for_seat(button, aggressor)
        return int(self.POSITION_RANK.get(actor_name, 0)) > int(self.POSITION_RANK.get(aggressor_name, 0))


class _SyntheticLeakBotV26:
    ACTION_FOLD = 0
    ACTION_CHECK = 1
    ACTION_CALL = 2
    ACTION_RAISE_SMALL = 3
    ACTION_RAISE_MEDIUM = 4
    ACTION_RAISE_LARGE = 5
    ACTION_ALL_IN = 6
    PREFLOP_OPEN_RAISE_TO_BB = {
        ACTION_RAISE_SMALL: 2.25,
        ACTION_RAISE_MEDIUM: 2.50,
        ACTION_RAISE_LARGE: 3.00,
    }
    POSTFLOP_BET_POT_MULTIPLIERS = {
        ACTION_RAISE_SMALL: 0.33,
        ACTION_RAISE_MEDIUM: 0.66,
        ACTION_RAISE_LARGE: 1.00,
    }
    FACING_BET_RAISE_TO_MULTIPLIERS = {
        ACTION_RAISE_SMALL: 2.20,
        ACTION_RAISE_MEDIUM: 2.80,
        ACTION_RAISE_LARGE: 3.50,
    }
    RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"
    RANK_VALUE = {rank: 14 - idx for idx, rank in enumerate(RANK_ORDER_HIGH_TO_LOW)}
    ALL_RAISE_ACTIONS = (ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE, ACTION_ALL_IN)

    def __init__(self, pkrs, player_id: int, style: str, tracker: _SyntheticHandTrackerV26):
        self.pkrs = pkrs
        self.player_id = int(player_id)
        self.style = str(style or "nit").lower()
        self.tracker = tracker
        self._rng = random.Random(10_000 + self.player_id * 97 + sum(ord(ch) for ch in self.style))

    def reset_for_new_hand(self) -> None:
        self.tracker.reset()

    def _rank_text(self, card) -> str:
        return RANK_TEXT_BY_VALUE.get(int(card.rank), "2")

    def _canonical_preflop_key(self, cards) -> str:
        if len(cards) != 2:
            return "72o"
        ordered = sorted(cards, key=lambda card: self.RANK_ORDER_HIGH_TO_LOW.index(self._rank_text(card)))
        rank1 = self._rank_text(ordered[0])
        rank2 = self._rank_text(ordered[1])
        if rank1 == rank2:
            return f"{rank1}{rank2}"
        suited = int(ordered[0].suit) == int(ordered[1].suit)
        return f"{rank1}{rank2}{'s' if suited else 'o'}"

    def _combo_count_for_key(self, hand_key: str) -> int:
        if len(hand_key) == 2:
            return 6
        return 4 if hand_key.endswith("s") else 12

    def _hand_strength_score(self, hand_key: str) -> float:
        if len(hand_key) == 2:
            pair_rank = self.RANK_VALUE.get(hand_key[0], 2)
            return float(85.0 + (pair_rank * 6.0) + (3.0 if pair_rank >= 11 else 0.0) + (3.0 if pair_rank <= 6 else 0.0))
        high_rank = self.RANK_VALUE.get(hand_key[0], 2)
        low_rank = self.RANK_VALUE.get(hand_key[1], 2)
        suited = hand_key.endswith("s")
        gap = high_rank - low_rank
        connector_bonus = {1: 9.0, 2: 6.0, 3: 3.0, 4: 0.0}.get(gap, -3.0)
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

    def _preflop_strength(self, state) -> float:
        cards = list(state.players_state[self.player_id].hand)
        if len(cards) != 2:
            return 0.35
        hand_key = self._canonical_preflop_key(cards)
        keys = []
        for i, high_rank in enumerate(self.RANK_ORDER_HIGH_TO_LOW):
            for low_rank in self.RANK_ORDER_HIGH_TO_LOW[i:]:
                if high_rank == low_rank:
                    keys.append(f"{high_rank}{low_rank}")
                else:
                    keys.append(f"{high_rank}{low_rank}s")
                    keys.append(f"{high_rank}{low_rank}o")
        ordered = [
            key
            for _, key in sorted(
                ((self._hand_strength_score(key), key) for key in keys),
                reverse=True,
            )
        ]
        total_combos = float(sum(self._combo_count_for_key(key) for key in ordered))
        cumulative = 0.0
        percentile = 1.0
        for key in ordered:
            cumulative += float(self._combo_count_for_key(key))
            if key == hand_key:
                percentile = cumulative / total_combos
                break
        strength = max(0.02, 1.0 - float(percentile))
        opponent_penalty = max(0, sum(1 for player in state.players_state if player.active) - 2) * 0.03
        return float(max(0.02, min(0.98, strength - opponent_penalty)))

    def _has_flush_draw(self, cards) -> float:
        suit_counts = [0, 0, 0, 0]
        for card in cards:
            suit_counts[int(card.suit)] += 1
        return 1.0 if max(suit_counts, default=0) >= 4 else 0.0

    def _has_straight_draw(self, cards) -> float:
        if len(cards) < 4:
            return 0.0
        ranks = {int(card.rank) for card in cards}
        if 12 in ranks:
            ranks.add(-1)
        ordered = sorted(ranks)
        for idx in range(len(ordered)):
            window = ordered[idx : idx + 4]
            if len(window) == 4 and window[-1] - window[0] <= 4:
                return 1.0
        return 0.0

    def _best_hand_class(self, hole_cards, board_cards) -> int:
        cards = list(hole_cards) + list(board_cards)
        if len(hole_cards) != 2:
            return 0
        if len(board_cards) < 3:
            return 0
        rank_counts = [0] * 13
        suit_counts = [0] * 4
        unique_ranks = set()
        for card in cards:
            rank_idx = int(card.rank)
            suit_idx = int(card.suit)
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
            return 7
        if flush:
            return 6
        if straight:
            return 5
        if trips:
            return 4
        if pair_count >= 2:
            return 3
        if max(rank_counts) >= 2:
            return 2
        if self._has_flush_draw(cards) > 0.5 or self._has_straight_draw(cards) > 0.5:
            return 1
        return 0

    def _preferred_raise_action(self, legal_mask: np.ndarray, preferred_actions) -> Optional[int]:
        for action_id in preferred_actions:
            if 0 <= int(action_id) < len(legal_mask) and float(legal_mask[int(action_id)]) > 0.5:
                return int(action_id)
        return None

    def _normalize_masked_policy(self, policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        probs = np.asarray(policy, dtype=np.float32).reshape(-1)
        mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
        probs = probs * np.where(mask > 0.5, 1.0, 0.0)
        total = float(probs.sum())
        if total <= 1e-8:
            legal = np.where(mask > 0.5)[0]
            if legal.size == 0:
                return np.zeros_like(mask, dtype=np.float32)
            probs = np.zeros_like(mask, dtype=np.float32)
            probs[legal] = 1.0 / float(legal.size)
            return probs
        return (probs / total).astype(np.float32)

    def _sample_action(self, probs: np.ndarray) -> int:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        total = float(arr.sum())
        if total <= 0.0:
            return int(np.argmax(arr))
        draw = self._rng.random() * total
        running = 0.0
        for idx, value in enumerate(arr):
            running += float(value)
            if draw <= running:
                return int(idx)
        return int(np.argmax(arr))

    def _effective_stack_bb(self, state) -> float:
        actor_stack = float(state.players_state[self.player_id].stake)
        active_stacks = [float(player.stake) for player in state.players_state if player.active]
        effective_stack = min([actor_stack] + active_stacks) if active_stacks else actor_stack
        return float(effective_stack / max(1.0, self.tracker.big_blind))

    def _all_in_allowed(self, state) -> bool:
        if int(state.stage) != 0:
            return True
        return self._effective_stack_bb(state) <= 40.0 or int(self.tracker.preflop_raise_count) >= 2

    def _abstract_raise_additional(self, state, action_id: int) -> Optional[float]:
        player_state = state.players_state[self.player_id]
        current_bet = float(player_state.bet_chips)
        highest_bet = float(state.min_bet)
        call_amount = max(0.0, highest_bet - current_bet)
        remaining_after_call = max(0.0, float(player_state.stake) - call_amount)
        if remaining_after_call <= 0:
            return None
        if int(action_id) == self.ACTION_ALL_IN:
            if not self._all_in_allowed(state):
                return None
            return float(remaining_after_call)

        street = int(state.stage)
        if street == 0 and int(self.tracker.preflop_raise_count) == 0:
            base_size_bb = self.PREFLOP_OPEN_RAISE_TO_BB.get(int(action_id))
            if base_size_bb is None:
                return None
            limper_bonus = min(1.0, 0.5 * float(self.tracker.preflop_call_count))
            target_total = (base_size_bb + limper_bonus) * self.tracker.big_blind
        elif street == 0:
            in_position = self.tracker.is_in_position(self.player_id, self.tracker.preflop_last_raiser, int(state.button))
            effective_stack_bb = self._effective_stack_bb(state)
            if int(action_id) == self.ACTION_RAISE_LARGE and (
                int(self.tracker.preflop_raise_count) >= 2 or effective_stack_bb <= 40.0
            ):
                return float(remaining_after_call)
            multiplier = 3.0 if in_position else 3.5
            if int(action_id) == self.ACTION_RAISE_MEDIUM:
                multiplier += 0.5
            elif int(action_id) == self.ACTION_RAISE_LARGE:
                multiplier += 1.0
            target_total = highest_bet * multiplier
        elif call_amount <= 1e-6:
            multiplier = self.POSTFLOP_BET_POT_MULTIPLIERS.get(int(action_id))
            if multiplier is None:
                return None
            target_total = current_bet + (multiplier * max(float(state.pot), self.tracker.big_blind))
        else:
            multiplier = self.FACING_BET_RAISE_TO_MULTIPLIERS.get(int(action_id))
            if multiplier is None:
                return None
            target_total = highest_bet * multiplier

        additional_raise = max(1.0, float(target_total) - highest_bet)
        additional_raise = min(float(additional_raise), remaining_after_call)
        if int(action_id) == self.ACTION_RAISE_LARGE and street == 0 and self._effective_stack_bb(state) <= 40.0 and self._all_in_allowed(state):
            additional_raise = remaining_after_call
        return float(additional_raise) if additional_raise > 0 else None

    def _legal_mask_v25(self, state) -> np.ndarray:
        legal_mask = np.zeros(7, dtype=np.float32)
        player_state = state.players_state[self.player_id]
        facing_bet = float(max(0.0, float(state.min_bet) - float(player_state.bet_chips))) > 1e-9
        legal_actions = list(state.legal_actions)
        if facing_bet:
            if self.pkrs.ActionEnum.Fold in legal_actions:
                legal_mask[self.ACTION_FOLD] = 1.0
            if self.pkrs.ActionEnum.Call in legal_actions:
                legal_mask[self.ACTION_CALL] = 1.0
        else:
            if self.pkrs.ActionEnum.Check in legal_actions:
                legal_mask[self.ACTION_CHECK] = 1.0
        if self.pkrs.ActionEnum.Raise in legal_actions:
            seen_amounts = set()
            for action_id in self.ALL_RAISE_ACTIONS:
                additional = self._abstract_raise_additional(state, action_id)
                if additional is None:
                    continue
                signature = round(float(additional), 6)
                if signature in seen_amounts:
                    continue
                seen_amounts.add(signature)
                legal_mask[int(action_id)] = 1.0
        return legal_mask

    def _hand_summary(self, state) -> Dict[str, float]:
        hole_cards = list(state.players_state[self.player_id].hand)
        board_cards = list(state.public_cards)
        to_call_bb = float(max(0.0, float(state.min_bet) - float(state.players_state[self.player_id].bet_chips))) / self.tracker.big_blind
        pot_bb = float(state.pot) / self.tracker.big_blind
        return {
            "street": float(int(state.stage)),
            "preflop_strength": self._preflop_strength(state),
            "hand_class": float(self._best_hand_class(hole_cards, board_cards)),
            "pot_odds": float(to_call_bb / max(to_call_bb + pot_bb, 1e-6)),
            "facing_bet": 1.0 if to_call_bb > 1e-6 else 0.0,
        }

    def _safe_prior_policy(self, state, legal_mask: np.ndarray) -> np.ndarray:
        summary = self._hand_summary(state)
        pre = float(summary["preflop_strength"])
        cls = int(summary["hand_class"])
        pot_odds = float(summary["pot_odds"])
        facing = bool(summary["facing_bet"] > 0.5)
        policy = np.zeros(7, dtype=np.float32)
        small_raise = self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_SMALL,))
        medium_raise = self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_MEDIUM, self.ACTION_RAISE_SMALL))
        large_raise = self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_LARGE, self.ACTION_RAISE_MEDIUM))
        all_in_raise = self._preferred_raise_action(legal_mask, (self.ACTION_ALL_IN,))
        if int(summary["street"]) == 0:
            if facing:
                if pre >= 0.88:
                    if all_in_raise is not None:
                        policy[int(all_in_raise)] += 0.25
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.30
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.45
                elif pre >= 0.72:
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.22
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.18
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.60
                elif pre >= 0.48 or pot_odds <= 0.24:
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.68
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.32
                else:
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.85
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.15
            else:
                if pre >= 0.82 and medium_raise is not None:
                    policy[int(medium_raise)] += 0.52
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.22
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.14
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.12
                elif pre >= 0.62 and medium_raise is not None:
                    policy[int(medium_raise)] += 0.48
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.20
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.32
                elif pre >= 0.56 and small_raise is not None:
                    policy[int(small_raise)] += 0.64
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.08
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.32
                else:
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.92
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.08
        else:
            if facing:
                if cls >= 6:
                    if all_in_raise is not None:
                        policy[int(all_in_raise)] += 0.18
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.30
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.12
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.40
                elif cls >= 4:
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.14
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.12
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.58
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.16
                elif cls >= 3:
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.08
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.06
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.61
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.25
                elif cls == 1 and pot_odds <= 0.26:
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.60
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.40
                else:
                    if float(legal_mask[self.ACTION_FOLD]) > 0.5:
                        policy[self.ACTION_FOLD] += 0.80
                    if float(legal_mask[self.ACTION_CALL]) > 0.5:
                        policy[self.ACTION_CALL] += 0.20
            else:
                if cls >= 6:
                    if all_in_raise is not None:
                        policy[int(all_in_raise)] += 0.08
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.34
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.18
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.12
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.28
                elif cls >= 4:
                    if large_raise is not None:
                        policy[int(large_raise)] += 0.16
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.20
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.08
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.56
                elif cls >= 2:
                    if medium_raise is not None:
                        policy[int(medium_raise)] += 0.10
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.14
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.76
                else:
                    if float(legal_mask[self.ACTION_CHECK]) > 0.5:
                        policy[self.ACTION_CHECK] += 0.92
                    if small_raise is not None:
                        policy[int(small_raise)] += 0.08
        return self._normalize_masked_policy(policy, legal_mask)

    def _synthetic_policy(self, state, legal_mask: np.ndarray) -> np.ndarray:
        policy = self._safe_prior_policy(state, legal_mask)
        summary = self._hand_summary(state)
        if self.style == "maniac":
            for action_id, boost in (
                (self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_SMALL,)), 0.18),
                (self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_MEDIUM,)), 0.22),
                (self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_LARGE,)), 0.18),
                (self._preferred_raise_action(legal_mask, (self.ACTION_ALL_IN,)), 0.10),
            ):
                if action_id is not None:
                    policy[int(action_id)] += boost
            policy[self.ACTION_FOLD] *= 0.35
        elif self.style == "overfolder" and summary["facing_bet"] > 0.5:
            policy[self.ACTION_FOLD] += 0.35
            policy[self.ACTION_CALL] *= 0.50
        elif self.style == "station" and summary["facing_bet"] > 0.5:
            policy[self.ACTION_CALL] += 0.25
            policy[self.ACTION_FOLD] *= 0.55
        elif self.style == "over3better" and int(summary["street"]) == 0:
            re_raise = self._preferred_raise_action(legal_mask, (self.ACTION_RAISE_MEDIUM, self.ACTION_RAISE_LARGE))
            if re_raise is not None:
                policy[int(re_raise)] += 0.22
        elif self.style == "overcaller" and summary["facing_bet"] > 0.5:
            policy[self.ACTION_CALL] += 0.20
        return self._normalize_masked_policy(policy, legal_mask)

    def _check_call_action(self, state):
        if self.pkrs.ActionEnum.Check in state.legal_actions:
            return self.pkrs.Action(self.pkrs.ActionEnum.Check)
        if self.pkrs.ActionEnum.Call in state.legal_actions:
            return self.pkrs.Action(self.pkrs.ActionEnum.Call)
        if self.pkrs.ActionEnum.Fold in state.legal_actions:
            return self.pkrs.Action(self.pkrs.ActionEnum.Fold)
        return self.pkrs.Action(next(iter(state.legal_actions)))

    def _to_real_action(self, state, abstract_action: int):
        if int(abstract_action) == self.ACTION_FOLD:
            if self.pkrs.ActionEnum.Fold in state.legal_actions:
                return self.pkrs.Action(self.pkrs.ActionEnum.Fold)
            return self._check_call_action(state)
        if int(abstract_action) in (self.ACTION_CHECK, self.ACTION_CALL):
            return self._check_call_action(state)
        if self.pkrs.ActionEnum.Raise not in state.legal_actions:
            return self._check_call_action(state)
        additional = self._abstract_raise_additional(state, int(abstract_action))
        if additional is None:
            return self._check_call_action(state)
        player_state = state.players_state[self.player_id]
        call_amount = max(0.0, float(state.min_bet) - float(player_state.bet_chips))
        remaining_after_call = max(0.0, float(player_state.stake) - call_amount)
        if additional >= remaining_after_call - RAISE_EPSILON_V26 and remaining_after_call - RAISE_EPSILON_V26 > 0:
            additional = remaining_after_call - RAISE_EPSILON_V26
        action = self.pkrs.Action(self.pkrs.ActionEnum.Raise, float(additional))
        test_state = state.apply_action(action)
        if test_state.status == self.pkrs.StateStatus.HighBet:
            safer_amount = max(0.01, min(remaining_after_call - RAISE_EPSILON_V26, float(additional) - RAISE_EPSILON_V26))
            if safer_amount > 0 and safer_amount != float(additional):
                safer_action = self.pkrs.Action(self.pkrs.ActionEnum.Raise, float(safer_amount))
                safer_state = state.apply_action(safer_action)
                if safer_state.status == self.pkrs.StateStatus.Ok:
                    return safer_action
            return self._check_call_action(state)
        return action

    def choose_action(self, state):
        self.tracker.observe_state(state)
        legal_mask = self._legal_mask_v25(state)
        if float(legal_mask.sum()) <= 0.0:
            return self._check_call_action(state)
        policy = self._synthetic_policy(state, legal_mask)
        abstract_action = self._sample_action(policy)
        return self._to_real_action(state, abstract_action)


class DeepCFRTrainerV26GUI(DeepCFRTrainerV26):
    """Adapter that lets the old Tk dashboard drive the external DeepCFR repo."""

    def __init__(self, config: Optional[DeepCFRGuiConfigV26] = None):
        super().__init__(config or DeepCFRGuiConfigV26())
        self.config: DeepCFRGuiConfigV26
        self._imports_loaded = False
        self._pkrs = None
        self._deep_cfr_agent_cls = None
        self._random_agent_cls = None
        self._cfr_traverse_with_opponents = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._parallel_disabled_reason: str = ""

        self._training_agent = None
        self._evaluation_checkpoint_path = self.flagship_checkpoint if self.flagship_checkpoint.exists() else None
        self._current_model_path: Optional[Path] = self._evaluation_checkpoint_path
        self._last_checkpoint_path: Optional[Path] = None
        self._training_run_dir: Optional[Path] = None
        self._training_opponents: Optional[List[object]] = None
        self._training_opponent_wrappers: Optional[List[object]] = None
        self._training_opponent_specs: Optional[List[Optional[dict]]] = None
        self._training_mode_state: Optional[str] = None
        self._training_reference_dir: Optional[Path] = None
        self._training_reference_checkpoint: Optional[Path] = None
        self._phase_root_dir: Optional[Path] = None
        self._phase_dirs: Dict[str, Path] = {}
        self._phase_index = 0
        self._phase_iteration = 0
        self._phase_traversals_collected = 0
        self._cached_checkpoint_pool_dir: Optional[Path] = None
        self._cached_checkpoint_pool_paths: List[Path] = []
        self._cached_latest_checkpoint: Optional[Path] = None
        self._gui_auto_dir: Optional[Path] = None

        self.traversals_completed = 0
        self.traverser_decisions = 0
        self.learner_steps = 0
        self.invalid_state_count = 0
        self.invalid_action_count = 0

        self._last_advantage_loss = 0.0
        self._last_strategy_loss = 0.0
        self._ema_advantage_loss = 0.0
        self._ema_strategy_loss = 0.0
        self._last_chunk_learner_steps = 0
        self._last_chunk_regret_steps = 0
        self._last_chunk_strategy_steps = 0
        self._last_perf_breakdown_ms = {
            "total_time": 0.0,
            "sim_time": 0.0,
            "nn_time": 0.0,
            "mc_equity_time": 0.0,
            "overhead_time": 0.0,
        }
        self._recent_profit_bb: deque[float] = deque()
        self._recent_profit_hand_counts: deque[int] = deque()
        self._recent_vpip: deque[float] = deque()
        self._recent_pfr: deque[float] = deque()
        self._recent_three_bet: deque[float] = deque()
        self._recent_prejam: deque[float] = deque()
        self._recent_flop_seen: deque[float] = deque()
        self._recent_actions_per_hand: deque[float] = deque()
        self._recent_preflop_actions_per_hand: deque[float] = deque()
        self._recent_action_histograms: deque[np.ndarray] = deque()
        self._recent_preflop_action_histograms: deque[np.ndarray] = deque()
        self._recent_postflop_action_histograms: deque[np.ndarray] = deque()
        self._recent_position_profit_maps: deque[Dict[str, float]] = deque()
        self._recent_speed_hands: deque[int] = deque()
        self._recent_speed_seconds: deque[float] = deque()
        self._position_profit_windows = {
            name: deque(maxlen=self.config.averaging_window_traversals)
            for name in POSITION_NAMES_V26
        }
        self._last_monitor_report: Optional[EvaluationReport] = None
        self._snapshot = self._build_snapshot("Idle")

    def _ensure_repo_imports(self) -> None:
        if self._imports_loaded:
            return
        self._ensure_repo_available()
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        import pokers as pkrs
        from src.agents.random_agent import RandomAgent
        from src.core.deep_cfr import DeepCFRAgent
        from src.training.train import _cfr_traverse_with_opponents

        self._pkrs = pkrs
        self._random_agent_cls = RandomAgent
        self._deep_cfr_agent_cls = DeepCFRAgent
        self._cfr_traverse_with_opponents = _cfr_traverse_with_opponents
        self._imports_loaded = True

    @staticmethod
    def _device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _new_agent(self, player_id: int = 0):
        self._ensure_repo_imports()
        return self._deep_cfr_agent_cls(
            player_id=player_id,
            num_players=self.config.num_players,
            device=self._device(),
        )

    def _load_agent_from_checkpoint(self, checkpoint_path: Path, *, player_id: int) -> object:
        agent = self._new_agent(player_id=player_id)
        agent.load_model(str(checkpoint_path))
        return agent

    def _serialize_training_agent(self) -> Optional[dict]:
        if self._training_agent is None or bool(getattr(self._training_agent, "_legacy_mode", False)):
            return None
        return {
            "iteration": int(getattr(self._training_agent, "iteration_count", 0)),
            "advantage_net": self._training_agent.advantage_net.state_dict(),
            "strategy_net": self._training_agent.strategy_net.state_dict(),
            "min_bet_size": float(getattr(self._training_agent, "min_bet_size", 0.1)),
            "max_bet_size": float(getattr(self._training_agent, "max_bet_size", 3.0)),
            "advantage_buffer": list(getattr(self._training_agent.advantage_memory, "buffer", [])),
            "advantage_priorities": list(getattr(self._training_agent.advantage_memory, "priorities", [])),
            "advantage_position": int(getattr(self._training_agent.advantage_memory, "position", 0)),
            "advantage_max_priority": float(getattr(self._training_agent.advantage_memory, "_max_priority", 1.0)),
            "strategy_memory": list(getattr(self._training_agent, "strategy_memory", [])),
        }

    def _hydrate_training_agent(self, payload: dict) -> object:
        agent = self._new_agent(player_id=self.config.eval_hero_seat)
        agent.iteration_count = int(payload.get("iteration", 0))
        agent.advantage_net.load_state_dict(payload["advantage_net"])
        agent.strategy_net.load_state_dict(payload["strategy_net"])
        agent.advantage_net.eval()
        agent.strategy_net.eval()
        agent.min_bet_size = float(payload.get("min_bet_size", 0.1))
        agent.max_bet_size = float(payload.get("max_bet_size", 3.0))
        advantage_buffer = list(payload.get("advantage_buffer", []))
        advantage_priorities = list(payload.get("advantage_priorities", []))
        if len(advantage_buffer) == len(advantage_priorities):
            agent.advantage_memory.buffer = list(advantage_buffer)
            agent.advantage_memory.priorities = list(advantage_priorities)
            agent.advantage_memory.position = int(payload.get("advantage_position", len(advantage_buffer))) % max(1, agent.advantage_memory.capacity)
            agent.advantage_memory._max_priority = float(payload.get("advantage_max_priority", 1.0))
        strategy_samples = list(payload.get("strategy_memory", []))
        agent.strategy_memory.clear()
        agent.strategy_memory.extend(strategy_samples)
        return agent

    def _restore_recent_windows(self, payload: dict) -> None:
        recent = payload if isinstance(payload, dict) else {}
        self._recent_profit_bb.clear()
        self._recent_profit_bb.extend(float(value) for value in list(recent.get("profit_bb", [])))
        self._recent_profit_hand_counts.clear()
        self._recent_profit_hand_counts.extend(int(value) for value in list(recent.get("profit_hands", [])))
        for attr_name, key in (
            ("_recent_vpip", "vpip"),
            ("_recent_pfr", "pfr"),
            ("_recent_three_bet", "three_bet"),
            ("_recent_prejam", "prejam"),
            ("_recent_flop_seen", "flop_seen"),
            ("_recent_actions_per_hand", "actions_per_hand"),
            ("_recent_preflop_actions_per_hand", "preflop_actions_per_hand"),
        ):
            series = getattr(self, attr_name)
            series.clear()
            series.extend(float(value) for value in list(recent.get(key, [])))
        for attr_name, key in (
            ("_recent_action_histograms", "action_histograms"),
            ("_recent_preflop_action_histograms", "preflop_action_histograms"),
            ("_recent_postflop_action_histograms", "postflop_action_histograms"),
        ):
            series = getattr(self, attr_name)
            series.clear()
            series.extend(np.asarray(value, dtype=np.float64) for value in list(recent.get(key, [])))
        missing_monitor_entries = max(0, len(self._recent_profit_hand_counts) - len(self._recent_action_histograms))
        for _ in range(missing_monitor_entries):
            self._recent_action_histograms.append(np.zeros(ACTION_COUNT_V26, dtype=np.float64))
            self._recent_preflop_action_histograms.append(np.zeros(ACTION_COUNT_V26, dtype=np.float64))
            self._recent_postflop_action_histograms.append(np.zeros(ACTION_COUNT_V26, dtype=np.float64))
        self._recent_position_profit_maps.clear()
        self._recent_position_profit_maps.extend(
            {name: float((mapping or {}).get(name, 0.0)) for name in POSITION_NAMES_V26}
            for mapping in list(recent.get("position_profit_maps", []))
        )
        while len(self._recent_position_profit_maps) < len(self._recent_profit_hand_counts):
            self._recent_position_profit_maps.append({name: 0.0 for name in POSITION_NAMES_V26})
        raw_positions = recent.get("position_profit_windows", {})
        if isinstance(raw_positions, dict):
            for name, series in self._position_profit_windows.items():
                series.clear()
                series.extend(float(value) for value in list(raw_positions.get(name, [])))
        self._recent_speed_hands.clear()
        self._recent_speed_hands.extend(int(value) for value in list(recent.get("speed_hands", [])))
        self._recent_speed_seconds.clear()
        self._recent_speed_seconds.extend(float(value) for value in list(recent.get("speed_seconds", [])))
        self._trim_monitor_windows()
        self._trim_speed_window()

    def _checkpoint_pool_paths(self) -> List[Path]:
        models_dir = self.preferred_models_dir()
        return self._checkpoint_pool_paths_from_dir(models_dir)

    @staticmethod
    def _is_gui_checkpoint_payload(payload: object) -> bool:
        return isinstance(payload, dict) and str(payload.get("format_version", "")) == GUI_CHECKPOINT_FORMAT

    def _is_external_model_checkpoint(self, path: Path) -> bool:
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            return False
        if self._is_gui_checkpoint_payload(payload):
            return False
        return isinstance(payload, dict) and "advantage_net" in payload and "strategy_net" in payload

    def _latest_external_checkpoint(self) -> Optional[Path]:
        return self._cached_latest_checkpoint

    def latest_checkpoint(self) -> Optional[Path]:
        return self._latest_external_checkpoint()

    def latest_models_dir(self) -> Optional[Path]:
        checkpoint = self.latest_checkpoint()
        return checkpoint.parent if checkpoint is not None else None

    def preferred_checkpoint(self) -> Optional[Path]:
        latest = self.latest_checkpoint()
        if latest is not None:
            return latest
        if self.flagship_checkpoint.exists():
            return self.flagship_checkpoint
        return None

    def preferred_models_dir(self) -> Optional[Path]:
        latest = self.latest_models_dir()
        if latest is not None:
            return latest
        if self.flagship_models_dir.exists():
            return self.flagship_models_dir
        return None

    def _active_training_mode(self) -> str:
        mode = str(self.config.training_monitor_mode or "").strip().lower()
        if mode in {"random", "self_play", "mixed", "phased"}:
            return mode
        return "phased"

    def _active_phase_spec(self) -> Optional[dict]:
        phased_plan = self._phased_training_plan()
        while self._phase_index < len(phased_plan) and int(phased_plan[self._phase_index]["iterations"]) <= 0:
            self._phase_index += 1
        if self._phase_index >= len(phased_plan):
            return None
        return phased_plan[self._phase_index]

    def _active_phase_mode(self) -> str:
        spec = self._active_phase_spec()
        return str(spec["mode"]) if spec is not None else "mixed"

    def _active_monitor_mode(self) -> str:
        training_mode = self._active_training_mode()
        return self._active_phase_mode() if training_mode == "phased" else training_mode

    def planned_total_traversals(self) -> int:
        return int(
            sum(
                max(0, int(spec["iterations"])) * max(1, int(spec["traversals_per_iteration"]))
                for spec in self._phased_training_plan()
            )
        )

    def _phased_training_plan(self) -> tuple[dict, ...]:
        return (
            {
                "name": "phase1",
                "mode": "random",
                "iterations": max(0, int(self.config.phase1_iterations)),
                "traversals_per_iteration": max(1, int(self.config.phase1_traversals_per_iteration)),
            },
            {
                "name": "phase2",
                "mode": "self_play",
                "iterations": max(0, int(self.config.phase2_iterations)),
                "traversals_per_iteration": max(1, int(self.config.phase2_traversals_per_iteration)),
            },
            {
                "name": "phase3",
                "mode": "mixed",
                "iterations": max(0, int(self.config.phase3_iterations)),
                "traversals_per_iteration": max(1, int(self.config.phase3_traversals_per_iteration)),
            },
        )

    def _reset_training_context(self) -> None:
        self._training_opponents = None
        self._training_opponent_wrappers = None
        self._training_opponent_specs = None
        self._training_mode_state = None
        self._training_reference_dir = None
        self._training_reference_checkpoint = None

    def _invalidate_checkpoint_cache(self) -> None:
        self._cached_checkpoint_pool_dir = None
        self._cached_checkpoint_pool_paths = []
        self._cached_latest_checkpoint = None

    def _reset_phase_state(self) -> None:
        self._phase_root_dir = None
        self._phase_dirs = {}
        self._phase_index = 0
        self._phase_iteration = 0
        self._phase_traversals_collected = 0
        self._gui_auto_dir = None

    def _trim_monitor_windows(self) -> None:
        max_hands = int(self.config.utility_averaging_window_hands)
        current_total = int(sum(self._recent_profit_hand_counts))
        while self._recent_profit_hand_counts and current_total > max_hands:
            overflow = current_total - max_hands
            head_weight = int(self._recent_profit_hand_counts[0])
            if head_weight <= overflow:
                self._recent_profit_hand_counts.popleft()
                self._recent_profit_bb.popleft()
                self._recent_vpip.popleft()
                self._recent_pfr.popleft()
                self._recent_three_bet.popleft()
                self._recent_prejam.popleft()
                self._recent_flop_seen.popleft()
                self._recent_actions_per_hand.popleft()
                self._recent_preflop_actions_per_hand.popleft()
                self._recent_action_histograms.popleft()
                self._recent_preflop_action_histograms.popleft()
                self._recent_postflop_action_histograms.popleft()
                self._recent_position_profit_maps.popleft()
                current_total -= head_weight
                continue
            keep_fraction = float(head_weight - overflow) / float(head_weight)
            self._recent_profit_hand_counts[0] = int(head_weight - overflow)
            self._recent_action_histograms[0] = self._recent_action_histograms[0] * keep_fraction
            self._recent_preflop_action_histograms[0] = self._recent_preflop_action_histograms[0] * keep_fraction
            self._recent_postflop_action_histograms[0] = self._recent_postflop_action_histograms[0] * keep_fraction
            current_total -= overflow
            break

    def _trim_speed_window(self) -> None:
        max_hands = int(self.config.utility_averaging_window_hands)
        current_total = int(sum(self._recent_speed_hands))
        while self._recent_speed_hands and current_total > max_hands:
            overflow = current_total - max_hands
            head_hands = int(self._recent_speed_hands[0])
            head_seconds = float(self._recent_speed_seconds[0])
            if head_hands <= overflow:
                self._recent_speed_hands.popleft()
                self._recent_speed_seconds.popleft()
                current_total -= head_hands
                continue
            keep_fraction = float(head_hands - overflow) / float(head_hands)
            self._recent_speed_hands[0] = int(head_hands - overflow)
            self._recent_speed_seconds[0] = float(head_seconds * keep_fraction)
            current_total -= overflow
            break

    def _ensure_phase_root_dir(self) -> Path:
        if self._phase_root_dir is None:
            self._phase_root_dir = self.create_run_dir(f"{self.config.nickname}_phased")
        return self._phase_root_dir

    def _phase_dir(self, phase_name: str) -> Path:
        existing = self._phase_dirs.get(phase_name)
        if existing is not None:
            return existing
        root = self._ensure_phase_root_dir()
        phase_dir = root / phase_name
        (phase_dir / "models").mkdir(parents=True, exist_ok=True)
        (phase_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._phase_dirs[phase_name] = phase_dir
        return phase_dir

    def _phase_models_dir(self, phase_name: str) -> Path:
        return self._phase_dir(phase_name) / "models"

    def _phase_final_checkpoint(self, phase_name: str) -> Optional[Path]:
        spec = next((item for item in self._phased_training_plan() if item["name"] == phase_name), None)
        if spec is None:
            return None
        checkpoint_path = self._phase_models_dir(phase_name) / self._raw_checkpoint_name(str(spec["mode"]), int(spec["iterations"]))
        return checkpoint_path if checkpoint_path.exists() else None

    def _parallel_worker_count(self, traversals_to_run: int) -> int:
        if not bool(self.config.parallel_traversal_enabled):
            return 0
        if traversals_to_run < max(1, int(self.config.parallel_traversal_min_traversals)):
            return 0
        if self._training_agent is None or bool(getattr(self._training_agent, "_legacy_mode", False)):
            return 0
        configured = max(0, int(self.config.parallel_traversal_workers))
        if configured <= 1:
            return 0
        return min(configured, max(1, int(traversals_to_run)))

    def _ensure_process_pool(self, worker_count: int) -> ProcessPoolExecutor:
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=int(worker_count))
        return self._process_pool

    def _shutdown_process_pool(self) -> None:
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=False, cancel_futures=True)
            self._process_pool = None

    def _build_opponents_from_specs(self, specs: List[Optional[dict]]) -> List[object]:
        opponents: List[object] = [None] * self.config.num_players
        for seat in range(self.config.num_players):
            if seat == self.config.eval_hero_seat:
                continue
            spec = specs[seat] if seat < len(specs) else None
            if spec is None:
                continue
            kind = str(spec.get("kind", "")).strip().lower()
            if kind == "random":
                opponents[seat] = self._random_agent_cls(seat)
            elif kind == "checkpoint":
                checkpoint_path = Path(str(spec.get("path", ""))).resolve()
                opponents[seat] = self._load_agent_from_checkpoint(checkpoint_path, player_id=seat)
        return opponents

    def _wrap_training_opponents(self, opponent_agents: List[object]) -> List[object]:
        hero_seat = int(self.config.eval_hero_seat)
        wrapped: List[object] = [None] * self.config.num_players
        for seat in range(self.config.num_players):
            if seat == hero_seat or opponent_agents[seat] is None:
                continue
            wrapped[seat] = _PerspectiveAgentWrapperV26(opponent_agents[seat])
        return wrapped

    def _checkpoint_for_self_play(self) -> Optional[Path]:
        current = self._current_model_path
        if current is not None and current.exists() and self._is_external_model_checkpoint(current):
            return current
        if self._evaluation_checkpoint_path is not None and self._evaluation_checkpoint_path.exists():
            return self._evaluation_checkpoint_path
        checkpoint = self.preferred_checkpoint()
        if checkpoint is not None and checkpoint.exists():
            return checkpoint
        return None

    def _checkpoint_pool_paths_from_dir(self, models_dir: Optional[Path]) -> List[Path]:
        if models_dir is None or not models_dir.exists():
            return []
        resolved_dir = models_dir.resolve()
        if self._cached_checkpoint_pool_dir == resolved_dir:
            return list(self._cached_checkpoint_pool_paths)
        paths = sorted(path for path in resolved_dir.glob("*.pt") if path.is_file() and self._is_external_model_checkpoint(path))
        self._cached_checkpoint_pool_dir = resolved_dir
        self._cached_checkpoint_pool_paths = list(paths)
        return list(paths)

    def _select_mixed_training_opponent_specs(self, checkpoint_dir: Optional[Path] = None) -> List[Optional[dict]]:
        self._ensure_repo_imports()
        checkpoint_paths = self._checkpoint_pool_paths_from_dir(checkpoint_dir or self.preferred_models_dir())
        if not checkpoint_paths:
            return [None if seat == self.config.eval_hero_seat else {"kind": "random"} for seat in range(self.config.num_players)]

        selected_paths = random.sample(
            checkpoint_paths,
            min(max(1, self.config.num_players - 1), len(checkpoint_paths)),
        )
        opponent_specs: List[Optional[dict]] = [None] * self.config.num_players
        if self.config.num_players > 1:
            opponent_specs[1] = {"kind": "random"}
        current_pos = 1
        for checkpoint_path in selected_paths:
            if current_pos != 1:
                opponent_specs[current_pos] = {"kind": "checkpoint", "path": str(checkpoint_path.resolve())}
            current_pos += 1
            if current_pos >= self.config.num_players:
                current_pos = 1

        for seat in range(self.config.num_players):
            if seat == self.config.eval_hero_seat:
                continue
            if opponent_specs[seat] is None:
                opponent_specs[seat] = {"kind": "random"}
        return opponent_specs

    def _configure_training_mode(self, mode: str, iteration: int) -> None:
        normalized_mode = str(mode).strip().lower()
        if self._training_mode_state != normalized_mode:
            self._reset_training_context()
            self._training_mode_state = normalized_mode

        if normalized_mode == "mixed":
            refresh_interval = 1000
            if self._training_opponents is None or (iteration % refresh_interval) == 1:
                if self._training_reference_dir is None:
                    self._training_reference_dir = self.preferred_models_dir()
                self._training_opponent_specs = self._select_mixed_training_opponent_specs(self._training_reference_dir)
                self._training_opponents = self._build_opponents_from_specs(self._training_opponent_specs)
                self._training_opponent_wrappers = self._wrap_training_opponents(self._training_opponents)
            return

        if normalized_mode == "self_play":
            if self._training_opponents is not None:
                return
            if self._training_reference_checkpoint is None:
                self._training_reference_checkpoint = self._checkpoint_for_self_play()
            checkpoint_path = self._training_reference_checkpoint
            if checkpoint_path is None:
                self._training_opponent_specs = [None if seat == self.config.eval_hero_seat else {"kind": "random"} for seat in range(self.config.num_players)]
                self._training_opponents = self._build_opponents_from_specs(self._training_opponent_specs)
                self._training_opponent_wrappers = self._wrap_training_opponents(self._training_opponents)
                return
            self._training_opponent_specs = [None] * self.config.num_players
            for seat in range(self.config.num_players):
                if seat == self.config.eval_hero_seat:
                    continue
                self._training_opponent_specs[seat] = {"kind": "checkpoint", "path": str(Path(checkpoint_path).resolve())}
            self._training_opponents = self._build_opponents_from_specs(self._training_opponent_specs)
            self._training_opponent_wrappers = self._wrap_training_opponents(self._training_opponents)

    def _start_phase(self, phase_index: int) -> None:
        self._phase_index = int(phase_index)
        self._phase_iteration = 0
        self._phase_traversals_collected = 0
        self._reset_training_context()
        spec = self._active_phase_spec()
        if spec is None:
            self._training_agent = None
            self._training_run_dir = None
            self._current_model_path = None
            return

        phase_name = str(spec["name"])
        phase_mode = str(spec["mode"])
        self._training_run_dir = self._phase_dir(phase_name)
        self._training_agent = self._new_agent(player_id=self.config.eval_hero_seat)
        self._training_agent.iteration_count = 0
        self._current_model_path = None
        self._training_mode_state = None

        if phase_mode == "self_play":
            self._training_reference_checkpoint = self._phase_final_checkpoint("phase1")
        elif phase_mode == "mixed":
            self._training_reference_dir = self._phase_models_dir("phase2")

    def _ensure_training_program_started(self) -> None:
        if self._active_training_mode() != "phased":
            return
        if self._training_agent is None and self._active_phase_spec() is not None:
            self._start_phase(self._phase_index)

    def _training_agent_payload_for_workers(self) -> Optional[dict]:
        if self._training_agent is None:
            return None
        if not hasattr(self._training_agent, "advantage_net") or not hasattr(self._training_agent, "strategy_net"):
            return None
        if not hasattr(self._training_agent.advantage_net, "state_dict") or not hasattr(self._training_agent.strategy_net, "state_dict"):
            return None
        payload = self._serialize_training_agent()
        if payload is None:
            return None
        return payload

    def _parallel_traversal_tasks(self, phase_mode: str, iteration: int, traversals_to_run: int, *, button_offset: int) -> List[dict]:
        worker_count = self._parallel_worker_count(traversals_to_run)
        if worker_count <= 0:
            return []
        agent_payload = self._training_agent_payload_for_workers()
        if agent_payload is None:
            return []
        base = traversals_to_run // worker_count
        remainder = traversals_to_run % worker_count
        tasks: List[dict] = []
        start_index = 0
        for worker_idx in range(worker_count):
            count = base + (1 if worker_idx < remainder else 0)
            if count <= 0:
                continue
            if phase_mode == "random":
                seed_button_pairs = [
                    (random.randint(0, 10_000), random.randint(0, self.config.num_players - 1))
                    for _ in range(count)
                ]
            else:
                seed_button_pairs = [
                    (
                        random.randint(0, 10_000),
                        (button_offset + start_index + idx) % self.config.num_players,
                    )
                    for idx in range(count)
                ]
            tasks.append(
                {
                    "repo_root": str(self.repo_root),
                    "hero_seat": int(self.config.eval_hero_seat),
                    "num_players": int(self.config.num_players),
                    "small_blind": float(self.config.small_blind),
                    "big_blind": float(self.config.big_blind),
                    "stake": float(self.config.stake),
                    "iteration": int(iteration),
                    "mode": str(phase_mode),
                    "agent_payload": agent_payload,
                    "opponent_specs": copy.deepcopy(self._training_opponent_specs) if self._training_opponent_specs is not None else None,
                    "seeds_and_buttons": seed_button_pairs,
                }
            )
            start_index += count
        return tasks

    def _merge_parallel_results(self, results: List[dict]) -> None:
        if self._training_agent is None:
            return
        for result in results:
            for experience, priority in list(result.get("advantage_samples", [])):
                self._training_agent.advantage_memory.add(experience, float(priority))
            for sample in list(result.get("strategy_samples", [])):
                self._training_agent.strategy_memory.append(sample)

    def _run_parallel_traversals(self, phase_mode: str, iteration: int, traversals_to_run: int, *, button_offset: int) -> bool:
        tasks = self._parallel_traversal_tasks(phase_mode, iteration, traversals_to_run, button_offset=button_offset)
        if not tasks:
            return False
        try:
            pool = self._ensure_process_pool(len(tasks))
            results = list(pool.map(_v26_collect_traversal_samples, tasks))
            self._merge_parallel_results(results)
            self._parallel_disabled_reason = ""
            return True
        except (BrokenProcessPool, OSError, RuntimeError, MemoryError) as exc:
            self._parallel_disabled_reason = str(exc)
            self._shutdown_process_pool()
            self.config.parallel_traversal_enabled = False
            return False

    def _complete_training_iteration(self, phase_mode: str, iteration: int) -> None:
        self._last_advantage_loss = float(
            self._training_agent.train_advantage_network(batch_size=int(self.config.advantage_batch_size))
        )
        self._last_chunk_learner_steps += 1
        self._last_chunk_regret_steps += 1
        self.learner_steps += 1
        self.config.current_iteration = self.learner_steps
        if iteration % 10 == 0:
            self._last_strategy_loss = float(
                self._training_agent.train_strategy_network(batch_size=int(self.config.strategy_batch_size))
            )
            self._last_chunk_strategy_steps += 1
        self._ema_advantage_loss = self._last_advantage_loss if self.learner_steps == 1 else (0.15 * self._last_advantage_loss + 0.85 * self._ema_advantage_loss)
        self._ema_strategy_loss = self._last_strategy_loss if self.learner_steps == 1 else (0.15 * self._last_strategy_loss + 0.85 * self._ema_strategy_loss)
        if iteration % 100 == 0:
            self._save_external_training_checkpoint(phase_mode, iteration)

    def _run_one_training_iteration(self, phase_mode: str, iteration: int, traversals_to_run: int) -> None:
        if phase_mode != "random":
            self._configure_training_mode(phase_mode, iteration)
            if self._run_parallel_traversals(
                phase_mode,
                iteration,
                traversals_to_run,
                button_offset=int(self._phase_traversals_collected),
            ):
                return

        if phase_mode == "random":
            if self._run_parallel_traversals(
                phase_mode,
                iteration,
                traversals_to_run,
                button_offset=0,
            ):
                return
            random_agents = [self._random_agent_cls(seat) for seat in range(self.config.num_players)]
            for _ in range(traversals_to_run):
                state = self._pkrs.State.from_seed(
                    n_players=self.config.num_players,
                    button=random.randint(0, self.config.num_players - 1),
                    sb=float(self.config.small_blind),
                    bb=float(self.config.big_blind),
                    stake=float(self.config.stake),
                    seed=random.randint(0, 10_000),
                )
                self._training_agent.cfr_traverse(state, iteration, random_agents)
            return

        for traverser_idx in range(traversals_to_run):
            state = self._pkrs.State.from_seed(
                n_players=self.config.num_players,
                button=(self._phase_traversals_collected + traverser_idx) % self.config.num_players,
                sb=float(self.config.small_blind),
                bb=float(self.config.big_blind),
                stake=float(self.config.stake),
                seed=random.randint(0, 10_000),
            )
            self._cfr_traverse_with_opponents(
                self._training_agent,
                state,
                iteration,
                self._training_opponent_wrappers,
                verbose=False,
            )

    def _raw_checkpoint_name(self, mode: str, iteration: int) -> str:
        if mode == "self_play":
            return f"selfplay_checkpoint_iter_{iteration}.pt"
        if mode == "mixed":
            return f"mixed_checkpoint_iter_{iteration}.pt"
        return f"checkpoint_iter_{iteration}.pt"

    def _save_external_training_checkpoint(self, mode: str, iteration: int) -> Optional[Path]:
        if self._training_agent is None or bool(getattr(self._training_agent, "_legacy_mode", False)):
            return None
        if self._training_run_dir is None:
            self._training_run_dir = self.create_run_dir(f"{self.config.nickname}_{mode}")
        checkpoint_path = self._training_run_dir / "models" / self._raw_checkpoint_name(mode, iteration)
        torch.save(
            {
                "iteration": int(iteration),
                "advantage_net": self._training_agent.advantage_net.state_dict(),
                "strategy_net": self._training_agent.strategy_net.state_dict(),
            },
            str(checkpoint_path),
        )
        self._current_model_path = checkpoint_path
        self._evaluation_checkpoint_path = checkpoint_path
        self._cached_latest_checkpoint = checkpoint_path
        self._invalidate_checkpoint_cache()
        return checkpoint_path

    def _runtime_config_overrides(self) -> dict:
        return {
            "traversals_per_chunk": int(self.config.traversals_per_chunk),
            "utility_averaging_window_hands": int(self.config.utility_averaging_window_hands),
            "advantage_batch_size": int(self.config.advantage_batch_size),
            "strategy_batch_size": int(self.config.strategy_batch_size),
            "parallel_traversal_enabled": bool(self.config.parallel_traversal_enabled),
            "parallel_traversal_workers": int(self.config.parallel_traversal_workers),
            "parallel_traversal_min_traversals": int(self.config.parallel_traversal_min_traversals),
            "phase1_iterations": int(self.config.phase1_iterations),
            "phase1_traversals_per_iteration": int(self.config.phase1_traversals_per_iteration),
            "phase2_iterations": int(self.config.phase2_iterations),
            "phase2_traversals_per_iteration": int(self.config.phase2_traversals_per_iteration),
            "phase3_iterations": int(self.config.phase3_iterations),
            "phase3_traversals_per_iteration": int(self.config.phase3_traversals_per_iteration),
        }

    def _hero_agent_for_evaluation(self):
        if self._training_agent is not None:
            return self._training_agent
        if self._evaluation_checkpoint_path is None:
            raise FileNotFoundError("No v26 checkpoint is available for evaluation.")
        return self._load_agent_from_checkpoint(self._evaluation_checkpoint_path, player_id=self.config.eval_hero_seat)

    def _clone_training_agent_for_seat(self, seat: int):
        clone = self._new_agent(player_id=seat)
        clone.iteration_count = int(getattr(self._training_agent, "iteration_count", 0))
        clone.advantage_net.load_state_dict(self._training_agent.advantage_net.state_dict())
        clone.strategy_net.load_state_dict(self._training_agent.strategy_net.state_dict())
        clone.advantage_net.eval()
        clone.strategy_net.eval()
        clone.min_bet_size = float(getattr(self._training_agent, "min_bet_size", 0.1))
        clone.max_bet_size = float(getattr(self._training_agent, "max_bet_size", 3.0))
        return clone

    def _build_opponents(self, mode: str) -> List[object]:
        self._ensure_repo_imports()
        hero_seat = int(self.config.eval_hero_seat)
        opponents: List[object] = [None] * self.config.num_players
        mode_value = str(mode or "").strip().lower()

        if mode_value in SYNTHETIC_OPPONENT_STYLES:
            tracker = _SyntheticHandTrackerV26(self.config.big_blind)
            for seat in range(self.config.num_players):
                if seat == hero_seat:
                    continue
                opponents[seat] = _SyntheticLeakBotV26(self._pkrs, seat, mode_value, tracker)
            return opponents

        if mode_value == "self_play":
            if self._training_agent is not None and not bool(getattr(self._training_agent, "_legacy_mode", False)):
                for seat in range(self.config.num_players):
                    if seat == hero_seat:
                        continue
                    opponents[seat] = self._clone_training_agent_for_seat(seat)
            elif self._evaluation_checkpoint_path is not None:
                for seat in range(self.config.num_players):
                    if seat == hero_seat:
                        continue
                    opponents[seat] = self._load_agent_from_checkpoint(self._evaluation_checkpoint_path, player_id=seat)
            else:
                for seat in range(self.config.num_players):
                    if seat == hero_seat:
                        continue
                    opponents[seat] = self._random_agent_cls(seat)
            return opponents

        if mode_value == "checkpoints":
            checkpoint_paths = self._checkpoint_pool_paths()
            if checkpoint_paths:
                idx = 0
                for seat in range(self.config.num_players):
                    if seat == hero_seat:
                        continue
                    checkpoint_path = checkpoint_paths[idx % len(checkpoint_paths)]
                    opponents[seat] = self._load_agent_from_checkpoint(checkpoint_path, player_id=seat)
                    idx += 1
                return opponents

        for seat in range(self.config.num_players):
            if seat == hero_seat:
                continue
            opponents[seat] = self._random_agent_cls(seat)
        return opponents

    @staticmethod
    def _starting_hand_key(hand) -> tuple[int, int, bool]:
        cards = sorted(hand, key=lambda card: int(card.rank), reverse=True)
        high_rank = RANK_TEXT_BY_VALUE.get(int(cards[0].rank), "?")
        low_rank = RANK_TEXT_BY_VALUE.get(int(cards[1].rank), "?")
        suited = int(cards[0].suit) == int(cards[1].suit)
        return RANK_INDEX_HIGH_TO_LOW.get(high_rank, 12), RANK_INDEX_HIGH_TO_LOW.get(low_rank, 12), suited

    @staticmethod
    def _record_grid(grid: List[List[float]], counts: List[List[int]], hand_key: tuple[int, int, bool], hit: bool) -> None:
        high_idx, low_idx, suited = hand_key
        row, col = (high_idx, low_idx) if suited else (low_idx, high_idx)
        counts[row][col] += 1
        if hit:
            grid[row][col] += 1.0

    def _evaluate_hand(self, hero_agent, opponents: List[object], seed: int) -> dict:
        self._ensure_repo_imports()
        hero_seat = int(self.config.eval_hero_seat)
        pkrs = self._pkrs
        for opponent in opponents:
            if opponent is not None and hasattr(opponent, "reset_for_new_hand"):
                opponent.reset_for_new_hand()
        state = pkrs.State.from_seed(
            n_players=self.config.num_players,
            button=seed % self.config.num_players,
            sb=float(self.config.small_blind),
            bb=float(self.config.big_blind),
            stake=float(self.config.stake),
            seed=int(seed),
        )
        initial_hand = tuple(state.players_state[hero_seat].hand)
        hand_key = self._starting_hand_key(initial_hand)
        position_name = _position_name_for_seat(int(state.button), hero_seat, self.config.num_players)

        hero_vpip = False
        hero_rfi = False
        hero_pfr = False
        hero_three_bet = False
        hero_prejam = False
        prior_preflop_raises = 0
        hero_actions = 0
        total_actions = 0
        preflop_actions = 0
        action_hist = [0] * ACTION_COUNT_V26
        pre_action_hist = [0] * ACTION_COUNT_V26
        post_action_hist = [0] * ACTION_COUNT_V26
        flop_seen = False
        turn_seen = False
        river_seen = False
        showdown_seen = False

        while not state.final_state:
            stage_value = int(state.stage)
            current_player = int(state.current_player)
            action = hero_agent.choose_action(state) if current_player == hero_seat else opponents[current_player].choose_action(state)
            bucket = _action_bucket(int(action.action))

            total_actions += 1
            if stage_value == 0:
                preflop_actions += 1
            if current_player == hero_seat:
                hero_actions += 1
                action_hist[bucket] += 1
                if stage_value == 0:
                    pre_action_hist[bucket] += 1
                    action_value = int(action.action)
                    if action_value in (2, 3):
                        hero_vpip = True
                    if action_value == 3:
                        hero_pfr = True
                        if prior_preflop_raises == 0:
                            hero_rfi = True
                        if prior_preflop_raises >= 1:
                            hero_three_bet = True
                        player_state = state.players_state[hero_seat]
                        call_amount = max(0.0, float(state.min_bet) - float(player_state.bet_chips))
                        remaining_stake = max(0.0, float(player_state.stake) - call_amount)
                        if remaining_stake > 0 and float(getattr(action, "amount", 0.0)) >= remaining_stake - 1e-9:
                            hero_prejam = True
                else:
                    post_action_hist[bucket] += 1

            if stage_value == 0 and _is_raise_action(int(action.action)):
                prior_preflop_raises += 1

            new_state = state.apply_action(action)
            if int(new_state.status) != int(pkrs.StateStatus.Ok):
                self.invalid_action_count += 1
                break

            if int(new_state.stage) >= 1 and bool(new_state.players_state[hero_seat].active):
                flop_seen = True
            if int(new_state.stage) >= 2 and bool(new_state.players_state[hero_seat].active):
                turn_seen = True
            if int(new_state.stage) >= 3 and bool(new_state.players_state[hero_seat].active):
                river_seen = True
            if int(new_state.stage) >= 4 and bool(new_state.players_state[hero_seat].active):
                showdown_seen = True
            state = new_state

        reward_bb = float(state.players_state[hero_seat].reward) / float(self.config.big_blind)
        return {
            "hand_key": hand_key,
            "position_name": position_name,
            "profit_bb": reward_bb,
            "win": reward_bb > 0.0,
            "vpip": hero_vpip,
            "rfi": hero_rfi,
            "pfr": hero_pfr,
            "three_bet": hero_three_bet,
            "prejam": hero_prejam,
            "flop_seen": flop_seen,
            "turn_seen": turn_seen,
            "river_seen": river_seen,
            "showdown_seen": showdown_seen,
            "showdown_won": showdown_seen and reward_bb > 0.0,
            "hero_actions": hero_actions,
            "total_actions": total_actions,
            "preflop_actions": preflop_actions,
            "action_histogram": action_hist,
            "preflop_action_histogram": pre_action_hist,
            "postflop_action_histogram": post_action_hist,
        }

    def _evaluate(self, mode: str, num_hands: int) -> EvaluationReport:
        start = time.time()
        hero_agent = self._hero_agent_for_evaluation()
        opponents = self._build_opponents(mode)
        hands = max(1, int(num_hands))

        total_profit = 0.0
        wins = 0
        hero_vpip = 0
        hero_rfi = 0
        hero_pfr = 0
        hero_three_bet = 0
        hero_prejam = 0
        hero_flop_seen = 0
        total_actions = 0
        total_preflop_actions = 0
        action_hist = [0] * ACTION_COUNT_V26
        pre_action_hist = [0] * ACTION_COUNT_V26
        post_action_hist = [0] * ACTION_COUNT_V26
        counts = _new_postflop_counts()
        position_profit = {name: [] for name in POSITION_NAMES_V26}
        position_hands = {name: 0 for name in POSITION_NAMES_V26}
        position_vpip = {name: 0 for name in POSITION_NAMES_V26}
        position_rfi = {name: 0 for name in POSITION_NAMES_V26}
        position_pfr = {name: 0 for name in POSITION_NAMES_V26}
        position_three_bet = {name: 0 for name in POSITION_NAMES_V26}

        vpip_grid = _zero_grid()
        rfi_grid = _zero_grid()
        pfr_grid = _zero_grid()
        three_bet_grid = _zero_grid()
        vpip_counts = [[0 for _ in range(13)] for _ in range(13)]
        rfi_counts = [[0 for _ in range(13)] for _ in range(13)]
        pfr_counts = [[0 for _ in range(13)] for _ in range(13)]
        three_bet_counts = [[0 for _ in range(13)] for _ in range(13)]
        vpip_grid_by_position = {name: _zero_grid() for name in POSITION_NAMES_V26}
        rfi_grid_by_position = {name: _zero_grid() for name in POSITION_NAMES_V26}
        pfr_grid_by_position = {name: _zero_grid() for name in POSITION_NAMES_V26}
        three_bet_grid_by_position = {name: _zero_grid() for name in POSITION_NAMES_V26}
        vpip_counts_by_position = {name: [[0 for _ in range(13)] for _ in range(13)] for name in POSITION_NAMES_V26}
        rfi_counts_by_position = {name: [[0 for _ in range(13)] for _ in range(13)] for name in POSITION_NAMES_V26}
        pfr_counts_by_position = {name: [[0 for _ in range(13)] for _ in range(13)] for name in POSITION_NAMES_V26}
        three_bet_counts_by_position = {name: [[0 for _ in range(13)] for _ in range(13)] for name in POSITION_NAMES_V26}

        for hand_index in range(hands):
            hand_stats = self._evaluate_hand(hero_agent, opponents, seed=10_000 + hand_index)
            total_profit += float(hand_stats["profit_bb"])
            wins += int(hand_stats["win"])
            hero_vpip += int(hand_stats["vpip"])
            hero_rfi += int(hand_stats["rfi"])
            hero_pfr += int(hand_stats["pfr"])
            hero_three_bet += int(hand_stats["three_bet"])
            hero_prejam += int(hand_stats["prejam"])
            hero_flop_seen += int(hand_stats["flop_seen"])
            total_actions += int(hand_stats["total_actions"])
            total_preflop_actions += int(hand_stats["preflop_actions"])
            counts["hands"] += 1
            counts["flop_seen"] += int(hand_stats["flop_seen"])
            counts["turn_seen"] += int(hand_stats["turn_seen"])
            counts["river_seen"] += int(hand_stats["river_seen"])
            counts["showdown_seen"] += int(hand_stats["showdown_seen"])
            counts["showdown_won"] += int(hand_stats["showdown_won"])
            for idx in range(ACTION_COUNT_V26):
                action_hist[idx] += int(hand_stats["action_histogram"][idx])
                pre_action_hist[idx] += int(hand_stats["preflop_action_histogram"][idx])
                post_action_hist[idx] += int(hand_stats["postflop_action_histogram"][idx])

            position_name = str(hand_stats["position_name"])
            position_hands[position_name] += 1
            position_profit[position_name].append(float(hand_stats["profit_bb"]))
            position_vpip[position_name] += int(hand_stats["vpip"])
            position_rfi[position_name] += int(hand_stats["rfi"])
            position_pfr[position_name] += int(hand_stats["pfr"])
            position_three_bet[position_name] += int(hand_stats["three_bet"])

            hand_key = hand_stats["hand_key"]
            self._record_grid(vpip_grid, vpip_counts, hand_key, bool(hand_stats["vpip"]))
            self._record_grid(rfi_grid, rfi_counts, hand_key, bool(hand_stats["rfi"]))
            self._record_grid(pfr_grid, pfr_counts, hand_key, bool(hand_stats["pfr"]))
            self._record_grid(three_bet_grid, three_bet_counts, hand_key, bool(hand_stats["three_bet"]))
            self._record_grid(vpip_grid_by_position[position_name], vpip_counts_by_position[position_name], hand_key, bool(hand_stats["vpip"]))
            self._record_grid(rfi_grid_by_position[position_name], rfi_counts_by_position[position_name], hand_key, bool(hand_stats["rfi"]))
            self._record_grid(pfr_grid_by_position[position_name], pfr_counts_by_position[position_name], hand_key, bool(hand_stats["pfr"]))
            self._record_grid(three_bet_grid_by_position[position_name], three_bet_counts_by_position[position_name], hand_key, bool(hand_stats["three_bet"]))

        def _finalize_grid(grid: List[List[float]], grid_counts: List[List[int]]) -> List[List[float]]:
            finalized = _zero_grid()
            for row in range(13):
                for col in range(13):
                    count = grid_counts[row][col]
                    finalized[row][col] = float(grid[row][col] / count) if count > 0 else 0.0
            return finalized

        vpip_grid = _finalize_grid(vpip_grid, vpip_counts)
        rfi_grid = _finalize_grid(rfi_grid, rfi_counts)
        pfr_grid = _finalize_grid(pfr_grid, pfr_counts)
        three_bet_grid = _finalize_grid(three_bet_grid, three_bet_counts)
        vpip_grid_by_position = {name: _finalize_grid(vpip_grid_by_position[name], vpip_counts_by_position[name]) for name in POSITION_NAMES_V26}
        rfi_grid_by_position = {name: _finalize_grid(rfi_grid_by_position[name], rfi_counts_by_position[name]) for name in POSITION_NAMES_V26}
        pfr_grid_by_position = {name: _finalize_grid(pfr_grid_by_position[name], pfr_counts_by_position[name]) for name in POSITION_NAMES_V26}
        three_bet_grid_by_position = {name: _finalize_grid(three_bet_grid_by_position[name], three_bet_counts_by_position[name]) for name in POSITION_NAMES_V26}

        position_avg_profit = {name: _safe_mean(values) for name, values in position_profit.items()}
        vpip_by_position = {name: float(position_vpip[name] / position_hands[name]) if position_hands[name] > 0 else 0.0 for name in POSITION_NAMES_V26}
        rfi_by_position = {name: float(position_rfi[name] / position_hands[name]) if position_hands[name] > 0 else 0.0 for name in POSITION_NAMES_V26}
        pfr_by_position = {name: float(position_pfr[name] / position_hands[name]) if position_hands[name] > 0 else 0.0 for name in POSITION_NAMES_V26}
        three_bet_by_position = {name: float(position_three_bet[name] / position_hands[name]) if position_hands[name] > 0 else 0.0 for name in POSITION_NAMES_V26}

        postflop_counts_by_position = {name: _new_postflop_counts() for name in POSITION_NAMES_V26}
        for name in POSITION_NAMES_V26:
            postflop_counts_by_position[name]["hands"] = int(position_hands[name])
        postflop_rates_by_position = {name: _postflop_rates(postflop_counts_by_position[name]) for name in POSITION_NAMES_V26}
        zero_grid = _zero_grid()
        return EvaluationReport(
            mode=str(mode),
            hands=hands,
            avg_profit_bb=float(total_profit / hands),
            win_rate=float(wins / hands),
            vpip=float(hero_vpip / hands),
            rfi=float(hero_rfi / hands),
            pfr=float(hero_pfr / hands),
            three_bet=float(hero_three_bet / hands),
            preflop_jam_rate=float(hero_prejam / hands),
            flop_seen_rate=float(hero_flop_seen / hands),
            avg_actions_per_hand=float(total_actions / hands),
            avg_preflop_actions_per_hand=float(total_preflop_actions / hands),
            blueprint_coverage_pct=1.0,
            illegal_action_count=int(self.invalid_action_count),
            runtime_seconds=float(time.time() - start),
            action_histogram=action_hist,
            preflop_action_histogram=pre_action_hist,
            postflop_action_histogram=post_action_hist,
            postflop_conditioned_rates_by_street=_new_conditioned_rates(),
            postflop_conditioned_counts_by_street=_new_conditioned_counts(),
            position_avg_profit_bb=position_avg_profit,
            vpip_by_position=vpip_by_position,
            rfi_by_position=rfi_by_position,
            pfr_by_position=pfr_by_position,
            three_bet_by_position=three_bet_by_position,
            vpip_hand_grid=vpip_grid,
            rfi_hand_grid=rfi_grid,
            pfr_hand_grid=pfr_grid,
            three_bet_hand_grid=three_bet_grid,
            vpip_hand_grid_by_position=vpip_grid_by_position,
            rfi_hand_grid_by_position=rfi_grid_by_position,
            pfr_hand_grid_by_position=pfr_grid_by_position,
            three_bet_hand_grid_by_position=three_bet_grid_by_position,
            postflop_rates=_postflop_rates(counts),
            postflop_counts={key: int(value) for key, value in counts.items()},
            postflop_rates_by_position=postflop_rates_by_position,
            postflop_counts_by_position=postflop_counts_by_position,
            postflop_hand_grids={metric: zero_grid for metric in POSTFLOP_RATE_KEYS},
            postflop_hand_grids_by_position={name: {metric: zero_grid for metric in POSTFLOP_RATE_KEYS} for name in POSITION_NAMES_V26},
            postflop_profit_by_stage={"all_hands": float(total_profit / hands), "flop_seen": 0.0, "turn_seen": 0.0, "river_seen": 0.0, "showdown_seen": 0.0},
        )

    def _build_snapshot(self, status: str) -> TrainingSnapshot:
        report = self._last_monitor_report
        report_action_hist = (
            [int(round(value)) for value in np.sum(np.stack(list(self._recent_action_histograms)), axis=0).tolist()]
            if self._recent_action_histograms
            else (report.action_histogram if report is not None else [0] * ACTION_COUNT_V26)
        )
        report_pre_hist = (
            [int(round(value)) for value in np.sum(np.stack(list(self._recent_preflop_action_histograms)), axis=0).tolist()]
            if self._recent_preflop_action_histograms
            else (report.preflop_action_histogram if report is not None else [0] * ACTION_COUNT_V26)
        )
        report_post_hist = (
            [int(round(value)) for value in np.sum(np.stack(list(self._recent_postflop_action_histograms)), axis=0).tolist()]
            if self._recent_postflop_action_histograms
            else (report.postflop_action_histogram if report is not None else [0] * ACTION_COUNT_V26)
        )
        weighted_avg_utility_bb = _weighted_mean(self._recent_profit_bb, self._recent_profit_hand_counts)
        utility_window_hands = int(sum(self._recent_profit_hand_counts))
        recent_speed_hands = int(sum(self._recent_speed_hands))
        recent_speed_seconds = float(sum(self._recent_speed_seconds))
        hands_per_second = float(recent_speed_hands / recent_speed_seconds) if recent_speed_seconds > 0.0 else 0.0
        weighted_position_utility = _weighted_position_mean(
            self._recent_position_profit_maps,
            self._recent_profit_hand_counts,
            list(POSITION_NAMES_V26),
        ) if self._recent_position_profit_maps else {name: _safe_mean(values) for name, values in self._position_profit_windows.items()}
        return TrainingSnapshot(
            status=str(status),
            traversals_completed=int(self.traversals_completed),
            traverser_decisions=int(self.traverser_decisions),
            exploration_epsilon=0.0,
            advantage_buffer_size=int(len(getattr(self._training_agent, "advantage_memory", []))) if self._training_agent is not None else 0,
            strategy_buffer_size=int(len(getattr(self._training_agent, "strategy_memory", []))) if self._training_agent is not None else 0,
            postflop_value_buffer_size=0,
            regret_loss=float(self._last_advantage_loss),
            strategy_loss=float(self._last_strategy_loss),
            postflop_value_loss=0.0,
            ema_regret_loss=float(self._ema_advantage_loss),
            ema_strategy_loss=float(self._ema_strategy_loss),
            ema_postflop_value_loss=0.0,
            avg_utility_bb=float(weighted_avg_utility_bb if utility_window_hands > 0 else (report.avg_profit_bb if report is not None else 0.0)),
            vpip=float(_weighted_mean(self._recent_vpip, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.vpip if report is not None else 0.0)),
            pfr=float(_weighted_mean(self._recent_pfr, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.pfr if report is not None else 0.0)),
            three_bet=float(_weighted_mean(self._recent_three_bet, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.three_bet if report is not None else 0.0)),
            preflop_jam_rate=float(_weighted_mean(self._recent_prejam, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.preflop_jam_rate if report is not None else 0.0)),
            flop_seen_rate=float(_weighted_mean(self._recent_flop_seen, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.flop_seen_rate if report is not None else 0.0)),
            avg_actions_per_hand=float(_weighted_mean(self._recent_actions_per_hand, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.avg_actions_per_hand if report is not None else 0.0)),
            avg_preflop_actions_per_hand=float(_weighted_mean(self._recent_preflop_actions_per_hand, self._recent_profit_hand_counts) if utility_window_hands > 0 else (report.avg_preflop_actions_per_hand if report is not None else 0.0)),
            blueprint_coverage_pct=1.0,
            utility_window_count=int(utility_window_hands),
            style_window_count=int(utility_window_hands),
            position_window_size=int(utility_window_hands),
            action_entropy=float(_action_entropy_from_histogram(report_action_hist)),
            invalid_state_count=int(self.invalid_state_count),
            invalid_action_count=int(self.invalid_action_count),
            hands_per_second=float(hands_per_second),
            learner_steps=int(self.learner_steps),
            chunk_learner_steps=int(self._last_chunk_learner_steps),
            chunk_regret_steps=int(self._last_chunk_regret_steps),
            chunk_strategy_steps=int(self._last_chunk_strategy_steps),
            chunk_postflop_value_steps=0,
            chunk_advantage_samples=int(len(getattr(self._training_agent, "advantage_memory", []))) if self._training_agent is not None else 0,
            chunk_strategy_samples=int(len(getattr(self._training_agent, "strategy_memory", []))) if self._training_agent is not None else 0,
            chunk_postflop_value_samples=0,
            postflop_samples_per_traversal=0.0,
            checkpoint_pool_size=len(self._checkpoint_pool_paths()),
            action_histogram=list(report_action_hist),
            preflop_action_histogram=list(report_pre_hist),
            postflop_action_histogram=list(report_post_hist),
            postflop_conditioned_rates_by_street=_new_conditioned_rates(),
            postflop_conditioned_counts_by_street=_new_conditioned_counts(),
            position_avg_utility_bb=weighted_position_utility,
            perf_breakdown_ms=dict(self._last_perf_breakdown_ms),
            infoset_count=int(len(getattr(self._training_agent, "advantage_memory", [])) + len(getattr(self._training_agent, "strategy_memory", []))) if self._training_agent is not None else 0,
            pruning_active=False,
            discount_active=False,
            last_discount_factor=1.0,
            algorithm_name="deep_cfr_6max_external",
            monitor_mode=str(self._active_monitor_mode()),
            timestamp=float(time.time()),
        )

    def _refresh_snapshot(self, status: str) -> TrainingSnapshot:
        self._snapshot = self._build_snapshot(status)
        return self._snapshot

    def get_snapshot(self) -> TrainingSnapshot:
        return copy.deepcopy(self._snapshot)

    def train_for_traversals(self, num_traversals: int) -> TrainingSnapshot:
        self._ensure_repo_imports()
        requested = max(1, int(num_traversals))
        training_mode = self._active_training_mode()
        start = time.time()
        start_traversals = int(self.traversals_completed)
        self._last_chunk_learner_steps = 0
        self._last_chunk_regret_steps = 0
        self._last_chunk_strategy_steps = 0

        if training_mode == "phased":
            self._ensure_training_program_started()
            remaining = requested
            while remaining > 0:
                spec = self._active_phase_spec()
                if spec is None or self._training_agent is None:
                    break
                phase_mode = str(spec["mode"])
                traversals_per_iteration = int(spec["traversals_per_iteration"])
                iteration = int(self._phase_iteration) + 1
                self._training_agent.iteration_count = iteration
                traversals_needed = max(1, traversals_per_iteration - int(self._phase_traversals_collected))
                traversals_to_run = min(remaining, traversals_needed)
                self._run_one_training_iteration(phase_mode, iteration, traversals_to_run)
                self._phase_traversals_collected += traversals_to_run
                self.traversals_completed += traversals_to_run
                self.traverser_decisions += traversals_to_run
                remaining -= traversals_to_run
                if self._phase_traversals_collected >= traversals_per_iteration:
                    self._complete_training_iteration(phase_mode, iteration)
                    self._phase_iteration += 1
                    self._phase_traversals_collected = 0
                    if self._phase_iteration >= int(spec["iterations"]):
                        self._phase_index += 1
                        if self._phase_index < len(self._phased_training_plan()):
                            self._start_phase(self._phase_index)
                        else:
                            self._training_agent = None
                            self._training_run_dir = None
                            self._current_model_path = self._phase_final_checkpoint("phase3")
                            break
        else:
            if self._training_agent is None:
                self._training_agent = self._new_agent(player_id=self.config.eval_hero_seat)
                self._current_model_path = None
                self._training_run_dir = self.create_run_dir(f"{self.config.nickname}_{training_mode}")
            iteration = int(getattr(self._training_agent, "iteration_count", 0)) + 1
            self._training_agent.iteration_count = iteration
            self._run_one_training_iteration(training_mode, iteration, requested)
            self.traversals_completed += requested
            self.traverser_decisions += requested
            self._complete_training_iteration(training_mode, iteration)

        if self._last_chunk_learner_steps > 0:
            monitor_hands = max(1, int(requested))
            monitor_mode = "heuristics"
            effective_mode = self._active_monitor_mode()
            if effective_mode == "self_play":
                monitor_mode = "self_play"
            elif effective_mode == "mixed":
                monitor_mode = "checkpoints"
            self._last_monitor_report = self._evaluate(monitor_mode, monitor_hands)
            report_hands = int(max(0, getattr(self._last_monitor_report, "hands", monitor_hands)))
            action_histogram = np.asarray(
                getattr(self._last_monitor_report, "action_histogram", [0] * ACTION_COUNT_V26),
                dtype=np.float64,
            )
            preflop_action_histogram = np.asarray(
                getattr(self._last_monitor_report, "preflop_action_histogram", [0] * ACTION_COUNT_V26),
                dtype=np.float64,
            )
            postflop_action_histogram = np.asarray(
                getattr(self._last_monitor_report, "postflop_action_histogram", [0] * ACTION_COUNT_V26),
                dtype=np.float64,
            )
            position_avg_profit_bb = getattr(self._last_monitor_report, "position_avg_profit_bb", {}) or {}
            self._recent_profit_bb.append(float(self._last_monitor_report.avg_profit_bb))
            self._recent_profit_hand_counts.append(report_hands)
            self._recent_vpip.append(float(self._last_monitor_report.vpip))
            self._recent_pfr.append(float(self._last_monitor_report.pfr))
            self._recent_three_bet.append(float(self._last_monitor_report.three_bet))
            self._recent_prejam.append(float(self._last_monitor_report.preflop_jam_rate))
            self._recent_flop_seen.append(float(self._last_monitor_report.flop_seen_rate))
            self._recent_actions_per_hand.append(float(self._last_monitor_report.avg_actions_per_hand))
            self._recent_preflop_actions_per_hand.append(float(self._last_monitor_report.avg_preflop_actions_per_hand))
            self._recent_action_histograms.append(action_histogram)
            self._recent_preflop_action_histograms.append(preflop_action_histogram)
            self._recent_postflop_action_histograms.append(postflop_action_histogram)
            self._recent_position_profit_maps.append(
                {name: float(position_avg_profit_bb.get(name, 0.0)) for name in POSITION_NAMES_V26}
            )
            self._trim_monitor_windows()
            for name, value in position_avg_profit_bb.items():
                self._position_profit_windows[name].append(float(value))

        elapsed_ms = (time.time() - start) * 1000.0
        processed_hands = int(max(0, self.traversals_completed - start_traversals))
        elapsed_seconds = float(elapsed_ms / 1000.0)
        if processed_hands > 0 and elapsed_seconds > 0.0:
            self._recent_speed_hands.append(processed_hands)
            self._recent_speed_seconds.append(elapsed_seconds)
            self._trim_speed_window()
        self._last_perf_breakdown_ms = {
            "total_time": float(elapsed_ms),
            "sim_time": float(elapsed_ms),
            "nn_time": 0.0,
            "mc_equity_time": 0.0,
            "overhead_time": 0.0,
        }
        if self.traversals_completed > 0 and self.traversals_completed % max(1, int(self.config.checkpoint_interval)) == 0:
            if self._gui_auto_dir is None:
                self._gui_auto_dir = self.create_run_dir(f"{self.config.nickname}_gui_auto")
            auto_dir = self._gui_auto_dir
            checkpoint_iteration = int(self.learner_steps)
            self._last_checkpoint_path = auto_dir / "models" / f"gui_checkpoint_iter_{checkpoint_iteration}.pt"
            self.save_checkpoint(str(self._last_checkpoint_path))
        return self._refresh_snapshot("Training")

    def evaluate_vs_heuristics(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("heuristics", num_hands)

    def evaluate_self_play(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("self_play", num_hands)

    def evaluate_network_only(self, num_hands: int) -> EvaluationReport:
        return self.evaluate_self_play(num_hands)

    def evaluate_vs_checkpoint_pool(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("checkpoints", num_hands)

    def evaluate_vs_synthetic_style(self, style: str, num_hands: int) -> EvaluationReport:
        return self._evaluate(style or "heuristics", num_hands)

    def evaluate_vs_leak_pool(self, num_hands: int) -> EvaluationReport:
        return self.evaluate_vs_synthetic_style("nit", num_hands)

    def evaluate_eval_suite(self, num_hands: int) -> ExploitSuiteReport:
        hands = max(1, int(num_hands))
        leak_reports = {
            style: self.evaluate_vs_synthetic_style(style, hands)
            for style in SYNTHETIC_OPPONENT_STYLES
        }
        robust_reports = {
            "heuristics": self.evaluate_vs_heuristics(hands),
            "self_play": self.evaluate_self_play(hands),
            "checkpoints": self.evaluate_vs_checkpoint_pool(hands),
        }
        return ExploitSuiteReport(
            hands_per_mode=hands,
            leak_reports=leak_reports,
            robust_reports=robust_reports,
            avg_leak_profit_bb=float(np.mean([report.avg_profit_bb for report in leak_reports.values()])) if leak_reports else 0.0,
            avg_robust_profit_bb=float(np.mean([report.avg_profit_bb for report in robust_reports.values()])),
            avg_leak_win_rate=float(np.mean([report.win_rate for report in leak_reports.values()])) if leak_reports else 0.0,
            avg_robust_win_rate=float(np.mean([report.win_rate for report in robust_reports.values()])),
            suite_name="eval_suite",
        )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "format_version": GUI_CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "metrics": {
                "traversals_completed": self.traversals_completed,
                "traverser_decisions": self.traverser_decisions,
                "learner_steps": self.learner_steps,
                "invalid_state_count": self.invalid_state_count,
                "invalid_action_count": self.invalid_action_count,
                "last_advantage_loss": self._last_advantage_loss,
                "last_strategy_loss": self._last_strategy_loss,
                "ema_advantage_loss": self._ema_advantage_loss,
                "ema_strategy_loss": self._ema_strategy_loss,
            },
            "evaluation_checkpoint_path": str(self._evaluation_checkpoint_path) if self._evaluation_checkpoint_path else "",
            "current_model_path": str(self._current_model_path) if self._current_model_path else "",
            "phase_state": {
                "phase_root_dir": str(self._phase_root_dir) if self._phase_root_dir else "",
                "phase_dirs": {key: str(value) for key, value in self._phase_dirs.items()},
                "phase_index": int(self._phase_index),
                "phase_iteration": int(self._phase_iteration),
                "phase_traversals_collected": int(self._phase_traversals_collected),
                "training_run_dir": str(self._training_run_dir) if self._training_run_dir else "",
                "training_reference_dir": str(self._training_reference_dir) if self._training_reference_dir else "",
                "training_reference_checkpoint": str(self._training_reference_checkpoint) if self._training_reference_checkpoint else "",
            },
            "recent_windows": {
                "profit_bb": list(self._recent_profit_bb),
                "profit_hands": list(self._recent_profit_hand_counts),
                "vpip": list(self._recent_vpip),
                "pfr": list(self._recent_pfr),
                "three_bet": list(self._recent_three_bet),
                "prejam": list(self._recent_prejam),
                "flop_seen": list(self._recent_flop_seen),
                "actions_per_hand": list(self._recent_actions_per_hand),
                "preflop_actions_per_hand": list(self._recent_preflop_actions_per_hand),
                "action_histograms": [hist.tolist() for hist in self._recent_action_histograms],
                "preflop_action_histograms": [hist.tolist() for hist in self._recent_preflop_action_histograms],
                "postflop_action_histograms": [hist.tolist() for hist in self._recent_postflop_action_histograms],
                "position_profit_maps": list(self._recent_position_profit_maps),
                "speed_hands": list(self._recent_speed_hands),
                "speed_seconds": list(self._recent_speed_seconds),
                "position_profit_windows": {name: list(values) for name, values in self._position_profit_windows.items()},
            },
            "training_agent": self._serialize_training_agent(),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        resolved = Path(path).resolve()
        payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and str(payload.get("format_version", "")) == GUI_CHECKPOINT_FORMAT:
            runtime_overrides = self._runtime_config_overrides()
            config_payload = payload.get("config", {})
            for key, value in config_payload.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            for key, value in runtime_overrides.items():
                setattr(self.config, key, value)
            metrics = payload.get("metrics", {})
            self.traversals_completed = int(metrics.get("traversals_completed", 0))
            self.traverser_decisions = int(metrics.get("traverser_decisions", 0))
            self.learner_steps = int(metrics.get("learner_steps", 0))
            self.invalid_state_count = int(metrics.get("invalid_state_count", 0))
            self.invalid_action_count = int(metrics.get("invalid_action_count", 0))
            self._last_advantage_loss = float(metrics.get("last_advantage_loss", 0.0))
            self._last_strategy_loss = float(metrics.get("last_strategy_loss", 0.0))
            self._ema_advantage_loss = float(metrics.get("ema_advantage_loss", 0.0))
            self._ema_strategy_loss = float(metrics.get("ema_strategy_loss", 0.0))
            eval_path = str(payload.get("evaluation_checkpoint_path", "")).strip()
            self._evaluation_checkpoint_path = Path(eval_path).resolve() if eval_path else None
            current_model_path = str(payload.get("current_model_path", "")).strip()
            self._current_model_path = Path(current_model_path).resolve() if current_model_path else None
            phase_state = payload.get("phase_state", {}) if isinstance(payload.get("phase_state"), dict) else {}
            phase_root_dir = str(phase_state.get("phase_root_dir", "")).strip()
            self._phase_root_dir = Path(phase_root_dir).resolve() if phase_root_dir else None
            raw_phase_dirs = phase_state.get("phase_dirs", {})
            self._phase_dirs = {
                str(key): Path(str(value)).resolve()
                for key, value in raw_phase_dirs.items()
                if str(value).strip()
            } if isinstance(raw_phase_dirs, dict) else {}
            self._phase_index = int(phase_state.get("phase_index", 0))
            self._phase_iteration = int(phase_state.get("phase_iteration", 0))
            self._phase_traversals_collected = int(phase_state.get("phase_traversals_collected", 0))
            training_run_dir = str(phase_state.get("training_run_dir", "")).strip()
            self._training_run_dir = Path(training_run_dir).resolve() if training_run_dir else None
            training_reference_dir = str(phase_state.get("training_reference_dir", "")).strip()
            self._training_reference_dir = Path(training_reference_dir).resolve() if training_reference_dir else None
            training_reference_checkpoint = str(phase_state.get("training_reference_checkpoint", "")).strip()
            self._training_reference_checkpoint = Path(training_reference_checkpoint).resolve() if training_reference_checkpoint else None
            training_payload = payload.get("training_agent")
            self._training_agent = self._hydrate_training_agent(training_payload) if isinstance(training_payload, dict) else None
            self._restore_recent_windows(payload.get("recent_windows", {}))
            self.config.current_iteration = self.learner_steps
            self._last_monitor_report = None
            self._shutdown_process_pool()
            self._invalidate_checkpoint_cache()
            self._reset_training_context()
            if self._active_training_mode() == "phased":
                if training_reference_dir:
                    self._training_reference_dir = Path(training_reference_dir).resolve()
                if training_reference_checkpoint:
                    self._training_reference_checkpoint = Path(training_reference_checkpoint).resolve()
            self._refresh_snapshot("Loaded")
            return

        agent = self._load_agent_from_checkpoint(resolved, player_id=self.config.eval_hero_seat)
        self.traversals_completed = 0
        self.traverser_decisions = 0
        self.learner_steps = int(getattr(agent, "iteration_count", 0))
        self.invalid_state_count = 0
        self.invalid_action_count = 0
        self._last_advantage_loss = 0.0
        self._last_strategy_loss = 0.0
        self._ema_advantage_loss = 0.0
        self._ema_strategy_loss = 0.0
        self.config.current_iteration = self.learner_steps
        self._current_model_path = resolved
        self._evaluation_checkpoint_path = resolved
        self._training_agent = None if bool(getattr(agent, "_legacy_mode", False)) else agent
        self._last_monitor_report = None
        self._shutdown_process_pool()
        self._invalidate_checkpoint_cache()
        self._cached_latest_checkpoint = resolved if self._is_external_model_checkpoint(resolved) else None
        self._training_run_dir = None
        self._reset_phase_state()
        self._reset_training_context()
        self._refresh_snapshot("Loaded")

    def shutdown(self) -> None:
        self._training_agent = None
        self._training_run_dir = None
        self._shutdown_process_pool()
        self._reset_phase_state()
        self._reset_training_context()
        return None


PluribusGuiTrainerV26 = DeepCFRTrainerV26GUI


__all__ = [
    "ACTION_NAMES_V26",
    "DeepCFRGuiConfigV26",
    "DeepCFRTrainerV26GUI",
    "EvaluationReport",
    "ExploitSuiteReport",
    "PluribusGuiTrainerV26",
    "SYNTHETIC_OPPONENT_STYLES",
    "TrainingSnapshot",
]
