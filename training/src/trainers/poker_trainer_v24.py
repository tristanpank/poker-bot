from __future__ import annotations

import copy
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
MODELS_DIR = os.path.join(SRC_ROOT, "models")
WORKERS_DIR = os.path.join(SRC_ROOT, "workers")
for _path in (FEATURES_DIR, MODELS_DIR, WORKERS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from poker_state_v24 import ACTION_COUNT_V21, POSITION_NAMES_V21, STATE_DIM_V21
from poker_worker_v24 import (
    SYNTHETIC_OPPONENT_STYLES,
    build_runtime_policy_config,
    freeze_policy_snapshot,
    run_policy_hand,
    run_traversal,
    run_traversal_batch_mp,
)
from tabular_policy_v24 import TabularNode, TabularPolicySnapshot, deserialize_node_store, serialize_node_store

RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"
POSTFLOP_RATE_KEYS = ("flop_seen", "turn_seen", "river_seen", "showdown_seen", "showdown_won", "cbet_flop", "fold_vs_cbet_flop", "cbet_turn", "fold_vs_cbet_turn")
POSTFLOP_COUNT_KEYS = ("hands", "flop_seen", "turn_seen", "river_seen", "showdown_seen", "showdown_won", "cbet_flop_opportunity", "cbet_flop_taken", "fold_vs_cbet_flop_opportunity", "fold_vs_cbet_flop", "cbet_turn_opportunity", "cbet_turn_taken", "fold_vs_cbet_turn_opportunity", "fold_vs_cbet_turn")
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
POSTFLOP_CONDITION_RATE_COUNT_KEYS = {
    "check_when_legal": ("check_when_legal_hits", "check_when_legal_opportunities"),
    "bet_raise_when_checked_to": ("bet_raise_when_checked_to_hits", "bet_raise_when_checked_to_opportunities"),
    "aggressive_when_checked_to": ("aggressive_when_checked_to_hits", "aggressive_when_checked_to_opportunities"),
    "fold_when_facing_bet": ("fold_when_facing_bet_hits", "fold_when_facing_bet_opportunities"),
    "call_when_facing_bet": ("call_when_facing_bet_hits", "call_when_facing_bet_opportunities"),
    "raise_when_facing_bet": ("raise_when_facing_bet_hits", "raise_when_facing_bet_opportunities"),
}


@dataclass
class DeepCFRConfig:
    num_players: int = 6
    small_blind: int = 5
    big_blind: int = 10
    state_dim: int = STATE_DIM_V21
    action_count: int = ACTION_COUNT_V21
    action_abstraction_name: str = "conservative_5a"
    algorithm_name: str = "tabular_mccfr_6max"
    traversals_per_player_per_iteration: int = 2048
    traversals_per_chunk: int = 2048
    strategy_interval_traversals: int = 1000
    training_monitor_interval_traversals: int = 32
    prune_after_iteration: int = 128
    lcfr_after_iteration: int = 128
    discount_interval_iterations: int = 2
    negative_regret_floor: float = -300_000_000.0
    checkpoint_interval: int = 2000
    max_checkpoint_pool: int = 32
    averaging_window_traversals: int = 8192
    evaluation_mode: str = "heuristics"
    training_monitor_mode: str = "self_play"
    eval_hero_seat: int = 0
    checkpoint_pool: tuple = field(default_factory=tuple)
    synthetic_opponent_style: str = ""
    current_iteration: int = 0
    parallel_rollouts: bool = True

    # Legacy v24 fields are tolerated but unused in the tabular reset.
    hidden_dim: int = 160
    device: str = os.getenv("POKER_V24_DEVICE", "cpu").strip().lower() or "cpu"
    learning_rate: float = 2e-4
    policy_learning_rate: float = 2e-4
    advantage_batch_size: int = 768
    policy_batch_size: int = 768
    advantage_capacity: int = 131_072
    strategy_capacity: int = 1_000_000
    advantage_train_steps: int = 0
    policy_train_steps: int = 0
    full_branch_depth: int = 1
    warm_start_iterations: int = 0
    training_guardrail_phaseout_traversals: int = 0
    training_teacher_guidance_start_mix: float = 0.0
    teacher_guidance_mix_current: float = 0.0
    training_prior_fallback_start_mix: float = 0.0
    prior_fallback_mix_current: float = 0.0
    training_action_epsilon_start: float = 0.0
    training_action_epsilon_current: float = 0.0
    safe_fallback_enabled: bool = False
    live_policy_stabilization_enabled: bool = False
    rollout_workers: int = max(1, max(1, (os.cpu_count() or 2) - 1))
    rollout_worker_chunk_size: int = 64
    quantize_local_actor_model: bool = False
    quantize_local_opponent_models: bool = False
    preflop_blueprint_name: str = ""


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

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


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

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ExploitSuiteReport:
    hands_per_mode: int
    leak_reports: Dict[str, EvaluationReport]
    robust_reports: Dict[str, EvaluationReport]
    avg_leak_profit_bb: float
    avg_robust_profit_bb: float
    avg_leak_win_rate: float
    avg_robust_win_rate: float
    suite_name: str = "exploit_suite"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _new_postflop_counts() -> Dict[str, int]:
    return {key: 0 for key in POSTFLOP_COUNT_KEYS}


def _new_postflop_conditioned_counts() -> Dict[str, Dict[str, int]]:
    return {
        street: {
            count_key: 0
            for hit, opp in POSTFLOP_CONDITION_RATE_COUNT_KEYS.values()
            for count_key in (hit, opp)
        }
        for street in POSTFLOP_CONDITION_STREET_KEYS
    }


def _merge_conditioned_counts(total: Dict[str, Dict[str, int]], update: Optional[Dict[str, Dict[str, int]]]) -> None:
    if not isinstance(update, dict):
        return
    for street in POSTFLOP_CONDITION_STREET_KEYS:
        for hit, opp in POSTFLOP_CONDITION_RATE_COUNT_KEYS.values():
            total[street][hit] += int(update.get(street, {}).get(hit, 0))
            total[street][opp] += int(update.get(street, {}).get(opp, 0))


def _format_conditioned_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        street: {
            metric: {"hits": int(counts.get(street, {}).get(hit, 0)), "opportunities": int(counts.get(street, {}).get(opp, 0))}
            for metric, (hit, opp) in POSTFLOP_CONDITION_RATE_COUNT_KEYS.items()
        }
        for street in POSTFLOP_CONDITION_STREET_KEYS
    }


def _conditioned_rates(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    return {
        street: {
            metric: float(counts.get(street, {}).get(hit, 0) / int(counts.get(street, {}).get(opp, 0)))
            if int(counts.get(street, {}).get(opp, 0)) > 0
            else 0.0
            for metric, (hit, opp) in POSTFLOP_CONDITION_RATE_COUNT_KEYS.items()
        }
        for street in POSTFLOP_CONDITION_STREET_KEYS
    }


def _postflop_rates(counts: Dict[str, int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, (hit, opp) in POSTFLOP_RATE_COUNT_KEYS.items():
        denom = int(counts.get(opp, 0))
        out[key] = float(counts.get(hit, 0) / denom) if denom > 0 else 0.0
    return out


def _hand_key_to_grid_indices(hand_key: Optional[str]) -> Optional[tuple[int, int]]:
    key = str(hand_key or "").strip()
    if len(key) < 2:
        return None
    rank_a = key[0]
    rank_b = key[1]
    if rank_a not in RANK_ORDER_HIGH_TO_LOW or rank_b not in RANK_ORDER_HIGH_TO_LOW:
        return None
    idx_a = RANK_ORDER_HIGH_TO_LOW.index(rank_a)
    idx_b = RANK_ORDER_HIGH_TO_LOW.index(rank_b)
    if len(key) == 2:
        return idx_a, idx_b
    if key.endswith("s"):
        return min(idx_a, idx_b), max(idx_a, idx_b)
    return max(idx_a, idx_b), min(idx_a, idx_b)


def _rate_grid_from_counts(hit_counts: np.ndarray, total_counts: np.ndarray) -> List[List[float]]:
    grid = np.zeros((13, 13), dtype=np.float32)
    nonzero = total_counts > 0
    grid[nonzero] = hit_counts[nonzero] / total_counts[nonzero]
    return grid.tolist()


class DeepCFRTrainerV24:
    def __init__(self, config: Optional[DeepCFRConfig] = None):
        self.config = config or DeepCFRConfig()
        self.node_store: Dict[str, TabularNode] = {}
        self.actor_snapshot: TabularPolicySnapshot = freeze_policy_snapshot(self.node_store, {"traversals_completed": 0, "iteration": 0})
        self._actor_snapshot_dirty = False
        self._node_store_version = 0
        self._worker_update_payload: Optional[Dict[str, object]] = None
        self.checkpoint_pool: List[TabularPolicySnapshot] = [copy.deepcopy(self.actor_snapshot)]
        self.traversals_completed = 0
        self.traverser_decisions = 0
        self.learner_steps = 0
        self.invalid_state_count = 0
        self.invalid_action_count = 0
        self._stats_window_size = max(1, int(self.config.averaging_window_traversals))
        self._recent_utilities = deque(maxlen=self._stats_window_size)
        self._recent_actions = deque(maxlen=self._stats_window_size)
        self._recent_preflop_actions = deque(maxlen=self._stats_window_size)
        self._recent_postflop_actions = deque(maxlen=self._stats_window_size)
        self._recent_vpip = deque(maxlen=self._stats_window_size)
        self._recent_pfr = deque(maxlen=self._stats_window_size)
        self._recent_three_bet = deque(maxlen=self._stats_window_size)
        self._recent_prejam = deque(maxlen=self._stats_window_size)
        self._recent_flop_seen = deque(maxlen=self._stats_window_size)
        self._recent_actions_per_hand = deque(maxlen=self._stats_window_size)
        self._recent_preflop_actions_per_hand = deque(maxlen=self._stats_window_size)
        self._position_profit_windows = {seat: deque(maxlen=self._stats_window_size) for seat in range(self.config.num_players)}
        self._conditioned_counts = _new_postflop_conditioned_counts()
        self._cumulative_perf = {key: 0.0 for key in ("state_init_time", "chance_time", "traverser_state_time", "opponent_state_time", "regret_infer_time", "strategy_infer_time", "branch_clone_time", "apply_time")}
        self._chunk_learner_steps = 0
        self._chunk_regret_steps = 0
        self._chunk_strategy_steps = 0
        self._chunk_advantage_samples = 0
        self._chunk_strategy_samples = 0
        self._outer_iteration = 0
        self._next_seat = 0
        self._start_time = time.time()
        self._last_discount_factor = 1.0
        self._rollout_executor: Optional[ProcessPoolExecutor] = None
        self._snapshot = self._make_snapshot("Idle")

    def _consume_result(self, result) -> None:
        self.traversals_completed += 1
        self.traverser_decisions += int(result.traverser_decisions)
        self.invalid_state_count += int(result.invalid_state_count)
        self.invalid_action_count += int(result.invalid_action_count)
        if bool(getattr(result, "monitor_sampled", False)):
            self._recent_utilities.append(float(result.utility_bb))
            self._recent_vpip.append(1.0 if result.vpip else 0.0)
            self._recent_pfr.append(1.0 if result.pfr else 0.0)
            self._recent_three_bet.append(1.0 if result.three_bet else 0.0)
            self._recent_prejam.append(1.0 if result.preflop_jam else 0.0)
            self._recent_flop_seen.append(1.0 if result.flop_seen else 0.0)
            self._recent_actions_per_hand.append(float(result.total_actions))
            self._recent_preflop_actions_per_hand.append(float(result.preflop_actions))
            self._position_profit_windows[int(result.traverser_seat)].append(float(result.utility_bb))
            for action_id, count in enumerate(np.asarray(result.action_counts).reshape(-1)):
                self._recent_actions.extend([action_id] * int(count))
            for action_id, count in enumerate(np.asarray(result.preflop_action_counts).reshape(-1)):
                self._recent_preflop_actions.extend([action_id] * int(count))
            for action_id, count in enumerate(np.asarray(result.postflop_action_counts).reshape(-1)):
                self._recent_postflop_actions.extend([action_id] * int(count))
            _merge_conditioned_counts(self._conditioned_counts, getattr(result, "postflop_conditioned_counts", None))
        for key, value in (result.perf_breakdown or {}).items():
            if key in self._cumulative_perf:
                self._cumulative_perf[key] += float(value)
        self._chunk_learner_steps += 1
        self._chunk_regret_steps += int(result.traverser_decisions)

    def _refresh_actor_snapshot(self, append_checkpoint: bool = False) -> None:
        self.actor_snapshot = freeze_policy_snapshot(
            self.node_store,
            {
                "traversals_completed": int(self.traversals_completed),
                "iteration": int(self._outer_iteration),
                "infosets": int(len(self.node_store)),
            },
        )
        self._actor_snapshot_dirty = False
        if append_checkpoint:
            self.checkpoint_pool.append(copy.deepcopy(self.actor_snapshot))
            if len(self.checkpoint_pool) > int(self.config.max_checkpoint_pool):
                self.checkpoint_pool = self.checkpoint_pool[-int(self.config.max_checkpoint_pool) :]

    def _ensure_actor_snapshot_current(self, append_checkpoint: bool = False) -> None:
        if bool(self._actor_snapshot_dirty) or append_checkpoint:
            self._refresh_actor_snapshot(append_checkpoint=append_checkpoint)

    def _effective_rollout_workers(self) -> int:
        configured = max(1, int(getattr(self.config, "rollout_workers", 1)))
        return min(configured, max(1, int(os.cpu_count() or configured)))

    @staticmethod
    def _default_rollout_workers() -> int:
        return max(1, max(1, (os.cpu_count() or 2) - 1))

    def _current_worker_snapshot_signature(self) -> str:
        return f"table_v{int(self._node_store_version)}"

    def _invalidate_worker_snapshot_sync(self) -> None:
        self._worker_update_payload = None

    def _build_full_worker_snapshot_payload(self) -> Dict[str, object]:
        return {
            "__snapshot_mode__": "full",
            "node_store": serialize_node_store(self.node_store),
        }

    def _accumulate_node_delta_payload(self, total: Dict[str, Dict[str, object]], update: Dict[str, Dict[str, object]]) -> None:
        if not isinstance(update, dict):
            return
        for infoset_key, payload in update.items():
            if not isinstance(payload, dict):
                continue
            key = str(infoset_key)
            if key not in total:
                total[key] = {
                    "legal_mask": np.asarray(payload.get("legal_mask", np.zeros(self.config.action_count, dtype=np.float32)), dtype=np.float32).copy(),
                    "regret_sum_delta": np.asarray(payload.get("regret_sum_delta", np.zeros(self.config.action_count, dtype=np.float32)), dtype=np.float32).copy(),
                    "strategy_sum_delta": np.asarray(payload.get("strategy_sum_delta", np.zeros(self.config.action_count, dtype=np.float32)), dtype=np.float32).copy(),
                    "visit_delta": int(payload.get("visit_delta", 0)),
                }
                continue
            total[key]["legal_mask"] = np.maximum(
                np.asarray(total[key]["legal_mask"], dtype=np.float32),
                np.asarray(payload.get("legal_mask", total[key]["legal_mask"]), dtype=np.float32),
            )
            total[key]["regret_sum_delta"] = (
                np.asarray(total[key]["regret_sum_delta"], dtype=np.float32)
                + np.asarray(payload.get("regret_sum_delta", np.zeros_like(total[key]["regret_sum_delta"])), dtype=np.float32)
            )
            total[key]["strategy_sum_delta"] = (
                np.asarray(total[key]["strategy_sum_delta"], dtype=np.float32)
                + np.asarray(payload.get("strategy_sum_delta", np.zeros_like(total[key]["strategy_sum_delta"])), dtype=np.float32)
            )
            total[key]["visit_delta"] = int(total[key]["visit_delta"]) + int(payload.get("visit_delta", 0))

    def _advance_worker_snapshot_version(self, batch_deltas: Optional[Dict[str, Dict[str, object]]], *, force_full_resync: bool) -> None:
        previous_signature = self._current_worker_snapshot_signature()
        self._node_store_version += 1
        if force_full_resync or not isinstance(batch_deltas, dict):
            self._worker_update_payload = None
            return
        self._worker_update_payload = {
            "__snapshot_mode__": "delta",
            "base_signature": previous_signature,
            "delta_payload": batch_deltas,
        }

    def _ensure_rollout_executor(self) -> Optional[ProcessPoolExecutor]:
        workers = self._effective_rollout_workers()
        if not bool(getattr(self.config, "parallel_rollouts", False)) or workers <= 1:
            return None
        current_workers = int(getattr(self._rollout_executor, "_max_workers", 0) or 0) if self._rollout_executor is not None else 0
        executor_shutdown = bool(getattr(self._rollout_executor, "_shutdown_thread", False)) if self._rollout_executor is not None else False
        executor_broken = bool(getattr(self._rollout_executor, "_broken", False)) if self._rollout_executor is not None else False
        if self._rollout_executor is None or current_workers != workers or executor_shutdown or executor_broken:
            if self._rollout_executor is not None:
                self._rollout_executor.shutdown(wait=False, cancel_futures=True)
            self._invalidate_worker_snapshot_sync()
            self._rollout_executor = ProcessPoolExecutor(max_workers=workers)
        return self._rollout_executor

    def _recreate_rollout_executor(self) -> Optional[ProcessPoolExecutor]:
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=False, cancel_futures=True)
            self._rollout_executor = None
        self._invalidate_worker_snapshot_sync()
        return self._ensure_rollout_executor()

    def _submit_rollout_batch_task(
        self,
        executor: ProcessPoolExecutor,
        *,
        hand_seeds: List[int],
        traverser_seats: List[int],
        actor_state_payload,
        config_payload: Dict[str, object],
        snapshot_signature: str,
    ):
        try:
            return executor.submit(
                run_traversal_batch_mp,
                hand_seeds,
                traverser_seats,
                actor_state_payload,
                None,
                config_payload,
                snapshot_signature,
            )
        except BrokenProcessPool:
            executor = self._recreate_rollout_executor()
            if executor is None:
                raise
            return executor.submit(
                run_traversal_batch_mp,
                hand_seeds,
                traverser_seats,
                actor_state_payload,
                None,
                config_payload,
                snapshot_signature,
            )
        except RuntimeError as exc:
            if "cannot schedule new futures after shutdown" not in str(exc).lower():
                raise
            executor = self._recreate_rollout_executor()
            if executor is None:
                raise
            return executor.submit(
                run_traversal_batch_mp,
                hand_seeds,
                traverser_seats,
                actor_state_payload,
                None,
                config_payload,
                snapshot_signature,
            )

    def _run_serial_traversals(self, hand_seeds: List[int], traverser_seats: List[int]) -> None:
        for hand_seed, traverser_seat in zip(hand_seeds, traverser_seats):
            result = run_traversal(int(hand_seed), int(traverser_seat), self.actor_snapshot, self.actor_snapshot, self.config)
            self._consume_result(result)

    def _degrade_parallel_rollouts(self) -> None:
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=False, cancel_futures=True)
            self._rollout_executor = None
        self._invalidate_worker_snapshot_sync()
        self.config.parallel_rollouts = False

    def _merge_node_deltas(self, node_deltas: Dict[str, Dict[str, object]]) -> None:
        if not isinstance(node_deltas, dict):
            return
        for infoset_key, payload in node_deltas.items():
            if not isinstance(payload, dict):
                continue
            legal_mask = np.asarray(payload.get("legal_mask", np.zeros(self.config.action_count, dtype=np.float32)), dtype=np.float32).reshape(-1)
            node = self.node_store.get(str(infoset_key))
            if node is None:
                node = TabularNode.new(legal_mask)
                self.node_store[str(infoset_key)] = node
            else:
                node.merge_legal_mask(legal_mask)
            regret_delta = np.asarray(payload.get("regret_sum_delta", np.zeros_like(node.regret_sum)), dtype=np.float32).reshape(-1)
            strategy_delta = np.asarray(payload.get("strategy_sum_delta", np.zeros_like(node.strategy_sum)), dtype=np.float32).reshape(-1)
            if regret_delta.shape == node.regret_sum.shape:
                node.regret_sum += regret_delta
            if strategy_delta.shape == node.strategy_sum.shape:
                node.strategy_sum += strategy_delta
            node.visits += int(payload.get("visit_delta", 0))

    def _maybe_discount_tables(self) -> bool:
        interval = max(1, int(getattr(self.config, "discount_interval_iterations", 2)))
        if int(self._outer_iteration) < int(getattr(self.config, "lcfr_after_iteration", 400)):
            self._last_discount_factor = 1.0
            return False
        if int(self._outer_iteration) <= 0 or int(self._outer_iteration) % interval != 0:
            self._last_discount_factor = 1.0
            return False
        scale_index = max(1, int(self._outer_iteration // interval))
        factor = float(scale_index / float(scale_index + 1))
        for node in self.node_store.values():
            node.regret_sum *= factor
            node.strategy_sum *= factor
        self._last_discount_factor = factor
        return True

    def _make_snapshot(self, status: str) -> TrainingSnapshot:
        action_hist = np.bincount(np.array(self._recent_actions, dtype=np.int64), minlength=self.config.action_count) if self._recent_actions else np.zeros(self.config.action_count, dtype=np.int64)
        pre_hist = np.bincount(np.array(self._recent_preflop_actions, dtype=np.int64), minlength=self.config.action_count) if self._recent_preflop_actions else np.zeros(self.config.action_count, dtype=np.int64)
        post_hist = np.bincount(np.array(self._recent_postflop_actions, dtype=np.int64), minlength=self.config.action_count) if self._recent_postflop_actions else np.zeros(self.config.action_count, dtype=np.int64)
        probs = action_hist.astype(np.float64) / max(1.0, float(action_hist.sum()))
        entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))) if np.any(probs > 0) else 0.0
        elapsed = max(time.time() - self._start_time, 1e-6)
        return TrainingSnapshot(
            status=status,
            traversals_completed=self.traversals_completed,
            traverser_decisions=self.traverser_decisions,
            exploration_epsilon=0.0,
            advantage_buffer_size=0,
            strategy_buffer_size=0,
            postflop_value_buffer_size=0,
            regret_loss=0.0,
            strategy_loss=0.0,
            postflop_value_loss=0.0,
            ema_regret_loss=0.0,
            ema_strategy_loss=0.0,
            ema_postflop_value_loss=0.0,
            avg_utility_bb=float(np.mean(self._recent_utilities)) if self._recent_utilities else 0.0,
            vpip=float(np.mean(self._recent_vpip)) if self._recent_vpip else 0.0,
            pfr=float(np.mean(self._recent_pfr)) if self._recent_pfr else 0.0,
            three_bet=float(np.mean(self._recent_three_bet)) if self._recent_three_bet else 0.0,
            preflop_jam_rate=float(np.mean(self._recent_prejam)) if self._recent_prejam else 0.0,
            flop_seen_rate=float(np.mean(self._recent_flop_seen)) if self._recent_flop_seen else 0.0,
            avg_actions_per_hand=float(np.mean(self._recent_actions_per_hand)) if self._recent_actions_per_hand else 0.0,
            avg_preflop_actions_per_hand=float(np.mean(self._recent_preflop_actions_per_hand)) if self._recent_preflop_actions_per_hand else 0.0,
            blueprint_coverage_pct=0.0,
            utility_window_count=len(self._recent_utilities),
            style_window_count=len(self._recent_vpip),
            position_window_size=max(len(window) for window in self._position_profit_windows.values()),
            action_entropy=entropy,
            invalid_state_count=self.invalid_state_count,
            invalid_action_count=self.invalid_action_count,
            hands_per_second=float(self.traversals_completed / elapsed),
            learner_steps=self.learner_steps,
            chunk_learner_steps=self._chunk_learner_steps,
            chunk_regret_steps=self._chunk_regret_steps,
            chunk_strategy_steps=self._chunk_strategy_steps,
            chunk_postflop_value_steps=0,
            chunk_advantage_samples=self._chunk_advantage_samples,
            chunk_strategy_samples=self._chunk_strategy_samples,
            chunk_postflop_value_samples=0,
            postflop_samples_per_traversal=0.0,
            checkpoint_pool_size=len(self.checkpoint_pool),
            action_histogram=action_hist.astype(int).tolist(),
            preflop_action_histogram=pre_hist.astype(int).tolist(),
            postflop_action_histogram=post_hist.astype(int).tolist(),
            postflop_conditioned_rates_by_street=_conditioned_rates(self._conditioned_counts),
            postflop_conditioned_counts_by_street=_format_conditioned_counts(self._conditioned_counts),
            position_avg_utility_bb={POSITION_NAMES_V21[seat]: (float(np.mean(window)) if window else 0.0) for seat, window in self._position_profit_windows.items()},
            perf_breakdown_ms={key: float(value / max(1, self.traversals_completed) * 1000.0) for key, value in self._cumulative_perf.items()},
            infoset_count=len(self.node_store),
            pruning_active=bool(self._outer_iteration >= int(self.config.prune_after_iteration)),
            discount_active=bool(self._outer_iteration >= int(self.config.lcfr_after_iteration)),
            last_discount_factor=float(self._last_discount_factor),
            algorithm_name=str(self.config.algorithm_name),
            monitor_mode="self_play",
            timestamp=time.time(),
        )

    def _refresh_snapshot(self, status: str) -> None:
        self._snapshot = self._make_snapshot(status)

    def _run_parallel_seat_batch(self, seat: int, seat_target: int) -> Optional[Dict[str, Dict[str, object]]]:
        executor = self._ensure_rollout_executor()
        workers = self._effective_rollout_workers()
        starting_seed = self.traversals_completed + 1
        if executor is None or workers <= 1 or seat_target < 2:
            self._run_serial_traversals(
                [starting_seed + offset for offset in range(seat_target)],
                [int(seat)] * seat_target,
            )
            return None

        base_chunk = max(1, int(getattr(self.config, "rollout_worker_chunk_size", 8)))
        snapshot_signature = self._current_worker_snapshot_signature()
        config_payload = asdict(self.config)
        config_payload.pop("checkpoint_pool", None)
        hand_seeds = [starting_seed + offset for offset in range(seat_target)]
        traverser_seats = [int(seat)] * seat_target
        task_ranges = [(start_idx, min(seat_target, start_idx + base_chunk)) for start_idx in range(0, seat_target, base_chunk)]
        prime_count = min(workers, len(task_ranges))
        cached_update_payload = self._worker_update_payload
        full_snapshot_payload: Optional[Dict[str, object]] = None

        def get_full_snapshot_payload() -> Dict[str, object]:
            nonlocal full_snapshot_payload
            if full_snapshot_payload is None:
                full_snapshot_payload = self._build_full_worker_snapshot_payload()
            return full_snapshot_payload

        task_specs = []
        future_map = {}
        for task_idx, (start_idx, end_idx) in enumerate(task_ranges):
            actor_state_payload = None
            if task_idx < prime_count:
                actor_state_payload = cached_update_payload if cached_update_payload is not None else get_full_snapshot_payload()
            spec = {
                "hand_seeds": hand_seeds[start_idx:end_idx],
                "traverser_seats": traverser_seats[start_idx:end_idx],
                "actor_state_payload": actor_state_payload,
            }
            task_specs.append(spec)
            future = self._submit_rollout_batch_task(
                executor,
                hand_seeds=spec["hand_seeds"],
                traverser_seats=spec["traverser_seats"],
                actor_state_payload=spec["actor_state_payload"],
                config_payload=config_payload,
                snapshot_signature=snapshot_signature,
            )
            future_map[future] = spec
        accumulated_deltas: Dict[str, Dict[str, object]] = {}
        completed_specs: set[tuple[int, ...]] = set()
        while future_map:
            for future in as_completed(list(future_map.keys())):
                spec = future_map.pop(future)
                spec_key = tuple(int(seed) for seed in spec["hand_seeds"])
                try:
                    batch_result = future.result()
                except RuntimeError as exc:
                    message = str(exc)
                    if "Worker snapshot cache miss" in message:
                        spec["actor_state_payload"] = get_full_snapshot_payload()
                        executor = self._ensure_rollout_executor()
                        if executor is None:
                            raise
                        retry_future = self._submit_rollout_batch_task(
                            executor,
                            hand_seeds=spec["hand_seeds"],
                            traverser_seats=spec["traverser_seats"],
                            actor_state_payload=spec["actor_state_payload"],
                            config_payload=config_payload,
                            snapshot_signature=snapshot_signature,
                        )
                        future_map[retry_future] = spec
                        continue
                    if "process pool was terminated abruptly" in message.lower():
                        remaining_specs = [pending for pending in task_specs if tuple(int(seed) for seed in pending["hand_seeds"]) not in completed_specs]
                        self._degrade_parallel_rollouts()
                        for pending in remaining_specs:
                            self._run_serial_traversals(pending["hand_seeds"], pending["traverser_seats"])
                        return None
                    if "cannot schedule new futures after shutdown" in message.lower():
                        executor = self._recreate_rollout_executor()
                        if executor is None:
                            raise
                        spec["actor_state_payload"] = get_full_snapshot_payload()
                        retry_future = self._submit_rollout_batch_task(
                            executor,
                            hand_seeds=spec["hand_seeds"],
                            traverser_seats=spec["traverser_seats"],
                            actor_state_payload=spec["actor_state_payload"],
                            config_payload=config_payload,
                            snapshot_signature=snapshot_signature,
                        )
                        future_map[retry_future] = spec
                        continue
                    raise
                except BrokenProcessPool:
                    remaining_specs = [pending for pending in task_specs if tuple(int(seed) for seed in pending["hand_seeds"]) not in completed_specs]
                    self._degrade_parallel_rollouts()
                    for pending in remaining_specs:
                        self._run_serial_traversals(pending["hand_seeds"], pending["traverser_seats"])
                    return None
                self._accumulate_node_delta_payload(accumulated_deltas, getattr(batch_result, "node_deltas", {}))
                self._merge_node_deltas(getattr(batch_result, "node_deltas", {}))
                for result in getattr(batch_result, "results", ()):
                    self._consume_result(result)
                completed_specs.add(spec_key)
                break
        return accumulated_deltas

    @staticmethod
    def _is_process_pool_runtime_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "process pool was terminated abruptly" in message or "cannot schedule new futures after shutdown" in message

    def _build_eval_config(self, mode: str):
        eval_config = build_runtime_policy_config(copy.copy(self.config))
        eval_config.evaluation_mode = "synthetic" if mode in SYNTHETIC_OPPONENT_STYLES else mode
        eval_config.current_iteration = self._outer_iteration
        eval_config.checkpoint_pool = tuple(self.checkpoint_pool)
        eval_config.synthetic_opponent_style = mode if mode in SYNTHETIC_OPPONENT_STYLES else ""
        return eval_config

    def _collect_evaluation_hands_serial(self, num_hands: int, eval_config) -> List[object]:
        results = []
        for hand_idx in range(max(1, int(num_hands))):
            eval_config.eval_hero_seat = hand_idx % self.config.num_players
            results.append(run_policy_hand(hand_idx + 1, self.actor_snapshot, eval_config))
        return results

    def _collect_evaluation_hands(self, num_hands: int, eval_config) -> List[object]:
        return self._collect_evaluation_hands_serial(num_hands, eval_config)

    def _build_evaluation_report(self, mode: str, hand_results: List[object], runtime_seconds: float) -> EvaluationReport:
        hands_played = max(1, int(len(hand_results)))
        action_hist = np.zeros(self.config.action_count, dtype=np.int64)
        pre_hist = np.zeros(self.config.action_count, dtype=np.int64)
        post_hist = np.zeros(self.config.action_count, dtype=np.int64)
        position_profit = {seat: [] for seat in range(self.config.num_players)}
        vpip_attempts = {seat: 0 for seat in range(self.config.num_players)}
        pfr_attempts = {seat: 0 for seat in range(self.config.num_players)}
        three_attempts = {seat: 0 for seat in range(self.config.num_players)}
        vpip_opp = {seat: 0 for seat in range(self.config.num_players)}
        rfi_attempts = {seat: 0 for seat in range(self.config.num_players)}
        rfi_opp = {seat: 0 for seat in range(self.config.num_players)}
        hand_counts = np.zeros((13, 13), dtype=np.int64)
        hand_vpip_hits = np.zeros((13, 13), dtype=np.int64)
        hand_rfi_counts = np.zeros((13, 13), dtype=np.int64)
        hand_rfi_hits = np.zeros((13, 13), dtype=np.int64)
        hand_pfr_hits = np.zeros((13, 13), dtype=np.int64)
        hand_three_bet_hits = np.zeros((13, 13), dtype=np.int64)
        position_hand_counts = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_vpip_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_rfi_counts = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_rfi_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_pfr_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_three_bet_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        total_profit = wins = illegal = vpip = pfr = three = prejam = flop_seen = 0
        total_actions = total_preflop_actions = 0
        counts = _new_postflop_counts()
        counts_by_pos = {name: _new_postflop_counts() for name in POSITION_NAMES_V21}
        conditioned = _new_postflop_conditioned_counts()

        for hand in hand_results:
            total_profit += float(hand.hero_profit_bb)
            wins += 1 if hand.win else 0
            illegal += int(hand.illegal_action_count)
            vpip += 1 if hand.vpip else 0
            pfr += 1 if hand.pfr else 0
            three += 1 if hand.three_bet else 0
            prejam += 1 if hand.preflop_jam else 0
            flop_seen += 1 if hand.flop_seen else 0
            total_actions += int(hand.total_actions)
            total_preflop_actions += int(hand.preflop_actions)
            action_hist += np.asarray(hand.action_counts, dtype=np.int64)
            pre_hist += np.asarray(hand.preflop_action_counts, dtype=np.int64)
            post_hist += np.asarray(hand.postflop_action_counts, dtype=np.int64)
            seat = int(hand.hero_seat)
            name = POSITION_NAMES_V21[seat]
            position_profit[seat].append(float(hand.hero_profit_bb))
            vpip_attempts[seat] += 1 if hand.vpip else 0
            pfr_attempts[seat] += 1 if hand.pfr else 0
            three_attempts[seat] += 1 if hand.three_bet else 0
            vpip_opp[seat] += 1
            if hand.rfi_opportunity:
                rfi_opp[seat] += 1
            if hand.rfi_attempt:
                rfi_attempts[seat] += 1
            hand_grid_indices = _hand_key_to_grid_indices(getattr(hand, "hero_hand_key", None))
            if hand_grid_indices is not None:
                row, col = hand_grid_indices
                hand_counts[row, col] += 1
                position_hand_counts[seat][row, col] += 1
                if hand.vpip:
                    hand_vpip_hits[row, col] += 1
                    position_hand_vpip_hits[seat][row, col] += 1
                if hand.rfi_opportunity:
                    hand_rfi_counts[row, col] += 1
                    position_hand_rfi_counts[seat][row, col] += 1
                    if hand.rfi_attempt:
                        hand_rfi_hits[row, col] += 1
                        position_hand_rfi_hits[seat][row, col] += 1
                if hand.pfr:
                    hand_pfr_hits[row, col] += 1
                    position_hand_pfr_hits[seat][row, col] += 1
                if hand.three_bet:
                    hand_three_bet_hits[row, col] += 1
                    position_hand_three_bet_hits[seat][row, col] += 1
            hand_postflop_counts = {
                "hands": 1,
                "flop_seen": 1 if hand.flop_seen else 0,
                "turn_seen": 1 if hand.turn_seen else 0,
                "river_seen": 1 if hand.river_seen else 0,
                "showdown_seen": 1 if hand.showdown_seen else 0,
                "showdown_won": 1 if hand.showdown_won else 0,
                "cbet_flop_opportunity": 1 if hand.cbet_flop_opportunity else 0,
                "cbet_flop_taken": 1 if hand.cbet_flop_taken else 0,
                "fold_vs_cbet_flop_opportunity": 1 if hand.fold_vs_cbet_flop_opportunity else 0,
                "fold_vs_cbet_flop": 1 if hand.fold_vs_cbet_flop else 0,
                "cbet_turn_opportunity": 1 if hand.cbet_turn_opportunity else 0,
                "cbet_turn_taken": 1 if hand.cbet_turn_taken else 0,
                "fold_vs_cbet_turn_opportunity": 1 if hand.fold_vs_cbet_turn_opportunity else 0,
                "fold_vs_cbet_turn": 1 if hand.fold_vs_cbet_turn else 0,
            }
            for key, value in hand_postflop_counts.items():
                counts[key] += int(value)
                counts_by_pos[name][key] += int(value)
            _merge_conditioned_counts(conditioned, hand.postflop_conditioned_counts)

        vpip_hand_grid = _rate_grid_from_counts(hand_vpip_hits, hand_counts)
        rfi_hand_grid = _rate_grid_from_counts(hand_rfi_hits, hand_rfi_counts)
        pfr_hand_grid = _rate_grid_from_counts(hand_pfr_hits, hand_counts)
        three_bet_hand_grid = _rate_grid_from_counts(hand_three_bet_hits, hand_counts)
        vpip_hand_grid_by_position = {
            POSITION_NAMES_V21[seat]: _rate_grid_from_counts(position_hand_vpip_hits[seat], position_hand_counts[seat])
            for seat in range(self.config.num_players)
        }
        rfi_hand_grid_by_position = {
            POSITION_NAMES_V21[seat]: _rate_grid_from_counts(position_hand_rfi_hits[seat], position_hand_rfi_counts[seat])
            for seat in range(self.config.num_players)
        }
        pfr_hand_grid_by_position = {
            POSITION_NAMES_V21[seat]: _rate_grid_from_counts(position_hand_pfr_hits[seat], position_hand_counts[seat])
            for seat in range(self.config.num_players)
        }
        three_bet_hand_grid_by_position = {
            POSITION_NAMES_V21[seat]: _rate_grid_from_counts(position_hand_three_bet_hits[seat], position_hand_counts[seat])
            for seat in range(self.config.num_players)
        }
        zero_grid = np.zeros((13, 13), dtype=np.float32).tolist()
        return EvaluationReport(
            mode=mode,
            hands=hands_played,
            avg_profit_bb=float(total_profit / hands_played),
            win_rate=float(wins / hands_played),
            vpip=float(vpip / hands_played),
            rfi=float(sum(rfi_attempts.values()) / max(1, sum(rfi_opp.values()))) if sum(rfi_opp.values()) > 0 else 0.0,
            pfr=float(pfr / hands_played),
            three_bet=float(three / hands_played),
            preflop_jam_rate=float(prejam / hands_played),
            flop_seen_rate=float(flop_seen / hands_played),
            avg_actions_per_hand=float(total_actions / hands_played),
            avg_preflop_actions_per_hand=float(total_preflop_actions / hands_played),
            blueprint_coverage_pct=0.0,
            illegal_action_count=illegal,
            runtime_seconds=float(runtime_seconds),
            action_histogram=action_hist.astype(int).tolist(),
            preflop_action_histogram=pre_hist.astype(int).tolist(),
            postflop_action_histogram=post_hist.astype(int).tolist(),
            postflop_conditioned_rates_by_street=_conditioned_rates(conditioned),
            postflop_conditioned_counts_by_street=_format_conditioned_counts(conditioned),
            position_avg_profit_bb={POSITION_NAMES_V21[seat]: (float(np.mean(values)) if values else 0.0) for seat, values in position_profit.items()},
            vpip_by_position={POSITION_NAMES_V21[seat]: float(vpip_attempts[seat] / max(1, vpip_opp[seat])) for seat in range(self.config.num_players)},
            rfi_by_position={POSITION_NAMES_V21[seat]: float(rfi_attempts[seat] / max(1, rfi_opp[seat])) if rfi_opp[seat] > 0 else 0.0 for seat in range(self.config.num_players)},
            pfr_by_position={POSITION_NAMES_V21[seat]: float(pfr_attempts[seat] / max(1, vpip_opp[seat])) for seat in range(self.config.num_players)},
            three_bet_by_position={POSITION_NAMES_V21[seat]: float(three_attempts[seat] / max(1, vpip_opp[seat])) for seat in range(self.config.num_players)},
            vpip_hand_grid=vpip_hand_grid,
            rfi_hand_grid=rfi_hand_grid,
            pfr_hand_grid=pfr_hand_grid,
            three_bet_hand_grid=three_bet_hand_grid,
            vpip_hand_grid_by_position=vpip_hand_grid_by_position,
            rfi_hand_grid_by_position=rfi_hand_grid_by_position,
            pfr_hand_grid_by_position=pfr_hand_grid_by_position,
            three_bet_hand_grid_by_position=three_bet_hand_grid_by_position,
            postflop_rates=_postflop_rates(counts),
            postflop_counts={key: int(value) for key, value in counts.items()},
            postflop_rates_by_position={name: _postflop_rates(pos_counts) for name, pos_counts in counts_by_pos.items()},
            postflop_counts_by_position={name: {key: int(value) for key, value in pos_counts.items()} for name, pos_counts in counts_by_pos.items()},
            postflop_hand_grids={metric: zero_grid for metric in POSTFLOP_RATE_KEYS},
            postflop_hand_grids_by_position={name: {metric: zero_grid for metric in POSTFLOP_RATE_KEYS} for name in POSITION_NAMES_V21},
            postflop_profit_by_stage={stage: 0.0 for stage in ("all_hands", "flop_seen", "turn_seen", "river_seen", "showdown_seen")},
        )

    def train_for_traversals(self, num_traversals: int) -> TrainingSnapshot:
        requested = max(1, int(num_traversals))
        self._chunk_learner_steps = 0
        self._chunk_regret_steps = 0
        self._chunk_strategy_steps = 0
        self._chunk_advantage_samples = 0
        self._chunk_strategy_samples = 0
        remaining = requested
        self.config.live_node_store = self.node_store
        self.config.current_iteration = self._outer_iteration
        while remaining > 0:
            seat = self._next_seat % self.config.num_players
            self._next_seat += 1
            seat_target = min(int(self.config.traversals_per_player_per_iteration), remaining)
            self.config.current_iteration = self._outer_iteration
            batch_deltas = self._run_parallel_seat_batch(seat, seat_target)
            remaining -= seat_target
            self._outer_iteration += 1
            self.learner_steps = self._outer_iteration
            discount_applied = self._maybe_discount_tables()
            self._advance_worker_snapshot_version(batch_deltas, force_full_resync=bool(discount_applied or batch_deltas is None))
            self._chunk_strategy_steps += 1
            if seat_target > 0:
                self._actor_snapshot_dirty = True
        append_checkpoint = bool(self.traversals_completed > 0 and self.traversals_completed % max(1, int(self.config.checkpoint_interval)) == 0)
        self._ensure_actor_snapshot_current(append_checkpoint=append_checkpoint)
        self._refresh_snapshot("Training")
        return self.get_snapshot()

    def train_forever(self, stop_event, pause_event, snapshot_callback=None) -> None:
        self._refresh_snapshot("Training")
        while not stop_event.is_set():
            if pause_event.is_set():
                self._refresh_snapshot("Paused")
                time.sleep(0.1)
                continue
            snapshot = self.train_for_traversals(self.config.traversals_per_chunk)
            if snapshot_callback is not None:
                snapshot_callback(snapshot)
        self._refresh_snapshot("Stopped")

    def _evaluate(self, mode: str, num_hands: int) -> EvaluationReport:
        start = time.time()
        self._ensure_actor_snapshot_current()
        eval_config = self._build_eval_config(mode)
        hand_results = self._collect_evaluation_hands(num_hands, eval_config)
        return self._build_evaluation_report(mode, hand_results, time.time() - start)

    def evaluate_vs_heuristics(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("heuristics", num_hands)

    def evaluate_self_play(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("self_play", num_hands)

    def evaluate_network_only(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("self_play", num_hands)

    def evaluate_vs_checkpoint_pool(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("checkpoints", num_hands)

    def evaluate_vs_v21_table(self, num_hands: int) -> EvaluationReport:
        raise RuntimeError("Legacy v21 table evaluation was removed from v24.")

    def evaluate_vs_leak_pool(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("synthetic", num_hands)

    def evaluate_vs_synthetic_style(self, style: str, num_hands: int) -> EvaluationReport:
        return self._evaluate(str(style or "").strip().lower(), num_hands)

    def evaluate_eval_suite(self, num_hands: int) -> ExploitSuiteReport:
        hands = max(1, int(num_hands))
        leak_reports = {style: self.evaluate_vs_synthetic_style(style, hands) for style in SYNTHETIC_OPPONENT_STYLES}
        robust_reports = {
            "heuristics": self.evaluate_vs_heuristics(hands),
            "checkpoints": self.evaluate_vs_checkpoint_pool(hands),
        }
        return ExploitSuiteReport(
            hands_per_mode=hands,
            leak_reports=leak_reports,
            robust_reports=robust_reports,
            avg_leak_profit_bb=float(np.mean([report.avg_profit_bb for report in leak_reports.values()])) if leak_reports else 0.0,
            avg_robust_profit_bb=float(np.mean([report.avg_profit_bb for report in robust_reports.values()])) if robust_reports else 0.0,
            avg_leak_win_rate=float(np.mean([report.win_rate for report in leak_reports.values()])) if leak_reports else 0.0,
            avg_robust_win_rate=float(np.mean([report.win_rate for report in robust_reports.values()])) if robust_reports else 0.0,
            suite_name="eval_suite",
        )

    def evaluate_exploit_suite(self, num_hands: int) -> ExploitSuiteReport:
        hands = max(1, int(num_hands))
        leak_reports = {style: self.evaluate_vs_synthetic_style(style, hands) for style in SYNTHETIC_OPPONENT_STYLES}
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
            avg_robust_profit_bb=float(np.mean([report.avg_profit_bb for report in robust_reports.values()])) if robust_reports else 0.0,
            avg_leak_win_rate=float(np.mean([report.win_rate for report in leak_reports.values()])) if leak_reports else 0.0,
            avg_robust_win_rate=float(np.mean([report.win_rate for report in robust_reports.values()])) if robust_reports else 0.0,
            suite_name="exploit_suite",
        )

    def save_checkpoint(self, path: str) -> None:
        self._ensure_actor_snapshot_current()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "format_version": "tabular_mccfr_v24",
            "config": asdict(self.config),
            "node_store": serialize_node_store(self.node_store),
            "actor_snapshot": self.actor_snapshot.to_payload(),
            "checkpoint_pool": [snapshot.to_payload() for snapshot in self.checkpoint_pool],
            "metrics": {
                "traversals_completed": self.traversals_completed,
                "traverser_decisions": self.traverser_decisions,
                "learner_steps": self.learner_steps,
                "invalid_state_count": self.invalid_state_count,
                "invalid_action_count": self.invalid_action_count,
                "outer_iteration": self._outer_iteration,
                "next_seat": self._next_seat,
                "last_discount_factor": self._last_discount_factor,
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or str(payload.get("format_version", "")) != "tabular_mccfr_v24":
            raise RuntimeError("Checkpoint is incompatible with the tabular MCCFR v24 format.")
        config_payload = payload.get("config", {})
        if isinstance(config_payload, dict):
            for key, value in config_payload.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        if str(getattr(self.config, "algorithm_name", "")) == "tabular_mccfr_6max":
            self.config.parallel_rollouts = True
            self.config.rollout_workers = max(int(getattr(self.config, "rollout_workers", 1)), self._default_rollout_workers())
            self.config.rollout_worker_chunk_size = max(8, int(getattr(self.config, "rollout_worker_chunk_size", 1)))
        self.node_store = deserialize_node_store(payload.get("node_store", {}))
        actor_snapshot_payload = payload.get("actor_snapshot")
        self.actor_snapshot = (
            TabularPolicySnapshot.from_payload(actor_snapshot_payload)
            if isinstance(actor_snapshot_payload, dict)
            else freeze_policy_snapshot(self.node_store, {"traversals_completed": 0, "iteration": 0})
        )
        self._actor_snapshot_dirty = False
        pool_payload = payload.get("checkpoint_pool", [])
        self.checkpoint_pool = [
            TabularPolicySnapshot.from_payload(snapshot_payload)
            for snapshot_payload in pool_payload
            if isinstance(snapshot_payload, dict)
        ]
        if not self.checkpoint_pool:
            self.checkpoint_pool = [copy.deepcopy(self.actor_snapshot)]
        metrics = payload.get("metrics", {})
        self.traversals_completed = int(metrics.get("traversals_completed", 0))
        self.traverser_decisions = int(metrics.get("traverser_decisions", 0))
        self.learner_steps = int(metrics.get("learner_steps", 0))
        self.invalid_state_count = int(metrics.get("invalid_state_count", 0))
        self.invalid_action_count = int(metrics.get("invalid_action_count", 0))
        self._outer_iteration = int(metrics.get("outer_iteration", 0))
        self._next_seat = int(metrics.get("next_seat", 0))
        self._last_discount_factor = float(metrics.get("last_discount_factor", 1.0))
        self._node_store_version = 0
        self._worker_update_payload = None
        self.config.current_iteration = self._outer_iteration
        self.config.live_node_store = self.node_store
        for window in (
            self._recent_utilities,
            self._recent_actions,
            self._recent_preflop_actions,
            self._recent_postflop_actions,
            self._recent_vpip,
            self._recent_pfr,
            self._recent_three_bet,
            self._recent_prejam,
            self._recent_flop_seen,
            self._recent_actions_per_hand,
            self._recent_preflop_actions_per_hand,
        ):
            window.clear()
        for window in self._position_profit_windows.values():
            window.clear()
        self._conditioned_counts = _new_postflop_conditioned_counts()
        self._cumulative_perf = {key: 0.0 for key in self._cumulative_perf}
        self._refresh_snapshot("Loaded")

    def get_snapshot(self) -> TrainingSnapshot:
        return copy.deepcopy(self._snapshot)

    def shutdown(self) -> None:
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=False, cancel_futures=True)
            self._rollout_executor = None
        self._invalidate_worker_snapshot_sync()
        return None


DeepCFRTrainerV21 = DeepCFRTrainerV24
