import copy
import math
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
TRAINING_ROOT = os.path.dirname(SRC_ROOT)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
MODELS_DIR = os.path.join(SRC_ROOT, "models")
WORKERS_DIR = os.path.join(SRC_ROOT, "workers")
DEFAULT_V21_CHECKPOINT_PATH = os.path.join(TRAINING_ROOT, "models", "poker_agent_v21_deepcfr.pt")
DEFAULT_V23_CHECKPOINT_PATH = os.path.join(TRAINING_ROOT, "models", "poker_agent_v23_deepcfr.pt")
for path in (FEATURES_DIR, MODELS_DIR, WORKERS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from poker_model_v23 import PokerDeepCFRNet as PokerDeepCFRNetV23
from poker_model_v23 import load_compatible_state_dict as load_compatible_state_dict_v23
from poker_model_v24 import AdvantageBuffer, PokerDeepCFRNet, StrategyBuffer, load_compatible_state_dict
from poker_state_v24 import (
    ACTION_COUNT_V21,
    ACTION_NAMES_V21,
    OPPONENT_PROFILE_DEFAULT_V24,
    OPPONENT_PROFILE_DEFAULT_SLOT_V24,
    OPPONENT_PROFILE_FEATURE_NAMES_V24,
    OPPONENT_PROFILE_PER_SLOT_DIM_V24,
    POSITION_NAMES_V21,
    STATE_DIM_V21,
)
from poker_worker_v24 import SYNTHETIC_OPPONENT_STYLES, run_policy_hand, run_traversal, run_traversal_batch_mp
from gpu_rollout_inference_v24 import GPURolloutInferenceService

RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"


@dataclass
class DeepCFRConfig:
    num_players: int = 6
    small_blind: int = 5
    big_blind: int = 10
    state_dim: int = STATE_DIM_V21
    action_count: int = ACTION_COUNT_V21
    hidden_dim: int = 1024
    full_branch_depth: int = 1
    max_branch_actions: int = 0
    learning_rate: float = 1e-4
    regret_batch_size: int = 1024
    strategy_batch_size: int = 2048
    exploit_batch_size: int = 2048
    advantage_capacity: int = 500_000
    strategy_capacity: int = 1_000_000
    exploit_capacity: int = 1_000_000
    warmup_advantage_samples: int = 32768
    regret_update_interval: int = 1024
    strategy_update_interval: int = 2048
    exploit_update_interval: int = 2048
    publish_snapshot_interval: int = 8 #change to 4 or 8
    checkpoint_interval: int = 2_000
    max_checkpoint_pool: int = 64
    pool_growth_warmup_multiplier: float = 2.0
    averaging_window_traversals: int = 2048 * 8 # probably at least traversals_per_chunk so we don't lie
    clip_grad_norm: float = 0.6
    traversals_per_chunk: int = 2048 # increase for throughput 
    use_training_pool: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    evaluation_mode: str = "heuristics"
    eval_hero_seat: int = 0
    checkpoint_pool: tuple = field(default_factory=tuple)
    seed_v21_count: int = 5
    seed_v21_checkpoint_path: str = DEFAULT_V21_CHECKPOINT_PATH
    seed_v23_count: int = 5
    seed_v23_checkpoint_path: str = DEFAULT_V23_CHECKPOINT_PATH
    rollout_workers: int = 10 
    rollout_worker_chunk_size: int = 32
    parallel_rollouts: bool = True
    rollout_auto_scale_workers: bool = True
    rollout_worker_reserve_cores: int = 2
    rollout_worker_auto_cap: int = 0
    opponent_hold_traversals: int = 256
    opponent_policy_mode: str = "snapshot"
    synthetic_opponent_style: str = ""
    synthetic_opponent_probability: float = 0.35
    synthetic_opponent_styles: tuple = SYNTHETIC_OPPONENT_STYLES
    exploit_prior_enabled: bool = True
    exploit_prior_strength: float = 0.55
    exploit_min_confidence: float = 0.10
    exploit_only_preflop_unopened: bool = False
    exploit_teacher_mix: float = 0.65
    exploit_safety_weight: float = 0.20
    opponent_profile_decay: float = 0.997
    opponent_profile_short_alpha: float = 0.25
    opponent_profile_long_alpha: float = 0.05
    opponent_profile_confidence_scale: float = 256.0
    opponent_profile: tuple = OPPONENT_PROFILE_DEFAULT_V24
    opponent_profiles_by_seat: dict = field(default_factory=dict)
    gpu_rollout_inference: bool = torch.cuda.is_available()
    gpu_inference_max_batch_size: int = 1024
    gpu_inference_max_wait_ms: float = 2.5
    gpu_inference_model_cache_size: int = 6
    gpu_rollout_remote_opponents: bool = False
    quantize_local_opponent_models: bool = True


@dataclass
class TrainingSnapshot:
    status: str
    traversals_completed: int
    traverser_decisions: int
    advantage_buffer_size: int
    strategy_buffer_size: int
    regret_loss: float
    strategy_loss: float
    avg_utility_bb: float
    vpip: float
    pfr: float
    three_bet: float
    utility_window_count: int
    style_window_count: int
    position_window_size: int
    action_entropy: float
    invalid_state_count: int
    invalid_action_count: int
    hands_per_second: float
    learner_steps: int
    checkpoint_pool_size: int
    action_histogram: List[int]
    position_avg_utility_bb: Dict[str, float]
    perf_breakdown_ms: Dict[str, float]
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
    pfr: float
    three_bet: float
    illegal_action_count: int
    runtime_seconds: float
    action_histogram: List[int]
    position_avg_profit_bb: Dict[str, float]
    vpip_by_position: Dict[str, float]
    pfr_by_position: Dict[str, float]
    three_bet_by_position: Dict[str, float]
    vpip_hand_grid: List[List[float]]
    pfr_hand_grid: List[List[float]]
    three_bet_hand_grid: List[List[float]]
    vpip_hand_grid_by_position: Dict[str, List[List[float]]]
    pfr_hand_grid_by_position: Dict[str, List[List[float]]]
    three_bet_hand_grid_by_position: Dict[str, List[List[float]]]

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

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class DeepCFRTrainerV24:
    def __init__(self, config: Optional[DeepCFRConfig] = None):
        self.config = config or DeepCFRConfig()
        self.config.opponent_profile = tuple(OPPONENT_PROFILE_DEFAULT_V24)
        self.config.opponent_profiles_by_seat = {}
        self.config.opponent_policy_mode = "snapshot"
        self.config.synthetic_opponent_style = ""
        self.device = torch.device(self.config.device)
        self.model = PokerDeepCFRNet(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim,
            action_dim=self.config.action_count,
        ).to(self.device)

        regret_params = (
            list(self.model.input_layer.parameters())
            + list(self.model.block1.parameters())
            + list(self.model.block2.parameters())
            + list(self.model.block3.parameters())
            + list(self.model.regret_head.parameters())
        )
        strategy_params = (
            list(self.model.input_layer.parameters())
            + list(self.model.block1.parameters())
            + list(self.model.block2.parameters())
            + list(self.model.block3.parameters())
            + list(self.model.strategy_head.parameters())
        )
        exploit_params = (
            list(self.model.input_layer.parameters())
            + list(self.model.block1.parameters())
            + list(self.model.block2.parameters())
            + list(self.model.block3.parameters())
            + list(self.model.exploit_head.parameters())
        )
        self.regret_optimizer = torch.optim.Adam(regret_params, lr=self.config.learning_rate)
        self.strategy_optimizer = torch.optim.Adam(strategy_params, lr=self.config.learning_rate)
        self.exploit_optimizer = torch.optim.Adam(exploit_params, lr=self.config.learning_rate)

        self.advantage_buffer = AdvantageBuffer(
            capacity=self.config.advantage_capacity,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_count,
        )
        self.strategy_buffer = StrategyBuffer(
            capacity=self.config.strategy_capacity,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_count,
        )
        self.exploit_buffer = StrategyBuffer(
            capacity=self.config.exploit_capacity,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_count,
        )

        self.actor_snapshot = self.model.clone_cpu()
        self._v21_seed_template = self._load_seed_template(
            getattr(self.config, "seed_v21_checkpoint_path", ""),
            label="v21",
        )
        self._v23_seed_template = self._load_seed_template(
            getattr(self.config, "seed_v23_checkpoint_path", ""),
            label="v23",
        )
        self.checkpoint_pool: List[object] = []
        self._reset_checkpoint_pool()

        self._lock = threading.Lock()
        self._seed_rng = np.random.default_rng(int(time.time()))
        self._start_time = time.time()

        self.traversals_completed = 0
        self.traverser_decisions = 0
        self.learner_steps = 0
        self.regret_steps = 0
        self.strategy_steps = 0
        self.exploit_steps = 0
        self.invalid_state_count = 0
        self.invalid_action_count = 0

        self._adv_pending = 0
        self._strat_pending = 0
        self._exploit_pending = 0
        self._last_regret_loss = 0.0
        self._last_strategy_loss = 0.0
        self._last_exploit_loss = 0.0

        self._stats_window_size = max(1, int(getattr(self.config, "averaging_window_traversals", 16_384)))
        self._recent_utilities = deque(maxlen=self._stats_window_size)
        self._recent_actions = deque(maxlen=self._stats_window_size)
        self._recent_vpip = deque(maxlen=self._stats_window_size)
        self._recent_pfr = deque(maxlen=self._stats_window_size)
        self._recent_three_bet = deque(maxlen=self._stats_window_size)
        self._position_profit_windows = {
            seat: deque(maxlen=self._stats_window_size) for seat in range(self.config.num_players)
        }
        self._recent_traversal_times = deque(maxlen=256)
        self._cumulative_work_time = 0.0
        self._cumulative_rollout_time = 0.0
        self._cumulative_regret_time = 0.0
        self._cumulative_strategy_time = 0.0
        self._cumulative_snapshot_time = 0.0
        self._cumulative_rollout_detail = {
            "state_init_time": 0.0,
            "chance_time": 0.0,
            "traverser_state_time": 0.0,
            "opponent_state_time": 0.0,
            "regret_infer_time": 0.0,
            "strategy_infer_time": 0.0,
            "branch_clone_time": 0.0,
            "apply_time": 0.0,
        }
        self._snapshot = self._make_snapshot(status="Idle")
        self._rollout_executor: Optional[ProcessPoolExecutor] = None
        self._snapshot_version: int = 0
        self._opponent_profiles: Dict[str, Dict[str, float]] = {}
        self._held_opponent_snapshot: Optional[object] = None
        self._held_opponent_policy_mode: str = "snapshot"
        self._held_synthetic_opponent_style: str = ""
        self._held_opponent_profile_key: str = ""
        self._held_opponent_block_start: int = -1
        self._held_opponent_block_end: int = -1
        checkpoint_interval = max(0, int(getattr(self.config, "checkpoint_interval", 0)))
        self._next_checkpoint_traverser_decisions = checkpoint_interval if checkpoint_interval > 0 else 0
        self._gpu_rollout_service: Optional[GPURolloutInferenceService] = None
        self._gpu_registered_model_keys: set[str] = set()
        self._maybe_start_gpu_rollout_service()
        self._sync_gpu_rollout_registry()

    @staticmethod
    def _state_dict_from_payload(payload: object) -> Dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload is not a dictionary.")
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if any(str(key).endswith("input_layer.weight") for key in payload.keys()):
            return payload
        raise ValueError("Could not find model state dict in checkpoint payload.")

    @staticmethod
    def _infer_model_dims(
        state_dict: Dict[str, torch.Tensor],
        fallback_state_dim: int,
        fallback_hidden_dim: int,
        fallback_action_dim: int,
    ) -> tuple[int, int, int]:
        state_dim = int(fallback_state_dim)
        hidden_dim = int(fallback_hidden_dim)
        action_dim = int(fallback_action_dim)

        input_weight = None
        regret_out_weight = None
        for key, tensor in state_dict.items():
            name = str(key)
            if input_weight is None and name.endswith("input_layer.weight"):
                input_weight = tensor
            if regret_out_weight is None and name.endswith("regret_head.2.weight"):
                regret_out_weight = tensor

        if input_weight is not None and getattr(input_weight, "ndim", 0) == 2:
            hidden_dim = int(input_weight.shape[0])
            state_dim = int(input_weight.shape[1])
        if regret_out_weight is not None and getattr(regret_out_weight, "ndim", 0) == 2:
            action_dim = int(regret_out_weight.shape[0])

        return state_dim, hidden_dim, action_dim

    def _load_seed_template(self, path_value: object, label: str) -> Optional[object]:
        path = str(path_value or "").strip()
        checkpoint_label = str(label or "seed")
        if not path:
            return None
        if not os.path.isfile(path):
            print(f"[v24] {checkpoint_label} seed checkpoint not found: {path}")
            return None

        try:
            payload = torch.load(path, map_location="cpu")
            state_dict = self._state_dict_from_payload(payload)
            state_dim, hidden_dim, action_dim = self._infer_model_dims(
                state_dict,
                fallback_state_dim=self.config.state_dim,
                fallback_hidden_dim=self.config.hidden_dim,
                fallback_action_dim=self.config.action_count,
            )
            if int(action_dim) == 5:
                seed_model = PokerDeepCFRNetV23(
                    state_dim=state_dim,
                    hidden_dim=hidden_dim,
                    action_dim=action_dim,
                    init_weights=False,
                )
                load_compatible_state_dict_v23(seed_model, state_dict)
                setattr(seed_model, "_policy_family", "legacy_v23")
            else:
                seed_model = PokerDeepCFRNet(
                    state_dim=self.config.state_dim,
                    hidden_dim=hidden_dim,
                    action_dim=self.config.action_count,
                    init_weights=False,
                )
                load_compatible_state_dict(seed_model, state_dict)
                setattr(seed_model, "_policy_family", "v24")
            seed_model.eval()
            seed_model.to("cpu")
            for param in seed_model.parameters():
                param.requires_grad_(False)
            return seed_model
        except Exception as exc:
            print(f"[v24] Failed to load {checkpoint_label} seed checkpoint from '{path}': {exc}")
            return None

    def _reset_checkpoint_pool(self) -> None:
        self.checkpoint_pool = []
        v21_seed_count = max(0, int(getattr(self.config, "seed_v21_count", 0)))
        v23_seed_count = max(0, int(getattr(self.config, "seed_v23_count", 0)))
        if self._v21_seed_template is not None and v21_seed_count > 0:
            for _ in range(v21_seed_count):
                self.checkpoint_pool.append(self._v21_seed_template.clone_cpu())
        if self._v23_seed_template is not None and v23_seed_count > 0:
            for _ in range(v23_seed_count):
                self.checkpoint_pool.append(self._v23_seed_template.clone_cpu())
        self.checkpoint_pool.append(self.actor_snapshot)
        if len(self.checkpoint_pool) > self.config.max_checkpoint_pool:
            self.checkpoint_pool = self.checkpoint_pool[-self.config.max_checkpoint_pool :]
        self._held_opponent_snapshot = None
        self._held_opponent_policy_mode = "snapshot"
        self._held_synthetic_opponent_style = ""
        self._held_opponent_profile_key = ""
        self._held_opponent_block_start = -1
        self._held_opponent_block_end = -1

    def _make_snapshot(self, status: str) -> TrainingSnapshot:
        action_hist = [0] * self.config.action_count
        if self._recent_actions:
            counts = np.bincount(np.array(self._recent_actions, dtype=np.int64), minlength=self.config.action_count)
            action_hist = counts.astype(int).tolist()
            probs = counts.astype(np.float64) / max(1.0, float(counts.sum()))
            nonzero = probs > 0
            action_entropy = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))
        else:
            action_entropy = 0.0

        pos_stats = {}
        for seat, window in self._position_profit_windows.items():
            if window:
                pos_stats[POSITION_NAMES_V21[seat]] = float(np.mean(window))
            else:
                pos_stats[POSITION_NAMES_V21[seat]] = 0.0

        if self._recent_traversal_times:
            avg_traversal_time = float(np.mean(self._recent_traversal_times))
            hands_per_second = 1.0 / max(avg_traversal_time, 1e-6)
        else:
            elapsed = max(time.time() - self._start_time, 1e-6)
            hands_per_second = float(self.traversals_completed / elapsed)

        traversals = max(1, self.traversals_completed)
        tracked_time = (
            self._cumulative_rollout_time
            + self._cumulative_regret_time
            + self._cumulative_strategy_time
            + self._cumulative_snapshot_time
        )
        other_time = max(0.0, self._cumulative_work_time - tracked_time)
        perf_breakdown_ms = {
            "total_time": float(self._cumulative_work_time / traversals * 1000.0),
            "rollout_total_time": float(self._cumulative_rollout_time / traversals * 1000.0),
            "state_init_time": float(self._cumulative_rollout_detail["state_init_time"] / traversals * 1000.0),
            "chance_time": float(self._cumulative_rollout_detail["chance_time"] / traversals * 1000.0),
            "traverser_state_time": float(self._cumulative_rollout_detail["traverser_state_time"] / traversals * 1000.0),
            "opponent_state_time": float(self._cumulative_rollout_detail["opponent_state_time"] / traversals * 1000.0),
            "regret_infer_time": float(self._cumulative_rollout_detail["regret_infer_time"] / traversals * 1000.0),
            "strategy_infer_time": float(self._cumulative_rollout_detail["strategy_infer_time"] / traversals * 1000.0),
            "branch_clone_time": float(self._cumulative_rollout_detail["branch_clone_time"] / traversals * 1000.0),
            "apply_time": float(self._cumulative_rollout_detail["apply_time"] / traversals * 1000.0),
            "regret_train_time": float(self._cumulative_regret_time / traversals * 1000.0),
            "strategy_train_time": float(self._cumulative_strategy_time / traversals * 1000.0),
            "snapshot_time": float(self._cumulative_snapshot_time / traversals * 1000.0),
            "other_time": float(other_time / traversals * 1000.0),
        }
        return TrainingSnapshot(
            status=status,
            traversals_completed=self.traversals_completed,
            traverser_decisions=self.traverser_decisions,
            advantage_buffer_size=len(self.advantage_buffer),
            strategy_buffer_size=len(self.strategy_buffer),
            regret_loss=self._last_regret_loss,
            strategy_loss=self._last_strategy_loss,
            avg_utility_bb=float(np.mean(self._recent_utilities)) if self._recent_utilities else 0.0,
            vpip=float(np.mean(self._recent_vpip)) if self._recent_vpip else 0.0,
            pfr=float(np.mean(self._recent_pfr)) if self._recent_pfr else 0.0,
            three_bet=float(np.mean(self._recent_three_bet)) if self._recent_three_bet else 0.0,
            utility_window_count=len(self._recent_utilities),
            style_window_count=len(self._recent_vpip),
            position_window_size=self._stats_window_size,
            action_entropy=action_entropy,
            invalid_state_count=self.invalid_state_count,
            invalid_action_count=self.invalid_action_count,
            hands_per_second=hands_per_second,
            learner_steps=self.learner_steps,
            checkpoint_pool_size=len(self.checkpoint_pool),
            action_histogram=action_hist,
            position_avg_utility_bb=pos_stats,
            perf_breakdown_ms=perf_breakdown_ms,
            timestamp=time.time(),
        )

    def _refresh_snapshot(self, status: str) -> None:
        with self._lock:
            self._snapshot = self._make_snapshot(status=status)

    def _publish_actor_snapshot(self) -> None:
        start = time.perf_counter()
        self.actor_snapshot = self.model.clone_cpu()
        self._snapshot_version += 1
        if self._allow_pool_growth():
            self.checkpoint_pool.append(self.actor_snapshot)
            if len(self.checkpoint_pool) > self.config.max_checkpoint_pool:
                self.checkpoint_pool.pop(0)
        elif self.checkpoint_pool:
            # Keep the newest actor available without increasing pool size during freeze.
            self.checkpoint_pool[-1] = self.actor_snapshot
        else:
            self.checkpoint_pool = [self.actor_snapshot]
        self._sync_gpu_rollout_registry()
        self._cumulative_snapshot_time += time.perf_counter() - start

    def _allow_pool_growth(self) -> bool:
        freeze_target = int(
            max(0.0, float(getattr(self.config, "pool_growth_warmup_multiplier", 0.0)))
            * int(self.config.warmup_advantage_samples)
        )
        return len(self.advantage_buffer) >= freeze_target

    def _ensure_rollout_executor(self) -> Optional[ProcessPoolExecutor]:
        workers = self._effective_rollout_workers()
        if not self.config.parallel_rollouts or workers <= 1:
            return None
        current_workers = int(getattr(self._rollout_executor, "_max_workers", 0) or 0) if self._rollout_executor is not None else 0
        if self._rollout_executor is None or current_workers != workers:
            if self._rollout_executor is not None:
                self._rollout_executor.shutdown(wait=False, cancel_futures=False)
            self._rollout_executor = ProcessPoolExecutor(max_workers=workers)
        return self._rollout_executor

    def _effective_rollout_workers(self) -> int:
        configured = max(1, int(getattr(self.config, "rollout_workers", 1)))
        if not bool(getattr(self.config, "rollout_auto_scale_workers", False)):
            return configured
        cpu_count = int(os.cpu_count() or configured)
        reserve = max(0, int(getattr(self.config, "rollout_worker_reserve_cores", 0)))
        auto_target = max(1, cpu_count - reserve)
        auto_cap = max(0, int(getattr(self.config, "rollout_worker_auto_cap", 0)))
        if auto_cap > 0:
            auto_target = min(auto_target, auto_cap)
        return max(configured, auto_target)

    def _linear_cfr_weight(self) -> float:
        traversal_idx = self.traversals_completed + 1
        # Linear CFR weighting: w_t = t.
        # This gives early iterations influence that decays ~2 / (T * (T + 1)).
        return float(traversal_idx)

    def _sample_training_opponent_snapshot(self) -> object:
        if not self.config.use_training_pool or not self.checkpoint_pool:
            return self.actor_snapshot
        if len(self.checkpoint_pool) == 1:
            return self.checkpoint_pool[0]
        weights = np.arange(1, len(self.checkpoint_pool) + 1, dtype=np.float64)
        probs = weights / weights.sum()
        idx = int(self._seed_rng.choice(len(self.checkpoint_pool), p=probs))
        return self.checkpoint_pool[idx]

    def _synthetic_style_candidates(self) -> List[str]:
        configured = getattr(self.config, "synthetic_opponent_styles", SYNTHETIC_OPPONENT_STYLES)
        if not isinstance(configured, (list, tuple)):
            return list(SYNTHETIC_OPPONENT_STYLES)
        styles = [str(style).strip().lower() for style in configured if str(style).strip()]
        valid = [style for style in styles if style in SYNTHETIC_OPPONENT_STYLES]
        return valid or list(SYNTHETIC_OPPONENT_STYLES)

    def _sample_training_opponent_assignment(self) -> tuple[Optional[PokerDeepCFRNet], str, str, str]:
        styles = self._synthetic_style_candidates()
        synthetic_prob = float(max(0.0, min(1.0, getattr(self.config, "synthetic_opponent_probability", 0.0))))
        if styles and synthetic_prob > 1e-6 and float(self._seed_rng.random()) < synthetic_prob:
            style = str(self._seed_rng.choice(styles))
            return None, f"synthetic_{style}", "synthetic", style

        snapshot = self._sample_training_opponent_snapshot()
        return snapshot, self._opponent_profile_key_for_snapshot(snapshot), "snapshot", ""

    def _opponent_hold_span(self) -> int:
        return max(1, int(getattr(self.config, "opponent_hold_traversals", 1)))

    def _opponent_block_bounds(self, traversal_index: int) -> tuple[int, int]:
        hold = self._opponent_hold_span()
        start = int(traversal_index // hold) * hold
        end = start + hold
        return start, end

    def _opponent_assignment_for_traversal(
        self, traversal_index: int
    ) -> tuple[Optional[object], str, Dict[int, tuple], int, int, str, str]:
        block_start, block_end = self._opponent_block_bounds(traversal_index)
        keep_current = (
            bool(self._held_opponent_profile_key)
            and self._held_opponent_block_start == block_start
            and int(traversal_index) < int(self._held_opponent_block_end)
        )
        if not keep_current:
            snapshot, profile_key, policy_mode, synthetic_style = self._sample_training_opponent_assignment()
            self._held_opponent_snapshot = snapshot
            self._held_opponent_policy_mode = policy_mode
            self._held_synthetic_opponent_style = synthetic_style
            self._held_opponent_profile_key = profile_key
            self._held_opponent_block_start = block_start
            self._held_opponent_block_end = block_end

        snapshot = self._held_opponent_snapshot
        policy_mode = str(self._held_opponent_policy_mode or "snapshot")
        synthetic_style = str(self._held_synthetic_opponent_style or "")
        if snapshot is None and policy_mode != "synthetic":
            snapshot = self.actor_snapshot
        profile_key = str(
            self._held_opponent_profile_key
            or (self._opponent_profile_key_for_snapshot(snapshot) if snapshot is not None else "synthetic_nit")
        )
        opponent_profiles_by_seat = self._opponent_profiles_by_seat(profile_key)
        return snapshot, profile_key, opponent_profiles_by_seat, block_start, block_end, policy_mode, synthetic_style

    def _new_opponent_profile_record(self) -> Dict[str, float]:
        record: Dict[str, float] = {"hands_played_total": 0.0}
        for metric_name, _, _ in self._opponent_metric_specs():
            record[f"{metric_name}_hits_total"] = 0.0
            record[f"{metric_name}_opp_total"] = 0.0
            record[f"{metric_name}_short"] = 0.0
            record[f"{metric_name}_long"] = 0.0
        return record

    @staticmethod
    def _opponent_metric_specs() -> tuple:
        return (
            ("vpip", "vpip_counts", "preflop_opportunities"),
            ("pfr", "pfr_counts", "preflop_opportunities"),
            ("three_bet", "three_bet_counts", "faced_open_opportunities"),
            ("fold_to_open", "fold_vs_open_counts", "faced_open_opportunities"),
            ("fold_to_three_bet", "fold_vs_three_bet_counts", "faced_three_bet_opportunities"),
            ("call_open", "call_vs_open_counts", "faced_open_opportunities"),
            ("squeeze", "squeeze_counts", "squeeze_opportunities"),
            ("fold_to_cbet_flop", "fold_vs_cbet_flop_counts", "faced_cbet_flop_opportunities"),
            ("fold_to_cbet_turn", "fold_vs_cbet_turn_counts", "faced_cbet_turn_opportunities"),
            ("aggression", "aggression_counts", "aggression_opportunities"),
        )

    def _opponent_profile_key_for_snapshot(self, snapshot: object) -> str:
        if snapshot is self.actor_snapshot:
            return "actor_snapshot"
        return f"checkpoint_{id(snapshot)}"

    @staticmethod
    def _opponent_profile_key_for_seat(opponent_key: str, seat: int) -> str:
        return f"{str(opponent_key)}:seat_{int(seat)}"

    def _opponent_profile_record_from_store(self, store: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
        record = store.get(key)
        if record is None:
            record = self._new_opponent_profile_record()
            store[key] = record
        else:
            defaults = self._new_opponent_profile_record()
            for metric_key, metric_default in defaults.items():
                if metric_key not in record:
                    record[metric_key] = float(metric_default)
        return record

    def _opponent_profile_record(self, key: str) -> Dict[str, float]:
        return self._opponent_profile_record_from_store(self._opponent_profiles, key)

    def _opponent_profile_tuple_for_record(self, record: Dict[str, float]) -> tuple:
        confidence_scale = float(max(1.0, getattr(self.config, "opponent_profile_confidence_scale", 256.0)))
        hands_played_total = float(max(0.0, record.get("hands_played_total", 0.0)))
        confidence = float(max(0.0, min(1.0, hands_played_total / confidence_scale)))
        trend_mix = float(max(0.0, min(1.0, confidence * 1.5)))

        values: List[float] = []
        for metric_name, _, _ in self._opponent_metric_specs():
            short_rate = float(max(0.0, min(1.0, record.get(f"{metric_name}_short", 0.0))))
            long_rate = float(max(0.0, min(1.0, record.get(f"{metric_name}_long", 0.0))))
            stat_mean = (trend_mix * short_rate) + ((1.0 - trend_mix) * long_rate)
            values.append(float(max(0.0, min(1.0, stat_mean))))
        values.append(confidence)
        if len(values) != OPPONENT_PROFILE_PER_SLOT_DIM_V24:
            return tuple(OPPONENT_PROFILE_DEFAULT_SLOT_V24)
        return tuple(values)

    def _opponent_profile_tuple_for_seat_from_store(
        self, store: Dict[str, Dict[str, float]], opponent_key: str, seat: int
    ) -> tuple:
        seat_key = self._opponent_profile_key_for_seat(opponent_key, seat)
        record = self._opponent_profile_record_from_store(store, seat_key)
        return self._opponent_profile_tuple_for_record(record)

    def _opponent_profile_tuple_for_seat(self, opponent_key: str, seat: int) -> tuple:
        return self._opponent_profile_tuple_for_seat_from_store(self._opponent_profiles, opponent_key, seat)

    def _opponent_profiles_by_seat_from_store(
        self, store: Dict[str, Dict[str, float]], opponent_key: str
    ) -> Dict[int, tuple]:
        profiles: Dict[int, tuple] = {}
        for seat in range(max(1, int(self.config.num_players))):
            profiles[seat] = self._opponent_profile_tuple_for_seat_from_store(store, opponent_key, seat)
        return profiles

    def _opponent_profiles_by_seat(self, opponent_key: str) -> Dict[int, tuple]:
        return self._opponent_profiles_by_seat_from_store(self._opponent_profiles, opponent_key)

    def _update_opponent_profile_from_result_store(
        self, store: Dict[str, Dict[str, float]], key: str, result
    ) -> None:
        preflop_stats = getattr(result, "preflop_stats", None)
        if not isinstance(preflop_stats, dict) or not preflop_stats:
            return

        seats = range(max(0, int(self.config.num_players)))
        traverser = int(getattr(result, "traverser_seat", getattr(result, "hero_seat", 0)))

        hands_played = preflop_stats.get("hands_played", [])
        short_alpha = float(max(0.0, min(1.0, getattr(self.config, "opponent_profile_short_alpha", 0.25))))
        long_alpha = float(max(0.0, min(1.0, getattr(self.config, "opponent_profile_long_alpha", 0.05))))

        for seat in seats:
            if seat == traverser:
                continue
            seat_key = self._opponent_profile_key_for_seat(key, seat)
            record = self._opponent_profile_record_from_store(store, seat_key)
            if seat < len(hands_played):
                record["hands_played_total"] += float(hands_played[seat])

            for metric_name, hit_stat_key, opp_stat_key in self._opponent_metric_specs():
                hit_values = preflop_stats.get(hit_stat_key, [])
                opp_values = preflop_stats.get(opp_stat_key, [])
                hits = float(hit_values[seat]) if seat < len(hit_values) else 0.0
                opportunities = float(opp_values[seat]) if seat < len(opp_values) else 0.0
                record[f"{metric_name}_hits_total"] += hits
                record[f"{metric_name}_opp_total"] += opportunities
                if opportunities > 1e-9:
                    observed_rate = float(max(0.0, min(1.0, hits / opportunities)))
                    prev_short = float(record.get(f"{metric_name}_short", 0.0))
                    prev_long = float(record.get(f"{metric_name}_long", 0.0))
                    record[f"{metric_name}_short"] = ((1.0 - short_alpha) * prev_short) + (short_alpha * observed_rate)
                    record[f"{metric_name}_long"] = ((1.0 - long_alpha) * prev_long) + (long_alpha * observed_rate)

    def _update_opponent_profile_from_result(self, key: str, result) -> None:
        self._update_opponent_profile_from_result_store(self._opponent_profiles, key, result)

    def _should_save_checkpoint(self) -> bool:
        checkpoint_interval = max(0, int(getattr(self.config, "checkpoint_interval", 0)))
        if checkpoint_interval <= 0:
            return False
        if self._next_checkpoint_traverser_decisions <= 0:
            self._next_checkpoint_traverser_decisions = checkpoint_interval
        return self.traverser_decisions >= self._next_checkpoint_traverser_decisions

    def _mark_checkpoint_published(self) -> None:
        checkpoint_interval = max(0, int(getattr(self.config, "checkpoint_interval", 0)))
        if checkpoint_interval <= 0:
            self._next_checkpoint_traverser_decisions = 0
            return
        if self._next_checkpoint_traverser_decisions <= 0:
            self._next_checkpoint_traverser_decisions = checkpoint_interval
        while self.traverser_decisions >= self._next_checkpoint_traverser_decisions:
            self._next_checkpoint_traverser_decisions += checkpoint_interval

    def _gpu_rollout_enabled(self) -> bool:
        return bool(
            getattr(self.config, "gpu_rollout_inference", False)
            and self.device.type == "cuda"
            and self.config.parallel_rollouts
            and int(self.config.rollout_workers) > 1
        )

    def _maybe_start_gpu_rollout_service(self) -> None:
        if not self._gpu_rollout_enabled():
            return
        if self._gpu_rollout_service is not None:
            return
        service = GPURolloutInferenceService(
            device=str(self.device),
            max_batch_size=int(getattr(self.config, "gpu_inference_max_batch_size", 512)),
            max_wait_ms=float(getattr(self.config, "gpu_inference_max_wait_ms", 1.5)),
            model_cache_size=int(getattr(self.config, "gpu_inference_model_cache_size", 6)),
        )
        service.start()
        self._gpu_rollout_service = service

    @staticmethod
    def _snapshot_policy_family(snapshot: Optional[object]) -> str:
        return str(getattr(snapshot, "_policy_family", "v24") or "v24")

    @staticmethod
    def _gpu_model_key_for_snapshot(snapshot: Optional[object], actor_snapshot: Optional[object]) -> str:
        if snapshot is None or snapshot is actor_snapshot:
            return "actor_snapshot"
        return f"checkpoint_{id(snapshot)}"

    def _register_gpu_snapshot(self, snapshot: Optional[object]) -> str:
        model_key = self._gpu_model_key_for_snapshot(snapshot, self.actor_snapshot)
        if self._gpu_rollout_service is None:
            return model_key
        if self._snapshot_policy_family(snapshot) != "v24":
            return model_key
        if model_key != "actor_snapshot" and model_key in self._gpu_registered_model_keys:
            return model_key
        source = self.actor_snapshot if snapshot is None else snapshot
        self._gpu_rollout_service.register_snapshot(model_key, source)
        self._gpu_registered_model_keys.add(model_key)
        return model_key

    def _sync_gpu_rollout_registry(self) -> None:
        service = self._gpu_rollout_service
        if service is None:
            return
        active_keys = {"actor_snapshot"}
        service.register_snapshot("actor_snapshot", self.actor_snapshot)
        self._gpu_registered_model_keys.add("actor_snapshot")
        if bool(getattr(self.config, "gpu_rollout_remote_opponents", False)):
            for snapshot in self.checkpoint_pool:
                if snapshot is self.actor_snapshot or self._snapshot_policy_family(snapshot) != "v24":
                    continue
                active_keys.add(self._register_gpu_snapshot(snapshot))
        service.retain_model_keys(active_keys)
        self._gpu_registered_model_keys.intersection_update(active_keys)

    def _gpu_runtime_config(self, actor_key: str, opponent_key: str) -> Dict[str, object]:
        service = self._gpu_rollout_service
        if service is None:
            return {}
        return {
            "gpu_rollout_inference_enabled": True,
            "gpu_inference_endpoint": service.address,
            "gpu_inference_authkey": service.authkey,
            "gpu_actor_model_key": str(actor_key),
            "gpu_opponent_model_key": str(opponent_key),
            "gpu_remote_opponents": bool(getattr(self.config, "gpu_rollout_remote_opponents", False)),
        }

    def _masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = weight / torch.clamp(weight.mean(), min=1e-6)
        masked_sq = ((pred - target) ** 2) * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        per_row = masked_sq.sum(dim=1) / denom
        return (per_row * weight).mean()

    def _masked_policy_loss(self, logits: torch.Tensor, target_policy: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = weight / torch.clamp(weight.mean(), min=1e-6)
        masked_logits = logits.masked_fill(mask <= 0.5, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        per_row = -(target_policy * log_probs).sum(dim=1)
        return (per_row * weight).mean()

    def _train_regret_step(self) -> None:
        start = time.perf_counter()
        batch = self.advantage_buffer.sample(self.config.regret_batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        legal_masks = torch.as_tensor(batch["legal_masks"], dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        self.regret_optimizer.zero_grad(set_to_none=True)
        pred = self.model.forward_regret(states)
        loss = self._masked_mse_loss(pred, targets, legal_masks, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        self.regret_optimizer.step()

        self._last_regret_loss = float(loss.item())
        self.regret_steps += 1
        self.learner_steps += 1
        self._cumulative_regret_time += time.perf_counter() - start

    def _train_strategy_step(self) -> None:
        start = time.perf_counter()
        batch = self.strategy_buffer.sample(self.config.strategy_batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        legal_masks = torch.as_tensor(batch["legal_masks"], dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        self.strategy_optimizer.zero_grad(set_to_none=True)
        logits = self.model.forward_strategy(states)
        loss = self._masked_policy_loss(logits, targets, legal_masks, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        self.strategy_optimizer.step()

        self._last_strategy_loss = float(loss.item())
        self.strategy_steps += 1
        self.learner_steps += 1
        self._cumulative_strategy_time += time.perf_counter() - start

    def _train_exploit_step(self) -> None:
        start = time.perf_counter()
        batch = self.exploit_buffer.sample(self.config.exploit_batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        legal_masks = torch.as_tensor(batch["legal_masks"], dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        self.exploit_optimizer.zero_grad(set_to_none=True)
        logits = self.model.forward_exploit(states)
        loss = self._masked_policy_loss(logits, targets, legal_masks, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        self.exploit_optimizer.step()

        self._last_exploit_loss = float(loss.item())
        self.exploit_steps += 1
        self.learner_steps += 1
        self._cumulative_strategy_time += time.perf_counter() - start

    def _maybe_train(self) -> None:
        if len(self.advantage_buffer) < self.config.warmup_advantage_samples:
            return

        while self._adv_pending >= self.config.regret_update_interval:
            self._train_regret_step()
            self._adv_pending -= self.config.regret_update_interval
            if self.learner_steps % self.config.publish_snapshot_interval == 0:
                self._publish_actor_snapshot()
        while len(self.strategy_buffer) > 0 and self._strat_pending >= self.config.strategy_update_interval:
            self._train_strategy_step()
            self._strat_pending -= self.config.strategy_update_interval
            if self.learner_steps % self.config.publish_snapshot_interval == 0:
                self._publish_actor_snapshot()
        while len(self.exploit_buffer) > 0 and self._exploit_pending >= self.config.exploit_update_interval:
            self._train_exploit_step()
            self._exploit_pending -= self.config.exploit_update_interval
            if self.learner_steps % self.config.publish_snapshot_interval == 0:
                self._publish_actor_snapshot()

    def _consume_traversal_result(
        self,
        result,
        iter_elapsed: float,
        opponent_profile_key: Optional[str] = None,
        allow_training: bool = True,
    ) -> None:
        if opponent_profile_key:
            self._update_opponent_profile_from_result(opponent_profile_key, result)

        for key, value in result.perf_breakdown.items():
            if key in self._cumulative_rollout_detail:
                self._cumulative_rollout_detail[key] += float(value)
        self._cumulative_rollout_time += float(sum(result.perf_breakdown.values()))

        sample_weight = self._linear_cfr_weight()
        weighted_adv_samples = [
            (state, legal_mask, target, sample_weight) for state, legal_mask, target, _ in result.advantage_samples
        ]
        weighted_strategy_samples = [
            (state, legal_mask, target, sample_weight) for state, legal_mask, target, _ in result.strategy_samples
        ]
        weighted_exploit_samples = [
            (state, legal_mask, target, sample_weight * float(max(1e-3, weight)))
            for state, legal_mask, target, weight in getattr(result, "exploit_samples", [])
        ]

        self.advantage_buffer.extend(weighted_adv_samples)
        self.strategy_buffer.extend(weighted_strategy_samples)
        self.exploit_buffer.extend(weighted_exploit_samples)
        self._adv_pending += len(result.advantage_samples)
        self._strat_pending += len(result.strategy_samples)
        self._exploit_pending += len(getattr(result, "exploit_samples", []))

        self.traversals_completed += 1
        self.traverser_decisions += int(result.traverser_decisions)
        self.invalid_state_count += int(result.invalid_state_count)
        self.invalid_action_count += int(result.invalid_action_count)
        self._recent_utilities.append(float(result.unclipped_utility_bb))
        self._recent_vpip.append(1.0 if result.vpip else 0.0)
        self._recent_pfr.append(1.0 if result.pfr else 0.0)
        self._recent_three_bet.append(1.0 if result.three_bet else 0.0)
        self._position_profit_windows[result.traverser_seat].append(float(result.unclipped_utility_bb))
        for action_id, count in enumerate(result.action_counts.tolist()):
            if count > 0:
                self._recent_actions.extend([action_id] * int(count))

        if allow_training:
            self._maybe_train()
            if self._should_save_checkpoint():
                self._publish_actor_snapshot()
                self._mark_checkpoint_published()
        self._cumulative_work_time += float(iter_elapsed)
        self._recent_traversal_times.append(float(iter_elapsed))

    def _flush_pending_updates_after_rollouts(self) -> None:
        self._maybe_train()
        if self._should_save_checkpoint():
            self._publish_actor_snapshot()
            self._mark_checkpoint_published()

    def _run_sequential_traversals(self, num_traversals: int) -> None:
        for _ in range(max(1, int(num_traversals))):
            iter_start = time.perf_counter()
            traversal_index = int(self.traversals_completed)
            traverser_seat = traversal_index % self.config.num_players
            seed = int(self._seed_rng.integers(0, 2**31 - 1))
            (
                opponent_snapshot,
                opponent_profile_key,
                opponent_profiles_by_seat,
                _,
                _,
                opponent_policy_mode,
                synthetic_opponent_style,
            ) = self._opponent_assignment_for_traversal(traversal_index)
            self.config.opponent_profile = tuple(OPPONENT_PROFILE_DEFAULT_V24)
            self.config.opponent_profiles_by_seat = {
                int(seat): tuple(profile_values) for seat, profile_values in opponent_profiles_by_seat.items()
            }
            self.config.opponent_policy_mode = str(opponent_policy_mode or "snapshot")
            self.config.synthetic_opponent_style = str(synthetic_opponent_style or "")
            result = run_traversal(seed, traverser_seat, self.actor_snapshot, opponent_snapshot, self.config)
            iter_elapsed = time.perf_counter() - iter_start
            self._consume_traversal_result(result, iter_elapsed, opponent_profile_key=opponent_profile_key)

    def _run_parallel_traversals(self, num_traversals: int) -> None:
        executor = self._ensure_rollout_executor()
        if executor is None:
            self._run_sequential_traversals(num_traversals)
            return

        count = max(1, int(num_traversals))
        workers = max(1, int(self._effective_rollout_workers()))
        base_chunk = max(1, int(self.config.rollout_worker_chunk_size))
        target_tasks = max(1, workers)
        adaptive_chunk = int(math.ceil(count / float(target_tasks)))
        chunk_size = max(1, min(base_chunk, adaptive_chunk))
        use_gpu_rollout = self._gpu_rollout_enabled()
        if use_gpu_rollout:
            self._maybe_start_gpu_rollout_service()
            self._sync_gpu_rollout_registry()
        actor_state = {k: v.detach().cpu() for k, v in self.actor_snapshot.state_dict().items()}
        start_traversal = self.traversals_completed
        future_meta = {}
        chunk_start = 0
        chunk_idx = 0
        while chunk_start < count:
            traversal_index = int(start_traversal + chunk_start)
            (
                opponent_snapshot,
                opponent_profile_key,
                opponent_profiles_by_seat,
                _,
                block_end,
                opponent_policy_mode,
                synthetic_opponent_style,
            ) = self._opponent_assignment_for_traversal(traversal_index)
            hold_remaining = max(1, int(block_end - traversal_index))
            chunk_len = min(chunk_size, count - chunk_start, hold_remaining)
            seeds = [int(self._seed_rng.integers(0, 2**31 - 1)) for _ in range(chunk_len)]
            traverser_seats = [
                int((start_traversal + chunk_start + offset) % self.config.num_players) for offset in range(chunk_len)
            ]
            config_dict = asdict(self.config)
            config_dict["opponent_profile"] = tuple(OPPONENT_PROFILE_DEFAULT_V24)
            config_dict["opponent_profiles_by_seat"] = {
                int(seat): tuple(profile_values) for seat, profile_values in opponent_profiles_by_seat.items()
            }
            config_dict["opponent_policy_mode"] = str(opponent_policy_mode or "snapshot")
            config_dict["synthetic_opponent_style"] = str(synthetic_opponent_style or "")
            actor_model_key = self._gpu_model_key_for_snapshot(self.actor_snapshot, self.actor_snapshot)
            if use_gpu_rollout:
                self._register_gpu_snapshot(self.actor_snapshot)
            opponent_family = self._snapshot_policy_family(opponent_snapshot)
            config_dict["opponent_snapshot_family"] = opponent_family
            remote_opponents = bool(
                use_gpu_rollout
                and getattr(self.config, "gpu_rollout_remote_opponents", False)
                and opponent_family == "v24"
            )
            if opponent_snapshot is None or opponent_snapshot is self.actor_snapshot:
                opponent_state = None if remote_opponents else actor_state
                opponent_model_key = actor_model_key
            else:
                opponent_model_key = (
                    self._register_gpu_snapshot(opponent_snapshot) if use_gpu_rollout else self._gpu_model_key_for_snapshot(opponent_snapshot, self.actor_snapshot)
                )
                opponent_state = (
                    None if remote_opponents else {k: v.detach().cpu() for k, v in opponent_snapshot.state_dict().items()}
                )
            if use_gpu_rollout:
                config_dict.update(self._gpu_runtime_config(actor_model_key, opponent_model_key))
            config_dict["gpu_remote_opponents"] = remote_opponents
            opponent_signature = "self" if opponent_snapshot is self.actor_snapshot else str(id(opponent_snapshot))
            actor_cache_signature = (
                f"remote_actor:{actor_model_key}:{config_dict.get('gpu_inference_endpoint')}"
                if use_gpu_rollout
                else f"cpu_actor:{self._snapshot_version}:{self.config.state_dim}:{self.config.hidden_dim}:{self.config.action_count}"
            )
            if str(opponent_policy_mode or "snapshot") != "snapshot":
                opponent_cache_signature = (
                    f"opponent_mode:{opponent_policy_mode}:{synthetic_opponent_style}:"
                    f"{int(bool(config_dict.get('quantize_local_opponent_models', False)))}"
                )
            elif opponent_snapshot is None or opponent_snapshot is self.actor_snapshot:
                opponent_cache_signature = (
                    f"opponent_self:{self._snapshot_version}:{self.config.state_dim}:{self.config.hidden_dim}:"
                    f"{self.config.action_count}:{int(bool(config_dict.get('quantize_local_opponent_models', False)))}"
                )
            else:
                opponent_cache_signature = (
                    f"opponent_checkpoint:{opponent_signature}:{int(bool(config_dict.get('quantize_local_opponent_models', False)))}"
                )
            config_dict["actor_cache_signature"] = actor_cache_signature
            config_dict["opponent_cache_signature"] = opponent_cache_signature
            signature = (
                f"{self._snapshot_version}:"
                f"{self.config.state_dim}:{self.config.hidden_dim}:{self.config.action_count}:"
                f"{opponent_signature}:"
                f"{config_dict['opponent_policy_mode']}:{config_dict['synthetic_opponent_style']}"
            )
            future = executor.submit(
                run_traversal_batch_mp,
                seeds,
                traverser_seats,
                actor_state,
                opponent_state,
                config_dict,
                signature,
            )
            future_meta[future] = {
                "chunk_len": chunk_len,
                "opponent_profile_key": opponent_profile_key,
            }
            chunk_start += chunk_len
            chunk_idx += 1

        for future in as_completed(list(future_meta.keys())):
            meta = future_meta[future]
            chunk_len = int(meta.get("chunk_len", 0))
            opponent_profile_key = str(meta.get("opponent_profile_key", "") or "")
            results = future.result()
            if not results:
                continue
            for result in results:
                iter_elapsed = max(1e-6, float(sum(result.perf_breakdown.values())))
                self._consume_traversal_result(
                    result,
                    iter_elapsed,
                    opponent_profile_key=opponent_profile_key,
                    allow_training=not use_gpu_rollout,
                )
        if use_gpu_rollout:
            self._flush_pending_updates_after_rollouts()

    def train_for_traversals(self, num_traversals: int) -> TrainingSnapshot:
        status = "Training"
        requested = max(1, int(num_traversals))
        workers = self._effective_rollout_workers()
        min_parallel = max(2, workers * 2)
        if self.config.parallel_rollouts and workers > 1 and requested >= min_parallel:
            self._run_parallel_traversals(requested)
        else:
            self._run_sequential_traversals(requested)

        self._refresh_snapshot(status=status)
        return self.get_snapshot()

    def train_forever(self, stop_event, pause_event, snapshot_callback=None) -> None:
        self._refresh_snapshot(status="Training")
        while not stop_event.is_set():
            if pause_event.is_set():
                self._refresh_snapshot(status="Paused")
                time.sleep(0.1)
                continue
            snapshot = self.train_for_traversals(self.config.traversals_per_chunk)
            if snapshot_callback is not None:
                snapshot_callback(snapshot)
        self._refresh_snapshot(status="Stopped")

    def _synthetic_style_for_eval_hand(self, mode: str, hand_idx: int, total_hands: int) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in SYNTHETIC_OPPONENT_STYLES:
            return normalized
        if normalized != "leak_pool":
            return ""

        styles = self._synthetic_style_candidates()
        if not styles:
            return ""
        hands_per_style = max(1, int(math.ceil(max(1, int(total_hands)) / float(len(styles)))))
        style_idx = min(len(styles) - 1, int(hand_idx // hands_per_style))
        return str(styles[style_idx])

    def _evaluate(self, mode: str, num_hands: int) -> EvaluationReport:
        start = time.time()
        action_hist = np.zeros(self.config.action_count, dtype=np.int64)
        total_profit = 0.0
        wins = 0
        illegal_action_count = 0
        position_profit = {seat: [] for seat in range(self.config.num_players)}
        vpip_count = 0
        pfr_count = 0
        three_bet_count = 0
        vpip_attempts = {seat: 0 for seat in range(self.config.num_players)}
        pfr_attempts = {seat: 0 for seat in range(self.config.num_players)}
        three_bet_attempts = {seat: 0 for seat in range(self.config.num_players)}
        vpip_opportunities = {seat: 0 for seat in range(self.config.num_players)}
        hand_counts = np.zeros((13, 13), dtype=np.int64)
        hand_vpip_hits = np.zeros((13, 13), dtype=np.int64)
        hand_pfr_hits = np.zeros((13, 13), dtype=np.int64)
        hand_three_bet_hits = np.zeros((13, 13), dtype=np.int64)
        position_hand_counts = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_vpip_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_pfr_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}
        position_hand_three_bet_hits = {seat: np.zeros((13, 13), dtype=np.int64) for seat in range(self.config.num_players)}

        eval_config = replace(self.config)
        eval_config.opponent_profile = OPPONENT_PROFILE_DEFAULT_V24
        eval_config.opponent_profiles_by_seat = {}
        eval_config.opponent_policy_mode = "snapshot"
        eval_config.synthetic_opponent_style = ""
        eval_mode = "heuristics"
        checkpoint_pool: tuple = tuple()
        if mode == "checkpoints":
            eval_mode = "checkpoints"
            checkpoint_pool = tuple(self.checkpoint_pool[:-1] or [self.actor_snapshot])
        elif mode == "v21_table":
            if self._v21_seed_template is None:
                raise RuntimeError(
                    "v21 evaluation checkpoint could not be loaded. "
                    "Check DeepCFRConfig.seed_v21_checkpoint_path."
                )
            eval_mode = "checkpoints"
            checkpoint_pool = (self._v21_seed_template.clone_cpu(),)
        elif str(mode).strip().lower() in SYNTHETIC_OPPONENT_STYLES or str(mode).strip().lower() == "leak_pool":
            eval_mode = "synthetic"

        eval_config.evaluation_mode = eval_mode
        eval_config.checkpoint_pool = checkpoint_pool
        total_eval_hands = max(1, int(num_hands))
        eval_opponent_profiles: Dict[str, Dict[str, float]] = {}

        for hand_idx in range(total_eval_hands):
            eval_config.eval_hero_seat = hand_idx % self.config.num_players
            synthetic_style = self._synthetic_style_for_eval_hand(mode, hand_idx, total_eval_hands)
            if synthetic_style:
                eval_key = f"synthetic_{synthetic_style}"
                eval_config.evaluation_mode = "synthetic"
                eval_config.synthetic_opponent_style = synthetic_style
                eval_config.opponent_profiles_by_seat = self._opponent_profiles_by_seat_from_store(
                    eval_opponent_profiles,
                    eval_key,
                )
            else:
                eval_key = ""
                eval_config.evaluation_mode = eval_mode
                eval_config.synthetic_opponent_style = ""
                eval_config.opponent_profiles_by_seat = {}
            seed = int(self._seed_rng.integers(0, 2**31 - 1))
            hand = run_policy_hand(seed, self.actor_snapshot, eval_config)
            if eval_key:
                self._update_opponent_profile_from_result_store(eval_opponent_profiles, eval_key, hand)
            total_profit += float(hand.hero_profit_bb)
            action_hist += hand.action_counts
            illegal_action_count += int(hand.illegal_action_count)
            wins += 1 if hand.win else 0
            position_profit[hand.hero_seat].append(float(hand.hero_profit_bb))
            vpip_count += 1 if hand.vpip else 0
            pfr_count += 1 if hand.pfr else 0
            three_bet_count += 1 if hand.three_bet else 0
            vpip_attempts[hand.hero_seat] += 1 if hand.vpip else 0
            pfr_attempts[hand.hero_seat] += 1 if hand.pfr else 0
            three_bet_attempts[hand.hero_seat] += 1 if hand.three_bet else 0
            vpip_opportunities[hand.hero_seat] += 1
            if hand.hero_hand_key:
                key = hand.hero_hand_key
                if len(key) >= 2:
                    rank_a = key[0]
                    rank_b = key[1]
                    if rank_a in RANK_ORDER_HIGH_TO_LOW and rank_b in RANK_ORDER_HIGH_TO_LOW:
                        idx_a = RANK_ORDER_HIGH_TO_LOW.index(rank_a)
                        idx_b = RANK_ORDER_HIGH_TO_LOW.index(rank_b)
                        if len(key) == 2:
                            row, col = idx_a, idx_b
                        elif key.endswith("s"):
                            row, col = min(idx_a, idx_b), max(idx_a, idx_b)
                        else:
                            row, col = max(idx_a, idx_b), min(idx_a, idx_b)
                        seat = int(hand.hero_seat)
                        hand_counts[row, col] += 1
                        position_hand_counts[seat][row, col] += 1
                        if hand.vpip:
                            hand_vpip_hits[row, col] += 1
                            position_hand_vpip_hits[seat][row, col] += 1
                        if hand.pfr:
                            hand_pfr_hits[row, col] += 1
                            position_hand_pfr_hits[seat][row, col] += 1
                        if hand.three_bet:
                            hand_three_bet_hits[row, col] += 1
                            position_hand_three_bet_hits[seat][row, col] += 1

        position_stats = {}
        for seat, values in position_profit.items():
            position_stats[POSITION_NAMES_V21[seat]] = float(np.mean(values)) if values else 0.0

        vpip_pos_stats = {}
        pfr_pos_stats = {}
        three_bet_pos_stats = {}
        for seat in range(self.config.num_players):
            denom = max(1, vpip_opportunities[seat])
            vpip_pos_stats[POSITION_NAMES_V21[seat]] = float(vpip_attempts[seat] / denom)
            pfr_pos_stats[POSITION_NAMES_V21[seat]] = float(pfr_attempts[seat] / denom)
            three_bet_pos_stats[POSITION_NAMES_V21[seat]] = float(three_bet_attempts[seat] / denom)

        vpip_hand_grid = np.zeros((13, 13), dtype=np.float32)
        pfr_hand_grid = np.zeros((13, 13), dtype=np.float32)
        three_bet_hand_grid = np.zeros((13, 13), dtype=np.float32)
        nonzero = hand_counts > 0
        vpip_hand_grid[nonzero] = hand_vpip_hits[nonzero] / hand_counts[nonzero]
        pfr_hand_grid[nonzero] = hand_pfr_hits[nonzero] / hand_counts[nonzero]
        three_bet_hand_grid[nonzero] = hand_three_bet_hits[nonzero] / hand_counts[nonzero]

        vpip_hand_grid_by_position: Dict[str, List[List[float]]] = {}
        pfr_hand_grid_by_position: Dict[str, List[List[float]]] = {}
        three_bet_hand_grid_by_position: Dict[str, List[List[float]]] = {}
        for seat in range(self.config.num_players):
            seat_name = POSITION_NAMES_V21[seat]
            seat_counts = position_hand_counts[seat]
            seat_nonzero = seat_counts > 0

            seat_vpip = np.zeros((13, 13), dtype=np.float32)
            seat_pfr = np.zeros((13, 13), dtype=np.float32)
            seat_three_bet = np.zeros((13, 13), dtype=np.float32)
            seat_vpip[seat_nonzero] = position_hand_vpip_hits[seat][seat_nonzero] / seat_counts[seat_nonzero]
            seat_pfr[seat_nonzero] = position_hand_pfr_hits[seat][seat_nonzero] / seat_counts[seat_nonzero]
            seat_three_bet[seat_nonzero] = position_hand_three_bet_hits[seat][seat_nonzero] / seat_counts[seat_nonzero]

            vpip_hand_grid_by_position[seat_name] = seat_vpip.tolist()
            pfr_hand_grid_by_position[seat_name] = seat_pfr.tolist()
            three_bet_hand_grid_by_position[seat_name] = seat_three_bet.tolist()

        return EvaluationReport(
            mode=mode,
            hands=max(1, int(num_hands)),
            avg_profit_bb=float(total_profit / max(1, int(num_hands))),
            win_rate=float(wins / max(1, int(num_hands))),
            vpip=float(vpip_count / max(1, int(num_hands))),
            pfr=float(pfr_count / max(1, int(num_hands))),
            three_bet=float(three_bet_count / max(1, int(num_hands))),
            illegal_action_count=illegal_action_count,
            runtime_seconds=float(time.time() - start),
            action_histogram=action_hist.astype(int).tolist(),
            position_avg_profit_bb=position_stats,
            vpip_by_position=vpip_pos_stats,
            pfr_by_position=pfr_pos_stats,
            three_bet_by_position=three_bet_pos_stats,
            vpip_hand_grid=vpip_hand_grid.tolist(),
            pfr_hand_grid=pfr_hand_grid.tolist(),
            three_bet_hand_grid=three_bet_hand_grid.tolist(),
            vpip_hand_grid_by_position=vpip_hand_grid_by_position,
            pfr_hand_grid_by_position=pfr_hand_grid_by_position,
            three_bet_hand_grid_by_position=three_bet_hand_grid_by_position,
        )

    def evaluate_vs_heuristics(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("heuristics", num_hands)

    def evaluate_vs_checkpoint_pool(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("checkpoints", num_hands)

    def evaluate_vs_v21_table(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("v21_table", num_hands)

    def evaluate_vs_leak_pool(self, num_hands: int) -> EvaluationReport:
        return self._evaluate("leak_pool", num_hands)

    def evaluate_vs_synthetic_style(self, style: str, num_hands: int) -> EvaluationReport:
        normalized = str(style or "").strip().lower()
        if normalized not in SYNTHETIC_OPPONENT_STYLES:
            raise ValueError(f"Unknown synthetic opponent style: {style}")
        return self._evaluate(normalized, num_hands)

    def evaluate_exploit_suite(self, num_hands: int) -> ExploitSuiteReport:
        hands = max(1, int(num_hands))
        leak_reports: Dict[str, EvaluationReport] = {}
        for style in self._synthetic_style_candidates():
            leak_reports[str(style)] = self.evaluate_vs_synthetic_style(style, hands)

        robust_reports: Dict[str, EvaluationReport] = {
            "heuristics": self.evaluate_vs_heuristics(hands),
            "checkpoints": self.evaluate_vs_checkpoint_pool(hands),
        }
        if self._v21_seed_template is not None:
            robust_reports["v21_table"] = self.evaluate_vs_v21_table(hands)

        leak_profit = [report.avg_profit_bb for report in leak_reports.values()]
        robust_profit = [report.avg_profit_bb for report in robust_reports.values()]
        leak_win_rates = [report.win_rate for report in leak_reports.values()]
        robust_win_rates = [report.win_rate for report in robust_reports.values()]

        return ExploitSuiteReport(
            hands_per_mode=hands,
            leak_reports=leak_reports,
            robust_reports=robust_reports,
            avg_leak_profit_bb=float(np.mean(leak_profit)) if leak_profit else 0.0,
            avg_robust_profit_bb=float(np.mean(robust_profit)) if robust_profit else 0.0,
            avg_leak_win_rate=float(np.mean(leak_win_rates)) if leak_win_rates else 0.0,
            avg_robust_win_rate=float(np.mean(robust_win_rates)) if robust_win_rates else 0.0,
        )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "regret_optimizer": self.regret_optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "exploit_optimizer": self.exploit_optimizer.state_dict(),
            "config": asdict(self.config),
            "opponent_profiles": copy.deepcopy(self._opponent_profiles),
            "metrics": {
                "traversals_completed": self.traversals_completed,
                "traverser_decisions": self.traverser_decisions,
                "learner_steps": self.learner_steps,
                "regret_steps": self.regret_steps,
                "strategy_steps": self.strategy_steps,
                "exploit_steps": self.exploit_steps,
                "invalid_state_count": self.invalid_state_count,
                "invalid_action_count": self.invalid_action_count,
                "last_regret_loss": self._last_regret_loss,
                "last_strategy_loss": self._last_strategy_loss,
                "last_exploit_loss": self._last_exploit_loss,
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        load_compatible_state_dict(self.model, payload["model_state_dict"])
        self.regret_optimizer.load_state_dict(payload["regret_optimizer"])
        if "strategy_optimizer" in payload:
            self.strategy_optimizer.load_state_dict(payload["strategy_optimizer"])
        if "exploit_optimizer" in payload:
            self.exploit_optimizer.load_state_dict(payload["exploit_optimizer"])

        metrics = payload.get("metrics", {})
        self.traversals_completed = int(metrics.get("traversals_completed", self.traversals_completed))
        self.traverser_decisions = int(metrics.get("traverser_decisions", self.traverser_decisions))
        self.learner_steps = int(metrics.get("learner_steps", self.learner_steps))
        self.regret_steps = int(metrics.get("regret_steps", self.regret_steps))
        self.strategy_steps = int(metrics.get("strategy_steps", self.strategy_steps))
        self.exploit_steps = int(metrics.get("exploit_steps", self.exploit_steps))
        self.invalid_state_count = int(metrics.get("invalid_state_count", self.invalid_state_count))
        self.invalid_action_count = int(metrics.get("invalid_action_count", self.invalid_action_count))
        self._last_regret_loss = float(metrics.get("last_regret_loss", self._last_regret_loss))
        self._last_strategy_loss = float(metrics.get("last_strategy_loss", self._last_strategy_loss))
        self._last_exploit_loss = float(metrics.get("last_exploit_loss", self._last_exploit_loss))
        loaded_profiles = payload.get("opponent_profiles", {})
        if isinstance(loaded_profiles, dict):
            cleaned_profiles: Dict[str, Dict[str, float]] = {}
            for key, value in loaded_profiles.items():
                if not isinstance(value, dict):
                    continue
                cleaned_profiles[str(key)] = {
                    str(metric): float(metric_value) for metric, metric_value in value.items()
                }
            self._opponent_profiles = cleaned_profiles

        self.actor_snapshot = self.model.clone_cpu()
        checkpoint_interval = max(0, int(getattr(self.config, "checkpoint_interval", 0)))
        self._next_checkpoint_traverser_decisions = checkpoint_interval if checkpoint_interval > 0 else 0
        while checkpoint_interval > 0 and self._next_checkpoint_traverser_decisions <= self.traverser_decisions:
            self._next_checkpoint_traverser_decisions += checkpoint_interval
        self._reset_checkpoint_pool()
        self._sync_gpu_rollout_registry()
        self._refresh_snapshot(status="Loaded")

    def get_snapshot(self) -> TrainingSnapshot:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def shutdown(self) -> None:
        executor = self._rollout_executor
        self._rollout_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)
        if self._gpu_rollout_service is not None:
            self._gpu_rollout_service.stop()
            self._gpu_rollout_service = None
        self._gpu_registered_model_keys.clear()


# Backward-compatible alias for any legacy imports.
DeepCFRTrainerV21 = DeepCFRTrainerV24


