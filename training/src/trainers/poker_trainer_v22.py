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
for path in (FEATURES_DIR, MODELS_DIR, WORKERS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from poker_model_v22 import AdvantageBuffer, PokerDeepCFRNet, StrategyBuffer
from poker_state_v22 import ACTION_COUNT_V21, ACTION_NAMES_V21, POSITION_NAMES_V21, STATE_DIM_V21
from poker_worker_v22 import run_policy_hand, run_traversal, run_traversal_batch_mp

RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"


@dataclass
class DeepCFRConfig:
    num_players: int = 6
    small_blind: int = 5
    big_blind: int = 10
    state_dim: int = STATE_DIM_V21
    action_count: int = ACTION_COUNT_V21
    hidden_dim: int = 1024 # clearly needs at least 1024 to perform well, should try more
    full_branch_depth: int = 1
    max_branch_actions: int = 0
    learning_rate: float = 1e-4
    regret_batch_size: int = 1024
    strategy_batch_size: int = 2048
    advantage_capacity: int = 500_000
    strategy_capacity: int = 1_000_000
    warmup_advantage_samples: int = 32768
    regret_update_interval: int = 1024
    strategy_update_interval: int = 2048
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
    rollout_workers: int = 10 
    rollout_worker_chunk_size: int = 16
    parallel_rollouts: bool = True


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


class DeepCFRTrainerV21:
    def __init__(self, config: Optional[DeepCFRConfig] = None):
        self.config = config or DeepCFRConfig()
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
        self.regret_optimizer = torch.optim.Adam(regret_params, lr=self.config.learning_rate)
        self.strategy_optimizer = torch.optim.Adam(strategy_params, lr=self.config.learning_rate)

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

        self.actor_snapshot = self.model.clone_cpu()
        self._v21_seed_template = self._load_v21_seed_template()
        self.checkpoint_pool: List[PokerDeepCFRNet] = []
        self._reset_checkpoint_pool()

        self._lock = threading.Lock()
        self._seed_rng = np.random.default_rng(int(time.time()))
        self._start_time = time.time()

        self.traversals_completed = 0
        self.traverser_decisions = 0
        self.learner_steps = 0
        self.regret_steps = 0
        self.strategy_steps = 0
        self.invalid_state_count = 0
        self.invalid_action_count = 0

        self._adv_pending = 0
        self._strat_pending = 0
        self._last_regret_loss = 0.0
        self._last_strategy_loss = 0.0

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

    def _load_v21_seed_template(self) -> Optional[PokerDeepCFRNet]:
        path = str(getattr(self.config, "seed_v21_checkpoint_path", "") or "").strip()
        if not path:
            return None
        if not os.path.isfile(path):
            print(f"[v22] Seed checkpoint not found: {path}")
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
            seed_model = PokerDeepCFRNet(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                init_weights=False,
            )
            seed_model.load_state_dict(state_dict, strict=True)
            seed_model.eval()
            seed_model.to("cpu")
            for param in seed_model.parameters():
                param.requires_grad_(False)
            return seed_model
        except Exception as exc:
            print(f"[v22] Failed to load v21 seed checkpoint from '{path}': {exc}")
            return None

    def _reset_checkpoint_pool(self) -> None:
        self.checkpoint_pool = []
        seed_count = max(0, int(getattr(self.config, "seed_v21_count", 0)))
        if self._v21_seed_template is not None and seed_count > 0:
            for _ in range(seed_count):
                self.checkpoint_pool.append(self._v21_seed_template.clone_cpu())
        self.checkpoint_pool.append(self.actor_snapshot)
        if len(self.checkpoint_pool) > self.config.max_checkpoint_pool:
            self.checkpoint_pool = self.checkpoint_pool[-self.config.max_checkpoint_pool :]

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
        self._cumulative_snapshot_time += time.perf_counter() - start

    def _allow_pool_growth(self) -> bool:
        freeze_target = int(
            max(0.0, float(getattr(self.config, "pool_growth_warmup_multiplier", 0.0)))
            * int(self.config.warmup_advantage_samples)
        )
        return len(self.advantage_buffer) >= freeze_target

    def _ensure_rollout_executor(self) -> Optional[ProcessPoolExecutor]:
        if not self.config.parallel_rollouts or int(self.config.rollout_workers) <= 1:
            return None
        if self._rollout_executor is None:
            self._rollout_executor = ProcessPoolExecutor(max_workers=int(self.config.rollout_workers))
        return self._rollout_executor

    def _linear_cfr_weight(self) -> float:
        traversal_idx = self.traversals_completed + 1
        # Linear CFR weighting: w_t = t.
        # This gives early iterations influence that decays ~2 / (T * (T + 1)).
        return float(traversal_idx)

    def _sample_training_opponent_snapshot(self) -> PokerDeepCFRNet:
        if not self.config.use_training_pool or not self.checkpoint_pool:
            return self.actor_snapshot
        if len(self.checkpoint_pool) == 1:
            return self.checkpoint_pool[0]
        weights = np.arange(1, len(self.checkpoint_pool) + 1, dtype=np.float64)
        probs = weights / weights.sum()
        idx = int(self._seed_rng.choice(len(self.checkpoint_pool), p=probs))
        return self.checkpoint_pool[idx]

    def _should_save_checkpoint(self) -> bool:
        return self.traverser_decisions > 0 and (self.traverser_decisions % self.config.checkpoint_interval) == 0

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

    def _consume_traversal_result(self, result, iter_elapsed: float) -> None:
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

        self.advantage_buffer.extend(weighted_adv_samples)
        self.strategy_buffer.extend(weighted_strategy_samples)
        self._adv_pending += len(result.advantage_samples)
        self._strat_pending += len(result.strategy_samples)

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

        self._maybe_train()
        if self._should_save_checkpoint():
            self._publish_actor_snapshot()
        self._cumulative_work_time += float(iter_elapsed)
        self._recent_traversal_times.append(float(iter_elapsed))

    def _run_sequential_traversals(self, num_traversals: int) -> None:
        for _ in range(max(1, int(num_traversals))):
            iter_start = time.perf_counter()
            traverser_seat = self.traversals_completed % self.config.num_players
            seed = int(self._seed_rng.integers(0, 2**31 - 1))
            opponent_snapshot = self._sample_training_opponent_snapshot()
            result = run_traversal(seed, traverser_seat, self.actor_snapshot, opponent_snapshot, self.config)
            iter_elapsed = time.perf_counter() - iter_start
            self._consume_traversal_result(result, iter_elapsed)

    def _run_parallel_traversals(self, num_traversals: int) -> None:
        executor = self._ensure_rollout_executor()
        if executor is None:
            self._run_sequential_traversals(num_traversals)
            return

        count = max(1, int(num_traversals))
        workers = max(1, int(self.config.rollout_workers))
        base_chunk = max(1, int(self.config.rollout_worker_chunk_size))
        target_tasks = max(1, workers)
        adaptive_chunk = int(math.ceil(count / float(target_tasks)))
        chunk_size = max(1, min(base_chunk, adaptive_chunk))
        actor_state = {k: v.detach().cpu() for k, v in self.actor_snapshot.state_dict().items()}
        config_dict = asdict(self.config)
        start_traversal = self.traversals_completed
        future_meta = {}
        chunk_count = int(math.ceil(count / float(chunk_size)))

        for chunk_idx in range(chunk_count):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(count, chunk_start + chunk_size)
            chunk_len = chunk_end - chunk_start
            seeds = [int(self._seed_rng.integers(0, 2**31 - 1)) for _ in range(chunk_len)]
            traverser_seats = [
                int((start_traversal + chunk_start + offset) % self.config.num_players) for offset in range(chunk_len)
            ]
            opponent_snapshot = self._sample_training_opponent_snapshot()
            if opponent_snapshot is self.actor_snapshot:
                opponent_state = None
            else:
                opponent_state = {k: v.detach().cpu() for k, v in opponent_snapshot.state_dict().items()}
            opponent_signature = "self" if opponent_snapshot is self.actor_snapshot else str(id(opponent_snapshot))
            signature = (
                f"{self._snapshot_version}:"
                f"{self.config.state_dim}:{self.config.hidden_dim}:{self.config.action_count}:"
                f"{opponent_signature}"
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
            future_meta[future] = chunk_len

        for future in as_completed(list(future_meta.keys())):
            chunk_len = future_meta[future]
            results = future.result()
            if not results:
                continue
            for result in results:
                iter_elapsed = max(1e-6, float(sum(result.perf_breakdown.values())))
                self._consume_traversal_result(result, iter_elapsed)

    def train_for_traversals(self, num_traversals: int) -> TrainingSnapshot:
        status = "Training"
        requested = max(1, int(num_traversals))
        min_parallel = max(2, int(self.config.rollout_workers) * 2)
        if self.config.parallel_rollouts and self.config.rollout_workers > 1 and requested >= min_parallel:
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

        eval_config.evaluation_mode = eval_mode
        eval_config.checkpoint_pool = checkpoint_pool

        for hand_idx in range(max(1, int(num_hands))):
            eval_config.eval_hero_seat = hand_idx % self.config.num_players
            seed = int(self._seed_rng.integers(0, 2**31 - 1))
            hand = run_policy_hand(seed, self.actor_snapshot, eval_config)
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

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "regret_optimizer": self.regret_optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "config": asdict(self.config),
            "metrics": {
                "traversals_completed": self.traversals_completed,
                "traverser_decisions": self.traverser_decisions,
                "learner_steps": self.learner_steps,
                "regret_steps": self.regret_steps,
                "strategy_steps": self.strategy_steps,
                "invalid_state_count": self.invalid_state_count,
                "invalid_action_count": self.invalid_action_count,
                "last_regret_loss": self._last_regret_loss,
                "last_strategy_loss": self._last_strategy_loss,
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.regret_optimizer.load_state_dict(payload["regret_optimizer"])
        self.strategy_optimizer.load_state_dict(payload["strategy_optimizer"])

        metrics = payload.get("metrics", {})
        self.traversals_completed = int(metrics.get("traversals_completed", self.traversals_completed))
        self.traverser_decisions = int(metrics.get("traverser_decisions", self.traverser_decisions))
        self.learner_steps = int(metrics.get("learner_steps", self.learner_steps))
        self.regret_steps = int(metrics.get("regret_steps", self.regret_steps))
        self.strategy_steps = int(metrics.get("strategy_steps", self.strategy_steps))
        self.invalid_state_count = int(metrics.get("invalid_state_count", self.invalid_state_count))
        self.invalid_action_count = int(metrics.get("invalid_action_count", self.invalid_action_count))
        self._last_regret_loss = float(metrics.get("last_regret_loss", self._last_regret_loss))
        self._last_strategy_loss = float(metrics.get("last_strategy_loss", self._last_strategy_loss))

        self.actor_snapshot = self.model.clone_cpu()
        self._reset_checkpoint_pool()
        self._refresh_snapshot(status="Loaded")

    def get_snapshot(self) -> TrainingSnapshot:
        with self._lock:
            return copy.deepcopy(self._snapshot)
