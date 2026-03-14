import math
import sys
import warnings
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

warnings.filterwarnings("ignore", message="optree is installed but the version is too old.*", category=FutureWarning)
import torch
from pokerkit import Card

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for rel in ("features", "models", "workers", "trainers"):
    path = str(SRC_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from poker_state_v24 import ACTION_CALL, ACTION_CHECK, ACTION_COUNT_V21, ACTION_FOLD
import poker_trainer_v24 as trainer_v24
from poker_trainer_v24 import DeepCFRConfig, DeepCFRTrainerV24
from tabular_policy_v24 import TabularNode, average_policy, regret_matching, uniform_legal_policy
import poker_worker_v24 as worker_v24

pytestmark = [
    pytest.mark.filterwarnings("ignore:A card being dealt .* is not recommended to be dealt.*:UserWarning"),
]


class FakePot:
    def __init__(self, amount: float):
        self.amount = float(amount)


class FakeState:
    def __init__(
        self,
        *,
        actor_index: int,
        hole_cards,
        board_cards,
        stacks,
        bets,
        pot_amount: float,
        can_fold: bool,
        can_check_or_call: bool,
        can_raise: bool,
        min_raise: int = 0,
        max_raise: int = 0,
    ):
        self.actor_index = int(actor_index)
        self.hole_cards = hole_cards
        self.board_cards = board_cards
        self.stacks = list(stacks)
        self.bets = list(bets)
        self.pots = [FakePot(pot_amount)]
        self._can_fold = bool(can_fold)
        self._can_check_or_call = bool(can_check_or_call)
        self._can_raise = bool(can_raise)
        self.min_completion_betting_or_raising_to_amount = int(min_raise)
        self.max_completion_betting_or_raising_to_amount = int(max_raise)

    def can_fold(self):
        return self._can_fold

    def can_check_or_call(self):
        return self._can_check_or_call

    def can_complete_bet_or_raise_to(self):
        return self._can_raise


def cards(spec: str):
    return list(Card.parse(spec))


def make_ctx(
    *,
    current_street: int,
    preflop_raise_count: int = 0,
    preflop_call_count: int = 0,
    street_raise_count: int = 0,
    last_aggressor=None,
    preflop_last_raiser=None,
):
    return SimpleNamespace(
        big_blind=10,
        small_blind=5,
        current_street=int(current_street),
        street_raise_count=int(street_raise_count),
        preflop_raise_count=int(preflop_raise_count),
        preflop_call_count=int(preflop_call_count),
        preflop_last_raiser=preflop_last_raiser,
        last_aggressor=last_aggressor,
        contributions=[0.0] * 6,
        in_hand=[True] * 6,
    )


def make_hand_result(
    *,
    hero_seat: int,
    hero_hand_key: str | None,
    vpip: bool = False,
    rfi_attempt: bool = False,
    rfi_opportunity: bool = False,
    pfr: bool = False,
    three_bet: bool = False,
):
    zeros = np.zeros(ACTION_COUNT_V21, dtype=np.int64)
    return SimpleNamespace(
        hero_profit_bb=0.0,
        hero_seat=int(hero_seat),
        action_counts=zeros.copy(),
        preflop_action_counts=zeros.copy(),
        postflop_action_counts=zeros.copy(),
        postflop_conditioned_counts=trainer_v24._new_postflop_conditioned_counts(),
        illegal_action_count=0,
        win=False,
        vpip=bool(vpip),
        pfr=bool(pfr),
        three_bet=bool(three_bet),
        preflop_jam=False,
        flop_seen=False,
        turn_seen=False,
        river_seen=False,
        showdown_seen=False,
        showdown_won=False,
        total_actions=0,
        preflop_actions=0,
        postflop_actions=0,
        blueprint_decisions=0,
        preflop_decisions=0,
        cbet_flop_opportunity=False,
        cbet_flop_taken=False,
        fold_vs_cbet_flop_opportunity=False,
        fold_vs_cbet_flop=False,
        cbet_turn_opportunity=False,
        cbet_turn_taken=False,
        fold_vs_cbet_turn_opportunity=False,
        fold_vs_cbet_turn=False,
        rfi_opportunity=bool(rfi_opportunity),
        rfi_attempt=bool(rfi_attempt),
        hero_hand_key=hero_hand_key,
        preflop_stats={},
    )


def test_infoset_key_ignores_scalar_noise_but_changes_when_bucket_changes():
    base_state = FakeState(
        actor_index=5,
        hole_cards=[[], [], [], [], [], cards("As Ah")],
        board_cards=[],
        stacks=[1000] * 6,
        bets=[5, 10, 0, 0, 0, 0],
        pot_amount=15,
        can_fold=True,
        can_check_or_call=True,
        can_raise=True,
        min_raise=20,
        max_raise=1000,
    )
    same_bucket_state = FakeState(
        actor_index=5,
        hole_cards=[[], [], [], [], [], cards("As Ah")],
        board_cards=[],
        stacks=[1005, 995, 1000, 1000, 1000, 1000],
        bets=[5, 10, 0, 0, 0, 0],
        pot_amount=16,
        can_fold=True,
        can_check_or_call=True,
        can_raise=True,
        min_raise=20,
        max_raise=1000,
    )
    changed_bucket_state = FakeState(
        actor_index=5,
        hole_cards=[[], [], [], [], [], cards("As Ah")],
        board_cards=[],
        stacks=[1000] * 6,
        bets=[5, 40, 0, 0, 0, 0],
        pot_amount=80,
        can_fold=True,
        can_check_or_call=True,
        can_raise=True,
        min_raise=80,
        max_raise=1000,
    )
    ctx = make_ctx(current_street=0)
    base_key, _, _ = worker_v24.build_infoset_key(base_state, 5, ctx)
    same_key, _, _ = worker_v24.build_infoset_key(same_bucket_state, 5, ctx)
    changed_key, _, _ = worker_v24.build_infoset_key(changed_bucket_state, 5, ctx)
    assert base_key == same_key
    assert base_key != changed_key


def test_regret_matching_and_average_policy_fallbacks():
    legal_mask = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    uniform = uniform_legal_policy(legal_mask)
    assert uniform.tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0, 0.0])

    regrets = np.array([-2.0, 5.0, 1.0, 0.0, 0.0], dtype=np.float32)
    sigma = regret_matching(regrets, legal_mask)
    assert sigma.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0])

    avg = average_policy(np.zeros(ACTION_COUNT_V21, dtype=np.float32), legal_mask, regret_sum=np.array([3.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert avg[ACTION_FOLD] > avg[ACTION_CALL]


def test_pruning_and_lcfr_discount_apply_to_tabular_nodes():
    node = TabularNode.new(np.ones(ACTION_COUNT_V21, dtype=np.float32))
    node.regret_sum[:] = np.array([-5.0, 3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    cfg = SimpleNamespace(current_iteration=3, prune_after_iteration=1, negative_regret_floor=-1.0)
    assert worker_v24._should_prune_action(node, ACTION_FOLD, cfg)
    assert not worker_v24._should_prune_action(node, ACTION_CALL, cfg)

    trainer = DeepCFRTrainerV24(DeepCFRConfig(discount_interval_iterations=2, lcfr_after_iteration=2))
    trainer.node_store["test"] = node
    trainer._outer_iteration = 4
    trainer._maybe_discount_tables()
    assert trainer._last_discount_factor == pytest.approx(2.0 / 3.0)
    assert node.regret_sum[ACTION_CHECK] == pytest.approx(2.0)


def test_short_training_run_creates_infosets_and_self_play_aliases_network_only():
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=2,
        traversals_per_chunk=2,
        checkpoint_interval=4,
        averaging_window_traversals=64,
        parallel_rollouts=False,
    )
    trainer = DeepCFRTrainerV24(cfg)
    snapshot = trainer.train_for_traversals(8)
    assert snapshot.traversals_completed == 8
    assert snapshot.infoset_count > 0
    assert snapshot.checkpoint_pool_size >= 1
    assert snapshot.monitor_mode == "self_play"
    assert len(snapshot.action_histogram) == ACTION_COUNT_V21

    self_play = trainer.evaluate_self_play(12)
    network_only = trainer.evaluate_network_only(12)
    assert self_play.avg_profit_bb == pytest.approx(network_only.avg_profit_bb)
    assert self_play.action_histogram == network_only.action_histogram


def test_checkpoint_round_trip_and_legacy_format_rejection(tmp_path):
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=2,
        traversals_per_chunk=2,
        checkpoint_interval=4,
        averaging_window_traversals=64,
        parallel_rollouts=False,
    )
    trainer = DeepCFRTrainerV24(cfg)
    trainer.train_for_traversals(8)
    before = trainer.evaluate_self_play(8)

    checkpoint_path = tmp_path / "tabular_v24.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    reloaded = DeepCFRTrainerV24(cfg)
    reloaded.load_checkpoint(str(checkpoint_path))
    after = reloaded.evaluate_self_play(8)

    assert after.avg_profit_bb == pytest.approx(before.avg_profit_bb)
    assert after.action_histogram == before.action_histogram
    assert reloaded.config.parallel_rollouts is True
    assert reloaded.config.rollout_workers >= reloaded._default_rollout_workers()

    legacy_path = tmp_path / "legacy_v24.pt"
    torch.save({"model_state_dict": {}, "advantage_state_dict": {}}, legacy_path)
    with pytest.raises(RuntimeError, match="incompatible"):
        reloaded.load_checkpoint(str(legacy_path))


def test_parallel_training_run_updates_table_and_metrics():
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=4,
        traversals_per_chunk=4,
        training_monitor_interval_traversals=4,
        checkpoint_interval=8,
        averaging_window_traversals=64,
        parallel_rollouts=True,
        rollout_workers=2,
        rollout_worker_chunk_size=2,
    )
    trainer = DeepCFRTrainerV24(cfg)
    try:
        snapshot = trainer.train_for_traversals(8)
        assert snapshot.traversals_completed == 8
        assert snapshot.infoset_count > 0
        assert snapshot.monitor_mode == "self_play"
        assert sum(snapshot.action_histogram) > 0
    finally:
        trainer.shutdown()


def test_submit_rollout_batch_task_recovers_from_broken_pool(monkeypatch):
    trainer = DeepCFRTrainerV24(DeepCFRConfig(parallel_rollouts=True, rollout_workers=2, rollout_worker_chunk_size=2))

    class FlakyExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            raise BrokenProcessPool("broken on submit")

    class HealthyExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            return "submitted"

    flaky = FlakyExecutor()
    healthy = HealthyExecutor()

    monkeypatch.setattr(trainer, "_recreate_rollout_executor", lambda: healthy)

    result = trainer._submit_rollout_batch_task(
        flaky,
        hand_seeds=[1, 2],
        traverser_seats=[0, 0],
        actor_state_payload=None,
        config_payload={"num_players": 6},
        snapshot_signature="sig_submit_retry",
    )

    assert result == "submitted"
    assert flaky.calls == 1
    assert healthy.calls == 1


def test_parallel_batch_degrades_to_serial_when_pool_breaks(monkeypatch):
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=2,
        traversals_per_chunk=2,
        parallel_rollouts=True,
        rollout_workers=2,
        rollout_worker_chunk_size=1,
    )
    trainer = DeepCFRTrainerV24(cfg)

    class DummyExecutor:
        pass

    broken_future = Future()
    broken_future.set_exception(BrokenProcessPool("A process pool was terminated abruptly while the future was running or pending."))

    monkeypatch.setattr(trainer, "_ensure_rollout_executor", lambda: DummyExecutor())
    monkeypatch.setattr(trainer, "_effective_rollout_workers", lambda: 2)
    monkeypatch.setattr(trainer, "_submit_rollout_batch_task", lambda *args, **kwargs: broken_future)

    def fake_run_traversal(hand_seed, traverser_seat, actor_snapshot, opponent_snapshot, config):
        del actor_snapshot, opponent_snapshot, config
        return worker_v24.TraversalResult(
            utility_bb=0.0,
            unclipped_utility_bb=0.0,
            traverser_seat=int(traverser_seat),
            monitor_sampled=False,
        )

    monkeypatch.setattr(trainer_v24, "run_traversal", fake_run_traversal)

    result = trainer._run_parallel_seat_batch(0, 2)

    assert result is None
    assert trainer.config.parallel_rollouts is False
    assert trainer.traversals_completed == 2


def test_worker_batch_cache_reuses_snapshot_signature():
    worker_v24._WORKER_BASE_SNAPSHOT_SIGNATURE = None
    worker_v24._WORKER_BASE_NODE_STORE = {}
    config_dict = {
        "num_players": 6,
        "small_blind": 5,
        "big_blind": 10,
        "parallel_rollouts": False,
        "training_monitor_interval_traversals": 999999,
    }

    first = worker_v24.run_traversal_batch_mp([11], [0], {}, None, config_dict, "sig_cache")
    second = worker_v24.run_traversal_batch_mp([13], [0], None, None, config_dict, "sig_cache")

    assert first.results
    assert second.results
    assert worker_v24._WORKER_BASE_SNAPSHOT_SIGNATURE == "sig_cache"


def test_worker_batch_cache_accepts_delta_update_then_reuses_new_signature():
    worker_v24._WORKER_BASE_SNAPSHOT_SIGNATURE = None
    worker_v24._WORKER_BASE_NODE_STORE = {}
    config_dict = {
        "num_players": 6,
        "small_blind": 5,
        "big_blind": 10,
        "parallel_rollouts": False,
        "training_monitor_interval_traversals": 999999,
    }

    full_payload = {"__snapshot_mode__": "full", "node_store": {}}
    delta_payload = {"__snapshot_mode__": "delta", "base_signature": "sig_a", "delta_payload": {}}

    first = worker_v24.run_traversal_batch_mp([17], [0], full_payload, None, config_dict, "sig_a")
    second = worker_v24.run_traversal_batch_mp([19], [0], delta_payload, None, config_dict, "sig_b")
    third = worker_v24.run_traversal_batch_mp([23], [0], None, None, config_dict, "sig_b")

    assert first.results
    assert second.results
    assert third.results
    assert worker_v24._WORKER_BASE_SNAPSHOT_SIGNATURE == "sig_b"


def test_checkpoint_eval_uses_fixed_snapshot_per_seat(monkeypatch):
    calls = {"seat_maps": 0}
    snapshot_ids_by_actor = {}

    snapshot_a = worker_v24.freeze_policy_snapshot({}, {"name": "a"})
    snapshot_b = worker_v24.freeze_policy_snapshot({}, {"name": "b"})

    def fake_seat_map(config, rng, hero_seat):
        del config, rng, hero_seat
        calls["seat_maps"] += 1
        return {1: snapshot_a, 2: snapshot_b, 3: snapshot_a, 4: snapshot_b, 5: snapshot_a}

    original_snapshot_policy = worker_v24._snapshot_policy

    def tracking_snapshot_policy(snapshot, state, actor, hand_ctx):
        if snapshot is not None and int(actor) != 0:
            snapshot_ids_by_actor.setdefault(int(actor), set()).add(id(snapshot))
        return original_snapshot_policy(snapshot, state, actor, hand_ctx)

    monkeypatch.setattr(worker_v24, "_checkpoint_snapshot_by_seat", fake_seat_map)
    monkeypatch.setattr(worker_v24, "_snapshot_policy", tracking_snapshot_policy)

    config = worker_v24.build_runtime_policy_config(
        {
            "evaluation_mode": "checkpoints",
            "eval_hero_seat": 0,
            "checkpoint_pool": (snapshot_a, snapshot_b),
            "num_players": 6,
            "small_blind": 5,
            "big_blind": 10,
        }
    )
    worker_v24.run_policy_hand(7, snapshot_a, config)
    assert calls["seat_maps"] == 1
    assert snapshot_ids_by_actor
    assert all(len(snapshot_ids) == 1 for snapshot_ids in snapshot_ids_by_actor.values())


def test_run_policy_hand_records_canonical_hero_hand_key(monkeypatch):
    fake_state = SimpleNamespace(
        status=False,
        hole_cards=[[], [], [], [], [], cards("As Ks")],
        board_cards=[],
        stacks=[1000] * 6,
        bets=[0] * 6,
        pots=[],
    )
    fake_ctx = SimpleNamespace(
        starting_stacks=[1000] * 6,
        big_blind=10,
        total_actions=0,
        preflop_actions=0,
        flop_seen=False,
        turn_seen=False,
        river_seen=False,
        in_hand=[True] * 6,
    )

    monkeypatch.setattr(worker_v24, "_create_state_and_context", lambda rng, config: (fake_state, fake_ctx))

    result = worker_v24.run_policy_hand(
        7,
        worker_v24.freeze_policy_snapshot({}, {}),
        worker_v24.build_runtime_policy_config({"eval_hero_seat": 5, "num_players": 6}),
    )

    assert result.hero_hand_key == "AKs"


def test_evaluation_report_populates_preflop_hand_grids():
    trainer = DeepCFRTrainerV24(DeepCFRConfig(parallel_rollouts=False))
    report = trainer._build_evaluation_report(
        "self_play",
        [
            make_hand_result(hero_seat=5, hero_hand_key="AKs", vpip=True, rfi_opportunity=True, rfi_attempt=True, pfr=True),
            make_hand_result(hero_seat=5, hero_hand_key="AKo", vpip=False, rfi_opportunity=True, rfi_attempt=False, pfr=False),
            make_hand_result(hero_seat=4, hero_hand_key="QQ", vpip=True, rfi_opportunity=False, rfi_attempt=False, pfr=True),
        ],
        0.1,
    )

    assert report.vpip_hand_grid[0][1] == pytest.approx(1.0)
    assert report.rfi_hand_grid[0][1] == pytest.approx(1.0)
    assert report.vpip_hand_grid[1][0] == pytest.approx(0.0)
    assert report.rfi_hand_grid[1][0] == pytest.approx(0.0)
    assert report.vpip_hand_grid[2][2] == pytest.approx(1.0)
    assert report.vpip_hand_grid_by_position["BTN"][0][1] == pytest.approx(1.0)
    assert report.rfi_hand_grid_by_position["BTN"][1][0] == pytest.approx(0.0)
    assert report.pfr_hand_grid_by_position["CO"][2][2] == pytest.approx(1.0)


def test_parallel_evaluation_matches_serial_results():
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=2,
        traversals_per_chunk=2,
        checkpoint_interval=4,
        averaging_window_traversals=64,
        parallel_rollouts=False,
    )
    trainer = DeepCFRTrainerV24(cfg)
    try:
        trainer.train_for_traversals(8)
        serial_report = trainer.evaluate_vs_heuristics(12)

        trainer.config.parallel_rollouts = True
        trainer.config.rollout_workers = 2
        trainer.config.rollout_worker_chunk_size = 3
        parallel_report = trainer.evaluate_vs_heuristics(12)

        assert parallel_report.avg_profit_bb == pytest.approx(serial_report.avg_profit_bb)
        assert parallel_report.win_rate == pytest.approx(serial_report.win_rate)
        assert parallel_report.action_histogram == serial_report.action_histogram
        assert parallel_report.preflop_action_histogram == serial_report.preflop_action_histogram
    finally:
        trainer.shutdown()


def test_eval_suite_runs_baseline_pool():
    cfg = DeepCFRConfig(
        traversals_per_player_per_iteration=2,
        traversals_per_chunk=2,
        checkpoint_interval=4,
        averaging_window_traversals=64,
        parallel_rollouts=True,
        rollout_workers=2,
        rollout_worker_chunk_size=2,
    )
    trainer = DeepCFRTrainerV24(cfg)
    try:
        trainer.train_for_traversals(8)
        report = trainer.evaluate_eval_suite(4)

        assert report.suite_name == "eval_suite"
        assert set(report.robust_reports.keys()) == {"heuristics", "checkpoints"}
        assert set(report.leak_reports.keys()) == set(trainer_v24.SYNTHETIC_OPPONENT_STYLES)
        assert report.hands_per_mode == 4
    finally:
        trainer.shutdown()
