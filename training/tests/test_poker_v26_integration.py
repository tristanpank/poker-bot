import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for rel in ("agents", "trainers", "features"):
    path = str(SRC_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from poker_gui_trainer_v26 import DeepCFRGuiConfigV26, DeepCFRTrainerV26GUI, SYNTHETIC_OPPONENT_STYLES
import poker_gui_trainer_v26 as gui_trainer_v26
from poker_trainer_v26 import DeepCFRConfigV26, DeepCFRTrainerV26
import poker_agent_v26 as agent_v26


def make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "readme.md").write_text("stub", encoding="utf-8")
    flagship_dir = repo_root / "flagship_models" / "first"
    flagship_dir.mkdir(parents=True)
    (flagship_dir / "mixed_checkpoint_iter_11200.pt").write_text("stub", encoding="utf-8")
    (flagship_dir / "1-model.pt").write_text("stub", encoding="utf-8")
    return repo_root


def test_flagship_defaults_are_detected(tmp_path):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )

    assert trainer.flagship_models_dir == (repo_root / "flagship_models" / "first").resolve()
    assert trainer.flagship_checkpoint == (repo_root / "flagship_models" / "first" / "mixed_checkpoint_iter_11200.pt").resolve()
    assert trainer.preferred_models_dir() == trainer.flagship_models_dir
    assert trainer.preferred_checkpoint() == trainer.flagship_checkpoint


def test_install_requirements_calls_pip_in_repo(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    calls = []

    def fake_run(args, *, cwd=None):
        calls.append((list(args), cwd))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(trainer, "_run_external", fake_run)

    trainer.install_requirements(upgrade_pip=True)

    assert calls[0][0] == ["-m", "pip", "install", "--upgrade", "pip"]
    assert calls[1][0] == ["-m", "pip", "install", "-r", "requirements.txt"]


def test_start_training_builds_module_command(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    calls = []

    def fake_run(args, *, cwd=None):
        calls.append((list(args), cwd))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(trainer, "_run_external", fake_run)

    run_dir = trainer.start_training(iterations=7, traversals=9, verbose=True, strict=True)

    assert run_dir.exists()
    assert (run_dir / "models").exists()
    assert (run_dir / "logs").exists()
    assert calls[0][0][:3] == ["-m", "src.training.train", "--iterations"]
    assert "--save-dir" in calls[0][0]
    assert "--log-dir" in calls[0][0]
    assert "--verbose" in calls[0][0]
    assert "--strict" in calls[0][0]


def test_resume_defaults_to_flagship_checkpoint(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    calls = []

    def fake_run(args, *, cwd=None):
        calls.append((list(args), cwd))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(trainer, "_run_external", fake_run)

    trainer.resume_training(iterations=3, traversals=5)

    assert "--checkpoint" in calls[0][0]
    checkpoint_index = calls[0][0].index("--checkpoint") + 1
    assert Path(calls[0][0][checkpoint_index]) == trainer.flagship_checkpoint


def test_play_cli_defaults_to_flagship_models_dir(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    calls = []

    def fake_run(args, *, cwd=None):
        calls.append((list(args), cwd))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(trainer, "_run_external", fake_run)

    models_dir = trainer.play_cli(no_shuffle=True, strict=True)

    assert models_dir == trainer.flagship_models_dir
    assert calls[0][0][:2] == ["-m", "scripts.play"]
    assert "--models-dir" in calls[0][0]
    models_dir_index = calls[0][0].index("--models-dir") + 1
    assert Path(calls[0][0][models_dir_index]) == trainer.flagship_models_dir
    assert "--no-shuffle" in calls[0][0]
    assert "--strict" in calls[0][0]


def test_mixed_training_uses_flagship_dir_without_training_prefix(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26(
        DeepCFRConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    calls = []

    def fake_run(args, *, cwd=None):
        calls.append((list(args), cwd))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(trainer, "_run_external", fake_run)

    trainer.mixed_training(iterations=4, traversals=6)

    assert "--checkpoint-dir" in calls[0][0]
    checkpoint_dir_index = calls[0][0].index("--checkpoint-dir") + 1
    assert Path(calls[0][0][checkpoint_dir_index]) == trainer.flagship_models_dir
    prefix_index = calls[0][0].index("--model-prefix") + 1
    assert calls[0][0][prefix_index] == ""


def test_agent_cli_status_runs_without_error(tmp_path, capsys):
    repo_root = make_repo(tmp_path)
    exit_code = agent_v26.main(
        [
            "--repo-root",
            str(repo_root),
            "--results-root",
            str(tmp_path / "results"),
            "status",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Flagship Models:" in captured.out
    assert "Default Play Models:" in captured.out


def test_agent_v26_reuses_legacy_dashboard_module():
    legacy_gui = agent_v26.configure_legacy_gui_module()

    assert legacy_gui.MODEL_VERSION == "v26"
    assert legacy_gui.MODEL_LABEL == "V26"
    assert legacy_gui.ACTION_NAMES == ["fold", "check_call", "raise"]
    assert legacy_gui.DeepCFRTrainerV25 is DeepCFRTrainerV26GUI
    assert legacy_gui.DEFAULT_CHECKPOINT_FILENAME.endswith("_v26_resume_checkpoint.pt")
    assert "maniac" in legacy_gui.EVAL_MODE_VALUES
    assert legacy_gui.TrainingGUI._v26_ingest_patched is True


def test_v26_exposes_v25_synthetic_styles():
    assert SYNTHETIC_OPPONENT_STYLES == (
        "nit",
        "overfolder",
        "overcaller",
        "over3better",
        "station",
        "maniac",
    )


def test_gui_v26_defaults_to_phased_training():
    trainer = DeepCFRTrainerV26GUI()
    assert trainer.config.training_monitor_mode == "phased"
    assert trainer.config.traversals_per_chunk == 400
    assert trainer.config.utility_averaging_window_hands == 10000
    assert trainer.config.advantage_batch_size == 256
    assert trainer.config.strategy_batch_size == 256
    assert trainer.config.parallel_traversal_workers == 2
    assert trainer.config.phase1_iterations == 2
    assert trainer.config.phase1_traversals_per_iteration == 200
    assert trainer.config.phase2_iterations == 4
    assert trainer.config.phase2_traversals_per_iteration == 400
    assert trainer.config.phase3_iterations == 8
    assert trainer.config.phase3_traversals_per_iteration == 400
    assert trainer.planned_total_traversals() == (2 * 200) + (4 * 400) + (8 * 400)


def test_gui_checkpoint_pool_skips_wrapper_snapshots(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    models_dir = tmp_path / "pool"
    models_dir.mkdir()
    raw_checkpoint = models_dir / "mixed_checkpoint_iter_10.pt"
    gui_checkpoint = models_dir / "gui_checkpoint_iter_10.pt"
    torch.save({"iteration": 10, "advantage_net": {}, "strategy_net": {}}, raw_checkpoint)
    torch.save({"format_version": "deepcfr_v26_gui", "training_agent": {}}, gui_checkpoint)

    monkeypatch.setattr(trainer, "preferred_models_dir", lambda: models_dir)

    assert trainer._checkpoint_pool_paths() == [raw_checkpoint]


def test_gui_parallel_task_builder_partitions_traversals():
    trainer = DeepCFRTrainerV26GUI()
    trainer.config.parallel_traversal_enabled = True
    trainer.config.parallel_traversal_workers = 2
    trainer.config.parallel_traversal_min_traversals = 2
    trainer._training_agent = object()
    trainer._training_opponent_specs = [None, {"kind": "random"}, {"kind": "random"}, {"kind": "random"}, {"kind": "random"}, {"kind": "random"}]
    trainer._training_agent_payload_for_workers = lambda: {"iteration": 1, "advantage_net": {}, "strategy_net": {}}

    tasks = trainer._parallel_traversal_tasks("random", 3, 5, button_offset=0)

    assert len(tasks) == 2
    assert sum(len(task["seeds_and_buttons"]) for task in tasks) == 5
    assert all(task["mode"] == "random" for task in tasks)


def test_gui_training_follows_phased_schedule(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )

    class DummyAgent:
        def __init__(self, label):
            self.label = label
            self.player_id = 0
            self.iteration_count = 0
            self.advantage_memory = []
            self.strategy_memory = []
            self.strategy_train_calls = 0
            self.random_calls = 0

        def cfr_traverse(self, *args, **kwargs):
            self.random_calls += 1

        def train_advantage_network(self, batch_size=128):
            return 1.25

        def train_strategy_network(self, batch_size=128):
            self.strategy_train_calls += 1
            return 0.5

    class DummyPkrs:
        class State:
            @staticmethod
            def from_seed(**kwargs):
                return {"seed": kwargs["seed"], "button": kwargs["button"]}

    agents = []
    traversal_calls = []
    saved_raw = []

    monkeypatch.setattr(trainer, "_ensure_repo_imports", lambda: None)
    trainer.config.phase1_iterations = 100
    trainer.config.phase1_traversals_per_iteration = 1
    trainer.config.phase2_iterations = 100
    trainer.config.phase2_traversals_per_iteration = 1
    trainer.config.phase3_iterations = 100
    trainer.config.phase3_traversals_per_iteration = 1
    trainer._pkrs = DummyPkrs
    trainer._random_agent_cls = lambda seat: f"random-{seat}"
    trainer._new_agent = lambda player_id=0: agents.append(DummyAgent(f"agent-{len(agents)+1}")) or agents[-1]
    trainer._load_agent_from_checkpoint = lambda checkpoint_path, player_id=0: DummyAgent(f"loaded-{player_id}")
    trainer._select_mixed_training_opponents = lambda *_args: [None, "opp1", "opp2", "opp3", "opp4", "opp5"]
    trainer._wrap_training_opponents = lambda opponents: list(opponents)
    trainer._cfr_traverse_with_opponents = (
        lambda learning_agent, state, iteration, opponent_wrappers, verbose=False: traversal_calls.append(
            (learning_agent, state, iteration, tuple(opponent_wrappers))
        )
    )
    def fake_save_external_training_checkpoint(mode, iteration):
        saved_raw.append((mode, iteration))
        phase_name = "phase1" if mode == "random" else "phase2" if mode == "self_play" else "phase3"
        phase_dir = trainer._phase_models_dir(phase_name)
        checkpoint_path = phase_dir / trainer._raw_checkpoint_name(mode, iteration)
        checkpoint_path.write_text("stub", encoding="utf-8")
        return checkpoint_path

    trainer._save_external_training_checkpoint = fake_save_external_training_checkpoint
    trainer._evaluate = lambda mode, num_hands: SimpleNamespace(
        avg_profit_bb=0.0,
        vpip=0.0,
        pfr=0.0,
        three_bet=0.0,
        preflop_jam_rate=0.0,
        flop_seen_rate=0.0,
        avg_actions_per_hand=0.0,
        avg_preflop_actions_per_hand=0.0,
        position_avg_profit_bb={name: 0.0 for name in trainer._position_profit_windows},
    )
    trainer._refresh_snapshot = lambda status: {"status": status}

    snapshot = trainer.train_for_traversals(205)

    assert snapshot == {"status": "Training"}
    assert trainer.learner_steps == 205
    assert agents[0].random_calls == 100
    assert len(traversal_calls) == 105
    assert saved_raw[:2] == [("random", 100), ("self_play", 100)]
    assert trainer._phase_index == 2
    assert trainer._active_phase_mode() == "mixed"


def test_gui_skips_monitor_eval_until_iteration_completes(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )

    class DummyAgent:
        def __init__(self):
            self.player_id = 0
            self.iteration_count = 0
            self.advantage_memory = []
            self.strategy_memory = []

        def cfr_traverse(self, *args, **kwargs):
            return None

        def train_advantage_network(self, batch_size=128):
            raise AssertionError("should not train before a full iteration completes")

        def train_strategy_network(self, batch_size=128):
            raise AssertionError("should not train before a full iteration completes")

    class DummyPkrs:
        class State:
            @staticmethod
            def from_seed(**kwargs):
                return {"seed": kwargs["seed"], "button": kwargs["button"]}

    eval_calls = []
    monkeypatch.setattr(trainer, "_ensure_repo_imports", lambda: None)
    trainer.config.phase1_iterations = 1
    trainer.config.phase1_traversals_per_iteration = 10
    trainer.config.phase2_iterations = 0
    trainer.config.phase3_iterations = 0
    trainer._pkrs = DummyPkrs
    trainer._random_agent_cls = lambda seat: f"random-{seat}"
    trainer._new_agent = lambda player_id=0: DummyAgent()
    trainer._evaluate = lambda mode, num_hands: eval_calls.append((mode, num_hands))
    trainer._refresh_snapshot = lambda status: {"status": status}

    snapshot = trainer.train_for_traversals(1)

    assert snapshot == {"status": "Training"}
    assert trainer.learner_steps == 0
    assert eval_calls == []


def test_gui_snapshot_uses_true_monitor_hand_count_and_weighted_bb_average():
    trainer = DeepCFRTrainerV26GUI()
    trainer._recent_profit_bb.extend([1.0, -1.0])
    trainer._recent_profit_hand_counts.extend([100, 300])
    trainer._recent_vpip.extend([0.2, 0.6])
    trainer._recent_action_histograms.extend(
        [
            np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 30.0, 0.0], dtype=np.float64),
        ]
    )
    trainer._recent_preflop_action_histograms.extend(
        [
            np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 30.0, 0.0], dtype=np.float64),
        ]
    )
    trainer._recent_postflop_action_histograms.extend(
        [
            np.asarray([5.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 15.0, 0.0], dtype=np.float64),
        ]
    )
    trainer._recent_position_profit_maps.extend(
        [
            {name: 1.0 for name in gui_trainer_v26.POSITION_NAMES_V26},
            {name: -1.0 for name in gui_trainer_v26.POSITION_NAMES_V26},
        ]
    )
    trainer._recent_speed_hands.extend([100, 300])
    trainer._recent_speed_seconds.extend([10.0, 30.0])

    snapshot = trainer._build_snapshot("Testing")

    assert snapshot.utility_window_count == 400
    assert snapshot.avg_utility_bb == pytest.approx(-0.5)
    assert snapshot.vpip == pytest.approx(0.5)
    assert snapshot.action_histogram == [10, 30, 0]
    assert snapshot.hands_per_second == pytest.approx(10.0)


def test_gui_profit_window_trims_to_last_10000_hands():
    trainer = DeepCFRTrainerV26GUI()
    trainer._recent_profit_bb.extend([1.0, 2.0, 3.0])
    trainer._recent_profit_hand_counts.extend([4000, 4000, 4000])
    gui_trainer_v26._trim_weighted_window(
        trainer._recent_profit_bb,
        trainer._recent_profit_hand_counts,
        trainer.config.utility_averaging_window_hands,
    )

    assert sum(trainer._recent_profit_hand_counts) == 10000
    assert list(trainer._recent_profit_hand_counts) == [2000, 4000, 4000]


def test_gui_checkpoint_load_preserves_runtime_overrides_and_training_buffers(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )

    class DummyMemory:
        def __init__(self):
            self.buffer = [("exp",)]
            self.priorities = [0.5]
            self.position = 1
            self.capacity = 10
            self._max_priority = 2.0

        def add(self, experience, priority=None):
            self.buffer.append(experience)
            self.priorities.append(priority if priority is not None else 1.0)

        def __len__(self):
            return len(self.buffer)

    class DummyAgent:
        def __init__(self):
            self.iteration_count = 7
            self.min_bet_size = 0.1
            self.max_bet_size = 3.0
            self.advantage_net = SimpleNamespace(state_dict=lambda: {"a": 1}, load_state_dict=lambda state: None, eval=lambda: None)
            self.strategy_net = SimpleNamespace(state_dict=lambda: {"b": 2}, load_state_dict=lambda state: None, eval=lambda: None)
            self.advantage_memory = DummyMemory()
            self.strategy_memory = deque([("s",)], maxlen=10)

    hydrated = DummyAgent()
    hydrated.advantage_memory = DummyMemory()
    hydrated.advantage_memory.buffer = []
    hydrated.advantage_memory.priorities = []
    hydrated.strategy_memory = deque([], maxlen=10)

    trainer._training_agent = DummyAgent()
    trainer._recent_profit_bb.extend([1.0])
    trainer._recent_profit_hand_counts.extend([400])
    trainer.config.parallel_traversal_workers = 4
    trainer.config.utility_averaging_window_hands = 10000
    trainer.config.phase1_iterations = 12
    trainer.config.phase1_traversals_per_iteration = 345
    trainer.config.phase2_iterations = 34
    trainer.config.phase2_traversals_per_iteration = 456
    trainer.config.phase3_iterations = 56
    trainer.config.phase3_traversals_per_iteration = 567
    monkeypatch.setattr(trainer, "_hydrate_training_agent", lambda payload: hydrated)

    checkpoint_path = tmp_path / "gui_checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path))

    reloaded = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )
    reloaded.config.parallel_traversal_workers = 4
    reloaded.config.utility_averaging_window_hands = 10000
    reloaded.config.phase1_iterations = 12
    reloaded.config.phase1_traversals_per_iteration = 345
    reloaded.config.phase2_iterations = 34
    reloaded.config.phase2_traversals_per_iteration = 456
    reloaded.config.phase3_iterations = 56
    reloaded.config.phase3_traversals_per_iteration = 567

    def restore_with_buffers(payload):
        agent = hydrated
        agent.iteration_count = int(payload.get("iteration", 0))
        agent.advantage_memory.buffer = list(payload.get("advantage_buffer", []))
        agent.advantage_memory.priorities = list(payload.get("advantage_priorities", []))
        agent.advantage_memory.position = int(payload.get("advantage_position", 0))
        agent.advantage_memory._max_priority = float(payload.get("advantage_max_priority", 1.0))
        agent.strategy_memory.clear()
        agent.strategy_memory.extend(list(payload.get("strategy_memory", [])))
        return agent

    monkeypatch.setattr(reloaded, "_hydrate_training_agent", restore_with_buffers)
    reloaded.load_checkpoint(str(checkpoint_path))

    assert reloaded.config.parallel_traversal_workers == 4
    assert reloaded.config.utility_averaging_window_hands == 10000
    assert reloaded.config.phase1_iterations == 12
    assert reloaded.config.phase1_traversals_per_iteration == 345
    assert reloaded.config.phase2_iterations == 34
    assert reloaded.config.phase2_traversals_per_iteration == 456
    assert reloaded.config.phase3_iterations == 56
    assert reloaded.config.phase3_traversals_per_iteration == 567
    assert list(reloaded._recent_profit_hand_counts) == [400]
    assert list(reloaded._training_agent.advantage_memory.buffer) == [("exp",)]
    assert list(reloaded._training_agent.strategy_memory) == [("s",)]


def test_gui_monitor_eval_uses_full_completed_chunk_size(tmp_path, monkeypatch):
    repo_root = make_repo(tmp_path)
    trainer = DeepCFRTrainerV26GUI(
        DeepCFRGuiConfigV26(repo_root=str(repo_root), results_root=str(tmp_path / "results"))
    )

    class DummyAgent:
        def __init__(self):
            self.player_id = 0
            self.iteration_count = 0
            self.advantage_memory = []
            self.strategy_memory = []

        def cfr_traverse(self, *args, **kwargs):
            return None

        def train_advantage_network(self, batch_size=128):
            return 0.0

        def train_strategy_network(self, batch_size=128):
            return 0.0

    class DummyPkrs:
        class State:
            @staticmethod
            def from_seed(**kwargs):
                return {"seed": kwargs["seed"], "button": kwargs["button"]}

    eval_calls = []
    monkeypatch.setattr(trainer, "_ensure_repo_imports", lambda: None)
    trainer.config.phase1_iterations = 1
    trainer.config.phase1_traversals_per_iteration = 400
    trainer.config.phase2_iterations = 0
    trainer.config.phase3_iterations = 0
    trainer.config.checkpoint_interval = 10_000
    trainer._pkrs = DummyPkrs
    trainer._random_agent_cls = lambda seat: f"random-{seat}"
    trainer._new_agent = lambda player_id=0: DummyAgent()
    trainer._evaluate = lambda mode, num_hands: eval_calls.append((mode, num_hands)) or SimpleNamespace(
        avg_profit_bb=0.0,
        vpip=0.0,
        pfr=0.0,
        three_bet=0.0,
        preflop_jam_rate=0.0,
        flop_seen_rate=0.0,
        avg_actions_per_hand=0.0,
        avg_preflop_actions_per_hand=0.0,
        position_avg_profit_bb={name: 0.0 for name in trainer._position_profit_windows},
        hands=int(num_hands),
        action_histogram=[0, 0, 0],
        preflop_action_histogram=[0, 0, 0],
        postflop_action_histogram=[0, 0, 0],
    )
    trainer._refresh_snapshot = lambda status: {"status": status}

    trainer.train_for_traversals(400)

    assert eval_calls == [("checkpoints", 400)]


def test_external_gui_stage_and_card_adapters_work(tmp_path):
    pkrs = pytest.importorskip("pokers")
    repo_root = Path(DeepCFRConfigV26().repo_root).resolve()
    if not (repo_root / "scripts" / "poker_gui.py").exists():
        pytest.skip("External DeepCFR repo is not available")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import scripts.poker_gui as poker_gui

    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=1)
    card = state.players_state[0].hand[0]

    assert poker_gui.stage_to_string(pkrs.Stage.Preflop) == "Preflop"
    assert poker_gui.card_display_parts(card)[0]
    assert poker_gui.CardWidget.update_display.__name__ == "_patched_card_widget_update_display"
    assert poker_gui.PokerTable.update_stage.__name__ == "_patched_poker_table_update_stage"


def test_legacy_checkpoint_mapping_prefers_check_call_on_action_zero():
    pkrs = pytest.importorskip("pokers")
    repo_root = Path(DeepCFRConfigV26().repo_root).resolve()
    if not (repo_root / "src" / "core" / "deep_cfr.py").exists():
        pytest.skip("External DeepCFR repo is not available")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from src.core.deep_cfr import DeepCFRAgent

    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=1)
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.load_model(str(repo_root / "flagship_models" / "first" / "mixed_checkpoint_iter_11200.pt"))

    while not state.final_state and state.current_player != 0:
        legal = list(state.legal_actions)
        if pkrs.ActionEnum.Call in legal:
            action = pkrs.Action(pkrs.ActionEnum.Call)
        elif pkrs.ActionEnum.Check in legal:
            action = pkrs.Action(pkrs.ActionEnum.Check)
        else:
            action = pkrs.Action(pkrs.ActionEnum.Fold)
        state = state.apply_action(action)

    assert pkrs.ActionEnum.Call in state.legal_actions
    assert pkrs.ActionEnum.Fold in state.legal_actions
    assert agent._legacy_action_id_to_pokers_action(0, state).action == pkrs.ActionEnum.Call
    assert agent._legacy_action_id_to_pokers_action(1, state).action == pkrs.ActionEnum.Fold


def test_external_cfr_traverse_encodes_each_infoset_once(monkeypatch):
    repo_root = Path(DeepCFRConfigV26().repo_root).resolve()
    if not (repo_root / "src" / "core" / "deep_cfr.py").exists():
        pytest.skip("External DeepCFR repo is not available")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import src.core.deep_cfr as deep_cfr

    encode_calls = []

    def fake_encode_state(state, player_id=0):
        encode_calls.append((state, player_id))
        return [0.0] * 156

    class FakeMemory:
        def __init__(self):
            self.items = []

        def add(self, experience, priority=None):
            self.items.append((experience, priority))

        def __len__(self):
            return len(self.items)

    class FakeNet:
        def __call__(self, inputs):
            batch = inputs.shape[0]
            return torch.zeros((batch, 3), dtype=torch.float32), torch.zeros((batch, 1), dtype=torch.float32)

    class FakeAction:
        pass

    class FakePlayerState:
        reward = 0.0

    class FakeState:
        def __init__(self, final_state=False):
            self.final_state = final_state
            self.current_player = 0
            self.players_state = [FakePlayerState()]
            self.status = None

        def apply_action(self, action):
            next_state = FakeState(final_state=True)
            next_state.players_state[0].reward = 0.0
            next_state.status = deep_cfr.pkrs.StateStatus.Ok
            return next_state

    agent = deep_cfr.DeepCFRAgent(player_id=0, num_players=1, device="cpu")
    agent.advantage_net = FakeNet()
    agent.advantage_memory = FakeMemory()
    agent.strategy_memory = []
    agent.get_legal_action_types = lambda state: [0, 1]
    agent.action_type_to_pokers_action = lambda action_type, state, bet_size_multiplier=None: FakeAction()

    monkeypatch.setattr(deep_cfr, "encode_state", fake_encode_state)

    value = agent.cfr_traverse(FakeState(final_state=False), iteration=1, random_agents=[None])

    assert value == 0.0
    assert len(encode_calls) == 1


def test_external_probability_normalizer_fixes_sum_drift():
    repo_root = Path(DeepCFRConfigV26().repo_root).resolve()
    if not (repo_root / "src" / "core" / "deep_cfr.py").exists():
        pytest.skip("External DeepCFR repo is not available")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import src.core.deep_cfr as deep_cfr

    probs = deep_cfr._normalize_probabilities([0.2, 0.2, 0.6000003])

    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert float(np.sum(probs)) == pytest.approx(1.0)
