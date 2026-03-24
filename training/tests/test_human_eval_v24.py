import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pokerkit import Card

warnings.filterwarnings("ignore", message="optree is installed but the version is too old.*", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for rel in ("features", "models", "workers", "agents", "trainers"):
    path = str(SRC_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import human_eval_v24
import poker_worker_v24 as worker_v24
from poker_model_v24 import PokerDeepCFRNet
from poker_state_v24 import ACTION_CALL, ACTION_COUNT_V21, ACTION_FOLD
from tabular_policy_v24 import TabularPolicySnapshot, freeze_policy_snapshot

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


def make_ctx(*, current_street: int):
    return SimpleNamespace(
        big_blind=10,
        small_blind=5,
        current_street=int(current_street),
        street_raise_count=0,
        preflop_raise_count=0,
        preflop_call_count=0,
        preflop_last_raiser=None,
        last_aggressor=None,
        contributions=[0.0] * 6,
        in_hand=[True] * 6,
    )


def _facing_open_state() -> FakeState:
    return FakeState(
        actor_index=5,
        hole_cards=[[], [], [], [], [], cards("As Kh")],
        board_cards=[],
        stacks=[1000] * 6,
        bets=[5, 10, 0, 0, 0, 0],
        pot_amount=15,
        can_fold=True,
        can_check_or_call=True,
        can_raise=False,
    )


def test_load_policy_accepts_tabular_v24_checkpoint(tmp_path):
    snapshot = freeze_policy_snapshot({}, {"traversals_completed": 7, "iteration": 3})
    checkpoint_path = tmp_path / "tabular_v24.pt"
    torch.save(
        {
            "format_version": "tabular_mccfr_v24",
            "config": {
                "algorithm_name": "tabular_mccfr_6max",
                "action_abstraction_name": "conservative_5a",
            },
            "actor_snapshot": snapshot.to_payload(),
            "node_store": {},
            "checkpoint_pool": [snapshot.to_payload()],
            "metrics": {"traversals_completed": 7, "outer_iteration": 3},
        },
        checkpoint_path,
    )

    policy, leaf_model, config = human_eval_v24.load_policy(str(checkpoint_path))

    assert isinstance(policy, TabularPolicySnapshot)
    assert leaf_model is None
    assert config["algorithm_name"] == "tabular_mccfr_6max"
    assert policy.metadata["iteration"] == 3


def test_load_policy_accepts_legacy_neural_v24_checkpoint(tmp_path):
    model = PokerDeepCFRNet(init_weights=False)
    checkpoint_path = tmp_path / "legacy_v24.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "action_abstraction_name": "conservative_5a",
            },
        },
        checkpoint_path,
    )

    policy, leaf_model, config = human_eval_v24.load_policy(str(checkpoint_path))

    assert isinstance(policy, PokerDeepCFRNet)
    assert leaf_model is None
    assert config["action_abstraction_name"] == "conservative_5a"
    assert policy.action_dim == ACTION_COUNT_V21


def test_load_policy_disables_weights_only_mode(monkeypatch):
    calls = {}
    snapshot = freeze_policy_snapshot({}, {"iteration": 1})

    def fake_load(path, map_location=None, weights_only=None):
        calls["path"] = path
        calls["map_location"] = map_location
        calls["weights_only"] = weights_only
        return {
            "format_version": "tabular_mccfr_v24",
            "config": {"algorithm_name": "tabular_mccfr_6max"},
            "actor_snapshot": snapshot.to_payload(),
        }

    monkeypatch.setattr(human_eval_v24.torch, "load", fake_load)

    policy, leaf_model, config = human_eval_v24.load_policy("dummy_checkpoint.pt")

    assert isinstance(policy, TabularPolicySnapshot)
    assert leaf_model is None
    assert config["algorithm_name"] == "tabular_mccfr_6max"
    assert calls["map_location"] == "cpu"
    assert calls["weights_only"] is False


def test_format_post_hand_log_reveals_model_cards_after_hand():
    state = SimpleNamespace(
        hole_cards=[
            [],
            [],
            [],
            [],
            [],
            cards("As Kh"),
        ],
        board_cards=cards("Qh Js 9c"),
        stacks=[980, 990, 1000, 1010, 1020, 1030],
    )
    hand_ctx = SimpleNamespace(
        hole_cards_by_seat=[
            cards("2c 2d"),
            cards("3c 3d"),
            cards("4c 4d"),
            cards("5c 5d"),
            cards("6c 6d"),
            cards("As Kh"),
        ]
    )

    text = human_eval_v24._format_post_hand_log(
        hand_index=12,
        hero_profit_bb=3.0,
        hero_seat=5,
        state=state,
        hand_ctx=hand_ctx,
        clone_name_by_seat={0: "Clone 1", 1: "Clone 2", 2: "Clone 3", 3: "Clone 4", 4: "Clone 5"},
        clone_id_by_seat={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        clone_bankroll_chips=[980, 990, 1000, 1010, 1020],
        big_blind=10,
    )

    assert "Hand #12 complete. You won +3.00 BB." in text
    assert "Board: Qh Js 9c" in text
    assert "You (BTN (5)): Ash Kd" not in text
    assert "You (BTN (5)): As Kh [103.0 BB]" in text
    assert "Revealed opponent cards:" in text
    assert "Clone 1 (SB (0)): 2c 2d [98.0 BB]" in text
    assert "Clone 5 (CO (4)): 6c 6d [102.0 BB]" in text


def test_policy_action_for_snapshot_supports_tabular_and_neural_sources():
    state = _facing_open_state()
    hand_ctx = make_ctx(current_street=0)
    config = worker_v24.build_runtime_policy_config({"evaluation_mode": "heuristics"})

    tabular_snapshot = freeze_policy_snapshot({}, {})
    tabular_action, tabular_details = worker_v24._policy_action_for_snapshot(
        tabular_snapshot,
        state,
        5,
        hand_ctx,
        random.Random(11),
        config=config,
        return_details=True,
    )

    neural_snapshot = PokerDeepCFRNet(init_weights=False)
    neural_action, neural_details = worker_v24._policy_action_for_snapshot(
        neural_snapshot,
        state,
        5,
        hand_ctx,
        random.Random(17),
        config=config,
        return_details=True,
    )

    legal_actions = {ACTION_FOLD, ACTION_CALL}
    assert tabular_action in legal_actions
    assert neural_action in legal_actions
    assert float(np.sum(tabular_details["policy"])) == pytest.approx(1.0)
    assert float(np.sum(neural_details["policy"])) == pytest.approx(1.0)
    assert set(np.flatnonzero(tabular_details["legal_mask"] > 0.5)) == legal_actions
    assert set(np.flatnonzero(neural_details["legal_mask"] > 0.5)) == legal_actions
