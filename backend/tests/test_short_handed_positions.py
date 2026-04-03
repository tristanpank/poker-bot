import importlib

import numpy as np
import pytest

from backend.models.schemas import CardSchema, GameStateRequest, PlayerState
from backend.services.game_service import get_game_service


def _three_handed_button_state(*, version: str) -> GameStateRequest:
    return GameStateRequest(
        community_cards=[],
        pot=15,
        players=[
            PlayerState(position=0, stack=995, bet=5, hole_cards=None, is_bot=False, is_active=True, has_acted=False),
            PlayerState(position=1, stack=990, bet=10, hole_cards=None, is_bot=False, is_active=True, has_acted=False),
            PlayerState(
                position=2,
                stack=1000,
                bet=0,
                hole_cards=[CardSchema(rank="K", suit="h"), CardSchema(rank="9", suit="h")],
                is_bot=True,
                is_active=True,
                has_acted=False,
            ),
        ],
        bot_position=2,
        starting_stacks=[1000, 1000, 1000],
        current_bet=10,
        big_blind=10,
        current_player_idx=2,
        street_raise_count=0,
        preflop_raise_count=0,
        preflop_call_count=0,
        preflop_last_raiser=None,
        last_aggressor=None,
        model_version=version,
    )


@pytest.mark.parametrize("version", ["v24", "v25"])
def test_short_handed_observation_maps_three_handed_button_to_btn_bucket(version: str):
    game_service = get_game_service()
    game_state = _three_handed_button_state(version=version)

    observation, _ = game_service.build_observation(game_state, version=version)

    position_slice = observation[4:10]
    assert position_slice.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("module_name", ["preflop_blueprint_v24", "preflop_blueprint_v25"])
def test_short_handed_preflop_blueprint_maps_three_handed_button_to_btn(monkeypatch, module_name: str):
    importlib.import_module("backend.services.game_service")
    module = importlib.import_module(module_name)

    captured: dict[str, int] = {}
    hole_cards = [
        next(module.Card.parse("Kh")),
        next(module.Card.parse("9h")),
    ]

    def fake_build_chart(stack_bucket, spot_name, actor_seat, aggressor_seat, in_position):
        captured["actor_seat"] = int(actor_seat)
        hand_key = module.canonical_preflop_hand_key(hole_cards)
        return {hand_key: ("open_raise", "fold")}

    monkeypatch.setattr(module, "_build_chart", fake_build_chart)

    action_count = 5 if module_name.endswith("v24") else 7
    policy, meta = module.preflop_blueprint_policy(
        hole_cards=hole_cards,
        actor_seat=2,
        legal_mask=np.ones(action_count, dtype=np.float32),
        effective_stack_bb=100.0,
        to_call_bb=0.0,
        preflop_raise_count=0,
        preflop_call_count=0,
        aggressor_seat=None,
        player_count=3,
    )

    assert captured["actor_seat"] == 5
    assert meta["covered"] is True
    assert float(policy.sum()) == pytest.approx(1.0)
