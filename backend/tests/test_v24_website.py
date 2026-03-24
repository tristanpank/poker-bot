import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.models.schemas import CardSchema, GameStateRequest, PlayerState
from backend.routers.poker import router as poker_router
from backend.poker_versions import ACTION_NAMES_V24
from backend.services.game_service import get_game_service
from backend.services.model_service import get_model_service


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(poker_router)
    return TestClient(app)


@pytest.fixture
def v24_game_state() -> GameStateRequest:
    return GameStateRequest(
        community_cards=[],
        pot=15,
        players=[
            PlayerState(
                position=0,
                stack=995,
                bet=5,
                hole_cards=[CardSchema(rank="A", suit="h"), CardSchema(rank="K", suit="s")],
                is_bot=True,
                is_active=True,
                has_acted=False,
            ),
            PlayerState(
                position=1,
                stack=990,
                bet=10,
                hole_cards=None,
                is_bot=False,
                is_active=True,
                has_acted=False,
            ),
        ],
        bot_position=0,
        starting_stacks=[1000, 1000],
        current_bet=10,
        big_blind=10,
        current_player_idx=0,
        street_raise_count=0,
        preflop_raise_count=0,
        preflop_call_count=0,
        preflop_last_raiser=None,
        last_aggressor=None,
        model_version="v24",
    )


@pytest.fixture
def v24_premium_open_state() -> GameStateRequest:
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
                hole_cards=[CardSchema(rank="A", suit="h"), CardSchema(rank="A", suit="s")],
                is_bot=True,
                is_active=True,
                has_acted=False,
            ),
            PlayerState(position=3, stack=1000, bet=0, hole_cards=None, is_bot=False, is_active=True, has_acted=False),
            PlayerState(position=4, stack=1000, bet=0, hole_cards=None, is_bot=False, is_active=True, has_acted=False),
            PlayerState(position=5, stack=1000, bet=0, hole_cards=None, is_bot=False, is_active=True, has_acted=False),
        ],
        bot_position=2,
        starting_stacks=[1000] * 6,
        current_bet=10,
        big_blind=10,
        current_player_idx=2,
        street_raise_count=0,
        preflop_raise_count=0,
        preflop_call_count=0,
        preflop_last_raiser=None,
        last_aggressor=None,
        model_version="v24",
    )


def test_v24_apply_frontend_action_tracks_preflop_context(v24_game_state: GameStateRequest):
    game_service = get_game_service()

    next_state, round_complete, applied_raise_amt = game_service.apply_frontend_action(
        v24_game_state,
        0,
        "call",
    )

    assert not round_complete
    assert applied_raise_amt is None
    assert next_state.preflop_call_count == 1
    assert next_state.preflop_raise_count == 0
    assert next_state.last_aggressor is None
    assert next_state.starting_stacks == [1000, 1000]


@pytest.mark.requires_models
def test_v24_tabular_model_service_accepts_backend_observation(v24_game_state: GameStateRequest):
    game_service = get_game_service()
    model_service = get_model_service()

    observation, equity = game_service.build_observation(v24_game_state, version="v24")
    legal_actions = game_service.get_legal_actions(v24_game_state, version="v24")
    action_id, q_values = model_service.get_action(observation, legal_actions, version="v24")

    assert observation.shape == (91,)
    assert 0.0 <= equity <= 1.0
    assert action_id in legal_actions
    assert set(q_values.keys()) == set(ACTION_NAMES_V24.values())


@pytest.mark.requires_models
def test_v24_premium_preflop_open_does_not_fold(v24_premium_open_state: GameStateRequest):
    game_service = get_game_service()
    model_service = get_model_service()
    observation, _ = game_service.build_observation(v24_premium_open_state, version="v24")
    legal_actions = game_service.get_legal_actions(v24_premium_open_state, version="v24")
    action_id, q_values = model_service.get_action(observation, legal_actions, version="v24")

    assert action_id != 0
    assert q_values["FOLD"] < max(q_values["CALL"], q_values["RAISE_SMALL"], q_values["RAISE_LARGE"])


@pytest.mark.requires_models
def test_v24_step_endpoint_uses_tabular_checkpoint(client: TestClient, v24_game_state: GameStateRequest):
    response = client.post(
        "/poker/step",
        json={
            "actor": "bot",
            "model_version": "v24",
            "game_state": v24_game_state.model_dump(),
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["applied_action"]["action"] in {"fold", "call", "raise_amt"}
    assert data["game_state"]["model_version"] == "v24"
    assert "street_raise_count" in data["game_state"]
    assert "preflop_raise_count" in data["game_state"]
    assert "preflop_call_count" in data["game_state"]


@pytest.mark.requires_models
def test_v24_step_endpoint_keeps_aa_preflop_live(client: TestClient, v24_premium_open_state: GameStateRequest):
    response = client.post(
        "/poker/step",
        json={
            "actor": "bot",
            "model_version": "v24",
            "game_state": v24_premium_open_state.model_dump(),
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["applied_action"]["action"] != "fold"
