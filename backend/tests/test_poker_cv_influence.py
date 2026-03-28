"""Tests for CV-influenced bot action routing on /poker/step."""

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if "redis" not in sys.modules:
    redis_module = types.ModuleType("redis")
    redis_asyncio_module = types.ModuleType("redis.asyncio")
    redis_exceptions_module = types.ModuleType("redis.exceptions")

    class _DummyRedis:  # pragma: no cover - test bootstrap only
        pass

    class _DummyRedisConnectionError(Exception):
        pass

    class _DummyRedisTimeoutError(Exception):
        pass

    redis_asyncio_module.Redis = _DummyRedis
    redis_asyncio_module.from_url = lambda *args, **kwargs: _DummyRedis()
    redis_exceptions_module.ConnectionError = _DummyRedisConnectionError
    redis_exceptions_module.TimeoutError = _DummyRedisTimeoutError
    redis_module.asyncio = redis_asyncio_module
    redis_module.exceptions = redis_exceptions_module

    sys.modules["redis"] = redis_module
    sys.modules["redis.asyncio"] = redis_asyncio_module
    sys.modules["redis.exceptions"] = redis_exceptions_module

from backend.routers.poker import router as poker_router
from backend.services.game_service import GameService


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(poker_router)
    return TestClient(app)


def test_bot_step_returns_original_and_cv_adjusted_action(client: TestClient, monkeypatch):
    monkeypatch.setattr(GameService, "build_observation", lambda self, game_state, version=None: ([0.0], 0.57))

    class DummyModelService:
        def get_action(self, observation, legal_actions, version=None):
            del observation, legal_actions, version
            return 0, {
                "FOLD": 0.55,
                "CHECK": 0.0,
                "CALL": 0.45,
                "RAISE_SMALL": 0.05,
                "RAISE_LARGE": 0.01,
            }

    monkeypatch.setattr("backend.services.model_service.get_model_service", lambda: DummyModelService())

    response = client.post(
        "/poker/step",
        json={
            "actor": "bot",
            "model_version": "v24",
            "game_state": {
                "session_id": "test-step-cv-influence",
                "community_cards": [],
                "pot": 30,
                "players": [
                    {
                        "position": 0,
                        "stack": 990,
                        "bet": 10,
                        "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "Q", "suit": "s"}],
                        "is_bot": True,
                        "is_active": True,
                        "has_acted": False,
                    },
                    {
                        "position": 1,
                        "stack": 980,
                        "bet": 20,
                        "hole_cards": None,
                        "is_bot": False,
                        "is_active": True,
                        "has_acted": True,
                    },
                ],
                "bot_position": 0,
                "starting_stacks": [1000, 1000],
                "current_bet": 20,
                "big_blind": 10,
                "current_player_idx": 0,
                "street_raise_count": 0,
                "preflop_raise_count": 0,
                "preflop_call_count": 0,
                "preflop_last_raiser": None,
                "last_aggressor": 1,
                "cv_reads": {
                    "1": {
                        "position": 1,
                        "last_window_max_bluff_delta": 24.0,
                    }
                },
                "pot_aggressors": [1],
            },
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    applied = data["applied_action"]

    assert applied["action"] == "call"
    assert applied["original_action"] == "fold"
    assert applied["cv_influence_applied"] is True
    assert applied["cv_bluff_risk_level"] == "elevated"
    assert applied["cv_act_max"] == pytest.approx(24.0)
    assert applied["q_values"]["CALL"] > applied["original_q_values"]["CALL"]
    assert data["game_state"]["players"][0]["bet"] == 20
    assert data["game_state"]["players"][0]["stack"] == 980
