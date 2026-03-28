"""Focused tests for per-player CV acting-window summaries on poker steps."""

import sys
import time
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
from backend.services.session_service import (
    append_webcam_metric_sample,
    clear_webcam_metric_history,
)


@pytest.fixture(autouse=True)
def clear_metric_history_between_tests():
    clear_webcam_metric_history()
    yield
    clear_webcam_metric_history()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(poker_router)
    return TestClient(app)


def _base_game_state(*, session_id: str, players: list[dict], current_player_idx: int, current_bet: int, pot: int):
    return {
        "session_id": session_id,
        "community_cards": [],
        "pot": pot,
        "players": players,
        "bot_position": 0,
        "starting_stacks": [1000 for _ in players],
        "current_bet": current_bet,
        "big_blind": 10,
        "current_player_idx": current_player_idx,
        "street_raise_count": 0,
        "preflop_raise_count": 0,
        "preflop_call_count": 0,
        "preflop_last_raiser": None,
        "last_aggressor": None,
        "cv_reads": {},
        "pot_aggressors": [],
        "model_version": "v24",
    }


def test_opponent_raise_tracks_window_summary_and_marks_pot_aggressor(client: TestClient):
    session_id = "test-session-cv-raise"
    cv_session_id = f"{session_id}__p1"
    now_ms = int(time.time() * 1000.0)
    window_start_ms = now_ms - 5_000

    append_webcam_metric_sample(
        cv_session_id,
        {"bluffRisk": 35.0, "bluffDelta": 3.0},
        timestamp_ms=window_start_ms - 800,
    )
    append_webcam_metric_sample(
        cv_session_id,
        {"bluffRisk": 52.0, "bluffDelta": 10.0},
        timestamp_ms=window_start_ms + 600,
    )
    append_webcam_metric_sample(
        cv_session_id,
        {"bluffRisk": 65.0, "bluffDelta": 20.0},
        timestamp_ms=window_start_ms + 1_700,
    )

    game_state = _base_game_state(
        session_id=session_id,
        current_player_idx=1,
        current_bet=10,
        pot=20,
        players=[
            {
                "position": 0,
                "stack": 990,
                "bet": 10,
                "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "s"}],
                "is_bot": True,
                "is_active": True,
                "has_acted": True,
            },
            {
                "position": 1,
                "stack": 990,
                "bet": 10,
                "hole_cards": None,
                "is_bot": False,
                "is_active": True,
                "has_acted": False,
            },
        ],
    )
    game_state["cv_reads"] = {
        "1": {
            "position": 1,
            "current_window_started_at_ms": window_start_ms,
        }
    }

    response = client.post(
        "/poker/step",
        json={
            "actor": "opponent",
            "action": "raise_amt",
            "raise_amt": 20,
            "game_state": game_state,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    read = data["game_state"]["cv_reads"]["1"]

    assert read["last_window_sample_count"] == 2
    assert read["last_window_avg_bluff_delta"] == pytest.approx(15.0)
    assert read["last_window_max_bluff_delta"] == pytest.approx(20.0)
    assert read["orbit_avg_bluff_delta"] == pytest.approx(15.0)
    assert read["orbit_max_bluff_delta"] == pytest.approx(20.0)
    assert read["orbit_window_count"] == 1
    assert read["orbit_sample_count"] == 2
    assert read["was_aggressor_this_pot"] is True
    assert data["game_state"]["pot_aggressors"] == [1]


def test_opponent_call_tracks_window_summary_and_primes_next_opponent_window(client: TestClient):
    session_id = "test-session-cv-call"
    cv_session_id = f"{session_id}__p1"
    now_ms = int(time.time() * 1000.0)
    window_start_ms = now_ms - 4_000

    append_webcam_metric_sample(
        cv_session_id,
        {"bluffRisk": 48.0, "bluffDelta": 6.0},
        timestamp_ms=window_start_ms + 250,
    )
    append_webcam_metric_sample(
        cv_session_id,
        {"bluffRisk": 58.0, "bluffDelta": 14.0},
        timestamp_ms=window_start_ms + 1_250,
    )

    game_state = _base_game_state(
        session_id=session_id,
        current_player_idx=1,
        current_bet=10,
        pot=25,
        players=[
            {
                "position": 0,
                "stack": 990,
                "bet": 10,
                "hole_cards": [{"rank": "Q", "suit": "h"}, {"rank": "Q", "suit": "s"}],
                "is_bot": True,
                "is_active": True,
                "has_acted": True,
            },
            {
                "position": 1,
                "stack": 995,
                "bet": 5,
                "hole_cards": None,
                "is_bot": False,
                "is_active": True,
                "has_acted": False,
            },
            {
                "position": 2,
                "stack": 990,
                "bet": 10,
                "hole_cards": None,
                "is_bot": False,
                "is_active": True,
                "has_acted": False,
            },
        ],
    )
    game_state["cv_reads"] = {
        "1": {
            "position": 1,
            "current_window_started_at_ms": window_start_ms,
        }
    }

    response = client.post(
        "/poker/step",
        json={
            "actor": "opponent",
            "action": "call",
            "game_state": game_state,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    actor_read = data["game_state"]["cv_reads"]["1"]
    next_read = data["game_state"]["cv_reads"]["2"]

    assert actor_read["last_window_sample_count"] == 2
    assert actor_read["last_window_avg_bluff_delta"] == pytest.approx(10.0)
    assert actor_read["last_window_max_bluff_delta"] == pytest.approx(14.0)
    assert actor_read["was_aggressor_this_pot"] is False
    assert data["game_state"]["pot_aggressors"] == []
    assert data["game_state"]["current_player_idx"] == 2
    assert next_read["position"] == 2
    assert next_read["current_window_started_at_ms"] is not None
    assert next_read["current_window_started_at_ms"] >= actor_read["last_window_ended_at_ms"]
