"""
FastAPI test suite using TestClient.

Run with: pytest backend/tests/test_api.py -v -s
(The -s flag shows print output)
"""

import pytest
import json
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.schemas import GameStateRequest
from backend.poker_versions import ACTION_NAMES_V24, ACTION_NAMES_V25
from backend.services.game_service import GameService


def log_response(test_name: str, response):
    """Pretty print API response for debugging."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2)}")
    except:
        print(f"Response: {response.text}")
    print(f"{'='*60}\n")


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /poker/health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/poker/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should have expected fields."""
        response = client.get("/poker/health")
        log_response("GET /poker/health", response)
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "available_models" in data
        assert isinstance(data["available_models"], list)


class TestRootEndpoint:
    """Tests for the root / endpoint."""
    
    def test_root_returns_200(self, client):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_has_name_and_version(self, client):
        """Root should return API name and version."""
        response = client.get("/")
        log_response("GET /", response)
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestModelsEndpoint:
    """Tests for the /poker/models endpoint."""
    
    def test_models_returns_200(self, client):
        """Models endpoint should return 200 OK."""
        response = client.get("/poker/models")
        assert response.status_code == 200
    
    def test_models_returns_list(self, client):
        """Models endpoint should return a list."""
        response = client.get("/poker/models")
        log_response("GET /poker/models", response)
        data = response.json()
        
        assert isinstance(data, list)


class TestWarmupEndpoint:
    """Tests for the /poker/warmup endpoint."""

    @pytest.mark.requires_models
    def test_warmup_returns_200(self, client):
        response = client.post("/poker/warmup", params={"version": "v24"})
        assert response.status_code == 200

    @pytest.mark.requires_models
    def test_warmup_response_structure(self, client):
        response = client.post("/poker/warmup", params={"version": "v24"})
        log_response("POST /poker/warmup", response)
        data = response.json()

        assert data["status"] == "ready"
        assert data["version"] == "v24"
        assert data["model_loaded"] is True
        assert "already_loaded" in data


class TestActionEndpoint:
    """Tests for the /poker/action endpoint."""
    
    @pytest.fixture
    def valid_game_state(self):
        """A valid game state for testing."""
        return {
            "community_cards": [],
            "pot": 15,
            "players": [
                {
                    "position": 0,
                    "stack": 990,
                    "bet": 5,
                    "hole_cards": [
                        {"rank": "A", "suit": "h"},
                        {"rank": "K", "suit": "s"}
                    ],
                    "is_bot": True,
                    "is_active": True
                },
                {
                    "position": 1,
                    "stack": 990,
                    "bet": 10,
                    "hole_cards": None,
                    "is_bot": False,
                    "is_active": True
                }
            ],
            "bot_position": 0,
            "current_bet": 10,
            "big_blind": 10,
            "model_version": "v24",
        }
    
    @pytest.mark.requires_models
    def test_action_returns_200(self, client, valid_game_state):
        """Action endpoint should return 200 OK for valid input."""
        response = client.post("/poker/action", json=valid_game_state)
        assert response.status_code == 200
    
    @pytest.mark.requires_models
    def test_action_response_structure(self, client, valid_game_state):
        """Action response should have all required fields."""
        response = client.post("/poker/action", json=valid_game_state)
        log_response("POST /poker/action (Preflop AK)", response)
        data = response.json()
        
        assert "action" in data
        assert "action_id" in data
        assert "equity" in data
        assert "hand_strength_category" in data
        assert "q_values" in data
    
    @pytest.mark.requires_models
    def test_action_id_is_valid(self, client, valid_game_state):
        """Action ID should be within the v24 action space."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        assert 0 <= data["action_id"] <= max(ACTION_NAMES_V24)
    
    @pytest.mark.requires_models
    def test_action_name_matches_id(self, client, valid_game_state):
        """Action name should match the action ID."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        expected_name = ACTION_NAMES_V24[data["action_id"]]
        assert data["action"] == expected_name
    
    @pytest.mark.requires_models
    def test_equity_is_valid_range(self, client, valid_game_state):
        """Equity should be between 0 and 1."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        assert 0 <= data["equity"] <= 1
    
    @pytest.mark.requires_models
    def test_hand_strength_is_valid(self, client, valid_game_state):
        """Hand strength category should be a valid value."""
        valid_categories = ["Trash", "Marginal", "Decent", "Strong", "Monster"]
        
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        assert data["hand_strength_category"] in valid_categories
    
    @pytest.mark.requires_models
    def test_q_values_has_all_actions(self, client, valid_game_state):
        """Q-values should include the current v24 action set."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        expected_keys = set(ACTION_NAMES_V24.values())
        assert set(data["q_values"].keys()) == expected_keys
    
    @pytest.mark.requires_models
    def test_action_with_flop(self, client):
        """Test action with community cards on the flop."""
        game_state = {
            "community_cards": [
                {"rank": "A", "suit": "s"},
                {"rank": "K", "suit": "h"},
                {"rank": "7", "suit": "d"}
            ],
            "pot": 50,
            "players": [
                {
                    "position": 0,
                    "stack": 950,
                    "bet": 0,
                    "hole_cards": [
                        {"rank": "A", "suit": "h"},
                        {"rank": "K", "suit": "s"}
                    ],
                    "is_bot": True,
                    "is_active": True
                },
                {
                    "position": 1,
                    "stack": 950,
                    "bet": 0,
                    "hole_cards": None,
                    "is_bot": False,
                    "is_active": True
                }
            ],
            "bot_position": 0,
            "current_bet": 0,
            "big_blind": 10,
            "model_version": "v24",
        }
        
        response = client.post("/poker/action", json=game_state)
        log_response("POST /poker/action (Flop: Top Two Pair)", response)
        assert response.status_code == 200
        
        data = response.json()
        # With top two pair, equity should be high
        assert data["equity"] > 0.5

    def test_v25_runtime_action_route_uses_game_service_resolver(self, client, monkeypatch, valid_game_state):
        game_state = dict(valid_game_state)
        game_state["model_version"] = "v25"

        expected_q = {name: 0.0 for name in ACTION_NAMES_V25.values()}
        expected_q["CALL"] = 0.75
        expected_q["RAISE_SMALL"] = 0.25

        monkeypatch.setattr(
            GameService,
            "get_runtime_action_for_actor",
            lambda self, game_state, actor_index, version=None: (2, dict(expected_q)),
        )

        class DummyModelService:
            def get_action(self, *args, **kwargs):
                raise AssertionError("Legacy observation-only inference should not be used for v25 runtime resolving.")

        monkeypatch.setattr("backend.services.model_service.get_model_service", lambda: DummyModelService())

        response = client.post("/poker/action", json=game_state)
        assert response.status_code == 200
        data = response.json()
        assert data["action_id"] == 2
        assert data["action"] == "CALL"
        assert data["q_values"] == expected_q


class TestV25RuntimeResolving:
    def test_game_service_runtime_action_for_actor_uses_resolved_policy(self, monkeypatch):
        import numpy as np
        import backend.services.game_service as game_service_module

        game_state = GameStateRequest.model_validate(
            {
                "community_cards": [],
                "pot": 15,
                "players": [
                    {
                        "position": 0,
                        "stack": 990,
                        "bet": 5,
                        "hole_cards": [
                            {"rank": "A", "suit": "h"},
                            {"rank": "K", "suit": "s"},
                        ],
                        "is_bot": True,
                        "is_active": True,
                    },
                    {
                        "position": 1,
                        "stack": 990,
                        "bet": 10,
                        "hole_cards": None,
                        "is_bot": False,
                        "is_active": True,
                    },
                ],
                "bot_position": 0,
                "current_bet": 10,
                "big_blind": 10,
                "current_player_idx": 0,
                "model_version": "v25",
            }
        )

        class DummyModelService:
            def load_model(self, version):
                assert version == "v25"
                return object()

        monkeypatch.setattr("backend.services.model_service.get_model_service", lambda: DummyModelService())

        def fake_runtime_policy(snapshot, state, actor, hand_ctx, rng, config=None, return_details=False, sample_action=True, **kwargs):
            del snapshot, state, actor, hand_ctx, rng, kwargs
            assert config.runtime_subgame_resolving_enabled is True
            assert sample_action is False
            policy = np.array([0.05, 0.0, 0.70, 0.20, 0.05, 0.0, 0.0], dtype=np.float32)
            if return_details:
                return 2, {"policy": policy, "legal_mask": np.array([1, 0, 1, 1, 1, 0, 0], dtype=np.float32), "resolved": True}
            return 2

        monkeypatch.setattr(game_service_module, "runtime_policy_action_for_snapshot_v25", fake_runtime_policy)

        service = GameService()
        action_id, q_values = service.get_runtime_action_for_actor(game_state, 0, version="v25")

        assert action_id == 2
        assert q_values["CALL"] == pytest.approx(0.70)
        assert q_values["RAISE_SMALL"] == pytest.approx(0.20)
        assert set(q_values.keys()) == set(ACTION_NAMES_V25.values())


class TestActionEndpointErrors:
    """Tests for error handling in the action endpoint."""
    
    def test_missing_players_returns_422(self, client):
        """Missing required field should return 422."""
        game_state = {
            "community_cards": [],
            "pot": 15,
            # Missing players
            "bot_position": 0,
            "current_bet": 10,
            "big_blind": 10
        }
        
        response = client.post("/poker/action", json=game_state)
        assert response.status_code == 422
    
    def test_no_bot_player_returns_400(self, client):
        """No bot player should return 400."""
        game_state = {
            "community_cards": [],
            "pot": 15,
            "players": [
                {
                    "position": 0,
                    "stack": 990,
                    "bet": 5,
                    "hole_cards": None,
                    "is_bot": False,  # No bot!
                    "is_active": True
                },
                {
                    "position": 1,
                    "stack": 990,
                    "bet": 10,
                    "hole_cards": None,
                    "is_bot": False,
                    "is_active": True
                }
            ],
            "bot_position": 0,
            "current_bet": 10,
            "big_blind": 10
        }
        
        response = client.post("/poker/action", json=game_state)
        assert response.status_code == 400
    
    def test_bot_without_hole_cards_returns_400(self, client):
        """Bot without hole cards should return 400."""
        game_state = {
            "community_cards": [],
            "pot": 15,
            "players": [
                {
                    "position": 0,
                    "stack": 990,
                    "bet": 5,
                    "hole_cards": None,  # No hole cards!
                    "is_bot": True,
                    "is_active": True
                },
                {
                    "position": 1,
                    "stack": 990,
                    "bet": 10,
                    "hole_cards": None,
                    "is_bot": False,
                    "is_active": True
                }
            ],
            "bot_position": 0,
            "current_bet": 10,
            "big_blind": 10
        }
        
        response = client.post("/poker/action", json=game_state)
        assert response.status_code == 400
