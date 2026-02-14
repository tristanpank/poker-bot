"""
FastAPI test suite using TestClient.

Run with: pytest backend/tests/test_api.py -v -s
(The -s flag shows print output)
"""

import pytest
import json
from fastapi.testclient import TestClient

from backend.main import app


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
            "big_blind": 10
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
        """Action ID should be between 0 and 5."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        assert 0 <= data["action_id"] <= 5
    
    @pytest.mark.requires_models
    def test_action_name_matches_id(self, client, valid_game_state):
        """Action name should match the action ID."""
        action_names = {
            0: "FOLD",
            1: "CALL",
            2: "RAISE_SMALL",
            3: "RAISE_MEDIUM",
            4: "RAISE_LARGE",
            5: "ALL_IN"
        }
        
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        expected_name = action_names[data["action_id"]]
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
        """Q-values should include all 6 actions."""
        response = client.post("/poker/action", json=valid_game_state)
        data = response.json()
        
        expected_keys = {"FOLD", "CALL", "RAISE_SMALL", "RAISE_MEDIUM", "RAISE_LARGE", "ALL_IN"}
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
            "big_blind": 10
        }
        
        response = client.post("/poker/action", json=game_state)
        log_response("POST /poker/action (Flop: Top Two Pair)", response)
        assert response.status_code == 200
        
        data = response.json()
        # With top two pair, equity should be high
        assert data["equity"] > 0.5


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
