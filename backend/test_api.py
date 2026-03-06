#!/usr/bin/env python
"""
Test script to verify the backend API functionality.

Run with: source venv/bin/activate && python backend/test_api.py
"""

import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.poker_versions import get_action_names, get_version_spec

def test_card_conversion():
    """Test Card conversion from schema to pokerkit."""
    print("Testing card conversion...")
    from backend.services.game_service import card_schema_to_pokerkit
    from backend.models.schemas import CardSchema
    
    card = CardSchema(rank='A', suit='h')
    result = card_schema_to_pokerkit(card)
    print(f"  ✅ Converted 'Ah': rank={result.rank}, suit={result.suit}")
    
    card2 = CardSchema(rank='K', suit='s')
    result2 = card_schema_to_pokerkit(card2)
    print(f"  ✅ Converted 'Ks': rank={result2.rank}, suit={result2.suit}")


def test_game_service():
    """Test GameService observation building."""
    print("\nTesting GameService...")
    from backend.services.game_service import get_game_service
    from backend.models.schemas import GameStateRequest, PlayerState, CardSchema
    
    game_service = get_game_service()
    
    # Create a test game state (preflop with AK)
    game_state = GameStateRequest(
        community_cards=[],
        pot=15,
        players=[
            PlayerState(
                position=0,
                stack=990,
                bet=5,
                hole_cards=[CardSchema(rank='A', suit='h'), CardSchema(rank='K', suit='s')],
                is_bot=True,
                is_active=True
            ),
            PlayerState(
                position=1,
                stack=990,
                bet=10,
                hole_cards=None,
                is_bot=False,
                is_active=True
            )
        ],
        bot_position=0,
        current_bet=10,
        big_blind=10,
        model_version="v24"
    )
    
    observation, equity = game_service.build_observation(game_state, version=game_state.model_version)
    print(f"  ✅ Built observation: shape={observation.shape}, equity={equity:.2f}")
    
    legal_actions = game_service.get_legal_actions(game_state, version=game_state.model_version)
    print(f"  ✅ Legal actions: {legal_actions}")


def test_model_service():
    """Test ModelService loading and inference."""
    print("\nTesting ModelService...")
    from backend.services.model_service import get_model_service
    import numpy as np
    
    model_service = get_model_service()
    
    available = model_service.get_available_models()
    print(f"  Available models: {available}")
    
    if not available:
        print("  ⚠️ No models found - skipping inference test")
        return
    
    # Load default model
    version = available[-1]  # Use latest
    try:
        model = model_service.load_model(version)
        print(f"  ✅ Loaded model: {version}")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        assert False, f"Failed to load model: {e}"
    
    # Test inference
    spec = get_version_spec(version)
    observation = np.zeros(spec.state_dim, dtype=np.float32)
    legal_actions = list(range(spec.action_dim))
    
    action, q_values = model_service.get_action(observation, legal_actions, version)
    print(f"  ✅ Inference: action={action}, q_values={list(q_values.values())[:3]}...")


def test_full_pipeline():
    """Test the complete action pipeline."""
    print("\nTesting full action pipeline...")
    from backend.services.game_service import get_game_service
    from backend.services.model_service import get_model_service
    from backend.models.schemas import (
        GameStateRequest, PlayerState, CardSchema, 
        HAND_STRENGTH_CATEGORIES
    )
    from backend.services.game_service import compute_hand_strength_category
    
    game_service = get_game_service()
    model_service = get_model_service()
    
    available = model_service.get_available_models()
    if not available:
        print("  ⚠️ No models - skipping full pipeline test")
        return
    
    # Create game state
    game_state = GameStateRequest(
        community_cards=[],
        pot=15,
        players=[
            PlayerState(
                position=0,
                stack=990,
                bet=5,
                hole_cards=[CardSchema(rank='A', suit='h'), CardSchema(rank='K', suit='s')],
                is_bot=True,
                is_active=True
            ),
            PlayerState(
                position=1,
                stack=990,
                bet=10,
                hole_cards=None,
                is_bot=False,
                is_active=True
            )
        ],
        bot_position=0,
        current_bet=10,
        big_blind=10,
        model_version="v24"
    )
    
    # Build observation
    version = game_state.model_version
    observation, equity = game_service.build_observation(game_state, version=version)
    legal_actions = game_service.get_legal_actions(game_state, version=version)
    
    # Get action
    action_id, q_values = model_service.get_action(observation, legal_actions, version=version)
    
    # Calculate amount
    amount = game_service.calculate_raise_amount(action_id, game_state, version=version)
    
    # Get hand strength
    strength_cat = compute_hand_strength_category(equity)
    
    action_names = get_action_names(version)
    print(f"  âœ… Action: {action_names[action_id]} (id={action_id})")
    print(f"  ✅ Amount: {amount}")
    print(f"  ✅ Equity: {equity:.2%}")
    print(f"  ✅ Hand Strength: {HAND_STRENGTH_CATEGORIES[strength_cat]}")


if __name__ == "__main__":
    print("=" * 50)
    print("Backend API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Card Conversion", test_card_conversion),
        ("Game Service", test_game_service),
        ("Model Service", test_model_service),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    for name, success, error in results:
        status = "✅ PASS" if success else f"❌ FAIL: {error}"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed! 🎉" if all_passed else "Some tests failed."))


