"""
Poker API router with endpoints for bot actions and health checks.
"""

from fastapi import APIRouter, HTTPException, Depends

from backend.models.schemas import (
    GameStateRequest,
    ActionResponse,
    HealthResponse,
    ModelInfo,
    ACTION_NAMES,
    HAND_STRENGTH_CATEGORIES,
)
from backend.services.model_service import get_model_service, ModelService
from backend.services.game_service import get_game_service, GameService, compute_hand_strength_category


router = APIRouter(prefix="/poker", tags=["poker"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_service: ModelService = Depends(get_model_service)
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns API status and available models.
    """
    available_models = model_service.get_available_models()
    model_loaded = any(model_service.is_loaded(v) for v in available_models)
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        available_models=available_models,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    model_service: ModelService = Depends(get_model_service)
) -> list[ModelInfo]:
    """
    List available model versions.
    """
    available = model_service.get_available_models()
    models = []
    
    for version in available:
        path = model_service.settings.get_model_path(version)
        models.append(ModelInfo(
            version=version,
            path=str(path),
            state_dim=520 if version >= "v15" else 385,
            num_actions=6,
        ))
    
    return models


@router.post("/action", response_model=ActionResponse)
async def get_action(
    game_state: GameStateRequest,
    model_service: ModelService = Depends(get_model_service),
    game_service: GameService = Depends(get_game_service),
) -> ActionResponse:
    """
    Get the bot's recommended action for the current game state.
    
    This is the primary endpoint for IRL gameplay. Send the current
    game state and receive the bot's action recommendation.
    """
    try:
        # Build observation from game state
        observation, equity = game_service.build_observation(game_state)
        
        # Get legal actions
        legal_actions = game_service.get_legal_actions(game_state)
        
        # Get model's action
        version = game_state.model_version
        action_id, q_values = model_service.get_action(
            observation, 
            legal_actions,
            version=version
        )
        
        # Calculate raise amount if applicable
        amount = game_service.calculate_raise_amount(action_id, game_state)
        
        # Get hand strength category
        strength_cat = compute_hand_strength_category(equity)
        strength_name = HAND_STRENGTH_CATEGORIES.get(strength_cat, "Unknown")
        
        return ActionResponse(
            action=ACTION_NAMES[action_id],
            action_id=action_id,
            amount=amount,
            equity=round(equity, 4),
            hand_strength_category=strength_name,
            q_values=q_values,
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
