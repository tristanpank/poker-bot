"""
Poker API router with endpoints for bot actions and health checks.
"""

import re

from fastapi import APIRouter, HTTPException, Depends

from backend.config import get_settings
from backend.models.schemas import (
    GameStateRequest,
    ActionResponse,
    AppliedAction,
    HandResolveRequest,
    HandResolveResponse,
    HealthResponse,
    LegalActionsResponse,
    ModelInfo,
    PokerStepRequest,
    PokerStepResponse,
    ACTION_NAMES,
    HAND_STRENGTH_CATEGORIES,
)
from backend.services.game_service import get_game_service, GameService, compute_hand_strength_category


router = APIRouter(prefix="/poker", tags=["poker"])


def _version_to_int(version: str) -> int:
    match = re.search(r"(\d+)", (version or "").lower())
    if match is None:
        return 0
    return int(match.group(1))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns API status and available models.
    """
    settings = get_settings()
    available_models = settings.get_available_models()
    
    return HealthResponse(
        status="healthy",
        # Keep health lightweight and independent of torch/model runtime imports.
        model_loaded=False,
        available_models=available_models,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """
    List available model versions.
    """
    settings = get_settings()
    available = settings.get_available_models()
    models = []
    
    for version in available:
        path = settings.get_model_path(version)
        version_num = _version_to_int(version)
        models.append(ModelInfo(
            version=version,
            path=str(path),
            state_dim=98 if version_num >= 21 else (520 if version_num >= 15 else 385),
            num_actions=5 if version_num >= 21 else 6,
        ))
    
    return models


@router.post("/legal", response_model=LegalActionsResponse)
async def get_legal_actions(
    game_state: GameStateRequest,
    game_service: GameService = Depends(get_game_service),
) -> LegalActionsResponse:
    """Return legal actions for the current actor index in the provided state."""
    info = game_service.get_legal_action_info(game_state, game_state.current_player_idx)
    return LegalActionsResponse(**info)


@router.post("/step", response_model=PokerStepResponse)
async def step_action(
    request: PokerStepRequest,
    game_service: GameService = Depends(get_game_service),
) -> PokerStepResponse:
    """
    Apply one action on the backend and return updated state + next legal action set.

    - `actor="bot"`: backend queries the model, normalizes to fold/check/call/raise_amt, applies it.
    - `actor="opponent"`: backend validates and applies the provided frontend action.
    """
    try:
        game_state = request.game_state
        actor_index = int(game_state.current_player_idx)
        if actor_index < 0:
            raise ValueError("No active actor: current_player_idx is -1")
        if actor_index >= len(game_state.players):
            raise ValueError("current_player_idx is out of range")

        if request.actor == "bot":
            if not game_state.players[actor_index].is_bot:
                raise ValueError("Current actor is not the bot")

            observation, equity = game_service.build_observation(game_state)
            legal_action_ids = game_service.get_legal_action_ids_for_actor(game_state, actor_index)
            if not legal_action_ids:
                raise ValueError("No legal actions available for bot actor")

            try:
                from backend.services.model_service import get_model_service
                model_service = get_model_service()
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model service unavailable: {str(e)}",
                ) from e

            version = request.model_version or game_state.model_version
            action_id, q_values = model_service.get_action(observation, legal_action_ids, version=version)
            normalized_action = game_service.model_action_to_frontend(action_id)
            raise_amt = game_service.calculate_raise_amount_for_actor(action_id, game_state, actor_index)

            next_state, round_complete, applied_raise_amt = game_service.apply_frontend_action(
                game_state,
                actor_index,
                normalized_action,
                raise_amt,
            )

            strength_cat = compute_hand_strength_category(equity)
            strength_name = HAND_STRENGTH_CATEGORIES.get(strength_cat, "Unknown")
            applied_action = AppliedAction(
                action=normalized_action,
                raise_amt=applied_raise_amt,
                action_id=action_id,
                equity=round(equity, 4),
                hand_strength_category=strength_name,
                q_values=q_values,
            )
        else:
            if request.action is None:
                raise ValueError("action is required when actor='opponent'")

            next_state, round_complete, applied_raise_amt = game_service.apply_frontend_action(
                game_state,
                actor_index,
                request.action,
                request.raise_amt,
            )
            applied_action = AppliedAction(
                action=request.action,
                raise_amt=applied_raise_amt if request.action == "raise_amt" else None,
            )

        legal_info = game_service.get_legal_action_info(next_state, next_state.current_player_idx)
        return PokerStepResponse(
            game_state=next_state,
            applied_action=applied_action,
            legal_actions=LegalActionsResponse(**legal_info),
            round_complete=round_complete,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")


@router.post("/resolve", response_model=HandResolveResponse)
async def resolve_hand(
    request: HandResolveRequest,
    game_service: GameService = Depends(get_game_service),
) -> HandResolveResponse:
    """
    Resolve final hand result for the bot using current game state and revealed opponent cards.
    """
    try:
        opponent_map = {
            int(item.player_index): {"hole_cards": item.hole_cards, "mucked": item.mucked}
            for item in request.opponents
        }
        resolved = game_service.resolve_hand_result(
            request.game_state,
            request.starting_stacks,
            opponent_map,
        )
        return HandResolveResponse(**resolved)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resolve error: {str(e)}")


@router.post("/action", response_model=ActionResponse)
async def get_action(
    game_state: GameStateRequest,
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

        # Lazily import model service so route registration doesn't require torch.
        try:
            from backend.services.model_service import get_model_service
            model_service = get_model_service()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model service unavailable: {str(e)}",
            ) from e
        
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
