"""
Poker API router with endpoints for bot actions and health checks.
"""

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
    ModelWarmupResponse,
    PokerStepRequest,
    PokerStepResponse,
    HAND_STRENGTH_CATEGORIES,
)
from backend.poker_versions import get_action_names, get_version_spec
from backend.services.game_service import get_game_service, GameService, compute_hand_strength_category


router = APIRouter(prefix="/poker", tags=["poker"])


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
        spec = get_version_spec(version)
        models.append(ModelInfo(
            version=version,
            path=str(path),
            state_dim=spec.state_dim,
            num_actions=spec.action_dim,
        ))
    
    return models


@router.post("/warmup", response_model=ModelWarmupResponse)
async def warmup_model(version: str | None = None) -> ModelWarmupResponse:
    """
    Explicitly load a model into memory so the first real action request is fast.
    """
    settings = get_settings()
    target_version = str(version or settings.model_version).lower()

    try:
        from backend.services.model_service import get_model_service

        model_service = get_model_service()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable: {str(e)}",
        ) from e

    try:
        already_loaded = model_service.is_loaded(target_version)
        model_service.load_model(target_version)
        return ModelWarmupResponse(
            status="ready",
            version=target_version,
            model_loaded=True,
            already_loaded=already_loaded,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup error: {str(e)}")


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

            version = request.model_version or game_state.model_version
            observation, equity = game_service.build_observation(game_state, version=version)
            legal_action_ids = game_service.get_legal_action_ids_for_actor(game_state, actor_index, version=version)
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

            action_id, q_values = model_service.get_action(observation, legal_action_ids, version=version)
            normalized_action = game_service.model_action_to_frontend(action_id)
            raise_amt = game_service.calculate_raise_amount_for_actor(action_id, game_state, actor_index, version=version)

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
        version = game_state.model_version
        observation, equity = game_service.build_observation(game_state, version=version)
        
        # Get legal actions
        legal_actions = game_service.get_legal_actions(game_state, version=version)

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
        action_id, q_values = model_service.get_action(
            observation, 
            legal_actions,
            version=version
        )
        
        # Calculate raise amount if applicable
        amount = game_service.calculate_raise_amount(action_id, game_state, version=version)
        
        # Get hand strength category
        strength_cat = compute_hand_strength_category(equity)
        strength_name = HAND_STRENGTH_CATEGORIES.get(strength_cat, "Unknown")
        action_names = get_action_names(version)
        
        return ActionResponse(
            action=action_names.get(action_id, f"ACTION_{action_id}"),
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
