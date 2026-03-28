"""
Poker API router with endpoints for bot actions and health checks.
"""

import time

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
    PlayerCvRead,
    PokerStepRequest,
    PokerStepResponse,
    HAND_STRENGTH_CATEGORIES,
)
from backend.poker_versions import get_action_names, get_version_spec, version_to_int
from backend.services.game_service import get_game_service, GameService, compute_hand_strength_category
from backend.services.session_service import summarize_webcam_metric_window


router = APIRouter(prefix="/poker", tags=["poker"])

_CV_SIGNAL_WATCH_ACT_MAX = 8.0
_CV_SIGNAL_ELEVATED_ACT_MAX = 16.0
_CV_SIGNAL_MAX_ACT_MAX = 28.0


def _derive_cv_bluff_risk_level_from_act_max(act_max: float | None) -> str | None:
    if act_max is None:
        return None
    if act_max >= _CV_SIGNAL_ELEVATED_ACT_MAX:
        return "elevated"
    if act_max >= _CV_SIGNAL_WATCH_ACT_MAX:
        return "watch"
    return "low"


def _get_relevant_cv_act_max(game_state: GameStateRequest) -> float | None:
    pot_aggressors = {int(position) for position in (game_state.pot_aggressors or [])}
    relevant_positions = {
        int(player.position)
        for player in game_state.players
        if (not player.is_bot) and (player.is_active or int(player.position) in pot_aggressors)
    }
    if not relevant_positions:
        return None

    strongest_act_max: float | None = None
    for read in (game_state.cv_reads or {}).values():
        if int(read.position) not in relevant_positions:
            continue
        if read.last_window_max_bluff_delta is None:
            continue
        if strongest_act_max is None or read.last_window_max_bluff_delta > strongest_act_max:
            strongest_act_max = float(read.last_window_max_bluff_delta)
    return strongest_act_max


def _reweight_bot_action_for_cv_signal(
    game_state: GameStateRequest,
    *,
    actor_index: int,
    version: str | None,
    legal_action_ids: list[int],
    original_action_id: int,
    original_q_values: dict[str, float],
) -> tuple[int, dict[str, float], bool, float | None, str | None]:
    act_max = _get_relevant_cv_act_max(game_state)
    bluff_risk_level = _derive_cv_bluff_risk_level_from_act_max(act_max)
    if bluff_risk_level != "elevated" or act_max is None:
        return original_action_id, dict(original_q_values), False, act_max, bluff_risk_level

    severity = max(0.0, min(1.0, (act_max - _CV_SIGNAL_ELEVATED_ACT_MAX) / (_CV_SIGNAL_MAX_ACT_MAX - _CV_SIGNAL_ELEVATED_ACT_MAX)))
    actor = game_state.players[actor_index]
    to_call = max(0.0, float(game_state.current_bet - actor.bet))
    action_names = get_action_names(version)
    adjusted_q_values = dict(original_q_values)
    base_shift = 0.08 + (0.18 * severity)

    def _bump(action_name: str, delta: float) -> None:
        if action_name not in adjusted_q_values:
            adjusted_q_values[action_name] = 0.0
        adjusted_q_values[action_name] = float(adjusted_q_values[action_name] + delta)

    if to_call > 0.0:
        _bump("FOLD", -base_shift)
        _bump("CALL", base_shift)
        _bump("RAISE_SMALL", base_shift * 0.18)
        _bump("RAISE_MEDIUM", base_shift * 0.12)
        _bump("RAISE_LARGE", -base_shift * 0.08)
        _bump("ALL_IN", -base_shift * 0.12)
    else:
        _bump("CHECK", base_shift * 0.55)
        _bump("RAISE_SMALL", base_shift * 0.18)
        _bump("RAISE_MEDIUM", base_shift * 0.10)
        _bump("RAISE_LARGE", base_shift * 0.04)

    adjusted_action_id = max(
        legal_action_ids,
        key=lambda action_id: adjusted_q_values.get(action_names.get(action_id, f"ACTION_{action_id}"), float("-inf")),
    )
    return int(adjusted_action_id), adjusted_q_values, True, act_max, bluff_risk_level


def _copy_cv_reads(game_state: GameStateRequest) -> dict[str, PlayerCvRead]:
    return {
        str(position): read.model_copy(deep=True)
        for position, read in (game_state.cv_reads or {}).items()
    }


def _merge_cv_window_summary(
    existing: PlayerCvRead,
    summary: dict[str, object],
) -> PlayerCvRead:
    sample_count = int(summary.get("sample_count") or 0)
    existing.last_window_started_at_ms = int(summary.get("started_at_ms") or 0) or None
    existing.last_window_ended_at_ms = int(summary.get("ended_at_ms") or 0) or None
    existing.last_window_sample_count = sample_count
    existing.last_window_avg_bluff_delta = (
        float(summary["avg_bluff_delta"])
        if summary.get("avg_bluff_delta") is not None
        else None
    )
    existing.last_window_max_bluff_delta = (
        float(summary["max_bluff_delta"])
        if summary.get("max_bluff_delta") is not None
        else None
    )
    existing.current_window_started_at_ms = None

    if sample_count <= 0 or existing.last_window_avg_bluff_delta is None:
        return existing

    prior_sample_count = int(existing.orbit_sample_count)
    total_sample_count = prior_sample_count + sample_count
    if prior_sample_count > 0 and existing.orbit_avg_bluff_delta is not None:
        existing.orbit_avg_bluff_delta = (
            (existing.orbit_avg_bluff_delta * prior_sample_count)
            + (existing.last_window_avg_bluff_delta * sample_count)
        ) / float(total_sample_count)
    else:
        existing.orbit_avg_bluff_delta = existing.last_window_avg_bluff_delta

    if existing.last_window_max_bluff_delta is not None:
        if existing.orbit_max_bluff_delta is None:
            existing.orbit_max_bluff_delta = existing.last_window_max_bluff_delta
        else:
            existing.orbit_max_bluff_delta = max(
                existing.orbit_max_bluff_delta,
                existing.last_window_max_bluff_delta,
            )

    existing.orbit_sample_count = total_sample_count
    existing.orbit_window_count += 1
    return existing


def _update_cv_read_tracking(
    game_state: GameStateRequest,
    next_state: GameStateRequest,
    *,
    actor_index: int,
    applied_action: str,
    now_ms: int,
) -> None:
    cv_reads = _copy_cv_reads(game_state)
    pot_aggressors = list(dict.fromkeys(int(pos) for pos in (game_state.pot_aggressors or [])))

    if game_state.session_id and 0 <= actor_index < len(game_state.players):
        actor = game_state.players[actor_index]
        actor_position = int(actor.position)
        actor_key = str(actor_position)
        actor_read = cv_reads.get(actor_key, PlayerCvRead(position=actor_position))

        if not actor.is_bot:
            window_start_ms = actor_read.current_window_started_at_ms
            if window_start_ms is None:
                window_start_ms = max(0, now_ms - 8_000)

            summary = summarize_webcam_metric_window(
                f"{game_state.session_id}__p{actor_position}",
                started_at_ms=window_start_ms,
                ended_at_ms=now_ms,
            )
            actor_read = _merge_cv_window_summary(actor_read, summary)

            if applied_action == "raise_amt":
                actor_read.was_aggressor_this_pot = True
                if actor_position not in pot_aggressors:
                    pot_aggressors.append(actor_position)

            cv_reads[actor_key] = actor_read

    next_state.session_id = game_state.session_id
    next_state.cv_reads = cv_reads
    next_state.pot_aggressors = pot_aggressors

    next_actor_index = int(next_state.current_player_idx)
    if 0 <= next_actor_index < len(next_state.players):
        next_actor = next_state.players[next_actor_index]
        if not next_actor.is_bot:
            next_key = str(int(next_actor.position))
            next_read = cv_reads.get(next_key, PlayerCvRead(position=int(next_actor.position)))
            next_read.current_window_started_at_ms = now_ms
            cv_reads[next_key] = next_read


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
        step_now_ms = int(time.time() * 1000.0)
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
            original_action_id = int(action_id)
            original_q_values = dict(q_values)
            original_normalized_action = game_service.model_action_to_frontend(original_action_id)
            original_raise_amt = game_service.calculate_raise_amount_for_actor(
                original_action_id,
                game_state,
                actor_index,
                version=version,
            )

            adjusted_action_id, adjusted_q_values, cv_influence_applied, cv_act_max, cv_bluff_risk_level = _reweight_bot_action_for_cv_signal(
                game_state,
                actor_index=actor_index,
                version=version,
                legal_action_ids=legal_action_ids,
                original_action_id=original_action_id,
                original_q_values=original_q_values,
            )
            normalized_action = game_service.model_action_to_frontend(adjusted_action_id)
            raise_amt = game_service.calculate_raise_amount_for_actor(
                adjusted_action_id,
                game_state,
                actor_index,
                version=version,
            )

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
                action_id=adjusted_action_id,
                equity=round(equity, 4),
                hand_strength_category=strength_name,
                q_values=adjusted_q_values,
                original_action=original_normalized_action,
                original_raise_amt=original_raise_amt,
                original_action_id=original_action_id,
                original_q_values=original_q_values,
                cv_influence_applied=cv_influence_applied,
                cv_act_max=round(cv_act_max, 1) if cv_act_max is not None else None,
                cv_bluff_risk_level=cv_bluff_risk_level,
            )
            _update_cv_read_tracking(
                game_state,
                next_state,
                actor_index=actor_index,
                applied_action=normalized_action,
                now_ms=step_now_ms,
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
            _update_cv_read_tracking(
                game_state,
                next_state,
                actor_index=actor_index,
                applied_action=request.action,
                now_ms=step_now_ms,
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
        
        if version_to_int(version) >= 25:
            action_id, q_values = game_service.get_runtime_action_for_actor(
                game_state,
                game_state.current_player_idx,
                version=version,
            )
        else:
            action_id, q_values = model_service.get_action(
                observation,
                legal_actions,
                version=version,
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
