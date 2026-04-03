"""
Webcam session router for opponent bluff-data webcam connections.

Allows the main bot user to generate a join code, opponents to join
with their webcam, and provides connection status polling.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.session_service import (
    generate_webcam_code,
    lookup_webcam_code,
    join_webcam_session,
    disconnect_webcam,
    disconnect_webcam_by_cv_session,
    reconnect_webcam,
    reconnect_webcam_by_cv_session,
    get_webcam_status,
)


router = APIRouter(prefix="/session/webcam", tags=["webcam"])


class GenerateCodeRequest(BaseModel):
    """Request to generate a join code for a game session."""
    session_id: str = Field(..., min_length=1)


class GenerateCodeResponse(BaseModel):
    """Response with the generated join code."""
    code: str


class JoinRequest(BaseModel):
    """Request from an opponent to join a webcam session."""
    code: str = Field(..., min_length=1, max_length=10)
    player_position: int = Field(..., ge=0, le=5)
    player_name: str | None = Field(default=None, max_length=60)


class JoinResponse(BaseModel):
    """Response after successfully joining a webcam session."""
    session_id: str
    cv_session_id: str
    player_name: str


class DisconnectRequest(BaseModel):
    """Request to disconnect an opponent from a webcam session."""
    session_id: str = Field(..., min_length=1)
    player_position: int | None = Field(default=None, ge=0, le=5)
    cv_session_id: str | None = Field(default=None, min_length=1)


class ReconnectRequest(BaseModel):
    """Request to reconnect an opponent to a webcam session."""
    session_id: str = Field(..., min_length=1)
    player_position: int | None = Field(default=None, ge=0, le=5)
    cv_session_id: str | None = Field(default=None, min_length=1)


@router.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: GenerateCodeRequest) -> GenerateCodeResponse:
    """Generate a short join code for the given game session."""
    code = await generate_webcam_code(request.session_id)
    return GenerateCodeResponse(code=code)


@router.post("/join", response_model=JoinResponse)
async def join(request: JoinRequest) -> JoinResponse:
    """
    Join a webcam session using a code and player position.

    Returns the session_id and a cv_session_id to use when streaming
    frames to the CV pipeline.
    """
    session_id = await lookup_webcam_code(request.code)
    if session_id is None:
        raise HTTPException(status_code=404, detail="Invalid or expired code")

    try:
        resolved_player_name = (request.player_name or "").strip() or f"Player {request.player_position + 1}"
        cv_session_id = await join_webcam_session(session_id, request.player_position, resolved_player_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JoinResponse(session_id=session_id, cv_session_id=cv_session_id, player_name=resolved_player_name)


@router.get("/status/{session_id}")
async def status(session_id: str) -> dict[str, Any]:
    """Return webcam connection status for all opponents in a session."""
    return await get_webcam_status(session_id)


@router.get("/status-by-code/{code}")
async def status_by_code(code: str) -> dict[str, Any]:
    """Return webcam connection status for the session behind a join code."""
    session_id = await lookup_webcam_code(code)
    if session_id is None:
        raise HTTPException(status_code=404, detail="Invalid or expired code")

    status = await get_webcam_status(session_id)
    status["session_id"] = session_id
    return status


@router.post("/disconnect", status_code=200)
async def disconnect(request: DisconnectRequest) -> dict[str, str]:
    """Mark an opponent as disconnected from the webcam session."""
    if request.cv_session_id:
        await disconnect_webcam_by_cv_session(request.session_id, request.cv_session_id)
    elif request.player_position is not None:
        await disconnect_webcam(request.session_id, request.player_position)
    else:
        raise HTTPException(status_code=422, detail="player_position or cv_session_id is required")
    return {"status": "disconnected"}


@router.post("/reconnect", status_code=200)
async def reconnect(request: ReconnectRequest) -> dict[str, str]:
    """Mark an opponent as reconnected to the webcam session."""
    if request.cv_session_id:
        await reconnect_webcam_by_cv_session(request.session_id, request.cv_session_id)
    elif request.player_position is not None:
        await reconnect_webcam(request.session_id, request.player_position)
    else:
        raise HTTPException(status_code=422, detail="player_position or cv_session_id is required")
    return {"status": "reconnected"}
