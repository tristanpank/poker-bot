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
    player_position: int = Field(..., ge=1, le=5)


class JoinResponse(BaseModel):
    """Response after successfully joining a webcam session."""
    session_id: str
    cv_session_id: str


class DisconnectRequest(BaseModel):
    """Request to disconnect an opponent from a webcam session."""
    session_id: str = Field(..., min_length=1)
    player_position: int = Field(..., ge=1, le=5)


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
        cv_session_id = await join_webcam_session(session_id, request.player_position)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JoinResponse(session_id=session_id, cv_session_id=cv_session_id)


@router.get("/status/{session_id}")
async def status(session_id: str) -> dict[str, Any]:
    """Return webcam connection status for all opponents in a session."""
    return await get_webcam_status(session_id)


@router.post("/disconnect", status_code=200)
async def disconnect(request: DisconnectRequest) -> dict[str, str]:
    """Mark an opponent as disconnected from the webcam session."""
    await disconnect_webcam(request.session_id, request.player_position)
    return {"status": "disconnected"}
