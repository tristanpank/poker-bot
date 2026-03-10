"""
Session management router for saving/loading game state via Redis.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.session_service import (
    save_session,
    load_session,
    delete_session,
)


router = APIRouter(prefix="/session", tags=["session"])


class SessionCreateRequest(BaseModel):
    """Request body for creating / updating a session."""
    session_id: str = Field(..., min_length=1, description="Client-generated UUID")
    data: dict[str, Any] = Field(..., description="Full session snapshot")


@router.post("", status_code=201)
async def create_session(request: SessionCreateRequest):
    """Create a new session in Redis."""
    await save_session(request.session_id, request.data)
    return {"status": "created", "session_id": request.session_id}


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Load a session from Redis. Returns 404 if not found."""
    data = await load_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "data": data}


@router.put("/{session_id}")
async def update_session(session_id: str, request: dict[str, Any]):
    """Overwrite session data in Redis."""
    await save_session(session_id, request)
    return {"status": "updated", "session_id": session_id}


@router.delete("/{session_id}", status_code=204)
async def remove_session(session_id: str):
    """Delete a session from Redis."""
    await delete_session(session_id)
    return None
