"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# Action constants matching the model
ACTION_NAMES = {
    0: "FOLD",
    1: "CALL", 
    2: "RAISE_SMALL",
    3: "RAISE_MEDIUM",
    4: "RAISE_LARGE",
    5: "ALL_IN",
}

HAND_STRENGTH_CATEGORIES = {
    0: "Trash",
    1: "Marginal", 
    2: "Decent",
    3: "Strong",
    4: "Monster",
}


class CardSchema(BaseModel):
    """Representation of a playing card."""
    model_config = ConfigDict(
        json_schema_extra={"example": {"rank": "A", "suit": "s"}}
    )
    
    rank: str = Field(..., description="Card rank: 2-9, T, J, Q, K, A")
    suit: str = Field(..., description="Card suit: s (spades), h (hearts), d (diamonds), c (clubs)")


class PlayerState(BaseModel):
    """State of a single player at the table."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "position": 0,
                "stack": 990,
                "bet": 10,
                "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "s"}],
                "is_bot": True,
                "is_active": True
            }
        }
    )
    
    position: int = Field(..., ge=0, le=5, description="Player position (0-5 for 6-max)")
    stack: int = Field(..., ge=0, description="Player's current chip stack")
    bet: int = Field(default=0, ge=0, description="Player's current bet in this round")
    hole_cards: Optional[list[CardSchema]] = Field(
        default=None, 
        description="Player's hole cards (only provided for the bot)"
    )
    is_bot: bool = Field(default=False, description="Whether this player is the bot")
    is_active: bool = Field(default=True, description="Whether player is still in the hand")


class GameStateRequest(BaseModel):
    """Complete game state for requesting bot action."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "community_cards": [
                    {"rank": "A", "suit": "s"},
                    {"rank": "K", "suit": "h"},
                    {"rank": "Q", "suit": "d"}
                ],
                "pot": 150,
                "players": [
                    {
                        "position": 0,
                        "stack": 950,
                        "bet": 50,
                        "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "s"}],
                        "is_bot": True,
                        "is_active": True
                    },
                    {
                        "position": 1,
                        "stack": 1000,
                        "bet": 50,
                        "hole_cards": None,
                        "is_bot": False,
                        "is_active": True
                    }
                ],
                "bot_position": 0,
                "current_bet": 50,
                "big_blind": 10,
                "model_version": "v18"
            }
        }
    )
    
    # Session tracking (optional, for future Redis integration)
    session_id: Optional[str] = Field(default=None, description="Optional session ID for state tracking")
    
    # Board state
    community_cards: list[CardSchema] = Field(
        default_factory=list,
        description="Community cards on the board (0-5 cards)"
    )
    pot: int = Field(..., ge=0, description="Total pot size in chips")
    
    # Player states
    players: list[PlayerState] = Field(
        ..., 
        min_length=2, 
        max_length=6,
        description="State of all players at the table"
    )
    
    # Bot information
    bot_position: int = Field(..., ge=0, le=5, description="The bot's position at the table")
    
    # Betting state
    current_bet: int = Field(default=0, ge=0, description="Current bet to call")
    big_blind: int = Field(default=10, gt=0, description="Big blind amount")
    
    # Model selection
    model_version: Optional[str] = Field(
        default=None, 
        description="Model version to use (e.g., 'v18'). Uses default if not specified."
    )


class ActionResponse(BaseModel):
    """Bot's action response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "RAISE_MEDIUM",
                "action_id": 3,
                "amount": 75,
                "equity": 0.72,
                "hand_strength_category": "Strong",
                "q_values": {
                    "FOLD": -5.2,
                    "CALL": 3.1,
                    "RAISE_SMALL": 4.5,
                    "RAISE_MEDIUM": 5.8,
                    "RAISE_LARGE": 4.2,
                    "ALL_IN": 1.1
                }
            }
        }
    )
    
    action: str = Field(..., description="Action name (e.g., 'RAISE_MEDIUM')")
    action_id: int = Field(..., ge=0, le=5, description="Action ID (0-5)")
    
    # Betting details
    amount: Optional[int] = Field(
        default=None, 
        description="Chip amount for raises/bets. None for fold/call."
    )
    
    # Analysis info
    equity: float = Field(..., ge=0, le=1, description="Estimated hand equity (0-1)")
    hand_strength_category: str = Field(..., description="Hand strength category")
    
    # Debug info
    q_values: Optional[dict[str, float]] = Field(
        default=None,
        description="Q-values for each action (for debugging)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    available_models: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    version: str
    path: str
    state_dim: int = 520
    num_actions: int = 6


class CvAnalyzeRequest(BaseModel):
    """Frame analysis request from frontend webcam stream."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "sessionId": "session-123",
                "timestamp": 1765935300000,
                "width": 160,
                "height": 90,
                "rgbaBase64": "AAECAwQFBgcICQ==",
                "streamFps": 24.3,
            }
        }
    )

    session_id: str = Field(..., alias="sessionId")
    timestamp: int = Field(..., ge=0)
    width: int = Field(..., ge=1, le=2048)
    height: int = Field(..., ge=1, le=2048)
    rgba_base64: str = Field(..., alias="rgbaBase64", min_length=1)
    stream_fps: float = Field(..., alias="streamFps", ge=0)


class CvMetrics(BaseModel):
    """Computed CV and bluff-related metrics."""

    model_config = ConfigDict(populate_by_name=True)

    brightness: float = Field(..., ge=0, le=100)
    motion: float = Field(..., ge=0, le=100)
    edge_density: float = Field(..., alias="edgeDensity", ge=0, le=100)
    activity_zone: Literal["none", "left", "center", "right"] = Field(
        ..., alias="activityZone"
    )
    pulse_bpm: Optional[float] = Field(default=None, alias="pulseBpm", ge=0, le=240)
    pulse_confidence: float = Field(..., alias="pulseConfidence", ge=0, le=100)
    skin_coverage: float = Field(..., alias="skinCoverage", ge=0, le=100)
    stress: float = Field(..., ge=0, le=100)
    emotion: Literal["unknown", "calm", "focused", "tense", "agitated"]
    bluff_risk: float = Field(..., alias="bluffRisk", ge=0, le=100)
    bluff_level: Literal["low", "watch", "elevated"] = Field(..., alias="bluffLevel")
    baseline_progress: float = Field(..., alias="baselineProgress", ge=0, le=100)
    baseline_stress: float = Field(..., alias="baselineStress", ge=0, le=100)
    baseline_bluff: float = Field(..., alias="baselineBluff", ge=0, le=100)
    bluff_delta: float = Field(..., alias="bluffDelta", ge=-100, le=100)
    signal_quality: Literal["poor", "fair", "good"] = Field(..., alias="signalQuality")
    analysis_fps: float = Field(..., alias="analysisFps", ge=0)
    stream_fps: float = Field(..., alias="streamFps", ge=0)
    updated_at: str = Field(..., alias="updatedAt")


class CvAnalyzeResponse(BaseModel):
    """Response payload for frame analysis."""

    metrics: CvMetrics


class CvSessionClearRequest(BaseModel):
    """Request payload for clearing per-session backend CV state."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
