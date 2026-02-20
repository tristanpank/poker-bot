"""
CV API router for backend frame analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.models.schemas import (
    CvAnalyzeRequest,
    CvAnalyzeResponse,
    CvSessionClearRequest,
)
from backend.services.cv_service import CvService, get_cv_service


router = APIRouter(prefix="/cv", tags=["cv"])


@router.post("/analyze", response_model=CvAnalyzeResponse)
async def analyze_frame(
    request: CvAnalyzeRequest,
    cv_service: CvService = Depends(get_cv_service),
) -> CvAnalyzeResponse:
    """Analyze a frame and return bluff/stress proxy metrics."""
    try:
        metrics = cv_service.analyze(request)
        return CvAnalyzeResponse(metrics=metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"CV analysis error: {str(exc)}") from exc


@router.post("/analyze-raw", response_model=CvAnalyzeResponse)
async def analyze_raw_frame(
    request: Request,
    session_id: str = Query(..., alias="sessionId"),
    timestamp: int = Query(..., ge=0),
    width: int = Query(..., ge=1, le=2048),
    height: int = Query(..., ge=1, le=2048),
    stream_fps: float = Query(..., alias="streamFps", ge=0),
    cv_service: CvService = Depends(get_cv_service),
) -> CvAnalyzeResponse:
    """Analyze a raw RGBA frame body and return bluff/stress proxy metrics."""
    try:
        raw_body = await request.body()
        metrics = cv_service.analyze_raw(
            session_id=session_id,
            timestamp=timestamp,
            width=width,
            height=height,
            stream_fps=stream_fps,
            rgba_bytes=raw_body,
        )
        return CvAnalyzeResponse(metrics=metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"CV analysis error: {str(exc)}") from exc


@router.delete("/session")
async def clear_session(
    request: CvSessionClearRequest,
    cv_service: CvService = Depends(get_cv_service),
) -> dict[str, bool]:
    """Clear session analysis state."""
    cv_service.clear_session(request.session_id)
    return {"ok": True}
