"""
FastAPI application entry point for the Poker Bot API.

Run with: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers.cv import router as cv_router
from backend.routers.session import router as session_router
from backend.routers.webcam import router as webcam_router

_poker_router = None
_get_model_service: Callable[[], Any] | None = None
_poker_import_error: Exception | None = None
_cv_router = None
_cv_import_error: Exception | None = None


def _configure_logging() -> None:
    """Ensure backend logs are visible in local development and tests."""
    settings = get_settings()
    level = logging.DEBUG if settings.debug else logging.INFO

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root_logger.setLevel(level)

    logging.getLogger("backend").setLevel(level)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)


def _init_poker_dependencies() -> None:
    global _poker_router, _get_model_service, _poker_import_error

    if _poker_router is not None or _get_model_service is not None or _poker_import_error is not None:
        return

    enable_setting = os.getenv("ENABLE_POKER_ROUTER", "1")
    should_enable = enable_setting == "1"

    # Keep CV backend startup resilient when poker/model deps are unavailable.
    if not should_enable:
        _poker_import_error = RuntimeError(
            "Poker router disabled (set ENABLE_POKER_ROUTER=1 to enable)."
        )
        return

    try:
        from backend.routers.poker import router as poker_router

        _poker_router = poker_router
    except Exception as exc:  # pragma: no cover - environment dependent
        _poker_import_error = exc


def _init_cv_dependencies() -> None:
    global _cv_router, _cv_import_error

    if _cv_router is not None or _cv_import_error is not None:
        return

    try:
        from backend.routers.cv import router as cv_router

        _cv_router = cv_router
    except Exception as exc:  # pragma: no cover - environment dependent
        _cv_import_error = exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Preloads the default model on startup for faster first request.
    """
    _configure_logging()
    _init_poker_dependencies()
    _init_cv_dependencies()

    if _poker_router is not None and os.getenv("ENABLE_POKER_PRELOAD", "1") == "1":
        try:
            from backend.services.model_service import get_model_service

            settings = get_settings()
            model_service = get_model_service()
            try:
                model_service.load_model(settings.model_version)
                print(f"Loaded default model: {settings.model_version}")
            except FileNotFoundError as e:
                print(f"Could not preload model: {e}")
        except Exception as e:
            print(f"Poker model preload skipped: {str(e)}")
    elif _poker_import_error is not None:
        print(f"Poker router disabled: {str(_poker_import_error)}")

    # Initialize Redis connection
    try:
        from backend.services.session_service import get_redis
        await get_redis()
        print("Redis connection established")
    except Exception as e:
        print(f"Redis connection failed (sessions disabled): {e}")
    
    yield
    
    # Cleanup Redis connection
    try:
        from backend.services.session_service import close_redis
        await close_redis()
        print("Redis connection closed")
    except Exception:
        pass
    print("Shutting down Poker Bot API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    _configure_logging()
    settings = get_settings()
    _init_poker_dependencies()
    _init_cv_dependencies()
    
    app = FastAPI(
        title=settings.app_name,
        description="API for serving trained poker bot models for IRL gameplay",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    if _poker_router is not None:
        app.include_router(_poker_router)
    app.include_router(cv_router)
    app.include_router(session_router)
    app.include_router(webcam_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/poker/health" if _poker_router is not None else None,
            "cv": "/cv/analyze-raw" if _cv_router is not None else None,
            "poker_enabled": _poker_router is not None,
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
