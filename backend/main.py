"""
FastAPI application entry point for the Poker Bot API.

Run with: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers import poker_router
from backend.services.model_service import get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Preloads the default model on startup for faster first request.
    """
    settings = get_settings()
    model_service = get_model_service()
    
    # Preload default model
    try:
        model_service.load_model(settings.model_version)
        print(f"✅ Loaded default model: {settings.model_version}")
    except FileNotFoundError as e:
        print(f"⚠️ Could not preload model: {e}")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down Poker Bot API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
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
    app.include_router(poker_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/poker/health",
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
