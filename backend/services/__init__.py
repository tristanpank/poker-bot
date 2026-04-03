"""Services for model inference, game state management, and CV analysis."""

from .game_service import GameService

__all__ = ["ModelService", "GameService", "CvService"]


def __getattr__(name: str):
    if name == "ModelService":
        from .model_service import ModelService

        return ModelService
    if name == "CvService":
        from .cv_service import CvService

        return CvService
    raise AttributeError(name)
