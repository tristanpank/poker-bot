"""API routers."""

from .cv import router as cv_router

__all__ = ["poker_router", "cv_router"]


def __getattr__(name: str):
    if name == "poker_router":
        from .poker import router as poker_router

        return poker_router
    raise AttributeError(name)
