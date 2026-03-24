"""API routers."""

__all__ = ["poker_router", "cv_router"]


def __getattr__(name: str):
    if name == "poker_router":
        from .poker import router as poker_router

        return poker_router
    if name == "cv_router":
        from .cv import router as cv_router

        return cv_router
    raise AttributeError(name)
