"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fata_cognita.api.routes import archetypes, inflection, predict, simulate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fata_cognita.api.deps import AppState


def create_app(state: AppState | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        state: Optional pre-loaded AppState (for testing). If None,
            artifacts will be loaded during lifespan startup.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Load model artifacts on startup and clean up on shutdown."""
        if state is not None:
            app.state.app_state = state
        else:
            from fata_cognita.api.deps import load_artifacts

            app.state.app_state = load_artifacts()
        yield

    app = FastAPI(
        title="Fata Cognita",
        description="Neural Actuarial Trajectory Model API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
    app.include_router(simulate.router, prefix="/api/v1", tags=["simulate"])
    app.include_router(inflection.router, prefix="/api/v1", tags=["inflection"])
    app.include_router(archetypes.router, prefix="/api/v1", tags=["archetypes"])

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return service health status."""
        return {"status": "ok"}

    return app
