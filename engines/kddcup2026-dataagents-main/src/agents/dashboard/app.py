"""FastAPI app factory for the traces dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.tracing.store import SQLiteTraceStore


def create_app(db_path: Path, static_dir: Path | None = None) -> Any:
    from fastapi import FastAPI, HTTPException
    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    app = FastAPI(title="DABench Traces Dashboard")
    store = SQLiteTraceStore(db_path)

    from agents.dashboard.api import create_api_routes

    create_api_routes(app, store)

    static_dir = static_dir or Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard static index not found at {index_path}",
            )
        return FileResponse(str(index_path))

    return app
