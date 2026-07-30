"""REST + SSE endpoints for the traces dashboard."""

import asyncio
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from typing import Any

from agents.tracing.store import SQLiteTraceStore


def _advance_stream_timestamp(last_ts: str, spans: list[dict[str, Any]]) -> str:
    latest = last_ts
    for span in spans:
        for key in ("started_at", "ended_at"):
            value = span.get(key)
            if isinstance(value, str) and value > latest:
                latest = value
    return latest


_db_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db")


def create_api_routes(app: Any, store: SQLiteTraceStore) -> None:
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse

    async def _to_thread(fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_db_pool, partial(fn, *args))

    @app.get("/api/runs")
    async def list_runs() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(await _to_thread(store.list_runs))

    @app.get("/api/traces")
    async def list_traces(  # pyright: ignore[reportUnusedFunction]
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> JSONResponse:
        result = await _to_thread(
            lambda: store.list_traces(
                run_id=run_id,
                status=status,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        )
        return JSONResponse(result)

    @app.get("/api/traces/{trace_id}")
    async def get_trace(trace_id: str) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        result = await _to_thread(store.get_trace_with_spans, trace_id)
        if result is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(result)

    @app.get("/api/spans/{span_id}")
    async def get_span(span_id: str) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        result = await _to_thread(store.get_span, span_id)
        if result is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(result)

    @app.get("/api/stats/rate-limit")
    async def rate_limit_stats(  # pyright: ignore[reportUnusedFunction]
        run_id: str | None = None,
    ) -> JSONResponse:
        return JSONResponse(await _to_thread(store.rate_limit_stats, run_id))

    @app.get("/api/stream")
    async def stream_events(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> StreamingResponse:
        run_id = request.query_params.get("run_id")

        async def event_generator() -> AsyncIterator[str]:
            last_ts = datetime.now(UTC).isoformat()
            while True:
                await asyncio.sleep(1.0)
                if await request.is_disconnected():
                    break
                new_spans = await _to_thread(store.get_spans_after, last_ts, run_id)
                if new_spans:
                    last_ts = _advance_stream_timestamp(last_ts, new_spans)
                    for span in new_spans:
                        event_type = "span_end" if span.get("ended_at") else "span_start"
                        yield f"event: {event_type}\ndata: {json.dumps(span, default=str)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
