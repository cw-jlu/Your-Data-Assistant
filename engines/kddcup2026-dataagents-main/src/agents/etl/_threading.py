"""ContextVar-propagating ThreadPoolExecutor submit."""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


def submit_in_context(
    pool: ThreadPoolExecutor, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Future[Any]:
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args, **kwargs)
