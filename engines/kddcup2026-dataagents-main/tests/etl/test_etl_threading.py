"""Tests for ContextVar-propagating submit — _threading.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

from agents.etl._threading import submit_in_context

_test_var: ContextVar[str] = ContextVar("_test_var", default="unset")


def test_submit_in_context_propagates_contextvar() -> None:
    token = _test_var.set("parent_value")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = submit_in_context(pool, _test_var.get)
            assert fut.result(timeout=5) == "parent_value"
    finally:
        _test_var.reset(token)


def test_plain_submit_does_not_propagate() -> None:
    """Verify the baseline: plain pool.submit does NOT see parent ContextVar."""
    token = _test_var.set("parent_value")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_test_var.get)
            assert fut.result(timeout=5) == "unset"
    finally:
        _test_var.reset(token)


def test_submit_in_context_with_args() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = submit_in_context(pool, add, 3, 7)
        assert fut.result(timeout=5) == 10


def test_submit_in_context_with_kwargs() -> None:
    def greet(name: str, prefix: str = "Hello") -> str:
        return f"{prefix} {name}"

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = submit_in_context(pool, greet, "K", prefix="Hi")
        assert fut.result(timeout=5) == "Hi K"
