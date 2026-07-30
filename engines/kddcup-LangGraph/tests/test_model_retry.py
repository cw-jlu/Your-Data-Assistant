from __future__ import annotations

import pytest

from data_agent_baseline.model_retry import (
    MODEL_REQUEST_RETRY_DELAYS_SECONDS,
    NETWORK_ERROR_RETRY_DELAYS_SECONDS,
    RATE_LIMIT_RETRY_DELAY_SECONDS,
    invoke_model_with_retries,
)


class StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ScriptedModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.invoke_count = 0
        self.invocations = []

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(messages)
        self.invoke_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_default_model_retry_uses_progressive_retry_delays() -> None:
    assert MODEL_REQUEST_RETRY_DELAYS_SECONDS == (5, 15, 30)
    assert NETWORK_ERROR_RETRY_DELAYS_SECONDS == (5,) * 10
    assert RATE_LIMIT_RETRY_DELAY_SECONDS == 5


def test_invoke_model_retries_retryable_status_with_progressive_delays() -> None:
    model = ScriptedModel(
        [
            StatusError("busy", 503),
            StatusError("still busy", 503),
            "ok",
        ]
    )
    events: list[dict[str, object]] = []
    sleeps: list[float] = []

    result = invoke_model_with_retries(
        model,
        messages=[],
        sleep_fn=sleeps.append,
        on_retry_event=events.append,
    )

    assert result == "ok"
    assert model.invoke_count == 3
    assert sleeps == [5, 15]
    assert [event["status_code"] for event in events] == [503, 503]
    assert all(event["retryable"] is True for event in events)
    assert all(event["will_retry"] is True for event in events)


def test_invoke_model_retries_429_with_progressive_delays_and_infinite_retry() -> None:
    model = ScriptedModel([StatusError("rate limited", 429)] * 12 + ["ok"])
    events: list[dict[str, object]] = []
    sleeps: list[float] = []

    result = invoke_model_with_retries(
        model,
        messages=[],
        sleep_fn=sleeps.append,
        on_retry_event=events.append,
    )

    assert result == "ok"
    assert model.invoke_count == 13
    assert sleeps == [5, 15, 30] + [10] * 9
    assert [event["status_code"] for event in events] == [429] * 12
    assert all(event["max_attempts"] is None for event in events)
    assert all(event["retryable"] is True for event in events)
    assert all(event["will_retry"] is True for event in events)


def test_invoke_model_does_not_retry_non_429_4xx() -> None:
    model = ScriptedModel([StatusError("bad request", 400), "unexpected"])
    events: list[dict[str, object]] = []

    with pytest.raises(StatusError, match="bad request"):
        invoke_model_with_retries(model, messages=[], on_retry_event=events.append)

    assert model.invoke_count == 1
    assert events[0]["status_code"] == 400
    assert events[0]["retryable"] is False
    assert events[0]["will_retry"] is False


def test_invoke_model_retries_connection_errors_ten_times() -> None:
    model = ScriptedModel([ConnectionError("server disconnected")] * 10 + ["ok"])
    events: list[dict[str, object]] = []
    sleeps: list[float] = []

    result = invoke_model_with_retries(
        model,
        messages=[],
        sleep_fn=sleeps.append,
        on_retry_event=events.append,
    )

    assert result == "ok"
    assert model.invoke_count == 11
    assert sleeps == [5, 15, 30] + [10] * 7
    assert len(events) == 10
    assert events[-1]["attempt"] == 10
    assert events[-1]["max_attempts"] is None
    assert all(event["error_type"] == "ConnectionError" for event in events)
    assert all(event["retryable"] is True for event in events)
    assert all(event["will_retry"] is True for event in events)


def test_invoke_model_does_not_retry_timeout() -> None:
    model = ScriptedModel([TimeoutError("Request timed out."), "unexpected"])
    events: list[dict[str, object]] = []

    with pytest.raises(TimeoutError, match="Request timed out"):
        invoke_model_with_retries(model, messages=[], on_retry_event=events.append)

    assert model.invoke_count == 1
    assert events[0]["error_type"] == "TimeoutError"
    assert events[0]["retryable"] is False
    assert events[0]["will_retry"] is False
