from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from data_agent_baseline.agents.langgraph_runtime import (
    LangGraphAgent,
    LangGraphAgentConfig,
    _discard_invalid_tool_calls,
    _is_non_action_stop,
)
from data_agent_baseline.benchmark.schema import (
    ContextAsset,
    ContextView,
    PublicTask,
    TaskAssets,
    TaskRecord,
)
from data_agent_baseline.config import DataInspectorConfig, ProcessValidatorConfig, ToolConfig
from data_agent_baseline.tools.registry import create_default_tool_registry


def _create_task(tmp_path: Path, task_id: str = "task_demo") -> PublicTask:
    task_dir = tmp_path / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample.csv").write_text("value\n1\n2\n", encoding="utf-8")
    (context_dir / "notes.md").write_text("# Notes\nhello\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id=task_id, difficulty="easy", question="List the value column."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage | BaseException]) -> None:
        self._responses = list(responses)
        self.invoke_count = 0
        self.invocations = []
        self.bound_tools = []
        self.tool_choice = None
        self.parallel_tool_calls = None

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(list(messages))
        self.invoke_count += 1
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class RetryableStatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_langgraph_agent_executes_tool_call_loop_and_submits_answer(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                id="response_1",
                additional_kwargs={"provider_note": "first"},
                response_metadata={"finish_reason": "tool_calls"},
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["value"], "rows": [["1"], ["2"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    assert result.answer.rows == [["1"], ["2"]]
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert model.tool_choice == "auto"
    assert model.parallel_tool_calls is False
    assert result.steps[0].model_request is not None
    assert result.steps[0].model_request["tool_choice"] == "auto"
    assert result.steps[0].model_request["parallel_tool_calls"] is False
    assert "submit_tool_result" in result.steps[0].model_request["tool_names"]
    assert "answer" not in result.steps[0].model_request["tool_names"]
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["response_id"] == "response_1"
    assert result.steps[0].model_response["finish_reason"] == "tool_calls"
    assert result.steps[0].model_response["tool_call_names"] == ["list_context"]


def test_langgraph_agent_attaches_stable_frame_images_without_leaking_base64_in_trace(
    tmp_path: Path,
) -> None:
    base_task = _create_task(tmp_path)
    (base_task.context_dir / "clip.mp4").write_bytes(b"raw video must not be attached")
    generated_context_dir = tmp_path / "generated_context"
    timeline_path = generated_context_dir / "video" / "clip_timeline.md"
    image_path = generated_context_dir / "video" / "clip_stable_frames" / "stable_001.jpg"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text("# Video Timeline\n\nspeech\n", encoding="utf-8")
    image_path.write_bytes(b"fake stable frame bytes")
    context_view = ContextView(
        source_context_dir=base_task.context_dir,
        generated_context_dir=generated_context_dir,
        assets=(
            ContextAsset(
                visible_path="video/clip_timeline.md",
                physical_path=timeline_path,
                source_path="video/clip.mp4",
                action="video_timeline",
                generated=True,
            ),
            ContextAsset(
                visible_path="video/clip_stable_frames/stable_001.jpg",
                physical_path=image_path,
                source_path="video/clip.mp4",
                action="video_stable_frame",
                generated=True,
            ),
        ),
    )
    task = PublicTask(
        record=base_task.record,
        assets=TaskAssets(
            task_dir=base_task.task_dir,
            context_dir=base_task.context_dir,
            context_view=context_view,
        ),
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )

    result = agent.run(task)

    assert result.succeeded is True
    initial_human_message = model.invocations[0][1]
    assert isinstance(initial_human_message.content, str)
    assert "video/clip_timeline.md" in initial_human_message.content
    assert "speech" in initial_human_message.content
    assert "video/clip_stable_frames/stable_001.jpg" in initial_human_message.content
    assert "read_context_image" in initial_human_message.content
    assert "raw video files are intentionally not attached" in initial_human_message.content
    request_summary = result.steps[0].model_request["last_message"]
    assert "content_part_types" not in request_summary
    assert "fake stable frame bytes" not in json.dumps(request_summary, ensure_ascii=False)
    assert "raw video must not be attached" not in json.dumps(request_summary, ensure_ascii=False)
    assert "ZmFrZSBzdGFibGUgZnJhbWUgYnl0ZXM=" not in json.dumps(request_summary, ensure_ascii=False)


def test_langgraph_agent_initial_context_prefers_video_summary(
    tmp_path: Path,
) -> None:
    base_task = _create_task(tmp_path)
    generated_context_dir = tmp_path / "generated_context"
    timeline_path = generated_context_dir / "video" / "clip_timeline.md"
    image_path = generated_context_dir / "video" / "clip_stable_frames" / "stable_001.jpg"
    summary_path = generated_context_dir / "video" / "clip_video_summary.md"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text("# Video Timeline\n\nfull transcript should stay out\n", encoding="utf-8")
    image_path.write_bytes(b"fake stable frame bytes")
    summary_path.write_text(
        "# Video Understanding Summary\n\nUse threshold 100. Original timeline: `video/clip_timeline.md`\n",
        encoding="utf-8",
    )
    context_view = ContextView(
        source_context_dir=base_task.context_dir,
        generated_context_dir=generated_context_dir,
        assets=(
            ContextAsset(
                visible_path="video/clip_timeline.md",
                physical_path=timeline_path,
                source_path="video/clip.mp4",
                action="video_timeline",
                generated=True,
            ),
            ContextAsset(
                visible_path="video/clip_stable_frames/stable_001.jpg",
                physical_path=image_path,
                source_path="video/clip.mp4",
                action="video_stable_frame",
                generated=True,
            ),
            ContextAsset(
                visible_path="video/clip_video_summary.md",
                physical_path=summary_path,
                source_path="video/clip.mp4",
                action="video_summary",
                generated=True,
            ),
        ),
    )
    task = PublicTask(
        record=base_task.record,
        assets=TaskAssets(
            task_dir=base_task.task_dir,
            context_dir=base_task.context_dir,
            context_view=context_view,
        ),
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )

    result = agent.run(task)

    assert result.succeeded is True
    initial_human_message = model.invocations[0][1]
    assert isinstance(initial_human_message.content, str)
    assert "Video Understanding Summary" in initial_human_message.content
    assert "Use threshold 100" in initial_human_message.content
    assert "pre-main video understanding agent summary" in initial_human_message.content
    assert "full transcript should stay out" not in initial_human_message.content
    assert "video/clip_stable_frames/stable_001.jpg" not in initial_human_message.content
    assert "read_context_image" in initial_human_message.content


def test_langgraph_agent_compresses_image_message_after_it_is_used(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "frame.jpg").write_bytes(b"fake jpg bytes")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_context_image",
                        "args": {"path": "frame.jpg", "detail": "high"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="The image shows the Alpha threshold clearly.",
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_3",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    second_request_image_message = model.invocations[1][-1]
    assert isinstance(second_request_image_message.content, list)
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in second_request_image_message.content
    )

    third_request_payload = json.dumps(
        [message.content for message in model.invocations[2]],
        ensure_ascii=False,
    )
    assert "Compressed image observation:" in third_request_payload
    assert "frame.jpg" in third_request_payload
    assert "detail: high" in third_request_payload
    assert "The image shows the Alpha threshold clearly." in third_request_payload
    assert "image_url" not in third_request_payload
    assert "data:image/jpeg;base64" not in third_request_payload

    third_model_step = result.steps[4]
    assert third_model_step.model_request is not None
    compressed_trace_messages = third_model_step.model_request["compressed_image_messages"]
    assert len(compressed_trace_messages) == 1
    assert compressed_trace_messages[0]["type"] == "human"
    assert "Compressed image observation:" in compressed_trace_messages[0]["content"]
    assert "frame.jpg" in compressed_trace_messages[0]["content"]
    assert "The image shows the Alpha threshold clearly." in compressed_trace_messages[0]["content"]


def test_langgraph_agent_keeps_unviewed_image_message_for_forced_answer(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "frame.jpg").write_bytes(b"fake jpg bytes")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_context_image",
                        "args": {"path": "frame.jpg", "detail": "low"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["best_effort"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    forced_request_contents = [message.content for message in model.invocations[1]]
    image_messages = [
        content
        for content in forced_request_contents
        if isinstance(content, list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
    ]
    assert len(image_messages) == 1
    forced_payload = json.dumps(forced_request_contents, ensure_ascii=False)
    assert "Compressed image observation:" not in forced_payload
    assert "data:image/jpeg;base64" in forced_payload


def test_langgraph_agent_compresses_multiple_images_in_one_message(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "first.jpg").write_bytes(b"first fake jpg bytes")
    (task.context_dir / "second.jpg").write_bytes(b"second fake jpg bytes")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_context_image",
                        "args": {"path": "first.jpg", "detail": "high"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "read_context_image",
                        "args": {"path": "second.jpg", "detail": "low"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="First frame has the threshold; second frame has the year.",
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_4",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    second_request_image_message = model.invocations[1][-1]
    assert isinstance(second_request_image_message.content, list)
    assert (
        sum(
            1
            for part in second_request_image_message.content
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        == 2
    )

    third_request_payload = json.dumps(
        [message.content for message in model.invocations[2]],
        ensure_ascii=False,
    )
    assert "Compressed image observation:" in third_request_payload
    assert "first.jpg" in third_request_payload
    assert "detail: high" in third_request_payload
    assert "second.jpg" in third_request_payload
    assert "detail: low" in third_request_payload
    assert "First frame has the threshold; second frame has the year." in third_request_payload
    assert "image_url" not in third_request_payload


def test_langgraph_agent_emits_live_trace_updates(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [len(update["steps"]) for update in trace_updates] == [1, 1, 2, 2, 2]
    assert trace_updates[0]["partial"] is True
    assert trace_updates[0]["steps"][-1]["node"] == "model"
    assert trace_updates[0]["steps"][-1]["status"] == "in_progress"
    assert trace_updates[1]["steps"][-1]["node"] == "model"
    assert "status" not in trace_updates[1]["steps"][-1]
    assert trace_updates[2]["steps"][-1]["node"] == "tool"
    assert trace_updates[2]["steps"][-1]["status"] == "in_progress"
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["succeeded"] is True
    assert trace_updates[-1]["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    json.dumps(trace_updates[-1], ensure_ascii=False)


def test_langgraph_agent_validates_answer_before_final_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "validate_answer"]
    assert result.steps[-1].ok is True
    assert trace_updates[-2]["steps"][-1]["node"] == "validate_answer"
    assert "status" not in trace_updates[-2]["steps"][-1]
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["steps"][-1]["node"] == "validate_answer"
    assert trace_updates[-1]["succeeded"] is True


def test_langgraph_agent_process_validates_answer_before_answer_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    process_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": {"intent_summary": "list values"},
            "raw_response": '{"valid": true}',
        }

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator", validate_process
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_process",
        "validate_answer",
    ]
    assert len(process_calls) == 1
    assert process_calls[0]["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    assert process_calls[0]["supporting_source_evidence"]["schema_version"] == 2
    assert process_calls[0]["submission_risk_report"]["source_tool"] == "execute_python"
    assert result.steps[3].model_request["supporting_evidence_count"] >= 0
    assert result.steps[3].model_request["has_video_evidence"] is False
    assert "process_validation_receipt" not in result.steps[2].model_response
    assert "process_receipt_matches_submission" not in result.steps[3].model_request
    assert result.semantic_ledger == {"intent_summary": "list values"}


def test_langgraph_agent_skips_process_validator_after_prior_pass_on_resubmission(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)

    def submit_message(value: str, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_tool_result",
                    "args": {
                        "tool_name": "execute_python",
                        "tool_args": {
                            "code": "print("
                            + repr(
                                json.dumps(
                                    {"columns": ["status"], "rows": [[value]]},
                                    ensure_ascii=False,
                                )
                            )
                            + ")",
                        },
                    },
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    model = ScriptedToolCallingModel(
        responses=[
            submit_message("first", "call_1"),
            submit_message("second", "call_2"),
        ]
    )
    process_calls = []
    answer_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": {"intent_summary": "approved once"},
            "raw_response": '{"valid": true}',
        }

    def validate_answer(**kwargs):  # noqa: ANN001
        answer_calls.append(kwargs)
        if len(answer_calls) == 1:
            return {
                "valid": False,
                "issues": ["needs a corrected shape"],
                "raw_response": '{"valid": false}',
            }
        return {"valid": True, "issues": [], "raw_response": '{"valid": true}'}

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator", validate_process
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate_answer
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=4,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_process",
        "validate_answer",
        "model",
        "tool",
        "validate_answer",
    ]
    assert len(process_calls) == 1
    assert len(answer_calls) == 2
    assert result.semantic_ledger == {"intent_summary": "approved once"}


def test_langgraph_agent_passes_verified_video_scope_to_answer_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print(" + repr(
                                    json.dumps(
                                        {"columns": ["procedure"], "rows": [["A"]]},
                                        ensure_ascii=False,
                                    )
                                ) + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    video_evidence = {
        "schema_version": 2,
        "evidence_items": [
            {
                "tool": "read_doc",
                "capabilities": ["document_text_fact"],
                "source": {"path": "video/briefing_timeline.md"},
            },
            {
                "tool": "read_context_image",
                "capabilities": ["visual_fact"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"locator": {"frame_path": "video/stable_006.jpg"}},
            },
            {
                "tool": "record_visual_evidence",
                "capabilities": ["visual_fact_receipt"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"evidence_excerpt": "Report scope: Top 3"},
            },
        ],
        "omitted_or_unusable": [],
    }
    validator_calls = []

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime._build_supporting_source_evidence",
        lambda *_args, **_kwargs: video_evidence,
    )

    def validate(**kwargs):  # noqa: ANN001
        validator_calls.append(kwargs)
        return {"valid": True, "issues": [], "raw_response": '{"valid": true}'}

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[-1].model_request["supporting_evidence_count"] == 3
    assert result.steps[-1].model_request["has_video_evidence"] is True
    assert validator_calls[0]["supporting_source_evidence"] == video_evidence


def test_langgraph_agent_process_validates_after_checkpoint_interval(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_context",
                    "args": {"max_depth": 1},
                    "id": f"call_{idx}",
                    "type": "tool_call",
                }
            ],
        )
        for idx in range(10)
    ]
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_tool_result",
                    "args": {
                        "tool_name": "execute_python",
                        "tool_args": {
                            "code": "print("
                            + repr(
                                json.dumps(
                                    {"columns": ["status"], "rows": [["ok"]]}, ensure_ascii=False
                                )
                            )
                            + ")",
                        },
                    },
                    "id": "call_answer",
                    "type": "tool_call",
                }
            ],
        )
    )
    model = ScriptedToolCallingModel(responses=responses)
    process_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": {"count": len(process_calls)},
            "raw_response": '{"valid": true}',
        }

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator", validate_process
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=12,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    validate_steps = [step for step in result.steps if step.node == "validate_process"]
    assert len(validate_steps) == 1
    assert validate_steps[0].model_request["model_count"] == 10
    assert validate_steps[0].model_request["has_answer"] is False
    assert len(process_calls) == 1


def test_langgraph_agent_process_validator_runs_once_when_tenth_model_answers(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_context",
                    "args": {"max_depth": 1},
                    "id": f"call_{idx}",
                    "type": "tool_call",
                }
            ],
        )
        for idx in range(9)
    ]
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_tool_result",
                    "args": {
                        "tool_name": "execute_python",
                        "tool_args": {
                            "code": "print("
                            + repr(
                                json.dumps(
                                    {"columns": ["status"], "rows": [["ok"]]}, ensure_ascii=False
                                )
                            )
                            + ")",
                        },
                    },
                    "id": "call_answer",
                    "type": "tool_call",
                }
            ],
        )
    )
    model = ScriptedToolCallingModel(responses=responses)
    process_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": {},
            "raw_response": '{"valid": true}',
        }

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator", validate_process
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=10,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    validate_steps = [step for step in result.steps if step.node == "validate_process"]
    assert len(validate_steps) == 1
    assert len(process_calls) == 1
    assert validate_steps[0].model_request["model_count"] == 10
    assert validate_steps[0].model_request["has_answer"] is True


def test_langgraph_agent_process_validation_failure_returns_to_model_step(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["bad"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    process_results = [
        {
            "valid": False,
            "issues": ["iodine field binding was not verified"],
            "required_next_actions": ["Probe atom.element distinct values"],
            "semantic_ledger": {"unverified_assumptions": ["iodine"]},
            "raw_response": '{"valid": false}',
        },
        {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": {"verified_claims": ["iodine"]},
            "raw_response": '{"valid": true}',
        },
    ]

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator",
        lambda **_: process_results.pop(0),
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=4,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10, retry_limit=1),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_process",
        "model",
        "tool",
        "validate_process",
        "validate_answer",
    ]
    assert result.steps[2].ok is False
    assert model.invocations[1][-1].content.startswith("Your recent work did NOT pass")


def test_langgraph_agent_process_validator_retry_limit_allows_answer_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    process_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        raise AssertionError("process validator should be skipped after retry limit")

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator", validate_process
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=10, retry_limit=0),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_process",
        "validate_answer",
    ]
    assert result.steps[2].ok is True
    assert result.steps[2].tool_results[0]["skipped"] is True
    assert result.steps[2].tool_results[0]["reason"] == "retry_limit_reached"
    assert result.steps[2].tool_results[0]["retry_limit_reached"] is True
    assert process_calls == []


def test_langgraph_agent_process_retry_limit_skips_validator_without_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    process_calls = []

    def validate_process(**kwargs):  # noqa: ANN001
        process_calls.append(kwargs)
        raise AssertionError("process validator should be skipped after retry limit")

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator",
        validate_process,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=3,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=1, retry_limit=0),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_process",
        "model",
        "tool",
        "validate_process",
        "validate_answer",
    ]
    skipped_steps = [step for step in result.steps if step.node == "validate_process"]
    assert len(skipped_steps) == 2
    assert all(step.tool_results[0]["skipped"] is True for step in skipped_steps)
    assert all(step.tool_results[0]["reason"] == "retry_limit_reached" for step in skipped_steps)
    assert process_calls == []
    assert len(model.invocations) >= 2
    assert "The process validator still found blocking issues" not in model.invocations[1][-1].content


def test_langgraph_agent_validation_failure_returns_to_model_step(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["extra"], "rows": [["bad"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    validation_results = [
        {"valid": False, "issues": ["extra column"], "raw_response": '{"valid": false}'},
        {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    ]

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: validation_results.pop(0),
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_answer",
        "model",
        "tool",
        "validate_answer",
    ]
    assert result.steps[2].ok is False
    assert result.steps[-1].ok is True
    assert model.invocations[1][-1].content.startswith("Your submitted answer did NOT pass")


def test_langgraph_agent_forces_answer_after_max_steps(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["best_effort"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "force_answer", "tool"]
    assert result.steps[2].model_request["forced_answer"] is True
    assert set(result.steps[2].model_request["tool_names"]) == {"submit_tool_result"}
    # thinking 模式不允许 object/required 形式的 tool_choice；force_answer 仅绑定
    # submit_tool_result 并配合强提示，用 "auto" 触发提交（详见 langgraph_runtime）。
    assert result.steps[2].model_request["tool_choice"] == "auto"
    assert result.steps[2].tool_calls[0]["name"] == "submit_tool_result"
    assert model.invocations[1][-1].content.startswith(
        "You have reached the maximum number of model steps"
    )
    assert {t.name for t in model.bound_tools} == {"submit_tool_result"}
    assert model.tool_choice == "auto"


def test_langgraph_agent_fails_when_forced_answer_does_not_call_submission_tool(
    tmp_path: Path,
) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="I need one more look.",
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="Still not enough evidence.",
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=1),
    )

    result = agent.run(task)

    assert result.succeeded is False
    assert result.failure_reason == "Agent did not submit an answer within max_steps."
    assert [step.node for step in result.steps] == ["model", "force_answer"]
    assert result.steps[1].model_request["forced_answer"] is True
    assert result.steps[1].tool_calls == []


def test_langgraph_agent_skips_validators_after_forced_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["extra"], "rows": [["bad"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    validator_calls = {"answer": 0, "process": 0}

    def fail_answer_validator(**_: object) -> dict[str, object]:
        validator_calls["answer"] += 1
        raise AssertionError("answer validator should be skipped after forced answer")

    def fail_process_validator(**_: object) -> dict[str, object]:
        validator_calls["process"] += 1
        raise AssertionError("process validator should be skipped after forced answer")

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        fail_answer_validator,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_process_validator",
        fail_process_validator,
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=1,
            enable_process_validator=True,
            process_validator=ProcessValidatorConfig(checkpoint_model_interval=1),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.failure_reason is None
    assert [step.node for step in result.steps] == ["model", "tool", "force_answer", "tool"]
    assert validator_calls == {"answer": 0, "process": 0}
    assert model.invoke_count == 2


def test_langgraph_agent_revalidates_same_answer_without_submission_source(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    bad_answer = {"columns": ["extra"], "rows": [["bad"]]}
    bad_code_1 = "print(" + repr(json.dumps(bad_answer, ensure_ascii=False)) + ")"
    bad_code_2 = (
        "import json\n"
        f"payload = {bad_answer!r}\n"
        "print(json.dumps(payload, ensure_ascii=False))"
    )
    good_answer = {"columns": ["status"], "rows": [["ok"]]}
    good_code = "print(" + repr(json.dumps(good_answer, ensure_ascii=False)) + ")"
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": bad_code_1},
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": bad_code_2},
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": good_code},
                        },
                        "id": "call_3",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    validation_results = [
        {"valid": False, "issues": ["extra column"], "raw_response": '{"valid": false}'},
        {"valid": False, "issues": ["still extra column"], "raw_response": '{"valid": false}'},
        {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    ]
    validator_calls = []

    def validate(**kwargs):  # noqa: ANN001
        validator_calls.append(kwargs)
        return validation_results.pop(0)

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=6, validation_retry_limit=3),
    )

    result = agent.run(task)

    validate_steps = [step for step in result.steps if step.node == "validate_answer"]
    assert result.succeeded is True
    assert len(validate_steps) == 3
    assert len(validator_calls) == 3
    assert validate_steps[0].ok is False
    assert validate_steps[1].ok is False
    assert validate_steps[1].model_response["cached"] is False
    assert validate_steps[1].tool_results[0]["cached"] is False
    assert "cache_hit_answer_fingerprint" not in validate_steps[1].model_request
    assert validate_steps[1].tool_results[0]["issues"] == ["still extra column"]
    assert validator_calls[0]["validation_history"] == []
    assert len(validator_calls[1]["validation_history"]) == 1
    assert validator_calls[1]["validation_history"][0]["issues"] == ["extra column"]
    assert "submission_context" not in validator_calls[0]
    assert "submission_context" not in validator_calls[1]
    assert validate_steps[2].ok is True


def test_langgraph_agent_passes_validation_history_for_new_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["extra"], "rows": [["bad"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    validation_results = [
        {"valid": False, "issues": ["extra column"], "raw_response": '{"valid": false}'},
        {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    ]
    validation_histories = []

    def validate(**kwargs):  # noqa: ANN001
        validation_histories.append(kwargs["validation_history"])
        return validation_results.pop(0)

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert validation_histories[0] == []
    assert len(validation_histories[1]) == 1
    assert validation_histories[1][0]["answer_columns"] == ["extra"]
    assert validation_histories[1][0]["answer_row_count"] == 1
    assert validation_histories[1][0]["valid"] is False
    assert validation_histories[1][0]["issues"] == ["extra column"]
    assert result.steps[-1].model_request["validation_history_count"] == 1


def test_langgraph_agent_does_not_pass_submit_tool_result_source_to_answer_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    code = (
        "import json\n"
        "print(json.dumps({'columns': ['value'], 'rows': [['1'], ['2']]}, "
        "ensure_ascii=False))"
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": code},
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    validator_calls = []

    def validate(**kwargs):  # noqa: ANN001
        validator_calls.append(kwargs)
        return {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'}

    def fake_explore(self, *, context_dir, task_id=""):  # noqa: ANN001
        del self, context_dir, task_id
        full_catalog = json.dumps(
            {
                "task_id": "task_demo",
                "query_surfaces": [],
                "documents": [],
                "media": [],
                "knowledge_documents": [
                    {
                        "path": "knowledge.md",
                        "content": "table: sample\nprimary_key: RecordNo\nfield: value",
                    }
                ],
                "semantic_uncertainties": [],
            },
            ensure_ascii=False,
        )
        return full_catalog, {}

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fake_explore,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(validator_calls) == 1
    assert validator_calls[0]["answer"] == {"columns": ["value"], "rows": [["1"], ["2"]]}
    assert "submission_context" not in validator_calls[0]
    assert validator_calls[0]["submission_field_context"]["status"] == "partial"
    assert any(
        warning["kind"] == "no_static_sql_found"
        for warning in validator_calls[0]["submission_field_context"]["warnings"]
    )
    assert validator_calls[0]["knowledge_docs"] == [
        {
            "path": "knowledge.md",
            "content": "table: sample\nprimary_key: RecordNo\nfield: value",
        }
    ]


def test_langgraph_agent_passes_structured_doc_manifest_fields_to_answer_validator(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    generated_dir = task.context_dir / ".generated" / "structured_doc"
    generated_dir.mkdir(parents=True)
    (generated_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "source_path": "doc/qt_dailyquote.md",
                        "target_table": "qt_dailyquote",
                        "registered_table": "qt_dailyquote",
                        "columns": ["secucode", "turnoverdeals", "tradingday"],
                        "file": "qt_dailyquote.jsonl",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (generated_dir / "qt_dailyquote.jsonl").write_text(
        json.dumps(
            {"secucode": "601908", "turnoverdeals": 71041, "tradingday": "2021-08-12"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {
                            "code": (
                                "import json\n"
                                "print(json.dumps({'columns': ['ok'], 'rows': [[1]]}))"
                            )
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_probe_query",
                            "tool_args": {
                                "queries": [
                                    "SELECT tradingday, turnoverdeals "
                                    "FROM qt_dailyquote WHERE secucode = '601908'"
                                ]
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    validator_calls = []

    def validate(**kwargs):  # noqa: ANN001
        validator_calls.append(kwargs)
        return {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'}

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=3),
    )

    result = agent.run(task)

    assert result.succeeded is True
    context = validator_calls[0]["submission_field_context"]
    assert context["status"] == "complete"
    universe = context["field_universe"][0]
    assert universe["kind"] == "structured_doc_table"
    assert universe["source_path"] == "doc/qt_dailyquote.md"
    assert universe["target_table"] == "qt_dailyquote"
    assert {field["name"] for field in universe["fields"]} == {
        "secucode",
        "turnoverdeals",
        "tradingday",
    }
    preview_universe = result.steps[-1].model_request["submission_field_context_preview"][
        "field_universe"
    ][0]
    assert preview_universe["source_path"] == "doc/qt_dailyquote.md"
    assert preview_universe["target_table"] == "qt_dailyquote"
    assert preview_universe["fields"] == ["secucode", "turnoverdeals", "tradingday"]


def test_langgraph_agent_truncates_answer_only_for_answer_validator_context(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    long_cell = "long-cell-" + ("x" * 80)
    code = (
        "import json\n"
        f"rows = [[{long_cell!r}] for _ in range(5)]\n"
        "print(json.dumps({'columns': ['value'], 'rows': rows}, ensure_ascii=False))"
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": code},
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    validator_calls = []

    def validate(**kwargs):  # noqa: ANN001
        validator_calls.append(kwargs)
        return {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'}

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(ToolConfig(max_output_tokens=5, max_list_items=2)),
        config=LangGraphAgentConfig(max_steps=2),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert len(result.answer.rows) == 5
    assert result.answer.rows[0] == [long_cell]
    validator_answer = validator_calls[0]["answer"]
    assert validator_answer["columns"] == ["value"]
    assert len(validator_answer["rows"]) == 3
    assert "内容已被截断" in validator_answer["rows"][0][0]
    assert "内容已被截断" in validator_answer["rows"][2]
    overview = validator_calls[0]["answer_structure_overview"]
    assert overview["row_count"] == 5
    assert overview["column_profiles"][0]["name"] == "value"
    assert "distinct_value_examples" in overview["column_profiles"][0]
    assert "row_samples" not in validator_calls[0]
    assert validator_calls[0]["answer_truncated"] is True
    assert "submission_risk_report" not in validator_calls[0]
    assert result.steps[-1].model_request["answer_row_count"] == 5
    assert result.steps[-1].model_request["validator_answer_truncated"] is True
    assert "submission_field_context_preview" in result.steps[-1].model_request


def test_langgraph_agent_rejected_answer_feedback_uses_truncated_answer_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    long_cell = "rejected-cell-" + ("y" * 80)
    code = (
        "import json\n"
        f"rows = [[{long_cell!r}] for _ in range(5)]\n"
        "rows = rows[:5]\n"
        "print(json.dumps({'columns': ['extra'], 'rows': rows}, ensure_ascii=False))"
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {"code": code},
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["value"], "rows": [["ok"]]}, ensure_ascii=False
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    validation_results = [
        {"valid": False, "issues": ["extra column"], "raw_response": '{"valid": false}'},
        {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    ]

    def validate(**_: object) -> dict[str, object]:
        return validation_results.pop(0)

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", validate
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(ToolConfig(max_output_tokens=5, max_list_items=2)),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    feedback = str(model.invocations[1][-1].content)
    assert "bounded validator-context structure overview with no row samples" in feedback
    assert "answer_structure_overview" in feedback
    assert "Programmatic source risk report" not in feedback
    assert "This validator only checks delivery" in feedback
    assert "内容已被截断" in feedback
    assert "row_index" not in feedback
    assert "semantic path" in feedback
    assert "Key formatting rules" in feedback
    assert long_cell not in feedback


def test_langgraph_agent_accepts_answer_when_validator_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    def fail_validator(**_: object) -> dict[str, object]:
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", fail_validator
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.steps[-1].node == "validate_answer"
    assert result.steps[-1].ok is False
    assert result.steps[-1].tool_results[0]["error"] == "validator unavailable"
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["steps"][-1]["tool_results"][0]["error"] == "validator unavailable"


def test_langgraph_agent_receives_problem_in_sft_aligned_user_message(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)

    def fake_explore(self, *, context_dir, task_id=""):  # noqa: ANN001
        full_catalog = '{"task_id": "task_demo", "assets": [{"path": "sample.csv", "kind": "csv", "size": 10}], "schemas": [{"asset_path": "sample.csv", "kind": "csv", "fields": [{"name": "value", "type": "integer", "cardinality": 2, "missing_count": 0, "distinct_values": ["1", "2"]}]}], "knowledge_documents": []}'
        return full_catalog, {}

    def fake_analyze_ambiguity(*, model, question, schemas=None, knowledge_docs=None):  # noqa: ANN001
        return {
            "question_intent": {
                "entities": ["value"],
                "filters": [],
                "metrics": [],
                "requested_output": "value column",
                "grain": "",
            },
            "ambiguities": [],
            "resolved_by_knowledge": [],
            "non_ambiguous_candidates": [
                {
                    "phrase": "value",
                    "candidate_fields": ["sample.csv.value"],
                    "note": "Value column.",
                }
            ],
        }

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fake_explore,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.analyze_ambiguity",
        fake_analyze_ambiguity,
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["value"], "rows": [["1"], ["2"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            enable_ambiguity_analysis=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_request = model.invocations[0]
    assert [message.type for message in first_request] == ["system", "human"]
    user_content = first_request[1].content
    context_index = user_content.index("<context_injection>")
    catalog_index = user_content.index("<lightweight_catalog>")
    analysis_index = user_content.index("<ambiguity_analysis>")
    action_index = user_content.index("<action_trigger>")
    assert context_index < catalog_index < analysis_index < action_index
    assert "User Question: List the value column." in user_content
    assert "\nQuestion: List the value column." not in user_content
    assert "## Task Input" not in user_content
    assert "## Problem Analysis" not in user_content
    assert "## Global Data Profile" not in user_content
    assert "## Action Instruction" not in user_content
    assert "All tool file paths are relative" not in user_content
    assert "Each turn should make progress" not in user_content
    assert "call `submit_tool_result`" not in user_content
    assert "<lightweight_catalog>" in user_content
    assert "</lightweight_catalog>" in user_content
    assert "sample.csv" in user_content
    assert "value" in user_content
    assert "<ambiguity_analysis>" in user_content
    assert "</ambiguity_analysis>" in user_content
    assert '"requested_output": "value column"' in user_content
    assert '"non_ambiguous_candidates"' in user_content
    assert '"sample.csv.value"' in user_content
    assert "does NOT field-bind" in user_content
    assert "verify" in user_content


def test_langgraph_agent_candidate_preamble_absent_when_no_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)

    def fake_explore(self, *, context_dir, task_id=""):  # noqa: ANN001
        full_catalog = '{"task_id": "task_demo", "assets": [{"path": "sample.csv", "kind": "csv", "size": 10}], "schemas": [{"asset_path": "sample.csv", "kind": "csv", "fields": [{"name": "value", "type": "integer", "cardinality": 2, "missing_count": 0, "distinct_values": ["1", "2"]}]}], "knowledge_documents": []}'
        return full_catalog, {}

    def fake_analyze_ambiguity(*, model, question, schemas=None, knowledge_docs=None):  # noqa: ANN001
        return {
            "question_intent": {
                "entities": ["value"],
                "filters": [],
                "metrics": [],
                "requested_output": "value column",
                "grain": "",
            },
            "ambiguities": [],
            "resolved_by_knowledge": [],
            "non_ambiguous_candidates": [],
        }

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fake_explore,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.analyze_ambiguity",
        fake_analyze_ambiguity,
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["value"], "rows": [["1"], ["2"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            enable_ambiguity_analysis=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)
    assert result.succeeded is True
    user_content = model.invocations[0][1].content
    assert "ambiguity analysis" in user_content


def test_langgraph_agent_emits_in_progress_trace_before_model_invoke(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []

    class ObservingModel(ScriptedToolCallingModel):
        def invoke(self, messages):  # noqa: ANN001
            pending_step = trace_updates[-1]["steps"][-1]
            assert pending_step["node"] == "model"
            assert pending_step["status"] == "in_progress"
            assert pending_step["model_request"] is not None
            return super().invoke(messages)

    model = ObservingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True


def test_langgraph_agent_retries_model_request_errors_with_backoff(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RetryableStatusError("temporary request failure 1", 503),
            RetryableStatusError("temporary request failure 2", 503),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert model.invoke_count == 3
    assert sleep_delays == [5, 15]
    assert [step.node for step in result.steps] == ["model", "tool"]
    assert result.steps[0].model_response is not None
    request_retry = result.steps[0].model_response["request_retry"]
    assert request_retry["status"] == "succeeded_after_retry"
    assert request_retry["retry_count"] == 2
    assert request_retry["request_error_count"] == 2
    assert [event["error"] for event in request_retry["errors"]] == [
        "temporary request failure 1",
        "temporary request failure 2",
    ]
    assert [event["error_type"] for event in request_retry["errors"]] == [
        "RetryableStatusError",
        "RetryableStatusError",
    ]
    assert [event["status_code"] for event in request_retry["errors"]] == [503, 503]
    assert all(event["retryable"] is True for event in request_retry["errors"])


def test_langgraph_agent_live_trace_records_model_retry_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RetryableStatusError("temporary request failure", 503),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    retry_updates = []
    for update in trace_updates:
        model_response = update["steps"][-1].get("model_response") or {}
        if model_response.get("request_retry"):
            retry_updates.append(update)
    assert retry_updates
    live_retry = retry_updates[0]["steps"][-1]["model_response"]["request_retry"]
    assert live_retry["status"] == "retrying"
    assert live_retry["retry_count"] == 1
    assert live_retry["request_error_count"] == 1
    assert live_retry["errors"][0]["error"] == "temporary request failure"
    assert live_retry["errors"][0]["error_type"] == "RetryableStatusError"
    assert live_retry["errors"][0]["status_code"] == 503
    assert live_retry["errors"][0]["retryable"] is True
    assert live_retry["errors"][0]["next_retry_delay_seconds"] == 5
    assert sleep_delays == [5]


def test_langgraph_agent_live_trace_records_global_exploration_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []

    def fail_explore(self, *, context_dir, task_id=""):
        raise RuntimeError("synthetic profiling failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    fail_steps = [
        update
        for update in trace_updates
        if update["steps"][-1]["node"] == "global_data_exploration"
        and update["steps"][-1]["ok"] is False
        and update["steps"][-1].get("status") != "in_progress"
    ]
    assert fail_steps
    fail_step = fail_steps[0]["steps"][-1]
    assert "synthetic profiling failure" in fail_step["tool_results"][0]["error"]


def test_langgraph_agent_finalizes_after_non_retryable_error(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RetryableStatusError("temporary request failure 1", 503),
            RetryableStatusError("temporary request failure 2", 503),
            RetryableStatusError("temporary request failure 3", 503),
            RuntimeError("non-retryable failure"),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is False
    assert "non-retryable failure" in result.failure_reason
    assert model.invoke_count == 4
    assert sleep_delays == [5, 15, 30]
    assert [step.node for step in result.steps] == ["model"]
    assert result.steps[0].ok is False
    assert result.steps[0].model_response is not None
    request_retry = result.steps[0].model_response["request_retry"]
    assert request_retry["status"] == "failed_after_retries"
    assert request_retry["retry_count"] == 3
    assert request_retry["request_error_count"] == 4
    assert request_retry["errors"][-1]["error"] == "non-retryable failure"
    assert request_retry["errors"][-1]["error_type"] == "RuntimeError"
    assert request_retry["errors"][-1]["retryable"] is False
    assert request_retry["errors"][-1]["will_retry"] is False


def test_langgraph_agent_handles_nullable_completion_token_details(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                id="response_nullable",
                response_metadata={
                    "id": "response_nullable",
                    "finish_reason": "tool_calls",
                    "token_usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 22,
                        "completion_tokens_details": None,
                    },
                },
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["response_id"] == "response_nullable"
    assert result.steps[0].model_response["input_tokens"] == 11
    assert result.steps[0].model_response["output_tokens"] == 22
    assert result.steps[0].model_response["reasoning_tokens"] is None


def test_langgraph_agent_records_tool_errors_without_crashing(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_csv", "args": {}, "id": "call_1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    tool_steps = [step for step in result.steps if step.node == "tool"]
    assert tool_steps[0].ok is False
    assert "error" in tool_steps[0].tool_results[0]
    assert "Tool is not available to the model" in tool_steps[0].tool_results[0]["error"]
    recovery_model_step = next(
        step for step in result.steps if step.node == "model" and step.step_index == 3
    )
    assert recovery_model_step.model_request is not None
    assert recovery_model_step.model_request["last_message"]["type"] == "tool"
    assert recovery_model_step.model_request["last_message"]["name"] == "read_csv"
    assert recovery_model_step.model_request["last_message"]["status"] == "error"


def test_langgraph_agent_uses_temporary_workspace_for_python(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {
                            "code": "from pathlib import Path\nPath('generated.txt').write_text('ok')\nprint('done')"
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert not (task.context_dir / "generated.txt").exists()
    python_tool_step = next(step for step in result.steps if step.node == "tool")
    tool_result = python_tool_step.tool_results[0]
    assert tool_result["ok"] is True
    assert "done" in json.dumps(tool_result["content"], ensure_ascii=False)


def test_langgraph_agent_truncates_tool_step_results_for_trace_and_messages(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {"code": "print('x' * 200)"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(ToolConfig(max_output_tokens=10, max_list_items=200)),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    python_tool_step = next(step for step in result.steps if step.node == "tool")
    output = python_tool_step.tool_results[0]["content"]["output"]
    assert output.startswith("x")
    assert "内容已被截断" in output
    assert len(output) < 200

    second_request_messages = model.invocations[1]
    tool_message = next(
        message
        for message in second_request_messages
        if getattr(message, "name", None) == "execute_python"
    )
    tool_payload = json.loads(str(tool_message.content))
    assert tool_payload["content"]["output"] == output
    assert "内容已被截断" in tool_payload["content"]["output"]


def test_langgraph_agent_records_reasoning_content_in_model_response(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "I should inspect the context first."},
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_step = result.steps[0]
    assert first_step.assistant_message == "I should inspect the context first."
    assert first_step.model_response is not None
    assert first_step.model_response["reasoning_content"] == "I should inspect the context first."
    assert first_step.model_response["reasoning_content_length"] == 35


def test_langgraph_agent_adds_clean_reasoning_content_to_history(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": "I should inspect available files before answering."
                },
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(model.invocations) == 2
    second_request_messages = model.invocations[1]
    assert [message.type for message in second_request_messages] == [
        "system",
        "human",
        "ai",
        "tool",
    ]
    assistant_message = second_request_messages[-2]
    tool_message = second_request_messages[-1]
    assert isinstance(tool_message, ToolMessage)
    assert assistant_message.content == "I should inspect available files before answering."
    assert assistant_message.tool_calls[0]["name"] == "list_context"
    assert assistant_message.additional_kwargs["reasoning_content"] == (
        "I should inspect available files before answering."
    )
    assert all(not key.startswith("_dab_") for key in assistant_message.additional_kwargs)
    assert all(
        not str(getattr(message, "content", "")).startswith(
            "Previous model reasoning_content from the last turn"
        )
        for message in second_request_messages
    )
    assert result.steps[2].model_request is not None
    assert result.steps[2].model_request["last_message"]["type"] == "tool"


def test_langgraph_agent_can_strip_reasoning_content_from_history(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content=(
                    "Visible note before the tool.\n"
                    "<tool_call><function=list_context></function></tool_call>"
                ),
                additional_kwargs={
                    "reasoning_content": "Hidden reasoning that should not become content."
                },
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, strip_reasoning_history=True),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].assistant_message == "Visible note before the tool."
    second_request_messages = model.invocations[1]
    assistant_message = second_request_messages[-2]
    assert assistant_message.content == "Visible note before the tool."
    assert "Hidden reasoning" not in assistant_message.content
    assert "reasoning_content" not in assistant_message.additional_kwargs
    assert "<tool_call>" not in assistant_message.content
    assert assistant_message.tool_calls[0]["name"] == "list_context"


def test_langgraph_agent_can_limit_reasoning_history_to_zero(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": "Reasoning that should be removed from requests."
                },
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, reasoning_history_limit=0),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["reasoning_content"] == (
        "Reasoning that should be removed from requests."
    )
    second_request_messages = model.invocations[1]
    assistant_message = second_request_messages[-2]
    assert assistant_message.content == ""
    assert "reasoning_content" not in assistant_message.additional_kwargs
    assert assistant_message.tool_calls[0]["name"] == "list_context"


def test_langgraph_agent_keeps_only_recent_reasoning_history(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "First reasoning."},
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "notes.md"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "Second reasoning."},
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 1},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_3",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=6, reasoning_history_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    third_request_messages = model.invocations[2]
    ai_messages = [message for message in third_request_messages if isinstance(message, AIMessage)]
    assert len(ai_messages) == 2
    assert ai_messages[0].content == ""
    assert "reasoning_content" not in ai_messages[0].additional_kwargs
    assert ai_messages[0].tool_calls[0]["name"] == "read_doc"
    assert ai_messages[1].content == "Second reasoning."
    assert ai_messages[1].additional_kwargs["reasoning_content"] == "Second reasoning."
    assert ai_messages[1].tool_calls[0]["name"] == "list_context"


def test_langgraph_agent_reasoning_history_limit_preserves_visible_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content=(
                    "Visible note before the tool.\n"
                    "<tool_call><function=list_context></function></tool_call>"
                ),
                additional_kwargs={"reasoning_content": "Reasoning that should be removed."},
                tool_calls=[
                    {
                        "name": "list_context",
                        "args": {"max_depth": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, reasoning_history_limit=0),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assistant_message = model.invocations[1][-2]
    assert assistant_message.content == "Visible note before the tool."
    assert "reasoning_content" not in assistant_message.additional_kwargs
    assert "<tool_call>" not in assistant_message.content


def test_langgraph_agent_recovers_pseudo_tool_call_from_reasoning_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "I need to read notes.md next.\n"
                        "<tool_call><function=read_doc><parameter=path>notes.md</parameter></function></tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=6, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert result.steps[0].tool_calls == [
        {
            "id": result.steps[0].tool_calls[0]["id"],
            "name": "read_doc",
            "args": {"path": "notes.md"},
        }
    ]
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["recovered_tool_call"] is True
    assert result.steps[0].model_response["recovered_tool_call_source"] == "reasoning_content"
    assert result.steps[0].model_response["recovered_tool_call_name"] == "read_doc"
    second_request_messages = model.invocations[1]
    assert [message.type for message in second_request_messages] == [
        "system",
        "human",
        "ai",
        "tool",
    ]
    assert second_request_messages[-2].content == "I need to read notes.md next."
    assert "<tool_call>" not in second_request_messages[-2].content
    assert second_request_messages[-2].tool_calls[0]["name"] == "read_doc"
    assert second_request_messages[-1].name == "read_doc"


def test_langgraph_agent_recovers_multiline_python_pseudo_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    code = "value = 1\nprint(value)\n"
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "<tool_call>\n"
                        "<function=execute_python>\n"
                        f"<parameter=code>\n{code}</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert result.steps[0].tool_calls[0]["name"] == "execute_python"
    assert result.steps[0].tool_calls[0]["args"]["code"] == code.strip()


def test_langgraph_agent_does_not_recover_unknown_pseudo_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "I should use the tool next.\n"
                        "<tool_call><function=unknown_tool><parameter=path>notes.md</parameter></function></tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool"]
    assert result.steps[0].model_response is not None
    assert "recovered_tool_call" not in result.steps[0].model_response
    assert model.invocations[1][-2].type == "ai"
    assert model.invocations[1][-2].content == "I should use the tool next."
    assert "<tool_call>" not in model.invocations[1][-2].content
    assert model.invocations[1][-1].type == "human"


def test_langgraph_agent_does_not_override_native_tool_call_with_pseudo_tool_call(
    tmp_path: Path,
) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "<tool_call><function=execute_python><parameter=code>print('wrong')</parameter></function></tool_call>"
                    )
                },
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "notes.md"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["done"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].tool_calls[0]["name"] == "read_doc"
    assert result.steps[0].model_response is not None
    assert "recovered_tool_call" not in result.steps[0].model_response


def test_langgraph_agent_retries_once_after_empty_stop(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool"]
    second_model_step = result.steps[2]
    assert second_model_step.model_request is not None
    assert second_model_step.model_request["last_message"]["type"] == "human"
    assert second_model_step.model_request["last_message"]["content_preview"].startswith(
        "Your previous response did not call a tool."
    )
    assert "immediately call a tool" in model.invocations[1][-1].content
    assert second_model_step.model_request["last_message"]["content_length"] == len(
        model.invocations[1][-1].content
    )
    assert [message.type for message in model.invocations[1]].count("system") == 1
    assert model.invocations[1][0].type == "system"


def test_empty_tool_calls_response_is_repairable() -> None:
    message = AIMessage(
        content="I have the answer and can submit it.",
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[],
    )

    assert _is_non_action_stop(message) is True


def test_invalid_tool_call_is_removed_before_repair_request(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    malformed_response = AIMessage(
        content="I will submit the answer.",
        response_metadata={"finish_reason": "tool_calls"},
        invalid_tool_calls=[
            {
                "name": "submit_tool_result",
                "args": '{"tool_name": "execute_python"',
                "id": "bad_call_1",
                "error": "Could not parse tool input",
            }
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "bad_call_1",
                    "function": {
                        "name": "submit_tool_result",
                        "arguments": '{"tool_name": "execute_python"',
                    },
                }
            ]
        },
    )
    model = ScriptedToolCallingModel(
        responses=[
            malformed_response,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print(" + repr(json.dumps({"columns": ["status"], "rows": [["recovered"]]})) + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool"]
    assert result.steps[0].model_response["invalid_tool_call_count"] == 1
    assert "not valid JSON" in result.steps[1].assistant_message
    assert "submit_tool_result" in result.steps[1].assistant_message
    assert "execute_python" in result.steps[1].assistant_message
    replayed_message = model.invocations[1][-2]
    assert isinstance(replayed_message, AIMessage)
    assert replayed_message.invalid_tool_calls == []
    assert replayed_message.additional_kwargs.get("tool_calls") is None


def test_discard_invalid_tool_calls_preserves_normal_message_content() -> None:
    message = AIMessage(
        content="Please retry.",
        invalid_tool_calls=[
            {"name": "read_doc", "args": "{bad", "id": "bad", "error": "invalid JSON"}
        ],
        additional_kwargs={"function_call": {"name": "read_doc", "arguments": "{bad"}},
    )

    cleaned = _discard_invalid_tool_calls(message)

    assert cleaned.content == "Please retry."
    assert cleaned.tool_calls == []
    assert cleaned.invalid_tool_calls == []
    assert cleaned.additional_kwargs == {}


def test_langgraph_agent_retries_after_non_tool_stop_with_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="I should inspect the context next, then continue.",
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool"]
    assert result.steps[0].assistant_message == "I should inspect the context next, then continue."
    assert "immediately call a tool" in model.invocations[1][-1].content


def test_langgraph_agent_resets_empty_stop_retry_after_tool_progress(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "missing.md"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["recovered_twice"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=8, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "repair",
        "model",
        "tool",
        "model",
        "repair",
        "model",
        "tool",
    ]
    assert sum(1 for step in result.steps if step.node == "repair") == 2
