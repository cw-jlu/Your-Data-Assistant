from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from data_agent_baseline import cli as cli_module
from data_agent_baseline.asr_runtime import (
    AsrPipelineRuntime,
    AsrRuntimeSettings,
    build_initial_prompt,
    build_term_request,
    parse_prompt_response,
    parse_term_response,
    sha256_text,
)
from data_agent_baseline.asr_eval.azure import (
    AZURE_API_VERSION,
    AzureFastTranscriptionClient,
    AzureFastTranscriptionError,
    extract_azure_reference,
    resolve_azure_credentials,
)
from data_agent_baseline.asr_eval.core import (
    classify_asr_run,
    discover_asr_samples,
    find_asr_run_dir,
)
from data_agent_baseline.asr_eval.pipeline import generate_azure_gold, run_candidate_asr
from data_agent_baseline.asr_eval.scoring import (
    edit_counts,
    normalize_chinese,
    normalize_english,
    score_asr_run,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "asr" / "azure_fast_transcription.json"
runner = CliRunner()


def _make_dataset(root: Path, task_ids: tuple[str, ...] = ("task_1", "task_2")) -> Path:
    dataset_root = root / "data"
    for task_id in task_ids:
        video = dataset_root / "input" / task_id / "context" / "video" / "briefing.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{task_id}".encode())
        (video.parents[2] / "task.json").write_text(
            json.dumps({"task_id": task_id, "question": f"Question for {task_id}"}),
            encoding="utf-8",
        )
        (video.parents[1] / "knowledge.md").write_text(
            f"# Knowledge for {task_id}\n",
            encoding="utf-8",
        )
    return dataset_root


def _write_wav(_: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)


class _FakeAzureClient:
    endpoint = "https://example.api.cognitive.microsoft.com"
    api_version = AZURE_API_VERSION

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        self.calls.append(audio_path.stem)
        if audio_path.stem == "task_2":
            return {
                "durationMilliseconds": 100,
                "combinedPhrases": [{"text": "Hello world"}],
                "phrases": [
                    {
                        "text": "Hello world",
                        "locale": "en-US",
                        "durationMilliseconds": 100
                    }
                ],
            }
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class _FakeWhisperModel:
    def __init__(self, *, fail_task_2: bool = False) -> None:
        self.fail_task_2 = fail_task_2
        self.calls: list[str] = []

    def transcribe(self, audio_path: str):
        task_id = Path(audio_path).stem
        self.calls.append(task_id)
        if task_id == "task_2" and self.fail_task_2:
            raise RuntimeError("synthetic failure")
        if task_id == "task_2":
            segments = [
                SimpleNamespace(start=0.0, end=0.1, text="Hello brave world")
            ]
            info = SimpleNamespace(language="en", language_probability=0.99, duration=0.1)
        else:
            segments = [SimpleNamespace(start=0.0, end=0.1, text="栏位 配置")]
            info = SimpleNamespace(language="zh", language_probability=0.99, duration=0.1)
        return iter(segments), info


def test_prompt_validation_and_language_specific_ui_terms() -> None:
    raw = json.dumps(
        {
            "domain_context": "公司档案与联系方式查询核对",
            "terms": [
                "字段",
                "字段",
                "批次",
                "配置",
                "加载",
                "面板",
                "看板",
                "队列",
                "筛选",
                "路径",
            ]
        },
        ensure_ascii=False,
    )
    terms = parse_term_response(raw, min_terms=1, max_terms=10)
    domain_context, scoped_terms = parse_prompt_response(
        raw,
        min_terms=1,
        max_terms=10,
    )
    assert domain_context == "公司档案与联系方式查询核对"
    assert scoped_terms == terms
    assert len(terms) == 9
    assert terms.count("字段") == 1
    assert "界面常见术语包括" in (
        build_initial_prompt(
            language="zh",
            domain_context=domain_context,
            dynamic_terms=terms,
            include_ui_terms=True,
        )
        or ""
    )
    assert "公司档案与联系方式查询核对" in (
        build_initial_prompt(
            language="zh",
            domain_context=domain_context,
            dynamic_terms=terms,
            include_ui_terms=True,
        )
        or ""
    )
    english_prompt = build_initial_prompt(
        language="en",
        dynamic_terms=["seasoned offering", "issue purpose"],
        include_ui_terms=True,
    )
    assert english_prompt is not None
    assert "字段" not in english_prompt

    chinese_request = build_term_request(
        question="查询董秘联系方式",
        knowledge_text="公司档案",
        language="zh",
        min_terms=1,
        max_terms=10,
    )
    english_request = build_term_request(
        question="Find major rights offering shareholders",
        knowledge_text="Rights offering entitlement guide",
        language="en",
        min_terms=1,
        max_terms=10,
    )
    assert "你要为中文 Whisper 生成" in chinese_request
    assert "Task question:" not in chinese_request
    assert "最多返回 10 个" in chinese_request
    assert "不得包含英文单词" in chinese_request
    assert "8 至" not in chinese_request
    assert "initial_prompt for English Whisper" in english_request
    assert "任务问题：" not in english_request
    assert "at most 10" in english_request
    assert "there is no minimum" in english_request

    english_domain, english_terms = parse_prompt_response(
        json.dumps(
            {
                "domain_context": "Rights offering entitlement review workflow",
                "terms": [
                    "Rights Offering",
                    "Entitlement Standard",
                    "Major Shareholder",
                    "Stock Code",
                    "Subscription Threshold",
                    "Compliance Review",
                    "A-share Company",
                    "Export List",
                    "股东名称",
                ],
            }
        ),
        min_terms=1,
        max_terms=10,
        language="en",
    )
    assert english_domain == "Rights offering entitlement review workflow"
    assert "股东名称" not in english_terms

    capped_terms = parse_term_response(
        json.dumps({"terms": [f"term-{index}" for index in range(20)]}),
        min_terms=1,
        max_terms=10,
    )
    assert len(capped_terms) == 10

    seven_terms = parse_term_response(
        json.dumps({"terms": [f"term-{index}" for index in range(7)]}),
        min_terms=1,
        max_terms=10,
    )
    assert len(seven_terms) == 7

    with pytest.raises(ValueError, match="any valid terms"):
        parse_term_response(
            json.dumps({"terms": []}),
            min_terms=1,
            max_terms=10,
        )

    _, chinese_only_terms = parse_prompt_response(
        json.dumps(
            {
                "domain_context": "涉及基金经理任职稳定性筛选",
                "terms": [
                    "任职天数",
                    "准入线",
                    "personalcode",
                    "RMB",
                    "基金经理",
                ],
            },
            ensure_ascii=False,
        ),
        min_terms=1,
        max_terms=10,
        language="zh",
    )
    assert chinese_only_terms == ["任职天数", "准入线", "基金经理"]


def test_runtime_loads_models_once_and_records_full_pipeline() -> None:
    factory_calls: list[str] = []

    class Tiny:
        def detect_language(self, _audio: object, **_kwargs: object):
            return "zh", 0.99, [("zh", 0.99), ("en", 0.01)]

    class Medium:
        prompts: list[str | None] = []

        def transcribe(self, _audio: object, **kwargs: object):
            self.prompts.append(kwargs.get("initial_prompt"))
            return iter(
                [SimpleNamespace(start=0.0, end=1.0, text="欄位與設定")]
            ), SimpleNamespace(language="zh", language_probability=0.99, duration=1.0)

    tiny = Tiny()
    medium = Medium()

    def factory(**kwargs: object):
        name = str(kwargs["model_size_or_path"])
        factory_calls.append(name)
        return tiny if name == "tiny" else medium

    class PromptModel:
        def invoke(self, _request: str) -> SimpleNamespace:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "domain_context": "年度定增项目审核与募资用途整理",
                        "terms": [
                            "年度批次",
                            "定增项目",
                            "公司名称",
                            "募资用途",
                            "受理日期",
                            "自然年度",
                            "增发记录",
                            "审核配置",
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    runtime = AsrPipelineRuntime(
        AsrRuntimeSettings(num_workers=8),
        model_factory=factory,
        audio_loader=lambda _path: object(),
    )
    for _ in range(2):
        result = runtime.transcribe(
            Path("task_6.wav"),
            stage="dynamic-plus-ui",
            question="年度定增审核批次",
            knowledge_text="增发目的",
            prompt_model=PromptModel(),
            prompt_model_name="qwen3.5-35b-a3b",
        )
        assert result["text"] == "栏位与设定"
        assert result["prompt"]["status"] == "ok"
        assert result["prompt"]["domain_context"] == "年度定增项目审核与募资用途整理"
        assert result["language_detection"]["status"] == "ok"
        assert "字段" in str(result["prompt"]["initial_prompt"])
    assert factory_calls == ["tiny", "medium"]


def test_runtime_safe_fallbacks_for_tiny_and_qwen_failures() -> None:
    class Tiny:
        def detect_language(self, _audio: object, **_kwargs: object):
            raise RuntimeError("tiny failed")

    class Medium:
        def transcribe(self, _audio: object, **kwargs: object):
            assert "language" not in kwargs
            assert "initial_prompt" not in kwargs
            return iter([]), SimpleNamespace(
                language="en",
                language_probability=0.9,
                duration=1.0,
            )

    runtime = AsrPipelineRuntime(
        AsrRuntimeSettings(),
        model_factory=lambda **kwargs: (
            Tiny() if kwargs["model_size_or_path"] == "tiny" else Medium()
        ),
        audio_loader=lambda _path: object(),
    )
    result = runtime.transcribe(
        Path("audio.wav"),
        stage="dynamic-plus-ui",
        prompt_model=None,
    )
    assert result["language_detection"]["status"] == "failed"
    assert result["prompt"]["status"] == "skipped_language_fallback"
    assert result["language"] == "en"


def test_normalization_handles_traditional_case_unicode_and_punctuation() -> None:
    assert normalize_chinese("欄位，ＡＢＣ １２３!") == "栏位abc123"
    assert normalize_chinese("十，十一，二十，一百零二") == "101120102"
    assert normalize_chinese("二零二四年六个月") == "2024年6个月"
    assert normalize_chinese("第十一名与第20名") == "第11名与第20名"
    assert normalize_chinese("60万0230到60万0240") == "600230到600240"
    assert normalize_chinese("两亿元") == "200000000元"
    assert normalize_english("Hello，WORLD's １２３") == ["hello", "world", "s", "123"]
    assert normalize_chinese("") == ""
    assert normalize_english("") == []


def test_edit_counts_reports_substitution_deletion_and_insertion() -> None:
    substitution = edit_counts("abc", "axc")
    deletion = edit_counts("abc", "ac")
    insertion = edit_counts("abc", "abxc")
    assert (substitution.substitutions, substitution.deletions, substitution.insertions) == (1, 0, 0)
    assert (deletion.substitutions, deletion.deletions, deletion.insertions) == (0, 1, 0)
    assert (insertion.substitutions, insertion.deletions, insertion.insertions) == (0, 0, 1)


def test_edit_counts_can_ignore_english_token_split_and_merge_variants() -> None:
    split = edit_counts(
        ["pop", "up", "and", "cutoff"],
        ["popup", "and", "cut", "off"],
        ignore_token_boundaries=True,
    )
    assert (split.substitutions, split.deletions, split.insertions) == (0, 0, 0)
    assert split.reference_length == 4

    real_error = edit_counts(
        ["pop", "up", "now"],
        ["popup", "later"],
        ignore_token_boundaries=True,
    )
    assert (
        real_error.substitutions,
        real_error.deletions,
        real_error.insertions,
    ) == (1, 0, 0)


def test_discover_asr_samples_finds_video_tasks_in_numeric_order(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_11", "task_2"))
    samples = discover_asr_samples(dataset_root)
    assert [sample.task_id for sample in samples] == ["task_2", "task_11"]
    assert all(len(sample.source_sha256) == 64 for sample in samples)


def test_asr_run_categories_and_recursive_lookup(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "asr"
    run_dir = output_root / "runs" / "prompt-ablation" / "prompt-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert find_asr_run_dir(output_root, "prompt-run") == run_dir.resolve()
    assert (
        classify_asr_run(
            run_id="medium-prompt-all",
            sample_count=30,
            stage="dynamic-prompt",
        )
        == "prompt-ablation"
    )
    assert (
        classify_asr_run(
            run_id="medium-speed-task6",
            sample_count=1,
            stage="medium-baseline",
        )
        == "speed-tests"
    )


def test_run_candidate_reads_prompt_inputs_when_dataset_root_is_input_dir(
    tmp_path: Path,
) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_1",))
    captured: dict[str, str] = {}

    class _RuntimeModel:
        def detect_language(
            self, *_: Any, **__: Any
        ) -> tuple[str, float, list[tuple[str, float]]]:
            return "zh", 0.99, [("zh", 0.99), ("en", 0.01)]

        def transcribe(self, *_: Any, **kwargs: Any) -> tuple[list[Any], Any]:
            captured["initial_prompt"] = str(kwargs.get("initial_prompt") or "")
            return [SimpleNamespace(start=0.0, end=1.0, text="字段")], SimpleNamespace(
                language="zh",
                language_probability=0.99,
            )

    class _PromptModel:
        def invoke(self, messages: list[Any]) -> SimpleNamespace:
            captured["request"] = str(messages)
            return SimpleNamespace(
                content=(
                    '{"domain_context":"任务业务范围与数据处理流程",'
                    '"terms":["字段","批次","配置","加载","面板","看板","卡片","队列"]}'
                )
            )

    _, manifest = run_candidate_asr(
        project_root=tmp_path,
        dataset_root=dataset_root / "input",
        run_id="input-root",
        model_name="medium",
        device="cpu",
        compute_type="int8",
        stage="dynamic-prompt",
        prompt_model=_PromptModel(),
        prompt_model_name="fake-qwen",
        model_factory=lambda **_: _RuntimeModel(),
        audio_loader=lambda _: [0.0],
        audio_decoder=_write_wav,
    )

    sample = manifest["samples"][0]
    assert sample["question_sha256"] != sha256_text("")
    assert sample["knowledge_sha256"] != sha256_text("")
    assert "Question for task_1" in captured["request"]
    assert "Knowledge for task_1" in captured["request"]


def test_extract_azure_reference_uses_dominant_phrase_locale() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    text, locale = extract_azure_reference(payload)
    assert text == "欄位 配置"
    assert locale == "zh-CN"


def test_azure_client_retries_429_without_leaking_key(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio, audio)

    class Response:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self.payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, Any]:
            return self.payload

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.headers: list[dict[str, str]] = []
            self.data: list[dict[str, str]] = []
            self.file_fields: list[set[str]] = []

        def post(self, _url: str, **kwargs: Any) -> Response:
            self.calls += 1
            self.headers.append(kwargs["headers"])
            self.data.append(kwargs["data"])
            self.file_fields.append(set(kwargs["files"]))
            if self.calls == 1:
                return Response(429, {"error": {"message": "quota"}})
            return Response(200, json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    http_client = Client()
    sleeps: list[float] = []
    client = AzureFastTranscriptionClient(
        endpoint="https://example.api.cognitive.microsoft.com",
        key="top-secret",
        client=http_client,
        sleep=sleeps.append,
    )
    payload = client.transcribe(audio)
    assert payload["combinedPhrases"][0]["text"] == "欄位 配置"
    assert http_client.calls == 2
    assert sleeps == [1.0]
    assert json.loads(http_client.data[0]["definition"]) == {
        "locales": ["zh-CN", "en-US"],
        "channels": [0],
        "profanityFilterMode": "Masked",
    }
    assert http_client.file_fields == [{"audio"}, {"audio"}]
    assert "top-secret" not in json.dumps(payload)


def test_missing_azure_credentials_fail_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    with pytest.raises(AzureFastTranscriptionError, match="AZURE_SPEECH_ENDPOINT"):
        resolve_azure_credentials(tmp_path)


def test_cli_generate_gold_reports_missing_credentials_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_1",))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    fake_config = SimpleNamespace(
        dataset=SimpleNamespace(root_path=dataset_root),
        run=SimpleNamespace(task_ids=None),
    )
    monkeypatch.setattr(cli_module, "load_app_config", lambda _: fake_config)
    monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "ASR_EVALUATION_DIR", tmp_path / "evaluation" / "asr")
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)

    result = runner.invoke(
        cli_module.app,
        [
            "asr-generate-gold",
            "--config",
            str(config_path),
            "--gold-id",
            "missing-credentials",
        ],
    )

    assert result.exit_code != 0
    assert "Missing Azure Speech credentials" in result.output
    assert not (tmp_path / "evaluation").exists()


def test_fixture_pipeline_generates_gold_run_and_reports(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path)
    evaluation_root = tmp_path / "evaluation" / "asr"
    artifacts_root = tmp_path / "artifacts" / "asr"
    azure = _FakeAzureClient()
    gold_path, gold = generate_azure_gold(
        project_root=tmp_path,
        dataset_root=dataset_root,
        gold_id="fixture-gold",
        azure_client=azure,
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
    )
    assert gold_path.is_file()
    assert gold["completed"] is True
    assert len(azure.calls) == 2
    assert (gold_path.parent / "raw" / "task_1.json").is_file()

    factory_calls: list[dict[str, str]] = []
    model = _FakeWhisperModel()

    def factory(**kwargs: str) -> _FakeWhisperModel:
        factory_calls.append(kwargs)
        return model

    run_path, run = run_candidate_asr(
        project_root=tmp_path,
        dataset_root=dataset_root,
        run_id="base-fixture",
        model_name="base",
        device="cpu",
        compute_type="int8",
        model_factory=factory,
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
        artifacts_root=artifacts_root,
    )
    assert run_path.is_file()
    assert run["completed"] is True
    assert len(factory_calls) == 1
    assert model.calls == ["task_1", "task_2"]

    score_path, score = score_asr_run(
        project_root=tmp_path,
        gold_id="fixture-gold",
        run_id="base-fixture",
        evaluation_root=evaluation_root,
        artifacts_root=artifacts_root,
    )
    assert score_path.is_file()
    assert score["language_summaries"]["zh"]["corpus_error_rate"] == 0
    assert score["language_summaries"]["en"]["insertions"] == 1
    assert score["language_summaries"]["en"]["corpus_error_rate"] == pytest.approx(0.5)
    assert score["overview"]["language_detection_accuracy"] == 1
    assert (score_path.parent / "per_sample.csv").is_file()
    assert (score_path.parent / "score_report.md").is_file()

    _, resumed_gold = generate_azure_gold(
        project_root=tmp_path,
        dataset_root=dataset_root,
        gold_id="fixture-gold",
        azure_client=azure,
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
    )
    assert resumed_gold["completed"] is True
    assert len(azure.calls) == 2


def test_failed_candidate_is_scored_as_full_deletion(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_2",))
    evaluation_root = tmp_path / "evaluation" / "asr"
    artifacts_root = tmp_path / "artifacts" / "asr"
    generate_azure_gold(
        project_root=tmp_path,
        dataset_root=dataset_root,
        gold_id="gold",
        azure_client=_FakeAzureClient(),
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
    )
    run_candidate_asr(
        project_root=tmp_path,
        dataset_root=dataset_root,
        run_id="failed",
        model_name="base",
        device="cpu",
        compute_type="int8",
        model_factory=lambda **_: _FakeWhisperModel(fail_task_2=True),
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
        artifacts_root=artifacts_root,
    )
    _, score = score_asr_run(
        project_root=tmp_path,
        gold_id="gold",
        run_id="failed",
        evaluation_root=evaluation_root,
        artifacts_root=artifacts_root,
    )
    row = score["samples"][0]
    assert row["candidate_succeeded"] is False
    assert row["deletions"] == 2
    assert row["error_rate"] == 1
    assert score["overview"]["success_rate"] == 0


def test_frozen_gold_rejects_changed_source_hash(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_1",))
    evaluation_root = tmp_path / "evaluation" / "asr"
    client = _FakeAzureClient()
    generate_azure_gold(
        project_root=tmp_path,
        dataset_root=dataset_root,
        gold_id="frozen",
        azure_client=client,
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
    )
    video = dataset_root / "input" / "task_1" / "context" / "video" / "briefing.mp4"
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="new gold_id"):
        generate_azure_gold(
            project_root=tmp_path,
            dataset_root=dataset_root,
            gold_id="frozen",
            azure_client=client,
            audio_decoder=_write_wav,
            evaluation_root=evaluation_root,
        )


def test_cli_asr_score_is_fully_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _make_dataset(tmp_path, ("task_1",))
    evaluation_root = tmp_path / "evaluation" / "asr"
    artifacts_root = tmp_path / "artifacts" / "asr"
    generate_azure_gold(
        project_root=tmp_path,
        dataset_root=dataset_root,
        gold_id="gold",
        azure_client=_FakeAzureClient(),
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
    )
    run_candidate_asr(
        project_root=tmp_path,
        dataset_root=dataset_root,
        run_id="run",
        model_name="base",
        device="cpu",
        compute_type="int8",
        model_factory=lambda **_: _FakeWhisperModel(),
        audio_decoder=_write_wav,
        evaluation_root=evaluation_root,
        artifacts_root=artifacts_root,
    )
    monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "ASR_EVALUATION_DIR", evaluation_root)
    monkeypatch.setattr(cli_module, "ASR_ARTIFACTS_DIR", artifacts_root)
    result = runner.invoke(
        cli_module.app,
        ["asr-score", "--gold-id", "gold", "--run-id", "run"],
    )
    assert result.exit_code == 0, result.output
    assert "zh CER" in result.output
    assert "relative to Azure" in result.output
