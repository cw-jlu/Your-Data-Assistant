from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

ASR_STAGES = (
    "medium-baseline",
    "tiny-route",
    "dynamic-prompt",
    "dynamic-plus-ui",
)
DEFAULT_UI_TERMS = (
    "字段",
    "批次",
    "配置",
    "加载",
    "面板",
    "看板",
    "卡片",
    "队列",
    "筛选",
    "核对",
    "列",
    "路径",
)
PROMPT_TEMPLATE_VERSION = "top1-question-filtered-terms-v9-max10-localized"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_uses_tiny(stage: str) -> bool:
    return stage in {"tiny-route", "dynamic-prompt", "dynamic-plus-ui"}


def stage_uses_dynamic_prompt(stage: str) -> bool:
    return stage in {"dynamic-prompt", "dynamic-plus-ui"}


def stage_uses_ui_terms(stage: str) -> bool:
    return stage == "dynamic-plus-ui"


def validate_stage(stage: str) -> str:
    normalized = str(stage).strip().lower()
    if normalized not in ASR_STAGES:
        raise ValueError(
            f"Unsupported ASR stage {stage!r}; expected one of {', '.join(ASR_STAGES)}."
        )
    return normalized


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content or "").strip()


def parse_prompt_response(
    raw_response: str,
    *,
    min_terms: int,
    max_terms: int,
    require_domain_context: bool = True,
    language: str | None = None,
) -> tuple[str | None, list[str]]:
    text = raw_response.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen term response does not contain a JSON object.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen term response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Qwen prompt response must be a JSON object.")
    raw_domain_context = payload.get("domain_context")
    domain_context: str | None = None
    if isinstance(raw_domain_context, str):
        domain_context = re.sub(r"\s+", " ", raw_domain_context).strip(
            " \t\r\n\"'，,。.;；"
        )
        if not 8 <= len(domain_context) <= 240:
            raise ValueError("Qwen domain_context must contain 8 to 240 characters.")
        if language == "en" and re.search(r"[\u3400-\u9fff]", domain_context):
            raise ValueError("English domain_context must not contain Chinese characters.")
    if require_domain_context and domain_context is None:
        raise ValueError("Qwen prompt response must contain domain_context.")

    raw_terms = payload.get("terms")
    if not isinstance(raw_terms, list):
        raise ValueError("Qwen term response must contain a terms array.")

    terms: list[str] = []
    seen: set[str] = set()
    for value in raw_terms:
        if not isinstance(value, str):
            continue
        term = re.sub(r"\s+", " ", value).strip(" \t\r\n,，;；。")
        if language == "en" and re.search(r"[\u3400-\u9fff]", term):
            continue
        if language == "zh" and re.search(r"[A-Za-z]", term):
            continue
        key = term.casefold()
        if not term or len(term) > 64 or "\n" in term or key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    if not terms:
        raise ValueError("Qwen prompt response does not contain any valid terms.")
    return domain_context, terms


def parse_term_response(
    raw_response: str,
    *,
    min_terms: int,
    max_terms: int,
) -> list[str]:
    _, terms = parse_prompt_response(
        raw_response,
        min_terms=min_terms,
        max_terms=max_terms,
        require_domain_context=False,
    )
    return terms


def build_term_request(
    *,
    question: str,
    knowledge_text: str,
    language: str,
    min_terms: int,
    max_terms: int,
) -> str:
    if language == "zh":
        return (
            "你要为中文 Whisper 生成 initial_prompt 的业务术语前半段。"
            "严格遵守三条规则："
            "①只以任务问题为筛选器，提取问题明确出现或紧扣问题的看板名、模块名、"
            "字段名、业务口径、实体标识，以及容易被识别成同音字的专有词；"
            "②knowledge.md 只用于核对这些候选词的中文规范写法或常用中文别名，"
            "绝不能从 knowledge.md 扩展问题范围，不能搬入整库字段；"
            f"③只保留正确的中文标准词，最多返回 {max_terms} 个，宁少勿滥，"
            "不要求凑足数量。专业术语中不得包含英文单词、英文缩写、拼音或"
            "英文数据库字段名；"
            "先生成 domain_context：用一个以“涉及”开头的简洁短语概括本题视频范围，"
            "只包含问题直接支持的对象、字段、口径和操作。不要回答问题，不要推断"
            "结果行，也不要加入问题未明确支持的相邻业务。"
            "不要选择字段、批次、配置、加载、面板、看板、卡片、队列、筛选、"
            "核对、列、路径等通用界面词，这些词会在中文 Prompt 中另外追加。"
            "中文使用规范简体字。"
            "只返回一个 JSON 对象，不要输出其他文字："
            f'{{"domain_context":"涉及……","terms":[...]}}。terms 最多包含 '
            f"{max_terms} 个互不重复的中文标准专业词，可以少于该数量。\n\n"
            f"任务问题：\n{question.strip()}\n\n"
            f"知识说明：\n{knowledge_text.strip() or '（空）'}"
        )
    if language == "en":
        return (
            "Generate the business-vocabulary prefix of an initial_prompt for English "
            "Whisper. Follow exactly three rules: "
            "(1) use the task question as the only filter; select dashboard or module "
            "names, field names, business definitions, entity identifiers, and "
            "specialist words directly stated by or tightly bound to the question; "
            "(2) use knowledge.md only to verify the standard spelling, English "
            "identifier, or common alias of those candidates; never expand the scope "
            "from knowledge.md or copy unrelated database fields; "
            f"(3) keep only correct standard terms, at most {max_terms} in total; "
            "there is no minimum, and precision is more important than quantity. "
            "Write domain_context as one concise English sentence containing only the "
            "objects, fields, definitions, and operations directly supported by the "
            "question. Do not answer the task or infer result rows. The domain_context "
            "and every term must be English or an ASCII identifier; output no Chinese. "
            "Return exactly one JSON object and no other text: "
            f'{{"domain_context":"...","terms":[...]}}. The terms array may contain '
            f"up to {max_terms} distinct standard professional terms.\n\n"
            f"Task question:\n{question.strip()}\n\n"
            f"Knowledge guide:\n{knowledge_text.strip() or '(empty)'}"
        )
    raise ValueError(f"Unsupported prompt language: {language!r}")


def generate_dynamic_terms(
    model: Any,
    *,
    question: str,
    knowledge_text: str,
    language: str,
    min_terms: int,
    max_terms: int,
) -> tuple[str, list[str], str, str]:
    request = build_term_request(
        question=question,
        knowledge_text=knowledge_text,
        language=language,
        min_terms=min_terms,
        max_terms=max_terms,
    )
    raw_response = _message_text(model.invoke(request))
    domain_context, terms = parse_prompt_response(
        raw_response,
        min_terms=min_terms,
        max_terms=max_terms,
        language=language,
    )
    assert domain_context is not None
    return domain_context, terms, raw_response, request


def build_initial_prompt(
    *,
    language: str | None,
    domain_context: str | None = None,
    dynamic_terms: list[str] | tuple[str, ...],
    include_ui_terms: bool,
    ui_terms: list[str] | tuple[str, ...] = DEFAULT_UI_TERMS,
) -> str | None:
    dynamic = [str(term).strip() for term in dynamic_terms if str(term).strip()]
    if language == "zh":
        parts = ["这是操作员在数据平台界面的讲解"]
        if domain_context:
            parts[0] += f"，{domain_context.strip('。')}"
        parts[0] += "。"
        if dynamic:
            parts.append(f"专业术语包括：{'、'.join(dynamic)}。")
        if include_ui_terms:
            parts.append(f"界面常见术语包括：{'、'.join(ui_terms)}。")
        return "".join(parts) if len(parts) > 1 else None
    if language == "en" and (domain_context or dynamic):
        parts = ["This is an operator explaining a data platform."]
        if domain_context:
            parts.append(f"{domain_context.strip('.')}.")
        if dynamic:
            parts.append(f"Professional terms: {', '.join(dynamic)}.")
        return " ".join(parts)
    return None


@dataclass(frozen=True, slots=True)
class AsrRuntimeSettings:
    model_name: str = "medium"
    detector_model_name: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    num_workers: int = 8
    language_threshold: float = 0.5
    prompt_min_terms: int = 1
    prompt_max_terms: int = 10
    ui_terms: tuple[str, ...] = DEFAULT_UI_TERMS


class AsrPipelineRuntime:
    def __init__(
        self,
        settings: AsrRuntimeSettings,
        *,
        model_factory: Callable[..., Any] | None = None,
        audio_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory or self._default_model_factory
        self._audio_loader = audio_loader or self._default_audio_loader
        self._models: dict[str, Any] = {}
        self._model_lock = threading.Lock()

    @staticmethod
    def _default_model_factory(**kwargs: Any) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is required for ASR.") from exc
        return WhisperModel(**kwargs)

    @staticmethod
    def _default_audio_loader(path: Path) -> Any:
        try:
            from faster_whisper.audio import decode_audio
        except ImportError as exc:
            raise RuntimeError("faster-whisper is required to decode ASR audio.") from exc
        return decode_audio(str(path), sampling_rate=16000)

    def _model(self, model_name: str) -> Any:
        model = self._models.get(model_name)
        if model is not None:
            return model
        with self._model_lock:
            model = self._models.get(model_name)
            if model is None:
                model = self._model_factory(
                    model_size_or_path=model_name,
                    device=self.settings.device,
                    compute_type=self.settings.compute_type,
                    cpu_threads=self.settings.cpu_threads,
                    num_workers=self.settings.num_workers,
                )
                self._models[model_name] = model
        return model

    def _detect_language(self, audio: Any) -> dict[str, Any]:
        started = perf_counter()
        try:
            detected, probability, probabilities = self._model(
                self.settings.detector_model_name
            ).detect_language(
                audio,
                language_detection_threshold=self.settings.language_threshold,
            )
            candidates = {
                str(language).strip().lower(): float(score)
                for language, score in probabilities
                if str(language).strip().lower() in {"zh", "en"}
            }
            if str(detected).strip().lower() in {"zh", "en"}:
                candidates.setdefault(str(detected).strip().lower(), float(probability))
            if not candidates:
                raise RuntimeError("tiny did not return a zh/en language candidate.")
            language, selected_probability = max(candidates.items(), key=lambda item: item[1])
            if selected_probability < self.settings.language_threshold:
                raise RuntimeError(
                    "tiny zh/en language confidence "
                    f"{selected_probability:.4f} is below "
                    f"{self.settings.language_threshold:.4f}."
                )
            return {
                "status": "ok",
                "language": language,
                "probability": selected_probability,
                "candidates": candidates,
                "elapsed_seconds": perf_counter() - started,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "language": None,
                "probability": None,
                "candidates": {},
                "elapsed_seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str,
        question: str = "",
        knowledge_text: str = "",
        prompt_model: Any | None = None,
        prompt_model_name: str | None = None,
    ) -> dict[str, Any]:
        effective_stage = validate_stage(stage)
        total_started = perf_counter()
        audio = self._audio_loader(audio_path) if stage_uses_tiny(effective_stage) else str(audio_path)

        language_detection = {
            "status": "not_requested",
            "language": None,
            "probability": None,
            "candidates": {},
            "elapsed_seconds": 0.0,
            "error": None,
        }
        routed_language: str | None = None
        if stage_uses_tiny(effective_stage):
            language_detection = self._detect_language(audio)
            if language_detection["status"] == "ok":
                routed_language = str(language_detection["language"])

        prompt_started = perf_counter()
        prompt_status = "not_requested"
        prompt_error: str | None = None
        raw_response: str | None = None
        request_text: str | None = None
        domain_context: str | None = None
        terms: list[str] = []
        if stage_uses_dynamic_prompt(effective_stage):
            if routed_language is None:
                prompt_status = "skipped_language_fallback"
            elif prompt_model is None:
                prompt_status = "failed"
                prompt_error = "Prompt model is unavailable."
            else:
                try:
                    request_text = build_term_request(
                        question=question,
                        knowledge_text=knowledge_text,
                        language=routed_language,
                        min_terms=self.settings.prompt_min_terms,
                        max_terms=self.settings.prompt_max_terms,
                    )
                    raw_response = _message_text(prompt_model.invoke(request_text))
                    domain_context, terms = parse_prompt_response(
                        raw_response,
                        min_terms=self.settings.prompt_min_terms,
                        max_terms=self.settings.prompt_max_terms,
                        language=routed_language,
                    )
                    prompt_status = "ok"
                except Exception as exc:  # noqa: BLE001
                    prompt_status = "failed"
                    prompt_error = f"{type(exc).__name__}: {exc}"

        initial_prompt = build_initial_prompt(
            language=routed_language,
            domain_context=domain_context,
            dynamic_terms=terms,
            include_ui_terms=stage_uses_ui_terms(effective_stage),
            ui_terms=self.settings.ui_terms,
        )
        prompt_elapsed = perf_counter() - prompt_started

        transcription_started = perf_counter()
        transcribe_kwargs: dict[str, Any] = {}
        if routed_language is not None:
            transcribe_kwargs["language"] = routed_language
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        segments_iter, info = self._model(self.settings.model_name).transcribe(
            audio,
            **transcribe_kwargs,
        )
        final_language = getattr(info, "language", None) or routed_language
        segments: list[dict[str, Any]] = []
        from data_agent_baseline.run.video_preprocessor import (
            repair_transcript_mojibake,
            simplify_chinese_transcript,
        )

        for segment in segments_iter:
            text = repair_transcript_mojibake(str(segment.text)).strip()
            text = simplify_chinese_transcript(text, language=final_language)
            if not text:
                continue
            segments.append(
                {
                    "start_sec": round(float(segment.start), 3),
                    "end_sec": round(float(segment.end), 3),
                    "text": text,
                }
            )
        transcription_elapsed = perf_counter() - transcription_started
        total_elapsed = perf_counter() - total_started
        return {
            "text": " ".join(item["text"] for item in segments).strip(),
            "segments": segments,
            "language": final_language,
            "language_probability": getattr(info, "language_probability", None),
            "model_duration_seconds": getattr(info, "duration", None),
            "stage": effective_stage,
            "language_detection": language_detection,
            "prompt": {
                "status": prompt_status,
                "template_version": PROMPT_TEMPLATE_VERSION,
                "model": prompt_model_name,
                "question_sha256": sha256_text(question),
                "knowledge_sha256": sha256_text(knowledge_text),
                "request_sha256": sha256_text(request_text or ""),
                "raw_response": raw_response,
                "domain_context": domain_context,
                "terms": terms,
                "initial_prompt": initial_prompt,
                "include_ui_terms": stage_uses_ui_terms(effective_stage)
                and routed_language == "zh",
                "elapsed_seconds": prompt_elapsed,
                "error": prompt_error,
            },
            "timings": {
                "language_detection_seconds": language_detection["elapsed_seconds"],
                "prompt_generation_seconds": prompt_elapsed,
                "transcription_seconds": transcription_elapsed,
                "total_seconds": total_elapsed,
            },
        }
