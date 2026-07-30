from __future__ import annotations

import os
import re
from dataclasses import dataclass

from kobushi_core.model import ModelMessage, OpenAIModelAdapter

try:
    from experiments.exp_155_phase_tool_visibility.prefix_cache import (
        with_prefix_cache_header,
    )
except Exception:  # pragma: no cover - lets the PoC run outside exp155.
    def with_prefix_cache_header(
        headers: dict[str, str], task_id: str | None = None
    ) -> dict[str, str]:
        del task_id
        return headers


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_FENCE_RE = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.S)


_EN_SYSTEM_PROMPT = """You are a COLUMN ADVISOR for a data-answering agent.

Return ONLY the minimal final CSV answer columns, separated by commas.
Do not solve the values. No explanation. No JSON. No markdown. One line only.

Rules:
- If the question asks which/list/name X, output X only; do not include the metric, filter, or sort field used to decide X.
- If the question asks show/retrieve/check X data/records/values, output X only.
- Do not include date/time/year/period columns unless explicitly requested.
- If the question asks X and Y, output X, Y.
- For count/how many questions, output count; add a group key only if grouped.
""".strip()


_ZH_SYSTEM_PROMPT = """你是数据问答系统的 COLUMN ADVISOR。

只输出最终 CSV 答案应该包含的最小列集合，用英文逗号分隔。
不要计算答案值。不要解释。不要 JSON。不要 markdown。只输出一行。

规则：
- 如果问题问“哪个/哪些/列出/名称 X”，只输出 X；不要输出用于判断 X 的指标、筛选或排序字段。
- 如果问题问“展示/查询/返回 X 的数据/记录/取值”，只输出 X。
- 没有明确要求时，不要加入日期、时间、年份、期间列。
- 如果问题明确问 X 和 Y，输出 X, Y。
- 数量/多少/count 题输出 count；只有分组统计时才加分组字段。
""".strip()


_EN_PROFILE_SUFFIX = """

You will also receive a compact DATA PROFILE. Use it to choose plausible column
semantics, but keep the same rules: only final answer columns, no implicit
date/time/year/period columns.
""".strip()


_ZH_PROFILE_SUFFIX = """

你还会收到一个简短的 DATA PROFILE。用它判断可能的列语义，但规则不变：
只输出最终答案列；没有明确要求时，不要加入日期、时间、年份、期间列。
""".strip()


@dataclass(frozen=True, slots=True)
class ColumnAdvisorResult:
    question: str
    columns: list[str]
    raw: str
    prompt_lang: str
    fallback_used: bool
    error: str | None = None

    @property
    def line(self) -> str:
        return ", ".join(self.columns)

    @property
    def column_count(self) -> int:
        return len(self.columns)

    def to_preamble_block(self) -> str:
        return (
            "# COLUMN ADVISOR HINT\n"
            "Question-only minimal final CSV columns. Column aliases are not the "
            "scoring target; match these value-vector semantics and do not add "
            "extra output columns.\n"
            f"columns: {self.line}\n"
            f"column_count: {self.column_count}\n\n"
        )


def _prefer_chinese_prompt(question: str) -> bool:
    cjk = len(_CJK_RE.findall(question))
    if cjk < 4:
        return False
    kana = len(_KANA_RE.findall(question))
    return kana == 0


def _make_model(task_id: str | None = None) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = (
        os.environ.get("COLUMN_ADVISOR_MODEL")
        or os.environ.get("AGENT_MODEL")
        or os.environ.get("MODEL_NAME")
        or "qwen3.5-35b-a3b"
    )
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        extra_headers=headers,
    )


def parse_column_line(raw: str) -> list[str]:
    """Parse the advisor's comma-only line, tolerating common wrapper mistakes."""
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    # Keep the first non-empty line. The prompt forbids explanations, but this
    # makes the PoC robust when a model still adds one.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            text = stripped
            break
    text = re.sub(r"^(columns?|列)\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().strip("[]").strip()
    parts = [part.strip().strip("\"'`") for part in re.split(r"[,，、]", text)]
    columns = [part for part in parts if part]
    return columns or ["answer"]


def _heuristic_columns(question: str) -> list[str]:
    """Cheap fallback used when the model is unavailable."""
    q = question.strip()
    q_lower = q.lower()
    zh = _prefer_chinese_prompt(q)
    haystack = q if zh else q_lower

    asks_count = bool(
        re.search(r"\b(how many|count|number of)\b|多少|几|数目|数量|共有|统计", haystack)
    )
    grouped = bool(re.search(r"\bby\b|per |group|分组|按.+展示|按.+统计", haystack))
    asks_time = bool(re.search(r"\bwhen|time|date\b|何时|什么时候|时间|日期", haystack))
    sorted_list = bool(
        re.search(r"ordered by|sorted by|sort|descending|ascending|由高到低|由低到高", haystack)
    )
    two_outputs = bool(
        re.search(r"\bname and (phone|telephone|email|code|amount|value)\b", q_lower)
        or re.search(r"(姓名|名称).{0,10}(电话|联系电话|代码|金额)", q)
        or re.search(r"(年度|年份).{0,10}(金额|总额)", q)
    )
    if asks_count and grouped:
        return ["group", "count"]
    if asks_count:
        return ["count"]
    if asks_time:
        return ["time"]
    if two_outputs:
        return ["first requested value", "second requested value"]
    if sorted_list:
        # The sort key should not be output unless separately requested.
        return ["listed item"]
    return ["answer"]


def advise_columns(
    question: str,
    *,
    task_id: str | None = None,
    model: OpenAIModelAdapter | None = None,
    use_model: bool = True,
) -> ColumnAdvisorResult:
    prompt_lang = "zh" if _prefer_chinese_prompt(question) else "en"
    if not use_model:
        cols = _heuristic_columns(question)
        return ColumnAdvisorResult(
            question=question,
            columns=cols,
            raw=", ".join(cols),
            prompt_lang=prompt_lang,
            fallback_used=True,
        )

    system = _ZH_SYSTEM_PROMPT if prompt_lang == "zh" else _EN_SYSTEM_PROMPT
    user = (
        f"{'用户问题' if prompt_lang == 'zh' else 'Question'}:\n"
        f"{question}\n\n"
        f"{'列' if prompt_lang == 'zh' else 'Columns'}:"
    )
    try:
        model = model or _make_model(task_id=task_id)
        raw = model.complete(
            [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=128,
        )
        return ColumnAdvisorResult(
            question=question,
            columns=parse_column_line(raw),
            raw=raw.strip(),
            prompt_lang=prompt_lang,
            fallback_used=False,
        )
    except Exception as exc:
        cols = _heuristic_columns(question)
        return ColumnAdvisorResult(
            question=question,
            columns=cols,
            raw=", ".join(cols),
            prompt_lang=prompt_lang,
            fallback_used=True,
            error=repr(exc),
        )


def advise_columns_with_profile(
    question: str,
    profile: str,
    *,
    task_id: str | None = None,
    model: OpenAIModelAdapter | None = None,
    use_model: bool = True,
) -> ColumnAdvisorResult:
    """Infer minimal output columns from question plus a compact context profile."""
    prompt_lang = "zh" if _prefer_chinese_prompt(question) else "en"
    if not use_model:
        cols = _heuristic_columns(question)
        return ColumnAdvisorResult(
            question=question,
            columns=cols,
            raw=", ".join(cols),
            prompt_lang=prompt_lang,
            fallback_used=True,
        )

    if prompt_lang == "zh":
        system = _ZH_SYSTEM_PROMPT + "\n\n" + _ZH_PROFILE_SUFFIX
        user = f"用户问题：\n{question}\n\nDATA PROFILE:\n{profile}\n\n列："
    else:
        system = _EN_SYSTEM_PROMPT + "\n\n" + _EN_PROFILE_SUFFIX
        user = f"Question:\n{question}\n\nDATA PROFILE:\n{profile}\n\nColumns:"
    try:
        model = model or _make_model(task_id=task_id)
        raw = model.complete(
            [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=128,
        )
        return ColumnAdvisorResult(
            question=question,
            columns=parse_column_line(raw),
            raw=raw.strip(),
            prompt_lang=prompt_lang,
            fallback_used=False,
        )
    except Exception as exc:
        cols = _heuristic_columns(question)
        return ColumnAdvisorResult(
            question=question,
            columns=cols,
            raw=", ".join(cols),
            prompt_lang=prompt_lang,
            fallback_used=True,
            error=repr(exc),
        )


def prompt_text(lang: str) -> str:
    """Expose the exact prompt for review/debug scripts."""
    normalized = lang.strip().lower()
    if normalized in {"zh", "cn", "chinese"}:
        return _ZH_SYSTEM_PROMPT
    return _EN_SYSTEM_PROMPT
