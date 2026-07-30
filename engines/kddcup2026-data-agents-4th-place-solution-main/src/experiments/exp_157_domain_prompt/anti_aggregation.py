"""Anti-aggregation classifier for exp_157.

The classifier is independent of the math advisor. It reads the user question
only and decides whether formula injection should be suppressed to preserve the
answer grain.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re

from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_157_domain_prompt.prefix_cache import with_prefix_cache_header


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")


@dataclass(frozen=True)
class AntiAggregationDecision:
    raw: str
    label: str
    prompt_lang: str

    @property
    def no_agg(self) -> bool:
        return self.label == "NO_AGG"


def _prefer_chinese_prompt(question: str) -> bool:
    cjk = len(_CJK_RE.findall(question))
    if cjk < 4:
        return False
    kana = len(_KANA_RE.findall(question))
    return kana == 0


def _make_model(task_id: str | None = None) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = os.environ.get("AGENT_MODEL") or os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")
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


_EN_SYS = """You classify whether the user question explicitly requires aggregation/calculation.
Do not solve the task.

Return exactly one line:
NO_AGG: <short reason>
or
AGG: <short reason>

Use NO_AGG when the question asks to find, check, show, display, list, or return
existing data, records, rows, values, examples, details, or a time series.
For NO_AGG, the answer should preserve source rows/values and should not use
SUM, AVG, COUNT, GROUP BY, or derived calculations.

Words inside field/entity names, such as total assets, total retail sales, GDP,
return rate, ratio, amount, or scale, do not by themselves request aggregation.

Use AGG only when the question explicitly asks for a calculation or statistic:
count, how many, how many distinct, average/mean, maximum/minimum, highest/lowest,
total/sum across records, ratio/percentage to compute, per-group, or for each group.

Mere "how much" is not enough for AGG if the question is asking to show existing values.
If unsure, return AGG."""


_ZH_SYS = """判断用户问题是否明确要求聚合/统计/重新计算。
不要解题。

只输出一行：
NO_AGG: <简短理由>
或
AGG: <简短理由>

以下情况输出 NO_AGG：
- 问题是在找/查/显示/列出/返回已有数据、已有记录、原始取值、明细、样例、时间序列。
- 这类问题应保留源表粒度，不要用 SUM、AVG、COUNT、GROUP BY 或重新计算。
- 字段名里的“总/总资产/国内生产总值/总额/比例/率/规模/金额”等词，不等于要求求和或重新计算。
- “各是多少”通常表示分别返回每个被问字段/项目的已有取值；除非同时出现合计、总和、平均、最大、最小、统计、分组等明确统计词，否则不要当作聚合。

以下情况输出 AGG：
- 问题明确要求统计或计算：计算、共有几年、多少个/多少人、distinct count、平均值、最大值、最小值、最高、最低、合计、总和、按组/分组/每类统计。
- 例如“最大值是多少”“平均值大于10”“总共有几年超过阈值”都属于 AGG。

单纯的“是多少/多少/金额大小”不足以说明要聚合；如果是在查已有数据，应输出 NO_AGG。
不确定时输出 AGG。"""


def _label_from_raw(raw: str) -> str:
    upper = raw.strip().upper()
    if upper.startswith("NO_AGG"):
        return "NO_AGG"
    if upper.startswith("AGG"):
        return "AGG"
    return "AGG"


def review_aggregation(question: str, task_id: str | None = None) -> AntiAggregationDecision:
    """Classify the question as NO_AGG or AGG."""
    lang = "zh" if _prefer_chinese_prompt(question) else "en"
    system = _ZH_SYS if lang == "zh" else _EN_SYS
    user = (
        f"{'用户问题' if lang == 'zh' else 'Question'}: {question}\n\n"
        f"{'判断' if lang == 'zh' else 'Classification'}:"
    )
    try:
        model = _make_model(task_id=task_id)
        raw = model.complete(
            [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=128,
        ).strip()
    except Exception as e:
        raw = f"AGG: classifier_error={e!r}"
    return AntiAggregationDecision(
        raw=raw,
        label=_label_from_raw(raw),
        prompt_lang=lang,
    )


def no_agg_note(prompt_lang: str) -> str:
    """Short guidance injected when the classifier says NO_AGG."""
    if prompt_lang == "zh":
        return (
            "# ANSWER SHAPE\n"
            "NO_AGG：保留源表行粒度、原始取值以及空值/空白行。"
            "除非用户明确要求统计或过滤，否则不要聚合或删除空值。\n\n"
        )
    return (
        "# ANSWER SHAPE\n"
        "NO_AGG: preserve source rows/values, including blank/NULL rows. "
        "Do not aggregate or drop NULLs unless explicitly asked.\n\n"
    )
