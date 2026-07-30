"""Qwen sub-agent PoC for indexing adversarial prose docs.

The goal is not to answer the benchmark question directly. The sub-agent turns
raw extracted PDF/MD text into a compact, evidence-linked index:
sections + per-record fields + line numbers. The main agent can then reason
over this index and spot-check evidence lines with read_doc.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelMessage, OpenAIModelAdapter

from experiments.exp_173_groupby_dim.prefix_cache import with_prefix_cache_header
from experiments.exp_173_groupby_dim.tools.filesystem import resolve_context_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def index_cache_root() -> Path:
    root = Path(os.environ.get("EXP173_PROSE_INDEX_ROOT", "artifacts/prose_doc_index"))
    if not root.is_absolute():
        root = repo_root() / root
    return root


def index_cache_path(task: PublicTask, relative_path: str) -> Path:
    rel = relative_path.replace("/", "__")
    return index_cache_root() / task.task_id / f"{rel}.json"


def make_model(temp: float = 0.0, *, task_id: str | None = None) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        max_tokens=32768,
        enable_thinking=False,
        extra_headers=headers,
    )


def _load_doc_text(task: PublicTask, relative_path: str) -> str:
    path = resolve_context_path(task, relative_path)
    if path.suffix.lower() == ".pdf":
        try:
            from experiments.exp_173_groupby_dim.pdf_text_cache import read_cached_pdf_text
            cached = read_cached_pdf_text(task, path)
            if cached is not None:
                return cached
        except Exception:
            pass
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _numbered_text(text: str, *, max_lines: int | None = None) -> str:
    lines = text.splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start=1))


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : i + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("JSON root is not an object")
                return parsed
    raise ValueError("could not find complete JSON object")


SYSTEM_PROMPT = """You are a document indexing sub-agent for data-analysis tasks.

Your job is NOT to answer the task directly. Convert raw line-numbered prose into
a compact JSON index with line evidence.

Rules:
- Output ONLY valid JSON. No markdown, no commentary.
- Do not infer values without explicit line evidence.
- Use null when a field is not found.
- Every non-null record field must have evidence line numbers.
- Prefer final/corrected values over earlier wrong drafts.
- Separate sections by purpose: identity/name/code, metric/profit, asset, per_share, date/period, other.
- Use the stable record id shown in text, usually a phrase like 战略单元 3865.
- For Chinese fund docs, distinguish:
  - 证券代码 / 证券交易代码 / 最终注册为 = requested fund code
  - 内部代码 / 内部编号 / 内部识别码 = internal id, not SecuCode
  - 交易简称 / 证券简称 / 市场简称 = displayed fund abbreviation
  - 交易代码 can be a name-like ticker label; do not use it as numeric SecuCode unless clearly numeric and described as 证券代码.
"""


def build_prompt(task: PublicTask, relative_path: str, numbered_doc: str) -> str:
    return f"""Task question:
{task.question}

Document path: {relative_path}

Return this JSON shape:
{{
  "doc_path": "{relative_path}",
  "join_key": "战略单元",
  "sections": [
    {{
      "name": "identity|profit|asset|per_share|date|other",
      "purpose": "short description",
      "start_line": 1,
      "end_line": 10
    }}
  ],
  "records": [
    {{
      "record_id": "3865",
      "display_name": "国投瑞盛",
      "secu_code": "161232",
      "total_profit": 79896578.43,
      "end_date": "2021-12-31",
      "include_for_question": true,
      "evidence": {{
        "identity_lines": [145, 146, 147],
        "code_lines": [146, 147],
        "profit_lines": [362, 363],
        "date_lines": [764, 765]
      }},
      "notes": "brief note on corrections/ambiguities"
    }}
  ],
  "top_records_for_question": [
    {{
      "rank": 1,
      "record_id": "2223",
      "display_name": "中欧成长A",
      "secu_code": "166006",
      "total_profit": 1656000000.0,
      "end_date": "2021-06-30",
      "evidence_lines": [87, 89, 335, 749]
    }}
  ],
  "warnings": []
}}

Index records relevant to the question. If the question asks for a top-10/ranking,
populate top_records_for_question with the ranked records after applying the
period/date criterion. Include enough non-top records in records to justify the
ranking boundary when visible.

Raw line-numbered document:
{numbered_doc}
"""


def build_doc_index(
    task: PublicTask,
    relative_path: str,
    *,
    model: OpenAIModelAdapter | None = None,
    force: bool = False,
    max_lines: int | None = None,
) -> dict[str, Any]:
    out_path = index_cache_path(task, relative_path)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    text = _load_doc_text(task, relative_path)
    numbered = _numbered_text(text, max_lines=max_lines)
    model = model or make_model(task_id=task.task_id)
    raw = model.complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=build_prompt(task, relative_path, numbered)),
        ],
        enable_thinking=False,
        max_tokens=32768,
    )
    parsed = _extract_json_object(raw)
    parsed.setdefault("doc_path", relative_path)
    parsed["_meta"] = {
        "task_id": task.task_id,
        "doc_path": relative_path,
        "source_total_lines": len(text.splitlines()),
        "indexed_max_lines": max_lines,
        "raw_model_chars": len(raw),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed
