"""LLM-as-judge vote: groups N attempts by signature, then asks a judge LLM
to pick the best candidate when groups disagree.

Pipeline:
  1. group_by_signature(attempts)          → {sig: [attempt_idx, ...]}
  2. summarize_attempt(answer, trace_path) → {sql, columns, rows_preview, n_rows}
  3. judge_attempts(question, summaries)   → picked index (0-based)

Falls back to rule-based adaptive_vote majority when:
  - Only one signature group exists (trivial agreement)
  - Judge call fails / model unavailable

The judge sees per-candidate: (a) the candidate's answer table (cols + first 5
rows), (b) the last `answer_from_sql` SQL string extracted from its trace.log,
and (c) the support count (= how many of the 3 attempts produced this sig).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_141_llm_judge_vote.adaptive_vote import _signature

MAX_ROWS_PREVIEW = 5
MAX_COL_VALUE_LEN = 40


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """One per unique signature group."""
    indices: tuple[int, ...]   # = original attempt indices that share this sig
    answer: AnswerTable        # = representative answer (first attempt in group)
    last_sql: str | None       # = SQL string from the trace, if extractable
    n_cols: int
    n_rows: int
    reasoning: dict[str, str]  # = phase-keyed thoughts: plan/explore/answer/verify


_SQL_RE_FROM_TRACE = re.compile(
    r"action=(?:answer_from_sql|execute_sql)[^\n]*?\bsql['\"]\s*[:=]\s*['\"]([^'\"]+)",
    re.IGNORECASE,
)
# Looser fallback: lines containing "sql:" or "'sql':"
_SQL_RE_LOOSE = re.compile(r"['\"]?sql['\"]?\s*[:=]\s*['\"]?(SELECT [^'\"\n]{20,800})", re.IGNORECASE)

# Per-step thought regex: capture phase + action + thought up to end of line.
_STEP_THOUGHT_RE = re.compile(
    r"phase=(?P<phase>[a-z]+) step (?P<step>\d+) action=(?P<action>[a-z_]+) thought=(?P<thought>[^\n]+)",
    re.IGNORECASE,
)

MAX_THOUGHT_CHARS = 400  # = cap per phase to keep judge prompt bounded


def _read_trace(trace_log_path: str | Path | None) -> str | None:
    if not trace_log_path:
        return None
    p = Path(trace_log_path)
    if not p.exists():
        return None
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return None


def extract_last_sql(trace_log_path: str | Path | None) -> str | None:
    """Read trace.log and return the last `answer_from_sql` SQL string."""
    text = _read_trace(trace_log_path)
    if text is None:
        return None
    matches = _SQL_RE_FROM_TRACE.findall(text)
    if matches:
        return matches[-1][:1500]
    matches = _SQL_RE_LOOSE.findall(text)
    if matches:
        return matches[-1][:1500]
    return None


def extract_reasoning(trace_log_path: str | Path | None) -> dict[str, str]:
    """Extract per-phase representative thought from trace.log.

    Returns dict keyed by phase: plan/explore/answer/verify. Each value is
    a short summary (= truncated). Heuristic per phase:
      - plan:    FIRST plan thought (= initial interpretation)
      - explore: LAST execute_sql/complete_phase thought in explore (= final discovery)
      - answer:  LAST answer_from_sql thought (= why this SQL)
      - verify:  LAST confirm_answer thought (= cross-check reasoning)
    """
    text = _read_trace(trace_log_path)
    if text is None:
        return {}
    # Group steps per phase
    by_phase: dict[str, list[tuple[str, str]]] = {}
    for m in _STEP_THOUGHT_RE.finditer(text):
        phase = m.group("phase").lower()
        action = m.group("action").lower()
        thought = m.group("thought").strip()
        by_phase.setdefault(phase, []).append((action, thought))

    out: dict[str, str] = {}
    if "plan" in by_phase and by_phase["plan"]:
        out["plan"] = by_phase["plan"][0][1][:MAX_THOUGHT_CHARS]
    if "explore" in by_phase:
        # Prefer the LAST complete_phase or terminal explore thought
        explore_steps = by_phase["explore"]
        # Take the very last explore thought (= summarizes findings before answer)
        if explore_steps:
            out["explore"] = explore_steps[-1][1][:MAX_THOUGHT_CHARS]
    if "answer" in by_phase:
        # Last answer_from_sql thought (= the final SQL justification)
        ans = [(a, t) for a, t in by_phase["answer"] if a == "answer_from_sql"]
        if not ans:
            ans = by_phase["answer"]
        if ans:
            out["answer"] = ans[-1][1][:MAX_THOUGHT_CHARS]
    if "verify" in by_phase:
        ver = [(a, t) for a, t in by_phase["verify"] if a == "confirm_answer"]
        if not ver:
            ver = by_phase["verify"]
        if ver:
            out["verify"] = ver[-1][1][:MAX_THOUGHT_CHARS]
    return out


def summarize_attempt(answer: AnswerTable, trace_log_path: str | Path | None,
                      indices: tuple[int, ...]) -> CandidateSummary:
    last_sql = extract_last_sql(trace_log_path)
    reasoning = extract_reasoning(trace_log_path)
    return CandidateSummary(
        indices=indices,
        answer=answer,
        last_sql=last_sql,
        n_cols=len(answer.columns) if answer else 0,
        n_rows=len(answer.rows) if (answer and answer.rows) else 0,
        reasoning=reasoning,
    )


def extract_reasoning_from_steps(steps: list[dict]) -> dict[str, str]:
    """Variant of extract_reasoning that reads structured `steps` list (= from
    trace.json) instead of a trace.log file. Each step is expected to have keys
    `phase`, `action`, `thought`."""
    if not steps:
        return {}
    by_phase: dict[str, list[tuple[str, str]]] = {}
    for st in steps:
        phase = (st.get("phase") or "").lower()
        action = (st.get("action") or "").lower()
        thought = (st.get("thought") or "").strip()
        if not thought or not phase:
            continue
        by_phase.setdefault(phase, []).append((action, thought))
    out: dict[str, str] = {}
    if "plan" in by_phase and by_phase["plan"]:
        out["plan"] = by_phase["plan"][0][1][:MAX_THOUGHT_CHARS]
    if "explore" in by_phase and by_phase["explore"]:
        out["explore"] = by_phase["explore"][-1][1][:MAX_THOUGHT_CHARS]
    if "answer" in by_phase:
        ans = [(a, t) for a, t in by_phase["answer"] if a == "answer_from_sql"]
        if not ans:
            ans = by_phase["answer"]
        if ans:
            out["answer"] = ans[-1][1][:MAX_THOUGHT_CHARS]
    if "verify" in by_phase:
        ver = [(a, t) for a, t in by_phase["verify"] if a == "confirm_answer"]
        if not ver:
            ver = by_phase["verify"]
        if ver:
            out["verify"] = ver[-1][1][:MAX_THOUGHT_CHARS]
    return out


def extract_last_sql_from_steps(steps: list[dict]) -> str | None:
    """Find the last step with action=answer_from_sql and pull its `sql` field."""
    if not steps:
        return None
    last_sql: str | None = None
    for st in steps:
        action = (st.get("action") or "").lower()
        if action == "answer_from_sql":
            ai = st.get("action_input") or {}
            sql = ai.get("sql") if isinstance(ai, dict) else None
            if isinstance(sql, str) and sql.strip():
                last_sql = sql.strip()
    return last_sql[:1500] if last_sql else None


def summarize_from_steps(answer: AnswerTable, steps: list[dict],
                         indices: tuple[int, ...]) -> CandidateSummary:
    return CandidateSummary(
        indices=indices,
        answer=answer,
        last_sql=extract_last_sql_from_steps(steps),
        n_cols=len(answer.columns) if answer else 0,
        n_rows=len(answer.rows) if (answer and answer.rows) else 0,
        reasoning=extract_reasoning_from_steps(steps),
    )


def _trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_table_preview(answer: AnswerTable) -> str:
    if not answer or not answer.columns:
        return "(empty)"
    cols = list(answer.columns)
    header = " | ".join(_trunc(str(c), MAX_COL_VALUE_LEN) for c in cols)
    sep = " | ".join("-" * min(MAX_COL_VALUE_LEN, max(3, len(_trunc(str(c), MAX_COL_VALUE_LEN)))) for c in cols)
    rows = answer.rows or []
    lines = [header, sep]
    for r in rows[:MAX_ROWS_PREVIEW]:
        lines.append(" | ".join(_trunc(str(v), MAX_COL_VALUE_LEN) for v in r))
    if len(rows) > MAX_ROWS_PREVIEW:
        lines.append(f"... ({len(rows) - MAX_ROWS_PREVIEW} more rows)")
    return "\n".join(lines)


def build_judge_prompt(question: str, summaries: list[CandidateSummary]) -> str:
    parts = [
        f"Question:\n{question}",
        "",
        f"Candidates ({len(summaries)}):",
    ]
    for i, c in enumerate(summaries):
        letter = chr(ord("A") + i)
        parts.append(f"\n[{letter}] support={len(c.indices)}, cols={c.n_cols}, rows={c.n_rows}")
        r = c.reasoning or {}
        if r.get("plan"):
            parts.append(f"  plan:    {_trunc(r['plan'], 240)}")
        if r.get("explore"):
            parts.append(f"  explore: {_trunc(r['explore'], 240)}")
        if r.get("answer"):
            parts.append(f"  answer:  {_trunc(r['answer'], 240)}")
        if r.get("verify"):
            parts.append(f"  verify:  {_trunc(r['verify'], 240)}")
        if c.last_sql:
            parts.append(f"  sql: {_trunc(c.last_sql, 400)}")
        parts.append("  table:")
        for line in _render_table_preview(c.answer).split("\n"):
            parts.append(f"    {line}")
    parts.extend([
        "",
        "Which candidate's answer is most accurate for the question?",
        "Reply:",
        "PICK: <letter>",
        "REASON: <one sentence>",
    ])
    return "\n".join(parts)


_PICK_RE = re.compile(r"PICK\s*:\s*([A-Z])", re.IGNORECASE)


def parse_judge_response(text: str, n_candidates: int) -> int | None:
    if not text:
        return None
    m = _PICK_RE.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    idx = ord(letter) - ord("A")
    if 0 <= idx < n_candidates:
        return idx
    return None


def group_by_signature(answers: list[AnswerTable]) -> list[CandidateSummary]:
    """Group answers by signature, preserving original index ordering."""
    sig_to_indices: dict[tuple, list[int]] = {}
    sig_to_rep: dict[tuple, AnswerTable] = {}
    for i, a in enumerate(answers):
        if not a or not a.rows:
            continue
        sig = _signature(a)
        sig_to_indices.setdefault(sig, []).append(i)
        sig_to_rep.setdefault(sig, a)
    # Order by support desc (= largest group first), then by first-seen index
    ordered_sigs = sorted(
        sig_to_indices.keys(),
        key=lambda s: (-len(sig_to_indices[s]), sig_to_indices[s][0]),
    )
    return [
        CandidateSummary(
            indices=tuple(sig_to_indices[s]),
            answer=sig_to_rep[s],
            last_sql=None,
            n_cols=len(sig_to_rep[s].columns),
            n_rows=len(sig_to_rep[s].rows or []),
            reasoning={},
        )
        for s in ordered_sigs
    ]


def judge_vote(
    answers: list[AnswerTable],
    question: str,
    model: ModelAdapter,
    trace_log_paths: list[str | Path | None] | None = None,
    steps_per_attempt: list[list[dict]] | None = None,
) -> tuple[AnswerTable | None, dict]:
    """Group → if 1 group skip, else summarize + LLM-judge.

    Returns (chosen_answer, judge_meta). `judge_meta` contains:
      - n_candidates: int (= unique sig groups)
      - chosen_letter: str (= "A"/"B"/...) or None
      - judge_text: full judge reply or None
      - fell_back: bool (= True if judge failed; first non-empty answer used)
    """
    if not answers:
        return None, {"n_candidates": 0, "fell_back": True}
    non_empty = [(i, a) for i, a in enumerate(answers) if a and a.rows]
    if not non_empty:
        return answers[0] if answers else None, {"n_candidates": 0, "fell_back": True}

    # Group by signature
    summaries = group_by_signature([a for _, a in non_empty])
    abs_indices = [i for i, _ in non_empty]
    enriched: list[CandidateSummary] = []
    for s in summaries:
        first_rel = s.indices[0]
        abs_idx = abs_indices[first_rel]
        abs_idx_tuple = tuple(abs_indices[r] for r in s.indices)
        # Prefer structured steps when provided (= cleaner extraction).
        if steps_per_attempt and abs_idx < len(steps_per_attempt) and steps_per_attempt[abs_idx]:
            enriched.append(summarize_from_steps(s.answer, steps_per_attempt[abs_idx], abs_idx_tuple))
        else:
            trace_path = trace_log_paths[abs_idx] if trace_log_paths and abs_idx < len(trace_log_paths) else None
            enriched.append(summarize_attempt(s.answer, trace_path, abs_idx_tuple))

    if len(enriched) == 1:
        return enriched[0].answer, {"n_candidates": 1, "fell_back": False, "skipped_judge": True}

    # Build judge prompt and call LLM
    prompt = build_judge_prompt(question, enriched)
    try:
        # Disable thinking + T=0 for deterministic fast judgment. Thinking
        # mode is 30× slower and produced the same final mean in earlier
        # POC (= 0.8229 both ways) — pick is selection, not creation.
        reply = model.complete(
            messages=[
                ModelMessage(role="system", content="You are a precise SQL judge."),
                ModelMessage(role="user", content=prompt),
            ],
            enable_thinking=False,
            max_tokens=2048,
        )
    except Exception as exc:
        # Fall back to largest group
        return enriched[0].answer, {
            "n_candidates": len(enriched),
            "fell_back": True,
            "judge_error": f"{type(exc).__name__}: {str(exc)[:120]}",
        }
    pick_idx = parse_judge_response(reply, len(enriched))
    if pick_idx is None:
        return enriched[0].answer, {
            "n_candidates": len(enriched),
            "fell_back": True,
            "judge_text": (reply or "")[:300],
            "parse_failed": True,
        }
    chosen = enriched[pick_idx]
    return chosen.answer, {
        "n_candidates": len(enriched),
        "fell_back": False,
        "chosen_letter": chr(ord("A") + pick_idx),
        "chosen_support": len(chosen.indices),
        "judge_text": (reply or "")[:300],
    }
