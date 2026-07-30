from __future__ import annotations

import json
import re
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


KNOWLEDGE_MAX_CHARS = 5000
_KNOWLEDGE_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_KNOWLEDGE_SECTION_RE = re.compile(r"^(#{2,3})\s", re.MULTILINE)
DOC_MAX_CHARS = 5000
DOC_PER_FILE_HEADER_OVERHEAD = 32   # rough budget for "### filename\n\n"


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. `action_input` MUST be a JSON object (dict), never a bare string. For execute_python use {"code": "..."}; for read_csv/read_json/read_doc/dataframe_describe/dataframe_head use {"path": "..."}.
9. The `answer` tool may soft-reject your first submission with `status: validation_blocking` and a list of warnings. Read each warning, fix the answer (e.g. include all enumerated columns, drop dup rows, populate empty columns), then re-call `answer`. If you are confident the warnings are wrong, re-call with `confirm: true` to bypass.

Answer normalization (apply when constructing the final `answer` table):
- Numbers: format with 2 decimal places (e.g. 4200000 → "4200000.00", 3.14159 → "3.14"). Pure integer-only ID columns may stay as-is.
- Dates: ISO 8601 YYYY-MM-DD (e.g. "2024-3-1" → "2024-03-01").
- Nulls, NaN, "none", "null" (any case): emit as empty string "".
- Strings: trim leading/trailing whitespace and \\r\\n; preserve case exactly.

Scoring policy:
- The grader matches each predicted column against gold by value-multiset signature (after normalization). Column names and row order are ignored.
- Recall is rewarded; extra predicted columns are only weakly penalized.
- When in doubt about including a column, INCLUDE it rather than omit.

Approach:
- On your first step, write a brief plan in `thought` (e.g. "Plan: 1) list_context, 2) inspect schema, 3) filter target rows, 4) compose answer").
- **In that first plan, explicitly enumerate the EXACT columns the gold answer requires.** Read the question phrase by phrase: nouns and noun-phrases that name the requested attributes are the columns; everything else (join keys, foreign IDs, timestamps used only for filtering, intermediate aggregates) MUST stay out of the answer. Adding auxiliary columns is the dominant zero-score failure mode — prefer omitting a borderline column over including it.
- When the question uses set semantics ("list", "which", "who are the", "name the"), apply DISTINCT / dedup before composing the answer table — multiple rows of the same entity is over-emission and zeroes the score.
- When the question is singular ("what is the …", "how many …", "the largest …"), the answer is a single value and the table should be 1×1 — never widen with bookkeeping columns.
- **Empty result is a valid answer.** If your query returns 0 rows, the answer may genuinely be empty — `rows: []` is committable. Commit empty with `confirm: true` when (a) the filter column's min/max overlaps the question's requested range (range is genuinely inside the data → 0 rows means no matches exist), OR (b) you have already tried ONE alternate table that could plausibly host the question's entity and it ALSO returned 0 rows. Do NOT loop trying further alt tables, do NOT widen the filter beyond the original question's intent. If the filter range is OUTSIDE the column's actual data range (e.g. question asks "June 2013" but the column min/max is "2012-08-01..2012-08-31"), the table is wrong — first verify there isn't a separate file (CSV / DB / JSON) that hosts the requested range BEFORE committing empty. Critical: before concluding "range is outside", inspect the FULL range of the column (e.g. via dataframe_describe or `SELECT MIN, MAX FROM table`), not just the first few rows — sample-based range estimates are unreliable on large CSVs.
- **Multi-source time-series lookup.** When the question asks about a specific date / month / year and the obvious table's range is OUTSIDE that, scan the ENTIRE context directory (csv/, json/, db/) for any file whose name or schema suggests a date-based aggregate (yearmonth, monthly, daily, summary, etc.). Some BIRD datasets split transactional detail and time-aggregated views into separate files — querying just the transactional file misses the answer.
- **Mirror the source schema for separate attributes.** When the question asks about attributes that exist as separate columns in the source (e.g. `home_team_goal` and `away_team_goal`, `first_name` and `last_name`, `min_value` and `max_value`), keep them as separate columns in your answer. Do NOT synthesize combined strings like `"1-1"` or `"John Doe"` when the source has separate fields — use the source's column names verbatim.
- **Pre-answer self-verification.** Before calling `answer`, write one short check in your `thought` confirming: (a) row count matches the question's expected cardinality (singular question → 1×1; plural question → all matching rows distinct), (b) each column maps to a phrase in the question, (c) values come from the source rows you just inspected (not paraphrased or invented).
- **Aggregation granularity (per-entity vs total).** Before computing an average, sum, or percentage, identify the GROUPING entity from the question wording. The phrase "of <entity>" or "per <entity>" makes that entity the grouping key — compute the within-entity value first, then aggregate across entities. Concrete:
  * "average monthly consumption **of customers**" → (total / N_customers) / 12 — NOT total / 12.
  * "average price **per product**" → revenue / N_distinct_products — NOT total revenue.
  * "average number of bonds the atoms have" — BIRD-style gold typically uses the raw `COUNT(detail_table.id) / COUNT(source_table.id)` over the JOINed rows (no DISTINCT). When the join expands the parent rows, both COUNTs grow together; the average is naturally 1.0 if every parent appears once per child. Match BIRD's convention rather than your own "dedupe parents first" intuition.
- **Percentage formulas (BIRD convention).** When the question asks "what percentage of X has property Y", gold almost always uses `CAST(COUNT(CASE WHEN Y THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(*)` over the SAME filtered population — NOT a separately-filtered numerator and denominator. Apply the property filter inside the CASE, leave the outer COUNT unfiltered (except for the population scope).
- **Column whitelist rule (BIRD-style scoring).** The grader compares predicted columns to gold by value-multiset. Question phrases like "which X has Y", "what is the X of Z", "list the Xs", "give their X" want ONLY the X column. The Y / Z / filter values are inputs, not outputs. Concrete patterns and the columns gold expects:
  * "Which event has the lowest cost?" → `event_name` only. NOT `event_name, cost`.
  * "List all the withdrawals that client 3356 makes" → only the transaction identifier column (e.g. `trans_id`). NOT the full transaction row.
  * "Give their consumption status in August 2012" → only `consumption`. NOT `customer_id, consumption`.
  * "What is the comment with the highest score?" → only the comment `text`. NOT `id, score, post_id, text, ...`.
  Before calling `answer`, re-read your `columns` list against the question. If a column was added because it "looks helpful" but the question never explicitly asked for it, REMOVE it. One natural-feeling extra column commonly turns a perfect score into zero.
  **Exception — multi-attribute questions.** When the question explicitly enumerates two or more attributes joined by "and" / "with" / "as well as" / "along with" (e.g. "List the **names and funding types** of schools", "Give me the **first and last name** of each member", "Show the **id, name, and price** of products"), keep ALL named attributes as separate columns — do NOT collapse to one. Drop "and" / "as well as" from candidate column names; keep the actual attribute names.
- For hard / extreme tier with context > 50 MB total, default to `dataframe_head` with explicit small `max_rows` (e.g. 50) over full `read_csv` — full reads of large CSVs blow up the prompt and can timeout the model call.
- On later steps, prefix `thought` with "Plan step N: ..." referencing the plan.
- For large tabular files, prefer dataframe_describe / dataframe_head over read_csv to avoid context-window blowup.
- The execute_python kernel is persistent within a task: a variable defined in one call survives the next, unless a timeout resets it.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response on the first step (plan-then-execute):
```json
{"thought":"Plan: 1) list_context, 2) inspect csv schema, 3) filter target rows, 4) compose answer","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you inspect a tabular file:
```json
{"thought":"Plan step 2: inspect schema of the main csv","action":"dataframe_describe","action_input":{"path":"csv/data.csv"}}
```

Example response when you run Python (note the `code` key — never put the code as a bare string):
```json
{"thought":"Plan step 3: load both json files and compute the join","action":"execute_python","action_input":{"code":"import json\nwith open('json/member.json') as f: m=json.load(f)\nprint(len(m))"}}
```

Example response when you have the final answer (normalized to 2dp):
```json
{"thought":"Plan step 4: compose answer with normalized values","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.50"]]}}
```

Example response for a genuinely-empty answer (filter matches no rows, range overlaps):
```json
{"thought":"Plan step 5: filter returns 0 rows and the column range overlaps the requested range — commit empty.","action":"answer","action_input":{"columns":["trans_id"],"rows":[],"confirm":true}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def _split_knowledge_sections(text: str) -> list[tuple[str, str]]:
    """Return ``[(header_line, body), ...]`` chunks split at H2/H3 boundaries.

    Anything before the first H2/H3 (preamble) is returned with an empty
    header_line. Used by the question-keyword reorder logic; preserves
    section text verbatim so re-joining is lossless.
    """
    chunks: list[tuple[str, str]] = []
    matches = list(_KNOWLEDGE_SECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    if matches[0].start() > 0:
        chunks.append(("", text[: matches[0].start()].rstrip()))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].rstrip()
        if section:
            line_end = section.find("\n")
            header = section if line_end < 0 else section[:line_end]
            chunks.append((header, section))
    return chunks


def _question_keywords(question: str) -> set[str]:
    if not question:
        return set()
    return {tok.lower() for tok in _KNOWLEDGE_KEYWORD_TOKEN_RE.findall(question)}


def _load_knowledge_md(
    context_dir: Path,
    max_chars: int = KNOWLEDGE_MAX_CHARS,
    *,
    question: str = "",
) -> str | None:
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.is_file():
        return None
    try:
        text = knowledge_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None

    # Fast path: text already fits the cap, no reorder needed.
    if len(text) <= max_chars:
        return text

    keywords = _question_keywords(question)
    if not keywords:
        return text[:max_chars].rstrip() + "\n... [knowledge.md truncated]"

    # Reorder: keep preamble in place, then put question-matching sections
    # before non-matching ones. Sections that fit within the cap are
    # included whole; the first section that overflows is truncated and
    # signals end-of-list.
    sections = _split_knowledge_sections(text)
    if not sections:
        return text[:max_chars].rstrip() + "\n... [knowledge.md truncated]"

    preamble = sections[0][1] if sections and sections[0][0] == "" else ""
    body_sections = sections[1:] if preamble else sections

    def _score(section: tuple[str, str]) -> int:
        body = section[1].lower()
        return sum(1 for kw in keywords if kw in body)

    matching = [s for s in body_sections if _score(s) > 0]
    rest = [s for s in body_sections if _score(s) == 0]
    ordered = matching + rest

    parts: list[str] = []
    remaining = max_chars
    truncated_any = False

    def _add_chunk(chunk: str) -> None:
        nonlocal remaining, truncated_any
        if remaining <= 0:
            truncated_any = True
            return
        if len(chunk) <= remaining:
            parts.append(chunk)
            remaining -= len(chunk) + 2  # blank-line separator
        else:
            parts.append(chunk[:remaining].rstrip())
            remaining = 0
            truncated_any = True

    if preamble:
        pre = preamble.strip()
        if pre:
            _add_chunk(pre)
    for _, section_text in ordered:
        _add_chunk(section_text)

    if not parts:
        return text[:max_chars].rstrip() + "\n... [knowledge.md truncated]"
    joined = "\n\n".join(parts)
    if truncated_any or len(joined) < len(text):
        joined = joined + "\n... [knowledge.md truncated]"
    return joined


def _load_doc_md(context_dir: Path, max_chars: int = DOC_MAX_CHARS) -> str | None:
    """Concatenate ``context/doc/*.md`` excerpts up to ``max_chars`` total.

    v2 holdout showed two of the worst-scoring tasks (both hard tier) had
    ``doc/`` supplementary documents that the agent never read because no
    tool surfaced them by default. Auto-injecting the prefix of every doc
    closes that gap — knowledge.md is universal and doc/ is the next-most-
    common signal carrier (12/50 in public set, ~100% of hard with mixed
    structured+unstructured sources).

    Files are sorted alphabetically; each gets a ``### filename`` header
    and as much body as the remaining budget allows. Returns ``None`` when
    the doc directory is missing, empty, or unreadable.
    """
    doc_dir = context_dir / "doc"
    if not doc_dir.is_dir():
        return None
    md_files = sorted(p for p in doc_dir.glob("*.md") if p.is_file())
    if not md_files:
        return None

    chunks: list[str] = []
    remaining = max_chars
    for path in md_files:
        if remaining <= DOC_PER_FILE_HEADER_OVERHEAD + 64:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        header = f"### {path.name}\n"
        body_budget = max(0, remaining - len(header))
        if body_budget <= 0:
            break
        if len(text) > body_budget:
            text = text[:body_budget].rstrip() + "\n... [truncated]"
        chunk = header + text
        chunks.append(chunk)
        remaining -= len(chunk) + 2  # account for "\n\n" separator
    if not chunks:
        return None
    return "\n\n".join(chunks)


def build_task_prompt(
    task: PublicTask,
    *,
    policy_hints: tuple[str, ...] | list[str] | None = None,
    preferred_tools: tuple[str, ...] | list[str] | None = None,
    avoid_tools: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Build the per-task user prompt.

    ``policy_hints`` / ``preferred_tools`` / ``avoid_tools`` come from the
    memory layer (``memory.policies.resolve_policy``) and let the runner
    inject task-shape-specific guidance without changing the system prompt.
    Empty / None means "no policy hints" — the prompt is identical to v6.
    """
    parts: list[str] = []
    knowledge = _load_knowledge_md(task.context_dir, question=task.question)
    if knowledge:
        parts.append("Domain knowledge from context/knowledge.md:\n" + knowledge)
    doc_text = _load_doc_md(task.context_dir)
    if doc_text:
        parts.append("Supplementary documents from context/doc/:\n" + doc_text)

    # Inject memory-layer hints right after domain knowledge so the agent
    # treats them as additional context, not as a hard constraint that
    # could override the question itself.
    hint_lines: list[str] = []
    if policy_hints:
        for hint in policy_hints:
            if hint:
                hint_lines.append(f"- {hint}")
    if preferred_tools:
        prefs = ", ".join(t for t in preferred_tools if t)
        if prefs:
            hint_lines.append(f"- Prefer these tools when applicable: {prefs}.")
    if avoid_tools:
        avoid = ", ".join(t for t in avoid_tools if t)
        if avoid:
            hint_lines.append(f"- Avoid these tools (they will not work well on this task): {avoid}.")
    if hint_lines:
        parts.append(
            "Task-shape advisories (derived from prior public-set observations; treat as hints, not constraints):\n"
            + "\n".join(hint_lines)
        )

    parts.append(f"Difficulty: {task.difficulty}")
    parts.append(f"Question: {task.question}")
    parts.append(
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )
    return "\n\n".join(parts)


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
