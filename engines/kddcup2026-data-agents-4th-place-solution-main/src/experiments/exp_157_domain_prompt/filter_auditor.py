"""Post-proposal filter auditor.

After answer_from_sql produces a SQL + result, an LLM checks each WHERE/JOIN/
HAVING clause against the question. Returns a verdict (PASS / NEEDS_FIX)
plus per-clause justification, so the agent can fix unjustified filters before
final commit.

GENERIC, no task-specific examples or thresholds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage


FILTER_AUDITOR_SYSTEM = """\
Audit the SQL's WHERE / HAVING clauses for two errors:

  (a) UNJUSTIFIED filter — a specific WHERE / HAVING clause is present in the
      SQL but the question never asked for it. Quote the SQL clause and explain
      which question phrase you tried to map it to.
  (b) MISSING filter   — a narrowing phrase in the question (= "for X", "in Y",
      "where Z") has no matching clause in the WHERE / HAVING. Quote the
      question phrase and the column it should filter.

Issues MUST quote a specific WHERE / HAVING clause from the SQL or a specific
phrase from the question. If you can't quote one, the issue is out of scope
and must not be reported.

Output PASS when every WHERE / HAVING clause traces to a question phrase AND
every narrowing phrase in the question has a matching clause.

Output format (exactly these labels):
VERDICT: PASS | NEEDS_FIX
ISSUES:
  - <one line per issue, each quoting a clause or phrase, "none" if PASS>
SUGGEST:
  - <one line per concrete fix, "none" if PASS>
"""


@dataclass(frozen=True, slots=True)
class FilterAuditResult:
    verdict: str  # "PASS" | "NEEDS_FIX" | "ERROR"
    issues: list[str]
    suggestions: list[str]
    raw_response: str

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"

    def feedback_text(self) -> str:
        """Format as observation text for the agent."""
        if self.passes:
            return "FILTER AUDIT: PASS — all clauses justified by the question."
        lines = [f"FILTER AUDIT: {self.verdict}"]
        if self.issues:
            lines.append("Issues:")
            for i in self.issues: lines.append(f"  - {i}")
        if self.suggestions:
            lines.append("Suggested fixes:")
            for s in self.suggestions: lines.append(f"  - {s}")
        return "\n".join(lines)


def audit_filters(
    *,
    question: str,
    sql: str,
    columns: list[str],
    rows: list[list] | None = None,
    n_rows: int = 0,
    schema_text: str = "",
    model: ModelAdapter,
) -> FilterAuditResult:
    """Return verdict on whether SQL filters trace to question phrases.

    Inputs:
      schema_text: catalog/schema description (= view names + columns) so the
        auditor can check column existence and types.
      rows: result rows (first 5-10 used as preview) so the auditor can spot
        duplicates and impossible values.
    """
    preview = ""
    if rows:
        head = rows[:5]
        preview = "\n".join(f"  {r}" for r in head)
    user_prompt = (
        f"## Question\n{question}\n\n"
        f"## Proposed SQL\n```sql\n{sql.strip()}\n```\n\n"
        f"## SQL result shape\n{n_rows} rows × {len(columns)} columns "
        f"(columns: {columns})\n"
        + (f"\n## Result preview (= first 5 rows)\n{preview}\n" if preview else "")
        + "\nAudit each WHERE / JOIN / HAVING clause + column existence/types + "
        "row-count plausibility. Output VERDICT, ISSUES, SUGGEST."
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=FILTER_AUDITOR_SYSTEM),
                ModelMessage(role="user", content=user_prompt),
            ],
            enable_thinking=False,
        )
    except Exception as exc:
        return FilterAuditResult(
            verdict="ERROR", issues=[f"audit LLM error: {exc}"],
            suggestions=[], raw_response="",
        )

    # Parse VERDICT line
    m = re.search(r"VERDICT\s*:\s*(\w+)", raw, flags=re.IGNORECASE)
    verdict = (m.group(1).upper() if m else "ERROR")
    if verdict not in ("PASS", "NEEDS_FIX"):
        verdict = "ERROR"

    # Parse ISSUES bullets (= lines after "ISSUES:" up to "SUGGEST:" or end)
    def _bullets(section_label: str) -> list[str]:
        m = re.search(
            rf"{section_label}\s*:\s*\n((?:\s*-\s*.+\n?)+)",
            raw, flags=re.IGNORECASE,
        )
        if not m: return []
        out = []
        for line in m.group(1).split("\n"):
            line = line.strip()
            if not line.startswith("-"): continue
            body = line.lstrip("-").strip()
            if body.lower() in ("none", "n/a", ""): continue
            out.append(body)
        return out

    issues = _bullets("ISSUES")
    suggestions = _bullets("SUGGEST")

    # If verdict says PASS but issues found, downgrade
    if verdict == "PASS" and issues: verdict = "NEEDS_FIX"
    return FilterAuditResult(
        verdict=verdict, issues=issues, suggestions=suggestions, raw_response=raw,
    )
