"""Domain-specific preamble notes for exp159."""
from __future__ import annotations

FINANCE_DOMAINS = {"fund", "stock", "macro"}

FINANCE_DOMAIN_NOTE = """# DOMAIN NOTE: finance

Preserve source values and choose the exact requested measure.

Before filtering, calculating, or formatting, check the unit encoded in the
question and the source column/value. Interpret 亿, 万, 万股, 百万, 元, %, and
percentage against the source unit. Do not multiply, divide, add percent signs,
remove percent signs, or reformat decimals unless the question or schema clearly
requires it.

When choosing output columns, distinguish numeric measures from labels, dates,
and IDs. If the question asks for an amount, rate, return, growth, volume, share
count, market value, profit, assets, liabilities, deposits, ratio, or count,
return the matching numeric measure column. Return names/text/codes only when
the question asks for entities, descriptions, or codes.

"""


def finance_domain_note(domain: str | None) -> str:
    key = (domain or "").strip().lower()
    return FINANCE_DOMAIN_NOTE if key in FINANCE_DOMAINS else ""
