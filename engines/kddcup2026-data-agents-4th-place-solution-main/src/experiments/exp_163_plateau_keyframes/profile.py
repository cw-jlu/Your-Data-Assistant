"""Per-column descriptive statistics for richer preambles.

Provides `column_profile_lines(df)` that returns a compact "schema with
stats" block. Per dtype we surface different summaries:

  numeric: dtype, n_null, n_unique, min, q25, median, q75, max, mean
  string:  dtype, n_null, n_unique, top-5 values (with counts), mean_len
  bool:    dtype, n_null, value_counts
  datelike: dtype, n_null, min, max
  other:   dtype, n_null, n_unique

Designed to be cheap (~constant work per column even for million-row tables
once the DataFrame is loaded) and compact (~80-200 chars per column).
"""
from __future__ import annotations

from typing import Any

_TOP_K = 5
_TRUNCATE_VAL_LEN = 30


def _truncate(s: Any, n: int = _TRUNCATE_VAL_LEN) -> str:
    text = str(s)
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _fmt_num(x: Any) -> str:
    try:
        if x is None:
            return "?"
        v = float(x)
    except Exception:
        return _truncate(x)
    if v != v:  # NaN
        return "nan"
    if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
        return f"{v:.3g}"
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e9:
        return f"{int(round(v))}"
    return f"{v:g}"


def column_profile_lines(df) -> list[str]:
    """Return one line per column summarizing dtype + descriptive stats."""
    import pandas as pd

    out: list[str] = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nnull = int(s.isna().sum())
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = -1

        prefix = f"  - {col}: {dtype}, null={nnull}, unique={nunique}"

        try:
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                non_null = s.dropna()
                if len(non_null) == 0:
                    out.append(prefix)
                    continue
                desc = non_null.describe(percentiles=[0.25, 0.5, 0.75])
                out.append(
                    f"{prefix}, min={_fmt_num(desc['min'])} q25={_fmt_num(desc['25%'])} "
                    f"median={_fmt_num(desc['50%'])} q75={_fmt_num(desc['75%'])} "
                    f"max={_fmt_num(desc['max'])} mean={_fmt_num(desc['mean'])}"
                )
            elif pd.api.types.is_bool_dtype(s):
                vc = s.value_counts(dropna=True)
                pairs = ", ".join(f"{k}={int(v)}" for k, v in vc.items())
                out.append(f"{prefix}, counts=[{pairs}]")
            elif pd.api.types.is_datetime64_any_dtype(s):
                non_null = s.dropna()
                if len(non_null) == 0:
                    out.append(prefix)
                    continue
                out.append(f"{prefix}, min={_fmt_num(non_null.min())} max={_fmt_num(non_null.max())}")
            else:
                # Treat as categorical/string. Show top-K values + counts.
                non_null = s.dropna().astype(str)
                if len(non_null) == 0:
                    out.append(prefix)
                    continue
                vc = non_null.value_counts().head(_TOP_K)
                top_pairs = ", ".join(
                    f"{_truncate(repr(k), _TRUNCATE_VAL_LEN)}({v})" for k, v in vc.items()
                )
                mean_len = float(non_null.str.len().mean()) if len(non_null) else 0.0
                out.append(
                    f"{prefix}, mean_len={mean_len:.1f}, top{min(_TOP_K, len(vc))}=[{top_pairs}]"
                )
        except Exception as exc:
            out.append(f"{prefix}, profile_error={exc!r}")
    return out


def csv_profile_text(path, max_chars: int = 8000) -> str:
    """Build a 'CSV profile' string: shape + per-col stats + first 5 rows.

    Returns an empty string if the file cannot be parsed at all.
    """
    import pandas as pd

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return ""
    parts: list[str] = []
    parts.append(f"shape: {df.shape}")
    parts.append("columns:")
    parts.extend(column_profile_lines(df))
    try:
        sample_n = min(5, len(df))
        if sample_n > 0:
            parts.append("\nfirst 5 rows:")
            parts.append(df.head(sample_n).to_string(index=False))
    except Exception:
        pass
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 30] + "\n[...profile truncated...]"
    return text


def sqlite_table_profile(conn, table: str, max_rows_for_stats: int = 100_000) -> list[str]:
    """Profile one SQLite table. Caller passes an open sqlite3 connection."""
    import pandas as pd

    out: list[str] = []
    # Pull at most max_rows_for_stats; for a 1M-row table that's still
    # representative for stats (relative error <0.1% on quantiles).
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {max_rows_for_stats}', conn)
    except Exception as exc:
        return [f"  [profile error: {exc!r}]"]
    out.append(f"  shape: {df.shape} (sampled to {max_rows_for_stats} rows for stats)")
    out.append("  columns:")
    for line in column_profile_lines(df):
        out.append("  " + line)
    return out


def json_array_profile(records: list[dict], max_chars: int = 6000) -> str:
    """Profile a list-of-dicts (top-level) JSON file."""
    import pandas as pd

    if not records or not isinstance(records[0], dict):
        return ""
    try:
        df = pd.DataFrame(records)
    except Exception:
        return ""
    parts = [f"shape: {df.shape}", "columns:"]
    parts.extend(column_profile_lines(df))
    try:
        sample_n = min(5, len(df))
        if sample_n > 0:
            parts.append("\nfirst 5 rows:")
            parts.append(df.head(sample_n).to_string(index=False))
    except Exception:
        pass
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 30] + "\n[...profile truncated...]"
    return text
