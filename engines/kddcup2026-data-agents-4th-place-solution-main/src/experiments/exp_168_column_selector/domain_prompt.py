"""Domain-specific preamble notes for exp_166.

A task-domain router classifies each task into fund / stock / macro / ehr /
other, and the matching note below is prepended to the preamble. The notes
encode mechanical, domain-universal rules derived from failure analysis and
validated against the external BULL (FinSQL) and EHRSQL conventions — they are
NOT tuned to any specific demo task. Notes are in English with the relevant
Chinese trigger terms quoted inline (tasks are bilingual).
"""
from __future__ import annotations

from pathlib import Path

FINANCE_DOMAINS = {"fund", "stock", "macro"}


MACRO_NOTE = """# DOMAIN NOTE: macro (national / economic statistics)

## Schema model — column names may be English or Chinese; confirm with describe_data.
- The tables are flat date-series with no foreign keys; do not join. Each table is one
  statistic family (CPI, GDP, import/export, money & banking, producer prices, ...).
  Find the single table that holds the requested measure.
- Keys: `EndDate` (截止日期) = period end (the usual sort column); `ReportPeriod`
  (数据统计期间) = the statistical period; region columns (`Province` / 省市,
  `ReportArea` / 统计区域类别) exist on some tables only.
- The answer is a named measure column (an index value, or a specific named metric).

## Scope words
- "我国 / 全国 / 咱们 / 我们 / 本国 / 国内" are rhetorical national scope, not a row filter
  and not an aggregation trigger. Do not add a region WHERE clause for them, and do not
  search for a "全国 / total" aggregate row.
- "多少 / 一共 / 总额 / 是多少" are not aggregation triggers — they read as "show the data".

## Output
- DEFAULT: SELECT the requested measure column(s) over ALL rows at the table's grain (one
  row per period / region). Do not collapse to one row.
- Aggregate ONLY when the question has an explicit aggregation word: 平均/average,
  最大/最小/highest/lowest, 总计/合计/sum, 排名/top-N, count/几年/几个/多少家.
- Filter on a region or period ONLY when the question explicitly names that value (a
  province name, a specific year).

"""


FUND_NOTE = """# DOMAIN NOTE: fund (公募基金 / 基金经理)

## Schema model — column names may be English or Chinese; confirm with describe_data.
- Decide which entity the question is about:
  - fund PRODUCT (most common): id typically `InnerCode` (基金内部编码); names
    `SecuAbbr` (基金简称) / `ChiNameAbbr` (中文名称缩写).
  - fund MANAGER (a person): id is a manager code (e.g. `PersonalCode` / 基金经理代码).
  - management company / "基金公司" (基金管理人 / InvestAdvisor) = the firm that runs the
    funds. This is NOT the 基金托管人 (trustee / custodian, which is a BANK); for a
    "基金公司 / fund company" question use the management-company source, never the
    trustee/custodian table.
- Most queries use a single table. When a question spans tables, join on the matching
  entity id (product id or manager id); derived tables may name these differently, so
  confirm the actual column with describe_data.
- `FundTypeName` (基金类别描述) = fund category.

## Output
- Prefer the single-row-per-entity snapshot table over a fan-out table (manager ×
  fund-type × time-interval); do not COUNT over a fan-out to fabricate a per-entity count.
- COUNT(*) by default. Use COUNT(DISTINCT key) only when the question has an explicit
  distinctness word (去重 / 唯一 / 不同 / 不重复 / distinct / unique). A large gap between
  COUNT(*) and COUNT(DISTINCT) is not evidence for DISTINCT.
- For a fund / manager / company name, prefer the short display name (基金简称 SecuAbbr /
  中文名称缩写 ChiNameAbbr) unless the question asks otherwise; use a code column only when
  代码 / code / 编号 is asked.

"""


STOCK_NOTE = """# DOMAIN NOTE: stock (A股 / 上市公司)

## Schema model — column names may be English or Chinese; confirm with describe_data.
- Entity = a listed company. The company identity columns repeat across tables: the
  company id (typically `CompanyCode` / 公司代码), the ticker code (`SecuCode` /
  所属基金/股票代码), the short name (`ChiNameAbbr` / 中文名称缩写) and the full legal
  name (`ChiName` / 公司中文名称).
- Most queries use a single table. When a question spans tables, join them on the company
  id — typically `CompanyCode` (公司代码), which repeats across stock tables.
- Time columns are per-table: `EndDate` (截止日期) = reporting-period end; announcement /
  registration dates are event-specific. Use the date the question actually means.

## Output
- For a company, output the short display name (`ChiNameAbbr` / 中文名称缩写), not the full
  legal name. Use a code column (`SecuCode` / 代码) only when the question asks for
  代码 / code / 编号.

"""


EHR_COMMON_PRIMER = """# EHR DOMAIN HINTS — shortcuts only; verify each against describe_data before relying on it.

## Record hierarchy — decide WHICH level the question means
person > hospital VISIT(=encounter=admission) > unit/ICU STAY > event(lab/procedure/diagnosis/med)
- "patient X" = the person.
- "hospital visit / encounter / admission" and "first|last|current visit" = the VISIT level,
  NOT the unit/ICU stay.
- lab and procedure are point events; they relate only by sharing the same visit/stay
  (there is no direct lab-to-procedure key).

## Time anchoring (this data follows the EHRSQL convention)
- "now / current" is anchored near the dataset's latest timestamp (the EHRSQL convention
  fixes "now" to a late constant, not the real present); read that anchor from the data
  (e.g. the maximum timestamp).
- "current / last hospital visit" = the patient's LATEST admission (active at that anchor);
  "first visit" = the earliest admission.

## Answer semantics — apply when writing the FINAL query
1. NAME not code: if the answer is a procedure/diagnosis/lab/drug NAME, resolve it through the
   name source (a *name column, or a dictionary table). Never return a raw ICD/item code.
2. SUPERLATIVE VALUE: "minimum/maximum <metric>" -> first restrict to value = (SELECT MIN/MAX(...)),
   THEN apply any time qualifier. Do not drop the value filter.
3. VISIT SCOPE: "during their first/last/current visit" filters at the VISIT level. If the patient
   has only one visit it is a no-op — do not add extra stay-level filtering.
4. TIME + TIES: "last/first time ..." orders by the TIME column only. If rows tie on that timestamp,
   do NOT invent a secondary key (id desc, etc.) — keep the time-only result.
5. OUTPUT SHAPE: return only the column(s) literally asked for (usually one: a time / name / value).
   Do not add descriptor/id/date columns unless requested. Confirm shape after seeing the data;
   do not pre-commit a column count.

"""


EICU_CARD = """# SOURCE = eICU  (lowercase tables; person ids are dash-formatted strings, not integers)
- person = uniquepid ; visit = patienthealthsystemstayid ; stay = patientunitstayid.
- Clinical tables (lab, diagnosis, treatment, medication, allergy, microlab, intakeoutput,
  vitalperiodic) are keyed on patientunitstayid ONLY — they do not carry the person/visit id.
- The standard `patient` mapping table may be absent here. The person->clinical-row link can live
  in a patient prose doc OR a bridge/billing table that carries both the person id and an event id.
  Discover the actual link with describe_data — do not assume a direct column.
- "procedure" -> the treatment data (treatmentname / treatmenttime).

"""


MIMIC_CARD = """# SOURCE = MIMIC  (UPPERCASE tables; numeric ids; D_* dictionaries)
- person = SUBJECT_ID ; visit = HADM_ID ; stay = ICUSTAY_ID.
- Clinical events carry SUBJECT_ID + HADM_ID directly (LABEVENTS, PROCEDURES_ICD, PRESCRIPTIONS,
  MICROBIOLOGYEVENTS). Only ICU tables also carry ICUSTAY_ID.
- NAMES need a dictionary join: PROCEDURES_ICD.ICD9_CODE->D_ICD_PROCEDURES.SHORT_TITLE ;
  DIAGNOSES_ICD.ICD9_CODE->D_ICD_DIAGNOSES ; LABEVENTS.ITEMID->D_LABITEMS.LABEL ;
  CHARTEVENTS.ITEMID->D_ITEMS.
- Time column = CHARTTIME ; age/cohort filters use ADMISSIONS.AGE via HADM_ID.

"""


_DOMAIN_NOTES = {
    "macro": MACRO_NOTE,
    "fund": FUND_NOTE,
    "stock": STOCK_NOTE,
    "eicu": EHR_COMMON_PRIMER + EICU_CARD,
    "mimic": EHR_COMMON_PRIMER + MIMIC_CARD,
    "ehr": EHR_COMMON_PRIMER,  # source undetected: common primer only
}


def domain_note(domain: str | None) -> str:
    """Return the per-domain note for fund/stock/macro; empty for ehr/other/None."""
    return _DOMAIN_NOTES.get((domain or "").strip().lower(), "")


# Backwards-compatible alias (exp_159 used finance_domain_note with one shared note).
def finance_domain_note(domain: str | None) -> str:
    return domain_note(domain)


# --- Modality note: domain-agnostic, driven purely by which files the task ships ---
_PROSE_EXTS = {".md", ".pdf", ".txt"}

_MODALITY_HEADER = "# DATA SOURCES — not every answer comes from a structured table.\n"
_PROSE_LINE = (
    "- Part of this task's data is in prose documents (md / pdf): narrative text, footnotes,\n"
    "  mixed sections, not a structured table. Read them; the answer may not be in the\n"
    "  tables alone.\n"
)
_VIDEO_LINE = (
    "- This task includes a video. Some criteria, thresholds, definitions, or the exact target\n"
    "  metric may be stated only in the video — locate them there before answering.\n"
)


def modality_note(context_dir: str | Path | None) -> str:
    """Context-driven, domain-agnostic note.

    Emitted only when the task ships prose documents (md/pdf/txt) or a video.
    A task with only structured tables (csv/json/db) gets nothing -- plain
    table/SQL work needs no caveat.
    """
    if not context_dir:
        return ""
    ctx = Path(context_dir)
    if not ctx.exists():
        return ""
    doc_dir = ctx / "doc"
    has_prose = doc_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in _PROSE_EXTS for p in doc_dir.iterdir()
    )
    video_dir = ctx / "video"
    has_video = video_dir.is_dir() and any(p.is_file() for p in video_dir.iterdir())
    if not (has_prose or has_video):
        return ""
    out = _MODALITY_HEADER
    if has_prose:
        out += _PROSE_LINE
    if has_video:
        out += _VIDEO_LINE
    return out + "\n"
