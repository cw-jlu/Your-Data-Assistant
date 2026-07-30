"""Tests for the video-rule → filter-recipe templater.

Each test pins a real findings.rules shape observed in production traces. The
templater must be tolerant of the wild variation in keys / nesting / language
across runs, surface a structured recipe when there is enough signal, and
return None when the rules are too sparse to be useful.
"""

from __future__ import annotations

from agents.video.rule_templater import (
    FilterRecipe,
    extract_filter_recipe,
    synthesize_filter_recipe,
)

# ---------- extract_filter_recipe: real production shapes ---------------------


def test_extract_task_25_clean_shape() -> None:
    """Reference run 005 task_25 — well-named keys, every slot present."""
    rules = {
        "field": "dailybenchgr",
        "field_cn": "本日基准增长率",
        "operator": ">",
        "threshold": "0.00%",
        "sort_field": "dailybenchgr",
        "sort_direction": "descending",
        "sort_cn": "降序 (由高到低)",
        "export_field": "SecuAbbr",
        "export_field_cn": "基金简称",
        "table": "mf_benchmarkgrowthrate",
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "dailybenchgr"
    assert r.field_label == "本日基准增长率"
    assert r.operator == ">"
    assert r.threshold == "0.00%"
    assert r.sort_field == "dailybenchgr"
    assert r.sort_descending is True
    assert r.export_fields == ("SecuAbbr",)
    assert r.table_hint == "mf_benchmarkgrowthrate"
    assert r.is_sufficient()


def test_extract_task_4_nested_threshold_dict() -> None:
    """task_4 — threshold is a nested dict with value/unit/operator inside."""
    rules = {
        "threshold": {
            "value": 2000000,
            "unit": "shares",
            "operator": "above",
            "description": "Shareholders with entitlement above this threshold",
        },
        "field": "应配股数 (Entitlement)",
        "status": "Active - locked for this quarter",
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "Entitlement"  # parenthesized ASCII ident preferred over Chinese
    assert r.field_label == "应配股数"
    assert r.operator == ">"  # "above" normalized
    assert r.threshold == "2000000"
    assert r.threshold_unit == "shares"


def test_extract_task_1_filter_field_with_paren_ident_and_chinese_threshold() -> None:
    """task_1 — filter_field carries '<中文> (<ASCII_IDENT>)' shape; threshold_text is Chinese number."""
    rules = {
        "filter_field": "流通A股股本 (AFLOATS)",
        "operator": ">",
        "threshold_value": "10,000,000,000",
        "threshold_text": "100亿",
        "statistical_year": "2019",
        "grouping_dimension": "二级行业 (SecondIndustryName)",
        "deduplication_rule": "按公司去重统计 (Deduplicate by CompanyCode)",
        "source_tables": ["lc_freefloat", "lc_exgindustry"],
        "join_key": "CompanyCode",
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "AFLOATS"
    assert r.field_label == "流通A股股本"
    assert r.operator == ">"
    assert r.threshold == "10,000,000,000"
    assert r.group_by == "SecondIndustryName"  # parenthesized ASCII preferred
    assert r.table_hint == "lc_freefloat"  # first of source_tables list


def test_extract_task_31_ranking_limit_and_output_fields_list() -> None:
    """task_31 — ranking_limit + output_fields list + batch_id all present."""
    rules = {
        "task_name": "利润总额排名",
        "statistical_period": "2021",
        "data_source_table": "mf_mainfinancialindexq",
        "aggregation_field": "totalprofit (季度利润)",
        "ranking_limit": 10,
        "batch_id": "BATCH-2021-Q4",
        "output_fields": ["SecuAbbr", "SecuCode", "TotalProfit"],
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "totalprofit"  # aggregation_field's paren ident
    assert r.sort_field == "totalprofit"
    assert r.top_n == 10
    assert r.batch_id == "BATCH-2021-Q4"
    assert r.table_hint == "mf_mainfinancialindexq"
    assert r.export_fields == ("SecuAbbr", "SecuCode", "TotalProfit")


def test_extract_task_35_snapshot_dates_no_operator() -> None:
    """task_35 — batch with multi snapshot dates but no comparator; still sufficient."""
    rules = {
        "batch_id": "B-2004",
        "batch_name": "2004 Annual Reporting Batch",
        "fields": ["单位存款/企业存款 (Corporate Deposits)", "财政性存款 (Fiscal Deposits)"],
        "unit": "万元",
        "source_table": "中国金融机构人民币信贷收支表",
        "snapshot_dates": ["2004-03-31", "2004-06-30", "2004-12-31"],
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.batch_id == "B-2004"
    assert r.snapshot_dates == ("2004-03-31", "2004-06-30", "2004-12-31")
    assert r.operator is None  # explicitly no comparator
    assert r.is_sufficient()  # structural-only (snapshot_dates) qualifies


def test_extract_task_41_group_by_and_unit_percent() -> None:
    """task_41 — group_by + Annualized return threshold with percent unit."""
    rules = {
        "batch_id": "BATCH-2024-Q3-007",
        "metric": "AnnualizedRRSinceStart",
        "operator": ">",
        "threshold": 10,
        "unit": "%",
        "apply_to": "All fund types",
        "group_by": "FundType",
        "aggregation": "Count",
        "output_format": "CSV",
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "AnnualizedRRSinceStart"
    assert r.operator == ">"
    assert r.threshold == "10"
    assert r.threshold_unit == "%"
    assert r.group_by == "FundType"
    assert r.batch_id == "BATCH-2024-Q3-007"


def test_extract_task_52_chinese_label_only_and_string_zero_threshold() -> None:
    """task_52 — string "0" threshold + Chinese-only field label."""
    rules = {
        "field": "performance",
        "field_label": "任职期间基金增长率",
        "operator": "<",
        "threshold": "0",
        "unit": "%",
        "batch_id": "BT-2024-0318",
        "config_name": "任职绩效准入线筛选",
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "performance"
    assert r.field_label == "任职期间基金增长率"
    assert r.operator == "<"
    assert r.threshold == "0"
    assert r.threshold_unit == "%"
    assert r.batch_id == "BT-2024-0318"


# ---------- operator normalization edge cases ---------------------------------


def test_operator_at_or_above_maps_to_gte() -> None:
    rules = {"field": "x", "operator": "at or above", "threshold": "5"}
    r = extract_filter_recipe(rules)
    assert r is not None and r.operator == ">="


def test_operator_chinese_dayu_dengyu_maps_to_gte() -> None:
    rules = {"field": "x", "operator": "大于等于", "threshold": "5"}
    r = extract_filter_recipe(rules)
    assert r is not None and r.operator == ">="


def test_operator_unknown_returns_none_operator() -> None:
    rules = {"field": "x", "operator": "approximately equals", "threshold": "5"}
    r = extract_filter_recipe(rules)
    # Unknown comparator → operator is None; recipe should still extract field/threshold
    # but is_sufficient() should be False (no filter, no ranking, no structural anchors).
    assert r is not None
    assert r.operator is None
    assert r.field == "x"
    assert not r.is_sufficient()


# ---------- sufficiency gating ------------------------------------------------


def test_is_sufficient_requires_filter_or_ranking_or_structural() -> None:
    assert not FilterRecipe(field="x").is_sufficient()  # no operator
    assert not FilterRecipe(operator=">").is_sufficient()  # no field
    assert FilterRecipe(field="x", operator=">").is_sufficient()
    assert FilterRecipe(sort_field="x", top_n=10).is_sufficient()
    assert FilterRecipe(batch_id="B-2004").is_sufficient()
    assert FilterRecipe(snapshot_dates=("2024-01-01", "2024-12-31")).is_sufficient()
    assert FilterRecipe(group_by="region").is_sufficient()


def test_empty_or_non_dict_rules_returns_none() -> None:
    assert extract_filter_recipe({}) is None
    assert extract_filter_recipe(None) is None
    assert extract_filter_recipe("not a dict") is None  # type: ignore[arg-type]


# ---------- synthesize_filter_recipe (end-to-end render) ----------------------


def test_synthesize_returns_none_for_insufficient_rules() -> None:
    """Rules with only metadata (no field/op/sort/top_n/structural) → None."""
    assert synthesize_filter_recipe({"task_name": "Some Task", "status": "Active"}) is None


def test_synthesize_returns_text_with_pandas_skeleton_for_clean_shape() -> None:
    rules = {
        "field": "dailybenchgr",
        "operator": ">",
        "threshold": "0.00%",
        "sort_field": "dailybenchgr",
        "sort_direction": "descending",
        "export_field": "SecuAbbr",
        "table": "mf_benchmarkgrowthrate",
    }
    text = synthesize_filter_recipe(rules)
    assert text is not None
    assert "video-rule recipe" in text
    assert "VERIFY" in text
    assert "dailybenchgr" in text
    assert "Pandas template" in text
    assert "```python" in text
    assert "mf_benchmarkgrowthrate" in text
    assert "ascending=False" in text  # descending → ascending=False
    assert "SecuAbbr" in text


def test_synthesize_warns_on_snapshot_multiplicity() -> None:
    rules = {
        "batch_id": "B-2004",
        "snapshot_dates": ["2004-03-31", "2004-06-30", "2004-12-31"],
    }
    text = synthesize_filter_recipe(rules)
    assert text is not None
    assert "snapshot_dates" in text
    assert "one row per date" in text


def test_render_top_n_without_threshold_keeps_template_meaningful() -> None:
    """Top-N-only videos (task_31 style) should still render a useful skeleton."""
    rules = {
        "aggregation_field": "totalprofit",
        "ranking_limit": 10,
        "batch_id": "BATCH-2021-Q4",
        "output_fields": ["SecuAbbr", "SecuCode"],
        "data_source_table": "mf_mainfinancialindexq",
    }
    text = synthesize_filter_recipe(rules)
    assert text is not None
    assert "top_n:        10" in text
    assert "BATCH-2021-Q4" in text
    assert "sort_values('totalprofit', ascending=False)" in text
    assert "head(10)" in text
    assert "SecuAbbr" in text and "SecuCode" in text


def test_render_descending_unknown_renders_question_mark() -> None:
    """When sort direction can't be parsed, render '?' rather than guessing."""
    rules = {
        "field": "x",
        "operator": ">",
        "threshold": "5",
        "sort_field": "x",
        # no sort_direction key at all
    }
    text = synthesize_filter_recipe(rules)
    assert text is not None
    assert "sort:         x ?" in text


# ---------- field cleanup edge cases ------------------------------------------


def test_paren_ident_field_extracts_ascii_name() -> None:
    rules = {"field": "二级行业 (SecondIndustryName)", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "SecondIndustryName"
    assert r.field_label == "二级行业"


def test_chinese_only_field_preserved() -> None:
    rules = {"field": "净资产收益率", "operator": ">", "threshold": "0.1"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "净资产收益率"
    assert r.field_label is None


def test_field_with_non_ascii_paren_content_strips_paren_only() -> None:
    """Parens that don't contain an ASCII identifier are stripped from the field."""
    rules = {"field": "增长率 (单位:%)", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.field == "增长率"


# ---------- export_fields coercion --------------------------------------------


def test_export_fields_single_string_becomes_one_element_tuple() -> None:
    rules = {"field": "x", "operator": ">", "threshold": "0", "select": "name"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.export_fields == ("name",)


def test_export_fields_dedups_with_paren_idents() -> None:
    rules = {
        "field": "x",
        "operator": ">",
        "threshold": "0",
        "output_fields": [
            "证券简称 (SecuAbbr)",
            "证券代码 (SecuCode)",
            "SecuAbbr",  # duplicate of first
        ],
    }
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.export_fields == ("SecuAbbr", "SecuCode")


# ---------- top_n coercion ----------------------------------------------------


def test_top_n_string_with_comma_coerced_to_int() -> None:
    rules = {"sort_field": "x", "top": "1,000"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.top_n == 1000


def test_top_n_zero_or_negative_rejected() -> None:
    rules = {"sort_field": "x", "top_n": 0}
    r = extract_filter_recipe(rules)
    # top_n=0 is dropped; without sort_field+top_n pair this isn't ranking; no other anchors → no recipe
    # but sort_field alone keeps recipe constructed
    assert r is not None
    assert r.top_n is None
    assert not r.is_sufficient()  # sort_field alone isn't enough


# ---------- structured rules wrapper -----------------------------------------


def test_extract_new_canonical_rules_wrapper() -> None:
    rules = {
        "canonical": {
            "filters": [
                {"field": "AFloats", "operator": ">", "value": "10000000000", "unit": "shares"}
            ],
            "group_by": ["SecondIndustryName"],
            "sort_by": [{"field": "AFloats", "direction": "desc"}],
            "top_n": 10,
            "tables": ["lc_freefloat"],
            "output_fields": ["SecondIndustryName"],
            "identifiers": {"batch_id": "B-1"},
            "snapshot_dates": ["2019-12-31"],
        },
        "rule_items": [
            {"id": "r1", "kind": "filter", "description": "AFloats greater than 10000000000"}
        ],
        "raw_observations": ["AFloats > 10000000000"],
        "task_specific": {},
    }

    r = extract_filter_recipe(rules)

    assert r is not None
    assert r.field == "AFloats"
    assert r.operator == ">"
    assert r.threshold == "10000000000"
    assert r.threshold_unit == "shares"
    assert r.group_by == "SecondIndustryName"
    assert r.sort_field == "AFloats"
    assert r.sort_descending is True
    assert r.top_n == 10
    assert r.table_hint == "lc_freefloat"
    assert r.export_fields == ("SecondIndustryName",)
    assert r.batch_id == "B-1"
    assert r.snapshot_dates == ("2019-12-31",)


def _multi_canonical_rules() -> dict:
    return {
        "canonical": {
            "filters": [
                {
                    "field": "AFloats",
                    "operator": ">",
                    "value": "10000000000",
                    "unit": "shares",
                    "source_rule_id": "r1",
                },
                {
                    "field": "ListedDate",
                    "operator": "<=",
                    "value": "2019-12-31",
                    "source_rule_id": "r2",
                },
            ],
            "logic": "AND",
            "group_by": ["SecondIndustryName", "Market"],
            "sort_by": [
                {"field": "AFloats", "direction": "desc"},
                {"field": "CompanyCode", "direction": "asc"},
            ],
            "top_n": 10,
            "tables": ["lc_freefloat", "lc_exgindustry"],
            "output_fields": ["SecondIndustryName", "CompanyCode"],
        },
        "rule_items": [
            {"id": "r1", "kind": "filter", "description": "AFloats greater than threshold"},
            {"id": "r2", "kind": "filter", "description": "Listed before end of 2019"},
        ],
        "raw_observations": [],
        "task_specific": {},
    }


def test_extract_new_canonical_rules_wrapper_preserves_multiple_conditions() -> None:
    r = extract_filter_recipe(_multi_canonical_rules())

    assert r is not None
    assert r.field == "AFloats"
    assert r.operator == ">"
    assert r.threshold == "10000000000"
    assert r.logic == "AND"
    assert len(r.filters) == 2
    assert r.filters[1].field == "ListedDate"
    assert r.filters[1].operator == "<="
    assert r.filters[1].threshold == "2019-12-31"
    assert r.group_by == "SecondIndustryName"
    assert r.group_by_fields == ("SecondIndustryName", "Market")
    assert r.table_hint == "lc_freefloat"
    assert r.table_hints == ("lc_freefloat", "lc_exgindustry")
    assert len(r.sorts) == 2
    assert r.sorts[1].field == "CompanyCode"
    assert r.sorts[1].descending is False


def test_synthesize_new_canonical_rules_wrapper_renders_multiple_conditions() -> None:
    text = synthesize_filter_recipe(_multi_canonical_rules())

    assert text is not None
    assert "filter_logic: AND" in text
    assert "AFloats > 10000000000 shares" in text
    assert "ListedDate <= 2019-12-31" in text
    assert "& (df['ListedDate'] <= '2019-12-31')" in text
    assert "sort_by:" in text
    assert "sort_values(['AFloats', 'CompanyCode'], ascending=[False, True])" in text
    assert "table_hints:  ['lc_freefloat', 'lc_exgindustry']" in text
    assert "group_by:     ['SecondIndustryName', 'Market']" in text


def test_canonical_list_values_survive_template_rendering() -> None:
    text = synthesize_filter_recipe(
        {
            "canonical": {
                "filters": [
                    {"field": "FundType", "operator": "in", "value": ["ETF", "LOF"]},
                    {"field": "ReturnPct", "operator": "between", "value": ["0", "10"]},
                ],
                "logic": "AND",
            },
            "rule_items": [],
            "raw_observations": [],
            "task_specific": {},
        }
    )

    assert text is not None
    assert "df['FundType'].isin(['ETF', 'LOF'])" in text
    assert "df['ReturnPct'].between('0', '10', inclusive='both')" in text


def test_new_rules_wrapper_falls_back_to_task_specific_legacy_shape() -> None:
    rules = {
        "canonical": {},
        "rule_items": [],
        "raw_observations": [],
        "task_specific": {
            "field": "dailybenchgr",
            "operator": ">",
            "threshold": "0.00%",
            "table": "mf_benchmarkgrowthrate",
        },
    }

    r = extract_filter_recipe(rules)

    assert r is not None
    assert r.field == "dailybenchgr"
    assert r.operator == ">"
    assert r.threshold == "0.00%"
    assert r.table_hint == "mf_benchmarkgrowthrate"


def test_synthesize_uses_new_canonical_rules_wrapper() -> None:
    text = synthesize_filter_recipe(
        {
            "canonical": {
                "filters": [{"field": "performance", "operator": "<", "value": 0, "unit": "%"}],
                "identifiers": {"batch_id": "BT-2024-0318"},
            },
            "rule_items": [],
            "raw_observations": [],
            "task_specific": {},
        }
    )

    assert text is not None
    assert "performance" in text
    assert "operator:     <" in text
    assert "BT-2024-0318" in text
