from __future__ import annotations

import pytest

from data_agent_baseline.tools.unit_normalizer import (
    generate_normalization_rule,
    get_factor,
    normalize_facts,
    normalize_unit,
    normalize_value,
)


class TestNormalizeUnit:
    def test_none_returns_none(self):
        assert normalize_unit(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_unit("") is None
        assert normalize_unit("  ") is None

    def test_canonical_forms_unchanged(self):
        assert normalize_unit("元") == "元"
        assert normalize_unit("万元") == "万元"
        assert normalize_unit("百万元") == "百万元"
        assert normalize_unit("亿元") == "亿元"
        assert normalize_unit("万亿元") == "万亿元"
        assert normalize_unit("%") == "%"

    def test_bare_scale_aliases(self):
        assert normalize_unit("万") == "万元"
        assert normalize_unit("百万") == "百万元"
        assert normalize_unit("千万") == "千万元"
        assert normalize_unit("亿") == "亿元"
        assert normalize_unit("十亿") == "十亿元"
        assert normalize_unit("万亿") == "万亿元"
        assert normalize_unit("千") == "千元"
        assert normalize_unit("百") == "百元"

    def test_currency_qualifier_suffixes_are_not_canonical_units(self):
        assert normalize_unit("万元人民币") is None
        assert normalize_unit("亿元人民币") is None

    def test_percentage_aliases(self):
        assert normalize_unit("百分比") == "%"
        assert normalize_unit("％") == "%"

    def test_basis_point_aliases(self):
        assert normalize_unit("基点") == "基点"
        assert normalize_unit("bps") == "基点"
        assert normalize_unit("bp") == "基点"

    def test_english_aliases(self):
        assert normalize_unit("million") == "百万元"
        assert normalize_unit("billion") == "十亿元"
        assert normalize_unit("trillion") == "万亿元"

    def test_unknown_unit_returns_none(self):
        assert normalize_unit("xyz_unknown") is None
        assert normalize_unit("公斤每平方米") is None


class TestGetFactor:
    def test_known_units(self):
        assert get_factor("元") == 1.0
        assert get_factor("万元") == 10_000.0
        assert get_factor("百万元") == 1_000_000.0
        assert get_factor("亿元") == 100_000_000.0
        assert get_factor("万亿元") == 1_000_000_000_000.0
        assert get_factor("%") == 0.01
        assert get_factor("基点") == 0.0001

    def test_alias_resolution(self):
        assert get_factor("万") == 10_000.0
        assert get_factor("亿") == 100_000_000.0

    def test_currency_qualifiers_use_scale_keyword_without_fx_conversion(self):
        assert get_factor("万元人民币") == 10_000.0
        assert get_factor("亿元港币") == 100_000_000.0
        assert get_factor("万美元") == 10_000.0
        assert get_factor("亿美元") == 100_000_000.0
        assert get_factor("百万美元") == 1_000_000.0

    def test_none_raises_key_error(self):
        with pytest.raises(KeyError, match="unit is None"):
            get_factor(None)

    def test_unknown_unit_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown unit"):
            get_factor("not_a_unit")


class TestNormalizeValue:
    """All normalisation goes to unit=1 — no target_unit parameter."""

    def test_yuan_passthrough(self):
        assert normalize_value(100.0, "元") == 100.0

    def test_wan_yuan_to_base(self):
        assert normalize_value(1.5, "万元") == 15_000.0

    def test_bai_wan_yuan_to_base(self):
        # 21.01965 百万元 → 21,019,650 (the task_59 bug fix)
        result = normalize_value(21.01965, "百万元")
        assert abs(result - 21_019_650.0) < 0.01

    def test_yi_yuan_to_base(self):
        assert normalize_value(3.2, "亿元") == 320_000_000.0

    def test_percent_to_base(self):
        # 15% → 0.15 (unit=1 means decimal)
        assert normalize_value(15, "%") == 0.15
        assert normalize_value(1.5, "%") == 0.015

    def test_none_raw_unit_passthrough(self):
        assert normalize_value(42.0, None) == 42.0

    def test_alias_resolution(self):
        assert normalize_value(6.97, "万") == 69_700.0

    def test_english_alias(self):
        assert normalize_value(2.5, "million") == 2_500_000.0


class TestNormalizeFacts:
    """normalize_facts no longer takes field_value_specs — always normalises to unit=1."""

    def test_converts_raw_dict_to_bare_number(self):
        facts = [
            {
                "line_id": 1,
                "entity_key": {"entity_id": "21"},
                "values": {
                    "depositswithcentralbank": {"raw_number": 1.81797, "raw_unit": "百万元"},
                    "entity_id": "21",
                },
            }
        ]
        result, log = normalize_facts(facts)
        assert len(result) == 1
        assert result[0]["values"]["depositswithcentralbank"] == pytest.approx(1_817_970.0)
        assert result[0]["values"]["entity_id"] == "21"
        assert len(log) == 1
        assert log[0]["converted"] == pytest.approx(1_817_970.0)

    def test_mixed_raw_and_bare_values(self):
        facts = [
            {
                "line_id": 1,
                "values": {
                    "totalassets": {"raw_number": 35.92615, "raw_unit": "百万元"},
                    "totalliabilities": 35.92615,
                },
            }
        ]
        result, log = normalize_facts(facts)
        assert result[0]["values"]["totalassets"] == pytest.approx(35_926_150.0)
        assert result[0]["values"]["totalliabilities"] == 35.92615
        assert len(log) == 1

    def test_unknown_unit_warning(self):
        facts = [
            {
                "line_id": 1,
                "values": {"field1": {"raw_number": 42.0, "raw_unit": "unknown_unit"}},
            }
        ]
        result, log = normalize_facts(facts)
        assert result[0]["values"]["field1"] == 42.0
        assert len(log) == 1
        assert "warning" in log[0]

    def test_multiple_facts(self):
        facts = [
            {"line_id": 1, "values": {"x": {"raw_number": 1.5, "raw_unit": "万元"}}},
            {"line_id": 2, "values": {"x": {"raw_number": 3.0, "raw_unit": "万元"}}},
            {"line_id": 3, "values": {"x": {"raw_number": 21.01965, "raw_unit": "百万元"}}},
        ]
        result, log = normalize_facts(facts)
        assert result[0]["values"]["x"] == pytest.approx(15_000.0)
        assert result[1]["values"]["x"] == pytest.approx(30_000.0)
        assert result[2]["values"]["x"] == pytest.approx(21_019_650.0)
        assert len(log) == 3

    def test_task59_scenario(self):
        """Simulate the task_59 bug: B008 values (元) + B010 values (百万元).
        Both should normalise to unit=1 (元)."""
        facts = [
            {"line_id": 117, "values": {"v": {"raw_number": 1_817_970, "raw_unit": "元"}}},
            {"line_id": 119, "values": {"v": {"raw_number": 3_052_730, "raw_unit": "元"}}},
            {"line_id": 175, "values": {"v": {"raw_number": 22_568_310, "raw_unit": "元"}}},
            {"line_id": 181, "values": {"v": {"raw_number": 21.01965, "raw_unit": "百万元"}}},
            {"line_id": 183, "values": {"v": {"raw_number": 21.26607, "raw_unit": "百万元"}}},
            {"line_id": 219, "values": {"v": {"raw_number": 20.94923, "raw_unit": "百万元"}}},
        ]
        result, log = normalize_facts(facts)

        assert result[0]["values"]["v"] == 1_817_970.0
        assert result[1]["values"]["v"] == 3_052_730.0
        assert result[2]["values"]["v"] == 22_568_310.0
        assert result[3]["values"]["v"] == pytest.approx(21_019_650.0)
        assert result[4]["values"]["v"] == pytest.approx(21_266_070.0)
        assert result[5]["values"]["v"] == pytest.approx(20_949_230.0)

    def test_percentage_normalized_to_decimal(self):
        """15% → 0.15 when normalised to unit=1."""
        facts = [
            {"line_id": 1, "values": {"rate": {"raw_number": 15, "raw_unit": "%"}}},
        ]
        result, log = normalize_facts(facts)
        assert result[0]["values"]["rate"] == 0.15


class TestGenerateNormalizationRule:
    def test_basic_rule(self):
        rule = generate_normalization_rule()
        assert "unit=1" in rule

    def test_rule_with_source_units(self):
        rule = generate_normalization_rule(["万元", "百万元"])
        assert "万元" in rule
        assert "百万元" in rule
