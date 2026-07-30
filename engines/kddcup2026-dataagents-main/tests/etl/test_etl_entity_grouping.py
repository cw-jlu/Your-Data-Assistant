"""Entity grouping: LLM paragraph grouping with deterministic verification.

The ETL prose pipeline numbers every paragraph, batches them to the LLM,
and has it emit ``<record_id>: <paragraph indices>`` groups; the claims
are then verified deterministically (anchor presence, hijack guard,
contested-index resolution with anchor-verified multi-membership for
group paragraphs, distinctness gate).  The builtin record-ID regex
survives only as total-failure fallback and as the verification
primitive.

Regression context (task_59, ed_otherdepositorycorpbs.md): the previous
LLM-discovered record-ID regex missed verbose label phrasings
("registered as 60") and captured digit fragments inside formatted numbers
("8,408,793" → "408"), which spawned spurious entity groups, hijacked
paragraphs, and silently dropped 3 of 50 entities from the converted CSV.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agents.etl._compress import (
    compress_prose,
)
from agents.etl._grouping import (
    GROUPING_BATCH_SIZE,
    group_paragraphs_by_entity,
    group_paragraphs_by_llm,
    grouping_head,
    make_grouping_batches,
    parse_grouping_response,
    verify_paragraph_groups,
)
from agents.etl._types import is_embedded_digit_run as _is_embedded_digit_run
from agents.llm.types import ModelResponse


@pytest.mark.parametrize(
    ("para", "expected"),
    [
        # Decimal points inside numerals never terminate the sentence.
        pytest.param(
            "Unit 144 reported 196,146,682.99 in Total Liabilities. The audit went well.",
            "Unit 144 reported 196,146,682.99 in Total Liabilities.",
            id="decimal-safe",
        ),
        # CJK terminators bound sentences without trailing whitespace.
        pytest.param(
            "档案286的资产总额为41,740,771。该季度审计顺利完成。",
            "档案286的资产总额为41,740,771。",
            id="cjk-terminator",
        ),
        # No terminator at all: the whole (collapsed) paragraph is the head.
        pytest.param(
            "Record 16 carries the figures\nfor the cycle without a closing mark",
            "Record 16 carries the figures for the cycle without a closing mark",
            id="no-terminator-fallback",
        ),
        # Rhetorical opener defers the ID to sentence two: head extends so the
        # grouping signal is not lost (in_meansofproductionpi, asset 1487).
        pytest.param(
            "The investigation now turns to a different methodology. "
            "The asset registered as 1487 used a baseline of 100. Scope national.",
            "The investigation now turns to a different methodology. "
            "The asset registered as 1487 used a baseline of 100.",
            id="extends-to-numberless-opener",
        ),
        pytest.param(
            "The investigation now turns to a different methodology. "
            "A second setup sentence still carries no label. "
            "The asset registered as 1487 used a baseline of 100. Scope national.",
            "The investigation now turns to a different methodology. "
            "A second setup sentence still carries no label. "
            "The asset registered as 1487 used a baseline of 100.",
            id="extends-past-two-sentences",
        ),
        # Pure narrative keeps expanding until the whole paragraph is shown.
        pytest.param(
            "Purely narrative filler about methodology. More filler follows here.",
            "Purely narrative filler about methodology. More filler follows here.",
            id="narrative-expands-to-whole-paragraph",
        ),
    ],
)
def test_grouping_head_keeps_record_signal(para: str, expected: str) -> None:
    assert grouping_head(para) == expected


@pytest.mark.parametrize("n", [1, 39, 40, 41, 46, 79, 80, 81, 606])
def test_grouping_batches_cover_all_paragraphs_without_weak_tail(n: int) -> None:
    """Regression: 606 paragraphs under fixed-stride slicing left a tail batch
    of 6 whose numbered-list anchor was too weak — the model answered in prose
    and the batch's paragraphs (the document's last five entities' Section XI
    data) all fell to the leftover path (task_53)."""
    paras = [f"Record {i} carries the figures for cycle {i}." for i in range(1, n + 1)]
    batches = make_grouping_batches(paras)

    flat = [i for indices, _ in batches for i in indices]
    assert flat == list(range(1, n + 1))

    sizes = [len(indices) for indices, _ in batches]
    assert max(sizes) <= GROUPING_BATCH_SIZE
    if len(sizes) > 1:
        assert min(sizes) >= GROUPING_BATCH_SIZE // 2
        assert max(sizes) - min(sizes) <= 1


def test_embedded_digit_fragments_detected() -> None:
    cases = [
        ("the sum of 8,408,793 in deposits", "408", True),  # comma-group middle
        ("the sum of 8,408,793 in deposits", "793", True),  # comma-group tail
        ("reached 671,408.57 overall", "57", True),  # decimal fraction
        ("concluded on September 30th, 2003.", "30", True),  # ordinal word
        ("period ending 2003", "200", True),  # fragment of longer run
        ("registered as 60 held assets", "60", False),  # real standalone id
        ("档案286的经理人", "286", False),  # CJK after digits is prose
        ("Record 25, filed yesterday", "25", False),  # prose comma after id
    ]
    for text, token, expected in cases:
        start = text.index(token)
        got = _is_embedded_digit_run(text, start, start + len(token))
        assert got is expected, (text, token)


@pytest.mark.parametrize(
    (
        "raw",
        "expected_groups",
        "expected_none_indices",
        "expected_groups_none",
        "expected_unparsed",
    ),
    [
        # Prose and fences around the group lines are tolerated; the two prose
        # lines are counted as unparsed, never fatal.
        pytest.param(
            "Here are the entity groupings you asked for:\n"
            "```\n"
            "16: 1, 2,3\n"
            "20: 4\n"
            "NONE: 5, 6\n"
            "```\n"
            "Let me know if you need anything else.",
            {"16": [1, 2, 3], "20": [4]},
            [5, 6],
            False,
            2,
            id="tolerates-fences-and-prose",
        ),
        # Lines echoing the input's paragraph numbering parse to the real rid.
        # Regression: task_13 mf_fcretscalerank batch 1 — the model prefixed every
        # line with the paragraph number ("1) 9: 1" = paragraph 1 → record 9), all
        # 40 lines fell to unparsed, and the whole batch dropped to the leftover
        # path.  The prefix needs ")"/"."/"）" after the digits, so unprefixed
        # "<rid>: ..." lines keep their rid.
        pytest.param(
            "1) 9: 1\n2. 10: 2\n3） 53: 3\n4) NONE: 4\n16: 5",
            {"9": [1], "10": [2], "53": [3], "16": [5]},
            [4],
            False,
            0,
            id="strips-enumeration-prefix",
        ),
        pytest.param(
            "2162, 2166 : 12\n2162、2166 ： 13",
            {"2162": [12, 13], "2166": [12, 13]},
            [],
            False,
            0,
            id="parses-multi-id-lines-with-spaces",
        ),
        pytest.param("GROUPS: NONE", {}, [], True, 0, id="groups-none-bare"),
        pytest.param("groups: none", {}, [], True, 0, id="groups-none-lowercase"),
        pytest.param(
            "The paragraphs form a time series.\nGROUPS: NONE",
            {},
            [],
            True,
            0,
            id="groups-none-after-prose",
        ),
    ],
)
def test_parse_grouping_response(
    raw: str,
    expected_groups: dict[str, list[int]],
    expected_none_indices: list[int],
    expected_groups_none: bool,
    expected_unparsed: int,
) -> None:
    groups, none_indices, groups_none, unparsed = parse_grouping_response(raw)

    assert groups == expected_groups
    assert none_indices == expected_none_indices
    assert groups_none is expected_groups_none
    assert unparsed == expected_unparsed


@pytest.mark.parametrize(
    ("paras", "groups", "expected"),
    [
        # Record 99 appears nowhere: the hallucinated group is dropped.
        pytest.param(
            [
                "Record 16 reports assets of 8,693,705 for the period in question.",
                "Record 20 reports liabilities of 1,413,113 in the same filing.",
                "Record 25 closed its books with 53,413,270 in total claims there.",
                "A general narrative paragraph with no identifiers inside it at all.",
            ],
            {"16": [1], "20": [2], "25": [3], "99": [4]},
            {"16": [1], "20": [2], "25": [3]},
            id="drops-hallucinated-group-without-anchor",
        ),
        # Paragraph 4 states Record 20's ID, so it cannot ride along with 16.
        pytest.param(
            [
                "Record 16 opens the filing cycle with the usual disclosures noted.",
                "Record 20 continues the cycle with separate balances reported here.",
                "Record 25 concludes the cycle with closing remarks and totals given.",
                "Additional figures for Record 20 appear in this trailing paragraph.",
            ],
            {"16": [1, 4], "20": [2], "25": [3]},
            {"16": [1], "20": [2, 4], "25": [3]},
            id="hijack-guard-reassigns-to-mentioned-entity",
        ),
        # Paragraph 2 states only Record 20's ID, so the contested claim is
        # resolved to 20 and dropped from 16.
        pytest.param(
            [
                "Record 16 holds the opening figures for this reporting period here.",
                "Record 20 holds the middle figures for this reporting period here.",
                "Record 25 holds the closing figures for this reporting period here.",
                "Record 30 holds the appendix figures for this reporting period here.",
            ],
            {"16": [1, 2], "20": [2], "25": [3], "30": [4]},
            {"16": [1], "20": [2], "25": [3], "30": [4]},
            id="resolves-contested-index-to-anchoring-claimant",
        ),
        # Paragraph 4 anchors neither claimant: ambiguous continuation, so it
        # is invalidated everywhere and routes to the leftover/positional path.
        pytest.param(
            [
                "Record 16 holds the opening figures for this reporting period here.",
                "Record 20 holds the middle figures for this reporting period here.",
                "Record 25 holds the closing figures for this reporting period here.",
                "A continuation paragraph with no identifier restated anywhere at all.",
            ],
            {"16": [1, 4], "20": [2, 4], "25": [3]},
            {"16": [1], "20": [2], "25": [3]},
            id="invalidates-contested-index-without-anchor",
        ),
        # The group paragraph states every claimant's ID standalone, so it is
        # granted membership in EACH claiming entity.
        pytest.param(
            [
                "Record 16 opens the cycle with its own dedicated figures here.",
                "Record 20 continues the cycle with its own dedicated figures here.",
                "Record 25 concludes the cycle with its own dedicated figures here.",
                "For records 16, 20 and 25 the shared composite index stood at 111.5.",
            ],
            {"16": [1, 4], "20": [2, 4], "25": [3, 4]},
            {"16": [1, 4], "20": [2, 4], "25": [3, 4]},
            id="allows-multi-membership-for-group-paragraph",
        ),
        # The distinctness gate rejects a dense quarter-style {1..4} run.
        pytest.param(
            [
                f"Quarter {i} GDP grew by 4.{i} percent across the seasonal periods."
                for i in (1, 2, 3, 4)
            ],
            {str(i): [i] for i in (1, 2, 3, 4)},
            None,
            id="gate-rejects-tiny-dense-id-run",
        ),
        # Sparse non-consecutive IDs pass the same gate untouched.
        pytest.param(
            [
                f"Record {rid} reports the balance for the filing period in question."
                for rid in (16, 20, 21, 25)
            ],
            {"16": [1], "20": [2], "21": [3], "25": [4]},
            {"16": [1], "20": [2], "21": [3], "25": [4]},
            id="gate-passes-sparse-id-run",
        ),
        # Index 99 (beyond the paragraph range) and 0 (below it) are dropped.
        pytest.param(
            [
                "Record 16 covers the first reporting block of the cycle in question.",
                "Record 20 covers the second reporting block of the cycle in question.",
                "Record 25 covers the third reporting block of the cycle in question.",
            ],
            {"16": [1, 99], "20": [2, 0], "25": [3]},
            {"16": [1], "20": [2], "25": [3]},
            id="drops-out-of-range-indices",
        ),
    ],
)
def test_verify_paragraph_groups(
    paras: list[str], groups: dict[str, list[int]], expected: dict[str, list[int]] | None
) -> None:
    assert verify_paragraph_groups(groups, paras) == expected


class _PromptDispatchAdapter:
    """Dispatch scripted responses on prompt content (never call order).

    Grouping batches run in a 4-worker pool, so arrival order is
    nondeterministic — tests must key responses on prompt markers.
    """

    def __init__(self, rules: list[tuple[str, str | Exception]]) -> None:
        self._rules = rules
        self.prompts: list[str] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        for marker, response in self._rules:
            if marker in prompt:
                if isinstance(response, Exception):
                    raise response
                return ModelResponse(content=response)
        raise AssertionError(f"no scripted response for prompt: {prompt[:200]}")


_LAST_PARA_INDEX = GROUPING_BATCH_SIZE + 1


def _multi_batch_paras(last_para: str) -> list[str]:
    """GROUPING_BATCH_SIZE+1 paragraphs: 3 anchored entities, filler, then
    ``last_para`` as the final paragraph — balanced batching puts it in batch 2."""
    paras = [
        "Record 16 opens the document with initial figures for the cycle.",
        "Record 20 carries the middle figures for the reporting cycle here.",
        "Record 25 carries further figures for the reporting cycle here.",
    ]
    paras += [
        "Purely narrative filler text about methodology and survey design."
        for _ in range(_LAST_PARA_INDEX - 4)
    ]
    paras.append(last_para)
    assert len(paras) == _LAST_PARA_INDEX
    return paras


def test_llm_grouping_merges_same_id_across_batches() -> None:
    paras = _multi_batch_paras("Record 16 returns at the end with appendix data for the cycle.")
    # Paragraph 5 carries a standalone 99 so only the batch-range filter —
    # not anchor verification — can stop batch 2's out-of-batch claim on it.
    paras[4] = "A filler paragraph that mentions the number 99 in passing prose."

    adapter = _PromptDispatchAdapter(
        [
            (f"{_LAST_PARA_INDEX}) Record 16 returns", f"16: {_LAST_PARA_INDEX}\n99: 5"),
            ("1) Record 16 opens", "16: 1\n20: 2\n25: 3"),
        ]
    )
    verified, llm_responded = group_paragraphs_by_llm(cast(Any, adapter), paras)

    assert llm_responded is True
    assert verified is not None
    # Same rid claimed in two batches merges into one document-ordered group.
    assert verified["16"] == [1, _LAST_PARA_INDEX]
    assert verified["20"] == [2]
    assert verified["25"] == [3]
    # Batch 2's claim on a paragraph outside its own range is filtered out.
    assert "99" not in verified
    assert len(adapter.prompts) == 2


def test_llm_grouping_groups_none_batch_contributes_nothing() -> None:
    paras = _multi_batch_paras("Record 30 closes the appendix with figures for the final cycle.")

    adapter = _PromptDispatchAdapter(
        [
            (f"{_LAST_PARA_INDEX}) Record 30 closes", "GROUPS: NONE"),
            ("1) Record 16 opens", "16: 1\n20: 2\n25: 3"),
        ]
    )
    verified, llm_responded = group_paragraphs_by_llm(cast(Any, adapter), paras)

    assert llm_responded is True
    assert verified == {"16": [1], "20": [2], "25": [3]}


def test_llm_grouping_failed_batch_routes_paras_to_leftover() -> None:
    paras = _multi_batch_paras("Record 30 closes the appendix with figures for the final cycle.")

    adapter = _PromptDispatchAdapter(
        [
            (
                f"{_LAST_PARA_INDEX}) Record 30 closes",
                RuntimeError("batch 2 infrastructure failure"),
            ),
            ("1) Record 16 opens", "16: 1\n20: 2\n25: 3"),
        ]
    )
    result = group_paragraphs_by_entity("\n\n".join(paras), adapter=cast(Any, adapter))

    assert result is not None
    groups, leftover = result
    # Healthy batches' verified groups are kept...
    assert sorted(groups) == ["16", "20", "25"]
    # ...and the failed batch's digit-bearing paragraph routes to leftover
    # instead of being silently dropped.
    assert paras[-1] in leftover


def test_grouping_multi_membership_routes_group_para_to_each_entity() -> None:
    paras = [
        "Record 16 opens the cycle with its own dedicated figures here.",
        "Record 20 continues the cycle with its own dedicated figures here.",
        "Record 25 concludes the cycle with its own dedicated figures here.",
        "For records 16, 20 and 25 the shared composite index stood at 111.5.",
    ]
    adapter = _PromptDispatchAdapter(
        [("numbered paragraph openings", "16: 1, 4\n20: 2, 4\n25: 3, 4")]
    )

    result = group_paragraphs_by_entity("\n\n".join(paras), adapter=cast(Any, adapter))

    assert result is not None
    groups, leftover = result
    # The group paragraph lands in EVERY claiming entity's context, so each
    # entity's chunk can extract its own value from it.
    assert sorted(groups) == ["16", "20", "25"]
    for rid in ("16", "20", "25"):
        assert paras[3] in groups[rid]
    assert leftover == []


def test_grouping_recovery_uses_shared_auto_anchor_patterns() -> None:
    paras = [
        "The case file associated with patient unit stay 101 reports a first metric.",
        "The follow-up note for unit stay 101 reports a second metric.",
        "The case file associated with patient unit stay 102 reports a first metric.",
        "The follow-up note for unit stay 102 reports a second metric.",
        "The case file associated with patient unit stay 103 reports a first metric.",
        "The follow-up note for unit stay 103 reports a second metric.",
    ]
    adapter = _PromptDispatchAdapter([("numbered paragraph openings", "101: 1\n102: 3\n103: 5")])

    result = group_paragraphs_by_entity("\n\n".join(paras), adapter=cast(Any, adapter))

    assert result is not None
    groups, leftover = result
    assert groups["101"] == [paras[0], paras[1]]
    assert groups["102"] == [paras[2], paras[3]]
    assert groups["103"] == [paras[4], paras[5]]
    assert leftover == []


def test_grouping_ignores_number_fragments_and_returns_leftover() -> None:
    paras = [
        "Record 16 held a combined ReserveAssets total of 8,693,705 overall.",
        "Record 21 reported the sum of 8,408,793 in DepositsWithCentralBank.",
        "Record 24 listed 53,413,270 in claims on other institutions there.",
        "The portfolio asset registered under 20 finalized its filings late.",
        "Narrative filler paragraph with no numbers in it at all, only words.",
    ]
    # No adapter: the builtin record-ID regex fallback path.
    result = group_paragraphs_by_entity("\n\n".join(paras))
    assert result is not None
    groups, leftover = result

    # No spurious entities from "8,693,705" / "8,408,793" / "53,413,270".
    assert sorted(groups) == ["16", "21", "24"]
    assert groups["16"] == [paras[0]]
    # The verbose-phrasing paragraph is not silently dropped: it carries a
    # standalone number, so it must come back as leftover for positional
    # processing.
    assert leftover == [paras[3]]


def test_grouping_fallback_uses_shared_auto_anchor_patterns() -> None:
    paras = [
        "The case file associated with patient unit stay 101 reports a first metric.",
        "The follow-up note for unit stay 101 reports a second metric.",
        "The case file associated with patient unit stay 102 reports a first metric.",
        "The follow-up note for unit stay 102 reports a second metric.",
        "The case file associated with patient unit stay 103 reports a first metric.",
        "The follow-up note for unit stay 103 reports a second metric.",
    ]

    result = group_paragraphs_by_entity("\n\n".join(paras))

    assert result is not None
    groups, leftover = result
    assert sorted(groups) == ["101", "102", "103"]
    assert groups["101"] == [paras[0], paras[1]]
    assert groups["102"] == [paras[2], paras[3]]
    assert groups["103"] == [paras[4], paras[5]]
    assert leftover == []


class _RecoveryAdapter:
    """Main pass emits entity 5 in marker casing and skips entity 7."""

    def __init__(self) -> None:
        self.chunk_prompts: list[str] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        prompt = messages[-1].content
        if "numbered paragraph openings" in prompt:
            return ModelResponse(content="5: 1\n6: 2\n7: 3")
        if "SYSTEM PROMPT" in prompt:
            return ModelResponse(content="short")
        self.chunk_prompts.append(prompt)
        if "[RECORD_ID: 7]" in prompt and "[RECORD_ID: 5]" not in prompt:
            return ModelResponse(content="record_id: 7 | metric: seventy")
        return ModelResponse(content="RECORD_ID: 5 | metric: fifty\n\nrecord_id: 6 | metric: sixty")


def test_compress_prose_recovers_missing_and_ignores_casing() -> None:
    paras = [
        f"Record {i} shows a metric value of interest in this cycle period." for i in (5, 6, 7)
    ]
    adapter = _RecoveryAdapter()

    result, *_ = compress_prose(
        cast(Any, adapter),
        "\n\n".join(paras),
        schema_columns=["record_id", "metric"],
        primary_key="record_id",
    )

    # Marker-style casing is normalized by the parser-backed integrity scan:
    # entity 5 is not treated as missing, so the main chunk is followed by
    # exactly one recovery chunk and that chunk re-requests entity 7 alone.
    # The raw compress output itself is no longer case-normalized here; the
    # downstream parser canonicalizes schema spelling once.
    assert "RECORD_ID: 5 | metric: fifty" in result
    assert len(adapter.chunk_prompts) == 2
    recovery = next(p for p in adapter.chunk_prompts if "[RECORD_ID: 5]" not in p)
    assert "[RECORD_ID: 7]" in recovery
    assert "[RECORD_ID: 6]" not in recovery
    # Entity 7 was missing from the main pass and recovered by the retry.
    assert "record_id: 7 | metric: seventy" in result


class _LeftoverAdapter:
    """Entity chunks get entity lines; the positional chunk gets entity 9."""

    def __init__(self) -> None:
        self.chunk_prompts: list[str] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        prompt = messages[-1].content
        if "numbered paragraph openings" in prompt:
            return ModelResponse(content="5: 1\n6: 2\n7: 3")
        if "SYSTEM PROMPT" in prompt:
            return ModelResponse(content="short")
        self.chunk_prompts.append(prompt)
        if "[RECORD_ID:" in prompt:
            return ModelResponse(
                content=(
                    "record_id: 5 | metric: fifty\n\n"
                    "record_id: 6 | metric: sixty\n\n"
                    "record_id: 7 | metric: seventy"
                )
            )
        return ModelResponse(content="record_id: 9 | metric: ninety")


def test_compress_prose_processes_unmatched_numeric_paragraphs() -> None:
    paras = [
        f"Record {i} shows a metric value of interest in this cycle period." for i in (5, 6, 7)
    ]
    paras.append("The special asset registered under 9 reported a metric of ninety.")
    adapter = _LeftoverAdapter()

    result, *_ = compress_prose(
        cast(Any, adapter),
        "\n\n".join(paras),
        schema_columns=["record_id", "metric"],
        primary_key="record_id",
    )

    # The unassigned paragraph went through a positional chunk (no markers)
    # instead of being dropped, and its record made it into the output.
    assert any("registered under 9" in p and "[RECORD_ID:" not in p for p in adapter.chunk_prompts)
    assert "record_id: 9 | metric: ninety" in result


class _GroupingDownAdapter:
    """Raises on every grouping call; answers guide and chunk prompts."""

    def __init__(self) -> None:
        self.chunk_prompts: list[str] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        prompt = messages[-1].content
        if "numbered paragraph openings" in prompt:
            raise RuntimeError("grouping endpoint down")
        if "SYSTEM PROMPT" in prompt:
            return ModelResponse(content="short")
        self.chunk_prompts.append(prompt)
        return ModelResponse(
            content=(
                "record_id: 5 | metric: fifty\n\n"
                "record_id: 6 | metric: sixty\n\n"
                "record_id: 7 | metric: seventy"
            )
        )


def test_grouping_falls_back_to_regex_when_all_llm_calls_fail() -> None:
    paras = [
        f"Record {i} shows a metric value of interest in this cycle period." for i in (5, 6, 7)
    ]
    adapter = _GroupingDownAdapter()

    result, *_ = compress_prose(
        cast(Any, adapter),
        "\n\n".join(paras),
        schema_columns=["record_id", "metric"],
        primary_key="record_id",
    )

    # Every grouping call failed on infrastructure, so the builtin regex
    # took over and entity-aware chunks with markers were still built.
    assert any("[RECORD_ID: 5]" in p for p in adapter.chunk_prompts)
    assert "record_id: 5 | metric: fifty" in result


class _GateRejectAdapter:
    """Groups a dense quarter-style run {1..4}; the gate must reject it."""

    def __init__(self) -> None:
        self.chunk_prompts: list[str] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        prompt = messages[-1].content
        if "numbered paragraph openings" in prompt:
            return ModelResponse(content="1: 1\n2: 2\n3: 3\n4: 4")
        if "SYSTEM PROMPT" in prompt:
            return ModelResponse(content="short")
        self.chunk_prompts.append(prompt)
        return ModelResponse(content="record_id: 1 | metric: one")


def test_grouping_rejected_by_gate_uses_positional_path() -> None:
    paras = [
        f"Record {i} grew by 4.{i} percent across the seasonal periods noted." for i in (1, 2, 3, 4)
    ]
    adapter = _GateRejectAdapter()

    result, *_ = compress_prose(
        cast(Any, adapter),
        "\n\n".join(paras),
        schema_columns=["record_id", "metric"],
        primary_key="record_id",
    )

    # The distinctness gate rejected the dense {1..4} run; the builtin regex
    # (which happily tags "Record 1...4") must NOT override that verdict —
    # the document is processed positionally, without [RECORD_ID: markers.
    assert adapter.chunk_prompts
    assert all("[RECORD_ID:" not in p for p in adapter.chunk_prompts)
    assert "record_id: 1 | metric: one" in result


def test_sample_paragraphs_by_section_returns_small_docs_in_full() -> None:
    """Under the token budget, every paragraph enters the schema sample."""
    from agents.etl._compress import sample_paragraphs_by_section

    sections = []
    for name in ("One", "Two", "Three"):
        paras = [f"{name} para {i} " + "filler words " * 5 for i in range(4)]
        sections.append(f"### Section {name}\n\n" + "\n\n".join(paras))
    sample = sample_paragraphs_by_section("\n\n".join(sections))

    for name in ("One", "Two", "Three"):
        for i in range(4):
            assert f"{name} para {i}" in sample


def test_sample_paragraphs_by_section_prefers_section_interior() -> None:
    """Over budget, centered sampling picks section-interior paragraphs.

    Section boundaries are intro/outro narrative where field mentions are
    weakest.  Regression: task_59 ed_consumerpriceindex — boundary-biased
    samples fed the schema call date narrative only and the CPI fields
    never entered the schema.
    """
    from agents.etl._compress import sample_paragraphs_by_section

    sections = []
    for name in ("One", "Two", "Three"):
        paras = [f"{name} para {i} " + "filler words " * 5 for i in range(9)]
        sections.append(f"### Section {name}\n\n" + "\n\n".join(paras))
    sample = sample_paragraphs_by_section("\n\n".join(sections), token_limit=150)

    for name in ("One", "Two", "Three"):
        assert f"{name} para 0" not in sample  # intro bookend skipped
        assert f"{name} para 8" not in sample  # outro bookend skipped
        assert any(f"{name} para {i}" in sample for i in range(1, 8))
