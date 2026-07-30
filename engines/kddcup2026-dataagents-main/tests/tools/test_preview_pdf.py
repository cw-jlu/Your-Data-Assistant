from __future__ import annotations

from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.preview import _group_pdf_paragraphs_by_entity, preview_file
from agents.tools.registry import ToolExecutionResult


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_pdf"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_pdf", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    import pymupdf  # pyright: ignore[reportMissingImports]

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        for page_lines in pages:
            page = doc.new_page()  # pyright: ignore[reportUnknownMemberType]
            y = 72
            for text in page_lines:
                page.insert_text((72, y), text, fontsize=11)  # pyright: ignore[reportUnknownMemberType]
                y += 36
        doc.save(str(path))  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]


def _preview_pdf(task: PublicTask, path: str) -> ToolExecutionResult:
    assert preview_file.input_model is not None
    args = preview_file.input_model.model_validate({"path": path})
    return preview_file.handler(task, args)


def test_preview_pdf_groups_record_entities_across_pages(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "profiles.pdf",
        [
            [
                "Record 101 management scale is",
            ],
            [
                "123.45 billion yuan.",
                "The highest degree field is Doctoral.",
                "Record 102 management scale is 88.00 billion yuan.",
                "The highest degree field is Master.",
                "Record 103 management scale is 9.50 billion yuan.",
            ],
        ],
    )

    result = _preview_pdf(task, "doc/profiles.pdf")

    assert result.ok is True
    content = result.content
    assert content["format"] == "pdf"
    assert content["page_count"] == 2
    assert content["entity_group_count"] == 3
    assert content["entity_sample_size"] == 3
    groups = {group["entity_id"]: group for group in content["entity_groups_sample"]}
    assert set(groups) == {"101", "102", "103"}
    record_101_text = "\n\n".join(paragraph["text"] for paragraph in groups["101"]["paragraphs"])
    assert "Record 101 management scale is 123.45 billion yuan." in record_101_text
    assert "The highest degree field is Doctoral." in record_101_text
    assert groups["101"]["paragraph_count"] == 2
    assert groups["101"]["paragraphs"][0]["page"] == 1
    assert groups["101"]["paragraphs"][1]["page"] == 2


def test_preview_pdf_shares_multi_entity_paragraphs(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "shared_profiles.pdf",
        [
            [
                "Record 101 has an equity fund scale of 123.45 billion yuan.",
                "Record 102 has an equity fund scale of 88.00 billion yuan.",
                "Record 101 and Record 102 have no bond fund records.",
                "This report now moves to the next asset class.",
            ]
        ],
    )

    content = _preview_pdf(task, "doc/shared_profiles.pdf").content

    groups = {group["entity_id"]: group for group in content["entity_groups_sample"]}
    assert set(groups) == {"101", "102"}
    shared = "Record 101 and Record 102 have no bond fund records."
    assert shared in "\n\n".join(paragraph["text"] for paragraph in groups["101"]["paragraphs"])
    assert shared in "\n\n".join(paragraph["text"] for paragraph in groups["102"]["paragraphs"])
    transition = "This report now moves to the next asset class."
    assert transition not in "\n\n".join(
        paragraph["text"] for paragraph in groups["101"]["paragraphs"]
    )
    assert transition not in "\n\n".join(
        paragraph["text"] for paragraph in groups["102"]["paragraphs"]
    )
    assert content["ungrouped_paragraph_count"] == 1


def test_preview_pdf_groups_patient_unit_stay_entities(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "patients.pdf",
        [
            [
                (
                    "The case file associated with patient unit stay 1688094 is "
                    "cross-referenced with PID 017-30133."
                ),
                "This patient required multiple specialist consultations.",
                (
                    "The patient registered under unit stay 504172 holds the "
                    "unique identifier 005-87465."
                ),
                "This patient followed the prescribed therapy plan closely.",
                (
                    "The admission for patient unit stay 3128357 is a female patient "
                    "who is 86 years of age."
                ),
                "This patient was transferred after clinical stabilization.",
                "The patient from unit stay 1688094 was later assigned to hospital 264.",
                "The case of patient unit stay 504172 was handled at hospital 142.",
                "The file for patient unit stay 3128357 indicates a height of 157.5 cm.",
            ]
        ],
    )

    content = _preview_pdf(task, "doc/patients.pdf").content

    groups = {group["entity_id"]: group for group in content["entity_groups_sample"]}
    assert set(groups) == {"1688094", "504172", "3128357"}
    assert content["entity_anchor_patterns"] == ["known", "auto:unit stay"]
    assert groups["1688094"]["paragraph_count"] == 3
    stay_1688094 = "\n\n".join(paragraph["text"] for paragraph in groups["1688094"]["paragraphs"])
    assert "patient unit stay 1688094" in stay_1688094
    assert "multiple specialist consultations" in stay_1688094
    assert "hospital 264" in stay_1688094


def test_preview_pdf_groups_english_entry_entities(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "diagnoses.pdf",
        [
            [
                f"The patient's diagnostic profile includes entry {10000 + index}."
                for index in range(12)
            ]
        ],
    )

    content = _preview_pdf(task, "doc/diagnoses.pdf").content

    groups = {group["entity_id"]: group for group in content["entity_groups_sample"]}
    assert content["entity_group_count"] == 12
    assert content["entity_anchor_patterns"] == ["known", "auto:entry"]
    assert set(groups) <= {str(10000 + index) for index in range(12)}


def test_preview_pdf_keeps_multiple_auto_anchor_aliases_separate(tmp_path: Path) -> None:
    paragraphs: list[dict[str, object]] = []
    for offset, record_id in enumerate(range(101, 113), start=1):
        paragraphs.append(
            {
                "paragraph_index": len(paragraphs) + 1,
                "page": 1,
                "text": f"投资标的 {record_id} 的初始描述为第 {offset} 条。",
            }
        )
    for offset, record_id in enumerate(range(101, 113), start=1):
        paragraphs.append(
            {
                "paragraph_index": len(paragraphs) + 1,
                "page": 2,
                "text": f"投资单元 {record_id} 的后续指标为第 {offset} 项。",
            }
        )

    groups, _ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert {"auto:标的", "auto:单元"} <= set(anchor_patterns)
    assert set(groups) == {str(record_id) for record_id in range(101, 113)}
    assert [paragraph["paragraph_index"] for paragraph in groups["101"]] == [1, 13]
    assert [paragraph["paragraph_index"] for paragraph in groups["102"]] == [2, 14]


def test_preview_pdf_does_not_promote_trading_codes_as_entities(tmp_path: Path) -> None:
    paragraphs = [
        {
            "paragraph_index": index,
            "page": 1,
            "text": f"该基金的市场简称为样例基金，交易代码为 {159900 + index}。",
        }
        for index in range(1, 13)
    ]

    groups, ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert anchor_patterns == ["known"]
    assert groups == {}
    assert ungrouped_count == 12


def test_preview_pdf_groups_cjk_number_then_entity_label(tmp_path: Path) -> None:
    paragraphs = [
        {
            "paragraph_index": index,
            "page": 1,
            "text": f"第 {1000 + index} 号资产项，对应产品的核心指标已完成复核。",
        }
        for index in range(1, 13)
    ]

    groups, _ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert "auto:资产项" in anchor_patterns
    assert set(groups) == {str(1000 + index) for index in range(1, 13)}


def test_preview_pdf_prefers_registry_ref_over_year_labels(tmp_path: Path) -> None:
    paragraphs = [
        {
            "paragraph_index": index,
            "page": 1,
            "text": (
                f"Regarding the year-end {2000 + index} report "
                f"(Registry Ref: {90 + index}), total savings were reviewed."
            ),
        }
        for index in range(1, 13)
    ]

    groups, _ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert "auto:registry ref" in anchor_patterns
    assert "auto:year-end" not in anchor_patterns
    assert set(groups) == {str(90 + index) for index in range(1, 13)}


def test_preview_pdf_auto_discovers_unknown_cjk_entity_label(tmp_path: Path) -> None:
    paragraph_texts = [
        "样本 101 的基金经理为甲，管理规模为 123.45 亿元。",
        "该基金经理最高学历为博士。",
        "样本 102 的基金经理为乙，管理规模为 88.00 亿元。",
        "该基金经理最高学历为硕士。",
        "样本 103 的基金经理为丙，管理规模为 9.50 亿元。",
        "该基金经理最高学历为本科。",
        "后续核查显示，样本 101 的权益基金规模为 111.00 亿元。",
        "后续核查显示，样本 102 的权益基金规模为 80.00 亿元。",
        "后续核查显示，样本 103 的权益基金规模为 7.00 亿元。",
    ]
    paragraphs = [
        {"paragraph_index": index, "page": 1, "text": text}
        for index, text in enumerate(paragraph_texts, start=1)
    ]

    groups, ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert set(groups) == {"101", "102", "103"}
    assert anchor_patterns == ["known", "auto:样本"]
    assert ungrouped_count == 0
    sample_101 = "\n\n".join(paragraph["text"] for paragraph in groups["101"])
    assert "最高学历为博士" in sample_101
    assert "权益基金规模为 111.00 亿元" in sample_101


def test_preview_pdf_discovers_auto_aliases_even_when_known_ids_exist(tmp_path: Path) -> None:
    paragraphs: list[dict[str, object]] = []
    for record_id in range(101, 113):
        paragraphs.append(
            {
                "paragraph_index": len(paragraphs) + 1,
                "page": 1,
                "text": f"档案 {record_id} 的主体身份已经核验。",
            }
        )
    for record_id in range(101, 113):
        paragraphs.append(
            {
                "paragraph_index": len(paragraphs) + 1,
                "page": 2,
                "text": f"索引 {record_id} 的补充指标已经复核。",
            }
        )

    groups, _ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert "auto:索引" in anchor_patterns
    assert [paragraph["paragraph_index"] for paragraph in groups["101"]] == [1, 13]
    assert [paragraph["paragraph_index"] for paragraph in groups["112"]] == [12, 24]


def test_preview_pdf_does_not_promote_internal_management_ids(tmp_path: Path) -> None:
    paragraphs = [
        {
            "paragraph_index": index,
            "page": 1,
            "text": f"该产品的内部管理档案编号为 {300000 + index}，仅用于后台系统同步。",
        }
        for index in range(1, 13)
    ]

    groups, ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert anchor_patterns == ["known"]
    assert groups == {}
    assert ungrouped_count == 12


def test_preview_pdf_does_not_promote_one_off_numeric_labels(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "hospitals.pdf",
        [
            [
                "The patient was admitted to hospital ID 264 for observation.",
                "Another patient was admitted to hospital ID 142 for observation.",
                "A third patient was admitted to hospital ID 420 for observation.",
            ]
        ],
    )

    content = _preview_pdf(task, "doc/hospitals.pdf").content

    assert content["entity_anchor_patterns"] == ["known"]
    assert content["entity_group_count"] == 0
    assert content["ungrouped_paragraph_count"] == 3


def test_preview_pdf_section_break_resets_entity_continuation(tmp_path: Path) -> None:
    paragraphs = [
        {
            "paragraph_index": 1,
            "page": 1,
            "text": "关于档案 21060 的审查，此条目为关于浙江越剑智能装备股份有限公司身份验证的最终记录。",
        },
        {
            "paragraph_index": 2,
            "page": 1,
            "text": "法证审查报告：目标公司资本沿革与股权交易深度剖析（续）",
        },
        {
            "paragraph_index": 3,
            "page": 1,
            "text": "在确认了各目标公司的法律与市场身份后，审查工作进入下一阶段。",
        },
        {"paragraph_index": 4, "page": 1, "text": "关于档案 21061 的审查，另一实体记录开始。"},
    ]

    groups, ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)

    assert anchor_patterns == ["known"]
    assert [paragraph["paragraph_index"] for paragraph in groups["21060"]] == [1]
    assert [paragraph["paragraph_index"] for paragraph in groups["21061"]] == [4]
    assert ungrouped_count == 2


def test_preview_pdf_samples_three_entities_deterministically(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "many_profiles.pdf",
        [[f"Record {record_id} field value is {record_id * 10}." for record_id in range(1, 6)]],
    )

    first = _preview_pdf(task, "doc/many_profiles.pdf").content
    second = _preview_pdf(task, "doc/many_profiles.pdf").content

    assert first["entity_group_count"] == 5
    assert first["entity_sample_size"] == 3
    assert len(first["entity_groups_sample"]) == 3
    assert first["entity_groups_sample"] == second["entity_groups_sample"]
    sampled_ids = {group["entity_id"] for group in first["entity_groups_sample"]}
    assert sampled_ids <= {"1", "2", "3", "4", "5"}


def test_preview_pdf_without_record_ids_returns_no_entity_groups(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "summary.pdf",
        [["This document has prose but no record identifier anchor."]],
    )

    content = _preview_pdf(task, "doc/summary.pdf").content

    assert content["format"] == "pdf"
    assert content["paragraph_count"] == 1
    assert content["entity_group_count"] == 0
    assert content["entity_sample_size"] == 0
    assert content["entity_groups_sample"] == []
    assert content["ungrouped_paragraph_count"] == 1
