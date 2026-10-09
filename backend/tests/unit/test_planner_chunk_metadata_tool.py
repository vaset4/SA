"""研究规划器按元数据筛块取证适配器测试（U3 切片 4，09-12 卡）。

覆盖点：
- `local_chunk_metadata` 进严格枚举：过既有校验、缺省补 evidence、自称最终答案被拒
- 筛块硬条件：块类型走严格枚举、页码元素下限 1
- 三个条件之间 AND、同一条件多个值 OR；章节匹配使用知识访问的共享定位规则
- 记录只回定位元数据（不含任何正文字段）、`usable_as_citation=False`、能过 `_filter_records`
- 超条数上限：按文档顺序取前 N 条 + 截断概览（含命中总数），概览不可引用
- 四态各可达：命中 success / 零命中 empty / 三条件全空 skipped / Milvus 报错 failed 且仍硬失败
- 计划没点名该类型时可选适配器不被启用，向量库零查询

作者：xxx
创建时间：2026-09-14 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import read_result

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from app.models import evidence_layers as layers
from app.models.research_planner_contracts import EvidencePolicy, ResearchClaimDraft
from app.nodes import rag_node
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    PlannerExecutionContext,
    PlannerUnitExecutor,
    _record_matches_source,
)
from app.services.internal_research.research_planner_pipeline import (
    _adapters_for_plan,
    execute_planner_state,
)
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_SOURCE = layers.PLANNER_SOURCE_LOCAL_CHUNK_METADATA
_PAPER = "upload_87b14b95"
_TITLE = "PDFormer"
_INTRO_PATH = f"{_TITLE} / 1 INTRODUCTION"
_PROBLEM_PATH = f"{_TITLE} / 3.1 PROBLEM DEFINITION"
_METHOD_PATH = f"{_TITLE} / 3.2 METHOD"
_APPENDIX_PATH = f"{_TITLE} / A.3.1 VISUALIZATIONS OF CHANNEL REPRESENTATIONS"
_CAP_SETTING = "rag_planner_chunk_metadata_max_rows"
# 真库 `list_chunk_rows_by_metadata` 的窄投影字段集：只有定位字段，没有正文字段
_ROW_FIELDS = ("chunk_id", "parent_id", "section", "section_path", "block_type", "page_num")
# 块定位记录形状（工作卡定死）：每条一个块，不含正文字段
_RECORD_FIELDS = {
    "evidence_id", "source", "layer", "paper_id", "chunk_id", "parent_id",
    "section", "page_num", "summary", "usable_as_citation", "provenance",
}


class _StubVectorStore:
    """返回预置章节路径与块行的向量库替身，记录窄查询入参、被绑定的读取身份，并照真库口径筛块。"""

    def __init__(
        self,
        paths: dict[str, list[str]] | None = None,
        rows: dict[str, list[dict]] | None = None,
    ) -> None:
        self._paths = paths or {}
        self._rows = rows or {}
        self.listed: list[str] = []
        self.filtered: list[dict[str, Any]] = []
        self.bound_user_ids: list[str] = []
        self.bound_snapshots: list[dict | None] = []

    def for_user(self, user_id: str) -> "_StubVectorStore":
        # 留痕本轮绑进来的身份，供断言筛块读库前确实带上了用户；替身无需真副本，仍返回自己
        self.bound_user_ids.append(user_id)
        return self

    def for_snapshot(self, snapshot: dict | None) -> "_StubVectorStore":
        # 留痕本轮盖章的发布快照，供断言直读沿用了同一版本；替身无需真副本，仍返回自己
        self.bound_snapshots.append(snapshot)
        return self

    def list_section_paths(self, paper_id: str) -> list[str]:
        self.listed.append(paper_id)
        return self._paths.get(paper_id, [])

    def list_chunk_rows_by_metadata(
        self,
        paper_id: str,
        *,
        section_paths: list[str] | None = None,
        block_types: list[str] | None = None,
        page_nums: list[int] | None = None,
    ) -> list[dict]:
        self.filtered.append({
            "paper_id": paper_id,
            "section_paths": list(section_paths or []),
            "block_types": list(block_types or []),
            "page_nums": list(page_nums or []),
        })
        return [
            _project(row) for row in self._rows.get(paper_id, [])
            if (not section_paths or row["section_path"] in set(section_paths))
            and (not block_types or row["block_type"] in set(block_types))
            and (not page_nums or row["page_num"] in set(page_nums))
        ]


class _FailingVectorStore(_StubVectorStore):
    """筛块时抛错的向量库替身，模拟 Milvus 基础设施故障。"""

    def list_chunk_rows_by_metadata(self, paper_id: str, **kwargs: Any) -> list[dict]:
        raise RuntimeError("milvus query failed")


def _project(row: dict[str, Any]) -> dict[str, Any]:
    """照真库窄投影只留定位字段：预置的正文字段不回给调用方，读正文是另一条路的活。"""
    return {field: row[field] for field in _ROW_FIELDS}


def _row(
    chunk_id: str,
    section_path: str,
    *,
    block_type: str = "table",
    page_num: int = 3,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "parent_id": f"p-{chunk_id}",
        "section": section_path.rsplit(" / ", 1)[-1],
        "section_path": section_path,
        "block_type": block_type,
        "page_num": page_num,
        "display_text": "block body text",
    }


def _state(paper_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "run_id": "chunk-metadata-test-run",
        "user_query": "第 3 页有哪些表格块",
        "focus": {"turn_focus": paper_ids or []},
        "project_scope": {"user_id": "u1", "paper_ids": paper_ids or [], "publication": {
            "user_id": "u1", "lease_id": "fixture-lease", "batch_ids": ["b" * 32],
            "paper_batches": {paper_id: "b" * 32 for paper_id in (paper_ids or [])},
        }},
    }


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="第 3 页有哪些表格块")


def _chunk_metadata_unit(
    *,
    sections: list[str] | None = None,
    block_types: list[str] | None = None,
    pages: list[int] | None = None,
) -> dict[str, Any]:
    unit = make_unit(
        "chunk-meta",
        goal="Locate matching chunks by metadata",
        action="filter the target paper's chunks by section, block type, and page",
        source_type=_SOURCE,
        allowed_claims=["paper_supported"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )
    policy = unit["evidence_policy"]
    policy["required_sections"] = [] if sections is None else sections
    policy["required_block_types"] = ["table"] if block_types is None else block_types
    policy["required_pages"] = [3] if pages is None else pages
    policy["locator_queries"] = []
    policy["min_citations_per_target"] = 0
    unit["claim_policy"]["bounded_inference_allowed"] = True
    unit["expected_output"].update({"deliverable_kind": "materials", "required_claim_types": []})
    return unit


def _metadata_unit() -> dict[str, Any]:
    unit = make_unit(
        "meta",
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _plan(units: list[dict[str, Any]]) -> Any:
    facet = {**make_facet("f1"), "deliverable_kind": units[0]["expected_output"]["deliverable_kind"]}
    result = validate_strategy_fixture(make_draft(units, explicit_facets=[facet]))
    assert result.plan is not None
    return result.plan


def _run_plan(units: list[dict[str, Any]], store: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """用真实可选适配器执行计划，返回 (plan, report)。"""
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan(units)
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))
    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    return plan, report


def _record_with(report: Any, provenance: str) -> dict[str, Any]:
    return next(item for item in report.evidence_records if item.get("provenance") == provenance)


def test_chunk_metadata_source_type_passes_existing_validation() -> None:
    result = validate_strategy_fixture(make_draft([_chunk_metadata_unit()], explicit_facets=[{
        **make_facet("f1"), "deliverable_kind": "materials",
    }]))

    assert result.failure is None
    assert result.plan is not None
    unit = result.plan.units[0]
    # 1. 新类型进严格枚举后过既有校验，缺省 output_role 按本地取证单元补成 evidence
    assert unit.evidence_policy.source_type == _SOURCE
    assert unit.output_role == "evidence"
    # 2. 纯材料策略不声明主张，因此不产生任何主张或引用许可
    assert unit.claim_policy.citation_required is False
    assert unit.claim_policy.allowed_claim_types == []
    # 3. 筛块硬条件在契约里原样保留
    assert unit.evidence_policy.required_block_types == ["table"]
    assert unit.evidence_policy.required_pages == [3]


def test_chunk_metadata_unit_cannot_claim_final_answer_role() -> None:
    unit = _chunk_metadata_unit()
    unit["output_role"] = "final_recommendation"

    result = validate_strategy_fixture(make_draft([unit], explicit_facets=[make_facet("f1")]))

    # 筛块记录不可引用、也当不了结论来源，与既有本地来源同一套门
    assert result.failure is not None
    assert result.failure.code == "invalid_plan_format"


def test_required_pages_must_be_positive() -> None:
    policy = EvidencePolicy(source_type=_SOURCE, required_pages=[3])
    assert policy.required_pages == [3]

    with pytest.raises(ValidationError):
        EvidencePolicy(source_type=_SOURCE, required_pages=[0])
    with pytest.raises(ValidationError):
        EvidencePolicy(source_type=_SOURCE, required_block_types=["chart"])


def test_chunk_metadata_records_carry_locator_fields_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StubVectorStore(
        {_PAPER: [_PROBLEM_PATH]},
        {_PAPER: [_row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH, page_num=3)]},
    )

    _, report = _run_plan([_chunk_metadata_unit(sections=["3.1"])], store, monkeypatch)

    # 1. 命中即 success，记录进单元证据；字段形状与工作卡定死的一致，没有任何正文字段
    assert report.capability_results == {_SOURCE: "success"}
    record = _record_with(report, "chunk_metadata_filter")
    assert set(record) == _RECORD_FIELDS
    assert not {"display_text", "full_text", "embedding_text", "excerpt"} & set(record)
    # 2. 定位元数据不可引用，靠 layer + source 过执行层过滤（按 layer 筛会误收论文级元数据）
    assert record["usable_as_citation"] is False
    assert record["layer"] == layers.EVIDENCE_LAYER_METADATA
    assert record["source"] == layers.EVIDENCE_SOURCE_MILVUS
    assert _record_matches_source(record, _SOURCE) is True
    assert _record_matches_source(
        {"layer": layers.EVIDENCE_LAYER_METADATA, "source": layers.EVIDENCE_SOURCE_INVENTORY},
        _SOURCE,
    ) is False
    assert record["evidence_id"] in report.results[0].evidence_ids
    # 3. summary 给得出定位：页码、块类型、章节路径都在，section 取块行的 section 列
    assert record["summary"] == f"第 3 页 · table · {_PROBLEM_PATH}"
    assert record["section"] == "3.1 PROBLEM DEFINITION"
    assert record["chunk_id"] == "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0"
    assert record["parent_id"] == "p-v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0"
    assert record["page_num"] == 3


def test_filters_combine_and_within_field_or(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore(
        {_PAPER: [_INTRO_PATH, _PROBLEM_PATH, _METHOD_PATH]},
        {
            _PAPER: [
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH, block_type="table", page_num=3),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1", _PROBLEM_PATH, block_type="paragraph", page_num=3),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:2", _METHOD_PATH, block_type="table", page_num=4),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:3", _INTRO_PATH, block_type="table", page_num=3),
            ]
        },
    )

    _, report = _run_plan(
        [_chunk_metadata_unit(sections=["3.1", "3.2"], block_types=["table"], pages=[3])],
        store,
        monkeypatch,
    )

    # 1. 条件之间 AND、同一条件多个值 OR：只有 3.1 节的第 3 页表格块命中
    assert store.filtered == [{
        "paper_id": _PAPER,
        "section_paths": [_PROBLEM_PATH, _METHOD_PATH],
        "block_types": ["table"],
        "page_nums": [3],
    }]
    # 2. 筛块前先绑本轮用户身份：论文 ID 来自计划草案，缺身份仓储层的归属谓词会整条跳过
    assert store.bound_user_ids == ["u1"]
    records = [item for item in report.evidence_records if item["provenance"] == "chunk_metadata_filter"]
    assert [record["chunk_id"] for record in records] == ["v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0"]


def test_section_number_needle_reuses_slice3_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore(
        {_PAPER: [_PROBLEM_PATH, _APPENDIX_PATH]},
        {
            _PAPER: [
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1", _APPENDIX_PATH),
            ]
        },
    )

    _, report = _run_plan(
        [_chunk_metadata_unit(sections=["3.1"], block_types=[], pages=[])], store, monkeypatch,
    )

    # 2. 纯编号按末段前缀匹配：3.1 只落在正文节上，附录 A.3.1 不参与筛块
    assert store.filtered[0]["section_paths"] == [_PROBLEM_PATH]
    records = [item for item in report.evidence_records if item["provenance"] == "chunk_metadata_filter"]
    assert [record["chunk_id"] for record in records] == ["v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0"]


def test_no_filter_conditions_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore({_PAPER: [_PROBLEM_PATH]}, {_PAPER: [_row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH)]})

    _, report = _run_plan(
        [_chunk_metadata_unit(sections=[], block_types=[], pages=[])], store, monkeypatch,
    )

    # 三个条件全空等于计划没给筛法：适配器在册也不调用，如实记 skipped 且向量库零查询
    assert store.listed == []
    assert store.filtered == []
    assert report.capability_results == {_SOURCE: "skipped"}
    assert report.results[0].status == "blocked"
    assert report.results[0].unresolved_reason == (
        "local_chunk_metadata requires a section, block type, or page filter"
    )


def test_no_match_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore(
        {_PAPER: [_PROBLEM_PATH]},
        {_PAPER: [_row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH, block_type="paragraph")]},
    )

    _, report = _run_plan([_chunk_metadata_unit(sections=["3.1"])], store, monkeypatch)

    # 查询成功但条件一条都不命中是如实的空：不冒充成功，也不产截断概览
    assert store.filtered != []
    assert report.evidence_records == []
    assert report.capability_results == {_SOURCE: "empty"}
    assert report.results[0].status == "insufficient_evidence"


def test_cap_keeps_first_rows_and_reports_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StubVectorStore(
        {_PAPER: [_PROBLEM_PATH]},
        {
            _PAPER: [
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:3", _PROBLEM_PATH, page_num=3),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1", _PROBLEM_PATH, page_num=3),
                _row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:2", _PROBLEM_PATH, page_num=3),
            ]
        },
    )
    monkeypatch.setattr(rag_node.settings, _CAP_SETTING, 2)

    _, report = _run_plan([_chunk_metadata_unit(sections=["3.1"])], store, monkeypatch)

    # 1. 超上限时按文档顺序（入库序号）取前 N 条，不打乱既有块序
    records = [item for item in report.evidence_records if item["provenance"] == "chunk_metadata_filter"]
    assert [record["chunk_id"] for record in records] == ["v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1", "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:2"]
    # 2. 未返回定位进入读取报告，不计为证据记录
    read_report = report.results[0].metadata["read_reports"][0]
    assert read_report["status"] == "partial"
    assert read_report["remaining"] == ({"paper_id": _PAPER, "chunk_id": "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:3", "reason": "metadata_row_limit"},)
    assert len(report.evidence_records) == 2


def test_chunk_metadata_milvus_failure_is_failed_and_still_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FailingVectorStore({_PAPER: [_PROBLEM_PATH]}, {})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_chunk_metadata_unit(sections=["3.1"])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))

    # 1. 适配器不吞错：原始报错换成带取证面的显式异常上抛
    with pytest.raises(EvidenceCapabilityError) as caught:
        asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    assert caught.value.source_type == _SOURCE
    assert caught.value.reason == "RuntimeError: milvus query failed"
    # 2. 到执行层仍是既有硬失败路径，四态记 failed 而不是被吞成 empty
    state = {**_state([_PAPER]), "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}
    payload = asyncio.run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    assert payload["error_code"] == "planner_execution_failed"
    assert payload["planner_loop_decision"] == "terminate_fail"
    assert payload["planner_evidence_capabilities"] == {_SOURCE: "failed"}


def test_missing_reader_identity_is_failed_not_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore({_PAPER: [_PROBLEM_PATH]}, {_PAPER: [_row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH)]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_chunk_metadata_unit(sections=["3.1"])])
    scopeless = {**_state([_PAPER]), "project_scope": {"paper_ids": [_PAPER]}}
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(scopeless))

    # 1. 缺身份时论文归属无从校验：带取证面的失败上抛，向量库一次都不查
    with pytest.raises(EvidenceCapabilityError) as caught:
        asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    assert store.listed == []
    assert store.filtered == []
    assert store.bound_user_ids == []
    assert caught.value.source_type == _SOURCE
    assert "读取身份" in caught.value.reason
    # 2. 到执行层走既有硬失败路径，四态记 failed 而不是被降级成 empty 冒充「查得到但是空」
    state = {**_state([_PAPER]), "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}
    payload = asyncio.run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    assert payload["error_code"] == "planner_execution_failed"
    assert payload["planner_loop_decision"] == "terminate_fail"
    assert payload["planner_evidence_capabilities"] == {_SOURCE: "failed"}


def test_adapter_not_selected_when_plan_does_not_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore({_PAPER: [_PROBLEM_PATH]}, {_PAPER: [_row("v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", _PROBLEM_PATH)]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_metadata_unit()])
    # 计划单元的论文详情取数用替身：本用例只盯「没点名的 chunk_metadata 不被启用」，
    # 详情这边不该真去读论文库
    adapters = _adapters_for_plan(
        plan,
        {},
        {**rag_node._planner_optional_adapters(_state([_PAPER])), "local_metadata": _metadata_adapter()},
    )

    assert _SOURCE not in adapters
    report = asyncio.run(
        PlannerUnitExecutor(
            adapters=adapters,
            reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
        ).execute_with_evidence(plan, _ctx())
    )

    # 1. 【知识取证】【定位资料】元信息仍用于定位，不能给正文推断授予完整证据状态
    assert report.results[0].status == "insufficient_evidence"
    assert store.listed == []
    assert store.filtered == []


def test_schema_keeps_chunk_metadata_filter_fields() -> None:
    """无参数；正式合同保留筛块条件，拒绝未知字段而不是清洗后通过。"""
    from pydantic import ValidationError
    from app.models.research_planner_contracts import EvidencePolicy

    payload = {
        "source_type": _SOURCE,
        "target_paper_ids": [_PAPER],
        "required_sections": ["3.1"],
        "required_block_types": ["table"],
        "required_pages": [3],
    }
    policy = EvidencePolicy.model_validate(payload, strict=True)
    assert policy.required_block_types == ["table"]
    assert policy.required_pages == [3]
    with pytest.raises(ValidationError):
        EvidencePolicy.model_validate({**payload, "unknown_filter": "x"}, strict=True)


def _metadata_adapter() -> Callable[..., Any]:
    async def run(unit, context) -> list[dict[str, Any]]:
        return read_result([
            {
                "evidence_id": f"metadata-{_PAPER}",
                "source": layers.EVIDENCE_SOURCE_INVENTORY,
                "layer": layers.EVIDENCE_LAYER_METADATA,
                "paper_id": _PAPER,
                "summary": "PDFormer",
                "usable_as_citation": False,
            }
        ])

    return run


class _StubReasoner:
    """把过滤后的证据原样编成一条推断草案，记录每次拿到的证据 ID。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, unit, evidence, dependency_results, *, read_reports=None, prepared_input=None) -> list[ResearchClaimDraft]:
        """参数为实际证据及已准备输入；返回本测试的固定草案，不调用模型。"""
        evidence_ids = [str(record.get("evidence_id")) for record in evidence]
        self.calls.append(evidence_ids)
        return [
            ResearchClaimDraft(
                text="The paper metadata was located.",
                claim_type="evidence_grounded_inference",
                supporting_evidence_ids=evidence_ids,
                reasoning_summary="Paper metadata came from the inventory record.",
            )
        ]


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> Any:
        raise AssertionError("capability failure must not reach claim drafting")
