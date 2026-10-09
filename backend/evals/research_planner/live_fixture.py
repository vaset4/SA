"""Deterministic live-branch fixtures for planner eval."""

from __future__ import annotations

import asyncio
import hashlib
import json
from uuid import uuid4
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Mapping
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeMaterial, KnowledgeReadReport
from app.models.research_contracts import EvidenceRecord
from app.models.research_planner_contracts import (
    DirectAnswerDraft,
    ProjectMemoryReadRequest,
    ResearchUnit,
    TurnDepthProposal,
)
from app.models.research_planner_draft_contracts import (
    ResearchClaimStrategyDraft, ResearchClaimStrategyDraftBatch, ResearchPlanStrategyDraft,
    ResearchReplanStrategyDraft,
)
from app.repositories import research_memory_repository
from app.services.knowledge_access_service import material_group_id
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials
from app.services.internal_research.research_planner_loop_state import read_decision
from app.services.internal_research.research_planner_consumer_budget import evidence_text_field
from app.services.internal_research import research_planner_pipeline
from app.services.internal_research.research_planner_pipeline import (
    analyze_planner_state,
    execute_planner_state,
    synthesize_planner_state,
    validate_planner_state,
)

from evals.research_planner.metrics import MINIMAL_DEPTH
from evals.contracts import freeze_record, read_snapshot, snapshot, validate_execution_input

PlannerRunner = Callable[..., Awaitable[dict[str, Any]]]

# 评测项目的固定设定：样本问的就是这几篇论文。深度硬约束按「本回合授权论文数」判，
# 授权范围为空时任何要求论文 Evidence 的深度都会当场越界，所以范围必须像真项目一样非空
_EVAL_PROJECT_PAPER_IDS = ("pdformer", "gman", "stgcn", "graph-wavenet", "dcrnn")
_EVAL_PROJECT_ID = "eval-project-001"
_EVAL_OWNER_USER_ID = "test-user-001"
_TERMINATE_FAIL = "terminate_fail"
_FIXTURE_BATCH_ID_CHARS = 32


def build_live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return asyncio.run(build_live_records_async(records, runner=run_planner_nodes))


async def build_live_records_async(
    records: list[dict[str, Any]],
    *,
    runner: PlannerRunner,
) -> list[dict[str, Any]]:
    """参数为案例和节点执行器；返回冻结案例、声明环境及实际观察的独立记录。"""
    live_records: list[dict[str, Any]] = []
    for record in records:
        environment = _fixture_environment(record)
        fixture = _RecordFixture(record=environment)
        input_snapshot = snapshot(fixture._initial_state())
        fixture_snapshot = snapshot(environment)
        payload = await fixture.run(runner, initial_state=read_snapshot(input_snapshot))
        frozen = freeze_record(
            record, mode="planner_live_fixture", execution_input=read_snapshot(input_snapshot),
            observation=payload, fixture=read_snapshot(fixture_snapshot),
        )
        live_records.append({**frozen, "replay": read_snapshot(snapshot(payload))})
    return live_records


def _fixture_environment(record: dict[str, Any]) -> dict[str, Any]:
    """参数为旧/新案例；返回独立替身环境，只有兼容边界读取旧 expected 环境字段。"""
    if isinstance(record.get("fixture"), dict):
        environment = read_snapshot(snapshot(record["fixture"]))
        if not isinstance(environment.get("seed"), dict) or not isinstance(environment.get("environment"), dict):
            raise ValueError("planner fixture requires seed and environment objects")
        return {**environment, "question": record.get("question", "")}
    expected = record["expected"] if isinstance(record.get("expected"), dict) else {}
    seed = record.get("replay")
    if not isinstance(seed, dict):
        raise ValueError("legacy planner fixture requires a replay seed object")
    fields = ("failure_stage", "depth", "memory_owner_user_id", "source_types")
    return read_snapshot(snapshot({
        "version": "planner-fixture-v1", "source": "legacy_expected_and_replay",
        "question": record.get("question", ""), "seed": seed,
        "environment": {key: expected[key] for key in fields if key in expected},
    }))


async def run_planner_nodes(
    state: dict[str, Any],
    *,
    llm: Any,
    settings: Any,
    adapters: Mapping[str, Any],
    optional_adapters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按图里 planner 四个节点的真实顺序跑一遍单轮，返回累积到最后的状态。

    评测要测的就是产品那条链路：深度判断、minimal 直答短路、项目记忆读取都长在
    analyze 节点里，绕开它另拼一条简化链路，测出来的不是线上会发生的事。选边判据与
    rag_node 的 route_after_planner_* 一致：硬失败就此终止、minimal 且无草案直接进
    synthesize、校验没产出可执行计划就不进执行。

    参数:
        state: 本条样本的初始状态，至少含 user_query 与 project_scope
        llm: fixture 的确定性 LLM 替身
        settings: fixture 配置，供计划单元上限与循环预算上界
        adapters: 默认启用的证据适配器
        optional_adapters: 按计划按需启用的可选适配器
    返回:
        累积状态：含 turn_depth / memory_read_result / research_plan /
        research_unit_results 与综合结果，供各项指标逐条取数
    """
    # ponytail: 单轮直跑不走 replan 回边 —— fixture 的草案是定死的，重规划结果必然相同；
    # 要测重规划得先让 fixture 按轮次给不同草案
    state = {**state, "run_id": str(uuid4())}
    state = {**state, **await analyze_planner_state(state, llm, settings)}
    if read_decision(state) == _TERMINATE_FAIL:
        return state
    # minimal 且没产草案才是真短路：正常 analyze 成功必产 planner_draft
    short_circuit = str(state.get("turn_depth") or "") == MINIMAL_DEPTH and not state.get("planner_draft")
    if not short_circuit:
        state = {**state, **validate_planner_state(state, settings)}
        if not state.get("research_plan"):
            return state
        state = {**state, **await execute_planner_state(
            state,
            llm=llm,
            settings=settings,
            adapters=adapters,
            optional_adapters=optional_adapters or {},
        )}
        if read_decision(state) == _TERMINATE_FAIL:
            return state
    return {**state, **await synthesize_planner_state(state, llm)}


@dataclass(frozen=True, kw_only=True)
class _RecordFixture:
    record: dict[str, Any]

    @property
    def stage(self) -> str:
        """无参数；返回声明环境中的故障注入阶段。"""
        return _stage(self.record)

    async def run(self, runner: PlannerRunner, *, initial_state: dict[str, Any]) -> dict[str, Any]:
        """参数为执行器和已冻结输入的副本；返回节点真实产物，不以预期终态改写结果。"""
        validate_execution_input(initial_state)
        # 记忆仓储换成按门如实过滤的内存实现：四道门与条目投影仍走产品代码，
        # 只有「库里有哪些行」是样本给定的，读取结果因此可复现又不放宽任何一道门
        context = _FixtureContext(user_id=initial_state["project_scope"]["user_id"])
        with (_fixture_memory_repository(self._memory_rows()),
              patch.object(research_planner_pipeline, "get_paper_repository", return_value=context),
              patch.object(research_planner_pipeline, "get_milvus_store", return_value=context),
              patch.object(research_planner_pipeline, "get_research_object_query_service", return_value=context)):
            return await runner(
                initial_state,
                llm=_FixtureLLM(self.record),
                settings=_FixtureSettings(),
                adapters={
                    "local_direct_text": self._direct_text_adapter,
                    "local_metadata": self._metadata_adapter,
                },
            )

    def _initial_state(self) -> dict[str, Any]:
        """按真实入图形状拼本条样本的初始状态。

        project_scope 是必须的：读记忆要靠它取归属人与项目，深度硬约束要靠它数
        本回合授权论文数，缺了这两样评测跑的就不是产品那条路。
        """
        environment = self.record["environment"]
        return {
            "user_query": self.record.get("question", ""),
            "chat_history": [],
            "project_scope": {
                "project_id": _EVAL_PROJECT_ID,
                "user_id": str(environment.get("memory_owner_user_id") or _EVAL_OWNER_USER_ID),
                "paper_ids": list(_EVAL_PROJECT_PAPER_IDS),
                "publication": {"user_id": str(environment.get("memory_owner_user_id") or _EVAL_OWNER_USER_ID),
                                "lease_id": "offline-fixture-lease", "batch_ids": ["b" * _FIXTURE_BATCH_ID_CHARS],
                                "paper_batches": {_EVAL_PROJECT_PAPER_IDS[0]: "b" * _FIXTURE_BATCH_ID_CHARS}},
            },
        }

    def _memory_rows(self) -> list[dict[str, Any]]:
        """取样本给定的记忆库行（仓储行形状，含四道门那四个轴）。"""
        replay = self.record["seed"]
        rows = replay.get("project_memory_rows")
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    async def _direct_text_adapter(self, unit: ResearchUnit, _context: Any) -> EvidenceReadResult:
        """参数为单元与隔离上下文；返回声明的模拟材料与统一读取报告，不调用真实检索。"""
        if self.stage == "execution":
            raise RuntimeError("local direct-text adapter raised retrieval scope error")
        records = tuple(_direct_record(item) for item in _result_for_unit(self.record, unit.unit_id).get("evidence_ids", []))
        return EvidenceReadResult(records, KnowledgeReadReport(operation="fixture_direct", status="complete"))

    async def _metadata_adapter(self, unit: ResearchUnit, _context: Any) -> EvidenceReadResult:
        """参数为单元与隔离上下文；返回原有元数据记录及不适用正文读取的报告。"""
        return EvidenceReadResult(
            (EvidenceRecord(evidence_id=f"metadata-{unit.unit_id}", source="inventory", layer="metadata"),),
            KnowledgeReadReport(operation="fixture_metadata", status="not_applicable"),
        )


@dataclass(frozen=True)
class _FixtureSettings:
    rag_research_planner_max_units: int = 8


@dataclass(frozen=True)
class _FixtureContext:
    """声明规划夹具的五篇模拟论文及模拟正文目录，无图谱边；作者：xxx。"""

    user_id: str

    async def get_papers_by_ids(self, paper_ids: list[str], *, user_id: str) -> list[dict[str, Any]]:
        """参数为论文集合和可信用户；返回固定夹具范围内论文，不访问业务库。"""
        if user_id != self.user_id:
            raise LookupError("fixture paper owner mismatch")
        return [{"paper_id": paper_id, "user_id": user_id, "title": paper_id,
                 "is_vectorized": paper_id == _EVAL_PROJECT_PAPER_IDS[0]}
                for paper_id in paper_ids if paper_id in _EVAL_PROJECT_PAPER_IDS]

    def for_user(self, user_id: str) -> _FixtureContext:
        """参数为读取用户；核对模拟材料归属后返回同一只读介质。"""
        if user_id != self.user_id:
            raise LookupError("fixture section owner mismatch")
        return self

    def for_snapshot(self, publication: dict[str, Any]) -> _FixtureContext:
        """参数为冻结发布身份；仅接受本夹具材料批次，不混读其它发布。"""
        if publication.get("paper_batches") != {_EVAL_PROJECT_PAPER_IDS[0]: "b" * _FIXTURE_BATCH_ID_CHARS}:
            raise ValueError("fixture publication mismatch")
        return self

    def list_section_paths(self, paper_id: str) -> list[str]:
        """参数为论文ID；返回模拟正文所在目录，其余声明论文尚未发布正文。"""
        if paper_id not in _EVAL_PROJECT_PAPER_IDS:
            raise LookupError("fixture paper outside scope")
        return ["Fixture"] if paper_id == _EVAL_PROJECT_PAPER_IDS[0] else []

    async def query_project_subgraph(self, paper_ids: list[str]) -> dict[str, Any]:
        """参数为论文集合；本夹具只声明正文材料，图谱节点和边均为零。"""
        if not set(paper_ids).issubset(_EVAL_PROJECT_PAPER_IDS):
            raise LookupError("fixture graph outside scope")
        return {"meta": {"node_count": 0, "edge_count": 0}}


@contextmanager
def _fixture_memory_repository(rows: list[dict[str, Any]]) -> Iterator[None]:
    """把记忆仓储单例临时换成内存实现，跑完还原。

    仓储没有公开的替换入口，工厂只认模块级单例，所以这里直接换那个单例。

    参数:
        rows: 本条样本给定的记忆库行
    """
    previous = research_memory_repository._repository
    research_memory_repository._repository = _FixtureMemoryRepository(rows=rows)
    try:
        yield
    finally:
        research_memory_repository._repository = previous


@dataclass(frozen=True)
class _FixtureMemoryRepository:
    """内存记忆仓储：只按调用方传进来的等值条件筛行，一道门都不放宽。

    四道资格门是读取侧作为查询条件传下来的，这里如实照筛：样本里那些别人的、
    待审的、已失效的行因此真的进不来，「泄漏率为 0」测的就是门在起作用。
    """

    rows: list[dict[str, Any]]

    async def list_questions(self, project_id: str, *, user_id: str) -> list[dict[str, Any]]:
        """参数为可信项目与用户；夹具未声明研究问题，返回实际空集合。"""
        return []

    async def get_item(self, memory_id: str, *, user_id: str, project_id: str) -> dict[str, Any] | None:
        """参数为记忆ID和可信归属；模型消费前重新读取当前夹具条目副本。"""
        return next((dict(row) for row in self.rows if row.get("memory_id") == memory_id
                     and row.get("user_id") == user_id and row.get("project_id") == project_id), None)

    async def ensure_schema(self) -> None:
        return None

    async def list_items(self, project_id: str, **filters: Any) -> list[dict[str, Any]]:
        """按项目与调用方给的等值条件筛行。

        参数:
            project_id: 归属项目
            filters: 读取侧传下来的四道门与收窄条件，None 表示该维不限
        返回:
            命中的记忆库行；顺序即样本给定顺序
        抛出:
            NotImplementedError: 用到本替身没实现的收窄维度，避免当成「没收窄」悄悄多返回
        """
        unsupported = [key for key in ("question_id", "paper_id", "tag") if filters.get(key) is not None]
        if unsupported:
            raise NotImplementedError(f"fixture memory repository has no narrowing for {unsupported}")
        wanted = {"project_id": project_id, **{key: value for key, value in filters.items() if value is not None}}
        return [row for row in self.rows if all(row.get(key) == value for key, value in wanted.items())]

    async def list_links_of_memory(self, _memory_id: str) -> list[dict[str, Any]]:
        return []


@dataclass(frozen=True)
class _FixtureLLM:
    record: dict[str, Any]

    def with_structured_output(self, schema: type[Any]) -> "_FixtureStructuredOutput":
        return _FixtureStructuredOutput(record=self.record, schema=schema)

    async def ainvoke(self, _messages: Any) -> AIMessage:
        """minimal 直答走的是纯文本调用：正文取样本 replay.answer。

        样本没给正文时照样返回空串，让产品那条「空正文如实失败」的路径暴露出来
        """
        replay = self.record["seed"]
        return AIMessage(content=str(replay.get("answer") or ""))


@dataclass(frozen=True)
class _FixtureStructuredOutput:
    record: dict[str, Any]
    schema: type[Any]

    async def ainvoke(self, _messages: Any) -> Any:
        """参数为当前节点实际提示；按请求的实时合同生成明确夹具输出，不支持的schema显式失败。"""
        if self.schema is DirectAnswerDraft:
            return DirectAnswerDraft(answer=str(self.record["seed"].get("answer") or ""), knowledge_bearing=False)
        if self.schema is ResearchPlanStrategyDraft:
            return _plan_draft(self.record)
        if self.schema is ResearchReplanStrategyDraft:
            draft = _plan_draft(self.record)
            return ResearchReplanStrategyDraft(units=draft.units)
        if self.schema is ResearchClaimStrategyDraftBatch:
            return ResearchClaimStrategyDraftBatch(claims=_claim_drafts(self.record, _unit_id_from_prompt(self.record, _messages)))
        if self.schema is TurnDepthProposal:
            return _depth_proposal(self.record)
        raise TypeError(f"unsupported fixture schema: {self.schema!r}")


def _plan_draft(record: dict[str, Any]) -> ResearchPlanStrategyDraft:
    """参数为固定模拟环境；返回当前策略schema，不向模型输出执行字段，也不修改旧种子。"""
    if _stage(record) == "planning":
        raise RuntimeError("unresolved paper aliases A/B in project scope")
    replay = record["seed"]
    plan = replay.get("research_plan") if isinstance(replay.get("research_plan"), dict) else {}
    raw_units = plan.get("units") or _fallback_units(record)
    units: list[dict[str, Any]] = []
    for index, unit in enumerate(raw_units, start=1):
        units.append(_complete_unit(unit, index, previous_units=units))
    facets = [{"facet_id": f"f{index}", "label": f"Facet {index}", "source": "explicit",
               "required": True, "required_paper_ids": [],
               "user_text_span": record["question"],
               "deliverable_kind": unit["expected_output"]["deliverable_kind"]}
              for index, unit in enumerate(units, 1)]
    if _stage(record) == "validation":
        facets.append({"facet_id": "f-missing", "label": "Missing", "source": "explicit",
                       "required": True, "deliverable_kind": "claims", "user_text_span": record["question"]})
    return ResearchPlanStrategyDraft.model_validate({
        "units": [_strategy_unit(unit) for unit in units], "requested_facets": facets, "answer_sections": [],
    })


def _strategy_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """参数为夹具明确生成的行动与产物；转换成模型策略输入，不输出内部政策副本。"""
    output = unit["expected_output"]
    intents = [{"claim_type": kind, "required": kind in output["required_claim_types"]}
               for kind in unit["claim_policy"]["allowed_claim_types"]] if output["deliverable_kind"] == "claims" else []
    return {key: unit[key] for key in ("unit_id", "goal", "research_action", "evidence_policy", "dependencies", "covered_facet_ids")} | {
        "output_goal": output["output_goal"], "deliverable_kind": output["deliverable_kind"], "claim_intents": intents,
        "output_role": unit["output_role"] if unit["evidence_policy"]["source_type"] == "inference_from_units" else None,
    }


def _complete_unit(unit: dict[str, Any], index: int, *, previous_units: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为原始模拟单元、序号和先前声明；返回新版可执行合同，不改写旧种子。"""
    source_type = str((unit.get("evidence_policy") or {}).get("source_type") or "local_direct_text")
    required_types = {"local_direct_text": ["paper_supported"], "inference_from_units": ["evidence_grounded_inference"],
                      "external_context_needed": ["external_needed"]}.get(source_type, [])
    return {
        "unit_id": unit.get("unit_id") or f"u{index}",
        "goal": unit.get("goal") or f"Unit {index}",
        "research_action": unit.get("research_action") or "collect local evidence",
        "evidence_policy": _policy(source_type, unit, index),
        "claim_policy": _claim_policy(source_type),
        "expected_output": {"output_goal": "validated planner eval finding",
                            "deliverable_kind": "claims" if required_types else "materials",
                            "required_claim_types": required_types},
        "output_role": _output_role(source_type),
        "dependencies": _dependencies(source_type, previous_units),
        "covered_facet_ids": [f"f{index}"],
    }


def _dependencies(source_type: str, previous_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """推理单元声明它依赖计划里前面那些取证单元。

    不声明依赖，推理单元的主张会因「来源单元未知」被整条拒掉、单元落成 blocked，
    多论文比较与混合目标那批样本就永远测不到推理与综合这一段。

    参数:
        source_type: 单元的取证来源
        previous_units: 计划里排在它前面的单元声明
    返回:
        依赖声明；非推理单元为空列表
    """
    if source_type != "inference_from_units":
        return []
    return [{"unit_id": unit["unit_id"], "required": True,
             "required_product": unit["expected_output"]["deliverable_kind"],
             "required_claim_types": unit["expected_output"]["required_claim_types"]} for unit in previous_units]


def _output_role(source_type: str) -> str | None:
    """按取证来源给单元的产出角色，取值对齐校验器那三条硬规则。

    推理单元不给角色会被判 missing_output_role 当场退回，草案永远进不了执行 ——
    多论文比较那批样本因此长期测不到计划与执行，只能测出「校验失败」。

    参数:
        source_type: 单元的取证来源
    返回:
        取证类单元 evidence、推理类单元 synthesis、外部背景类单元 None
    """
    if source_type == "inference_from_units":
        return "synthesis"
    if source_type == "external_context_needed":
        return None
    return "evidence"


def _fallback_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    environment = record["environment"]
    units: list[dict[str, Any]] = []
    for index, source_type in enumerate(environment.get("source_types") or [], start=1):
        units.append({
            "unit_id": f"u{index}",
            "goal": f"Fixture unit {index}",
            "research_action": "fixture planner action",
            "evidence_policy": {"source_type": source_type},
        })
    return units or [{"unit_id": "u1", "evidence_policy": {"source_type": "local_direct_text"}}]


def _policy(source_type: str, unit: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "target_paper_ids": [],
        "required_sections": [],
        "evidence_dimensions": [],
        "locator_queries": [unit.get("goal") or f"query {index}"],
        "min_citations_per_target": 1,
    }


def _claim_policy(source_type: str) -> dict[str, Any]:
    if source_type == "local_direct_text":
        return {"allowed_claim_types": ["paper_supported"], "citation_required": True}
    if source_type == "inference_from_units":
        return {"allowed_claim_types": ["evidence_grounded_inference"], "bounded_inference_allowed": True}
    return {"allowed_claim_types": ["external_needed"], "citation_required": False}


def _claim_drafts(record: dict[str, Any], unit_id: str) -> list[ResearchClaimStrategyDraft]:
    """参数为夹具和唯一匹配单元；返回该单元明确声明的实时主张选择。"""
    result = _result_for_unit(record, unit_id)
    return [_draft_from_claim(claim, unit_id, record=record) for claim in result.get("claims", []) if isinstance(claim, dict)]


def _draft_from_claim(claim: dict[str, Any], unit_id: str, *, record: dict[str, Any]) -> ResearchClaimStrategyDraft:
    """参数为模拟主张与单元；将夹具标签映射到实际模拟来源身份，不修改种子或参考标签。"""
    claim_type = str(claim.get("claim_type") or "paper_supported")
    supporting = claim.get("supporting_evidence_ids") if claim_type == "paper_supported" else []
    records = [_direct_record(label) for label in supporting or []]
    spans = []
    for evidence in records:
        field, body = evidence_text_field(evidence)
        spans.append({"evidence_id": evidence["evidence_id"],
                      "field": field, "start": 0, "end": len(body), "quote": body})
    return ResearchClaimStrategyDraft(
        text=str(claim.get("text") or "fixture claim"),
        claim_type=claim_type,
        supporting_evidence_ids=[record["evidence_id"] for record in records],
        supporting_spans=spans,
        source_claim_ids=_selected_fixture_claim_ids(record, claim),
        reasoning_summary=str(claim.get("reasoning_summary") or "Derived from fixture planner claims."),
        addressed_facet_ids=[f"f{unit_id[1:]}"] if unit_id.startswith("u") and unit_id[1:].isdigit() else [],
    )


def _selected_fixture_claim_ids(record: dict[str, Any], claim: dict[str, Any]) -> list[str]:
    """参数为旧种子和明确主张；仅单一上游主张可无歧义映射，不将整个上游默认为已选支持。"""
    if "source_claim_ids" in claim:
        return list(claim["source_claim_ids"])
    selected = []
    for unit_id in claim.get("source_unit_ids") or []:
        upstream = _result_for_unit(record, unit_id).get("claims") or []
        if len(upstream) != 1 or not isinstance(upstream[0], dict):
            raise ValueError(f"legacy fixture must select a unique upstream claim: {unit_id}")
        selected.append(f"{unit_id}.c1")
    return selected


def _result_for_unit(record: dict[str, Any], unit_id: str) -> dict[str, Any]:
    replay = record["seed"]
    for result in replay.get("research_unit_results") or []:
        if isinstance(result, dict) and result.get("unit_id") == unit_id:
            return result
    return {}


def _direct_record(evidence_id: str) -> EvidenceRecord:
    """参数为夹具来源标签；经正式转换生成新读取合同的模拟正文，不改写历史存档。"""
    label = str(evidence_id)
    paper_id = _EVAL_PROJECT_PAPER_IDS[0]
    batch_id = "b" * _FIXTURE_BATCH_ID_CHARS
    ordinal = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    chunk_id = f"v3:{batch_id}:{ordinal}"
    text = f"Fixture direct text for {label}."
    material = KnowledgeMaterial(
        paper_id=paper_id, batch_id=batch_id, group_kind="parent", group_locator=label,
        material_group_id=material_group_id(paper_id=paper_id, batch_id=batch_id, group_kind="parent", group_locator=label),
        text=text, source_spans=({"paper_id": paper_id, "batch_id": batch_id, "chunk_id": chunk_id,
                                  "field": "text", "start": 0, "end": len(text),
                                  "text_start": 0, "text_end": len(text)},),
        chunk_id=chunk_id, parent_id=label, section_path="Fixture", section="Fixture", page_num=0,
        source_location_status="legacy_chunk", block_type="paragraph",
    )
    return convert_knowledge_materials((material,))[0]


def _stage(record: dict[str, Any]) -> str:
    """参数为声明的替身环境；返回注入阶段，不读取评分 expected。"""
    return str(record["environment"].get("failure_stage") or "none")


def _unit_id_from_prompt(record: dict[str, Any], messages: Any) -> str:
    """参数为声明环境和实际任务投影；唯一匹配行动身份，缺失或含糊时明确拒绝。"""
    text = _message_text(messages)
    marker = "Unit and assigned requirements: "
    lines = [line[len(marker):] for line in text.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise ValueError("fixture claim prompt requires one task projection")
    projection = json.loads(lines[0])
    matches = [unit for unit in _plan_draft(record).units if unit.goal == projection.get("goal")
               and unit.research_action == projection.get("research_action")]
    if len(matches) != 1:
        raise ValueError("fixture claim prompt does not identify one declared action")
    return matches[0].unit_id


def _message_text(messages: Any) -> str:
    if isinstance(messages, list):
        return "\n".join(str(getattr(message, "content", message) or "") for message in messages)
    return str(getattr(messages, "content", messages) or "")


def _depth_proposal(record: dict[str, Any]) -> TurnDepthProposal:
    """按样本标注构造深度评估的结构化返回（模型这一步被替身顶掉）。

    深度取声明的 fixture 环境，测的是替身深度过完硬约束与后续链路有没有被改写、
    以及三档深度真实的执行差异；「模型判得准不准」要真调模型，确定性 replay 测不了。

    参数:
        record: 声明环境，取 environment.depth 与 environment.memory_owner_user_id
    返回:
        TurnDepthProposal：标注深度 + 与之相符的论文 Evidence 要求 + 本轮记忆读取判断
    """
    environment = record["environment"]
    depth = str(environment.get("depth") or "lookup").strip()
    # 三档深度各自该有的原因码与论文 Evidence 要求：minimal 不检索论文所以不得要求
    # 论文 Evidence，否则撞上深度评估器第一条硬约束
    reason_map = {
        "minimal": ("conversational_task", False),
        "lookup": ("single_fact_lookup", True),
        "research": ("multi_paper_synthesis", True),
    }
    reason_code, requires_paper_evidence = reason_map.get(depth, ("single_fact_lookup", True))
    return TurnDepthProposal(
        depth=depth,
        reason_code=reason_code,
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=str(record.get("question") or "fixture question"),
        project_memory=_memory_read_request(environment),
    )


def _memory_read_request(expected: Mapping[str, Any]) -> ProjectMemoryReadRequest | None:
    """标了记忆归属人的样本才要求本轮翻项目记忆。

    参数:
        expected: 样本标注，取 memory_owner_user_id
    返回:
        needed=True 的读取请求；样本没标归属人时返回 None，即本轮不翻记忆
    """
    if not str(expected.get("memory_owner_user_id") or "").strip():
        return None
    return ProjectMemoryReadRequest(needed=True, reason_code="project_history_recall")
