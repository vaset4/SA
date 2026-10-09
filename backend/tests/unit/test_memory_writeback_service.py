"""回合记忆写回与回合终态的单元测试。

覆盖四态决策表（skipped/empty/failed/success 互不归并）、prd §7.2 候选政策逐条筛选、
回合终态字段来源，以及回合级字段经真实编译图后仍能穿过两个 State 入口。

作者：xxx
2026-08-30 18:10:00
"""

from __future__ import annotations

from typing import Any
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.graph import END, START, StateGraph

from app.models.models import AgentState, RAGState
from app.services.internal_research import research_memory_writeback as writeback
from app.services.rag_trace_payload import TURN_LEVEL_STATE_KEYS, build_write_back_payload
from app.models.research_planner_contracts import ResearchClaim
from app.services.internal_research.research_planner_final_contracts import FinalDeliveryReport
from research_planner_test_utils import make_draft, make_unit, result_with_exposure, supporting_spans
from research_planner_strategy_test_utils import current_plan_fixture


@pytest.fixture(autouse=True)
def _isolated_source_boundary(monkeypatch):
    """参数为补丁工具；隔离原文读取端口，候选支持闭包、权限调用及模型发送顺序仍用正式代码。"""
    from app.services import memory_evidence_service

    async def validate(ref, *, user_id, project_id):
        """参数为支持引用和可信归属；确认写回传入原始论文和精确范围，不访问真实资料。"""
        assert (user_id, project_id) == ("user-1", "proj-1")
        assert ref["paper_id"] == "paper-a"
        return ref

    monkeypatch.setattr(memory_evidence_service, "validate_memory_evidence", validate)


class _FakeStructuredLLM:
    """替身模型：按给定提议返回结构化批次，并留下收到的提示供断言。"""

    def __init__(self, batch: writeback.TurnMemoryProposalBatch) -> None:
        self.batch = batch
        self.prompts: list[str] = []

    def with_structured_output(self, schema: Any) -> "_FakeStructuredLLM":
        assert schema is writeback.TurnMemoryProposalBatch
        return self

    async def ainvoke(self, messages: list[Any]) -> writeback.TurnMemoryProposalBatch:
        self.prompts.append(messages[0].content)
        return self.batch


def _proposal(**overrides: Any) -> writeback.TurnMemoryProposal:
    payload: dict[str, Any] = {
        "source_claim_id": "c-1",
        "title": "STGCN 的时空卷积在本项目数据上更稳",
        "body": "两篇对照实验都指向同一结论",
        "memory_type": "comparison_note",
        "is_synthesis": False,
        "durable_project_value": True,
        "reusable_judgment": True,
        "transient_or_lookup_detail": False,
    }
    payload.update(overrides)
    return writeback.TurnMemoryProposal(**payload)


def _batch(*proposals: writeback.TurnMemoryProposal, reason: str = "") -> writeback.TurnMemoryProposalBatch:
    return writeback.TurnMemoryProposalBatch(proposals=list(proposals), not_suitable_reason=reason)


def _research_state(**overrides: Any) -> dict[str, Any]:
    """一条正常收敛的 research 回合状态：已验证判断 + 本轮 Evidence + 定稿答案齐全。"""
    state: dict[str, Any] = {
        "user_query": "这两篇的实验设置差别在哪",
        "answer": "两篇的实验设置差别在采样率与骨架图构造。",
        "turn_depth": "research",
        "turn_depth_assessment": {
            "depth": "research",
            "reason_code": "cross_paper_comparison_required",
            "needs_project_memory": True,
            "memory_read_reason_code": "prior_judgment_needed",
        },
        "project_scope": {"project_id": "proj-1", "user_id": "user-1", "paper_ids": ["paper-a"]},
        "memory_read_result": {"status": "success", "items": [{"memory_id": "mem-1"}], "excluded_counts": {}},
        "source_status": {"planner": "ok", "planner_final": "ok"},
        "planner_loop_decision": "",
        # 计数是纯观察记录：取消四个额度字段后，这里只留实际步数
        "planner_budget": {
            "total_steps": 4,
        },
        "planner_final_blocks": [
            {"role": "finding", "text": "差别在采样率", "evidence_ids": ["ev-1", "ev-1", "ev-2"]},
        ],
        "missing_evidence": [],
        "evidence_records": [
            {"evidence_id": "ev-1", "paper_id": "paper-a", "chunk_id": "chunk-1"},
            {"evidence_id": "ev-2", "paper_id": "paper-b", "chunk_id": "chunk-2"},
        ],
        "research_unit_results": [
            {
                "unit_id": "u-1",
                "status": "evidence_found",
                "evidence_ids": ["ev-1"],
                "claims": [
                    {
                        "claim_id": "c-1",
                        "text": "论文 A 的采样率是 5 分钟",
                        "claim_type": "paper_supported",
                        "validation_status": "validated",
                        "supporting_evidence_ids": ["ev-1"],
                    },
                    {
                        "claim_id": "c-unvalidated",
                        "text": "论文 A 可能在夜间数据上更差",
                        "claim_type": "model_prior_hypothesis",
                        "validation_status": "needs_validation",
                    },
                ],
            }
        ],
    }
    state.update(overrides)
    return state


def _patch_persist(monkeypatch: pytest.MonkeyPatch, *, memory_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """替换落库通路，记录调用参数；memory_ids 为空表示一条都没落住。"""
    calls: list[dict[str, Any]] = []

    async def fake_persist_candidates(
        project_id,
        candidates,
        *,
        user_id,
        source_answer_id,
        source_conversation_id=None,
        source_session_id=None,
        source_delivery_revision=None,
    ):
        calls.append({
            "project_id": project_id,
            "candidates": candidates,
            "user_id": user_id,
            "source_answer_id": source_answer_id,
            "source_conversation_id": source_conversation_id,
            "source_session_id": source_session_id,
            "source_delivery_revision": source_delivery_revision,
        })
        return [{"item": {"memory_id": memory_id}} for memory_id in memory_ids or []]

    monkeypatch.setattr(writeback, "persist_candidates", fake_persist_candidates)
    return calls


def _supported_state(**overrides: Any) -> dict[str, Any]:
    """参数为场景覆盖；建立有实际暴露范围、原文坐标与O4支持事实的已提交回合。"""
    plan = current_plan_fixture(make_draft([make_unit("u-1", paper_ids=["paper-a"])]))
    text = "论文 A 的采样率是 5 分钟"
    record = {"evidence_id": "ev-1", "paper_id": "paper-a", "chunk_id": "chunk-1", "layer": "direct_text",
              "usable_as_citation": True, "excerpt": text, "source_spans": [{"paper_id": "paper-a", "chunk_id": "chunk-1",
              "field": "full_text", "start": 10, "end": 10 + len(text), "text_start": 0, "text_end": len(text)}]}
    claim = ResearchClaim(claim_id="c-1", text=text, claim_type="paper_supported", validation_status="validated",
        supporting_evidence_ids=["ev-1"], supporting_spans=supporting_spans([record]), addressed_facet_ids=["f1"])
    pending = ResearchClaim(claim_id="c-unvalidated", text="论文 A 可能在夜间数据上更差",
                            claim_type="model_prior_hypothesis", validation_status="needs_validation")
    result = result_with_exposure(plan.units[0], [record], [claim, pending])
    report = FinalDeliveryReport(attempt_id="final-1", entrypoint="normal", status="complete", generation_status="completed")
    return {**_research_state(), "research_plan": plan.model_dump(mode="json"),
            "memory_read_result": {"status": "empty", "items": [], "excluded_counts": {}},
            "research_unit_results": [result.model_dump(mode="json")], "evidence_records": [record],
            "final_generation_attempt": {"attempt_id": "final-1"}, "final_delivery_report": report.model_dump(mode="json"),
            **overrides}


async def _write_committed(monkeypatch, state: dict, *, request_id: str, llm: Any) -> dict:
    """参数为已保存状态与模型替身；走正式提交后入口，只隔离运行仓储和候选状态持久化。"""
    from app.repositories import rag_run_repository, result_delivery_repository

    run = {"run_id": request_id, "user_id": "user-1", "answer": state["answer"], "metadata": {
        **deepcopy(state), "delivery_revision": 1, "memory_writeback": {"status": "skipped", "reason_codes": ["awaiting_result_commit"]}}}

    async def get_run(**identity):
        """参数为可信查询身份；回读服务器已经提交的完整版本，不接收浏览器正文。"""
        assert identity == {"user_id": "user-1", "run_id": request_id}
        return deepcopy(run)

    async def record(**payload):
        """参数为独立候选处理状态；核对目标版本并更新，始终保留原已交付正文。"""
        assert payload["delivery_revision"] == 1
        run["metadata"]["memory_writeback"] = deepcopy(payload["outcome"])
        return True

    monkeypatch.setattr(rag_run_repository, "get_rag_run_repository", lambda: SimpleNamespace(get_run=get_run))
    monkeypatch.setattr(result_delivery_repository, "ResultDeliveryRepository", lambda: SimpleNamespace(record_memory_writeback=record))
    monkeypatch.setattr(writeback, "_curator_llm", lambda: llm)
    result = await writeback.after_result_committed({"run": deepcopy(run), "delivery_revision": 1})
    assert run["answer"] == state["answer"]
    assert run["metadata"]["memory_writeback"] == result
    return result


# ─── 深度闸门：非 research 深度压根不评估，原因码要说出真实深度 ───


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["minimal", "lookup"])
async def test_non_research_depth_skips_with_depth_reason(monkeypatch, depth) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal()))
    _patch_persist(monkeypatch, memory_ids=["mem-x"])

    result = await _write_committed(monkeypatch,
        _supported_state(turn_depth=depth), request_id="req-1", llm=llm
    )

    assert result["status"] == "skipped"
    assert result["reason_codes"] == [f"skipped_depth_{depth}"]
    assert result["candidate_ids"] == []
    # 深度没触发时不该调模型，也不该落库
    assert llm.prompts == []


# ─── 终态闸门：没正常收敛的回合记 skipped 而不是 empty ───


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"planner_loop_decision": "terminate_fail"}, "skipped_terminal_terminate_fail"),
        ({"planner_loop_decision": "terminate_honest"}, "skipped_terminal_terminate_honest"),
        ({"error_code": "evidence_not_found_in_scope"}, "skipped_terminal_error_evidence_not_found_in_scope"),
        ({"source_status": {"planner": "insufficient"}}, "skipped_terminal_evidence_insufficient"),
        ({"answer": ""}, "skipped_terminal_no_final_answer"),
        ({"research_unit_results": []}, "skipped_terminal_no_validated_claim"),
    ],
)
async def test_non_converged_turn_skips_with_terminal_reason(monkeypatch, overrides, expected) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal()))
    _patch_persist(monkeypatch, memory_ids=["mem-x"])

    result = await _write_committed(monkeypatch,
        _supported_state(**overrides), request_id="req-1", llm=llm
    )

    # 终态不合格 ≠ 评估过但没东西可记，这两态归并就分不清写回有没有跑
    assert result["status"] == "skipped"
    assert result["status"] != "empty"
    assert result["reason_codes"] == [expected]
    assert llm.prompts == []


# ─── 候选政策：prd §7.2 四条特征逐条淘汰，一条都不够格只能是 empty ───


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"durable_project_value": False}, "no_durable_project_value"),
        ({"reusable_judgment": False}, "no_reusable_judgment"),
        ({"transient_or_lookup_detail": True}, "transient_wording_or_lookup_detail"),
        ({"source_claim_id": "c-not-in-turn"}, "proposal_not_linked_to_validated_claim"),
        ({"source_claim_id": "c-unvalidated"}, "proposal_not_linked_to_validated_claim"),
    ],
)
async def test_policy_rejection_returns_empty_with_reason(monkeypatch, overrides, expected) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal(**overrides)))
    calls = _patch_persist(monkeypatch, memory_ids=["mem-x"])

    result = await _write_committed(monkeypatch,
        _supported_state(), request_id="req-1", llm=llm
    )

    assert result["status"] == "empty"
    assert result["reason_codes"] == [expected]
    assert result["candidate_ids"] == []
    # 空候选绝不落库，也绝不冒充 success
    assert calls == []


@pytest.mark.asyncio
async def test_untraceable_claim_cannot_gain_credit_by_changing_memory_type(monkeypatch) -> None:
    """O4支持断裂不能通过改名项目决策绕过；两种提议均不调用候选模型。"""
    state = _supported_state(evidence_records=[{"evidence_id": "ev-other"}])
    _patch_persist(monkeypatch, memory_ids=["mem-1"])

    rejected = await _write_committed(monkeypatch,
        state, request_id="req-1", llm=_FakeStructuredLLM(_batch(_proposal()))
    )
    accepted = await _write_committed(monkeypatch,
        state,
        request_id="req-1",
        llm=_FakeStructuredLLM(_batch(_proposal(memory_type="decision_note"))),
    )

    assert rejected["status"] == accepted["status"] == "skipped"
    assert rejected["reason_codes"] == ["skipped_terminal_no_validated_claim"]


@pytest.mark.asyncio
async def test_no_proposal_returns_empty_not_success(monkeypatch) -> None:
    llm = _FakeStructuredLLM(_batch(reason="本轮只是确认既有结论"))
    _patch_persist(monkeypatch, memory_ids=["mem-x"])

    result = await _write_committed(monkeypatch, _supported_state(), request_id="req-1", llm=llm)

    assert result["status"] == "empty"
    assert result["reason_codes"] == ["no_candidate_proposed"]


# ─── 成功路径与提示输入边界 ───


@pytest.mark.asyncio
async def test_research_turn_success_persists_with_request_id_identity(monkeypatch) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal(), _proposal(source_claim_id="c-1", title="第二条判断")))
    calls = _patch_persist(monkeypatch, memory_ids=["mem-1", "mem-2"])

    result = await _write_committed(monkeypatch, _supported_state(), request_id="req-42", llm=llm)

    assert result["status"] == "success"
    assert result["candidate_ids"] == ["mem-1", "mem-2"]
    assert result["reason_codes"] == ["validated_claim_with_evidence"]
    assert result["retryable"] is False
    # 来源身份使用已提交 run_id，落库归属取自项目范围
    assert calls[0]["source_answer_id"] == "req-42"
    assert calls[0]["source_delivery_revision"] == 1
    assert calls[0]["project_id"] == "proj-1"
    assert calls[0]["user_id"] == "user-1"
    # 候选一律走 curator 契约，三轴草稿形状不由写回决定
    assert [candidate.title for candidate in calls[0]["candidates"]] == [
        "STGCN 的时空卷积在本项目数据上更稳",
        "第二条判断",
    ]
    candidate = calls[0]["candidates"][0]
    assert candidate.source_claim_id == "c-1" and candidate.source_attempt_id == "final-1"
    assert [(ref.paper_id, ref.quote, ref.source_start) for ref in candidate.evidence] == [
        ("paper-a", "论文 A 的采样率是 5 分钟", 10),
    ]


@pytest.mark.asyncio
async def test_prompt_only_exposes_validated_claims(monkeypatch) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal()))
    _patch_persist(monkeypatch, memory_ids=["mem-1"])

    await _write_committed(monkeypatch, _supported_state(), request_id="req-1", llm=llm)

    prompt = llm.prompts[0]
    assert "claim_id=c-1" in prompt
    # 未验证的猜测不进提示，模型没机会在它上面提议
    assert "c-unvalidated" not in prompt
    assert "论文 A 可能在夜间数据上更差" not in prompt
    # 政策四条原文逐条进提示
    assert "对当前项目后续研究有持续价值" in prompt
    assert "具有清晰、可复用的认识或决定" in prompt
    assert "能追溯到本轮回答和真实 Evidence，或明确标为用户项目决策" in prompt
    assert "不是寒暄、临时措辞、单次 lookup 细节或尚未验证的猜测" in prompt


@pytest.mark.asyncio
async def test_candidate_count_capped(monkeypatch) -> None:
    proposals = [_proposal(title=f"判断 {index}") for index in range(5)]
    llm = _FakeStructuredLLM(_batch(*proposals))
    calls = _patch_persist(monkeypatch, memory_ids=["mem-1", "mem-2", "mem-3"])

    await _write_committed(monkeypatch, _supported_state(), request_id="req-1", llm=llm)

    assert len(calls[0]["candidates"]) == writeback._MAX_WRITEBACK_CANDIDATES


# ─── 失败路径：存储/模型出问题落 failed，答案不受影响 ───


@pytest.mark.asyncio
async def test_persistence_failure_returns_failed_and_retryable(monkeypatch) -> None:
    async def exploding_persist(*args: Any, **kwargs: Any):
        raise TimeoutError("memory store timeout")

    monkeypatch.setattr(writeback, "persist_candidates", exploding_persist)

    result = await _write_committed(monkeypatch,
        _supported_state(), request_id="req-1", llm=_FakeStructuredLLM(_batch(_proposal()))
    )

    assert result["status"] == "failed"
    assert result["retryable"] is True
    assert result["error_kind"] == "TimeoutError"
    assert result["candidate_ids"] == []


@pytest.mark.asyncio
async def test_model_failure_returns_failed_with_real_error_kind(monkeypatch) -> None:
    class _ExplodingLLM(_FakeStructuredLLM):
        async def ainvoke(self, messages: list[Any]):
            raise ValueError("structured output rejected")

    _patch_persist(monkeypatch, memory_ids=["mem-1"])

    result = await _write_committed(monkeypatch,
        _supported_state(), request_id="req-1", llm=_ExplodingLLM(_batch())
    )

    assert result["status"] == "failed"
    assert result["error_kind"] == "ValueError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,request_id",
    [
        ({"project_scope": {"user_id": "user-1"}}, "req-1"),
        ({"project_scope": {"project_id": "proj-1"}}, "req-1"),
        ({}, ""),
    ],
)
async def test_missing_identity_fails_without_retry(monkeypatch, overrides, request_id) -> None:
    llm = _FakeStructuredLLM(_batch(_proposal()))
    _patch_persist(monkeypatch, memory_ids=["mem-1"])

    result = await _write_committed(monkeypatch,
        _supported_state(**overrides), request_id=request_id, llm=llm
    )

    assert result["status"] == "failed"
    assert result["error_kind"] == "MissingWritebackIdentity"
    assert result["retryable"] is False
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_no_persisted_candidates_returns_empty(monkeypatch) -> None:
    """存储端返回零候选只能报empty，不能凭有效提议数量宣称保存成功。"""
    calls = _patch_persist(monkeypatch, memory_ids=[])

    result = await _write_committed(monkeypatch,
        _supported_state(),
        request_id="req-1",
        llm=_FakeStructuredLLM(_batch(_proposal())),
    )

    assert result["status"] == "empty"
    assert result["reason_codes"] == ["candidate_rejected_by_memory_type"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_invalid_candidate_type_is_visible_failure_after_commit(monkeypatch) -> None:
    """已提交回答不撤销，真实curator拒绝无效类型时记录候选失败而非伪造空成功。"""
    from app.services import research_memory_curator_service as curator

    repository = SimpleNamespace(ensure_schema=AsyncMock(), create_candidate=AsyncMock())
    monkeypatch.setattr(curator, "get_research_memory_repository", lambda: repository)
    result = await _write_committed(monkeypatch, _supported_state(), request_id="saved-run",
        llm=_FakeStructuredLLM(_batch(_proposal(memory_type="不存在的类型"))))
    assert result["status"] == "failed" and result["error_kind"] == "ValueError"
    repository.create_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_source_is_not_sent_to_candidate_model(monkeypatch) -> None:
    """已提交后来源被撤权时，候选发送前重新核验失败，不能把受限片段再发给模型。"""
    from app.services import memory_evidence_service

    monkeypatch.setattr(memory_evidence_service, "validate_memory_evidence", AsyncMock(side_effect=LookupError("项目已撤回论文")))
    llm = _FakeStructuredLLM(_batch(_proposal()))
    calls = _patch_persist(monkeypatch, memory_ids=["must-not-exist"])
    result = await _write_committed(monkeypatch, _supported_state(), request_id="saved-run", llm=llm)
    assert result["status"] == "failed" and result["error_kind"] == "LookupError"
    assert llm.prompts == [] and calls == []


# ─── 回合终态：字段来源与四态换算 ───


def test_turn_outcome_fields_come_from_real_sources() -> None:
    outcome = writeback.build_turn_outcome(
        _research_state(),
        memory_writeback={"status": "success", "candidate_ids": ["mem-9"]},
    )

    assert outcome["status"] == "completed"
    assert outcome["termination_reason"] == "converged"
    assert outcome["final_depth"] == "research"
    assert outcome["depth_reason_code"] == "cross_paper_comparison_required"
    assert outcome["memory_item_ids"] == ["mem-1"]
    # 引用证据取自定稿答案块并按出现顺序去重
    assert outcome["evidence_ids"] == ["ev-1", "ev-2"]
    assert outcome["loop_steps"] == 4
    assert outcome["missing_evidence"] == []
    assert outcome["error_kind"] == ""


def test_turn_outcome_depth_transitions_stay_empty() -> None:
    """循环内没有任何代码改深度，轨迹留空数组，不造生产方。"""
    outcome = writeback.build_turn_outcome(_research_state(), memory_writeback={"status": "skipped"})

    assert outcome["depth_transitions"] == []


@pytest.mark.parametrize(
    "decision,expected",
    [
        ("terminate_fail", ("failed", "terminate_fail")),
        ("terminate_honest", ("incomplete", "terminate_honest")),
        ("", ("completed", "converged")),
    ],
)
def test_turn_outcome_status_maps_from_loop_decision(decision, expected) -> None:
    outcome = writeback.build_turn_outcome(
        _research_state(planner_loop_decision=decision),
        memory_writeback={"status": "skipped"},
    )

    assert (outcome["status"], outcome["termination_reason"]) == expected


@pytest.mark.parametrize(
    "planner_status,expected",
    [("ok", "success"), ("partial", "empty"), ("insufficient", "empty"), ("failed", "failed")],
)
def test_capability_results_convert_planner_status_explicitly(planner_status, expected) -> None:
    outcome = writeback.build_turn_outcome(
        _research_state(source_status={"planner": planner_status}),
        memory_writeback={"status": "empty"},
    )

    assert outcome["capability_results"]["retrieve_paper_evidence"] == expected
    assert outcome["capability_results"]["read_project_memory"] == "success"
    assert outcome["capability_results"]["write_back_project_memory"] == "empty"


def test_capability_results_keep_failed_and_empty_separate() -> None:
    failed_read = writeback.build_turn_outcome(
        _research_state(memory_read_result={"status": "failed", "error_kind": "OSError"}),
        memory_writeback={"status": "failed", "error_kind": "TimeoutError"},
    )
    empty_read = writeback.build_turn_outcome(
        _research_state(memory_read_result={"status": "empty", "items": []}),
        memory_writeback={"status": "empty"},
    )

    assert failed_read["capability_results"]["read_project_memory"] == "failed"
    assert empty_read["capability_results"]["read_project_memory"] == "empty"


def test_capability_results_omit_never_invoked_memory_read() -> None:
    """没读过记忆就不进表；模型判定不必翻记忆才算 skipped。"""
    state = _research_state(turn_depth_assessment={"depth": "research", "reason_code": "r"})
    state.pop("memory_read_result")

    never_asked = writeback.build_turn_outcome(state, memory_writeback={"status": "skipped"})

    assert "read_project_memory" not in never_asked["capability_results"]

    decided_not_to_read = writeback.build_turn_outcome(
        {**state, "turn_depth_assessment": {"needs_project_memory": False, "reason_code": "r"}},
        memory_writeback={"status": "skipped"},
    )

    assert decided_not_to_read["capability_results"]["read_project_memory"] == "skipped"


# ─── 跨 State 与终结节点：字段必须真的穿过图入口和写回收尾 ───


def test_turn_level_key_lists_stay_in_sync() -> None:
    """state 侧清单和 trace summary 侧清单漂了，回合级字段就只进 state 不进 trace 列表接口。"""
    from app.middleware.trace_builders import _TURN_LEVEL_TRACE_KEYS

    assert set(TURN_LEVEL_STATE_KEYS) == set(_TURN_LEVEL_TRACE_KEYS)


def _turn_fields() -> dict[str, Any]:
    """返回回合与O4交付字段，验证新旧消费层都完整透传同次事实。"""
    delivered = _supported_state()
    return {
        "turn_depth": "research",
        "turn_depth_assessment": {"depth": "research", "reason_code": "cross_paper_comparison_required"},
        "depth_transitions": [],
        "memory_read_result": {"status": "success", "items": [{"memory_id": "mem-1"}]},
        "turn_outcome": {"status": "completed", "loop_steps": 4, "termination_reason": "converged"},
        "memory_writeback": {"status": "success", "candidate_ids": ["mem-1"]},
        "final_delivery_report": delivered["final_delivery_report"],
        "final_generation_attempt": delivered["final_generation_attempt"],
        "final_citation_bindings": [],
        "final_delivery_applicability": {"applicable": True, "reason": "research"},
    }


@pytest.mark.parametrize("schema", [AgentState, RAGState])
def test_turn_fields_survive_real_graph_entry(schema) -> None:
    """六键必须在两个 State 都声明：没声明的键在图入口被静默丢弃，手搓 dict 测不出来。"""
    turn_fields = _turn_fields()

    def terminal_node(state: dict[str, Any]) -> dict[str, Any]:
        return build_write_back_payload(
            state, trace=None, trace_summary=None, rag_trace_dict=None, turn_fields=turn_fields
        )

    builder = StateGraph(schema)
    builder.add_node("write_back", terminal_node)
    builder.add_edge(START, "write_back")
    builder.add_edge("write_back", END)

    result = builder.compile().invoke({"user_query": "比较这两篇的方法"})

    for key in TURN_LEVEL_STATE_KEYS:
        assert result.get(key) == turn_fields[key], f"{schema.__name__} 丢了 {key}"


def test_absent_turn_fields_do_not_clobber_planner_values() -> None:
    """普通 RAG 回合不产回合级字段，payload 不能给缺失键补 None 覆盖 planner 判好的深度。"""
    payload = build_write_back_payload(
        {"turn_depth": "research"}, trace=None, trace_summary=None, rag_trace_dict=None
    )

    assert "turn_depth" not in payload
    assert "memory_writeback" not in payload


class _StubObservability:
    """替身观测层：只记完成事件，不落库、不发指标。"""

    def __init__(self) -> None:
        self.events: list[str] = []

    def record_timed_event(self, event: str, stage: str, started_at: float, **kwargs: Any) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_graph_trace_defers_candidates_until_result_commit(monkeypatch) -> None:
    """图内只记录待提交事实，不提前创建候选；trace与响应保留一致的真实阶段。"""
    from app.middleware import observability, trace_context
    from app.services import rag_trace_service

    persist_calls = _patch_persist(monkeypatch, memory_ids=["mem-9"])
    monkeypatch.setattr(writeback, "_curator_llm", lambda: _FakeStructuredLLM(_batch(_proposal())))
    written: list[dict[str, Any]] = []

    async def fake_safe_write_trace(*, trace_data: dict[str, Any], **kwargs: Any) -> None:
        written.append(trace_data)

    monkeypatch.setattr(trace_context, "safe_write_trace", fake_safe_write_trace)
    token = trace_context.set_trace_ctx(
        trace_context.TraceContext(request_id="req-e2e", session_id="sess-e2e")
    )
    try:
        state = _research_state(
            trace=observability.RAGTrace(session_id="sess-e2e", query="这两篇的实验设置差别在哪")
        )
        payload = await rag_trace_service.execute_write_back(
            state,
            paper_db=None,
            settings=None,
            observability_module=_StubObservability(),
            with_trace_fn=rag_trace_service.with_trace,
            update_trace_generation_fn=rag_trace_service.update_trace_generation,
        )
    finally:
        trace_context.reset_trace_ctx(token)

    assert persist_calls == []
    assert payload["memory_writeback"]["candidate_ids"] == []
    assert payload["memory_writeback"]["reason_codes"] == ["awaiting_result_commit"]
    assert payload["turn_outcome"]["status"] == "completed"
    assert payload["turn_outcome"]["loop_steps"] == 4
    # trace 列表接口只投影 summary，评测取数的键少一个就查不到
    summary = written[0]["summary"]
    for key in ("turn_depth", "turn_depth_assessment", "memory_read_result", "turn_outcome", "memory_writeback"):
        assert key in summary, f"summary 丢了 {key}"
    assert summary["memory_writeback"]["candidate_ids"] == []
    assert summary["memory_writeback"]["reason_codes"] == ["awaiting_result_commit"]
    assert summary["turn_depth_assessment"]["memory_read_reason_code"] == "prior_judgment_needed"
    # 循环里没有改深度的生产方，state 就没有这个键，summary 也不补空数组冒充跑过
    assert "depth_transitions" not in summary
    assert summary["turn_outcome"]["depth_transitions"] == []


@pytest.mark.asyncio
async def test_execute_write_back_leaves_plain_rag_turn_without_turn_fields(monkeypatch) -> None:
    """没有 turn_depth 的普通问答不评估写回：给它编一份终态会让评测把没跑过当成跑过。"""
    from app.middleware import observability, trace_context
    from app.services import rag_trace_service

    async def exploding_writeback(*args: Any, **kwargs: Any):
        raise AssertionError("普通 RAG 回合不该触发记忆写回")

    monkeypatch.setattr(writeback, "write_back_turn_memory", exploding_writeback)

    async def fake_safe_write_trace(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(trace_context, "safe_write_trace", fake_safe_write_trace)
    token = trace_context.set_trace_ctx(
        trace_context.TraceContext(request_id="req-plain", session_id="sess-plain")
    )
    try:
        payload = await rag_trace_service.execute_write_back(
            {"answer": "普通答案", "trace": observability.RAGTrace(session_id="sess-plain", query="是什么")},
            paper_db=None,
            settings=None,
            observability_module=_StubObservability(),
            with_trace_fn=rag_trace_service.with_trace,
            update_trace_generation_fn=rag_trace_service.update_trace_generation,
        )
    finally:
        trace_context.reset_trace_ctx(token)

    for key in ("turn_depth", "turn_depth_assessment", "depth_transitions", "memory_read_result", "turn_outcome", "memory_writeback"):
        assert key not in payload
    for key in ("final_delivery_report", "final_generation_attempt", "final_citation_bindings", "final_delivery_applicability"):
        assert payload.get(key) is None


@pytest.mark.asyncio
async def test_turn_fields_cross_the_rag_bridge_into_main_graph(monkeypatch) -> None:
    """子图 → 主图的反向回填：白名单漏一个键，API 与 SSE 的 done payload 就都取不到它。"""
    from app.agents import agent

    turn_fields = _turn_fields()

    async def fake_run_rag_graph_stream(graph: Any, rag_state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "两篇的实验设置差别在采样率", "citations": [], **turn_fields}

    monkeypatch.setattr(agent, "create_rag_node", lambda: None)
    monkeypatch.setattr(agent, "_run_rag_graph_stream", fake_run_rag_graph_stream)

    builder = StateGraph(AgentState)
    builder.add_node("rag", agent.rag_node)
    builder.add_edge(START, "rag")
    builder.add_edge("rag", END)

    result = await builder.compile().ainvoke({"user_query": "比较这两篇的方法"})

    for key in TURN_LEVEL_STATE_KEYS:
        assert result.get(key) == turn_fields[key], f"主图丢了 {key}"



def test_partial_evidence_turn_is_not_recorded_as_completed() -> None:
    """交付了部分结果时回合终态是 incomplete：只答了一部分不许记成正常完成。"""
    outcome = writeback.build_turn_outcome(
        _research_state(source_status={"planner": "partial"}),
        memory_writeback={"status": "empty"},
    )

    assert outcome["status"] == "incomplete"
    assert outcome["termination_reason"] == "partial_evidence"


def test_full_evidence_turn_stays_completed() -> None:
    """对照：取证完整时终态照旧 completed，部分完成那档不许吃掉正常回合。"""
    outcome = writeback.build_turn_outcome(
        _research_state(source_status={"planner": "ok"}),
        memory_writeback={"status": "empty"},
    )

    assert outcome["status"] == "completed"
    assert outcome["termination_reason"] == "converged"


def test_other_terminal_states_keep_their_reason() -> None:
    """对照：硬失败与诚实收尾各自的终态与终止原因不变。"""
    failed = writeback.build_turn_outcome(
        _research_state(planner_loop_decision="terminate_fail", source_status={"planner": "partial"}),
        memory_writeback={"status": "skipped"},
    )
    honest = writeback.build_turn_outcome(
        _research_state(planner_loop_decision="terminate_honest", source_status={"planner": "partial"}),
        memory_writeback={"status": "skipped"},
    )

    assert (failed["status"], failed["termination_reason"]) == ("failed", "terminate_fail")
    assert (honest["status"], honest["termination_reason"]) == ("incomplete", "terminate_honest")
