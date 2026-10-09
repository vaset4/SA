"""Research planner failure visibility tests."""

from __future__ import annotations

from app.services.internal_research.research_planner_conversion import (
    build_evidence_not_found_response,
    build_planner_failure_response,
)
from app.services.internal_research.research_planner_pipeline import (
    _failure,
    _failure_from_exc,
)
from app.services.internal_research.research_planner_validator import validate_or_repair_plan, PlannerValidationOptions
from research_planner_strategy_test_utils import strategy_fixture
from app.services.llm_service import StructuredOutputError
from research_planner_test_utils import make_draft, make_facet


def test_repair_failure_no_fallback_to_execution() -> None:
    broken = strategy_fixture(make_draft(explicit_facets=[make_facet("f1"), make_facet("f2")]))
    result = validate_or_repair_plan(broken, repair_draft=broken,
        options=PlannerValidationOptions(question="text f1 text f2"))
    assert result.failure is not None
    response = build_planner_failure_response(result.failure)
    assert response["runtime_chain"] == "planner"
    assert response["source_status"]["planner"] == "failed"
    assert response["research_unit_results"] == []
    assert response["error_code"]
    assert response["error_summary"]


def test_failure_states_no_execution() -> None:
    result = validate_or_repair_plan("{not-json")
    assert result.failure is not None
    response = build_planner_failure_response(result.failure)
    assert response["source_status"]["local_evidence_execution"] == "not_run"
    # 1. 【研究失败】【事实来源】独立校验器没有运行回执，不能从错误身份补造读取说明
    assert "未执行本地证据检索" not in response["answer"]
    assert response["error_code"]
    assert isinstance(response["error_summary"], str) and response["error_summary"]


def test_failure_trace_complete() -> None:
    broken = strategy_fixture(make_draft(explicit_facets=[make_facet("f1"), make_facet("f2")]))
    result = validate_or_repair_plan(broken, repair_draft=broken,
        options=PlannerValidationOptions(question="text f1 text f2"))
    assert result.failure is not None
    trace = result.failure.trace
    assert trace["repair_attempted"] is True
    assert trace["repairs"][0]["code"] == "repair_draft_used"
    assert trace["validation_errors"][0]["code"] == "explicit_facet_uncovered"


def test_execution_error_code_does_not_invent_evidence_attempt() -> None:
    """无参数；错误码只说明故障，未提供本轮回执时不能宣称已启动本地读取。"""
    response = build_planner_failure_response(
        _failure("planner_execution_failed", "claim draft json parse failed")
    )

    assert response["source_status"]["local_evidence_execution"] == "not_run"
    assert "本轮本地证据检索已启动" not in response["answer"]
    assert response["error_code"] == "planner_execution_failed"
    assert "claim draft json parse failed" in response["error_summary"]


def test_structured_output_failure_maps_to_planner_failure_payload() -> None:
    """StructuredOutputError → PlannerFailure → 稳定 error_code/summary，证据 not_run。"""
    failure = _failure_from_exc(
        StructuredOutputError(
            "structured_output_failed",
            "LLM 结构化输出失败: schema=ResearchPlanDraft; error=unit_id Field required",
            user_summary="ResearchPlanDraft 结构化输出字段不符合契约，已停止后续规划",
            details={"repair_count": 1, "native_mode": "rejected"},
        ),
        "planner_analysis_failed",
    )
    response = build_planner_failure_response(failure)

    assert failure.code == "planner_structured_output_failed"
    assert failure.local_evidence_execution == "not_run"
    assert response["error_code"] == "planner_structured_output_failed"
    assert "不符合契约" in response["error_summary"]
    assert response["source_status"]["planner"] == "failed"
    assert response["source_status"]["local_evidence_execution"] == "not_run"
    # planner_trace 是 PlannerFailure.model_dump，structured 细节在 trace 子字典内
    assert response["planner_trace"]["trace"]["structured_output"]["details"]["repair_count"] == 1


def test_chat_outcome_planner_failed_not_answered() -> None:
    """chat 完成 outcome：planner failed 不得标成无差别 answered（复用真实判定 helper）。"""
    from app.api.routes.chat_api import _completed_qa_outcome

    result = {
        "answer": "可靠研究规划失败：结构化输出失败。本轮未执行本地证据检索。",
        "error_code": "planner_structured_output_failed",
        "error_summary": "结构化输出失败",
        "source_status": {"planner": "failed", "local_evidence_execution": "not_run"},
        "guardrails_action": None,
    }
    outcome = _completed_qa_outcome(result)
    assert outcome == "planner_failed"
    assert outcome != "answered"


def test_chat_outcome_evidence_not_found_distinct_from_failed() -> None:
    """AC6b：诚实收尾 error_code=evidence_not_found_in_scope 记 evidence_not_found，不混成 planner_failed。"""
    from app.api.routes.chat_api import _completed_qa_outcome

    honest = {
        "answer": "在你指定的 3 篇论文里检索过了，没找到关于「LSTTN 创新点」的内容。",
        "error_code": "evidence_not_found_in_scope",
        "source_status": {"planner": "no_evidence", "local_evidence_execution": "attempted"},
        "guardrails_action": None,
    }
    assert _completed_qa_outcome(honest) == "evidence_not_found"

    # 对照：真正的系统失败仍记 planner_failed
    failed = {
        "answer": "可靠研究规划失败：结构化输出失败。",
        "error_code": "planner_structured_output_failed",
        "source_status": {"planner": "failed"},
        "guardrails_action": None,
    }
    assert _completed_qa_outcome(failed) == "planner_failed"


def _honest_payload(paper_ids: list[str], authorized_count: int | None = None) -> dict:
    """按 scoped 论文列表跑一次诚实收尾组装。

    参数:
        paper_ids: 计划实际去检索的论文，只进结构化追溯字段
        authorized_count: 本轮授权论文数（决定正文档位）；缺省时按「计划点名的就是
            项目全部论文」这种一致情形取 paper_ids 的长度
    返回:
        诚实收尾载荷
    """
    return build_evidence_not_found_response(
        question="LSTTN 的创新点是什么",
        scoped_paper_ids=paper_ids,
        authorized_paper_count=len(paper_ids) if authorized_count is None else authorized_count,
        attempted_queries=["LSTTN innovation"],
        missing_evidence=[{"dimension": "method"}],
    )


def test_empty_project_says_missing_material_not_zero_papers() -> None:
    """空项目：正文说清项目里没有可核对的论文材料，不说"0 篇""这几篇论文"。"""
    payload = _honest_payload([])
    answer = payload["answer"]

    assert "这个项目里还没有论文" in answer
    assert "没有可以核对的材料" in answer
    # 指向下一步动作，用户看完知道该往项目里加论文
    assert "加进这个项目" in answer
    # 不存在的论文不能被指代，也不能出现"0 篇"这种别扭表述
    assert "这几篇论文" not in answer
    assert "0 篇" not in answer
    assert "0 篇" not in payload["error_summary"]


def test_scoped_papers_keep_original_wording() -> None:
    """有论文时措辞保持原样：查过了没找到，可能这几篇没涉及。"""
    payload = _honest_payload(["p1", "p2", "p3"])

    assert payload["answer"] == (
        "在你指定的 3 篇论文里检索过了，没找到关于「LSTTN 的创新点是什么」的内容，"
        "可能这几篇论文没涉及这个话题，或相关内容不在当前资料范围内。"
    )
    assert payload["error_summary"] == "在当前 3 篇论文范围内未检索到相关证据"


def test_honest_payload_keys_stable_across_paper_count() -> None:
    """两种措辞共用同一套载荷键与失败码，下游按同一形状消费。"""
    empty = _honest_payload([])
    scoped = _honest_payload(["p1"])

    assert set(empty) == set(scoped)
    assert set(empty["source_status"]) == set(scoped["source_status"])
    assert empty["error_code"] == scoped["error_code"] == "evidence_not_found_in_scope"
    assert empty["scoped_papers"] == []


def test_project_with_papers_is_never_told_it_is_empty() -> None:
    """项目里有论文、但模型草案没点名目标论文时，不许说「这个项目里还没有论文」。

    破了会怎样：用户项目里明明有 3 篇，却被告知项目是空的、要先去加论文，于是去
    重新上传已经在项目里的论文。论文数只能问项目真实范围，不能拿草案点名的目标
    论文顶替 —— 草案没点名时那个列表是空的。
    """
    payload = _honest_payload([], authorized_count=3)

    assert "这个项目里还没有论文" not in payload["answer"]
    assert "加进这个项目" not in payload["answer"]
    assert "3 篇" in payload["answer"]


def test_empty_project_with_ghost_target_still_says_missing_material() -> None:
    """项目是空的、草案却点名了一个不存在的论文时，仍按「缺可验证材料」说。

    破了会怎样：幽灵 id 让论文数算成 1，用户看到「在你指定的 1 篇论文里检索过了」，
    而项目里一篇都没有 —— 正是本单元要治的那句话换个形式又出来了。
    """
    payload = _honest_payload(["ghost-paper"], authorized_count=0)

    assert "这个项目里还没有论文" in payload["answer"]
    assert "1 篇" not in payload["answer"]
    # 草案点名的论文仍进结构化追溯字段，供排查「模型当时想查什么」
    assert payload["scoped_papers"] == [{"paper_id": "ghost-paper", "title": "ghost-paper"}]


def test_empty_project_trace_reason_names_the_real_cause() -> None:
    """空项目的 trace 归因要说「项目内没有论文」，不能说「scoped 论文范围内检索无命中」。

    检索这一轮确实发起过（同一份载荷里 local_evidence_execution 是 attempted、
    missing_evidence 带着 attempted_queries），所以问题不在「谎称检索过」，而在
    归因指错了地方：压根没有 scoped 论文范围可言，真正的原因是项目里没有材料。
    破了会怎样：排查的人照这句去调检索式和召回参数，而该做的是往项目里加论文。
    """
    empty_reason = _honest_payload([])["source_status_reasons"]["planner_final"]
    scoped_reason = _honest_payload(["p1"])["source_status_reasons"]["planner_final"]

    assert "项目内没有论文" in empty_reason
    assert "检索无命中" not in empty_reason
    # 有论文那条保持原归因：检索真发起过，只是没命中
    assert scoped_reason == "scoped 论文范围内检索无命中，诚实收尾"


def test_chat_outcome_partial_evidence_is_its_own_bucket() -> None:
    """部分完成自成一档：查到一部分、另一部分没查清，不许与全答对混记成 answered。"""
    from app.api.routes.chat_api import _completed_qa_outcome

    partial = {
        "answer": "论文 A 的方法有依据，论文 B 的效果缺少可比实验数据，无法比较。",
        "error_code": None,
        "source_status": {"planner": "partial", "local_evidence_execution": "attempted"},
        "guardrails_action": None,
    }

    assert _completed_qa_outcome(partial) == "partial_answer"
    assert _completed_qa_outcome(partial) != "answered"

    # 对照：完整作答仍记 answered，部分完成那档不许顺手把正常回合也吃掉
    complete = {
        "answer": "两篇论文的方法与效果如下。",
        "error_code": None,
        "source_status": {"planner": "ok", "local_evidence_execution": "attempted"},
        "guardrails_action": None,
    }

    assert _completed_qa_outcome(complete) == "answered"
