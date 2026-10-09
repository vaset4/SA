"""延续项目已有判断这一类主张的准入与呈现测试。

项目记忆是用户自己此前形成的判断，不是论文原文。这类主张一旦能凭空出现、
或者挂上证据 id、或者被标成已验证，用户就会在带引用的结论里读到自己的旧想法，
还以为系统从论文里查到了。这里守三件事：本轮真读到记忆才准用、不许带证据 id、
落到未验证一侧，以及正文里它与论文证据支持分栏可辨。

作者：xxx
2026-09-01 16:20:00
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.models.research_planner_contracts import (
    ResearchClaim,
    ResearchClaimDraft,
    ResearchPlan,
    ResearchUnit,
    ResearchUnitResult,
)
from app.services.internal_research import research_planner_synthesis as synthesis
from app.services.internal_research.research_planner_plan_prompt import render_project_memory_block
from app.services.internal_research.research_planner_reasoner import (
    ClaimValidationOutcome,
    ResearchUnitReasoner,
)
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer
from research_planner_strategy_test_utils import validate_strategy_fixture
from app.services.internal_research.research_planner_validator import PlannerValidationOptions
from research_planner_test_utils import make_draft, make_facet, make_unit

_MEMORY_CLAIM_TEXT = "接着项目已有判断：这条线仍按骨架图方向推进。"


def _memory_plan(*, opt_in: bool = True) -> ResearchPlan:
    """造一份经真实校验器产出的计划，按需让单元自己点名 memory_continuation。

    Args:
        opt_in: 草案的 allowed_claim_types 是否点名新类；False 用于对照组

    Returns:
        ResearchPlan: 校验/收紧后的可执行计划
    """
    allowed = ["evidence_grounded_inference"] + (["memory_continuation"] if opt_in else [])
    draft = make_draft([make_unit(allowed_claims=allowed, facets=["f1"])], explicit_facets=[make_facet("f1")])
    result = validate_strategy_fixture(draft, options=PlannerValidationOptions(memory_grounded_allowed=True))
    assert result.plan is not None
    return result.plan


def _memory_state(items: list[dict[str, Any]] | None = None, *, status: str = "success") -> dict[str, Any]:
    """造一个带记忆读取结果的 state 片段，供真实 render_project_memory_block 判定。"""
    return {"memory_read_result": {"status": status, "items": items if items is not None else []}}


def _memory_item() -> dict[str, Any]:
    """一条投影后的记忆条目，字段按 MemoryReadItem 注入所需的最小集给。"""
    return {
        "memory_id": "mem-1",
        "version": 1,
        "title": "骨架图构造是这个项目的主线",
        "body_summary": "前几轮对照实验都指向这个方向",
        "memory_type": "comparison_note",
        "maturity": "established",
        "source_kind": "turn_writeback",
        "provenance_label": "已确立",
    }


def _reason(unit: ResearchUnit, drafts: list[ResearchClaimDraft], state: dict[str, Any]) -> ClaimValidationOutcome:
    """按 pipeline 的接线方式跑一次真实 reason()：记忆门取自 render_project_memory_block。"""

    def provider(
        _unit: ResearchUnit,
        _records: list[dict[str, Any]],
        _deps: list[ResearchUnitResult],
        *,
        read_reports: list[dict[str, Any]] | None = None,
        prepared_input: Any = None,
    ) -> list[ResearchClaimDraft]:
        return drafts

    reasoner = ResearchUnitReasoner(
        draft_provider=provider,
        memory_grounded_allowed=bool(render_project_memory_block(state)),
    )
    return asyncio.run(reasoner.reason(unit, [], []))


def _memory_draft(*, supporting: list[str] | None = None) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text=_MEMORY_CLAIM_TEXT,
        claim_type="memory_continuation",
        supporting_evidence_ids=supporting or [],
        reasoning_summary="延续本项目已确立的判断",
    )


def test_memory_continuation_rejected_when_state_has_no_memory_read_result() -> None:
    """本轮压根没读记忆：state 里连 memory_read_result 这一键都没有，新类必须被拒。

    破了会怎样：任何一轮问答里模型都能编一条判断，标成「项目已有判断」，
    而这个项目从没读过记忆，用户读到的是凭空生成的「你以前认为」。
    """
    outcome = _reason(_memory_plan().units[0], [_memory_draft()], {})

    assert outcome.claims == []
    assert "requires project memory read" in outcome.rejected_reasons[0]


def test_memory_continuation_rejected_when_memory_read_failed() -> None:
    """记忆读取失败（status 非 success）：拿失败的读取结果不能当读到过。

    破了会怎样：库连不上或读取报错时反而放开了这一类，模型用它兜住整段回答，
    失败被一段看着像项目共识的正文盖住。
    """
    outcome = _reason(_memory_plan().units[0], [_memory_draft()], _memory_state([_memory_item()], status="error"))

    assert outcome.claims == []
    assert "requires project memory read" in outcome.rejected_reasons[0]


def test_memory_continuation_rejected_when_memory_items_empty() -> None:
    """读取成功但一条记忆都没有：空结果不构成「项目已有判断」。

    破了会怎样：新项目还没攒下任何判断，模型却能替它宣布已有立场。
    """
    outcome = _reason(_memory_plan().units[0], [_memory_draft()], _memory_state([]))

    assert outcome.claims == []
    assert "requires project memory read" in outcome.rejected_reasons[0]


def test_memory_continuation_allowed_when_memory_read_and_not_validated() -> None:
    """真读到记忆条目时放行，但落 needs_validation：它没有本轮 Evidence 支撑。

    破了会怎样：标成 validated 就进了可引用主张池，与论文结论同级，
    最终回答会给它配引用。
    """
    outcome = _reason(_memory_plan().units[0], [_memory_draft()], _memory_state([_memory_item()]))

    assert [claim.text for claim in outcome.claims] == [_MEMORY_CLAIM_TEXT]
    claim = outcome.claims[0]
    assert claim.claim_type == "memory_continuation"
    # 未验证是这一类的固定归属，不随本轮有没有证据变化
    assert claim.validation_status == "needs_validation"
    assert claim.supporting_evidence_ids == []


def test_memory_continuation_contract_rejects_supporting_evidence_ids() -> None:
    """契约层：这一类挂 supporting_evidence_ids 直接构造失败。

    破了会怎样：记忆条目被当成本轮证据 id 挂上去，最终回答按引用来源渲染，
    用户点进去发现那不是论文里的话。
    """
    with pytest.raises(ValidationError) as excinfo:
        ResearchClaim(
            claim_id="u1.c1",
            text=_MEMORY_CLAIM_TEXT,
            claim_type="memory_continuation",
            supporting_evidence_ids=["e1"],
        )

    assert "cannot carry supporting_evidence_ids" in str(excinfo.value)


def test_memory_continuation_reasoner_rejects_supporting_evidence_ids() -> None:
    """reasoner 层：即便本轮真读到记忆，挂了证据 id 一样拒。"""
    outcome = _reason(
        _memory_plan().units[0],
        [_memory_draft(supporting=["e1"])],
        _memory_state([_memory_item()]),
    )

    assert outcome.claims == []
    assert "cannot carry supporting_evidence_ids" in outcome.rejected_reasons[0]


def test_memory_continuation_dropped_when_plan_did_not_opt_in() -> None:
    """计划层没点名这一类时收紧掉：不跟着推断权限自动发放。

    破了会怎样：每个允许推断的单元都自带这张通行证，模型把它当没证据时的兜底。
    """
    unit = _memory_plan(opt_in=False).units[0]
    assert "memory_continuation" not in unit.claim_policy.allowed_claim_types

    outcome = _reason(unit, [_memory_draft()], _memory_state([_memory_item()]))

    assert outcome.claims == []
    assert "is not allowed" in outcome.rejected_reasons[0]


def test_memory_continuation_body_label_separates_it_from_paper_evidence() -> None:
    """正文分栏：延续判断有自己的小标题，不落在论文证据支持那一栏。

    破了会怎样：用户在「本地论文证据支持」标题下读到项目历史判断，
    以为这句话有论文可核对。
    """
    plan = _memory_plan()
    unit_id = plan.units[0].unit_id
    results = [
        ResearchUnitResult(
            unit_id=unit_id,
            status="evidence_found",
            evidence_ids=["e1"],
            claims=[
                ResearchClaim(
                    claim_id=f"{unit_id}.c1",
                    text="论文里写了注意力模块的接法。",
                    claim_type="paper_supported",
                    validation_status="validated",
                    supporting_evidence_ids=["e1"],
                ),
                ResearchClaim(
                    claim_id=f"{unit_id}.c2",
                    text=_MEMORY_CLAIM_TEXT,
                    claim_type="memory_continuation",
                    validation_status="needs_validation",
                ),
            ],
        )
    ]

    answer = synthesize_planner_answer(
        plan,
        results,
        project_paper_count=1,
        evidence_records=[{"evidence_id": "e1", "layer": "direct_text", "usable_as_citation": True}],
    )["answer"]

    memory_label = synthesis._CLAIM_LABELS["memory_continuation"]
    paper_label = synthesis._CLAIM_LABELS["paper_supported"]
    # 1. 两个标签都在正文里，且互不相同：读者能看出这是两种来源
    assert memory_label != paper_label
    assert memory_label in answer and paper_label in answer
    # 2. 延续判断的正文排在自己标题之后、论文证据那一栏之外
    assert answer.index(memory_label) < answer.index(_MEMORY_CLAIM_TEXT)
    assert answer.index(paper_label) < answer.index(memory_label)
    # 3. 标签自带非论文原文的身份提示，不能只是个中性词
    assert "非论文原文" in memory_label
