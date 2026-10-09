"""Planner claim validation tests."""

from __future__ import annotations

from app.models.research_planner_contracts import ResearchClaim, ResearchClaimDraft, ResearchUnit, ResearchUnitResult
from app.services.internal_research.research_planner_claim_provider import build_claim_prompt
from app.services.internal_research.research_planner_consumer_budget import CLAIM_PROMPT_VERSION, prepare_consumer_input
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer
from app.services.internal_research.research_planner_reasoner import (
    validate_claim_drafts,
    validate_unit_dependencies,
)
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit, supporting_spans, result_with_exposure

# 完整章节正文约 3000 字，关键结论落在第 2000 字之后：旧实现写死 1800 字裁会切掉它
_SECTION_BODY_CHARS = "前文" * 1500
_CONCLUSION_POSITION = 2000
_CONCLUSION_SENTENCE = "关键结论：本方法在全部基准上优于基线。"


def test_paper_supported_requires_evidence_ids() -> None:
    unit = _unit()
    report = validate_claim_drafts(unit, [], [], [_draft("paper_supported")])
    assert report.claims == []
    assert "supporting_evidence_ids" in report.errors[0]


def test_paper_supported_ids_resolve_to_payload_direct_text_records() -> None:
    result = validate_strategy_fixture(make_draft([make_unit(facets=["f1"])], explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    unit = result.plan.units[0]
    outcome = validate_claim_drafts(unit, [_direct_evidence()], [], [_draft("paper_supported", supporting=["e1"])])
    payload = synthesize_planner_answer(
        result.plan,
        [ResearchUnitResult(unit_id=unit.unit_id, status="evidence_found", evidence_ids=["e1"], claims=outcome.claims)],
        project_paper_count=1,
        evidence_records=[_direct_evidence()],
    )
    direct_ids = {record["evidence_id"] for record in payload["evidence_records"] if record.get("layer") == "direct_text"}
    claim = payload["research_unit_results"][0]["claims"][0]
    assert claim["claim_type"] == "paper_supported"
    assert set(claim["supporting_evidence_ids"]) <= direct_ids


def test_inference_requires_reasoning_summary() -> None:
    unit = _unit(allowed=["evidence_grounded_inference"], citation_required=False)
    draft = _draft("evidence_grounded_inference", supporting=["e1"])
    report = validate_claim_drafts(unit, [_direct_evidence()], [], [draft])
    assert report.claims == []
    assert "reasoning_summary" in report.errors[0]


def test_hypothesis_no_supporting_evidence() -> None:
    unit = _unit(allowed=["model_prior_hypothesis"])
    draft = _draft("model_prior_hypothesis", supporting=["e1"])
    report = validate_claim_drafts(unit, [_direct_evidence()], [], [draft])
    assert report.claims == []
    assert "cannot carry supporting_evidence_ids" in report.errors[0]


def test_required_dep_failure_blocks() -> None:
    unit = _unit_with_dep(required=True, claim_types=["paper_supported"])
    report = validate_unit_dependencies(unit, {"u1": _insufficient_result()})
    assert report.satisfied is False
    assert "status is insufficient_evidence" in report.blocking_reasons[0]


def test_optional_dep_failure_continues() -> None:
    unit = _unit_with_dep(required=False, claim_types=["paper_supported"])
    report = validate_unit_dependencies(unit, {"u1": _insufficient_result()})
    assert report.satisfied is True


def test_dep_satisfaction_claim_level() -> None:
    unit = _unit_with_dep(required=True, claim_types=["paper_supported"])
    upstream = ResearchUnitResult(
        unit_id="u1",
        status="inferred",
        claims=[
            _validated_claim(
                _draft(
                    "model_prior_hypothesis",
                    related=["e1"],
                    validation_tasks=["validate externally"],
                )
            )
        ],
    )
    report = validate_unit_dependencies(unit, {"u1": upstream})
    assert report.satisfied is False
    assert "paper_supported" in report.blocking_reasons[0]


def test_mixed_claim_types_no_promotion() -> None:
    unit = _unit(allowed=["paper_supported", "model_prior_hypothesis"])
    drafts = [
        _draft("paper_supported", supporting=["e1"]),
        _draft("model_prior_hypothesis", related=["e1"], validation_tasks=["run experiment"]),
    ]
    report = validate_claim_drafts(unit, [_direct_evidence()], [], drafts)
    assert [claim.claim_type for claim in report.claims] == ["paper_supported", "model_prior_hypothesis"]
    assert report.claims[1].supporting_evidence_ids == []
    assert report.claims[1].validation_status == "needs_validation"


def test_metadata_paper_supported_draft_is_rejected_without_silent_downgrade() -> None:
    unit = _metadata_unit()
    draft = _draft("paper_supported", supporting=["m1"])

    report = validate_claim_drafts(unit, [_metadata_evidence()], [], [draft])

    assert not report.claims
    assert "not allowed" in report.errors[0]


def test_inference_from_units_accepts_dependency_claim_source_unit() -> None:
    unit = _inference_unit_with_dep()
    upstream = _paper_result()
    draft = _inference_draft(source_units=["u1"])

    report = validate_claim_drafts(unit, [], [upstream], [draft], dependency_evidence_records=[_direct_evidence()])

    assert report.claims
    assert report.claims[0].source_unit_ids == ["u1"]


def test_inference_from_units_rejects_self_source_unit() -> None:
    unit = _inference_unit_with_dep()
    upstream = _paper_result()
    draft = _inference_draft(source_units=["u2"])

    report = validate_claim_drafts(unit, [], [upstream], [draft], dependency_evidence_records=[_direct_evidence()])

    assert report.claims == []
    assert any("source_claim_not_direct_dependency" in error for error in report.errors)


def test_inference_from_units_rejects_unvalidated_dependency_claims() -> None:
    unit = _inference_unit_with_dep()
    upstream = ResearchUnitResult(
        unit_id="u1",
        status="inferred",
        claims=[_paper_claim(validation_status="needs_validation")],
    )
    draft = _inference_draft(source_units=["u1"])

    report = validate_claim_drafts(unit, [], [upstream], [draft], dependency_evidence_records=[_direct_evidence()])

    assert report.claims == []
    assert "lack validated claims" in report.errors[0]


def test_inference_prompt_limits_source_unit_ids_to_direct_dependencies() -> None:
    unit = _chain_final_recommendation_unit()
    dependency_results = [
        _paper_result("u1"),
        _paper_result("u2"),
        _paper_result("u3"),
        _synthesis_result(),
    ]

    prompt = build_claim_prompt(unit, [], dependency_results)

    assert '"claim_id": "u4.c1"' in prompt
    assert '"claim_id": "u1.c1"' not in prompt
    assert "source_claim_ids must select the specific eligible claims from direct dependencies" in prompt
    assert "Ownership is derived by the server" in prompt


def test_final_recommendation_rejects_transitive_source_unit_ids() -> None:
    unit = _chain_final_recommendation_unit()
    synthesis = _synthesis_result()
    draft = _inference_draft(source_units=["u1", "u2", "u3", "u4"])

    upstream = [_paper_result(unit_id)
                for unit_id in ["u1", "u2", "u3"]]
    report = validate_claim_drafts(unit, [], [*upstream, synthesis], [draft], dependency_evidence_records=[_direct_evidence()])

    assert report.claims == []
    assert any("source_claim_not_direct_dependency" in error for error in report.errors)


def test_complete_section_reaches_claim_prompt_without_truncation() -> None:
    """一段完整章节正文（关键结论在 2000 字之后）必须原样进主张模型输入，不得被裁。

    本卡首条验收标准钉在这里：消费侧按配置总额度取正文，工具说完整取回的一整节要装得下。
    旧实现每条写死 1800 字裁且不说明，结论必然丢，这条用例必须红。
    """
    text = _SECTION_BODY_CHARS + _CONCLUSION_SENTENCE
    assert len(text) > _CONCLUSION_POSITION
    records = [{
        "evidence_id": "ev-long",
        "layer": "direct_text",
        "paper_id": "paper-a",
        "section": "Method",
        "expanded_text": text,
        "usable_as_citation": True,
    }]

    prompt = build_claim_prompt(_unit(), records, [])

    assert _CONCLUSION_SENTENCE in prompt
    assert "已截断" not in prompt


def test_claim_prompt_selects_visible_region_without_recounting_fenced_text() -> None:
    """无参数；代码块引用直接选可见原文区间，逐字引文仍严格核验，无返回值。"""
    # 1. 【研究主张】【区间选择】引文位于代码围栏内部，系统给出的范围覆盖整个可见原文
    quote = "6: Back-propagate gradients to update the spatial module."
    body = f"算法说明：\n```\n3: Shuffle the input and target channels.\n{quote}\n```\n"
    record = {**_direct_evidence(), "excerpt": body}
    unit = _unit()
    prepared = prepare_consumer_input([record], [], evidence_max_chars=1000, prompt_version=CLAIM_PROMPT_VERSION)
    visible = prepared.exposure.evidence[0]
    selection = visible.model_dump(include={"evidence_id", "evidence_version", "field", "start", "end"})
    draft = ResearchClaimDraft(text="更新空间模块", claim_type="paper_supported",
                               supporting_evidence_ids=[record["evidence_id"]],
                               supporting_spans=[{**selection, "quote": quote}])
    outcome = validate_claim_drafts(unit, [record], [], [draft], prepared_input=prepared)
    assert outcome.claims and not outcome.errors
    assert outcome.claims[0].supporting_spans == draft.supporting_spans
    assert (visible.start, visible.end) == (0, len(body))
    assert body.index(quote) > visible.start

    # 2. 【研究主张】【区间选择】误减围栏宽度或改写引文依然失败，不由程序修正或放行
    offset, fence_chars = body.index(quote), len("```\n")
    bad_selections = [
        draft.supporting_spans[0].model_copy(update={
            "start": offset - fence_chars, "end": offset + len(quote) - fence_chars}),
        draft.supporting_spans[0].model_copy(update={"quote": quote.replace("spatial", "temporal")}),
    ]
    for span in bad_selections:
        rejected = validate_claim_drafts(unit, [record], [], [draft.model_copy(update={"supporting_spans": [span]})],
                                         prepared_input=prepared)
        assert not rejected.claims
        assert f"support_quote_mismatch:{record['evidence_id']}" in rejected.errors

    # 3. 【研究主张】【区间选择】提示明确复制权威区间，不让模型重新计算引文字符位置
    prompt = build_claim_prompt(unit, [record], [], prepared_input=prepared)
    assert body in prompt
    assert "Copy evidence_id, field, start and end exactly from one matching evidence entry in Visible ranges" in prompt
    assert "not the quote's character offset" in prompt
    assert "Do not count or recalculate character positions or subtract Markdown fences" in prompt
    assert "quote must be a verbatim substring inside that selected region" in prompt
    assert "when finer location is uncertain" not in prompt
    assert prepared.exposure.prompt_version == "research-claim-v8"


def test_claim_prompt_keeps_author_explanations_and_reasoning_within_visible_support() -> None:
    """无参数；观察结果与作者归因原样送模，推理摘要不得新增未获支持的实验缺失断言，无返回值。"""
    body = ("Table 4 reports the full model and ablations. The replacement performs worse. "
            "The authors attribute this to transfer from short to long horizons.")
    record = {**_direct_evidence(), "excerpt": body}
    unit = _unit(allowed=["paper_supported", "evidence_grounded_inference"])
    prepared = prepare_consumer_input([record], [], evidence_max_chars=1000, prompt_version=CLAIM_PROMPT_VERSION)
    prompt = build_claim_prompt(unit, [record], [], prepared_input=prepared)
    assert body in prompt
    assert "Both text and reasoning_summary must stay within the selected support" in prompt
    assert "reasoning_summary must not introduce additional factual premises" in prompt
    assert "Absence from the visible excerpts does not establish that an experiment was never performed" in prompt
    assert "Preserve author attribution for proposed causal explanations" in prompt
    assert "distinct from directly observed experimental outcomes" in prompt


def _unit(*, allowed: list[str] | None = None, citation_required: bool = True):
    raw = make_unit(allowed_claims=allowed or ["paper_supported"], facets=["f1"])
    raw["claim_policy"]["citation_required"] = citation_required
    result = validate_strategy_fixture(make_draft([raw], explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan.units[0]


def _unit_with_dep(*, required: bool, claim_types: list[str]):
    dep = {"unit_id": "a", "required": required, "required_claim_types": claim_types, "reason": "needed"}
    raw_a = make_unit("a", facets=["f1"])
    raw_b = make_unit("b", goal="Use dependency", action="infer from dependency", deps=[dep], facets=["f2"])
    result = validate_strategy_fixture(
        make_draft([raw_a, raw_b], explicit_facets=[make_facet("f1"), make_facet("f2")])
    )
    assert result.plan is not None
    return result.plan.units[1]


def _metadata_unit():
    raw = make_unit(
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        facets=["f1"],
    )
    raw["claim_policy"]["citation_required"] = False
    raw["claim_policy"]["bounded_inference_allowed"] = True
    result = validate_strategy_fixture(make_draft([raw], explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan.units[0]


def _inference_unit_with_dep():
    dep = {"unit_id": "a", "required": True, "required_claim_types": ["paper_supported"], "reason": "needed"}
    raw_a = make_unit("a", facets=["f1"])
    raw_b = make_unit(
        "b",
        goal="Use dependency",
        action="infer from dependency",
        source_type="inference_from_units",
        allowed_claims=["evidence_grounded_inference"],
        deps=[dep],
        facets=["f2"],
        output_role="synthesis",
    )
    raw_b["claim_policy"]["citation_required"] = False
    result = validate_strategy_fixture(
        make_draft([raw_a, raw_b], explicit_facets=[make_facet("f1"), make_facet("f2")])
    )
    assert result.plan is not None
    return result.plan.units[1]


def _chain_final_recommendation_unit():
    units = [
        make_unit(
            "a",
            goal="Verify paper A method evidence",
            action="locate paper A method evidence",
            paper_ids=["paper-a"],
            facets=["f1"],
        ),
        make_unit(
            "b",
            goal="Verify paper B method evidence",
            action="locate paper B method evidence",
            paper_ids=["paper-b"],
            facets=["f2"],
        ),
        make_unit(
            "c",
            goal="Verify paper C method evidence",
            action="locate paper C method evidence",
            paper_ids=["paper-c"],
            facets=["f3"],
        ),
        _inference_unit(
            "d",
            goal="Synthesize cross-paper method gaps",
            action="compare validated paper claims",
            deps=[_dep("a"), _dep("b"), _dep("c")],
            facets=["f4"],
        ),
        _inference_unit(
            "e",
            goal="Recommend a concrete research direction",
            action="turn synthesis claims into a recommendation",
            deps=[_dep("d", claim_types=["evidence_grounded_inference"])],
            facets=["f5"],
        ),
    ]
    facets = [make_facet(f"f{index}") for index in range(1, 6)]
    result = validate_strategy_fixture(make_draft(units, explicit_facets=facets))
    assert result.plan is not None
    return result.plan.units[4]


def _inference_unit(
    unit_id: str,
    *,
    goal: str,
    action: str,
    deps: list[dict[str, object]],
    facets: list[str],
) -> dict[str, object]:
    raw = make_unit(
        unit_id,
        goal=goal,
        action=action,
        source_type="inference_from_units",
        allowed_claims=["evidence_grounded_inference"],
        deps=deps,
        facets=facets,
        output_role="final_recommendation" if unit_id == "e" else "synthesis",
    )
    raw["claim_policy"]["citation_required"] = False
    return raw


def _dep(unit_id: str, *, claim_types: list[str] | None = None) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "required": True,
        "required_claim_types": claim_types or ["paper_supported"],
        "reason": "needed",
    }


def _draft(
    claim_type: str,
    *,
    supporting: list[str] | None = None,
    related: list[str] | None = None,
    validation_tasks: list[str] | None = None,
) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="A claim",
        claim_type=claim_type,
        supporting_evidence_ids=supporting or [],
        supporting_spans=supporting_spans([_direct_evidence(), _metadata_evidence()], supporting or []),
        related_evidence_ids=related or [],
        reasoning_summary="",
        validation_tasks=validation_tasks or [],
    )


def _validated_claim(draft: ResearchClaimDraft):
    unit = _unit(allowed=[draft.claim_type], citation_required=False)
    report = validate_claim_drafts(unit, [_direct_evidence()], [], [draft])
    assert report.claims
    return report.claims[0]


def _paper_claim(validation_status: str = "validated", *, unit_id: str = "u1") -> ResearchClaim:
    return ResearchClaim(
        claim_id=f"{unit_id}.c1",
        text="The paper uses attention.",
        claim_type="paper_supported",
        validation_status=validation_status,
        supporting_evidence_ids=["e1"],
        supporting_spans=supporting_spans([_direct_evidence()]),
    )


def _inference_draft(source_units: list[str]) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="Dependency claims support a synthesis.",
        claim_type="evidence_grounded_inference",
        source_claim_ids=[f"{unit_id}.c1" for unit_id in source_units],
        reasoning_summary="The dependency claim supports this inference.",
    )


def _validated_inference_claim() -> ResearchClaim:
    return ResearchClaim(
        claim_id="u4.c1",
        text="Validated synthesis from upstream paper claims.",
        claim_type="evidence_grounded_inference",
        validation_status="validated",
        source_unit_ids=["u1", "u2", "u3"], source_claim_ids=["u1.c1", "u2.c1", "u3.c1"],
        reasoning_summary="Upstream paper claims support this synthesis.",
    )


def _direct_evidence() -> dict[str, object]:
    return {"evidence_id": "e1", "layer": "direct_text", "usable_as_citation": True, "excerpt": "The paper uses attention."}


def _metadata_evidence() -> dict[str, object]:
    return {"evidence_id": "m1", "layer": "metadata", "usable_as_citation": False, "summary": "Paper metadata"}


def _insufficient_result() -> ResearchUnitResult:
    return ResearchUnitResult(unit_id="u1", status="insufficient_evidence", unresolved_reason="no evidence")


def _paper_result(unit_id="u1"):
    """参数为真实归属；返回同一原文支持投影的测试事实。"""
    unit = ResearchUnit.model_validate(make_unit(unit_id))
    return result_with_exposure(unit, [_direct_evidence()], [_paper_claim(unit_id=unit_id)])


def _synthesis_result():
    """无参数；返回明确选择三条直接前置事实的合成结果。"""
    unit = ResearchUnit.model_validate(make_unit("u4", source_type="inference_from_units", output_role="synthesis",
           allowed_claims=["evidence_grounded_inference"], deps=[{"unit_id": key} for key in ["u1", "u2", "u3"]]))
    return result_with_exposure(unit, [], [_validated_inference_claim()], dependencies=[_paper_result(key) for key in ["u1", "u2", "u3"]])
