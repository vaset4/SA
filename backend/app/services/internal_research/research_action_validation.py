"""【研究循环】【动作准入】验证当前动作并追加已接受账本，不预设未来计划。作者：xxx。"""

from __future__ import annotations

import json
from typing import Any

from app.services.llm_observation import record_business_validation, record_structured_conversion

from pydantic import BaseModel, ValidationError

from app.models.research_action_contracts import (
    AnalyzeMaterials, ExecuteActions, InitialResearchDecision, InitialResearchDecisionOutput,
    ResearchAction, ResearchDecision, ResearchDecisionOutput,
)
from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION, ClaimPolicy, EvidencePolicy, ExpectedOutput, PlannerCoverageReview,
    PlannerFailure, PlannerMetadata, PlannerValidationResult, ResearchPlan, ResearchUnit, UnitDependency,
)
from app.services.internal_research.research_planner_validation_rules import (
    dependency_issues, fulfillment_contract_issues, normalize_coverage, scope_issues, user_presentation_issues,
)
from app.services.internal_research.research_planner_validator import PlannerValidationOptions
from app.services.internal_research.research_planner_loop_controller import fact_digest
from app.services.internal_research.research_planner_consumer_budget import evidence_version


def project_decision_output(raw: Any, *, initial: bool) -> ResearchDecision:
    """参数为新模型原响应和首次标记；严格校验后投影持久合同，旧字段不删改或重试。"""
    output_schema = InitialResearchDecisionOutput if initial else ResearchDecisionOutput
    # 1. 【研究决策】【原字段保留】替换供应端可能返回未校验实例，先保留嵌套多余字段再严格拒绝
    value = _decision_output_value(raw)
    output = (output_schema.model_validate_json(value, strict=True) if isinstance(value, str)
              else output_schema.model_validate(value, strict=True))
    stored_schema = InitialResearchDecision if initial else ResearchDecision
    # 2. 【研究决策】【持久投影】内部空默认仅表示未采集，原始模型字段已全部通过新合同校验
    stored = stored_schema.model_validate(output.model_dump(mode="python"), strict=True)
    record_structured_conversion(output, stored, schema=output_schema.__name__, target_schema=stored_schema.__name__)
    return stored


def _decision_output_value(value: Any) -> Any:
    """参数为决策响应中的原值；保留模型实例实际字段与嵌套结构，不用序列化删除未知字段。"""
    if isinstance(value, BaseModel):
        fields = {**value.__dict__, **(value.__pydantic_extra__ or {})}
        return {key: _decision_output_value(item) for key, item in fields.items()}
    if isinstance(value, dict):
        return {key: _decision_output_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decision_output_value(item) for item in value]
    return value


def parse_research_decision(raw: Any, *, initial: bool) -> ResearchDecision:
    """参数为模型原值及是否首次决策；返回严格动作对象，非法字段不做清洗。"""
    schema = InitialResearchDecision if initial else ResearchDecision
    if isinstance(raw, ResearchDecision):
        raw = raw.model_dump(mode="python")
    return schema.model_validate_json(raw, strict=True) if isinstance(raw, str) else schema.model_validate(raw, strict=True)


def decision_requirements(draft: ResearchDecision, baseline: PlannerCoverageReview | None) -> PlannerCoverageReview:
    """参数为决策和可信要求基准；首次建立要求，后续原样保留。"""
    if baseline is not None:
        return baseline
    if not isinstance(draft, InitialResearchDecision):
        raise ValueError("initial research decision requires task requirements")
    return PlannerCoverageReview.model_validate({
        "explicit_facets": [item.model_dump() for item in draft.requested_facets if item.source == "explicit"],
        "inferred_facets": [item.model_dump() for item in draft.requested_facets if item.source == "inferred"],
        "answer_sections": [item.model_dump() for item in draft.answer_sections],
    }, strict=True)


def validate_actions(
    raw: Any, *, options: PlannerValidationOptions, previous_plan: ResearchPlan | None,
    results: list[dict[str, Any]],
) -> tuple[PlannerValidationResult, PlannerCoverageReview | None]:
    """参数为原决策、可信约束和累计成果；返回动作账本或明确失败及独立的要求基准。"""
    if not options.question.strip():
        return _failure("missing_trusted_question", "research requires the current user question"), None
    try:
        draft = parse_research_decision(raw, initial=options.requirements_baseline is None)
    except ValidationError as exc:
        return _failure("invalid_action_format", str(exc)), None
    review = decision_requirements(draft, options.requirements_baseline)
    issues = user_presentation_issues(review, (options.question, *options.user_texts))
    issues.extend(fulfillment_contract_issues(review, [], allowed_paper_ids=options.allowed_paper_ids,
                                              require_complete=False))
    if issues:
        return _failure(issues[0].code, "; ".join(item.message for item in issues)), None
    if not isinstance(draft.decision, ExecuteActions):
        return PlannerValidationResult(plan=previous_plan), review
    # 1. 【研究循环】【批次准入】仅显式0关闭动作数量限制，正数配置保持原准入门槛
    if options.max_units != 0 and len(draft.decision.actions) > options.max_units:
        return _failure("too_many_actions", "current batch exceeds the configured planner action limit"), review
    prior_units = previous_plan.units if previous_plan else []
    known_results = {item["unit_id"] for item in results}
    units = [_project_action(action, f"u{len(prior_units) + index}", review)
             for index, action in enumerate(draft.decision.actions, 1)]
    for action in draft.decision.actions:
        if isinstance(action.input, AnalyzeMaterials):
            if action.input.analysis_mode == "memory" and not options.memory_grounded_allowed:
                return _failure("memory_intent_without_memory", "memory analysis requires available project memory"), review
            if any(dep.unit_id not in known_results for dep in action.input.dependencies):
                return _failure("action_input_unavailable", "analysis dependencies must refer to returned results"), review
            selection_error = _material_selection_error(action.input, results)
            if selection_error:
                return _failure("action_material_selection_invalid", selection_error), review
    cumulative = [*prior_units, *units]
    issues = scope_issues(cumulative, options.allowed_paper_ids) + dependency_issues(cumulative)
    coverage, coverage_issues = normalize_coverage(review, cumulative, require_complete=False)
    issues.extend(coverage_issues)
    if issues:
        return _failure(issues[0].code, "; ".join(item.message for item in issues)), review
    # 2. 【研究决策】【历史说明】新输出没有额外思路，旧计划中已保存的说明继续作为历史信息保留
    reasoning = draft.research_intent or (previous_plan.metadata.analyzer_reasoning if previous_plan else "")
    plan = ResearchPlan(question=options.question, plan_type="single_unit" if len(cumulative) == 1 else "multi_unit",
        units=cumulative, coverage_review=coverage, contract_version=CURRENT_RESEARCH_CONTRACT_VERSION,
        metadata=PlannerMetadata(execution_mode="incremental", pending_unit_ids=[unit.unit_id for unit in units],
                                 **({"analyzer_reasoning": reasoning} if reasoning else {})))
    return PlannerValidationResult(plan=plan), review


def _material_selection_error(action: AnalyzeMaterials, results: list[dict[str, Any]]) -> str:
    """参数为材料分析动作及已返回结果；选择须属于具体依赖，范围须来自实际未提供目录。"""
    by_id = {result["unit_id"]: result for result in results}
    ranges = {entry["range_id"]: entry for result in results
              for entry in result.get("metadata", {}).get("provenance", {}).get("exposure", {}).get("unprovided_ranges", [])}
    for dependency in action.dependencies:
        if dependency.required_product != "materials" or dependency.selections is None:
            continue
        ids = [item.evidence_id for item in dependency.selections]
        if len(ids) != len(set(ids)) or not set(ids).issubset(by_id[dependency.unit_id].get("evidence_ids", [])):
            return f"dependency {dependency.unit_id} selections contain unavailable or duplicate evidence_ids"
        for item in dependency.selections:
            if item.range_id is not None and ranges.get(item.range_id, {}).get("evidence_id") != item.evidence_id:
                return f"range_id is not an offered unprovided range of {item.evidence_id}"
    return ""


def _project_action(action: ResearchAction, unit_id: str, review: PlannerCoverageReview) -> ResearchUnit:
    """参数为有效动作、服务端身份和要求；返回当前能力的执行参数，不添加模型未选的来源。"""
    analysis = isinstance(action.input, AnalyzeMaterials)
    policy = action.input.model_dump(exclude={"dependencies", "output_role", "analysis_mode"})
    kinds = []
    if analysis:
        kinds = {"grounded": ["paper_supported", "evidence_grounded_inference"],
                 "hypothesis": ["model_prior_hypothesis"], "memory": ["memory_continuation"]}[action.input.analysis_mode]
    external = action.input.source_type == "external_context_needed"
    if external:
        kinds = ["external_needed"]
    # 1. 【研究循环】【实际分析】分析具备选择事实或推断的能力，不强迫两类结论都出现
    return ResearchUnit(unit_id=unit_id, goal=action.goal, research_action=action.goal,
        evidence_policy=EvidencePolicy.model_validate(policy, strict=True),
        claim_policy=ClaimPolicy(allowed_claim_types=kinds,
                                bounded_inference_allowed=bool(set(kinds) - {"paper_supported", "external_needed"})),
        expected_output=ExpectedOutput(output_goal=action.goal, deliverable_kind="claims" if analysis or external else "materials",
            answer_section_ids=[section.section_id for section in review.answer_sections
                                if set(section.facet_ids) & set(action.covered_facet_ids)]),
        output_role=action.input.output_role if analysis else (None if external else "evidence"),
        # 2. 【研究循环】【依赖投影】模型材料分支没有结论类型，内部保留既有空类型要求语义
        dependencies=[UnitDependency(unit_id=dep.unit_id, required_product=dep.required_product,
            required=dep.required, reason=dep.reason,
            required_claim_types=[] if dep.required_product == "materials" else [*dep.required_claim_types],
            selections=dep.selections if dep.required_product == "materials" else None)
            for dep in action.input.dependencies]
                     if analysis else [], covered_facet_ids=action.covered_facet_ids)


def _failure(code: str, message: str) -> PlannerValidationResult:
    """参数为准入错误；返回显式失败，原草案仍由图状态和调用观察保存。"""
    return PlannerValidationResult(failure=PlannerFailure(code=code, message=message))


def record_action_validation(raw: Any, result: PlannerValidationResult, *, initial: bool) -> None:
    """参数为真实动作、准入结果及首次标记；记录领域结论与错误，不把交付提议当已执行。"""
    record_business_validation(raw, {
        "validator": "research_planner", "schema": "InitialResearchDecision" if initial else "ResearchDecision",
        "phase": "current_action", "validation_stage": "action_admission",
        "valid": result.failure is None, "accepted": result.failure is None,
        "errors": [result.failure.model_dump(mode="json")] if result.failure else [],
        "accepted_plan": result.plan.model_dump(mode="json") if result.plan else None,
    })


def action_work_keys(raw: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """参数为决策及实际状态；读取绑定来源快照，分析绑定选中依赖，交付绑定当前成果。"""
    try:
        parsed = parse_research_decision(raw, initial="requested_facets" in raw)
        decision = parsed.decision.model_dump(mode="json")
    except ValidationError:
        return [fact_digest(raw)]
    if decision.get("kind") != "execute":
        return [fact_digest({"decision": {key: value for key, value in decision.items() if key != "reason"},
                             "facts": _selected_result_facts(state, None), "requirements": state.get("coverage_review")})]
    keys = []
    for action in decision.get("actions") or []:
        inputs = {key: sorted(value, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(value, list) else value
                  for key, value in (action.get("input") or {}).items()}
        basis: dict[str, Any] = {"input": inputs}
        publication = (state.get("project_scope") or {}).get("publication") or {}
        basis["publication"] = {key: value for key, value in publication.items() if key != "lease_id"}
        if inputs.get("source_type") == "inference_from_units":
            basis.update(goal=action.get("goal"), facets=action.get("covered_facet_ids"),
                         facts=_selected_result_facts(state, {dep["unit_id"] for dep in inputs["dependencies"]}))
            if inputs.get("analysis_mode") == "memory":
                basis["memory"] = state.get("memory_read_result")
        keys.append(fact_digest(basis))
    return sorted(keys)


def _selected_result_facts(state: dict[str, Any], selected: set[str] | None) -> dict[str, Any]:
    """参数为累计状态和选中成果；返回实际结论、状态和材料版本，不受无关新材料影响。"""
    results = [item for item in state.get("research_unit_results") or []
               if selected is None or item["unit_id"] in selected]
    facts = [{key: item.get(key) for key in ("unit_id", "status", "evidence_ids", "claims")} for item in results]
    ids = {eid for item in results for eid in item.get("evidence_ids", [])}
    ids.update(eid for item in results for claim in item.get("claims", []) for eid in claim.get("supporting_evidence_ids", []))
    versions = {item["evidence_id"]: evidence_version(item) for item in state.get("evidence_records") or []
                if item["evidence_id"] in ids}
    return {"results": facts, "evidence_versions": versions}
