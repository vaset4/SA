"""【研究规划】【测试输入】显式把旧测试构造器的业务选择写成实时策略。作者：xxx。"""

from dataclasses import replace

from app.models.research_planner_contracts import ResearchPlan
from app.services.internal_research.research_planner_validator import PlannerValidationOptions, validate_or_repair_plan


def strategy_fixture(raw: dict) -> dict:
    """参数为本地测试选择；返回新策略输入，不用于产品、旧运行或历史读取。"""
    units = []
    for unit in raw["units"]:
        output = unit["expected_output"]
        intents = [{"claim_type": kind, "required": kind in output["required_claim_types"]}
                   for kind in unit["claim_policy"]["allowed_claim_types"]]
        # 1. 【研究规划】【材料测试】材料任务只声明材料；旧测试构造器通用policy不是模型策略
        if output.get("deliverable_kind") == "materials":
            intents = []
        strategy = {key: unit[key] for key in ("unit_id", "goal", "research_action", "evidence_policy",
                    "operation_family", "covered_facet_ids") if key in unit}
        strategy.update(output_goal=output["output_goal"], deliverable_kind=output.get("deliverable_kind"),
                        claim_intents=intents, validation_tasks=output.get("validation_tasks", []),
                        dependencies=[{"required": True, **dep} for dep in unit.get("dependencies", [])])
        if unit.get("output_role") not in {None, "evidence"}:
            strategy["output_role"] = unit["output_role"]
        units.append(strategy)
    review = raw["coverage_review"]
    return {"units": units, "requested_facets": [*review["explicit_facets"], *review["inferred_facets"]],
            "answer_sections": review.get("answer_sections", [])}


def validate_strategy_fixture(raw: dict, *, options: PlannerValidationOptions | None = None):
    """参数为测试选择和可选领域选项；构造明确测试用户原话后调用真实策略验证入口。"""
    strategy = strategy_fixture(raw)
    spans = tuple(facet["user_text_span"] for facet in strategy["requested_facets"])
    opts = replace(options or PlannerValidationOptions(), question=raw["question"], user_texts=spans)
    return validate_or_repair_plan(strategy, options=opts)


def current_plan_fixture(raw: dict) -> ResearchPlan:
    """参数为下游单元测试的正式计划字段；显式构造当前版本，不改变ID或主张资格。"""
    return ResearchPlan.model_validate({**raw, "contract_version": "research-planner-v4"}, strict=True)
