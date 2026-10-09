"""Deterministic metrics for research planner replay reports.

深度与记忆这五项指标要的数据不在现有 20 条样本里，所以本模块显式区分「算出来是这个数」
与「这次根本没数据可算」：后者一律返回 NOT_MEASURED，不拿 0.0 或 1.0 顶替。

补样本时按下面的键名标注，指标不改代码就能开始出数：
- expected.depth: 本回合应有的处理深度，取值 minimal / lookup / research。
  只能依据深度定义标注，不许拿运行时选中的深度回填，否则评测只是在追认实现
- expected.should_escalate_depth: 本回合是否应当在循环中把深度调深（bool）
- expected.memory_owner_user_id: 本回合记忆归属人，用来核验 user_id 那道资格门

运行时载荷（replay）要带上这些键，指标才有数可算：
- turn_depth 或 turn_outcome.final_depth: 实际生效深度
- depth_transitions（顶层或 turn_outcome 内）: 深度变更记录。键在而值为空数组表示
  「本回合确实没改过深度」，键不在表示「这条链根本不记录深度变更」，两者不归并
- memory_read_result.items[]: 进上下文的记忆条目，且逐条带 user_id / review_state /
  status / stale_state 四个轴（注入侧投影已补齐这四个轴）
- evidence_records[]: 本轮 Evidence，带 evidence_id 与 usable_as_citation，
  引用可验证率据它判主张的引用能不能回溯到本轮证据
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from evals.contracts import (
    MetricResult,
    build_grading,
    freeze_record,
    measurement_coverage,
    project_legacy_record,
)

CATEGORY_NAMES = (
    "single_paper_lookup",
    "multi_paper_comparison",
    "mixed_goal",
    "insufficient_evidence",
    "external_context_needed",
    "conversational",
    "memory_lookup",
)
FAILURE_STAGES = (
    "none",
    "planning",
    "validation",
    "execution",
    "synthesis",
    "evidence_absence",
)
METRIC_NAMES = (
    "plan_valid_rate",
    "unit_decomposition_accuracy",
    "evidence_policy_accuracy",
    "unit_status_accuracy",
    "answer_partition_accuracy",
    "unsupported_claim_rate",
    "planner_failure_visibility",
    "depth_judgment_accuracy",
    "depth_escalation_recall",
    "unnecessary_retrieval_rate",
    "citation_verifiability_rate",
    "memory_injection_compliance_rate",
    "stale_memory_leak_rate",
)

# 指标取值：算得出就是 float，没数据可算时是 NOT_MEASURED 这个字符串
MetricValue = float | str

# 「本次没有数据可算这个指标」的显式取值。缺数据回 0.0 是假红、回 1.0 是假绿，
# 两者都会让缺标注看起来像结论；用字符串是为了让它在 JSON 与 markdown 里都一眼可辨
NOT_MEASURED = "not_measured"

# 三档深度由浅到深，升级判定按这个次序比档位；档位词表与运行时契约
# VALID_TURN_DEPTHS 同源，两边漂移由 test_eval_shape 的词表用例报警
DEPTH_ORDER: tuple[str, ...] = ("minimal", "lookup", "research")
MINIMAL_DEPTH = "minimal"
# 未知档位的档序，比任何合法档位都小，用来把认不出的取值挡在升级判定外
UNKNOWN_DEPTH_RANK = -1

# 会真的去翻论文的取证来源；inference_from_units 与 external_context_needed 不碰论文库
PAPER_SOURCE_TYPES = frozenset({"local_direct_text", "local_metadata", "local_structure_hint"})

# 记忆资格门里取值固定的三道：进上下文的条目要逐条对上
MEMORY_GATE_REQUIRED: tuple[tuple[str, str], ...] = (
    ("review_state", "accepted"),
    ("status", "confirmed"),
    ("stale_state", "fresh"),
)
# 第四道门是归属人，取值随样本走，所以单列
MEMORY_OWNER_AXIS = "user_id"
MEMORY_GATE_AXES: tuple[str, ...] = (MEMORY_OWNER_AXIS, *(axis for axis, _ in MEMORY_GATE_REQUIRED))

# 失效记忆的两种新鲜度，进了上下文即算泄漏
STALE_STATES = frozenset({"stale", "needs_rebind"})

# 必须由本轮论文 Evidence 支撑的主张类型：只有这类进引用可验证率的分母。
# 推断、假设、记忆延续本就不带引用，混进分母会把「本轮没有论文事实」算成「引用都可验证」
PAPER_FACT_CLAIM_TYPE = "paper_supported"
# 引文校验通过的主张状态，与运行时 ClaimValidationStatus 同源
VALIDATED_CLAIM_STATUS = "validated"


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [score_record(record) for record in records]
    return {
        "metrics": _aggregate_metrics(scored),
        "records": scored,
        "coverage_summary": _coverage_summary(scored),
    }


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """参数为旧记录或冻结合同；返回可重建评分、测量状态及兼容标量字段。"""
    frozen = freeze_record(record, mode="planner_replay")
    projection = project_legacy_record(frozen)
    expected = _mapping(projection.get("expected"))
    replay = _mapping(projection.get("replay"))
    failure_stage = _resolve_failure_stage(replay)
    measurements = _measurements(expected, replay)
    if "expected" in projection and not isinstance(projection["expected"], dict):
        measurements = [_unmeasured(name, "invalid_input", "truth.expected must be an object") for name in METRIC_NAMES]
    scores = {metric.name: metric.value if metric.status == "measured" else NOT_MEASURED for metric in measurements}
    return {
        "eval_case": frozen["eval_case"], "run_record": frozen["run_record"],
        "grading_result": build_grading(frozen, measurements, grader="research_planner"),
        "id": record.get("id"),
        "category": record.get("category"),
        "question": projection.get("question"),
        "expected_failure_stage": expected.get("failure_stage"),
        "failure_stage": failure_stage,
        "scores": scores,
        **scores,
    }


def _measurements(expected: dict[str, Any], replay: dict[str, Any]) -> list[MetricResult]:
    """参数为真值与实际观察；分别验证各指标所需字段并记录真实计数。"""
    return [
        _measure_plan(expected, replay), _measure_decomposition(expected, replay),
        _measure_expected_list("evidence_policy_accuracy", expected, replay),
        _measure_expected_list("unit_status_accuracy", expected, replay),
        _measure_expected_list("answer_partition_accuracy", expected, replay),
        _measure_claims("unsupported_claim_rate", replay), _measure_failure(expected, replay),
        _measure_depth(expected, replay), _measure_escalation(expected, replay),
        _measure_retrieval(expected, replay), _measure_claims("citation_verifiability_rate", replay),
        _measure_memory("memory_injection_compliance_rate", expected, replay),
        _measure_memory("stale_memory_leak_rate", expected, replay),
    ]


def _unmeasured(name: str, status: str, reason: str) -> MetricResult:
    """参数为指标、状态与定位原因；返回无数值的测量结果。"""
    return MetricResult(name=name, status=status, reason=reason)


def _ratio(name: str, numerator: int | float, denominator: int, *, reason: str) -> MetricResult:
    """参数为指标、实际分子/分母与原因；空分母退出测量，其余保留可追溯计数。"""
    if not denominator:
        return _unmeasured(name, "not_measured", f"not_applicable:{reason}")
    return MetricResult(name=name, status="measured", reason=reason, value=numerator / denominator,
                        numerator=numerator, denominator=denominator)


def _field_issue(name: str, holder: dict[str, Any], key: str, *, kind: type, layer: str) -> MetricResult | None:
    """参数为指标、字段容器和字段名及类型/层；返回缺失或类型错误，合法时返回 None。"""
    if key not in holder:
        return _unmeasured(name, "missing", f"{layer}.{key} is missing")
    if not isinstance(holder[key], kind) or (kind is int and isinstance(holder[key], bool)):
        return _unmeasured(name, "invalid_input", f"{layer}.{key} must be {kind.__name__}")
    return None


def _plan_issue(name: str, replay: dict[str, Any]) -> MetricResult | None:
    """参数为指标和观察；验证实际计划及单元列表，显式空计划表示未产出。"""
    if "research_plan" not in replay:
        return _unmeasured(name, "missing", "observation.research_plan is missing")
    plan = replay["research_plan"]
    if plan is None:
        return None
    if not isinstance(plan, dict):
        return _unmeasured(name, "invalid_input", "observation.research_plan must be an object or null")
    issue = _field_issue(name, plan, "units", kind=list, layer="observation.research_plan")
    if issue:
        return issue
    if any(not isinstance(unit, dict) for unit in plan["units"]):
        return _unmeasured(name, "invalid_input", "observation.research_plan.units must contain objects")
    return None


def _results_issue(name: str, replay: dict[str, Any], *, claims: bool = False) -> MetricResult | None:
    """参数为指标、观察及是否读取主张；缺少实际结果/主张列表时不以空列表代替。"""
    issue = _field_issue(name, replay, "research_unit_results", kind=list, layer="observation")
    if issue:
        return issue
    for result in replay["research_unit_results"]:
        if not isinstance(result, dict):
            return _unmeasured(name, "invalid_input", "observation.research_unit_results must contain objects")
        if not claims:
            continue
        issue = _field_issue(name, result, "claims", kind=list, layer="observation.research_unit_results[]")
        if issue:
            return issue
        if any(not isinstance(claim, dict) for claim in result["claims"]):
            return _unmeasured(name, "invalid_input", "observation.claims must contain objects")
    return None


def _measure_plan(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为真值和观察；仅非 minimal 案例评计划有效性，实际失败保持零分。"""
    name = "plan_valid_rate"
    if expected.get("depth") == MINIMAL_DEPTH:
        return _unmeasured(name, "not_measured", "not_applicable:minimal_turn")
    issue = _plan_issue(name, replay)
    if issue:
        return issue
    return _ratio(name, _score_plan_valid(replay), 1, reason="actual_plan_validity")


def _measure_decomposition(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为真值和观察；组合分保留原口径，不伪造跨组件分母。"""
    name = "unit_decomposition_accuracy"
    if expected.get("depth") == MINIMAL_DEPTH:
        return _unmeasured(name, "not_measured", "not_applicable:minimal_turn")
    if not ("min_units" in expected or "source_types" in expected):
        return _unmeasured(name, "missing", "truth.min_units/source_types is missing")
    for key, kind in (("min_units", int), ("source_types", list)):
        issue = _field_issue(name, expected, key, kind=kind, layer="truth") if key in expected else None
        if issue:
            return issue
    if expected.get("min_units", 0) < 0 or any(not isinstance(item, str) for item in expected.get("source_types", [])):
        return _unmeasured(name, "invalid_input", "truth decomposition constraints are invalid")
    if not expected.get("min_units") and not expected.get("source_types"):
        return _unmeasured(name, "not_measured", "not_applicable:no_decomposition_constraints")
    issue = _plan_issue(name, replay)
    if issue:
        return issue
    return MetricResult(name=name, status="measured", value=_score_unit_decomposition(expected, replay),
                        reason="mean_of_unit_count_and_source_coverage; counts_not_additive")


def _measure_expected_list(name: str, expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为列表型指标、真值和观察；按期望元素出现次数计数，不从缺标注推断满分。"""
    keys = {"evidence_policy_accuracy": "source_types", "unit_status_accuracy": "statuses",
            "answer_partition_accuracy": "partitions"}
    key = keys[name]
    if expected.get("depth") == MINIMAL_DEPTH and key not in expected:
        return _unmeasured(name, "not_measured", "not_applicable:minimal_turn")
    issue = _field_issue(name, expected, key, kind=list, layer="truth")
    if issue:
        return issue
    target = expected[key]
    if any(not isinstance(item, str) or not item.strip() for item in target):
        return _unmeasured(name, "invalid_input", f"truth.{key} must contain nonempty strings")
    if not target:
        return _unmeasured(name, "not_measured", f"not_applicable:no_expected_{key}")
    issue = _plan_issue(name, replay) if key == "source_types" else _results_issue(name, replay, claims=key == "partitions")
    if issue:
        return issue
    actual = _unit_source_types(_plan_units(replay)) if key == "source_types" else (
        _result_statuses(replay) if key == "statuses" else _actual_partitions(replay))
    counts = Counter(actual)
    hits = sum(min(counts.get(value, 0), count) for value, count in Counter(target).items())
    return _ratio(name, hits, len(target), reason=f"matched_expected_{key}")


def _measure_failure(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为预期与观察；失败阶段仅从实际 trace/result 解析，保留预期不符。"""
    name = "planner_failure_visibility"
    issue = _field_issue(name, expected, "failure_stage", kind=str, layer="truth")
    if issue:
        return issue
    if expected["failure_stage"] not in FAILURE_STAGES:
        return _unmeasured(name, "invalid_input", "truth.failure_stage is unknown")
    issue = _field_issue(name, replay, "planner_trace", kind=dict, layer="observation")
    if issue:
        return issue
    score = _score_failure_visibility(expected, replay, _resolve_failure_stage(replay))
    return _ratio(name, score, 1, reason="observed_failure_matches_expectation")


def _measure_depth(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为真值和观察；标注和实际深度都合法时才计入准确率。"""
    name = "depth_judgment_accuracy"
    issue = _field_issue(name, expected, "depth", kind=str, layer="truth")
    if issue:
        return issue
    actual = _actual_depth(replay)
    if expected["depth"] not in DEPTH_ORDER:
        return _unmeasured(name, "invalid_input", "truth.depth is unknown")
    if not actual:
        return _unmeasured(name, "missing", "observation.turn_depth/final_depth is missing")
    if actual not in DEPTH_ORDER:
        return _unmeasured(name, "invalid_input", "observation.depth is unknown")
    return _ratio(name, int(actual == expected["depth"]), 1, reason="actual_depth_matches_annotation")


def _measure_escalation(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为真值和观察；不要求升级退出分母，要求升级却缺记录则报告缺失。"""
    name = "depth_escalation_recall"
    issue = _field_issue(name, expected, "should_escalate_depth", kind=bool, layer="truth")
    if issue:
        return issue
    if not expected["should_escalate_depth"]:
        return _unmeasured(name, "not_measured", "not_applicable:escalation_not_required")
    outcome = _mapping(replay.get("turn_outcome"))
    holder = outcome if "depth_transitions" in outcome else replay
    issue = _field_issue(name, holder, "depth_transitions", kind=list, layer="observation")
    if issue:
        return issue
    transitions = holder["depth_transitions"]
    if any(not isinstance(item, dict) or item.get("from_depth") not in DEPTH_ORDER
           or item.get("to_depth") not in DEPTH_ORDER for item in transitions):
        return _unmeasured(name, "invalid_input", "observation.depth_transitions contains invalid depths")
    return _ratio(name, int(any(_is_escalation(item) for item in transitions)), 1, reason="depth_escalation_observed")


def _measure_retrieval(expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为真值和观察；仅标注 minimal 样本计论文检索率，缺实际计划不推断未检索。"""
    name = "unnecessary_retrieval_rate"
    issue = _field_issue(name, expected, "depth", kind=str, layer="truth")
    if issue:
        return issue
    if expected["depth"] not in DEPTH_ORDER:
        return _unmeasured(name, "invalid_input", "truth.depth is unknown")
    if expected["depth"] != MINIMAL_DEPTH:
        return _unmeasured(name, "not_measured", "not_applicable:non_minimal_turn")
    issue = _plan_issue(name, replay)
    if issue:
        return issue
    return _ratio(name, int(_paper_retrieval_happened(replay)), 1, reason="paper_retrieval_observed")


def _measure_claims(name: str, replay: dict[str, Any]) -> MetricResult:
    """参数为主张指标和观察；只核验实际主张，引用指标必须有实际 Evidence 载体。"""
    issue = _results_issue(name, replay, claims=True)
    if issue:
        return issue
    claims = _claims(replay)
    if name == "unsupported_claim_rate":
        return _ratio(name, sum(_claim_is_unsupported(claim) for claim in claims), len(claims), reason="unsupported_actual_claims")
    if any("claim_type" not in claim for claim in claims):
        return _unmeasured(name, "missing", "observation.claim_type is missing")
    if any(not isinstance(claim["claim_type"], str) or not claim["claim_type"] for claim in claims):
        return _unmeasured(name, "invalid_input", "observation.claim_type must be a nonempty string")
    claims = [claim for claim in claims if _is_paper_fact_claim(claim)]
    if not claims:
        return _unmeasured(name, "not_measured", "not_applicable:no_paper_fact_claims")
    issue = _field_issue(name, replay, "evidence_records", kind=list, layer="observation")
    if issue:
        return issue
    if any(not isinstance(item, dict) for item in replay["evidence_records"]):
        return _unmeasured(name, "invalid_input", "observation.evidence_records must contain objects")
    if any("evidence_id" not in item or "usable_as_citation" not in item for item in replay["evidence_records"]):
        return _unmeasured(name, "missing", "observation.evidence_records lacks citation identity/eligibility")
    if any(not isinstance(item["usable_as_citation"], bool) or not isinstance(item["evidence_id"], str)
           or not item["evidence_id"] for item in replay["evidence_records"]):
        return _unmeasured(name, "invalid_input", "observation.evidence_records has invalid citation identity/eligibility")
    citable = _citable_evidence_ids(replay)
    return _ratio(name, sum(_citation_is_verifiable(claim, citable) for claim in claims), len(claims), reason="verifiable_paper_claims")


def _measure_memory(name: str, expected: dict[str, Any], replay: dict[str, Any]) -> MetricResult:
    """参数为记忆指标、真值和观察；空读退出分母，缺资格轴不能冒充合规。"""
    issue = _field_issue(name, replay, "memory_read_result", kind=dict, layer="observation")
    if issue:
        return issue
    read = replay["memory_read_result"]
    issue = _field_issue(name, read, "items", kind=list, layer="observation.memory_read_result")
    if issue:
        return issue
    items = read["items"]
    if any(not isinstance(item, dict) for item in items):
        return _unmeasured(name, "invalid_input", "observation.memory items must contain objects")
    if not items:
        return _unmeasured(name, "not_measured", "not_applicable:no_injected_memory")
    axes = MEMORY_GATE_AXES if name == "memory_injection_compliance_rate" else ("stale_state",)
    if any(not str(item.get(axis) or "").strip() for item in items for axis in axes):
        return _unmeasured(name, "missing", "observation.memory items lack eligibility axes")
    if name == "stale_memory_leak_rate":
        count = sum(item["stale_state"] in STALE_STATES for item in items)
        return _ratio(name, count, len(items), reason="stale_injected_memory")
    issue = _field_issue(name, expected, "memory_owner_user_id", kind=str, layer="truth")
    if issue:
        return issue
    if not expected["memory_owner_user_id"].strip():
        return _unmeasured(name, "invalid_input", "truth.memory_owner_user_id is empty")
    count = sum(_passes_memory_gates(item, expected["memory_owner_user_id"]) for item in items)
    return _ratio(name, count, len(items), reason="eligible_injected_memory")


def _aggregate_metrics(scored: list[dict[str, Any]]) -> dict[str, MetricValue]:
    """
    按指标名跨样本聚合。

    参数:
        scored: 每条样本的打分结果
    返回:
        指标名 → 样本均值；该指标一条样本都没算出数时给 NOT_MEASURED 而不是 0.0，
        免得「没标注」和「全错」在报告里长得一样
    """
    metrics: dict[str, MetricValue] = {}
    for name in METRIC_NAMES:
        values = [float(item[name]) for item in scored if isinstance(item.get(name), (int, float))]
        metrics[name] = _mean(values) if values else NOT_MEASURED
    return metrics


def _score_plan_valid(replay: dict[str, Any]) -> float:
    units = _plan_units(replay)
    trace = _mapping(replay.get("planner_trace"))
    code = str(trace.get("code") or "").lower()
    if "analysis" in code or "validation" in code:
        return 0.0
    return 1.0 if units else 0.0


def _score_unit_decomposition(expected: dict[str, Any], replay: dict[str, Any]) -> float:
    units = _plan_units(replay)
    parts: list[float] = []
    min_units = int(expected.get("min_units") or 0)
    if min_units > 0:
        parts.append(min(len(units) / min_units, 1.0))
    source_types = _strings(expected.get("source_types"))
    if source_types:
        parts.append(_multiset_coverage(_unit_source_types(units), source_types))
    return _mean(parts)


def _score_failure_visibility(
    expected: dict[str, Any],
    replay: dict[str, Any],
    failure_stage: str,
) -> float:
    expected_stage = str(expected.get("failure_stage") or "none").strip()
    if expected_stage != failure_stage:
        return 0.0
    if expected_stage == "none":
        return 1.0
    if expected_stage in {"planning", "validation"}:
        return 1.0 if _trace_has_failure_code(replay) and not _result_statuses(replay) else 0.0
    if expected_stage == "evidence_absence":
        return 1.0 if _has_unresolved_reason(replay) else 0.0
    return 1.0 if _trace_has_failure_code(replay) else 0.0


def _is_paper_fact_claim(claim: dict[str, Any]) -> bool:
    """这条主张是不是「必须由本轮论文 Evidence 支撑」的那类。"""
    return str(claim.get("claim_type") or "").strip() == PAPER_FACT_CLAIM_TYPE


def _citable_evidence_ids(replay: dict[str, Any]) -> frozenset[str]:
    """
    取本回合可作引用的 Evidence id。

    只收 usable_as_citation 为真的记录：不可引用的命中虽然进了上下文，
    拿它当引用来源等于给用户一条回溯不到原文的引文。

    参数:
        replay: 本回合运行时载荷，取 evidence_records
    返回:
        可引用的 Evidence id 集合
    """
    ids = {
        str(record["evidence_id"]).strip()
        for record in (_mapping(item) for item in _list(replay.get("evidence_records")))
        if record.get("usable_as_citation") and str(record.get("evidence_id") or "").strip()
    }
    return frozenset(ids)


def _citation_is_verifiable(claim: dict[str, Any], citable: frozenset[str]) -> bool:
    """
    一条论文事实主张的引用是否可验证。

    三个条件缺一不可：引用非空（空引用是无出处断言）、逐条落在本轮可引用 Evidence 里
    （回溯不到本轮证据的引文等于凭印象编号）、主张本身通过引文校验。

    参数:
        claim: 一条论文事实主张
        citable: 本回合可引用的 Evidence id 集合
    返回:
        三个条件全满足为 True
    """
    evidence = set(_strings(claim.get("supporting_evidence_ids")))
    if not evidence or not evidence <= citable:
        return False
    return str(claim.get("validation_status") or "").strip() == VALIDATED_CLAIM_STATUS


def _resolve_failure_stage(replay: dict[str, Any]) -> str:
    trace = _mapping(replay.get("planner_trace"))
    explicit = str(trace.get("failure_stage") or replay.get("failure_stage") or "").strip()
    if explicit in FAILURE_STAGES:
        return explicit
    code = str(trace.get("code") or "").lower()
    if "analysis" in code or "planning" in code:
        return "planning"
    if "validation" in code:
        return "validation"
    if "execution" in code:
        return "execution"
    if "synthesis" in code:
        return "synthesis"
    if any(status in {"insufficient_evidence", "external_needed"} for status in _result_statuses(replay)):
        return "evidence_absence"
    return "none"


def _plan_units(replay: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _mapping(replay.get("research_plan"))
    return [_mapping(unit) for unit in _list(plan.get("units"))]


def _actual_depth(replay: dict[str, Any]) -> str:
    """
    取本回合最终生效的处理深度。

    参数:
        replay: 本回合运行时载荷
    返回:
        深度取值；载荷两处都没记深度时返回空串，表示无数据
    """
    outcome = _mapping(replay.get("turn_outcome"))
    return str(outcome.get("final_depth") or replay.get("turn_depth") or "").strip()


def _depth_rank(value: Any) -> int:
    """取深度档序，认不出的取值返回 UNKNOWN_DEPTH_RANK。"""
    depth = str(value or "").strip()
    if depth not in DEPTH_ORDER:
        return UNKNOWN_DEPTH_RANK
    return DEPTH_ORDER.index(depth)


def _is_escalation(transition: dict[str, Any]) -> bool:
    """一次深度变更是否为升级：两端档位都要认得出，降级与同档都不算。"""
    from_rank = _depth_rank(transition.get("from_depth"))
    to_rank = _depth_rank(transition.get("to_depth"))
    return from_rank > UNKNOWN_DEPTH_RANK and to_rank > from_rank


def _paper_retrieval_happened(replay: dict[str, Any]) -> bool:
    """本回合有没有去翻论文：计划里出现论文层取证单元即算发生，检索成本那时已经花掉。"""
    return any(source in PAPER_SOURCE_TYPES for source in _unit_source_types(_plan_units(replay)))


def _passes_memory_gates(item: dict[str, Any], owner: str) -> bool:
    """
    单条记忆是否同时满足四道资格门。

    参数:
        item: 进上下文的记忆条目，需带 user_id / review_state / status / stale_state
        owner: 本回合记忆归属人
    返回:
        四道门全过为 True
    """
    if str(item.get(MEMORY_OWNER_AXIS) or "").strip() != owner:
        return False
    return all(str(item.get(axis) or "").strip() == value for axis, value in MEMORY_GATE_REQUIRED)


def _unit_source_types(units: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for unit in units:
        policy = _mapping(unit.get("evidence_policy"))
        source = str(policy.get("source_type") or "").strip()
        if source:
            sources.append(source)
    return sources


def _result_statuses(replay: dict[str, Any]) -> list[str]:
    statuses: list[str] = []
    for result in _list(replay.get("research_unit_results")):
        status = str(_mapping(result).get("status") or "").strip()
        if status:
            statuses.append(status)
    return statuses


def _actual_partitions(replay: dict[str, Any]) -> list[str]:
    partitions = [claim["claim_type"] for claim in _claims(replay) if claim.get("claim_type")]
    for result in _list(replay.get("research_unit_results")):
        status = str(_mapping(result).get("status") or "").strip()
        if status in {"insufficient_evidence", "external_needed", "blocked"}:
            partitions.append(status)
    return partitions


def _claims(replay: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for result in _list(replay.get("research_unit_results")):
        claims.extend(_mapping(claim) for claim in _list(_mapping(result).get("claims")))
    return claims


def _claim_is_unsupported(claim: dict[str, Any]) -> bool:
    claim_type = str(claim.get("claim_type") or "").strip()
    evidence = _strings(claim.get("supporting_evidence_ids"))
    sources = _strings(claim.get("source_unit_ids"))
    reasoning = str(claim.get("reasoning_summary") or "").strip()
    if claim_type == "paper_supported":
        return not evidence
    if claim_type == "evidence_grounded_inference":
        return not (evidence or sources) or not reasoning
    if claim_type == "model_prior_hypothesis":
        return bool(evidence)
    return False


def _trace_has_failure_code(replay: dict[str, Any]) -> bool:
    trace = _mapping(replay.get("planner_trace"))
    return bool(str(trace.get("code") or trace.get("message") or "").strip())


def _has_unresolved_reason(replay: dict[str, Any]) -> bool:
    for result in _list(replay.get("research_unit_results")):
        payload = _mapping(result)
        reason = str(payload.get("unresolved_reason") or "").strip()
        if payload.get("status") in {"insufficient_evidence", "external_needed", "blocked"} and reason:
            return True
    return False


def _multiset_coverage(actual: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    actual_counts = Counter(actual)
    hits = 0
    for value, count in Counter(expected).items():
        hits += min(actual_counts.get(value, 0), count)
    return hits / len(expected)


def _coverage_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为逐条结果；返回类别、失败阶段和逐指标真实测量覆盖。"""
    categories = Counter(str(item.get("category") or "") for item in scored)
    stages = Counter(str(item.get("failure_stage") or "") for item in scored)
    return {
        "categories": {name: categories.get(name, 0) for name in CATEGORY_NAMES},
        "failure_stages": {name: stages.get(name, 0) for name in FAILURE_STAGES},
        "measurements": measurement_coverage(scored),
        "run_modes": dict(Counter(item["run_record"]["run_mode"] for item in scored)),
        "provenance": dict(Counter(item["run_record"]["provenance"] for item in scored)),
        "partial_records": sum(bool(item["run_record"]["missing_fields"]) for item in scored),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _list(value) if str(item or "").strip()]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
