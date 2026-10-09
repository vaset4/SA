"""Oracle 配对的阶段症状归因。作者：xxx。"""

from __future__ import annotations

import json
from typing import Any

from evals.evidence.report import extract_stage_observations

from .contracts import AttributionLabel, OracleAttribution, OraclePairSpec, OracleSideRun


def attribute_pair(actual: OracleSideRun, oracle: OracleSideRun, *, spec: OraclePairSpec | None = None) -> OracleAttribution:
    """参数为 Actual/Oracle 冻结运行及可选配对合同；返回观察驱动的多候选诊断。"""
    if actual.status != "completed" or oracle.status != "completed":
        return OracleAttribution(status="not_measured", reason="one_side_did_not_complete")
    actual_stages = _stages(actual)
    oracle_stages = _stages(oracle)
    if not actual_stages or not oracle_stages:
        return OracleAttribution(status="missing", reason="stage_observations_missing")
    target_ids = _target_ids(spec)
    if not target_ids:
        return OracleAttribution(status="not_measured", reason="target_evidence_ids_missing")
    labels: list[AttributionLabel] = []
    evidence: list[str] = []
    if _target_missing(actual_stages, target_ids) and _target_present(oracle_stages, target_ids):
        labels.append("retrieval_gap")
        evidence.append("oracle_target_observed_actual_target_missing")
    elif _target_present(actual_stages, target_ids) and not _target_in_context(actual_stages, target_ids):
        labels.append("selection_gap")
        evidence.append("actual_target_retrieved_but_not_in_model_context")
    elif _target_in_context(actual_stages, target_ids) and _target_in_context(oracle_stages, target_ids):
        if _responses_differ(actual, oracle):
            labels.append("downstream_use_candidate")
            evidence.append("both_sides_exposed_target_with_different_outputs")
    if not labels:
        labels.append("unknown")
        evidence.append("observed_stages_do_not_support_specific_attribution")
    return OracleAttribution(status="measured", labels=tuple(labels), reason="stage_symptom_observed", evidence=tuple(evidence))


def _stages(run: OracleSideRun) -> dict[str, list[dict[str, Any]]]:
    """参数为一侧运行；返回按阶段分组的结构化观察。"""
    payload = _object(run.observation_snapshot)
    raw = payload.get("stage_observations")
    if isinstance(raw, list):
        return _group(raw)
    try:
        observations = extract_stage_observations(payload)
    except Exception:
        return {}
    return _group([item.model_dump(mode="json") for item in observations])


def _group(items: list[Any]) -> dict[str, list[dict[str, Any]]]:
    """参数为阶段观察；按阶段名称分组有效对象。"""
    result: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("stage"), str):
            result.setdefault(item["stage"], []).append(item)
    return result


def _object(value: str) -> dict[str, Any]:
    """参数为 JSON 快照；返回对象。"""
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("oracle observation must contain an object")
    return payload


def _target_ids(spec: OraclePairSpec | None) -> frozenset[str]:
    """参数为配对合同；返回控制条件声明的目标 Evidence 身份。"""
    if spec is None:
        return frozenset()
    payload = _object(spec.control_snapshot)
    values = payload.get("target_evidence_ids")
    return frozenset(str(item) for item in values if str(item).strip()) if isinstance(values, list) else frozenset()


def _target_present(stages: dict[str, list[dict[str, Any]]], target_ids: frozenset[str]) -> bool:
    """参数为阶段观察和目标身份；判断目标是否实际出现。"""
    return bool(_target_events(stages, target_ids))


def _target_missing(stages: dict[str, list[dict[str, Any]]], target_ids: frozenset[str]) -> bool:
    """参数为阶段观察和目标身份；判断检索阶段未出现目标。"""
    return not bool(_target_events({key: value for key, value in stages.items() if key in _RETRIEVAL_STAGES}, target_ids))


def _target_in_context(stages: dict[str, list[dict[str, Any]]], target_ids: frozenset[str]) -> bool:
    """参数为阶段观察和目标身份；判断目标进入 claim/final 模型上下文。"""
    return bool(_target_events({key: value for key, value in stages.items() if key in _CONTEXT_STAGES}, target_ids))


def _target_events(stages: dict[str, list[dict[str, Any]]], target_ids: frozenset[str]) -> list[dict[str, Any]]:
    """参数为阶段观察和目标身份；返回包含目标 Evidence 的结构化事件。"""
    events: list[dict[str, Any]] = []
    for items in stages.values():
        for item in items:
            if _contains_target(item, target_ids):
                events.append(item)
    return events


def _contains_target(value: Any, target_ids: frozenset[str]) -> bool:
    """参数为任意结构化观察和目标身份；递归检查 Evidence ID。"""
    if isinstance(value, dict):
        if any(str(value.get(key) or "") in target_ids for key in ("evidence_id", "chunk_id", "span_id")):
            return True
        return any(_contains_target(child, target_ids) for child in value.values())
    if isinstance(value, list):
        return any(_contains_target(child, target_ids) for child in value)
    return False


_RETRIEVAL_STAGES = frozenset({"query_raw", "query_filtered", "merged_hits", "ranked_hits", "evidence_pool", "unit_evidence_pool"})
_CONTEXT_STAGES = frozenset({"claim_input", "final_input"})


def _responses_differ(actual: OracleSideRun, oracle: OracleSideRun) -> bool:
    """参数为两侧运行；判断实际最终回答或模型上下文是否不同。"""
    return _response(actual) != _response(oracle)


def _response(run: OracleSideRun) -> str:
    """参数为一侧运行；读取最终观察中的实际回答文本。"""
    payload = _object(run.observation_snapshot)
    value = payload.get("response") or payload.get("answer")
    return value if isinstance(value, str) else ""
