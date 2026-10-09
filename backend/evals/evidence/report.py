"""冻结观察的逐调用 Evidence 评分与分层报告。作者：xxx。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import ValidationError

from evals.contracts import (
    MetricResult, build_grading, grading_projection, record_contracts,
    snapshot, snapshot_hash,
)
from evals.evidence.alignment import EvidenceDataError, validate_paper_requirements
from evals.evidence.contracts import (
    EffectiveScope, EvidenceBenchmark, EvidenceGradingPolicy, ObservedItem, StageObservation,
)
from evals.evidence.metrics import grade_stage, group_hits, ratio_metric

EVIDENCE_GRADER_VERSION = "evidence-benchmark-v1"
EVIDENCE_CANDIDATE_GRADER_VERSION = "evidence-candidate-diagnostic-v1"
_STATUSES = ("measured", "not_measured", "missing", "invalid_input")
_RETRIEVAL_STAGES = frozenset(("query_raw", "query_filtered", "merged_hits", "ranked_hits"))
_MODEL_STAGES = {"claim_context": "claim_input", "final_context": "final_input"}
_UNKNOWN_BACKEND = "runtime_backend_missing"


def _effective_scope(raw: Any) -> EffectiveScope:
    """参数为当次执行记录的范围；只识别显式合同或已实际生效的本地检索范围。"""
    if not isinstance(raw, dict):
        return EffectiveScope(status="missing")
    if "status" in raw:
        return EffectiveScope.model_validate(raw)
    if raw.get("local_retrieval_allowed") and raw.get("user_id") and raw.get("filter_expr"):
        return EffectiveScope.model_validate({"status": "constrained", "allowed_paper_ids": raw.get("paper_ids")})
    return EffectiveScope(status="missing")


def _observed_item(raw: dict[str, Any], *, kind: str) -> ObservedItem:
    """参数为真实 hit/证据行及阶段类型；仅提取实际可见正文和身份。"""
    payload_kind = "child"
    text = raw.get("text") if "text" in raw else raw.get("display_text")
    if kind == "pool":
        text = raw.get("expanded_text") or raw.get("excerpt") or raw.get("summary")
        # 1. 证据转换可能截断展开父块，只将实际保留的正文映射回原文
        payload_kind = "model_excerpt"
    if kind == "model":
        text, payload_kind = raw.get("source_text"), "model_excerpt"
    values = {key: raw[key] if key in raw else None for key in ("paper_id", "chunk_id", "parent_id", "evidence_id", "index_snapshot_id")}
    rank = raw["rank"] if "rank" in raw else None
    score = raw["score"] if "score" in raw else None
    return ObservedItem.model_validate({
        **values, "payload_kind": payload_kind, "text_snapshot": text,
        "rank": rank if kind == "retrieval" else None, "score": score,
    })


def _declared_index(events: list[dict[str, Any]]) -> str | None:
    """参数为冻结事件；只接受实际执行前声明的一份索引身份。"""
    declarations = [event for event in events if event.get("stage") == "index_snapshot_declared"]
    if not declarations:
        return None
    if len(declarations) != 1 or events[0] is not declarations[0]:
        raise EvidenceDataError("invalid_input", "index_declaration_not_unique_or_not_before_execution")
    payload = declarations[0]["payload"]
    identity = payload.get("index_snapshot_id")
    if payload.get("source") != "eval_declared_before_execution" or not isinstance(identity, str) or not identity.strip():
        raise EvidenceDataError("invalid_input", "index_declaration_invalid")
    return identity


def _validate_event(event: Any) -> None:
    """参数为外部事件；校验字段类型及 scope 请求身份，避免畸形快照触发程序异常。"""
    if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
        raise EvidenceDataError("invalid_input", "evidence_observation_shape_invalid")
    if any(not isinstance(event.get(key), str) or not event[key] for key in ("event_id", "request_id", "stage")):
        raise EvidenceDataError("invalid_input", "evidence_observation_identity_invalid")
    if type(event.get("observed_at_ns")) is not int or event["observed_at_ns"] < 0:
        raise EvidenceDataError("invalid_input", "evidence_observation_time_invalid")
    scope = event.get("scope")
    if scope is None:
        return
    if not isinstance(scope, dict) or any(key not in scope for key in ("scope_id", "request_id", "parent_scope_id", "stage", "unit_id")):
        raise EvidenceDataError("invalid_input", "evidence_scope_identity_invalid")
    if any(not isinstance(scope[key], str) or not scope[key] for key in ("scope_id", "request_id", "stage")):
        raise EvidenceDataError("invalid_input", "evidence_scope_identity_invalid")
    if any(value is not None and (not isinstance(value, str) or not value.strip())
           for value in (scope["parent_scope_id"], scope["unit_id"])):
        raise EvidenceDataError("invalid_input", "evidence_scope_link_invalid")
    if scope["request_id"] != event["request_id"]:
        raise EvidenceDataError("invalid_input", "evidence_scope_request_mismatch")


def _capture_context(observed: dict[str, Any]) -> dict[str, Any]:
    """参数为原始观察；校验事件容器、唯一身份和关联，返回只读装配索引。"""
    events = observed.get("evidence_observations")
    if not isinstance(events, list):
        raise EvidenceDataError("missing", "evidence_observations_missing")
    for event in events:
        _validate_event(event)
    if any(event.get("schema_version") != "evidence-observation-v1" for event in events):
        raise EvidenceDataError("invalid_input", "evidence_observation_version_invalid")
    identities = [event.get("event_id") for event in events]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
        raise EvidenceDataError("invalid_input", "evidence_event_identity_invalid")
    request_ids = {event.get("request_id") for event in events}
    if len(request_ids) > 1:
        raise EvidenceDataError("invalid_input", "evidence_request_identity_mismatch")
    scoped_events = [event for event in events if isinstance(event.get("scope"), dict)]
    scopes = {event["scope"]["scope_id"]: event["scope"] for event in scoped_events}
    failed = {event["scope"]["scope_id"] for event in events if event.get("stage") == "scope_failed" and event.get("scope")}
    errors = observed.get("evidence_capture_errors", [])
    if not isinstance(errors, list) or any(not isinstance(error, dict) for error in errors):
        raise EvidenceDataError("invalid_input", "evidence_capture_errors_invalid")
    for error in errors:
        identity = error.get("scope_id")
        if identity is not None and (not isinstance(identity, str) or not identity.strip()):
            raise EvidenceDataError("invalid_input", "evidence_capture_scope_invalid")
        if not isinstance(error.get("stage"), str) or not error["stage"].strip():
            raise EvidenceDataError("invalid_input", "evidence_capture_stage_invalid")
    error_scopes = {(error.get("scope_id"), error.get("stage")) for error in errors}
    return {"events": events, "scopes": scopes, "failed": failed, "error_scopes": error_scopes,
            "index_id": _declared_index(events), "observed": observed}


def _event_stage(event: dict[str, Any], context: dict[str, Any]) -> StageObservation:
    """参数为一次检索或证据池事件和关联信息；返回保留原调用身份的阶段。"""
    if not isinstance(event.get("stage"), str):
        raise EvidenceDataError("invalid_input", "evidence_stage_missing")
    payload = event["payload"]
    scope = event.get("scope") or {}
    is_retrieval = event["stage"] in _RETRIEVAL_STAGES
    raw_items = payload.get("hits" if is_retrieval else "records")
    if raw_items is not None and (not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items)):
        raise EvidenceDataError("invalid_input", "stage_items_invalid")
    status = "complete" if raw_items is not None else "missing"
    if (scope.get("scope_id"), event["stage"]) in context["error_scopes"]:
        status = "failed"
    return StageObservation.model_validate({
        "stage": event["stage"] if is_retrieval else "evidence_pool", "call_id": event["event_id"],
        "scope_id": scope.get("scope_id"), "parent_scope_id": scope.get("parent_scope_id"), "unit_id": scope.get("unit_id"),
        "query": payload.get("query"), "retrieval_backend": payload.get("retrieval_backend") or _scope_backend(scope.get("scope_id"), context),
        "status": status, "reason": "stage_capture_failed" if status == "failed" else None,
        "items": tuple(_observed_item(item, kind="retrieval" if is_retrieval else "pool") for item in raw_items) if raw_items is not None else None,
        "scope": _effective_scope(payload.get("scope")),
        "index_snapshot_id": payload.get("index_snapshot_id") or context["index_id"],
    })


def _scope_backend(scope_id: str | None, context: dict[str, Any]) -> str:
    """参数为真实 scope 与事件索引；仅从关联取证调用或祖先范围的事实解析来源。"""
    backends: set[str] = set()
    for event in context["events"]:
        details = event["payload"].get("details")
        declared = details.get("retrieval_backend") if event["stage"] == "scope_start" and isinstance(details, dict) else None
        backend = event["payload"].get("retrieval_backend") or declared
        if not isinstance(backend, str) or not backend:
            continue
        current = (event.get("scope") or {}).get("scope_id")
        visited: set[str] = set()
        while current and current not in visited:
            if current == scope_id:
                backends.add(backend)
                break
            visited.add(current)
            current = context["scopes"].get(current, {}).get("parent_scope_id")
    if not backends:
        ancestors = _scope_ancestors(scope_id, context)
        backends = {event["payload"]["details"]["retrieval_backend"] for event in context["events"]
                    if event["stage"] == "scope_start" and (event.get("scope") or {}).get("scope_id") in ancestors
                    and isinstance(event["payload"].get("details"), dict)
                    and isinstance(event["payload"]["details"].get("retrieval_backend"), str)}
    if len(backends) == 1:
        return next(iter(backends))
    return "mixed_retrieval_backends" if backends else _UNKNOWN_BACKEND


def _scope_ancestors(scope_id: str | None, context: dict[str, Any]) -> set[str]:
    """参数为 scope 与关联表；返回祖先集合，循环身份明确判为无效输入。"""
    ancestors: set[str] = set()
    current = scope_id
    while current:
        if current in ancestors:
            raise EvidenceDataError("invalid_input", "evidence_scope_parent_cycle")
        ancestors.add(current)
        current = context["scopes"].get(current, {}).get("parent_scope_id")
    return ancestors


def _failed_query_stages(context: dict[str, Any]) -> list[tuple[StageObservation, str]]:
    """参数为真实失败 scope；保留检索尚未返回或后续过滤失败的具体阶段。"""
    stages: list[tuple[StageObservation, str]] = []
    for event in context["events"]:
        scope = event.get("scope") or {}
        if event["stage"] != "scope_failed" or scope.get("stage") != "retrieval_query":
            continue
        observed = {item["stage"] for item in context["events"] if (item.get("scope") or {}).get("scope_id") == scope["scope_id"]}
        missing = next((stage for stage in ("query_raw", "query_filtered") if stage not in observed), None)
        if missing is None:
            continue
        stages.append((StageObservation.model_validate({
            "stage": missing, "call_id": event["event_id"], "scope_id": scope["scope_id"],
            "parent_scope_id": scope.get("parent_scope_id"), "unit_id": scope.get("unit_id"),
            "retrieval_backend": _scope_backend(scope["scope_id"], context), "status": "failed",
            "reason": f"retrieval_scope_failed:{event['payload'].get('error_type', 'unknown')}",
        }), missing))
    return stages


def _capture_gap_stages(context: dict[str, Any]) -> list[tuple[StageObservation, str]]:
    """参数为实际采集错误；为缺失的整次检索或证据池保留独立阶段，避免均值忽略失败调用。"""
    stages: list[tuple[StageObservation, str]] = []
    for number, error in enumerate(context["observed"].get("evidence_capture_errors", [])):
        raw_stage = error.get("stage")
        if not isinstance(raw_stage, str) or not (raw_stage in _RETRIEVAL_STAGES or raw_stage.endswith("_evidence_pool")):
            continue
        scope_id = error.get("scope_id")
        stages.append((StageObservation.model_validate({
            "stage": raw_stage if raw_stage in _RETRIEVAL_STAGES else "evidence_pool",
            "call_id": f"capture-error-{number}", "scope_id": scope_id,
            "retrieval_backend": _scope_backend(scope_id, context), "status": "missing", "reason": "stage_capture_failed",
        }), raw_stage))
    return stages


def _message_texts(messages: Any) -> list[str]:
    """参数为实际模型消息；仅返回文本内容，不将元数据序列化为证据正文。"""
    if not isinstance(messages, list):
        raise EvidenceDataError("missing", "model_input_messages_missing")
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise EvidenceDataError("invalid_input", "model_input_message_invalid")
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return texts


def _model_stage(call: dict[str, Any], context: dict[str, Any]) -> StageObservation | None:
    """参数为真实模型调用和同请求事件；核对已发送文本后返回独立上下文阶段。"""
    scope_id = call.get("evidence_scope_id")
    scope = context["scopes"].get(scope_id)
    if scope is None or scope.get("stage") not in _MODEL_STAGES:
        return None
    evidence = [event for event in context["events"] if event["stage"] == "model_evidence_text"
                and (event.get("scope") or {}).get("scope_id") == scope_id]
    messages = _message_texts(call.get("input_messages"))
    payloads = [event["payload"] for event in evidence]
    valid_texts = all(isinstance(item.get("source_text"), str) and isinstance(item.get("rendered_text"), str)
                      and item["source_text"] in item["rendered_text"] for item in payloads)
    attempts = call.get("http_attempt_count")
    sent = call.get("cache_hit") is True or (type(attempts) is int and attempts > 0)
    texts_present = bool(messages) and sent and valid_texts and all(any(item["rendered_text"] in text for text in messages) for item in payloads)
    status = "complete" if texts_present else "missing"
    capture_failed = any((scope_id, stage) in context["error_scopes"] for stage in ("model_evidence_text", "model_call_observations"))
    if call.get("error") or scope_id in context["failed"] or capture_failed:
        status = "failed"
    return StageObservation.model_validate({
        "stage": _MODEL_STAGES[scope["stage"]], "call_id": call.get("call_id"), "scope_id": scope_id,
        "parent_scope_id": scope.get("parent_scope_id"), "unit_id": scope.get("unit_id"),
        "retrieval_backend": "model_context", "status": status,
        "reason": "model_visible_text_not_in_messages" if not texts_present else None,
        "items": tuple(_observed_item(item, kind="model") for item in payloads),
        "index_snapshot_id": context["index_id"],
    })


def _stage_observations(observed: dict[str, Any]) -> list[tuple[StageObservation, str]]:
    """参数为冻结原始观察；枚举每次真实阶段，保留未关联模型调用的缺口。"""
    context = _capture_context(observed)
    result: list[tuple[StageObservation, str]] = []
    for event in context["events"]:
        stage = event["stage"]
        if stage in _RETRIEVAL_STAGES or stage.endswith("_evidence_pool"):
            result.append((_event_stage(event, context), stage))
    calls = observed.get("model_call_observations", [])
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise EvidenceDataError("invalid_input", "model_call_observations_invalid")
    identities = [call.get("call_id") for call in calls]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
        raise EvidenceDataError("invalid_input", "model_call_identity_invalid")
    for call in calls:
        scope_id = call.get("evidence_scope_id")
        if scope_id is not None and (not isinstance(scope_id, str) or not scope_id.strip()):
            raise EvidenceDataError("invalid_input", "model_call_scope_invalid")
        stage = _model_stage(call, context)
        if stage is not None:
            result.append((stage, stage.stage))
    associated = {call.get("evidence_scope_id") for call in calls}
    for scope_id, scope in context["scopes"].items():
        if scope.get("stage") in _MODEL_STAGES and scope_id not in associated:
            result.append((StageObservation.model_validate({
                "stage": _MODEL_STAGES[scope["stage"]], "call_id": scope_id, "scope_id": scope_id,
                "retrieval_backend": "model_context", "status": "missing", "reason": "actual_model_call_missing",
            }), _MODEL_STAGES[scope["stage"]]))
    return [*result, *_capture_gap_stages(context), *_failed_query_stages(context)]


def _group_breakdowns(benchmark: EvidenceBenchmark, grade: Any) -> list[dict[str, Any]]:
    """参数为真值与阶段结果；按证据等级和每篇显式证明要求保留真实计数。"""
    template = next(metric for metric in grade.measurements if metric.name.startswith("evidence_group_"))
    subsets = [("importance", importance, tuple(group for group in benchmark.groups if group.importance == importance))
               for importance in ("critical", "important", "supporting")]
    subsets.extend(("paper", item.paper_id, item.groups) for item in benchmark.paper_requirements or ())
    paper_error = None
    try:
        validate_paper_requirements(benchmark)
    except EvidenceDataError as exc:
        paper_error = MetricResult(name=template.name, status=exc.status, reason=exc.reason)
    rows: list[dict[str, Any]] = []
    for dimension, label, groups in subsets:
        if dimension == "paper" and paper_error is not None:
            metric = paper_error
        elif template.status == "measured":
            covered = frozenset(grade.covered_span_ids)
            metric = ratio_metric(template.name, len(group_hits(groups, covered)), len(groups))
        else:
            metric = MetricResult(name=template.name, status=template.status, reason=template.reason)
        rows.append({"dimension": dimension, "label": label, "measurements": [metric.model_dump(mode="json")]})
    missing_papers = set(benchmark.target_paper_ids) - {item.paper_id for item in benchmark.paper_requirements or ()}
    for paper_id in sorted(missing_papers):
        metric = MetricResult(name=template.name, status="missing", reason="paper_requirements_missing")
        rows.append({"dimension": "paper", "label": paper_id, "measurements": [metric.model_dump(mode="json")]})
    return rows


def _grading_version(policy: EvidenceGradingPolicy) -> str:
    """参数为显式审核策略；返回对应评分版本，未知策略拒绝。"""
    if policy == "reviewed":
        return EVIDENCE_GRADER_VERSION
    if policy == "candidate_diagnostic":
        return EVIDENCE_CANDIDATE_GRADER_VERSION
    raise ValueError(f"unsupported evidence grading policy: {policy}")


def _error_row(
    record: dict[str, Any], *, status: str, reason: str,
    grading_policy: EvidenceGradingPolicy = "reviewed",
) -> dict[str, Any]:
    """参数为冻结运行与已识别输入问题；返回无虚构分数的独立评分行。"""
    case, run = record_contracts(record)
    metric = MetricResult.model_validate({"name": "evidence_observation_integrity", "status": status, "reason": reason})
    return {"case_id": case.case_id, "run_id": run.run_id, "stage": "observation_integrity", "call_id": None,
            "source_type": case.source_type, "retrieval_backend": "unknown", "task_family": None,
            "measurements": [metric.model_dump(mode="json")], "scope_violations": [], "breakdowns": [],
            "grading_result": build_grading(record, [metric], grader="evidence-benchmark", grader_version=_grading_version(grading_policy))}


def grade_record_stages(
    record: dict[str, Any], projection: dict[str, Any], *,
    grading_policy: EvidenceGradingPolicy = "reviewed",
) -> list[dict[str, Any]]:
    """参数为冻结记录及评分投影；生成独立逐阶段分数，原记录与旧评分保持不可变。"""
    raw = projection["truth"].get("evidence_benchmark")
    if raw is None:
        return [_error_row(record, status="missing", reason="evidence_benchmark_missing", grading_policy=grading_policy)]
    try:
        benchmark = EvidenceBenchmark.model_validate(raw)
        observations = _stage_observations(projection["observation"])
    except (EvidenceDataError, ValidationError) as exc:
        return [_error_row(record, status=exc.status if isinstance(exc, EvidenceDataError) else "invalid_input",
                           reason=str(exc), grading_policy=grading_policy)]
    case, run = record_contracts(record)
    rows: list[dict[str, Any]] = []
    for observation, raw_stage in observations:
        grade = grade_stage(benchmark, observation, policy=grading_policy)
        grading = build_grading(record, list(grade.measurements), grader="evidence-benchmark", grader_version=_grading_version(grading_policy))
        rows.append({
            "case_id": case.case_id, "run_id": run.run_id, "stage": observation.stage, "raw_stage": raw_stage,
            "call_id": observation.call_id, "scope_id": observation.scope_id, "parent_scope_id": observation.parent_scope_id,
            "unit_id": observation.unit_id, "query": observation.query,
            "top_k": benchmark.ranking_policy.top_k if benchmark.ranking_policy and observation.stage in _RETRIEVAL_STAGES else None,
            "source_type": case.source_type, "retrieval_backend": observation.retrieval_backend,
            "task_family": benchmark.task_family, "review_status": benchmark.review_status, "dataset_version": benchmark.dataset_version,
            "source_snapshot_ids": [source.snapshot_id for source in benchmark.sources], "index_snapshot_id": observation.index_snapshot_id,
            "observation_hash": snapshot_hash(snapshot(observation.model_dump(mode="json"))),
            **grade.model_dump(mode="json"), "grading_result": grading, "breakdowns": _group_breakdowns(benchmark, grade),
        })
    if not observations:
        rows.append(_error_row(record, status="missing", reason="evidence_stages_not_observed", grading_policy=grading_policy))
    for error in projection["observation"].get("evidence_capture_errors", []):
        rows.append(_error_row(record, status="missing", reason=f"evidence_capture_error:{snapshot(error)}", grading_policy=grading_policy))
    if projection["observation"].get("failure_reason"):
        rows.append(_error_row(record, status="missing", reason=f"run_failed:{projection['observation']['failure_reason']}",
                               grading_policy=grading_policy))
    return rows


def extract_stage_observations(observed: dict[str, Any]) -> tuple[StageObservation, ...]:
    """参数为冻结 Evidence 观察；返回实际出现的阶段观察，不补造缺失阶段。"""
    return tuple(observation for observation, _raw_stage in _stage_observations(observed))


def summarize_measurement_states(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """参数为逐调用结果；按指标保留四种状态计数，不把缺失当作零分。"""
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric in row["measurements"]:
            counters[metric["name"]][metric["status"]] += 1
    return {name: {f"{status}_count": counts[status] for status in _STATUSES} for name, counts in sorted(counters.items())}


def _case_aggregate(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为同题同指标的调用结果；每题等权，缺口存在时不生成完整质量均值。"""
    measurements = [row["measurement"] for row in rows]
    counts = Counter(metric["status"] for metric in measurements)
    complete = counts["measured"] == len(measurements)
    values = [metric["value"] for metric in measurements if metric["status"] == "measured"]
    scope_violation = any(metric["name"] == "scope_adherence" and metric["value"] == 0 for metric in measurements)
    mean = sum(values) / len(values) if complete and values else None
    if scope_violation:
        mean = 0.0
    return {"case_id": case_id, "value": mean, "complete": complete, "scope_violation": scope_violation,
            "coverage": {status: counts[status] for status in _STATUSES}, "observations": rows}


def group_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为逐调用明细；按题先聚合后分层，调用多的题不会获得额外权重。"""
    buckets: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        subsets = [
            {"dimension": "task_family", "label": row.get("task_family") or "unlabeled", "measurements": row["measurements"]},
            {"dimension": "source_type", "label": row.get("source_type") or "unlabeled", "measurements": row["measurements"]},
            *row["breakdowns"],
        ]
        for subset in subsets:
            for metric in subset["measurements"]:
                key = (subset["dimension"], subset["label"], row["stage"], row["retrieval_backend"], metric["name"],
                       row.get("source_type") or "unlabeled", row.get("review_status") or "unlabeled",
                       row.get("grading_result", {}).get("grader_version", "unrecorded"))
                buckets[key][row["case_id"]].append({"run_id": row["run_id"], "call_id": row["call_id"], "measurement": metric})
    groups: list[dict[str, Any]] = []
    for group_key, by_case in sorted(buckets.items()):
        cases = [_case_aggregate(case_id, values) for case_id, values in sorted(by_case.items())]
        values = [case["value"] for case in cases if case["value"] is not None]
        violation_count = sum(case["scope_violation"] for case in cases)
        groups.append({"dimension": group_key[0], "label": group_key[1], "stage": group_key[2], "retrieval_backend": group_key[3],
                       "metric": group_key[4], "source_type": group_key[5], "review_status": group_key[6],
                       "grader_version": group_key[7], "case_count": len(cases), "measured_case_count": len(values),
                       "mean": 0.0 if violation_count else (sum(values) / len(values) if values else None),
                       "scope_violation_case_count": violation_count, "case_results": cases})
    return groups


def build_evidence_report(
    records: list[dict[str, Any]], *, grading_policy: EvidenceGradingPolicy = "reviewed",
) -> dict[str, Any]:
    """参数为冻结运行；独立计算逐阶段与逐题分组结果，并保存原投影校验值。"""
    rows: list[dict[str, Any]] = []
    for record in records:
        case, run = record_contracts(record)
        rows.extend(grade_record_stages(record, grading_projection(case, run), grading_policy=grading_policy))
    violations = [{key: row.get(key) for key in ("case_id", "run_id", "stage", "call_id", "scope_violations")}
                  for row in rows if row["scope_violations"]]
    return {"grader_version": _grading_version(grading_policy), "grading_policy": grading_policy, "records": rows,
            "coverage": summarize_measurement_states(rows), "groups": group_report_rows(rows),
            "scope_violations": violations, "scope_violation_count": len(violations),
            "case_count": len({row["case_id"] for row in rows})}


def render_evidence_report(report: dict[str, Any]) -> list[str]:
    """参数为独立 Evidence 报告；返回 Markdown 行，明确呈现数据层、阶段和测量缺口。"""
    lines = ["", "## Evidence Benchmark", "", f"Grader: `{report['grader_version']}`",
             f"Policy: `{report.get('grading_policy', 'reviewed')}`; candidate remains unreviewed",
             f"Cases: {report['case_count']}; scope violations: {report['scope_violation_count']}", "",
             "| Metric | Measured | Not Measured | Missing | Invalid Input |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, counts in report["coverage"].items():
        lines.append(f"| {name} | " + " | ".join(str(counts[f"{status}_count"]) for status in _STATUSES) + " |")
    lines.extend(["", "### Stage And Source Results", "",
                  "| Case | Stage | Source | Metric | Status | Value | Counts | Reason |",
                  "| --- | --- | --- | --- | --- | ---: | --- | --- |"])
    for row in report["records"]:
        for metric in row["measurements"]:
            counts = f"{metric['numerator']}/{metric['denominator']}" if metric.get("denominator") is not None else ""
            cells = [row["case_id"], row["stage"], row["retrieval_backend"], metric["name"], metric["status"],
                     "" if metric["value"] is None else str(metric["value"]), counts, metric["reason"]]
            lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    return lines
