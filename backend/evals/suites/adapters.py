"""读取原报告并校验冻结运行，不调用评分器或当前运行配置。作者：xxx。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.agent_outcome.contracts import agent_record_contracts
from evals.contracts import (
    GradingResult, MetricResult, grading_projection, read_snapshot, rebuild_grading_input,
    record_contracts, snapshot, snapshot_hash,
)
from evals.evidence.contracts import StageObservation

from .contracts import (
    IDENTITY_NAMES, FrozenReportRef, IdentityCondition, IdentityName, ImportedRun,
    RecordedIdentity, ScoreSlice, SuiteInput, SuiteManifest,
)

_REPLAY_MODES = frozenset(("planner_replay", "planner_live_fixture", "agent_fixture"))
_OPTIONAL_FIXTURE_CONDITIONS = frozenset(("source", "index", "model", "embedding", "prompt", "retrieval"))
_RETRIEVAL_STAGES = frozenset(("query_raw", "query_filtered", "merged_hits", "ranked_hits"))


@dataclass(frozen=True)
class LoadedSource:
    """一次读取的原文件快照；解析时返回新对象，避免修改原文件内容。"""

    reference: FrozenReportRef
    report_snapshot: str
    records_snapshot: str
    identity: RecordedIdentity | None
    ragas_snapshot: str | None = None


def _object(value: Any, *, location: str) -> dict[str, Any]:
    """参数为外部 JSON 值和位置；只接受对象并返回独立字典。"""
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    return dict(value)


def _objects(value: Any, *, location: str) -> tuple[dict[str, Any], ...]:
    """参数为外部数组和位置；逐条验证对象，不丢弃畸形行。"""
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected an array")
    return tuple(_object(item, location=f"{location}[{index}]") for index, item in enumerate(value))


def _verified_text(path: Path, expected_digest: str) -> str:
    """参数为原文件和预期校验和；验证原始字节后返回 UTF-8 文本。"""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return content.decode("utf-8-sig")


def _resolve_reference(reference: FrozenReportRef, base_dir: Path) -> FrozenReportRef:
    """参数为引用及清单目录；返回绝对路径引用，不改变原清单对象。"""
    values = reference.model_dump()
    for name in ("report_path", "frozen_path", "identity_path", "ragas_path"):
        value = values[name]
        if value is not None:
            values[name] = str((base_dir / value).resolve())
    return FrozenReportRef.model_validate(values)


def read_frozen_source(reference: FrozenReportRef, *, base_dir: Path) -> LoadedSource:
    """参数为原产物引用和清单目录；返回验证后的原报告、记录及历史条件。"""
    resolved = _resolve_reference(reference, base_dir)
    report = read_snapshot(_verified_text(Path(resolved.report_path), resolved.report_sha256))
    if resolved.frozen_path is not None and resolved.frozen_sha256 is not None:
        text = _verified_text(Path(resolved.frozen_path), resolved.frozen_sha256)
        records = [read_snapshot(line) for line in text.splitlines() if line.strip()]
    else:
        records = list(_objects(report.get("records"), location="report.records"))
    identity = None
    if resolved.identity_path is not None and resolved.identity_sha256 is not None:
        text = _verified_text(Path(resolved.identity_path), resolved.identity_sha256)
        identity = RecordedIdentity.model_validate_json(text)
        if (identity.report_sha256, identity.frozen_sha256) != (resolved.report_sha256, resolved.frozen_sha256):
            raise ValueError("recorded identity is not bound to the source artifacts")
    ragas = None
    if resolved.ragas_path is not None and resolved.ragas_sha256 is not None:
        ragas = snapshot(read_snapshot(_verified_text(Path(resolved.ragas_path), resolved.ragas_sha256)))
        attempt = read_snapshot(ragas).get("attempt")
        if attempt is not None and attempt.get("frozen_sha256") != resolved.frozen_sha256:
            raise ValueError("grading_attempt_frozen_checksum_mismatch")
    return LoadedSource(resolved, snapshot(report), snapshot({"records": records}), identity, ragas)


def validate_original_grading(record: dict[str, Any], grading: GradingResult) -> None:
    """参数为原记录和原评分；核对案例、运行及投影校验和，不重新评分。"""
    rebuild_grading_input({**record, "grading_result": grading.model_dump(mode="json")})
    names = [measurement.name for measurement in grading.measurements]
    if len(names) != len(set(names)):
        raise ValueError("duplicate metric in original grading")


def _record_index(source: LoadedSource) -> dict[str, dict[str, Any]]:
    """参数为已读取来源；按真实 run_id 建立唯一索引，重复记录显式拒绝。"""
    records = _objects(read_snapshot(source.records_snapshot)["records"], location="frozen records")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        _case, run = record_contracts(record)
        if run.run_id in result:
            raise ValueError(f"duplicate frozen run: {run.run_id}")
        result[run.run_id] = record
    if source.identity is not None:
        unknown = {item.run_id for item in source.identity.runs} - result.keys()
        if unknown:
            raise ValueError(f"identity file references unknown runs: {sorted(unknown)}")
    return result


def _validate_agent(record: dict[str, Any]) -> None:
    """参数为 Agent 原记录；核对专属合同与通用合同，防止两份观察互相冒用。"""
    case, run = record_contracts(record)
    agent_case, agent_run = agent_record_contracts(record)
    if (agent_case.case_id, agent_case.case_version) != (case.case_id, case.case_version):
        raise ValueError("agent and generic case identity mismatch")
    fields = ("run_id", "case_id", "case_version", "run_mode", "input_snapshot", "observation_snapshot", "missing_fields")
    if any(getattr(agent_run, field) != getattr(run, field) for field in fields):
        raise ValueError("agent and generic run mismatch")
    if (agent_case.input_snapshot, agent_case.truth_snapshot) != (case.input_snapshot, case.truth_snapshot):
        raise ValueError("agent and generic case snapshot mismatch")
    if read_snapshot(agent_run.observation_snapshot).get("status") != agent_run.status:
        raise ValueError("agent status contradicts its frozen observation")
    if run.run_mode == "agent_fixture" and run.fixture_snapshot != agent_run.environment_snapshot:
        raise ValueError("agent fixture environment mismatch")


def _topology(record: dict[str, Any]) -> tuple[str, str]:
    """参数为原冻结记录；返回原审核状态与可观察的论文、任务及证据分组。"""
    case, run = record_contracts(record)
    truth = read_snapshot(case.truth_snapshot)
    inputs = read_snapshot(run.input_snapshot)
    benchmark = truth.get("evidence_benchmark")
    labels = _object(benchmark, location="evidence_benchmark") if isinstance(benchmark, dict) else {}
    agent_case = record.get("agent_case")
    if isinstance(agent_case, dict):
        labels = {**labels, **agent_case}
    review = labels.get("review_status", "unlabeled")
    if review not in {"candidate", "reviewed", "rejected", "unlabeled"}:
        raise ValueError("unknown source review status")
    topology = {
        "task_family": labels.get("task_family", record.get("category")),
        "paper_ids": labels.get("target_paper_ids", inputs.get("paper_ids")),
        "paper_keys": inputs.get("paper_keys"),
        "group_id": record.get("group_id"),
        "evidence_groups": labels.get("groups"),
        "critical_facts": truth.get("critical_facts"),
        "answerable": truth.get("answerable"),
    }
    return str(review), snapshot(topology)


def _recorded(name: IdentityName, payload: dict[str, Any]) -> IdentityCondition:
    """参数为条件名称及原记录内容；返回规范化的已记录身份。"""
    return IdentityCondition(name=name, status="recorded", snapshot=snapshot(payload), reason="original_frozen_record")


def stable_execution_input(value: str) -> str:
    """参数为真实输入快照；返回去除顶层运行身份和 publication 临时租约的比较副本。"""
    def project(payload: Any) -> Any:
        """参数为输入节点；返回递归复制后的稳定授权与批次字段。"""
        if isinstance(payload, list):
            return [project(item) for item in payload]
        if not isinstance(payload, dict):
            return payload
        result = {key: project(item) for key, item in payload.items()}
        publication = result.get("publication")
        if isinstance(publication, dict):
            result["publication"] = {key: item for key, item in publication.items() if key != "lease_id"}
        return result

    # 1. 【评测比较】【执行身份】新研究编号不改变问题条件，嵌套业务编号及原始快照仍完整保留
    inputs = {key: item for key, item in read_snapshot(value).items() if key != "run_id"}
    return snapshot(project(inputs))


def _intrinsic_conditions(record: dict[str, Any], manifest: SuiteManifest) -> dict[IdentityName, IdentityCondition]:
    """参数为原记录和清单；仅从冻结数据构造输入、真值、原文及索引身份。"""
    case, run = record_contracts(record)
    truth = read_snapshot(case.truth_snapshot)
    result: dict[IdentityName, IdentityCondition] = {
        "dataset": _recorded("dataset", {"suite_dataset_version": manifest.dataset_version, "case_version": case.case_version}),
        "input": _recorded("input", {"case_input": stable_execution_input(case.input_snapshot),
                                      "execution_input": stable_execution_input(run.input_snapshot)}),
        "truth": _recorded("truth", truth),
    }
    benchmark = truth.get("evidence_benchmark")
    if isinstance(benchmark, dict) and benchmark.get("sources"):
        result["source"] = _recorded("source", {"sources": benchmark["sources"]})
    observed = read_snapshot(run.observation_snapshot)
    observations = observed.get("evidence_observations")
    if run.run_mode in {"agent_live", "agent_fixture"}:
        observations = [event for turn in observed.get("turns", [])
                        for event in read_snapshot(turn.get("evidence_observation_snapshot", "{}")).get("evidence_observations", [])]
    if isinstance(observations, list):
        indices = {event["payload"]["index_snapshot_id"] for event in observations
                   if isinstance(event, dict) and isinstance(event.get("payload"), dict)
                   and isinstance(event["payload"].get("index_snapshot_id"), str)}
        if indices:
            result["index"] = _recorded("index", {"index_snapshot_ids": sorted(indices)})
    return result


def _conditions(record: dict[str, Any], source: LoadedSource, manifest: SuiteManifest) -> tuple[IdentityCondition, ...]:
    """参数为原记录、来源和清单；合并已记录条件，缺少历史配置时显式保留缺口。"""
    case, run = record_contracts(record)
    observed = read_snapshot(run.observation_snapshot)
    pure_retrieval = (run.run_mode == "retrieval_live" and observed.get("model_call_observations") == []
                      and "model_call_observations" not in run.missing_fields)
    if run.run_mode == "retrieval_live" and observed.get("model_call_observations"):
        raise ValueError("retrieval_live_contains_generation_calls")
    conditions = _intrinsic_conditions(record, manifest)
    historical = () if source.identity is None else tuple(
        condition for item in source.identity.runs if item.run_id == run.run_id for condition in item.conditions
    )
    for condition in historical:
        existing = conditions.get(condition.name)
        if existing is not None:
            if (existing.status, existing.snapshot) != (condition.status, condition.snapshot):
                raise ValueError(f"historical condition contradicts frozen {condition.name}: {run.run_id}")
            continue
        if condition.status == "not_applicable":
            fixture_allowed = (run.run_mode in _REPLAY_MODES and case.source_type in {"fixture", "agent_scenario"}
                               and condition.name in _OPTIONAL_FIXTURE_CONDITIONS)
            retrieval_allowed = pure_retrieval and condition.name in {"model", "prompt"}
            if not fixture_allowed and not retrieval_allowed:
                raise ValueError(f"condition cannot be not_applicable: {condition.name}")
        conditions[condition.name] = condition
    return tuple(conditions.get(name, IdentityCondition(
        name=name, status="missing", reason="historical_condition_not_recorded",
    )) for name in IDENTITY_NAMES)


def _import_run(record: dict[str, Any], source: LoadedSource, manifest: SuiteManifest) -> ImportedRun:
    """参数为原记录、来源和清单；构造只读运行载体，审核资格保持原样。"""
    case, run = record_contracts(record)
    review, topology = _topology(record)
    return ImportedRun.model_validate({
        "case": case, "run": run, "record_snapshot": snapshot(record),
        "review_status": review, "topology_snapshot": topology,
        "conditions": _conditions(record, source, manifest),
    })


def _validate_row(row: dict[str, Any], record: dict[str, Any], grading: GradingResult) -> None:
    """参数为原报告行、冻结记录及评分；检查重复展示的身份和测量是否一致。"""
    validate_original_grading(record, grading)
    case, run = record_contracts(record)
    if "eval_case" in row or "run_record" in row:
        if record_contracts(row) != (case, run):
            raise ValueError("report and companion frozen contracts differ")
    identities = (
        ("case_id", grading.case_id, "case identity"), ("run_id", grading.run_id, "run identity"),
        ("case_version", case.case_version, "case version"), ("source_type", case.source_type, "source metadata"),
        ("review_status", _topology(record)[0], "review status"),
        ("run_status", read_snapshot(run.observation_snapshot).get("status"), "status"),
    )
    for field, expected, label in identities:
        if field in row and row[field] != expected:
            raise ValueError(f"report row {label} contradicts frozen record")
    if "measurements" in row:
        values = _objects(row["measurements"], location="row.measurements")
        if tuple(MetricResult.model_validate(value) for value in values) != grading.measurements:
            raise ValueError("report measurements contradict original grading")


def _adapt_rows(source: LoadedSource, manifest: SuiteManifest, rows: tuple[dict[str, Any], ...]) -> SuiteInput:
    """参数为来源、清单及原评分行；按运行身份连接，不依赖文件行号。"""
    records = _record_index(source)
    slices: list[ScoreSlice] = []
    used: set[str] = set()
    for row in rows:
        grading = GradingResult.model_validate(row.get("grading_result"))
        record = records.get(grading.run_id)
        if record is None:
            raise ValueError(f"grading references absent frozen run: {grading.run_id}")
        _validate_row(row, record, grading)
        if source.reference.kind in {"agent", "workflow"}:
            _validate_agent(record)
        used.add(grading.run_id)
        slices.append(ScoreSlice(
            source_id=source.reference.source_id, kind=source.reference.kind, run_id=grading.run_id,
            stage=row.get("stage", source.reference.kind), call_id=row.get("call_id"),
            backend=row.get("retrieval_backend"), grading=grading, detail_snapshot=snapshot(row),
        ))
    if used != records.keys():
        raise ValueError("report and frozen records do not cover the same runs")
    runs = tuple(_import_run(record, source, manifest) for record in records.values())
    return SuiteInput(manifest=manifest, runs=runs, scores=tuple(slices))


def adapt_planner_report(source: LoadedSource, manifest: SuiteManifest) -> SuiteInput:
    """参数为 Planner 原报告及清单；保留原三层合同和评分状态。"""
    report = read_snapshot(source.report_snapshot)
    rows = _objects(report.get("records"), location="planner.records")
    for row in rows:
        record_contracts(row)
    result = _adapt_rows(source, manifest, rows)
    if any(score.grading.grader != "research_planner" for score in result.scores):
        raise ValueError("planner source contains another grader")
    return result


def adapt_evidence_report(source: LoadedSource, manifest: SuiteManifest) -> SuiteInput:
    """参数为 Evidence 原报告及清单；显式选择报告区域，保留每次检索阶段。"""
    report = read_snapshot(source.report_snapshot)
    if source.reference.report_region == "evidence_benchmark":
        report = _object(report.get("evidence_benchmark"), location="report.evidence_benchmark")
    result = _adapt_rows(source, manifest, _objects(report.get("records"), location="evidence.records"))
    if any(score.grading.grader != "evidence-benchmark" for score in result.scores):
        raise ValueError("evidence source contains another grader")
    _validate_evidence_slices(result)
    return result


def _validate_evidence_slices(inputs: SuiteInput) -> None:
    """参数为 Evidence 来源输入；核对阶段与原观测哈希，防止给检索评分换阶段标签。"""
    from evals.evidence.report import extract_stage_observations

    runs = {run.run.run_id: run for run in inputs.runs}
    observations: dict[str, dict[tuple[str, str], StageObservation]] = {}
    for score in inputs.scores:
        if score.stage == "observation_integrity":
            if any(metric.name != "evidence_observation_integrity" or metric.status == "measured"
                   for metric in score.grading.measurements):
                raise ValueError("invalid evidence integrity row")
            continue
        if score.run_id not in observations:
            observed = read_snapshot(runs[score.run_id].run.observation_snapshot)
            stages = extract_stage_observations(observed)
            observations[score.run_id] = {(stage.stage, stage.call_id): stage for stage in stages}
        expected = observations[score.run_id].get((score.stage, score.call_id or ""))
        if expected is None:
            raise ValueError("evidence stage observation checksum mismatch")
        _validate_evidence_metadata(score, expected, runs[score.run_id])


def _validate_evidence_metadata(score: ScoreSlice, observed: StageObservation, run: ImportedRun) -> None:
    """参数为评分行、原阶段及运行；将用于分组和比较的标签绑定到真实观察。"""
    row = read_snapshot(score.detail_snapshot)
    if row.get("observation_hash") != snapshot_hash(snapshot(observed.model_dump(mode="json"))):
        raise ValueError("evidence stage observation checksum mismatch")
    fields = ("query", "retrieval_backend", "index_snapshot_id", "scope_id", "parent_scope_id", "unit_id")
    if any(row.get(name) != getattr(observed, name) for name in fields):
        raise ValueError("evidence report metadata contradicts frozen stage")
    benchmark = _object(read_snapshot(run.case.truth_snapshot).get("evidence_benchmark"), location="evidence_benchmark")
    policy = benchmark.get("ranking_policy")
    top_k = policy.get("top_k") if isinstance(policy, dict) and score.stage in _RETRIEVAL_STAGES else None
    if row.get("top_k") != top_k:
        raise ValueError("evidence ranking metadata contradicts original policy")
    if row.get("dataset_version") != benchmark.get("dataset_version") or row.get("task_family") != benchmark.get("task_family"):
        raise ValueError("evidence benchmark metadata contradicts original case")


def adapt_agent_report(source: LoadedSource, manifest: SuiteManifest) -> SuiteInput:
    """参数为 Agent 原报告及清单；通过配套冻结运行核对实际 episode 评分。"""
    report = read_snapshot(source.report_snapshot)
    if report.get("schema_version") != "agent-outcome-report-v1":
        raise ValueError("unsupported agent report version")
    result = _adapt_rows(source, manifest, _objects(report.get("rows"), location="agent.rows"))
    if any(score.grading.grader != "agent_outcome" for score in result.scores):
        raise ValueError("agent source contains another grader")
    return result


def adapt_workflow_report(source: LoadedSource, manifest: SuiteManifest) -> SuiteInput:
    """参数为工作流原产物和清单；强校验冻结观察、评分投影及策略身份，不复用旧 task_success。"""
    from .policy import MeasurementPolicy

    report = read_snapshot(source.report_snapshot)
    if report.get("schema_version") != "agent-workflow-report-v1":
        raise ValueError("unsupported workflow report version")
    policy = MeasurementPolicy.model_validate(report.get("measurement_policy"))
    if manifest.measurement_policy_snapshot != snapshot(policy.model_dump(mode="json")):
        raise ValueError("workflow_policy_does_not_match_manifest")
    result = _adapt_rows(source, manifest, _objects(report.get("rows"), location="workflow.rows"))
    for score in result.scores:
        detail = read_snapshot(score.detail_snapshot)
        if score.grading.grader != "agent_workflow" or score.grading.grader_version != policy.workflow_grader_version:
            raise ValueError("workflow_grader_version_mismatch")
        if detail.get("grader_version") != score.grading.grader_version:
            raise ValueError("workflow_row_grader_version_mismatch")
    return result


def _merge_run(existing: ImportedRun, incoming: ImportedRun) -> ImportedRun:
    """参数为跨报告的同一次运行；只合并相容的历史条件，不覆盖矛盾原记录。"""
    if (existing.case, existing.run, existing.review_status, existing.topology_snapshot) != (
        incoming.case, incoming.run, incoming.review_status, incoming.topology_snapshot,
    ):
        raise ValueError(f"conflicting frozen run across reports: {incoming.run.run_id}")
    conditions = {condition.name: condition for condition in existing.conditions}
    for value in incoming.conditions:
        old = conditions[value.name]
        if old.status == "missing":
            conditions[value.name] = value
        elif value.status != "missing" and (old.status, old.snapshot) != (value.status, value.snapshot):
            raise ValueError(f"conflicting recorded condition: {value.name}")
    return existing.model_copy(update={"conditions": tuple(conditions[name] for name in IDENTITY_NAMES)})


def _validate_slots(manifest: SuiteManifest, runs: dict[str, ImportedRun]) -> None:
    """参数为清单和已导入运行；拒绝错题、遗漏引用和未声明的额外运行。"""
    declared = {slot.run_id for slot in manifest.slots if slot.run_id is not None}
    if declared != runs.keys():
        raise ValueError("manifest slots and imported runs do not match")
    for slot in manifest.slots:
        if slot.run_id is None:
            continue
        run = runs[slot.run_id]
        if (slot.case_id, slot.case_version) != (run.case.case_id, run.case.case_version):
            raise ValueError(f"slot references another case: {slot.repeat_id}")


def adapt_suite_sources(manifest: SuiteManifest, sources: tuple[LoadedSource, ...]) -> SuiteInput:
    """参数为清单和原产物；返回唯一运行与评分切片，拒绝重复导入制造样本。"""
    adapters = {"planner": adapt_planner_report, "evidence": adapt_evidence_report,
                "agent": adapt_agent_report, "workflow": adapt_workflow_report}
    runs: dict[str, ImportedRun] = {}
    scores: list[ScoreSlice] = []
    seen: set[tuple[str, ...]] = set()
    for source in sources:
        if (manifest.measurement_policy_snapshot is not None and source.reference.kind != "planner"
                and source.ragas_snapshot is None and manifest.grading_timing != "deferred"):
            raise ValueError("frozen_policy_requires_ragas_artifact")
        adapted = adapters[source.reference.kind](source, manifest)
        if source.ragas_snapshot is not None:
            adapted = _append_ragas_score(adapted, source)
        for run in adapted.runs:
            existing = runs.get(run.run.run_id)
            runs[run.run.run_id] = run if existing is None else _merge_run(existing, run)
        for score in adapted.scores:
            key = (score.run_id, score.grading.grader, score.stage, score.call_id or "")
            if key in seen:
                raise ValueError(f"duplicate original score slice: {key}")
            seen.add(key)
            scores.append(score)
    _validate_slots(manifest, runs)
    return SuiteInput(manifest=manifest, runs=tuple(runs.values()), scores=tuple(scores))


def _append_ragas_score(inputs: SuiteInput, source: LoadedSource) -> SuiteInput:
    """参数为已校验产品评分和RAGAS冻结文件；绑定同一原始run与投影，追加独立评分切片。"""
    row = read_snapshot(str(source.ragas_snapshot))
    grading = GradingResult.model_validate(row["grading_result"])
    run = next((run for run in inputs.runs if run.run.run_id == grading.run_id), None)
    if run is None or grading.grader != "ragas_frozen":
        raise ValueError("ragas_grading_run_or_grader_mismatch")
    _validate_row(row, read_snapshot(run.record_snapshot), grading)
    if inputs.manifest.measurement_policy_snapshot is not None:
        from .policy import MeasurementPolicy

        policy = MeasurementPolicy.model_validate_json(inputs.manifest.measurement_policy_snapshot)
        if (row.get("measurement_policy") != policy.model_dump(mode="json")
                or grading.grader_version != policy.retrieval_grader_version
                or {item.name for item in grading.measurements} != set(policy.ragas_metrics)):
            raise ValueError("ragas_frozen_policy_mismatch")
        if any(sample.get("metric") not in policy.ragas_metrics for sample in row.get("samples", [])):
            raise ValueError("ragas_disabled_metric_was_scheduled")
        from evals.ragas.frozen_bridge import _jobs

        expected = _jobs(grading_projection(run.case, run.run), row.get("kind"), retrieval_only=True)
        binding_fields = ("metric", "variant", "turn", "call_id", "context_stage", "retrieval_query", "input_snapshot", "input_hash")
        actual_inputs = [{key: sample.get(key) for key in binding_fields} for sample in row.get("samples", [])]
        expected_inputs = [{key: sample[key] for key in binding_fields} for sample in expected]
        if actual_inputs != expected_inputs:
            raise ValueError("ragas_sample_input_does_not_match_frozen_observation")
    score = ScoreSlice(source_id=source.reference.source_id, kind=source.reference.kind,
                       run_id=grading.run_id, stage="ragas", grading=grading, detail_snapshot=snapshot(row))
    return inputs.model_copy(update={"scores": (*inputs.scores, score)})


def load_suite_sources(manifest_path: Path) -> SuiteInput:
    """参数为清单路径；验证全部原文件后返回可重建的独立输入。"""
    manifest = SuiteManifest.model_validate_json(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.grading_attempt_snapshot is not None:
        attempt = read_snapshot(manifest.grading_attempt_snapshot)
        source = (manifest_path.parent / attempt["source_manifest"]).resolve()
        if source.parent != manifest_path.resolve().parent or source == manifest_path.resolve():
            raise ValueError("grading_manifest_parent_must_be_another_file_in_suite_root")
        _verified_text(source, attempt["source_manifest_sha256"])
    sources = tuple(read_frozen_source(reference, base_dir=manifest_path.parent) for reference in manifest.sources)
    resolved = manifest.model_copy(update={"sources": tuple(source.reference for source in sources)})
    return adapt_suite_sources(resolved, sources)
