"""校验真实论文归属与索引，生成可重复使用的固定评测输入。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from evals.agent_outcome.contracts import AgentCase, load_agent_case
from evals.contracts import create_case, snapshot, snapshot_hash
from evals.evidence.alignment import validate_benchmark
from evals.evidence.contracts import (
    CORE_EVIDENCE_METRICS, EvidenceBenchmark, EvidenceMeasurementPolicy, IndexSnapshot,
    RankingPolicy, SourceSnapshot, make_source_snapshot,
)
from evals.evidence.corpus import load_json_object, validate_case_assets

from .contracts import RunSlot
from .execution_contracts import (
    ExecutionJob, ExecutionProfile, InputArtifact, PreparedCase, PreparedJob, PreparedSuite,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_HASH_READ_SIZE = 1024 * 1024


class PreparationRuntime(Protocol):
    """准备过程的只读端口；实际实现复用业务归属和现役检索 provider。"""

    async def resolve_scope(self, profile: ExecutionProfile, *, paper_ids: tuple[str, ...]) -> dict[str, Any]:
        """参数为配置和请求论文；返回经业务权限校验的实际范围。"""
        ...

    async def verify_bindings(self, profile: ExecutionProfile) -> tuple[dict[str, Any], ...]:
        """参数为真实绑定；核对业务论文行和 PDF，返回固定元数据。"""
        ...

    async def export_index(self, sources: tuple[SourceSnapshot, ...]) -> tuple[IndexSnapshot, list[dict[str, Any]]]:
        """参数为已验证原文；导出现有完整索引及映射缺口。"""
        ...

    async def snapshot_initial(self, profile: ExecutionProfile, *, papers: tuple[dict[str, Any], ...]) -> str:
        """参数为配置和实际论文；返回固定问题、记忆、关联及元数据快照。"""
        ...


class PreparationFailure(ValueError):
    """携带可落盘的数据缺口；不会产生可冒充基线的套件清单。"""

    def __init__(self, reason: str, *, details: list[dict[str, Any]]) -> None:
        """参数为原因和实际缺口；保存明确诊断供 CLI 输出。"""
        super().__init__(reason)
        self.details = details


@dataclass(frozen=True, kw_only=True)
class CaseBindings:
    """集中保存已验证的论文、索引及 RAG 真值，避免调用点自行猜身份。"""

    profile: ExecutionProfile
    sources: tuple[SourceSnapshot, ...]
    index: IndexSnapshot
    rag_cases: dict[str, dict[str, Any]]
    scopes: dict[str, dict[str, Any]]


def sha256_file(path: Path) -> str:
    """参数为真实文件；流式计算 SHA-256，读取失败向调用方暴露。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_HASH_READ_SIZE):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, payload: Any) -> None:
    """参数为新文件和可序列化对象；以独占写入保存，拒绝覆盖历史产物。"""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_replaced_json(path: Path, payload: Any) -> None:
    """参数为会被反复更新的账本文件和可序列化对象；先写临时文件再原子替换。

    执行账本每完成一个槽位就要重写一次，半截 JSON 比没有文件更难排查，
    所以这里不复用独占写入的 write_new_json，而是走同卷 os.replace。
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def profile_path(value: str) -> Path:
    """参数为配置中的路径；相对路径统一从仓库根解析，不依赖调用目录。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def load_profile(path: Path) -> ExecutionProfile:
    """参数为配置文件；返回已填真实身份的配置，模板空值明确提示准备条件。"""
    raw = load_json_object(path)
    missing = [item.get("paper_key", "unknown") for item in raw.get("paper_bindings", []) if item.get("paper_id") is None]
    if raw.get("user_id") is None:
        missing.insert(0, "user_id")
    if missing:
        raise ValueError("尚未完成真实业务绑定：" + ", ".join(missing) + "；需通过正常入库建立归属，文档代号不能代替业务身份")
    return ExecutionProfile.model_validate(raw)


def _load_dataset(job: ExecutionJob) -> tuple[dict[str, Any], ...]:
    """参数为事前任务；校验题目唯一且覆盖声明，再按固定顺序返回案例。"""
    path = profile_path(job.dataset_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) or not row.get("id") for row in rows):
        raise ValueError(f"dataset has invalid case identities: {path}")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(job.case_ids) <= by_id.keys():
        raise ValueError(f"duplicate or missing declared cases: {path}")
    return tuple(by_id[identity] for identity in job.case_ids)


def verify_profile_sources(profile: ExecutionProfile) -> tuple[SourceSnapshot, ...]:
    """参数为配置；核对稳定候选包与实际原文，只为已验证 PDF 建立新业务身份。"""
    sources: dict[str, SourceSnapshot] = {}
    for job in profile.jobs:
        if job.kind == "planner_fixture":
            continue
        path = profile_path(job.dataset_path)
        manifest = load_json_object(path.parent / "manifest.json")
        if manifest.get("files", {}).get(path.name) != sha256_file(path):
            raise ValueError(f"candidate_package_checksum_mismatch:{path.name}")
        for key, raw in manifest["source_snapshots"].items():
            value = SourceSnapshot.model_validate(raw)
            if key in sources and sources[key] != value:
                raise ValueError(f"conflicting_source_snapshot:{key}")
            sources[key] = value
    bound: list[SourceSnapshot] = []
    for binding in profile.paper_bindings:
        source = sources.get(binding.paper_key)
        if source is None or source.pdf_sha256 != binding.pdf_sha256:
            raise ValueError(f"paper_binding_pdf_identity_mismatch:{binding.paper_key}")
        bound.append(make_source_snapshot(
            paper_id=binding.paper_id, pdf_sha256=source.pdf_sha256, blocks=source.blocks,
            artifact_manifest_snapshot=source.artifact_manifest_snapshot,
            parser_engine_version=source.parser_engine_version,
        ))
    benchmark = EvidenceBenchmark(dataset_version=profile.dataset_version, review_status="candidate", sources=tuple(bound), spans=(), groups=())
    rows = validate_case_assets(benchmark, uploads_dir=profile_path(profile.uploads_root), paper_dir=profile_path(profile.paper_root))
    if any(row["status"] != "measured" for row in rows):
        raise PreparationFailure("original_source_verification_failed", details=rows)
    return tuple(bound)


def bind_case_assets(raw: dict[str, Any], bindings: CaseBindings) -> dict[str, Any]:
    """参数为 RAG 候选和已验证绑定；更新原文引用与事前策略，不修改原案例。"""
    original = EvidenceBenchmark.model_validate(raw["evidence_benchmark"])
    by_digest = {source.pdf_sha256: source for source in bindings.sources}
    old_to_new = {source.snapshot_id: by_digest[source.pdf_sha256] for source in original.sources}
    paper_ids = {source.paper_id: by_digest[source.pdf_sha256].paper_id for source in original.sources}
    body = original.model_dump(mode="json")
    body.update({
        "dataset_version": bindings.profile.dataset_version,
        "sources": [source.model_dump(mode="json") for source in bindings.sources],
        "spans": [{**span.model_dump(mode="json"), "source_snapshot_id": old_to_new[span.source_snapshot_id].snapshot_id} for span in original.spans],
        "target_paper_ids": [paper_ids[identity] for identity in original.target_paper_ids],
        "index_snapshots": [bindings.index.model_dump(mode="json")],
        "ranking_policy": RankingPolicy(stage="query_raw", top_k=bindings.profile.retrieval_top_k).model_dump(mode="json"),
        "measurement_policy": EvidenceMeasurementPolicy(enabled_metrics=CORE_EVIDENCE_METRICS).model_dump(mode="json"),
        "ranking_judgments": [],
    })
    for key in ("paper_requirements", "target_sections"):
        if body.get(key) is not None:
            body[key] = [{**item, "paper_id": paper_ids[item["paper_id"]]} for item in body[key]]
    benchmark = EvidenceBenchmark.model_validate(body)
    validate_benchmark(benchmark, include_ranking=False)
    scope = bindings.scopes[raw["id"]]
    prepared = {
        **raw, "evidence_benchmark": benchmark.model_dump(mode="json"),
        "paper_ids": scope["paper_ids"], "user_id": bindings.profile.user_id,
        "project_id": bindings.profile.project_id, "project_scope": scope,
        "structured_query": {"project_scope": scope, "paper_ids": scope["paper_ids"]},
    }
    prepared["case_version"] = _case_version(prepared, bindings.profile.dataset_version)
    return prepared


def _case_version(record: dict[str, Any], version: str) -> str:
    """参数为新案例完整内容与数据版本；返回可区分绑定和策略变化的准备版本。"""
    return f"{version}:{snapshot_hash(snapshot(record))[:16]}"


def _agent_targets(scenario: dict[str, Any], bindings: CaseBindings) -> tuple[EvidenceBenchmark, dict[str, Any]]:
    """参数为场景和已绑定 RAG 候选；把独立 evidence_refs 解析为完整原文目标。"""
    if "evidence_benchmark" in scenario:
        bound = bind_case_assets(scenario, bindings)
        benchmark = EvidenceBenchmark.model_validate(bound["evidence_benchmark"])
        negative = scenario.get("answerability") == "insufficient_evidence"
        if not benchmark.groups and not negative:
            raise ValueError("answerable_agent_requires_complete_evidence_proof")
        return benchmark, {"required_span_ids": [span.span_id for span in benchmark.spans],
                           "required_group_ids": [group.group_id for group in benchmark.groups],
                           "required_paper_ids": list(benchmark.target_paper_ids)}
    spans: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    references = scenario.get("evidence_refs", [])
    required = {ref["evidence_key"] for ref in references}
    if not required:
        raise ValueError("agent evidence references are missing")
    for ref in references:
        benchmark = EvidenceBenchmark.model_validate(bindings.rag_cases[ref["case_id"]]["evidence_benchmark"])
        span = next((span for span in benchmark.spans if span.span_id == ref["evidence_key"]), None)
        if span is None:
            raise ValueError(f"agent_evidence_reference_unresolved:{ref}")
        spans[span.span_id] = span
        groups.update({group.group_id: group for group in benchmark.groups
                       if any(set(proof.span_ids) <= required for proof in group.proofs)})
    selected_groups = tuple(group.model_copy(update={"proofs": tuple(proof for proof in group.proofs if set(proof.span_ids) <= required)}) for group in groups.values())
    source_ids = {span.source_snapshot_id for span in spans.values()}
    papers = tuple(source.paper_id for source in bindings.sources if source.snapshot_id in source_ids)
    benchmark = EvidenceBenchmark(
        dataset_version=bindings.profile.dataset_version, task_family=scenario["task_family"],
        review_status=scenario["review_status"], sources=bindings.sources, spans=tuple(spans.values()),
        groups=selected_groups, target_paper_ids=papers, index_snapshots=(bindings.index,),
        ranking_policy=RankingPolicy(stage="query_raw", top_k=bindings.profile.retrieval_top_k),
        measurement_policy=EvidenceMeasurementPolicy(enabled_metrics=CORE_EVIDENCE_METRICS),
    )
    validate_benchmark(benchmark, include_ranking=False)
    if not selected_groups:
        raise ValueError("agent_evidence_references_do_not_resolve_a_complete_proof")
    return benchmark, {"required_span_ids": sorted(required), "required_group_ids": sorted(groups), "required_paper_ids": sorted(papers)}


def prepare_agent_case(scenario: dict[str, Any], bindings: CaseBindings) -> AgentCase:
    """参数为原场景和已验证绑定；冻结真实初始范围，将目标和后续消息留在独立真值。"""
    scope = bindings.scopes[scenario["id"]]
    keys = {binding.paper_key: binding.paper_id for binding in bindings.profile.paper_bindings}
    focus_key = scenario["initial_conditions"].get("focused_paper_key")
    focus = keys[focus_key] if focus_key is not None else None
    benchmark, targets = _agent_targets(scenario, bindings)
    raw = {
        **scenario,
        "input": {**scenario["input"], "user_id": bindings.profile.user_id,
                  "project_id": bindings.profile.project_id, "project_scope": scope, "paper_ids": scope["paper_ids"],
                  "structured_query": {"project_scope": scope, "paper_ids": [focus] if focus else list(scope["paper_ids"])}},
        "initial_conditions": {**scenario["initial_conditions"], "focused_paper_key": focus, "setup_status": "prepared"},
        "evidence_benchmark": benchmark.model_dump(mode="json"), "evidence_targets": targets,
    }
    raw["case_version"] = _case_version(raw, bindings.profile.dataset_version)
    return load_agent_case(raw)


async def _case_scopes(
    profile: ExecutionProfile, datasets: dict[str, tuple[dict[str, Any], ...]], *, runtime: PreparationRuntime,
) -> dict[str, dict[str, Any]]:
    """参数为配置、固定案例和归属端口；分别绑定授权范围，不把目标集合混入授权。"""
    keys = {binding.paper_key: binding.paper_id for binding in profile.paper_bindings}
    scopes: dict[str, dict[str, Any]] = {}
    for job in profile.jobs:
        if job.kind == "planner_fixture":
            continue
        for row in datasets[job.job_id]:
            if job.kind == "agent":
                paper_keys = row["initial_conditions"]["authorized_paper_keys"]
            else:
                paper_keys = row.get("authorized_paper_keys")
            if not isinstance(paper_keys, list) or not paper_keys or any(key not in keys for key in paper_keys):
                raise ValueError(f"case_authorized_paper_keys_missing_or_unbound:{row['id']}")
            requested = tuple(keys[key] for key in paper_keys)
            scope = await runtime.resolve_scope(profile, paper_ids=requested)
            if row["id"] in scopes and scopes[row["id"]] != scope:
                raise ValueError(f"logical_case_authorized_scope_differs_between_channels:{row['id']}")
            scopes[row["id"]] = scope
    return scopes


def _prepared_jobs(bindings: CaseBindings, datasets: dict[str, tuple[dict[str, Any], ...]]) -> tuple[PreparedJob, ...]:
    """参数为真实绑定和固定题集；冻结新案例并按事前重复次数展开槽位。"""
    jobs: list[PreparedJob] = []
    for job in bindings.profile.jobs:
        cases: list[PreparedCase] = []
        for raw in datasets[job.job_id]:
            record = bindings.rag_cases[raw["id"]] if job.kind == "retrieval" else dict(raw)
            if job.kind == "planner_fixture":
                record = {**record, "source_type": "fixture", "case_version": _case_version(record, bindings.profile.dataset_version)}
            if job.kind == "agent":
                agent = prepare_agent_case(raw, bindings)
                case = PreparedCase(case_id=agent.case_id, case_version=agent.case_version, record_snapshot=snapshot(agent.model_dump(mode="json")))
            else:
                contract = create_case(record)
                case = PreparedCase(case_id=contract.case_id, case_version=contract.case_version, record_snapshot=snapshot(record))
            cases.append(case)
        slots = tuple(RunSlot(case_id=case.case_id, case_version=case.case_version, repeat_id=str(repeat), run_id=None,
                              required_checks=job.required_checks) for case in cases for repeat in range(1, job.repetitions + 1))
        if bindings.profile.repetition_order == "round-first":
            slots = tuple(sorted(slots, key=lambda slot: int(slot.repeat_id)))
        jobs.append(PreparedJob(job_id=job.job_id, kind=job.kind, cases=tuple(cases), slots=slots,
                                retrieval_method=job.retrieval_method))
    return tuple(jobs)


async def prepare_suite(profile: ExecutionProfile, output_dir: Path, *, runtime: PreparationRuntime) -> PreparedSuite:
    """参数为配置、新目录和只读依赖；保存固定输入或具体准备失败，不创建虚构基线。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        prepared = await _prepare(profile, runtime)
    except Exception as exc:
        details = exc.details if isinstance(exc, PreparationFailure) else []
        write_new_json(output_dir / "preparation-failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "details": details,
        })
        raise
    write_new_json(output_dir / "prepared.json", prepared.model_dump(mode="json"))
    return prepared


async def _prepare(profile: ExecutionProfile, runtime: PreparationRuntime) -> PreparedSuite:
    """参数为配置与只读端口；按原文、归属、索引和初始状态顺序生成内容寻址准备包。"""
    sources = verify_profile_sources(profile)
    papers = await runtime.verify_bindings(profile)
    scope = await runtime.resolve_scope(profile, paper_ids=tuple(item.paper_id for item in profile.paper_bindings))
    index, issues = await runtime.export_index(sources)
    blocking = [issue for issue in issues if not is_text_mapping_gap(issue)]
    if blocking:
        raise PreparationFailure("actual_index_original_text_mapping_incomplete", details=issues)
    datasets = {job.job_id: _load_dataset(job) for job in profile.jobs}
    scopes = await _case_scopes(profile, datasets, runtime=runtime)
    bindings = CaseBindings(profile=profile, sources=sources, index=index, rag_cases={}, scopes=scopes)
    rag = {raw["id"]: bind_case_assets(raw, bindings) for job in profile.jobs if job.kind == "retrieval" for raw in datasets[job.job_id]}
    bindings = CaseBindings(profile=profile, sources=sources, index=index, rag_cases=rag, scopes=scopes)
    initial = await runtime.snapshot_initial(profile, papers=papers)
    artifacts = tuple(InputArtifact(path=job.dataset_path, sha256=sha256_file(profile_path(job.dataset_path))) for job in profile.jobs)
    draft = PreparedSuite.model_validate({
        "prepared_id": "0" * 64, "profile": profile, "scope_snapshot": snapshot(scope), "sources": sources,
        "index": index, "initial_snapshot": initial, "mapping_issues_snapshot": snapshot({"issues": issues}),
        "input_artifacts": artifacts, "jobs": _prepared_jobs(bindings, datasets),
    }, context={"building_identity": True})
    body = draft.model_dump(mode="json", exclude={"prepared_id"})
    return PreparedSuite.model_validate({**body, "prepared_id": snapshot_hash(snapshot(body))})


def is_text_mapping_gap(issue: dict[str, Any]) -> bool:
    """参数为严格原文映射诊断；仅正文未定位或多重定位允许逐指标缺失，其余准备错误仍阻断。"""
    reason = str(issue.get("reason", ""))
    return reason == "index_text_not_located" or reason.startswith("index_text_match_count=")
