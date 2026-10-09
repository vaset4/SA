"""采集实际运行条件、验证漂移并绑定不可覆盖的评测产物。作者：xxx。"""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from evals.contracts import read_snapshot, record_contracts, snapshot, snapshot_hash

from .contracts import FrozenReportRef, IdentityCondition, IdentityName, RecordedIdentity, ReportKind, RunConditions, SuiteManifest
from .execution_contracts import JobKind, PreparedSuite, ProducedRun
from .prepare import WORKSPACE_ROOT, sha256_file, write_new_json
from .runtime import LivePreparationRuntime

_MODEL_ATTRIBUTES = (
    "model", "temperature", "timeout", "min_request_interval_seconds", "request_exception_max_retries",
    "empty_response_max_retries", "retry_backoff_seconds",
)
_DEPENDENCIES = ("pydantic", "langgraph", "langchain-core", "requests", "pymilvus", "pypdf", "Pillow", "tokenizers")
_RETRIEVAL_SETTINGS = (
    "rag_top_k", "vector_backend", "rag_planner_lightrag_locator_enabled", "graph_rag_enabled",
    "milvus_collection_v2", "milvus_rrf_k", "vector_rerank_enabled", "vector_rerank_top_k",
)
_FIXTURE_OPTIONAL_CONDITIONS: tuple[IdentityName, ...] = ("source", "index", "model", "embedding", "prompt", "retrieval")
_GENERATION_CONDITIONS: tuple[IdentityName, ...] = ("model", "prompt")
_REPORT_KINDS: dict[JobKind, ReportKind] = {"retrieval": "evidence", "agent": "agent", "planner_fixture": "planner"}


@dataclass(frozen=True, kw_only=True)
class RuntimeBindings:
    """本次实际依赖和固定初始状态；不携带模型秘密或当前预期分数。"""

    prepared: PreparedSuite
    kind: JobKind
    live: LivePreparationRuntime | None = None
    fixture_snapshot: str | None = None
    retrieval_method: str | None = None


def safe_endpoint(value: str) -> str:
    """参数为实际客户端地址；保留协议、主机和路径，去除认证及查询密钥。"""
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def fingerprint_source_files(files: tuple[Path, ...], *, root: Path) -> dict[str, Any]:
    """参数为实际文件和命名根；返回包含未提交改动的内容身份，删除和新增也会变化。"""
    return {"files": {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)}}


def describe_model_client(client: Any) -> dict[str, Any]:
    """参数为实际模型客户端；只采集影响调用的公开配置，绝不序列化 api_key 或 headers。"""
    from app.config import get_settings
    from app.services.provider_request_queue import queue_policy_identity

    return {"provider_queue": queue_policy_identity(get_settings()),
            "client_type": f"{type(client).__module__}.{type(client).__qualname__}",
            "endpoint": safe_endpoint(client.base_url), **{name: getattr(client, name) for name in _MODEL_ATTRIBUTES}}


def _embedding_identity() -> dict[str, Any]:
    """无参数；API 记录实际服务配置，local 核对真实模型文件，不伪造远端权重身份。"""
    from app.config import normalize_model_api_url
    from app.services.embedding_client import get_embedding_client
    from app.services.hf_local import resolve_local_model_source

    client = get_embedding_client()
    if client._mode == "api":
        return {"mode": "api", "model": client._model_name, "dimension": client.dimension,
                "batch_size": client._batch_size, "normalized": True,
                "service": normalize_model_api_url(client._settings.embedding_api_base_url,
                                                   setting_name="EMBEDDING_API_BASE_URL"),
                "timeout": client._settings.embedding_api_timeout_seconds}
    source = Path(resolve_local_model_source(client._model_name))
    if not source.is_dir():
        raise ValueError("actual_embedding_model_snapshot_unavailable")
    files = tuple(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError("actual_embedding_model_snapshot_empty")
    return {"model": client._model_name, "dimension": client.dimension, "device": client._device,
            "batch_size": client._batch_size, **fingerprint_source_files(files, root=source)}


def _environment_identity(bindings: RuntimeBindings) -> dict[str, Any]:
    """参数为本次依赖；冻结固定初始数据和依赖版本，不把输出目录、run ID 或时间当条件。"""
    from app.config import get_settings

    settings = get_settings().model_dump()
    dependencies = _DEPENDENCIES
    if settings["embedding_mode"] == "local" or settings["rerank_mode"] == "local":
        dependencies += ("sentence-transformers",)
    configuration = {key: value for key, value in settings.items() if key.startswith(("cache_", "research_memory_", "rag_planner_"))
                     and not any(secret in key for secret in ("key", "secret", "password", "token", "url"))}
    configuration.update({name: settings[name] for name in ("embedding_mode", "rerank_mode", "chunk_tokenizer_sha256")})
    initial = bindings.fixture_snapshot if bindings.kind == "planner_fixture" else bindings.prepared.initial_snapshot
    return {"initial_snapshot_sha256": snapshot_hash(initial or "{}"), "cache_initial_state": "empty_process_local",
            **({"grading_timing": bindings.prepared.profile.grading_timing} if bindings.prepared.profile.grading_timing else {}),
            **({"repetition_order": bindings.prepared.profile.repetition_order} if bindings.prepared.profile.repetition_order else {}),
            "python": platform.python_version(), "system": platform.system(),
            "dependencies": {name: importlib.metadata.version(name) for name in dependencies}, "configuration": configuration}


def _recorded(name: IdentityName, value: dict[str, Any]) -> IdentityCondition:
    """参数为条件名和实际公开配置；返回可重建的已记录条件。"""
    return IdentityCondition(name=name, status="recorded", snapshot=snapshot(value), reason="captured_before_actual_execution")


def capture_conditions(bindings: RuntimeBindings) -> tuple[IdentityCondition, ...]:
    """参数为本次实际依赖；返回代码、评分器、模型、索引及环境的条件，不补写历史运行。"""
    product = WORKSPACE_ROOT / "backend" / "app"
    evaluator = WORKSPACE_ROOT / "backend" / "evals"
    conditions = [
        _recorded("code", fingerprint_source_files(tuple(product.rglob("*.py")), root=product)),
        _recorded("evaluator", _evaluator_identity(bindings, evaluator)),
        _recorded("environment", _environment_identity(bindings)),
    ]
    if bindings.kind == "planner_fixture":
        conditions.extend(IdentityCondition(name=name, status="not_applicable", reason="declared_deterministic_planner_fixture")
                          for name in _FIXTURE_OPTIONAL_CONDITIONS)
        return tuple(conditions)
    from app.config import get_settings

    settings = get_settings()
    conditions.extend([
        _recorded("embedding", _embedding_identity()),
        _recorded("index", {"index_snapshot_ids": [bindings.prepared.index.index_snapshot_id]}),
        _recorded("retrieval", {"fixed_query_top_k": bindings.prepared.profile.retrieval_top_k,
                               **({"method": bindings.retrieval_method} if bindings.retrieval_method is not None else {}),
                               "configuration": {name: getattr(settings, name) for name in _RETRIEVAL_SETTINGS if hasattr(settings, name)}}),
    ])
    if bindings.kind == "retrieval":
        conditions.extend(IdentityCondition(name=name, status="not_applicable", reason="retrieval_entry_without_generation") for name in _GENERATION_CONDITIONS)
    else:
        conditions.append(_recorded("model", _model_identities()))
        prompts = tuple(product.rglob("*prompt*.py"))
        conditions.append(_recorded("prompt", fingerprint_source_files(prompts, root=product)))
    return tuple(conditions)


def _evaluator_identity(bindings: RuntimeBindings, root: Path) -> dict[str, Any]:
    """参数为实际依赖和评分源码根；冻结策略及 judge 身份，避免接口变化被当作产品提升。"""
    return grading_identity(bindings.prepared.profile.measurement_policy, root=root)


def grading_identity(policy: Any, *, root: Path = WORKSPACE_ROOT / "backend" / "evals") -> dict[str, Any]:
    """参数为评分策略及源码根；返回当前实际judge配置和评分代码身份，不调用产品或模型。"""
    result = {**fingerprint_source_files(tuple(root.rglob("*.py")), root=root),
              "export_redaction_policy": "known-credentials-v1"}
    if policy is None:
        return result
    from app.config import get_settings

    settings = get_settings()
    from app.services.provider_request_queue import queue_policy_identity

    return {**result, "measurement_policy": policy.model_dump(mode="json"),
            "judge": {"model": settings.llm_model_name, "endpoint": safe_endpoint(settings.llm_base_url),
                      "temperature": 0.0, "ragas_version": importlib.metadata.version("ragas"),
                      "timeout": settings.llm_timeout, "provider_queue": queue_policy_identity(settings),
                      "min_request_interval_seconds": settings.ragas_llm_min_request_interval_seconds}}


def _model_identities() -> dict[str, Any]:
    """无参数；读取主链和记忆写回实际客户端，保留独立模型配置的差异。"""
    from app.services.llm_service import get_llm
    from app.services.research_memory_curator_service import _curator_llm

    return {"primary": describe_model_client(get_llm()), "memory_writeback": describe_model_client(_curator_llm())}


def verify_model_requests(record: dict[str, Any], conditions: tuple[IdentityCondition, ...]) -> tuple[str, ...]:
    """参数为实际记录和事前客户端身份；核对真实 HTTP 参数，缺失或矛盾时阻止质量比较。"""
    _case, run = record_contracts(record)
    if run.run_mode != "agent_live":
        return ()
    condition = next(item for item in conditions if item.name == "model")
    if condition.snapshot is None:
        return ("actual_model_identity_missing",)
    clients = read_snapshot(condition.snapshot).values()
    keys = ("model", "temperature", "endpoint")
    allowed = {snapshot({key: client[key] for key in keys}) for client in clients}
    problems: list[str] = []
    for turn in read_snapshot(run.observation_snapshot).get("turns", []):
        observed = read_snapshot(turn.get("evidence_observation_snapshot", "{}"))
        requests = [event["payload"] for event in observed.get("evidence_observations", []) if event["stage"] == "model_request_parameters"]
        uncached = [call for call in observed.get("model_call_observations", []) if not call["cache_hit"]]
        if uncached and not requests:
            problems.append("actual_model_request_parameters_missing")
        if any(snapshot({key: request.get(key) for key in keys}) not in allowed for request in requests):
            problems.append("actual_model_request_differs_from_recorded_client")
    return tuple(dict.fromkeys(problems))


def verify_conditions(before: tuple[IdentityCondition, ...], after: tuple[IdentityCondition, ...]) -> tuple[str, ...]:
    """参数为运行前后实际条件；返回每项漂移原因，任何漂移都不能作为正常前后提升。"""
    initial = {condition.name: (condition.status, condition.snapshot) for condition in before}
    current = {condition.name: (condition.status, condition.snapshot) for condition in after}
    return tuple(f"runtime_condition_drift:{name}" for name in sorted(initial.keys() | current.keys()) if initial.get(name) != current.get(name))


def write_recorded_identity(produced: ProducedRun, conditions: RunConditions, *, path: Path) -> RecordedIdentity:
    """参数为实际产物、原运行条件和新文件；按文件内容绑定身份，不接受其他 run 的条件。"""
    if produced.report_path is None or produced.frozen_path is None:
        raise ValueError("recorded_identity_requires_actual_report_and_frozen_artifacts")
    if produced.run_id != conditions.run_id:
        raise ValueError("recorded_conditions_must_match_produced_run")
    identity = RecordedIdentity(report_sha256=sha256_file(Path(produced.report_path)),
                                frozen_sha256=sha256_file(Path(produced.frozen_path)), runs=(conditions,))
    write_new_json(path, identity.model_dump(mode="json"))
    return identity


def _reference(produced: ProducedRun, *, kind: JobKind, root: Path,
               workflow: bool = False) -> FrozenReportRef | None:
    """参数为实际运行和入口；仅存在完整文件时返回真实清单引用，缺产物保留为漏跑。"""
    paths = (produced.report_path, produced.frozen_path, produced.identity_path)
    if not produced.run_id or any(path is None or not Path(path).is_file() for path in paths):
        return None
    report, frozen, identity = (Path(str(path)).resolve() for path in paths)
    ragas = report.parent / "ragas.json"
    if any(not path.is_relative_to(root.resolve()) for path in (report, frozen, identity)):
        raise ValueError("produced_artifact_outside_run_directory")
    return FrozenReportRef(
        source_id=f"{produced.job_id}:{produced.case_id}:{produced.repeat_id}",
        kind="workflow" if workflow and kind == "agent" else _REPORT_KINDS[kind],
        report_path=report.relative_to(root.resolve()).as_posix(), report_sha256=sha256_file(report),
        frozen_path=frozen.relative_to(root.resolve()).as_posix(), frozen_sha256=sha256_file(frozen),
        identity_path=identity.relative_to(root.resolve()).as_posix(), identity_sha256=sha256_file(identity),
        ragas_path=ragas.relative_to(root.resolve()).as_posix() if ragas.is_file() else None,
        ragas_sha256=sha256_file(ragas) if ragas.is_file() else None,
    )


def build_run_manifest(prepared: PreparedSuite, produced: tuple[ProducedRun, ...], *, output_dir: Path) -> Path | None:
    """参数为固定套件和实际产物；只用本次完整产物填槽，无产物时不伪造非空清单。"""
    jobs = {job.job_id: job for job in prepared.jobs}
    actual = {(run.job_id, run.case_id, run.case_version, run.repeat_id): run for run in produced}
    if len(actual) != len(produced):
        raise ValueError("duplicate_produced_execution_slot")
    references: list[FrozenReportRef] = []
    slots = []
    for job in prepared.jobs:
        for slot in job.slots:
            key = (job.job_id, slot.case_id, slot.case_version, slot.repeat_id)
            run = actual.pop(key, None)
            reference = _reference(run, kind=jobs[job.job_id].kind, root=output_dir,
                                   workflow=prepared.profile.measurement_policy is not None) if run else None
            if reference is not None:
                references.append(reference)
            slots.append(slot.model_copy(update={"run_id": run.run_id if run and reference else None}))
    if actual:
        raise ValueError("produced_run_not_in_prepared_slots")
    if not references:
        return None
    policy = prepared.profile.measurement_policy
    manifest = SuiteManifest(suite_id=prepared.profile.suite_id, purpose=prepared.profile.purpose,
                             dataset_version=prepared.profile.dataset_version, sources=tuple(references), slots=tuple(slots),
                             measurement_policy_snapshot=snapshot(policy.model_dump(mode="json")) if policy else None,
                             grading_timing=prepared.profile.grading_timing,
                             execution_snapshot=snapshot({"runs": [run.model_dump(mode="json") for run in produced]}))
    path = output_dir / "manifest.json"
    write_new_json(path, manifest.model_dump(mode="json"))
    return path
