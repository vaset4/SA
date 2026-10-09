"""统一评测的现役检索/Agent 依赖及进程内 IO 装配。作者：xxx。"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch
from uuid import uuid4

from evals.agent_outcome.contracts import AgentCase, agent_case_payload
from evals.agent_outcome.environment import AgentEnvironment, GraphFactory
from evals.agent_outcome.runner import run_episode
from evals.agent_outcome.runtime_storage import STORAGE_VERSION, EvaluationStorage
from evals.contracts import EvalCase, freeze_record, read_snapshot, snapshot, snapshot_hash
from evals.evidence.contracts import IndexSnapshot, SourceSnapshot
from evals.evidence.corpus import build_index_from_chunks
from evals.evidence.live_capture import CapturedRunFailure, FailedLiveCapture

from .execution_contracts import ExecutionProfile, PreparedSuite
from .prepare import PreparationFailure, profile_path, sha256_file

_MEMORY_MODULES = (
    "app.repositories.research_memory_repository", "app.services.research_memory_service",
    "app.services.research_memory_curator_service", "app.services.research_memory_injector_service",
    "app.services.internal_research.project_memory_reader",
)
_ORIGINAL_TEXT_PORTS = ("hybrid_search", "local_rag_search", "read_chunk", "get_chunk_row", "list_chunk_rows_by_paper")
# 【评测导出】【快照合同】这些运行/准备字段保存JSON对象；Evidence.text_snapshot保存正文，不能按后缀解析
_JSON_EXPORT_SNAPSHOT_FIELDS = frozenset({
    "input_snapshot", "truth_snapshot", "observation_snapshot", "fixture_snapshot",
    "environment_snapshot", "values_snapshot", "evidence_observation_snapshot",
    "artifact_manifest_snapshot", "record_snapshot", "scope_snapshot", "initial_snapshot",
    "mapping_issues_snapshot", "measurement_policy_snapshot", "execution_snapshot", "cleanup_snapshot",
})


def sanitize_frozen_record(record: dict[str, Any], *, secrets: tuple[str, ...]) -> dict[str, Any]:
    """参数为待导出记录及本次实际凭证；返回递归脱敏副本，保留原异常和产品对象不变。"""
    return _sanitize_export_value(record, secrets=secrets)


def export_secrets(kind: str) -> tuple[str, ...]:
    """参数为执行类型；返回本次实际客户端凭证，仅用于导出副本脱敏。"""
    if kind == "planner_fixture":
        return ()
    from app.config import get_settings
    from app.services.research_memory_curator_service import _curator_llm

    return (get_settings().llm_api_key, _curator_llm().api_key)


def _sanitize_export_value(value: Any, *, secrets: tuple[str, ...], field: str = "") -> Any:
    """参数为导出节点、实际凭证及字段名；解析冻结快照后脱敏，避免破坏JSON转义。"""
    if isinstance(value, dict):
        return {key: _sanitize_export_value(item, secrets=secrets, field=key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_export_value(item, secrets=secrets) for item in value]
    if not isinstance(value, str):
        return value
    if field in _JSON_EXPORT_SNAPSHOT_FIELDS:
        return snapshot(_sanitize_export_value(read_snapshot(value), secrets=secrets))
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    return value


class LivePreparationRuntime:
    """只读读取实际归属和现有 Milvus 索引，不写入业务论文或共享索引。"""

    def __init__(self) -> None:
        """无参数；延迟加载真实 provider，保存实际索引内容指纹。"""
        self._store: Any = None
        self.index_storage_identity: str | None = None
        self.publication: dict[str, Any] | None = None
        self.active_scopes: dict[str, dict[str, Any]] = {}

    @property
    def store(self) -> Any:
        """无参数；返回现役 provider 单例，初始化本身不创建评测索引。"""
        if self._store is None:
            from app.services.retrieval_provider_registry import get_default_vector_provider

            self._store = get_default_vector_provider()
        return self._store

    async def resolve_scope(self, profile: ExecutionProfile, *, paper_ids: tuple[str, ...]) -> dict[str, Any]:
        """参数为真实配置及请求论文；复用产品正常归属规则并核对未丢失请求论文。"""
        from app.services.upload_read_service import read_stable_project_scope

        scope = await read_stable_project_scope(profile.project_id, user_id=profile.user_id, paper_ids=list(paper_ids))
        if set(scope["paper_ids"]) != set(paper_ids) or not scope.get("local_retrieval_allowed"):
            raise ValueError("actual_business_scope_differs_from_requested_papers")
        if self.publication is None:
            self.publication = scope["publication"]
        return dict(scope)

    async def verify_bindings(self, profile: ExecutionProfile) -> tuple[dict[str, Any], ...]:
        """参数为已声明的实际论文；读取 owner 归属和 PDF 路径，严格核对内容身份。"""
        from app.repositories.paper_repository import get_paper_repository

        papers = await get_paper_repository().get_papers_by_ids([item.paper_id for item in profile.paper_bindings], user_id=profile.user_id)
        by_id = {paper["paper_id"]: paper for paper in papers}
        issues: list[dict[str, Any]] = []
        for binding in profile.paper_bindings:
            paper = by_id.get(binding.paper_id)
            if paper is None:
                issues.append({"paper_key": binding.paper_key, "paper_id": binding.paper_id, "reason": "paper_not_owned_or_not_active"})
                continue
            path = _business_pdf_path(paper, profile)
            if sha256_file(path) != binding.pdf_sha256:
                issues.append({"paper_key": binding.paper_key, "paper_id": binding.paper_id, "reason": "business_pdf_checksum_mismatch"})
        if issues:
            raise PreparationFailure("business_paper_bindings_unverified", details=issues)
        return tuple(deepcopy(papers))

    def require_existing_index(self) -> None:
        """无参数；确认当前 Milvus collection 已存在，缺失时失败而不是创建共享索引。"""
        from app.config import get_settings
        from app.services.retrieval_provider_registry import get_default_vector_provider_name

        if get_default_vector_provider_name() != "milvus":
            raise ValueError("live_index_export_requires_current_milvus_provider")
        if not self.store._get_client().has_collection(collection_name=get_settings().milvus_collection_v2):
            raise ValueError("actual_milvus_collection_missing")

    async def export_index(self, sources: tuple[SourceSnapshot, ...]) -> tuple[IndexSnapshot, list[dict[str, Any]]]:
        """参数为实际原文；导出全部 child/parent 和向量身份，映射缺口保持可审计。"""
        rows = await asyncio.to_thread(self._index_rows, sources)
        self.index_storage_identity = snapshot_hash(snapshot({"rows": rows}))
        chunks = [{**row, "text": row["display_text"]} for row in rows]
        index, issues = build_index_from_chunks(sources, chunks, retrieval_backend="milvus")
        available = {row["paper_id"] for row in rows}
        issues.extend({"paper_id": source.paper_id, "reason": "actual_index_has_no_chunks", "status": "missing"}
                      for source in sources if source.paper_id not in available)
        return index, issues

    def _index_rows(self, sources: tuple[SourceSnapshot, ...]) -> list[dict[str, Any]]:
        """参数为已验证论文；只读遍历真实索引，包含向量正文以检测内容相同但向量变化。"""
        from app.config import get_settings
        from app.services.milvus_store import _COUNT_BATCH_SIZE, _FIELD_OUTPUTS

        self.require_existing_index()
        client = self.store._get_client()
        if self.publication is None:
            raise ValueError("index_export_requires_frozen_publication")
        expression = "paper_id in " + json.dumps(sorted(source.paper_id for source in sources), ensure_ascii=False)
        expression += " and batch_id in " + json.dumps(self.publication["batch_ids"], ensure_ascii=True)
        iterator = client.query_iterator(collection_name=get_settings().milvus_collection_v2,
                                         filter=expression, output_fields=[*_FIELD_OUTPUTS, "dense_vector"], batch_size=_COUNT_BATCH_SIZE)
        rows: list[dict[str, Any]] = []
        try:
            while batch := iterator.next():
                rows.extend(dict(row) for row in batch)
        finally:
            iterator.close()
        return sorted(rows, key=lambda row: (str(row["paper_id"]), str(row["chunk_id"])))

    async def acquire_scope(self, scope: dict[str, Any]) -> dict[str, Any]:
        """参数为已核对的稳定范围；在发布仓储内原子核对批次并取得实际请求占用，返回新范围。"""
        from app.repositories.upload_publication_repository import get_upload_publication_repository

        publication = await get_upload_publication_repository().snapshot(
            scope["paper_ids"], user_id=scope["user_id"], expected_snapshot=scope["publication"])
        active = {**scope, "publication": publication}
        self.active_scopes[snapshot(scope)] = active
        return active

    async def release_scopes(self) -> None:
        """无参数；释放本进程实际领取的全部租约，释放错误向执行器传播。"""
        from app.repositories.upload_publication_repository import get_upload_publication_repository

        for scope in self.active_scopes.values():
            await get_upload_publication_repository().release_snapshot(scope["publication"])

    def execution_scope(self, scope: dict[str, Any]) -> dict[str, Any]:
        """参数为冻结授权范围；返回本次实际领取租约的范围，不允许猜测当前批次。"""
        return deepcopy(self.active_scopes[snapshot(scope)])

    async def snapshot_initial(self, profile: ExecutionProfile, *, papers: tuple[dict[str, Any], ...]) -> str:
        """参数为实际项目与论文；只读固定初始记忆、问题和关联，后续重复不再读变化中的业务仓储。"""
        from app.repositories.research_memory_repository import get_research_memory_repository

        initial: dict[str, Any] = {
            "schema_version": STORAGE_VERSION, "user_id": profile.user_id, "project_id": profile.project_id,
            "papers": list(papers), "questions": [], "items": [], "evidence_refs": [],
            "links": [], "question_links": [], "versions": [], "index_storage_identity": self.index_storage_identity,
        }
        if profile.project_id is None:
            return snapshot(initial)
        repo = get_research_memory_repository()
        initial["questions"] = await repo.list_questions(profile.project_id, user_id=profile.user_id, review_state=None)
        initial["items"] = await repo.list_items(profile.project_id, user_id=profile.user_id, review_state=None)
        initial["links"] = await repo.list_links(profile.project_id, user_id=profile.user_id)
        for item in initial["items"]:
            identity = item["memory_id"]
            initial["evidence_refs"].extend(await repo.list_evidence_refs(identity))
            initial["versions"].extend(await repo.list_versions(identity))
            initial["question_links"].extend({"memory_id": identity, "question_id": question}
                for question in await repo.list_question_ids(identity, user_id=profile.user_id, project_id=profile.project_id))
        return snapshot(initial)


def _business_pdf_path(paper: dict[str, Any], profile: ExecutionProfile) -> Path:
    """参数为真实论文行和允许目录；解析实际保存路径，拒绝缺文件或目录外读取。"""
    raw = paper.get("pdf_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"business_pdf_path_missing:{paper['paper_id']}")
    path = Path(raw)
    if not path.is_absolute():
        path = profile_path("backend") / path
    path = path.resolve()
    roots = (profile_path(profile.uploads_root), profile_path(profile.paper_root))
    if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
        raise ValueError(f"business_pdf_path_unavailable:{paper['paper_id']}")
    return path


@dataclass(frozen=True, kw_only=True)
class RetrievalRuntime:
    """一次检索的真实 provider 与已经验证的固定范围。"""

    store: Any
    scope_snapshot: str
    top_k: int
    index_id: str
    backend: str
    search: Callable[..., list[dict[str, Any]]]


def search_declared_retrieval(store: Any, query: str, top_k: int, *, scope: dict[str, Any], method: str) -> list[dict[str, Any]]:
    """参数为现役存储、问题、K、可信范围及方法；返回同一发布范围内的真实有序检索结果。"""
    from app.services.retrieval_service import require_local_retrieval_scope

    trusted_scope = require_local_retrieval_scope(scope)
    if method == "dense":
        return store.local_rag_search(query, top_k=top_k, filter_paper_ids=list(trusted_scope["paper_ids"]), scope=trusted_scope)
    if method == "hybrid":
        return store.search_scope(query, top_k=top_k, scope=trusted_scope)
    raise ValueError(f"unknown_retrieval_method:{method}")


async def run_retrieval_job(case: EvalCase, runtime: RetrievalRuntime) -> dict[str, Any]:
    """参数为固定问题和真实依赖；仅执行检索并冻结观察，绝不补造回答。"""
    from app.middleware.trace_context import (
        TraceContext, evidence_scope, export_evidence_observations, record_evidence_observation,
        record_retrieval_hits, reset_trace_ctx, set_trace_ctx,
    )

    inputs = read_snapshot(case.input_snapshot)
    scope = read_snapshot(runtime.scope_snapshot)
    invocation = {**inputs, "top_k": runtime.top_k, "project_scope": scope}
    trace = TraceContext(request_id=str(uuid4()), user_id=scope["user_id"])
    token = set_trace_ctx(trace)
    failure: CapturedRunFailure | None = None
    started = time.perf_counter()
    try:
        record_evidence_observation("index_snapshot_declared", {"index_snapshot_id": runtime.index_id, "source": "eval_declared_before_execution"})
        with evidence_scope("retrieval_query", details={"query": inputs["question"], "scope": scope, "retrieval_backend": runtime.backend}):
            hits = await asyncio.to_thread(runtime.search, runtime.store, inputs["question"], runtime.top_k, scope=scope)
            record_retrieval_hits("query_raw", hits, query=inputs["question"], scope=scope, retrieval_backend=runtime.backend)
    except Exception as exc:
        failed = FailedLiveCapture(request_id=trace.request_id, input_snapshot=snapshot(invocation),
                                  observation_snapshot=snapshot(export_evidence_observations(trace)),
                                  error_type=type(exc).__name__, error_message=str(exc))
        failure = CapturedRunFailure(str(exc), failed)
    finally:
        observed = export_evidence_observations(trace)
        observed["latency"] = {"retrieval_seconds": time.perf_counter() - started}
        reset_trace_ctx(token)
    if failure is not None:
        observed["failure_reason"] = f"{failure.capture.error_type}:{failure.capture.error_message}"
    raw = {"id": case.case_id, "case_version": case.case_version, "source_type": case.source_type,
           **read_snapshot(case.input_snapshot), **read_snapshot(case.truth_snapshot)}
    return freeze_record(raw, mode="retrieval_live", execution_input=invocation, observation=observed)


@dataclass(frozen=True, kw_only=True)
class AgentRuntime:
    """把真实图、已授权范围和独立介质集中绑定到一次 episode。"""

    prepared: PreparedSuite
    output_dir: Path
    live: LivePreparationRuntime
    graph_factory: GraphFactory

    def build_graph(self) -> Any:
        """无参数；返回透传现役流事件并逐条持久化的图边界，供强制中断后核查已完成步骤。"""
        return CapturedGraph(graph=self.graph_factory(), path=self.output_dir / "partial-events.jsonl",
                             secrets=export_secrets("agent"))

    def build_turn_input(self, case: AgentCase, user_message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """参数为案例、当轮新消息和已发生历史；返回具有独立研究身份的同范围图输入。"""
        base = agent_case_payload(case)["input"]
        scope = self.live.execution_scope(base["project_scope"])
        focus = base.get("focused_paper_key")
        # 1. 【评估入口】【研究身份】新用户轮次开始新研究；同输入流转沿用此身份，不借用评分编号
        payload: dict[str, Any] = {
            "run_id": str(uuid4()),
            "user_query": user_message, "user_id": self.prepared.profile.user_id,
            "project_id": self.prepared.profile.project_id, "project_scope": scope,
            "filter_expr": scope["filter_expr"], "chat_history": deepcopy(history),
            "structured_query": {**deepcopy(base["structured_query"]), "project_scope": scope},
            "dialogue_mode": "research", "source_status_reasons": {},
        }
        if focus is not None:
            payload["paper_focus"] = {"paper_ids": [focus], "mode": "explicit"}
        return payload


def build_agent_environment(runtime: AgentRuntime) -> AgentEnvironment:
    """参数为真实图和固定初始环境；返回不包含评分真值的集中执行环境。"""
    return AgentEnvironment(
        graph_factory=runtime.build_graph, state_builder=runtime.build_turn_input,
        declared={"user_id": runtime.prepared.profile.user_id, "index_snapshot_id": runtime.prepared.index.index_snapshot_id,
                  "initial_snapshot_sha256": snapshot_hash(runtime.prepared.initial_snapshot), "cache_initial_state": "empty_process_local"},
    )


@dataclass(frozen=True, kw_only=True)
class CapturedGraph:
    """透传产品图，并保留中断前已经实际发出的事件。"""

    graph: Any
    path: Path
    secrets: tuple[str, ...] = ()

    async def astream(self, inputs: dict[str, Any], *, stream_mode: list[str]) -> Any:
        """参数为真实图输入和流模式；逐条写出原始事件后原样返回，不改变节点输出。"""
        from evals.agent_outcome.runner import _to_jsonable

        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(snapshot(sanitize_frozen_record({"input": inputs}, secrets=self.secrets)) + "\n")
            stream.flush()
            async for item in self.graph.astream(inputs, stream_mode=stream_mode):
                stream.write(snapshot(sanitize_frozen_record({"event": _to_jsonable(item)}, secrets=self.secrets)) + "\n")
                stream.flush()
                yield item


def _bind_storage(stack: ExitStack, storage: EvaluationStorage) -> None:
    """参数为退出栈和真实隔离仓储；仅替换持久化端口，业务筛选及写回服务照常执行。"""
    for name in _MEMORY_MODULES:
        module = importlib.import_module(name)
        stack.enter_context(patch.object(module, "get_research_memory_repository", storage.repository))
    stale = importlib.import_module("app.services.research_memory_stale_service")
    stack.enter_context(patch.object(stale, "get_research_memory_repository", storage.repository))
    stack.enter_context(patch.object(stale, "mark_checked_stale", storage.mark_checked_stale))
    stack.enter_context(patch.object(stale, "record_source_checks", storage.record_source_checks))
    observer = importlib.import_module("app.repositories.observability_repository")
    stack.enter_context(patch.object(observer, "get_observability_repository", storage.repository))
    paper_db = importlib.import_module("app.services.paper_db")
    ports = {"get_paper": storage.get_paper, "list_papers": storage.list_papers,
             "list_paper_ids": storage.list_paper_ids, "write_log": storage.write_log_sync,
             "write_rag_trace": storage.write_rag_trace_sync}
    for name, implementation in ports.items():
        stack.enter_context(patch.object(paper_db, name, implementation))
    recommendation = importlib.import_module("app.services.recommendation")
    stack.enter_context(patch.object(recommendation, "push_notification", storage.push_notification))


@dataclass(frozen=True, kw_only=True)
class ModelRequestObserver:
    """在模型 HTTP 端口记录真实公开参数；不保存密钥，不改变请求与回复。"""

    send: Callable[..., Any]

    def __get__(self, client: Any, owner: Any) -> Callable[..., Any]:
        """参数为实际客户端和所属类；返回绑定该客户端的原请求观察器。"""
        return partial(self.invoke, client)

    def invoke(self, client: Any, headers: dict[str, Any], payload: dict[str, Any]) -> Any:
        """参数为真实客户端、请求头与请求体；仅记录公开模型参数后调用原端口。"""
        from app.middleware.trace_context import record_evidence_observation
        from .identity import safe_endpoint

        record_evidence_observation("model_request_parameters", {
            "model": payload["model"], "temperature": payload.get("temperature"),
            "endpoint": safe_endpoint(client.base_url),
        })
        return self.send(client, headers, payload)


@dataclass(frozen=True, kw_only=True)
class SourceFailure:
    """只在实际调用指定原文端口时施加已声明故障；声明本身不算故障发生。"""

    storage: EvaluationStorage
    message: str

    def raise_on_attempt(self, port: str, *args: Any, **kwargs: Any) -> Any:
        """参数为实际端口及调用实参；保存真实尝试并抛出声明的持续 TimeoutError。"""
        from app.middleware.trace_context import record_evidence_observation
        from evals.agent_outcome.runner import _to_jsonable

        self.storage.record_io("source_fault_attempt", {"port": port, "args": _to_jsonable(args), "kwargs": _to_jsonable(kwargs)})
        record_evidence_observation("source_port_failed", {"port": port, "fault_injected": True, "error_type": "TimeoutError", "error": self.message})
        raise TimeoutError(self.message)


def _bind_fault(stack: ExitStack, fault: dict[str, Any] | None, *, runtime: AgentRuntime, storage: EvaluationStorage) -> None:
    """参数为退出栈及独立故障声明；只绑定本评测进程现役原文端口，不改变业务终态。"""
    if fault is None:
        return
    expected = {"behavior": "each_actual_attempt_raises", "target_capability": "all_original_text_retrieval_and_read_ports", "persistence": "whole_case", "error_type": "TimeoutError"}
    if any(fault.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported_source_fault_contract")
    controller = SourceFailure(storage=storage, message=str(fault["error_message"]))
    for name in _ORIGINAL_TEXT_PORTS:
        stack.enter_context(patch.object(runtime.live.store, name, partial(controller.raise_on_attempt, name)))


@contextmanager
def isolated_runtime(runtime: AgentRuntime, *, fault: dict[str, Any] | None = None) -> Iterator[AgentEnvironment]:
    """参数为独立进程依赖及故障计划；保存真实 IO 并恢复端口，模型和业务节点保持现役实现。"""
    from app.services.cache import CacheManager
    from app.services.llm_service import OpenAICompatibleLLM

    storage = EvaluationStorage(runtime.prepared.initial_snapshot, output_dir=runtime.output_dir / "storage",
                                sanitize=partial(sanitize_frozen_record, secrets=export_secrets("agent")))
    environment = build_agent_environment(runtime)
    try:
        with ExitStack() as stack:
            _bind_storage(stack, storage)
            observer = ModelRequestObserver(send=OpenAICompatibleLLM._post_chat_completions)
            stack.enter_context(patch.object(OpenAICompatibleLLM, "_post_chat_completions", observer))
            stream_observer = ModelRequestObserver(send=OpenAICompatibleLLM._post_chat_completions_stream)
            stack.enter_context(patch.object(OpenAICompatibleLLM, "_post_chat_completions_stream", stream_observer))
            stack.enter_context(patch("app.services.cache._cache", CacheManager()))
            stack.enter_context(patch.object(runtime.live.store, "ensure_collection", runtime.live.require_existing_index))
            _bind_fault(stack, fault, runtime=runtime, storage=storage)
            yield environment
    finally:
        environment.environment_issues.extend(storage.errors)
        storage.persist_snapshot()


async def run_agent_job(case: AgentCase, runtime: AgentRuntime) -> dict[str, Any]:
    """参数为冻结场景及真实依赖；运行完整 episode 并保存隔离介质中的真实 IO 错误。"""
    from evals.agent_outcome.contracts import freeze_agent_run

    fault = read_snapshot(case.truth_snapshot).get("fault_injection")
    with isolated_runtime(runtime, fault=fault) as environment:
        started = time.perf_counter()
        record = await run_episode(case, environment, run_mode="agent_live")
        duration = time.perf_counter() - started
    if environment.environment_issues:
        run = record["agent_run"]
        observed = read_snapshot(run["observation_snapshot"])
        record = freeze_agent_run(case, observed["turns"], run_mode="agent_live", environment=environment.snapshot(),
                                  run_id=run["run_id"], status=run["status"], missing_fields=("environment_access",))
    return _capture_agent_resources(record, duration=duration)


def _capture_agent_resources(record: dict[str, Any], *, duration: float) -> dict[str, Any]:
    """参数为实际Agent记录与episode时钟；冻结模型原始用量和缺失数，不估填token或费用。"""
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    captures = [read_snapshot(turn["evidence_observation_snapshot"]) for turn in observed["turns"]]
    calls = [event["payload"] for capture in captures
             for event in capture.get("evidence_observations", [])
             if event["stage"] == "model_response_usage"]
    # 1. 【评测】【实际用量】某次用量采集失败后，其他调用的有效用量不能把整体缺失改成已测量
    usage_capture_failed = any(error["stage"] == "model_response_usage" for capture in captures
                               for error in capture.get("evidence_capture_errors", []))
    usage_complete = bool(calls) and not usage_capture_failed and all(call["usage_status"] == "measured" for call in calls)
    actual = {**observed, "latency": {"episode_seconds": duration},
              "usage": {"model_http_calls": calls, "observed_call_count": len(calls),
                        "status": "measured" if usage_complete else "missing"},
              "cost": {"status": "missing", "reason": "no_frozen_price_schedule"}}
    observation_snapshot = snapshot(actual)
    return {**record, "run_record": {**record["run_record"], "observation_snapshot": observation_snapshot},
            "agent_run": {**record["agent_run"], "observation_snapshot": observation_snapshot}}
