"""Ragas 评估输入适配层。"""

from __future__ import annotations

import asyncio
import json
import inspect
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from evals.contracts import OBSERVATION_FIELDS, freeze_record, validate_execution_input
from evals.evidence.live_capture import CapturedRunFailure, capture_live_run
from evals.ragas.eval_contracts import extract_contract_fields, with_contract_fields

REQUIRED_FIELDS = ("question", "retrieved_contexts", "response", "reference_answer")
LIVE_MODE_CHOICES = ("auto", "milvus", "graph", "fusion")
GRAPH_MODE_CHOICES = ("local", "global", "hybrid")
logger = logging.getLogger(__name__)


def normalize_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    """把单条样本规范化成稳定结构。"""
    question = str(record.get("question") or "").strip()
    response = str(record.get("response") or "").strip()
    reference_answer = str(record.get("reference_answer") or "").strip()
    retrieved_contexts = record.get("retrieved_contexts", [])
    if not question:
        raise ValueError("评估样本缺少 question")
    if not isinstance(retrieved_contexts, list):
        raise ValueError("评估样本缺少 retrieved_contexts")
    if any(not isinstance(item, str) for item in retrieved_contexts):
        raise ValueError("retrieved_contexts must contain only text strings")
    normalized_contexts = [item.strip() for item in retrieved_contexts if item.strip()]
    normalized: dict[str, Any] = {
        "question": question,
        "reference_answer": reference_answer,
    }
    if "retrieved_contexts" in record:
        normalized["retrieved_contexts"] = normalized_contexts
    if "response" in record:
        normalized["response"] = response
    if isinstance(record.get("metadata"), dict):
        normalized["metadata"] = record["metadata"]
    for field in ("id", "case_version", "dataset_version", "source_type", "eval_case", "run_record", *OBSERVATION_FIELDS):
        if field in record and field not in {"response", "retrieved_contexts"}:
            normalized[field] = deepcopy(record[field])
    return with_contract_fields(normalized, extract_contract_fields(record))


def load_jsonl_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """加载 frozen JSONL 样本集。"""
    path = Path(dataset_path)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(normalize_eval_record(json.loads(stripped)))
    if not records:
        raise ValueError(f"评估数据集为空: {path}")
    return records


def load_live_jsonl_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """加载 live 样本；问题必需，缺少参考答案时保留评分数据缺口。"""
    path = Path(dataset_path)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        question = str(payload.get("question") or payload.get("user_input") or "").strip()
        reference_answer = str(payload.get("reference_answer") or payload.get("reference") or "").strip()
        if not question:
            raise ValueError("live 评估样本缺少 question")
        record: dict[str, Any] = {
            "question": question,
            "reference_answer": reference_answer,
        }
        structured_query = payload.get("structured_query")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if not isinstance(structured_query, dict):
            structured_query = metadata.get("structured_query")
        if isinstance(structured_query, dict):
            record["structured_query"] = dict(structured_query)
        filter_expr = str(payload.get("filter_expr") or "").strip()
        if not filter_expr:
            filter_expr = str(metadata.get("filter_expr") or "").strip()
        if filter_expr:
            record["filter_expr"] = filter_expr
        paper_ids = _normalize_live_paper_ids(payload)
        if paper_ids:
            record["paper_ids"] = paper_ids
        paper_titles = _normalize_live_paper_titles(payload)
        if paper_titles:
            record["paper_titles"] = paper_titles
        for scope_key in ("library_id", "user_id", "project_id"):
            scope_value = str(payload.get(scope_key) or metadata.get(scope_key) or "").strip()
            if scope_value:
                record[scope_key] = scope_value
        dialogue_mode = _normalize_optional_dialogue_mode(payload.get("dialogue_mode") or metadata.get("dialogue_mode"))
        if dialogue_mode:
            record["dialogue_mode"] = dialogue_mode
        if metadata:
            record["metadata"] = dict(metadata)
        record = with_contract_fields(record, extract_contract_fields(payload))
        for key in ("id", "case_version", "dataset_version", "source_type"):
            if key in payload:
                record[key] = deepcopy(payload[key])
        records.append(record)
    if not records:
        raise ValueError(f"评估数据集为空: {path}")
    return records


def _normalize_live_paper_ids(payload: dict[str, Any]) -> list[str]:
    paper_ids = payload.get("paper_ids")
    if isinstance(paper_ids, list):
        normalized = [str(item).strip() for item in paper_ids if str(item).strip()]
        if normalized:
            return normalized
    paper_id = str(payload.get("paper_id") or "").strip()
    if paper_id:
        return [paper_id]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_paper_ids = metadata.get("paper_ids")
        if isinstance(metadata_paper_ids, list):
            normalized = [str(item).strip() for item in metadata_paper_ids if str(item).strip()]
            if normalized:
                return normalized
        metadata_paper_id = str(metadata.get("paper_id") or "").strip()
        if metadata_paper_id:
            return [metadata_paper_id]
    return []


def _normalize_live_paper_titles(payload: dict[str, Any]) -> list[str]:
    paper_titles = payload.get("paper_titles")
    if isinstance(paper_titles, list):
        normalized = [str(item).strip() for item in paper_titles if str(item).strip()]
        if normalized:
            return normalized
    paper_title = str(payload.get("paper_title") or "").strip()
    if paper_title:
        return [paper_title]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_paper_titles = metadata.get("paper_titles")
        if isinstance(metadata_paper_titles, list):
            normalized = [str(item).strip() for item in metadata_paper_titles if str(item).strip()]
            if normalized:
                return normalized
        metadata_paper_title = str(metadata.get("paper_title") or "").strip()
        if metadata_paper_title:
            return [metadata_paper_title]
    return []


class ReplayRAGAdapter:
    """最小可运行的 replay 适配器。"""

    adapter_name = "replay"

    def load(self, dataset_path: str | Path) -> list[dict[str, Any]]:
        return load_jsonl_dataset(dataset_path)

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.adapter_name}


class LiveRAGAdapter:
    """真实调用当前项目 rag_node 的 live adapter。"""

    adapter_name = "live"

    def __init__(
        self,
        *,
        mode: str = "auto",
        graph_mode: str = "hybrid",
        rag_node_factory: Callable[[], Any] | None = None,
        vector_provider_factory: Callable[[], Any] | None = None,
        evidence_index_snapshot_id: str | None = None,
    ) -> None:
        self.mode = _normalize_live_mode(mode)
        self.graph_mode = _normalize_graph_mode(graph_mode)
        self._rag_node_factory = rag_node_factory
        self._vector_provider_factory = vector_provider_factory
        self._evidence_index_snapshot_id = evidence_index_snapshot_id

    def load(self, dataset_path: str | Path) -> list[dict[str, Any]]:
        samples = [self._resolve_sample_scope(sample) for sample in load_live_jsonl_dataset(dataset_path)]
        return asyncio.run(self._load_async(samples))

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "mode": self.mode,
            "graph_mode": self.graph_mode,
            "requested_mode": self.mode,
            "requested_graph_mode": self.graph_mode,
        }

    async def _load_async(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """参数为已解析范围的样本；逐题执行真实图并冻结观察，失败保留该题的快照和原始异常。"""
        rag_graph = self._get_rag_graph()
        records: list[dict[str, Any]] = []
        for sample in samples:
            try:
                live_input = self._build_live_input(sample)
                result = await capture_live_run(
                    rag_graph, live_input, index_snapshot_id=self._evidence_index_snapshot_id,
                )
            except CapturedRunFailure as exc:
                message = str(self._wrap_live_eval_error(sample=sample, exc=exc))
                raise CapturedRunFailure(message, exc.capture) from exc
            except Exception as exc:
                raise self._wrap_live_eval_error(sample=sample, exc=exc) from exc
            record = normalize_eval_record(self._build_live_record(sample, result))
            records.append(freeze_record(record, mode="live", execution_input=live_input))
        return records

    def _get_rag_graph(self) -> Any:
        if self._rag_node_factory is not None:
            return self._rag_node_factory()
        from app.nodes.rag_node import create_rag_node

        return create_rag_node()

    def _build_live_input(self, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为执行样本；返回具有独立研究身份的 live 输入，并拒绝嵌套真值。"""
        sample = deepcopy(sample)
        # 1. 【评估入口】【研究身份】本次新研究身份随输入传给图，独立于样本、评分和请求追踪
        payload = {
            "user_query": sample["question"],
            "run_id": str(uuid4()),
            "source_status_reasons": {},
        }
        if self.mode != "auto":
            payload["retrieval_mode_override"] = self.mode
            payload["graph_mode_override"] = self.graph_mode
        dialogue_mode = _normalize_optional_dialogue_mode(sample.get("dialogue_mode"))
        if dialogue_mode:
            payload["dialogue_mode"] = dialogue_mode
        structured_query = sample.get("structured_query")
        if isinstance(structured_query, dict):
            payload["structured_query"] = dict(structured_query)
        elif sample.get("paper_ids"):
            payload["structured_query"] = {
                "paper_ids": list(sample["paper_ids"]),
                "resolved_scope": {"paper_ids": list(sample["paper_ids"])},
            }
        if sample.get("paper_titles"):
            structured_query = payload.get("structured_query")
            if not isinstance(structured_query, dict):
                structured_query = {}
            structured_query["paper_titles"] = list(sample["paper_titles"])
            payload["structured_query"] = structured_query
        if dialogue_mode:
            structured_query = payload.get("structured_query")
            if not isinstance(structured_query, dict):
                structured_query = {}
            structured_query["dialogue_mode"] = dialogue_mode
            payload["structured_query"] = structured_query
        if sample.get("project_id"):
            structured_query = payload.get("structured_query")
            if not isinstance(structured_query, dict):
                structured_query = {}
            resolved_scope = structured_query.get("resolved_scope")
            if not isinstance(resolved_scope, dict):
                resolved_scope = {}
            resolved_scope["project_id"] = str(sample["project_id"])
            if sample.get("paper_ids") and not resolved_scope.get("paper_ids"):
                resolved_scope["paper_ids"] = list(sample["paper_ids"])
            structured_query["project_id"] = str(sample["project_id"])
            structured_query["resolved_scope"] = resolved_scope
            payload["structured_query"] = structured_query
        filter_expr = str(sample.get("filter_expr") or "").strip()
        if filter_expr:
            payload["filter_expr"] = filter_expr
        validate_execution_input(payload)
        return payload

    def _resolve_sample_scope(self, sample: dict[str, Any]) -> dict[str, Any]:
        resolved = deepcopy(sample)
        if isinstance(sample.get("metadata"), dict):
            resolved["metadata"] = dict(sample["metadata"])
        scope_warnings: list[str] = []
        if not resolved.get("paper_ids") and resolved.get("paper_titles"):
            try:
                resolved["paper_ids"] = _resolve_live_paper_ids_from_titles(
                    resolved["paper_titles"],
                    library_id=str(resolved.get("library_id") or "").strip() or None,
                    user_id=str(resolved.get("user_id") or "").strip() or None,
                )
            except ValueError as exc:
                scope_warnings.append(str(exc))
        project_id = str(resolved.get("project_id") or "").strip()
        if project_id:
            project_context = _resolve_live_project_context(
                project_id=project_id,
                requested_paper_ids=resolved.get("paper_ids"),
                user_id=str(resolved.get("user_id") or "").strip() or "anonymous",
            )
            if project_context.get("paper_ids"):
                scope_warnings.extend(_scope_mismatch_warnings(resolved.get("paper_ids"), project_context.get("paper_ids")))
                resolved["paper_ids"] = list(project_context["paper_ids"])
            if project_context.get("filter_expr"):
                resolved["filter_expr"] = str(project_context["filter_expr"])
            if isinstance(project_context.get("resolved_scope"), dict):
                structured_query = resolved.get("structured_query")
                if not isinstance(structured_query, dict):
                    structured_query = {}
                structured_query["project_id"] = project_id
                existing_scope = structured_query.get("resolved_scope")
                merged_scope: dict[str, Any] = dict(existing_scope) if isinstance(existing_scope, dict) else {}
                merged_scope.update(project_context["resolved_scope"])
                structured_query["resolved_scope"] = merged_scope
                if resolved.get("paper_ids"):
                    structured_query["paper_ids"] = list(resolved["paper_ids"])
                resolved["structured_query"] = structured_query
        if scope_warnings:
            metadata = dict(resolved.get("metadata")) if isinstance(resolved.get("metadata"), dict) else {}
            metadata["scope_resolution_warnings"] = scope_warnings
            resolved["metadata"] = metadata
        return resolved

    def _build_live_record(self, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        requested_mode = self.mode
        requested_graph_mode = self.graph_mode
        resolved_mode = _resolve_live_result_mode(result)
        paper_sources = result["paper_sources"] if isinstance(result.get("paper_sources"), list) else []
        inventory_contexts = _normalize_inventory_contexts(
            result.get("inventory_contexts"),
            paper_sources,
        )
        if not resolved_mode:
            if inventory_contexts:
                resolved_mode = "inventory"
            else:
                resolved_mode = requested_mode if requested_mode != "auto" else "unknown"
        resolved_graph_mode = str(result.get("graph_mode") or "").strip().lower() or requested_graph_mode
        source_status = result.get("source_status") if isinstance(result.get("source_status"), dict) else {}
        search_results = result["search_results"] if isinstance(result.get("search_results"), list) else []
        graph_context = str(result.get("graph_context") or "").strip()
        vector_contexts = self._build_vector_contexts(search_results)
        retrieved_contexts = _merge_live_contexts(
            retrieval_mode=resolved_mode,
            vector_contexts=vector_contexts,
            graph_context=graph_context,
            inventory_contexts=inventory_contexts,
            failure_reason=str(result.get("failure_reason") or "").strip(),
        )
        response = str(result.get("answer") or "").strip()
        citations = result["citations"] if isinstance(result.get("citations"), list) else []
        chunk_citations = result["chunk_citations"] if isinstance(result.get("chunk_citations"), list) else []
        eval_citations = _normalize_eval_citations(citations, chunk_citations)
        metadata = {}
        if isinstance(sample.get("metadata"), dict):
            metadata.update({key: deepcopy(sample["metadata"][key]) for key in (
                "scope_resolution_warnings", "requires_citation", "citation_required", "retrieval_policy",
            ) if key in sample["metadata"]})
        metadata.update(
            {
                "adapter": self.adapter_name,
                "requested_mode": requested_mode,
                "requested_graph_mode": requested_graph_mode,
                "requested_dialogue_mode": _normalize_optional_dialogue_mode(sample.get("dialogue_mode")),
                "resolved_mode": resolved_mode,
                "resolved_graph_mode": resolved_graph_mode,
                "retrieval_mode": resolved_mode,
                "graph_mode": resolved_graph_mode,
                "source_status": source_status,
                "answer_confidence": str(result.get("answer_confidence") or "").strip(),
                "citations": eval_citations,
                "citations_count": len(eval_citations),
                "chunk_citations_count": len(chunk_citations),
            }
        )
        dialogue_mode = _normalize_optional_dialogue_mode(result.get("dialogue_mode")) or _normalize_optional_dialogue_mode(sample.get("dialogue_mode"))
        if dialogue_mode:
            metadata["dialogue_mode"] = dialogue_mode
        if result.get("retrieval_reasoning"):
            metadata["retrieval_reasoning"] = str(result["retrieval_reasoning"])
        inventory_source_ids = _normalize_inventory_source_ids(
            result.get("inventory_source_ids"),
            paper_sources,
        )
        if inventory_source_ids:
            metadata["paper_source_ids"] = inventory_source_ids
            metadata["paper_sources_count"] = len(inventory_source_ids)
        if inventory_contexts:
            metadata["inventory_contexts_count"] = len(inventory_contexts)
        synthesis_state = result.get("synthesis_state") if isinstance(result.get("synthesis_state"), dict) else {}
        if synthesis_state:
            metadata["synthesis_state"] = synthesis_state
            metadata["synthesis_status"] = str(synthesis_state.get("status") or "").strip()
            if isinstance(synthesis_state.get("evidence_layers"), dict):
                metadata["evidence_layers"] = dict(synthesis_state["evidence_layers"])
            elif isinstance(synthesis_state.get("evidence_layers"), list):
                metadata["evidence_layers"] = list(synthesis_state["evidence_layers"])
            if isinstance(synthesis_state.get("project_scope"), dict):
                metadata["project_scope"] = dict(synthesis_state["project_scope"])
            if isinstance(synthesis_state.get("paper_focus"), dict):
                metadata["paper_focus"] = dict(synthesis_state["paper_focus"])
            if isinstance(synthesis_state.get("supported_dimensions"), list):
                metadata["supported_dimensions"] = list(synthesis_state["supported_dimensions"])
            if isinstance(synthesis_state.get("unsupported_dimensions"), list):
                metadata["unsupported_dimensions"] = list(synthesis_state["unsupported_dimensions"])
            if isinstance(synthesis_state.get("open_questions"), list):
                metadata["open_questions"] = list(synthesis_state["open_questions"])
            if "citations_count" in synthesis_state:
                metadata["synthesis_citations_count"] = synthesis_state.get("citations_count")
        research_objects = result.get("research_objects") if isinstance(result.get("research_objects"), list) else []
        if research_objects:
            metadata["research_objects_count"] = len(research_objects)
        note_actions = result.get("note_actions") if isinstance(result.get("note_actions"), list) else []
        if note_actions:
            metadata["note_actions_count"] = len(note_actions)
        if sample.get("paper_ids"):
            metadata["paper_ids"] = list(sample["paper_ids"])
        if sample.get("paper_titles"):
            metadata["paper_titles"] = list(sample["paper_titles"])
        if sample.get("project_id"):
            metadata["project_id"] = str(sample["project_id"])
        if sample.get("filter_expr"):
            metadata["filter_expr"] = str(sample["filter_expr"])
        if isinstance(sample.get("structured_query"), dict):
            metadata["structured_query"] = dict(sample["structured_query"])
            resolved_scope = sample["structured_query"].get("resolved_scope")
            if isinstance(resolved_scope, dict):
                metadata["resolved_scope"] = dict(resolved_scope)
        live_record = {
            "question": sample["question"],
            "retrieved_contexts": retrieved_contexts,
            "response": response,
            "reference_answer": sample["reference_answer"],
            "metadata": metadata,
            "failure_reason": str(result.get("failure_reason") or ""),
        }
        live_record.update(_capture_runtime_observation(result))
        if "answer" not in result:
            live_record.pop("response")
        live_record["context_origin"] = "adapter_reconstructed_from_runtime_hits"
        for key in ("id", "case_version", "dataset_version", "source_type"):
            if key in sample:
                live_record[key] = deepcopy(sample[key])
        return with_contract_fields(live_record, extract_contract_fields(sample))

    def _build_vector_contexts(self, search_results: list[dict[str, Any]]) -> list[str]:
        contexts: list[str] = []
        already_read: set[str] = set()
        store_holder: dict[str, Any] = {"store": None}

        def get_vector_store() -> Any:
            if store_holder["store"] is None:
                store_holder["store"] = self._get_vector_store()
            return store_holder["store"]

        for hit in search_results:
            context = _extract_context_from_hit(
                hit,
                get_vector_store=get_vector_store,
                already_read=already_read,
            )
            if context:
                contexts.append(context)
        return contexts

    def _get_vector_store(self) -> Any:
        if self._vector_provider_factory is not None:
            return self._vector_provider_factory()
        from app.services.retrieval_provider_registry import get_default_vector_provider

        return get_default_vector_provider()

    def _wrap_live_eval_error(self, *, sample: dict[str, Any], exc: Exception) -> RuntimeError:
        question = str(sample.get("question") or "").strip()
        message = str(exc)
        if "Milvus health probe failed" in message:
            return RuntimeError(
                "live ragas 评估在调用真实 rag_node 时失败。"
                f" 样本问题: {question!r}。"
                " 根因是当前路由需要 Milvus，但本地 Milvus 不可用。"
                " 如果只是验证图检索链路，可先尝试 `python -m evals.ragas.evals ... --adapter live --mode graph`。"
                " 如果要评估 `--mode auto` 或 `--mode milvus/fusion` 的真实行为，请先启动 Milvus 后再重跑。"
                f" 原始错误: {message}"
            )
        return RuntimeError(
            "live ragas 评估在调用真实 rag_node 时失败。"
            f" 样本问题: {question!r}。"
            f" 原始错误: {message}"
        )


def _capture_runtime_observation(result: dict[str, Any]) -> dict[str, Any]:
    """参数为图返回；复制已有观察，trace 使用其序列化接口，未返回字段不补造。"""
    observation = {}
    for key in OBSERVATION_FIELDS:
        if key not in result or key in {"retrieved_contexts", "response"}:
            continue
        value = result[key]
        if key == "trace" and callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        observation[key] = deepcopy(value)
    return observation


def _normalize_live_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower() or "auto"
    if normalized not in LIVE_MODE_CHOICES:
        raise ValueError(f"不支持的 live 检索模式: {mode}")
    return normalized


def _normalize_optional_dialogue_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"chat", "lookup", "research"} else ""


def _normalize_graph_mode(graph_mode: str) -> str:
    normalized = str(graph_mode or "hybrid").strip().lower() or "hybrid"
    if normalized not in GRAPH_MODE_CHOICES:
        raise ValueError(f"不支持的 graph_mode: {graph_mode}")
    return normalized


def _resolve_live_result_mode(result: dict[str, Any]) -> str:
    for value in (
        result.get("retrieval_mode"),
        _nested_mode(result.get("trace")),
        _nested_mode(result.get("rag_trace")),
        _nested_mode(result.get("trace_summary")),
    ):
        mode = str(value or "").strip().lower()
        if mode:
            return mode
    return ""


def _nested_mode(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("mode") or value.get("retrieval_mode") or "").strip()
    return str(getattr(value, "mode", getattr(value, "retrieval_mode", "")) or "").strip()


def _merge_live_contexts(
    *,
    retrieval_mode: str,
    vector_contexts: list[str],
    graph_context: str,
    inventory_contexts: list[str],
    failure_reason: str,
) -> list[str]:
    if retrieval_mode == "inventory":
        contexts = list(inventory_contexts)
    elif retrieval_mode == "graph":
        contexts = [graph_context] if graph_context else []
    elif retrieval_mode == "fusion":
        contexts = [*vector_contexts, *([graph_context] if graph_context else [])]
    elif retrieval_mode == "milvus":
        contexts = list(vector_contexts)
    else:
        contexts = [*vector_contexts, *([graph_context] if graph_context else []), *inventory_contexts]
    if contexts:
        return contexts
    return []


def _normalize_eval_citations(
    citations: list[dict[str, Any]],
    chunk_citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_items = citations if citations else chunk_citations
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in source_items:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        chunk_id = str(item.get("chunk_id") or "").strip()
        section = str(item.get("section") or "").strip()
        dedupe_key = (paper_id, chunk_id, section)
        if dedupe_key in seen:
            continue

        payload: dict[str, Any] = {}
        for field in ("evidence_id", "paper_id", "chunk_id", "section", "page_num", "excerpt"):
            value = item.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    continue
                payload[field] = text
            else:
                payload[field] = value

        if not payload:
            continue
        seen.add(dedupe_key)
        normalized.append(payload)

    return normalized


def _extract_context_from_hit(
    hit: dict[str, Any],
    *,
    get_vector_store: Callable[[], Any],
    already_read: set[str],
) -> str:
    full_text = str(hit.get("full_text") or "").strip()
    if full_text:
        return full_text
    text = str(hit.get("text") or "").strip()
    if text:
        return text

    chunk_id = str(hit.get("chunk_id") or "").strip()
    if chunk_id:
        try:
            read_result = get_vector_store().read_chunk(chunk_id, already_read)
        except Exception as exc:
            logger.warning("live ragas 读取 chunk 全文失败: chunk_id=%s error=%s", chunk_id, exc)
            read_result = None
        if isinstance(read_result, dict):
            read_text = str(read_result.get("full_text") or "").strip()
            if read_text and not read_result.get("skipped"):
                return read_text

    return str(hit.get("preview") or "").strip()


def _normalize_inventory_contexts(raw_contexts: Any, paper_sources: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            text = str(item or "").strip()
            if text:
                contexts.append(text)
    if contexts:
        return contexts
    for paper in paper_sources:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or paper.get("paper_id") or "").strip()
        paper_id = str(paper.get("paper_id") or "").strip()
        authors = ", ".join(str(item).strip() for item in paper.get("authors", []) if str(item).strip())
        summary = str(paper.get("abstract") or "").strip()
        lines = []
        if title:
            lines.append(f"title: {title}")
        if paper_id:
            lines.append(f"paper_id: {paper_id}")
        if authors:
            lines.append(f"authors: {authors}")
        if summary:
            lines.append(f"abstract: {summary}")
        if lines:
            contexts.append("\n".join(lines))
    return contexts


def _normalize_inventory_source_ids(raw_ids: Any, paper_sources: list[dict[str, Any]]) -> list[str]:
    source_items = raw_ids if isinstance(raw_ids, list) else [
        paper.get("paper_id")
        for paper in paper_sources
        if isinstance(paper, dict)
    ]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in source_items:
        paper_id = str(item or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)
    return normalized


def _resolve_live_paper_ids_from_titles(
    paper_titles: list[str],
    *,
    library_id: str | None = None,
    user_id: str | None = None,
) -> list[str]:
    from app.services import paper_db

    resolved_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_title in paper_titles:
        title = str(raw_title or "").strip()
        if not title:
            continue
        matches = paper_db.search_by_exact_title(title, library_id=library_id, user_id=user_id)
        if not matches:
            raise ValueError(
                "live 评估样本里的 paper_title 未命中本地论文库。"
                f" title={title!r}。"
                " 请确认标题与库中 title 完全一致，或改用 paper_id/paper_ids。"
            )
        if len(matches) > 1:
            matched_ids = [
                paper_id
                for paper_id in (_paper_id_from_match(match) for match in matches)
                if paper_id
            ]
            raise ValueError(
                "live 评估样本里的 paper_title 命中了多篇论文，无法稳定解析作用域。"
                f" title={title!r} matched_paper_ids={matched_ids}。"
                " 请补充 user_id/library_id 缩小范围，或直接改用 paper_id/paper_ids。"
            )
        paper_id = _paper_id_from_match(matches[0])
        if not paper_id:
            raise ValueError(
                "live 评估样本里的 paper_title 命中的论文缺少 paper_id。"
                f" title={title!r}。"
                " 请修复本地论文库记录，或直接改用 paper_id/paper_ids。"
            )
        if paper_id and paper_id not in seen_ids:
            resolved_ids.append(paper_id)
            seen_ids.add(paper_id)
    return resolved_ids


def _paper_id_from_match(match: Any) -> str:
    if isinstance(match, dict):
        return str(match.get("paper_id") or "").strip()
    return str(getattr(match, "paper_id", "") or "").strip()


def _scope_mismatch_warnings(
    requested_paper_ids: Any,
    resolved_paper_ids: Any,
) -> list[str]:
    requested = _normalized_id_set(requested_paper_ids)
    resolved = _normalized_id_set(resolved_paper_ids)
    if not requested or not resolved or requested.issubset(resolved):
        return []
    return [
        "live 评估样本的显式 paper_ids 与 project scope 解析结果不一致。"
        f" requested_paper_ids={sorted(requested)} resolved_paper_ids={sorted(resolved)}。"
    ]


def _normalized_id_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item or "").strip()}


def _resolve_live_project_context(
    *,
    project_id: str,
    requested_paper_ids: list[str] | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    from app.services.project_scope_service import build_project_context

    return asyncio.run(
        build_project_context(
            project_id,
            requested_paper_ids=requested_paper_ids,
            user_id=user_id,
        )
    )


def build_ragas_dataset(records: list[dict[str, Any]]) -> Any:
    """把规范化样本适配成 Ragas EvaluationDataset。"""
    try:
        from ragas import EvaluationDataset
    except ImportError:
        EvaluationDataset = None
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError:
        try:
            from ragas import SingleTurnSample  # type: ignore[attr-defined]
        except ImportError as exc:
            raise RuntimeError("未安装 ragas，无法构建 EvaluationDataset") from exc

    samples = [_build_single_turn_sample(SingleTurnSample, record) for record in records]
    if EvaluationDataset is None:
        return samples
    if hasattr(EvaluationDataset, "from_list"):
        payload = [_sample_to_plain_dict(sample) for sample in samples]
        return EvaluationDataset.from_list(payload)
    try:
        return EvaluationDataset(samples=samples)
    except TypeError:
        return EvaluationDataset(samples)


def _build_single_turn_sample(single_turn_cls: Any, record: dict[str, Any]) -> Any:
    signature = inspect.signature(single_turn_cls)
    candidate_kwargs = {
        "user_input": record["question"],
        "question": record["question"],
        "retrieved_contexts": record["retrieved_contexts"],
        "response": record["response"],
        "answer": record["response"],
        "reference": record["reference_answer"],
        "reference_answer": record["reference_answer"],
    }
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    return single_turn_cls(**kwargs)


def _sample_to_plain_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "model_dump"):
        return sample.model_dump()
    if hasattr(sample, "dict"):
        return sample.dict()
    if isinstance(sample, dict):
        return sample
    return dict(sample)
