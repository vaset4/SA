"""Ragas 离线评估入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation, LLMResult

from evals.contracts import (
    CONTRACT_VERSION,
    build_grading,
    freeze_record,
    measurement_coverage,
    measurement_lines,
    merge_measurement_coverage,
    project_legacy_record,
    read_snapshot,
)
from evals.ragas.eval_contracts import contract_measurements
from evals.ragas.measurements import build_measurements, integrity_checks, ragas_eligible
from evals.ragas.rag_adapter import LiveRAGAdapter, ReplayRAGAdapter, build_ragas_dataset

DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "single_turn_eval.jsonl"
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "datasets" / "wave8_live_regression_manifest.json"
DEFAULT_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
DEFAULT_LOGS_DIR = Path(__file__).parent / "logs"
METRIC_NAMES = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
PROJECT_METRIC_NAMES = (
    "scope_adherence",
    "citation_presence_rate",
    "placeholder_citation_rate",
    "route_correctness",
    "critical_fact_coverage",
    "critical_fact_error_rate",
    "golden_evidence_presence_rate",
    "fact_contract_score",
    "answer_payload_present",
    "dialogue_mode_alignment",
    "synthesis_payload_present",
)
CONTRACT_METRIC_NAMES = ("scope_adherence", "citation_presence_rate", "placeholder_citation_rate")
DEBUG_METRIC_NAMES = (
    "route_correctness",
    "critical_fact_coverage",
    "critical_fact_error_rate",
    "golden_evidence_presence_rate",
    "fact_contract_score",
    "answer_payload_present",
    "dialogue_mode_alignment",
    "synthesis_payload_present",
)
SMOKE_QUALITY_FLOOR = 0.01
SMOKE_PLACEHOLDER_CITATION_CEILING = 0.0
SMOKE_SCOPE_ADHERENCE_FLOOR = 1.0
SMOKE_FACT_CONTRACT_FLOOR = 1.0
ADAPTER_CHOICES = ("replay", "live")
LIVE_MODE_CHOICES = ("auto", "milvus", "graph", "fusion")
GATE_MODE_CHOICES = ("smoke", "regression")
EXPLICIT_ROUTE_REQUEST_MODES = {"milvus", "graph", "fusion"}
KNOWN_ROUTE_RESOLVED_MODES = {"milvus", "graph", "fusion", "inventory", "none"}
INLINE_CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")
PLACEHOLDER_CITATION_PATTERN = re.compile(r"^(paper_id|chunk_id|citation|source|ref(?:erence)?)$", re.IGNORECASE)
METRIC_DIRECTIONS = {name: "higher" for name in METRIC_NAMES}
PROJECT_METRIC_DIRECTIONS = {
    "scope_adherence": "higher",
    "citation_presence_rate": "higher",
    "placeholder_citation_rate": "lower",
    "route_correctness": "higher",
    "critical_fact_coverage": "higher",
    "critical_fact_error_rate": "lower",
    "golden_evidence_presence_rate": "higher",
    "fact_contract_score": "higher",
    "answer_payload_present": "higher",
    "dialogue_mode_alignment": "higher",
    "synthesis_payload_present": "higher",
}


class _ProjectLangchainEmbeddings(Embeddings):
    """把项目本地 EmbeddingClient 适配成 ragas 可接收的 embeddings。"""

    def __init__(self, *, client: Any, model_name: str) -> None:
        self._client = client
        self.model = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._client.embed_texts([text])
        if not vectors:
            raise RuntimeError("项目 embedding 客户端没有返回 query 向量")
        return vectors[0]


class _ProjectRagasLLM:
    """把项目 OpenAICompatibleLLM 显式注入 ragas，避免默认 OpenAI evaluator。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout: int,
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        self.model = str(model or "").strip()
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip()
        self._default_temperature = float(temperature)
        self._timeout = int(timeout)
        self._min_request_interval_seconds = max(float(min_request_interval_seconds or 0.0), 0.0)
        if not self._api_key:
            raise RuntimeError("项目配置缺少 `LLM_API_KEY`，无法构造 ragas LLM")
        if not self._base_url:
            raise RuntimeError("项目配置缺少 `LLM_BASE_URL`，无法构造 ragas LLM")

    def set_run_config(self, run_config: Any) -> None:
        timeout = getattr(run_config, "timeout", None)
        if isinstance(timeout, (int, float)) and timeout > 0:
            self._timeout = int(timeout)

    def get_temperature(self, n: int) -> float:
        if n > 1:
            return max(self._default_temperature, 0.3)
        return self._default_temperature

    async def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        del stop, callbacks
        resolved_temperature = self.get_temperature(n) if temperature is None else float(temperature)
        payload = self._normalize_prompt(prompt)
        tasks = [
            self._ainvoke_once(payload=payload, temperature=resolved_temperature)
            for _ in range(max(int(n), 1))
        ]
        responses = await asyncio.gather(*tasks)
        return self._to_llm_result(responses)

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        del stop, callbacks
        resolved_temperature = self.get_temperature(n) if temperature is None else float(temperature)
        payload = self._normalize_prompt(prompt)
        responses = [
            self._invoke_once(payload=payload, temperature=resolved_temperature)
            for _ in range(max(int(n), 1))
        ]
        return self._to_llm_result(responses)

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return await self.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return bool(response.generations)

    def _build_client(self, *, temperature: float) -> Any:
        """参数为评分温度；构建使用实际发送准入和当前HTTP重试政策的项目客户端。"""
        from app.services.llm_service import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            model=self.model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature,
            timeout=self._timeout,
            min_request_interval_seconds=self._min_request_interval_seconds,
            **self._transport_options(),
        )

    def _transport_options(self) -> dict[str, Any]:
        """无参数；返回产品共用的传输重试配置，评分不额外叠加逻辑等待。"""
        settings = _load_project_settings()
        return {"request_exception_max_retries": settings.llm_http_max_retries,
                "retry_backoff_seconds": settings.llm_retry_backoff_seconds}

    async def _ainvoke_once(self, *, payload: Any, temperature: float) -> str:
        """参数为评分提示及温度；仅通过真实HTTP入口异步发送并返回文本。"""
        client = self._build_client(temperature=temperature)
        response = await client.ainvoke(payload, use_cache=False)
        return response.content

    def _invoke_once(self, *, payload: Any, temperature: float) -> str:
        """参数为评分提示及温度；仅通过真实HTTP入口同步发送并返回文本。"""
        client = self._build_client(temperature=temperature)
        response = client.invoke(payload, use_cache=False)
        return response.content

    def _normalize_prompt(self, prompt: Any) -> Any:
        if hasattr(prompt, "to_messages"):
            try:
                return prompt.to_messages()
            except Exception:
                pass
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    def _to_llm_result(self, responses: list[str]) -> LLMResult:
        generations = [
            Generation(
                text=response,
                generation_info={"finish_reason": "stop"},
            )
            for response in responses
        ]
        return LLMResult(
            generations=[generations],
            llm_output={
                "provider": "studyassistant-project-llm",
                "model": self.model,
                "generation_count": len(generations),
            },
        )


def run_evaluation(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    adapter: ReplayRAGAdapter | LiveRAGAdapter | None = None,
    evaluator: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    dataset_name: str | None = None,
    dataset_tags: list[str] | None = None,
    baseline_report_path: str | Path | None = None,
    gate_mode: str = "smoke",
    regression_tolerance: float = 0.03,
) -> dict[str, Any]:
    """执行一次离线评估，并把结果写入 experiments 目录。"""
    adapter = adapter or ReplayRAGAdapter()
    records = [freeze_record(record, mode=describe_adapter(adapter).get("adapter", "replay")) for record in adapter.load(dataset_path)]
    evaluator = evaluator or default_ragas_evaluator
    result = evaluator([project_legacy_record(record) for record in records])
    normalized = _normalize_evaluation_result(records, result)
    if result.get("evaluation_status") != "missing":
        _assert_evaluation_has_finite_metrics(normalized)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _utc_now().replace(":", "").replace(".", "")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_info = describe_adapter(adapter)
    normalized_gate_mode = _normalize_gate_mode(gate_mode)

    report = {
        "run_id": run_id,
        "dataset_path": str(Path(dataset_path)),
        "dataset_name": str(dataset_name or Path(dataset_path).stem),
        "dataset_tags": _normalize_dataset_tags(dataset_tags),
        "sample_count": len(records),
        "metrics": normalized["metrics"],
        "project_metrics": normalized["project_metrics"],
        "quality_metrics": normalized["metrics"],
        "contract_metrics": _select_metric_group(normalized["project_metrics"], CONTRACT_METRIC_NAMES),
        "debug_metrics": _select_metric_group(normalized["project_metrics"], DEBUG_METRIC_NAMES),
        "records": normalized["records"],
        "coverage_summary": _build_coverage_summary(normalized["records"]),
        "created_at": _utc_now(),
        "contract_version": CONTRACT_VERSION,
        "measurement_coverage": measurement_coverage(normalized["records"]),
    }
    report.update(adapter_info)
    if any("evidence_benchmark" in read_snapshot(record["eval_case"]["truth_snapshot"]) for record in records):
        from evals.evidence.report import build_evidence_report

        # 1. 【证据评测】【独立报告】仅消费冻结观察，证据分数不混入 RAGAS 门禁
        report["evidence_benchmark"] = build_evidence_report(records)
    if baseline_report_path:
        report["baseline_comparison"] = _build_baseline_comparison(
            report,
            baseline_report_path=baseline_report_path,
        )
    report["gate"] = _build_gate_result(
        report,
        gate_mode=normalized_gate_mode,
        regression_tolerance=regression_tolerance,
    )
    report["gate"]["checks"].extend(integrity_checks(normalized["records"]))
    report["gate"]["passed"] = all(check["passed"] for check in report["gate"]["checks"])
    json_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_markdown_report(report), encoding="utf-8")
    return {
        **report,
        "output_dir": str(run_dir),
        "json_report_path": str(json_path),
        "markdown_report_path": str(markdown_path),
    }


def run_evaluation_suite(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    baseline_report_path: str | Path | None = None,
    gate_mode: str = "smoke",
    regression_tolerance: float = 0.03,
) -> dict[str, Any]:
    """按 manifest 执行一组数据集，并生成 suite 级摘要。"""
    manifest = _load_dataset_manifest(manifest_path)
    normalized_gate_mode = _normalize_gate_mode(gate_mode)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"suite-{_utc_now().replace(':', '').replace('.', '')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets_output_dir = run_dir / "datasets"
    datasets_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_results: list[dict[str, Any]] = []
    for dataset_entry in manifest["datasets"]:
        adapter = build_adapter(
            adapter_name=str(dataset_entry.get("adapter") or manifest.get("default_adapter") or "live"),
            mode=str(dataset_entry.get("mode") or manifest.get("default_mode") or "auto"),
        )
        dataset_result = run_evaluation(
            dataset_path=dataset_entry["path"],
            output_dir=datasets_output_dir,
            adapter=adapter,
            dataset_name=str(dataset_entry.get("name") or Path(dataset_entry["path"]).stem),
            dataset_tags=list(dataset_entry.get("tags") or []),
            gate_mode="smoke",
            regression_tolerance=regression_tolerance,
        )
        dataset_results.append(_summarize_suite_dataset_result(dataset_result))

    report = {
        "run_id": run_id,
        "suite_name": str(manifest.get("suite_name") or Path(manifest["manifest_path"]).stem),
        "manifest_path": str(manifest["manifest_path"]),
        "description": str(manifest.get("description") or "").strip(),
        "dataset_count": len(dataset_results),
        "sample_count": sum(int(item.get("sample_count") or 0) for item in dataset_results),
        "metrics": _mean_metric_maps(dataset_results, field="metrics", metric_names=METRIC_NAMES),
        "project_metrics": _mean_metric_maps(dataset_results, field="project_metrics", metric_names=PROJECT_METRIC_NAMES),
        "coverage_summary": _aggregate_suite_coverage(dataset_results),
        "datasets": dataset_results,
        "contract_version": CONTRACT_VERSION,
        "measurement_coverage": merge_measurement_coverage([row.get("measurement_coverage", {}) for row in dataset_results]),
        "created_at": _utc_now(),
    }
    report["quality_metrics"] = report["metrics"]
    report["contract_metrics"] = _select_metric_group(report["project_metrics"], CONTRACT_METRIC_NAMES)
    report["debug_metrics"] = _select_metric_group(report["project_metrics"], DEBUG_METRIC_NAMES)
    if baseline_report_path:
        report["baseline_comparison"] = _build_suite_baseline_comparison(
            report,
            baseline_report_path=baseline_report_path,
        )
    report["gate"] = _build_suite_gate_result(
        report,
        gate_mode=normalized_gate_mode,
        regression_tolerance=regression_tolerance,
    )

    json_path = run_dir / "suite-report.json"
    markdown_path = run_dir / "suite-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_suite_markdown_report(report), encoding="utf-8")
    return {
        **report,
        "output_dir": str(run_dir),
        "json_report_path": str(json_path),
        "markdown_report_path": str(markdown_path),
    }


def default_ragas_evaluator(records: list[dict[str, Any]]) -> dict[str, Any]:
    """默认 Ragas evaluator。"""
    indices = [index for index, record in enumerate(records) if ragas_eligible(record)]
    if not indices:
        return {"evaluation_status": "missing", "records": [{} for _ in records]}
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError("未安装 ragas，无法执行离线评估") from exc
    settings = _load_project_settings()
    dataset = build_ragas_dataset([records[index] for index in indices])
    llm = _build_project_ragas_llm(settings=settings)
    embeddings = _build_project_ragas_embeddings(settings=settings)
    run_config = _build_ragas_run_config(settings=settings)
    llm.set_run_config(run_config)
    try:
        result = evaluate(
            dataset,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=True,
        )
    except Exception as exc:
        if _is_upstream_provider_error(exc):
            raise RuntimeError(
                "Ragas 已显式注入 StudyAssistant 项目现有的 LLM 与 embedding。"
                " 如果这里仍然失败，说明问题已经不在 ragas 默认 OpenAI 栈，而在项目当前 provider/model 本身："
                " 请检查 `LLM_MODEL_NAME/LLM_BASE_URL/LLM_API_KEY` 是否可用于评估，"
                " 以及上游是否拦截了评估提示词或禁止该模型执行此类打分请求。"
                f" 原始错误: {exc}"
            ) from exc
        raise
    if len(indices) == len(records):
        return {"raw_result": result}
    return {"raw_result": result, "record_indices": indices}


def _build_project_ragas_llm(settings: Any | None = None) -> _ProjectRagasLLM:
    settings = settings or _load_project_settings()
    return _ProjectRagasLLM(
        model=getattr(settings, "llm_model_name", ""),
        api_key=getattr(settings, "llm_api_key", ""),
        base_url=getattr(settings, "llm_base_url", ""),
        temperature=float(getattr(settings, "llm_temperature", 0.0) or 0.0),
        timeout=int(getattr(settings, "llm_timeout", 240) or 240),
        min_request_interval_seconds=float(
            getattr(settings, "ragas_llm_min_request_interval_seconds", 0.0) or 0.0
        ),
    )


def _build_project_ragas_embeddings(settings: Any | None = None) -> _ProjectLangchainEmbeddings:
    settings = settings or _load_project_settings()
    from app.services.embedding_client import get_embedding_client

    return _ProjectLangchainEmbeddings(
        client=get_embedding_client(),
        model_name=str(getattr(settings, "embedding_model", "") or ""),
    )


def _build_ragas_run_config(settings: Any | None = None) -> Any:
    settings = settings or _load_project_settings()
    from ragas.run_config import RunConfig

    timeout = int(getattr(settings, "llm_timeout", 240) or 240)
    max_workers = int(getattr(settings, "ragas_max_workers", 1) or 1)
    try:
        return RunConfig(timeout=timeout, max_workers=max_workers)
    except TypeError:
        return RunConfig(timeout=timeout)


def _bootstrap_openai_compat_env(settings: Any | None = None) -> dict[str, bool]:
    """
    为 ragas 默认 OpenAI evaluator 预热环境变量。

    优先使用用户显式设置的 `OPENAI_*`，若缺失则回退到项目现有的
    `LLM_API_KEY/LLM_BASE_URL`，这样在 backend 目录里可直接复用当前服务配置。
    """
    settings = settings or _load_project_settings()
    used_project_key = False
    used_project_base_url = False

    if not str(os.environ.get("OPENAI_API_KEY") or "").strip():
        project_key = str(getattr(settings, "llm_api_key", "") or "").strip()
        if project_key:
            os.environ["OPENAI_API_KEY"] = project_key
            used_project_key = True

    if not str(os.environ.get("OPENAI_BASE_URL") or "").strip():
        project_base_url = str(getattr(settings, "llm_base_url", "") or "").strip()
        if project_base_url:
            os.environ["OPENAI_BASE_URL"] = project_base_url
            used_project_base_url = True

    if not str(os.environ.get("OPENAI_API_KEY") or "").strip():
        raise RuntimeError(
            "未设置 `OPENAI_API_KEY`，且项目配置里也没有可复用的 `LLM_API_KEY`。"
            " 当前 `python -m evals.ragas.evals ...` 默认走 ragas 的 OpenAI-compatible evaluator。"
            " 请先设置 `OPENAI_API_KEY`；如果你使用 OpenAI-compatible 代理，也请一并设置 `OPENAI_BASE_URL`。"
        )

    return {
        "used_project_key": used_project_key,
        "used_project_base_url": used_project_base_url,
    }


def _load_project_settings() -> Any:
    from app.config import get_settings

    return get_settings()


def _is_openai_error(exc: Exception) -> bool:
    try:
        from openai import OpenAIError
    except Exception:
        return False
    return isinstance(exc, OpenAIError)


def _is_upstream_provider_error(exc: Exception) -> bool:
    if _is_openai_error(exc):
        return True
    message = str(exc).lower()
    markers = (
        "your request was blocked",
        "blocked",
        "unauthorized",
        "forbidden",
        "401",
        "403",
        "model not found",
        "permission denied",
        "authentication failed",
        "api key",
    )
    return any(marker in message for marker in markers)


def _normalize_evaluation_result(
    records: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    raw_result = result.get("raw_result", result)
    record_scores = result.get("records") or []
    aggregate_metrics = result.get("metrics") or {}

    if not record_scores and hasattr(raw_result, "to_pandas"):
        dataframe = raw_result.to_pandas()
        record_scores = dataframe.to_dict(orient="records")

    if not aggregate_metrics and isinstance(raw_result, dict):
        aggregate_metrics = {
            key: _coerce_metric_value(value)
            for key, value in raw_result.items()
            if isinstance(value, (int, float))
        }
        aggregate_metrics = {
            key: value
            for key, value in aggregate_metrics.items()
            if value is not None
        }

    if not aggregate_metrics and record_scores:
        aggregate_metrics = _mean_metrics(record_scores)

    if "record_indices" in result:
        indexed_scores = dict(zip(result["record_indices"], record_scores))
        record_scores = [indexed_scores.get(index, {}) for index in range(len(records))]

    merged_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        score_row = record_scores[index] if index < len(record_scores) else {}
        frozen = freeze_record(record)
        projected = project_legacy_record(frozen)
        project_scores = _compute_project_metric_row(projected, keep_missing=True)
        measurements = build_measurements(projected, project_scores, score_row)
        numeric = {item.name: round(item.value, 6) for item in measurements if item.value is not None}
        original = {key: value for key, value in frozen.items() if key not in (*METRIC_NAMES, *PROJECT_METRIC_NAMES)}
        merged_records.append({**original, **projected, **numeric,
                               "grading_result": build_grading(frozen, measurements, grader="ragas-and-contracts")})

    enriched_records, project_metrics = merged_records, _mean_custom_metrics(merged_records)

    return {
        "metrics": aggregate_metrics,
        "project_metrics": project_metrics,
        "records": enriched_records,
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [
            coerced
            for row in rows
            if (coerced := _coerce_metric_value(row.get(name))) is not None
        ]
        if values:
            metrics[name] = round(statistics.fmean(values), 6)
    return metrics


def _coerce_metric_value(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _normalize_gate_mode(gate_mode: str) -> str:
    normalized = str(gate_mode or "smoke").strip().lower() or "smoke"
    if normalized not in GATE_MODE_CHOICES:
        raise ValueError(f"不支持的 gate_mode: {gate_mode}")
    return normalized


def _normalize_dataset_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    normalized: list[str] = []
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _build_coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    requested_mode_counts: Counter[str] = Counter()
    resolved_mode_counts: Counter[str] = Counter()
    graph_mode_counts: Counter[str] = Counter()
    route_metadata_complete = 0
    scope_constraints = 0
    project_scope_constraints = 0
    filter_scope_constraints = 0
    paper_scope_constraints = 0
    scope_metadata_complete = 0
    citations_field = 0
    citations_count_field = 0
    chunk_citations_count_field = 0
    structured_citations = 0
    non_placeholder_inline_citation = 0
    citation_metadata_complete = 0
    answer_confidence_field = 0
    answer_confidence_value = 0
    source_status_field = 0
    retrieval_reasoning_field = 0
    payload_metadata_complete = 0
    dialogue_mode_field = 0
    synthesis_state_field = 0
    synthesis_status_field = 0
    evidence_layers_field = 0
    supported_dimensions_field = 0
    open_questions_field = 0
    research_metadata_complete = 0

    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        requested_mode = str(metadata.get("requested_mode") or "").strip().lower()
        resolved_mode = str(metadata.get("resolved_mode") or metadata.get("retrieval_mode") or "").strip().lower()
        graph_mode = str(metadata.get("graph_mode") or "").strip().lower()
        if requested_mode:
            requested_mode_counts[requested_mode] += 1
        if resolved_mode:
            resolved_mode_counts[resolved_mode] += 1
        if graph_mode:
            graph_mode_counts[graph_mode] += 1

        if _has_route_metadata(metadata):
            route_metadata_complete += 1

        has_scope_constraint = _has_scope_constraint(metadata)
        if has_scope_constraint:
            scope_constraints += 1
        if str(metadata.get("project_id") or "").strip():
            project_scope_constraints += 1
        if str(metadata.get("filter_expr") or "").strip():
            filter_scope_constraints += 1
        if _get_allowed_scope_paper_ids(metadata):
            paper_scope_constraints += 1
        if _has_scope_metadata(metadata):
            scope_metadata_complete += 1

        citations = metadata.get("citations") if isinstance(metadata.get("citations"), list) else []
        if "citations" in metadata and isinstance(metadata.get("citations"), list):
            citations_field += 1
        if "citations_count" in metadata:
            citations_count_field += 1
        if "chunk_citations_count" in metadata:
            chunk_citations_count_field += 1
        if citations:
            structured_citations += 1
        citation_presence = _compute_citation_presence_rate(record, metadata)
        if citation_presence is not None and citation_presence > 0:
            non_placeholder_inline_citation += 1
        if _has_citation_metadata(metadata):
            citation_metadata_complete += 1

        if "answer_confidence" in metadata:
            answer_confidence_field += 1
            if str(metadata.get("answer_confidence") or "").strip():
                answer_confidence_value += 1
        if "source_status" in metadata and isinstance(metadata.get("source_status"), dict):
            source_status_field += 1
        if "retrieval_reasoning" in metadata:
            retrieval_reasoning_field += 1
        if _has_payload_metadata(record, metadata):
            payload_metadata_complete += 1
        if _normalize_dialogue_mode(metadata.get("dialogue_mode") or metadata.get("requested_dialogue_mode")):
            dialogue_mode_field += 1
        if isinstance(metadata.get("synthesis_state"), dict):
            synthesis_state_field += 1
        if str(metadata.get("synthesis_status") or "").strip():
            synthesis_status_field += 1
        if _has_evidence_layers_payload(metadata.get("evidence_layers")):
            evidence_layers_field += 1
        if isinstance(metadata.get("supported_dimensions"), list):
            supported_dimensions_field += 1
        if isinstance(metadata.get("open_questions"), list):
            open_questions_field += 1
        if _has_research_metadata(metadata):
            research_metadata_complete += 1

    return {
        "sample_count": sample_count,
        "route": {
            "requested_mode_counts": _counter_to_dict(requested_mode_counts),
            "resolved_mode_counts": _counter_to_dict(resolved_mode_counts),
            "graph_mode_counts": _counter_to_dict(graph_mode_counts),
            "route_metadata_complete": _count_summary(route_metadata_complete, sample_count),
        },
        "scope": {
            "with_scope_constraint": _count_summary(scope_constraints, sample_count),
            "with_project_scope": _count_summary(project_scope_constraints, sample_count),
            "with_filter_scope": _count_summary(filter_scope_constraints, sample_count),
            "with_paper_scope": _count_summary(paper_scope_constraints, sample_count),
            "scope_metadata_complete": _count_summary(scope_metadata_complete, sample_count),
        },
        "citation": {
            "with_citations_field": _count_summary(citations_field, sample_count),
            "with_citations_count_field": _count_summary(citations_count_field, sample_count),
            "with_chunk_citations_count_field": _count_summary(chunk_citations_count_field, sample_count),
            "with_structured_citations": _count_summary(structured_citations, sample_count),
            "with_non_placeholder_citation_signal": _count_summary(non_placeholder_inline_citation, sample_count),
            "citation_metadata_complete": _count_summary(citation_metadata_complete, sample_count),
        },
        "payload": {
            "with_answer_confidence_field": _count_summary(answer_confidence_field, sample_count),
            "with_answer_confidence_value": _count_summary(answer_confidence_value, sample_count),
            "with_source_status_field": _count_summary(source_status_field, sample_count),
            "with_retrieval_reasoning_field": _count_summary(retrieval_reasoning_field, sample_count),
            "payload_metadata_complete": _count_summary(payload_metadata_complete, sample_count),
        },
        "research": {
            "with_dialogue_mode_field": _count_summary(dialogue_mode_field, sample_count),
            "with_synthesis_state_field": _count_summary(synthesis_state_field, sample_count),
            "with_synthesis_status_field": _count_summary(synthesis_status_field, sample_count),
            "with_evidence_layers_field": _count_summary(evidence_layers_field, sample_count),
            "with_supported_dimensions_field": _count_summary(supported_dimensions_field, sample_count),
            "with_open_questions_field": _count_summary(open_questions_field, sample_count),
            "research_metadata_complete": _count_summary(research_metadata_complete, sample_count),
        },
    }


def _count_summary(count: int, total: int) -> dict[str, Any]:
    rate = round((count / total), 6) if total else 0.0
    return {"count": count, "rate": rate}


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _coverage_count(summary: dict[str, Any], section: str, key: str) -> int:
    payload = (summary.get(section) or {}).get(key)
    if isinstance(payload, dict):
        return int(payload.get("count") or 0)
    return 0


def _has_route_metadata(metadata: dict[str, Any]) -> bool:
    return bool(str(metadata.get("requested_mode") or "").strip()) and bool(
        str(metadata.get("resolved_mode") or metadata.get("retrieval_mode") or "").strip()
    ) and bool(str(metadata.get("graph_mode") or "").strip()) and isinstance(metadata.get("source_status"), dict)


def _has_scope_metadata(metadata: dict[str, Any]) -> bool:
    if not _has_scope_constraint(metadata):
        return True
    return bool(_get_allowed_scope_paper_ids(metadata)) or bool(str(metadata.get("project_id") or "").strip()) or bool(
        str(metadata.get("filter_expr") or "").strip()
    ) or isinstance(metadata.get("resolved_scope"), dict)


def _has_citation_metadata(metadata: dict[str, Any]) -> bool:
    return (
        "citations" in metadata
        and isinstance(metadata.get("citations"), list)
        and "citations_count" in metadata
        and "chunk_citations_count" in metadata
    )


def _has_evidence_layers_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return False


def _has_payload_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return bool(str(record.get("response") or "").strip()) and isinstance(record.get("retrieved_contexts"), list) and bool(
        record.get("retrieved_contexts")
    ) and "answer_confidence" in metadata


def _has_research_metadata(metadata: dict[str, Any]) -> bool:
    synthesis_state = metadata.get("synthesis_state")
    return (
        bool(_normalize_dialogue_mode(metadata.get("dialogue_mode") or metadata.get("requested_dialogue_mode")))
        and isinstance(synthesis_state, dict)
        and bool(str(metadata.get("synthesis_status") or synthesis_state.get("status") or "").strip())
        and _has_evidence_layers_payload(metadata.get("evidence_layers") or synthesis_state.get("evidence_layers"))
    )


def _load_json_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"baseline report 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_baseline_comparison(report: dict[str, Any], *, baseline_report_path: str | Path) -> dict[str, Any]:
    baseline = _load_json_report(baseline_report_path)
    return {
        "baseline_report_path": str(Path(baseline_report_path)),
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "metrics": _compare_metric_groups(
            report.get("metrics") or {},
            baseline.get("metrics") or {},
            directions=METRIC_DIRECTIONS,
        ),
        "project_metrics": _compare_metric_groups(
            report.get("project_metrics") or {},
            baseline.get("project_metrics") or {},
            directions=PROJECT_METRIC_DIRECTIONS,
        ),
    }


def _compare_metric_groups(
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    directions: dict[str, str],
) -> dict[str, dict[str, Any]]:
    compared: dict[str, dict[str, Any]] = {}
    for name, direction in directions.items():
        current_value = _coerce_metric_value(current_metrics.get(name))
        baseline_value = _coerce_metric_value(baseline_metrics.get(name))
        if current_value is None or baseline_value is None:
            continue
        compared[name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": round(current_value - baseline_value, 6),
            "direction": direction,
        }
    return compared


def _build_gate_result(
    report: dict[str, Any],
    *,
    gate_mode: str,
    regression_tolerance: float,
) -> dict[str, Any]:
    sample_count = int(report.get("sample_count") or 0)
    coverage = report.get("coverage_summary") if isinstance(report.get("coverage_summary"), dict) else {}
    checks = [
        {
            "name": "finite_metrics_available",
            "passed": _report_has_finite_metrics(report),
            "details": "至少有一个聚合指标或样本级指标是有限值。",
        }
    ]

    if str(report.get("adapter") or "").strip().lower() == "live":
        checks.extend(
            [
                _build_count_gate_check(
                    "route_metadata_complete",
                    _coverage_count(coverage, "route", "route_metadata_complete"),
                    sample_count,
                ),
                _build_count_gate_check(
                    "scope_metadata_complete",
                    _coverage_count(coverage, "scope", "scope_metadata_complete"),
                    sample_count,
                ),
                _build_count_gate_check(
                    "citation_metadata_complete",
                    _coverage_count(coverage, "citation", "citation_metadata_complete"),
                    sample_count,
                ),
                _build_count_gate_check(
                    "payload_metadata_complete",
                    _coverage_count(coverage, "payload", "payload_metadata_complete"),
                    sample_count,
                ),
            ]
        )

    warnings: list[str] = []
    baseline_comparison = report.get("baseline_comparison") if isinstance(report.get("baseline_comparison"), dict) else None
    if gate_mode == "regression":
        if baseline_comparison is None:
            checks.append(
                {
                    "name": "baseline_report_available",
                    "passed": False,
                    "details": "regression gate 需要 baseline_report。",
                }
            )
        else:
            regressions = _collect_regressions(
                baseline_comparison,
                regression_tolerance=regression_tolerance,
            )
            checks.append(
                {
                    "name": "baseline_regression_within_tolerance",
                    "passed": not regressions,
                    "details": regressions or [f"所有指标变化都在容忍阈值 {regression_tolerance:.4f} 内。"],
                }
            )
    elif baseline_comparison is None:
        warnings.append("smoke gate 未做 baseline 对比，当前验证可运行性、字段完整性和 smoke 质量/契约底线。")
    if gate_mode == "smoke":
        checks.extend(_build_smoke_quality_checks(report))
        warnings.extend(_build_smoke_metadata_warnings(report))

    return {
        "mode": gate_mode,
        "passed": all(bool(item.get("passed")) for item in checks),
        "regression_tolerance": regression_tolerance,
        "checks": checks,
        "warnings": warnings,
    }


def _build_smoke_quality_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    project_metrics = report.get("project_metrics") if isinstance(report.get("project_metrics"), dict) else {}
    route_score = _coerce_metric_value(project_metrics.get("route_correctness"))
    placeholder_rate = _coerce_metric_value(project_metrics.get("placeholder_citation_rate"))
    scope_score = _coerce_metric_value(project_metrics.get("scope_adherence"))
    fact_contract_score = _coerce_metric_value(project_metrics.get("fact_contract_score"))
    return [
        {
            "name": "smoke_ragas_metrics_not_collapsed",
            "passed": not _ragas_metrics_collapsed(metrics),
            "details": f"RAGAS metrics must not all be at or below {SMOKE_QUALITY_FLOOR:.4f}.",
        },
        {
            "name": "smoke_route_signal_not_collapsed",
            "passed": route_score is None or route_score > 0.0,
            "details": _smoke_route_quality_details(route_score),
        },
        {
            "name": "smoke_placeholder_citations_absent",
            "passed": placeholder_rate is None or placeholder_rate <= SMOKE_PLACEHOLDER_CITATION_CEILING,
            "details": _smoke_placeholder_citation_details(placeholder_rate),
        },
        {
            "name": "smoke_scope_contract_intact",
            "passed": scope_score is None or scope_score >= SMOKE_SCOPE_ADHERENCE_FLOOR,
            "details": _smoke_scope_contract_details(scope_score),
        },
        {
            "name": "smoke_fact_contract_intact",
            "passed": fact_contract_score is None or fact_contract_score >= SMOKE_FACT_CONTRACT_FLOOR,
            "details": _smoke_fact_contract_details(fact_contract_score),
        },
    ]


def _smoke_route_quality_details(route_score: float | None) -> str:
    if route_score is None:
        return "route_correctness unavailable; skipped for this adapter/report."
    return f"route_correctness={route_score:.6f}; value must be greater than 0.0 when present."


def _smoke_placeholder_citation_details(placeholder_rate: float | None) -> str:
    if placeholder_rate is None:
        return "placeholder_citation_rate unavailable; skipped for this adapter/report."
    return (
        f"placeholder_citation_rate={placeholder_rate:.6f}; "
        f"value must be at or below {SMOKE_PLACEHOLDER_CITATION_CEILING:.6f} when present."
    )


def _smoke_scope_contract_details(scope_score: float | None) -> str:
    if scope_score is None:
        return "scope_adherence unavailable or not applicable; skipped for this adapter/report."
    return (
        f"scope_adherence={scope_score:.6f}; "
        f"value must be at least {SMOKE_SCOPE_ADHERENCE_FLOOR:.6f} when present."
    )


def _smoke_fact_contract_details(fact_contract_score: float | None) -> str:
    if fact_contract_score is None:
        return "fact_contract_score unavailable or not applicable; skipped for this adapter/report."
    return (
        f"fact_contract_score={fact_contract_score:.6f}; "
        f"value must be at least {SMOKE_FACT_CONTRACT_FLOOR:.6f} when present."
    )


def _build_smoke_metadata_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    coverage = report.get("coverage_summary") if isinstance(report.get("coverage_summary"), dict) else {}
    unknown_count = _resolved_mode_count(coverage, "unknown")
    if unknown_count > 0:
        warnings.append(f"resolved_mode contains unknown for {unknown_count} sample(s); inspect route metadata.")
    return warnings


def _ragas_metrics_collapsed(metrics: dict[str, Any]) -> bool:
    values = [_coerce_metric_value(metrics.get(name)) for name in METRIC_NAMES if name in metrics]
    finite_values = [value for value in values if value is not None]
    return bool(finite_values) and all(value <= SMOKE_QUALITY_FLOOR for value in finite_values)


def _resolved_mode_count(coverage: dict[str, Any], mode: str) -> int:
    route = coverage.get("route") if isinstance(coverage.get("route"), dict) else {}
    counts = route.get("resolved_mode_counts") if isinstance(route.get("resolved_mode_counts"), dict) else {}
    return int(counts.get(mode) or 0)


def _report_has_finite_metrics(report: dict[str, Any]) -> bool:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if any(_coerce_metric_value(value) is not None for value in metrics.values()):
        return True
    for record in report.get("records") or []:
        if not isinstance(record, dict):
            continue
        if any(_coerce_metric_value(record.get(name)) is not None for name in METRIC_NAMES):
            return True
    return False


def _build_count_gate_check(name: str, actual: int, expected: int) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "details": {"actual": actual, "expected": expected},
    }


def _collect_regressions(
    baseline_comparison: dict[str, Any],
    *,
    regression_tolerance: float,
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for group_name in ("metrics", "project_metrics"):
        group = baseline_comparison.get(group_name)
        if not isinstance(group, dict):
            continue
        regressions.extend(
            _collect_metric_group_regressions(
                group_name,
                group,
                regression_tolerance=regression_tolerance,
            )
        )
    return regressions


def _collect_metric_group_regressions(
    group_name: str,
    group: dict[str, Any],
    *,
    regression_tolerance: float,
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for metric_name, payload in group.items():
        if not isinstance(payload, dict):
            continue
        direction = str(payload.get("direction") or "higher").strip().lower() or "higher"
        delta = _coerce_metric_value(payload.get("delta"))
        if delta is None:
            continue
        if direction == "higher" and delta < -regression_tolerance:
            regressions.append(
                {
                    "group": group_name,
                    "metric": metric_name,
                    "delta": delta,
                    "direction": direction,
                }
            )
        if direction == "lower" and delta > regression_tolerance:
            regressions.append(
                {
                    "group": group_name,
                    "metric": metric_name,
                    "delta": delta,
                    "direction": direction,
                }
            )
    return regressions


def _load_dataset_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"manifest 缺少 datasets: {path}")
    normalized_datasets: list[dict[str, Any]] = []
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError(f"manifest dataset 条目必须是对象: {path}")
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            raise ValueError(f"manifest dataset 条目缺少 path: {path}")
        dataset_path = Path(raw_path)
        if not dataset_path.is_absolute():
            dataset_path = (path.parent / dataset_path).resolve()
        normalized_datasets.append(
            {
                **item,
                "path": str(dataset_path),
                "name": str(item.get("name") or dataset_path.stem),
                "tags": _normalize_dataset_tags(list(item.get("tags") or [])),
            }
        )
    return {
        **payload,
        "manifest_path": str(path),
        "datasets": normalized_datasets,
    }


def _summarize_suite_dataset_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": str(result.get("dataset_name") or Path(str(result.get("dataset_path") or "")).stem),
        "dataset_path": str(result.get("dataset_path") or ""),
        "dataset_tags": _normalize_dataset_tags(result.get("dataset_tags") if isinstance(result.get("dataset_tags"), list) else []),
        "adapter": str(result.get("adapter") or ""),
        "mode": str(result.get("mode") or ""),
        "graph_mode": str(result.get("graph_mode") or ""),
        "sample_count": int(result.get("sample_count") or 0),
        "metrics": dict(result.get("metrics") or {}),
        "project_metrics": dict(result.get("project_metrics") or {}),
        "coverage_summary": dict(result.get("coverage_summary") or {}),
        "measurement_coverage": dict(result.get("measurement_coverage") or {}),
        "gate": dict(result.get("gate") or {}),
        "json_report_path": str(result.get("json_report_path") or ""),
        "markdown_report_path": str(result.get("markdown_report_path") or ""),
        "output_dir": str(result.get("output_dir") or ""),
    }


def _mean_metric_maps(
    rows: list[dict[str, Any]],
    *,
    field: str,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in metric_names:
        weighted = []
        for row in rows:
            value = _coerce_metric_value((row.get(field) or {}).get(name))
            if value is None:
                continue
            coverage = row.get("measurement_coverage", {}).get(name)
            count = int(coverage["measured_count"]) if coverage is not None else 1
            if count:
                weighted.append((value, count))
        if weighted:
            metrics[name] = round(sum(value * count for value, count in weighted) / sum(count for _, count in weighted), 6)
    return metrics


def _select_metric_group(metrics: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: value
        for name in names
        if (value := _coerce_metric_value(metrics.get(name))) is not None
    }


def _aggregate_suite_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    sample_count = sum(int(item.get("sample_count") or 0) for item in rows)
    route_requested_counts: Counter[str] = Counter()
    route_resolved_counts: Counter[str] = Counter()
    route_graph_counts: Counter[str] = Counter()
    aggregated = {
        "sample_count": sample_count,
        "route": {},
        "scope": {},
        "citation": {},
        "payload": {},
    }
    scalar_keys = {
        "route": ("route_metadata_complete",),
        "scope": ("with_scope_constraint", "with_project_scope", "with_filter_scope", "with_paper_scope", "scope_metadata_complete"),
        "citation": (
            "with_citations_field",
            "with_citations_count_field",
            "with_chunk_citations_count_field",
            "with_structured_citations",
            "with_non_placeholder_citation_signal",
            "citation_metadata_complete",
        ),
        "payload": (
            "with_answer_confidence_field",
            "with_answer_confidence_value",
            "with_source_status_field",
            "with_retrieval_reasoning_field",
            "payload_metadata_complete",
        ),
    }

    for row in rows:
        coverage = row.get("coverage_summary") if isinstance(row.get("coverage_summary"), dict) else {}
        route = coverage.get("route") if isinstance(coverage.get("route"), dict) else {}
        route_requested_counts.update(route.get("requested_mode_counts") or {})
        route_resolved_counts.update(route.get("resolved_mode_counts") or {})
        route_graph_counts.update(route.get("graph_mode_counts") or {})
        for section, keys in scalar_keys.items():
            section_payload = coverage.get(section) if isinstance(coverage.get(section), dict) else {}
            target_section = aggregated[section]
            for key in keys:
                current_count = _coverage_count({section: section_payload}, section, key)
                target_section[key] = target_section.get(key, 0) + current_count

    aggregated["route"] = {
        "requested_mode_counts": _counter_to_dict(route_requested_counts),
        "resolved_mode_counts": _counter_to_dict(route_resolved_counts),
        "graph_mode_counts": _counter_to_dict(route_graph_counts),
        **{
            key: _count_summary(int(value), sample_count)
            for key, value in aggregated["route"].items()
        },
    }
    for section in ("scope", "citation", "payload"):
        aggregated[section] = {
            key: _count_summary(int(value), sample_count)
            for key, value in aggregated[section].items()
        }
    return aggregated


def _build_suite_baseline_comparison(report: dict[str, Any], *, baseline_report_path: str | Path) -> dict[str, Any]:
    baseline = _load_json_report(baseline_report_path)
    baseline_datasets = {
        str(item.get("dataset_name") or Path(str(item.get("dataset_path") or "")).stem): item
        for item in baseline.get("datasets") or []
        if isinstance(item, dict)
    }
    dataset_comparisons: dict[str, Any] = {}
    for item in report.get("datasets") or []:
        if not isinstance(item, dict):
            continue
        dataset_name = str(item.get("dataset_name") or "")
        baseline_item = baseline_datasets.get(dataset_name)
        if not baseline_item:
            continue
        dataset_comparisons[dataset_name] = {
            "metrics": _compare_metric_groups(
                item.get("metrics") or {},
                baseline_item.get("metrics") or {},
                directions=METRIC_DIRECTIONS,
            ),
            "project_metrics": _compare_metric_groups(
                item.get("project_metrics") or {},
                baseline_item.get("project_metrics") or {},
                directions=PROJECT_METRIC_DIRECTIONS,
            ),
        }
    return {
        "baseline_report_path": str(Path(baseline_report_path)),
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "metrics": _compare_metric_groups(
            report.get("metrics") or {},
            baseline.get("metrics") or {},
            directions=METRIC_DIRECTIONS,
        ),
        "project_metrics": _compare_metric_groups(
            report.get("project_metrics") or {},
            baseline.get("project_metrics") or {},
            directions=PROJECT_METRIC_DIRECTIONS,
        ),
        "datasets": dataset_comparisons,
    }


def _build_suite_gate_result(
    report: dict[str, Any],
    *,
    gate_mode: str,
    regression_tolerance: float,
) -> dict[str, Any]:
    datasets = [item for item in (report.get("datasets") or []) if isinstance(item, dict)]
    failed_dataset_gates = [str(item.get("dataset_name") or "") for item in datasets if not bool((item.get("gate") or {}).get("passed"))]
    checks: list[dict[str, Any]] = [
        {
            "name": "dataset_reports_generated",
            "passed": len(datasets) == int(report.get("dataset_count") or 0),
            "details": {"actual": len(datasets), "expected": int(report.get("dataset_count") or 0)},
        },
        {
            "name": "dataset_gates_passed",
            "passed": not failed_dataset_gates,
            "details": failed_dataset_gates or ["所有子数据集 gate 均通过。"],
        },
    ]

    warnings: list[str] = []
    warnings.extend(_collect_dataset_gate_warnings(datasets))
    baseline_comparison = report.get("baseline_comparison") if isinstance(report.get("baseline_comparison"), dict) else None
    if gate_mode == "regression":
        if baseline_comparison is None:
            checks.append(
                {
                    "name": "baseline_report_available",
                    "passed": False,
                    "details": "regression gate 需要 suite baseline_report。",
                }
            )
        else:
            regressions = _collect_regressions(
                baseline_comparison,
                regression_tolerance=regression_tolerance,
            )
            for dataset_name, dataset_payload in (baseline_comparison.get("datasets") or {}).items():
                if isinstance(dataset_payload, dict):
                    regressions.extend(
                        _collect_regressions(
                            dataset_payload,
                            regression_tolerance=regression_tolerance,
                        )
                    )
            checks.append(
                {
                    "name": "baseline_regression_within_tolerance",
                    "passed": not regressions,
                    "details": regressions or [f"所有 suite / dataset 指标变化都在容忍阈值 {regression_tolerance:.4f} 内。"],
                }
            )
    elif baseline_comparison is None:
        warnings.append("smoke gate 未做 suite baseline 对比，当前验证数据集可运行、字段摘要和 smoke 质量/契约底线。")
    if gate_mode == "smoke":
        checks.extend(_build_smoke_quality_checks(report))
        warnings.extend(_build_smoke_metadata_warnings(report))

    return {
        "mode": gate_mode,
        "passed": all(bool(item.get("passed")) for item in checks),
        "regression_tolerance": regression_tolerance,
        "checks": checks,
        "warnings": warnings,
    }


def _collect_dataset_gate_warnings(datasets: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for dataset in datasets:
        gate = dataset.get("gate") if isinstance(dataset.get("gate"), dict) else {}
        dataset_name = str(dataset.get("dataset_name") or "").strip() or "(unknown dataset)"
        for warning in gate.get("warnings") or []:
            warnings.append(f"{dataset_name}: {warning}")
    return warnings


def _mean_custom_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in PROJECT_METRIC_NAMES:
        values = [
            coerced
            for row in rows
            if (coerced := _coerce_metric_value(row.get(name))) is not None
        ]
        if values:
            metrics[name] = round(statistics.fmean(values), 6)
    return metrics


def _compute_project_metric_row(record: dict[str, Any], *, keep_missing: bool = False) -> dict[str, float | None]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    contract_scores = contract_measurements(record)
    scores = {
        "scope_adherence": _compute_scope_adherence(metadata),
        "citation_presence_rate": _compute_citation_presence_rate(record, metadata),
        "placeholder_citation_rate": _compute_placeholder_citation_rate(record),
        "route_correctness": _compute_route_correctness(metadata),
        **{name: metric.value for name, metric in contract_scores.items()},
        "answer_payload_present": _compute_answer_payload_present(record, metadata),
        "dialogue_mode_alignment": _compute_dialogue_mode_alignment(metadata),
        "synthesis_payload_present": _compute_synthesis_payload_present(metadata),
    }
    if keep_missing:
        return scores
    return {name: round(score, 6) for name, score in scores.items() if score is not None}


def _compute_scope_adherence(metadata: dict[str, Any]) -> float | None:
    if not _has_scope_constraint(metadata):
        return None

    allowed_paper_ids = _get_allowed_scope_paper_ids(metadata)
    inventory_paper_ids = _get_inventory_paper_ids(metadata)
    citations = metadata.get("citations") if isinstance(metadata.get("citations"), list) else []
    project_id = str(metadata.get("project_id") or "").strip()
    resolved_scope = metadata.get("resolved_scope") if isinstance(metadata.get("resolved_scope"), dict) else {}

    scores: list[float] = []
    scores.append(1.0 if metadata.get("filter_expr") or allowed_paper_ids or project_id else 0.0)

    if allowed_paper_ids:
        cited_paper_ids = {
            str(item.get("paper_id") or "").strip()
            for item in citations
            if isinstance(item, dict) and str(item.get("paper_id") or "").strip()
        }
        if cited_paper_ids:
            scores.append(1.0 if cited_paper_ids.issubset(allowed_paper_ids) else 0.0)
        if inventory_paper_ids:
            scores.append(1.0 if inventory_paper_ids.issubset(allowed_paper_ids) else 0.0)

    if project_id:
        scores.append(1.0 if str(resolved_scope.get("project_id") or "").strip() == project_id else 0.0)

    return statistics.fmean(scores) if scores else 0.0


def _compute_citation_presence_rate(record: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    policy = {**metadata, "rubric": record.get("rubric", metadata.get("rubric", {}))}
    if not _requires_citation_check(policy):
        return None
    citations_count = metadata.get("citations_count")
    if isinstance(citations_count, int):
        return 1.0 if citations_count > 0 else 0.0
    citations = metadata.get("citations") if isinstance(metadata.get("citations"), list) else []
    if citations:
        return 1.0
    tokens = _extract_inline_citation_tokens(str(record.get("response") or ""))
    non_placeholder_tokens = [token for token in tokens if not PLACEHOLDER_CITATION_PATTERN.match(token)]
    return 1.0 if non_placeholder_tokens else 0.0


def _requires_citation_check(metadata: dict[str, Any]) -> bool:
    for key in ("requires_citation", "citation_required"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
    rubric = metadata.get("rubric") if isinstance(metadata.get("rubric"), dict) else {}
    for key in ("requires_citation", "citation_required"):
        value = rubric.get(key)
        if isinstance(value, bool):
            return value
    retrieval_policy = str(metadata.get("retrieval_policy") or "").strip().lower()
    return retrieval_policy != "should_not_use"


def _compute_placeholder_citation_rate(record: dict[str, Any]) -> float:
    tokens = _extract_inline_citation_tokens(str(record.get("response") or ""))
    if not tokens:
        return 0.0
    placeholders = [token for token in tokens if PLACEHOLDER_CITATION_PATTERN.match(token)]
    return len(placeholders) / len(tokens)


def _compute_route_correctness(metadata: dict[str, Any]) -> float:
    requested_mode = str(metadata.get("requested_mode") or "").strip().lower()
    resolved_mode = str(metadata.get("resolved_mode") or metadata.get("retrieval_mode") or "").strip().lower()
    source_status = metadata.get("source_status") if isinstance(metadata.get("source_status"), dict) else {}
    scores: list[float] = []

    if resolved_mode:
        scores.append(1.0 if resolved_mode in KNOWN_ROUTE_RESOLVED_MODES else 0.0)
    if requested_mode in EXPLICIT_ROUTE_REQUEST_MODES:
        scores.append(1.0 if resolved_mode == requested_mode else 0.0)
    elif requested_mode and requested_mode != "auto":
        scores.append(0.0)

    source_score = _score_source_status(resolved_mode, source_status)
    if source_score is not None:
        scores.append(source_score)

    return statistics.fmean(scores) if scores else 0.0


def _compute_answer_payload_present(record: dict[str, Any], metadata: dict[str, Any]) -> float:
    scores = [
        1.0 if str(record.get("response") or "").strip() else 0.0,
        1.0 if isinstance(record.get("retrieved_contexts"), list) and bool(record.get("retrieved_contexts")) else 0.0,
    ]
    if str(metadata.get("adapter") or "").strip().lower() == "live":
        scores.extend(
            [
                1.0 if str(metadata.get("requested_mode") or "").strip() else 0.0,
                1.0 if str(metadata.get("resolved_mode") or metadata.get("retrieval_mode") or "").strip() else 0.0,
                1.0 if isinstance(metadata.get("source_status"), dict) else 0.0,
                1.0 if ("citations_count" in metadata or isinstance(metadata.get("citations"), list)) else 0.0,
                1.0 if str(metadata.get("answer_confidence") or "").strip() else 0.0,
            ]
        )
        if _has_scope_constraint(metadata):
            scores.append(
                1.0
                if metadata.get("filter_expr") or _get_allowed_scope_paper_ids(metadata) or str(metadata.get("project_id") or "").strip()
                else 0.0
            )
    return statistics.fmean(scores) if scores else 0.0


def _compute_dialogue_mode_alignment(metadata: dict[str, Any]) -> float:
    actual_mode = _normalize_dialogue_mode(metadata.get("dialogue_mode"))
    requested_mode = _normalize_dialogue_mode(metadata.get("requested_dialogue_mode"))
    synthesis_state = metadata.get("synthesis_state") if isinstance(metadata.get("synthesis_state"), dict) else {}
    synthesis_mode = _normalize_dialogue_mode(synthesis_state.get("dialogue_mode"))
    scores: list[float] = []

    if actual_mode:
        scores.append(1.0)
    if requested_mode:
        scores.append(1.0 if actual_mode == requested_mode else 0.0)
    if synthesis_mode:
        scores.append(1.0 if not actual_mode or synthesis_mode == actual_mode else 0.0)

    return statistics.fmean(scores) if scores else 0.0


def _compute_synthesis_payload_present(metadata: dict[str, Any]) -> float:
    synthesis_state = metadata.get("synthesis_state") if isinstance(metadata.get("synthesis_state"), dict) else {}
    if not synthesis_state:
        return 0.0
    synthesis_project_scope = synthesis_state.get("project_scope")
    synthesis_paper_focus = synthesis_state.get("paper_focus")
    synthesis_evidence_summary = synthesis_state.get("evidence_summary")
    synthesis_dimension_status = synthesis_state.get("dimension_status")
    synthesis_evidence_layers = synthesis_state.get("evidence_layers")
    synthesis_supported_dimensions = synthesis_state.get("supported_dimensions")
    synthesis_open_questions = synthesis_state.get("open_questions")

    scores = [
        1.0 if str(synthesis_state.get("status") or metadata.get("synthesis_status") or "").strip() else 0.0,
        1.0 if isinstance(synthesis_project_scope, dict) or isinstance(metadata.get("project_scope"), dict) else 0.0,
        1.0 if isinstance(synthesis_paper_focus, dict) or isinstance(metadata.get("paper_focus"), dict) else 0.0,
        1.0 if isinstance(synthesis_evidence_summary, dict) else 0.0,
        1.0 if isinstance(synthesis_dimension_status, dict) else 0.0,
        1.0
        if _has_evidence_layers_payload(synthesis_evidence_layers) or _has_evidence_layers_payload(metadata.get("evidence_layers"))
        else 0.0,
        1.0 if isinstance(synthesis_supported_dimensions, list) or isinstance(metadata.get("supported_dimensions"), list) else 0.0,
        1.0 if isinstance(synthesis_open_questions, list) or isinstance(metadata.get("open_questions"), list) else 0.0,
    ]
    citations_count = synthesis_state.get("citations_count", metadata.get("synthesis_citations_count"))
    if isinstance(citations_count, int):
        scores.append(1.0)
    return statistics.fmean(scores) if scores else 0.0


def _normalize_dialogue_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"chat", "lookup", "research"} else ""


def _score_source_status(resolved_mode: str, source_status: dict[str, Any]) -> float | None:
    if not resolved_mode or not source_status:
        return None
    milvus_status = str(source_status.get("milvus") or "").strip().lower()
    graph_status = str(source_status.get("graph") or "").strip().lower()
    inventory_status = str(source_status.get("inventory") or "").strip().lower()
    ok_statuses = {"ok", "empty", "skipped", "unused"}
    if resolved_mode == "inventory":
        return 1.0 if inventory_status in {"ok", "empty"} and milvus_status in ok_statuses and graph_status in ok_statuses else 0.0
    if resolved_mode == "milvus":
        return 1.0 if milvus_status == "ok" and graph_status in ok_statuses else 0.0
    if resolved_mode == "graph":
        return 1.0 if graph_status in {"ok", "empty"} and milvus_status in ok_statuses else 0.0
    if resolved_mode == "fusion":
        return 1.0 if milvus_status == "ok" and graph_status in {"ok", "empty"} else 0.0
    return None


def _has_scope_constraint(metadata: dict[str, Any]) -> bool:
    return bool(_get_allowed_scope_paper_ids(metadata)) or bool(str(metadata.get("project_id") or "").strip()) or bool(
        str(metadata.get("filter_expr") or "").strip()
    )


def _get_allowed_scope_paper_ids(metadata: dict[str, Any]) -> set[str]:
    candidates: list[str] = []
    paper_ids = metadata.get("paper_ids")
    if isinstance(paper_ids, list):
        candidates.extend(str(item).strip() for item in paper_ids if str(item).strip())
    resolved_scope = metadata.get("resolved_scope")
    if isinstance(resolved_scope, dict):
        resolved_paper_ids = resolved_scope.get("paper_ids")
        if isinstance(resolved_paper_ids, list):
            candidates.extend(str(item).strip() for item in resolved_paper_ids if str(item).strip())
    return {item for item in candidates if item}


def _get_inventory_paper_ids(metadata: dict[str, Any]) -> set[str]:
    candidates: list[str] = []
    paper_source_ids = metadata.get("paper_source_ids")
    if isinstance(paper_source_ids, list):
        candidates.extend(str(item).strip() for item in paper_source_ids if str(item).strip())
    paper_sources = metadata.get("paper_sources")
    if isinstance(paper_sources, list):
        candidates.extend(
            str(item.get("paper_id") or "").strip()
            for item in paper_sources
            if isinstance(item, dict) and str(item.get("paper_id") or "").strip()
        )
    return {item for item in candidates if item}


def _extract_inline_citation_tokens(text: str) -> list[str]:
    return [match.group(1).strip() for match in INLINE_CITATION_PATTERN.finditer(str(text or "")) if match.group(1).strip()]


def _assert_evaluation_has_finite_metrics(normalized: dict[str, Any]) -> None:
    metrics = normalized.get("metrics") or {}
    if any(_coerce_metric_value(value) is not None for value in metrics.values()):
        return

    for record in normalized.get("records") or []:
        if any(_coerce_metric_value(record.get(name)) is not None for name in METRIC_NAMES):
            return

    raise RuntimeError(
        "Ragas 评估已执行，但没有产出任何有限指标值。"
        " 这通常表示上游评估模型请求被拦截、provider 与当前项目模型注入链路不兼容，"
        " 或 credentials/model policy 阻止了实际评分。"
        " 请检查控制台里是否已有 `Your request was blocked`、401/403、模型不可用等上游报错。"
    )


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Ragas Evaluation Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- dataset_path: `{report['dataset_path']}`",
        f"- dataset_name: `{report.get('dataset_name', '')}`",
        f"- adapter: `{report.get('adapter', 'replay')}`",
    ]
    dataset_tags = report.get("dataset_tags")
    if isinstance(dataset_tags, list) and dataset_tags:
        lines.append(f"- dataset_tags: `{', '.join(dataset_tags)}`")
    if report.get("requested_mode"):
        lines.append(f"- requested_mode: `{report['requested_mode']}`")
    if report.get("requested_graph_mode"):
        lines.append(f"- requested_graph_mode: `{report['requested_graph_mode']}`")
    if report.get("mode"):
        lines.append(f"- mode: `{report['mode']}`")
    if report.get("graph_mode"):
        lines.append(f"- graph_mode: `{report['graph_mode']}`")
    lines.extend(
        [
            f"- sample_count: {report['sample_count']}",
            f"- created_at: `{report['created_at']}`",
            "",
            "## Aggregate Metrics",
            "",
        ]
    )
    metrics = report.get("metrics") or {}
    if metrics:
        for name in METRIC_NAMES:
            if name in metrics:
                lines.append(f"- {name}: {metrics[name]}")
    else:
        lines.append("- no metrics available")
    lines.extend(_build_project_metric_group_lines(report))
    coverage_summary = report.get("coverage_summary") if isinstance(report.get("coverage_summary"), dict) else {}
    if coverage_summary:
        lines.extend(["", "## Coverage Summary", ""])
        lines.extend(_build_coverage_markdown_lines(coverage_summary))
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    if gate:
        lines.extend(["", "## Gate", ""])
        lines.extend(_build_gate_markdown_lines(gate))
    baseline = report.get("baseline_comparison") if isinstance(report.get("baseline_comparison"), dict) else {}
    if baseline:
        lines.extend(["", "## Baseline Comparison", ""])
        lines.extend(_build_baseline_markdown_lines(baseline))
    lines.extend(["", "## Samples", ""])
    for index, record in enumerate(report.get("records") or [], start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                f"- question: {record.get('question', '')}",
                f"- response: {record.get('response', '')}",
                f"- reference_answer: {record.get('reference_answer', '')}",
            ]
        )
        for name in METRIC_NAMES:
            if name in record:
                lines.append(f"- {name}: {record[name]}")
        for name in PROJECT_METRIC_NAMES:
            if name in record:
                lines.append(f"- {name}: {record[name]}")
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if metadata:
            requested_mode = str(metadata.get("requested_mode") or "").strip()
            resolved_mode = str(metadata.get("resolved_mode") or metadata.get("retrieval_mode") or "").strip()
            dialogue_mode = str(metadata.get("dialogue_mode") or "").strip()
            if requested_mode:
                lines.append(f"- requested_mode: {requested_mode}")
            if resolved_mode:
                lines.append(f"- resolved_mode: {resolved_mode}")
            if dialogue_mode:
                lines.append(f"- dialogue_mode: {dialogue_mode}")
            graph_mode = str(metadata.get("graph_mode") or "").strip()
            if graph_mode:
                lines.append(f"- graph_mode: {graph_mode}")
            if metadata.get("project_id"):
                lines.append(f"- project_id: {metadata['project_id']}")
            if metadata.get("paper_ids"):
                lines.append(f"- paper_ids: {json.dumps(metadata['paper_ids'], ensure_ascii=False)}")
            if metadata.get("filter_expr"):
                lines.append(f"- filter_expr: {metadata['filter_expr']}")
            if "citations_count" in metadata:
                lines.append(f"- citations_count: {metadata['citations_count']}")
            if "chunk_citations_count" in metadata:
                lines.append(f"- chunk_citations_count: {metadata['chunk_citations_count']}")
            if "answer_confidence" in metadata:
                lines.append(f"- answer_confidence: {metadata.get('answer_confidence', '')}")
            if "synthesis_status" in metadata:
                lines.append(f"- synthesis_status: {metadata.get('synthesis_status', '')}")
            if "research_objects_count" in metadata:
                lines.append(f"- research_objects_count: {metadata['research_objects_count']}")
            if "note_actions_count" in metadata:
                lines.append(f"- note_actions_count: {metadata['note_actions_count']}")
        lines.append("")
    lines.extend(measurement_lines(report.get("measurement_coverage", {})))
    if "evidence_benchmark" in report:
        from evals.evidence.report import render_evidence_report

        lines.extend(render_evidence_report(report["evidence_benchmark"]))
    return "\n".join(lines).strip() + "\n"


def _build_suite_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Ragas Evaluation Suite Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- suite_name: `{report.get('suite_name', '')}`",
        f"- manifest_path: `{report.get('manifest_path', '')}`",
        f"- dataset_count: {report.get('dataset_count', 0)}",
        f"- sample_count: {report.get('sample_count', 0)}",
        f"- created_at: `{report.get('created_at', '')}`",
    ]
    if report.get("description"):
        lines.append(f"- description: {report['description']}")
    lines.extend(["", "## Aggregate Metrics", ""])
    metrics = report.get("metrics") or {}
    if metrics:
        for name in METRIC_NAMES:
            if name in metrics:
                lines.append(f"- {name}: {metrics[name]}")
    else:
        lines.append("- no metrics available")
    lines.extend(_build_project_metric_group_lines(report))
    coverage_summary = report.get("coverage_summary") if isinstance(report.get("coverage_summary"), dict) else {}
    if coverage_summary:
        lines.extend(["", "## Coverage Summary", ""])
        lines.extend(_build_coverage_markdown_lines(coverage_summary))
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    if gate:
        lines.extend(["", "## Gate", ""])
        lines.extend(_build_gate_markdown_lines(gate))
    baseline = report.get("baseline_comparison") if isinstance(report.get("baseline_comparison"), dict) else {}
    if baseline:
        lines.extend(["", "## Baseline Comparison", ""])
        lines.extend(_build_baseline_markdown_lines(baseline))
    lines.extend(["", "## Datasets", ""])
    for dataset in report.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        lines.extend(
            [
                f"### {dataset.get('dataset_name', '')}",
                "",
                f"- dataset_path: `{dataset.get('dataset_path', '')}`",
                f"- adapter: `{dataset.get('adapter', '')}`",
                f"- mode: `{dataset.get('mode', '')}`",
                f"- graph_mode: `{dataset.get('graph_mode', '')}`",
                f"- sample_count: {dataset.get('sample_count', 0)}",
            ]
        )
        dataset_tags = dataset.get("dataset_tags")
        if isinstance(dataset_tags, list) and dataset_tags:
            lines.append(f"- dataset_tags: `{', '.join(dataset_tags)}`")
        metrics = dataset.get("metrics") if isinstance(dataset.get("metrics"), dict) else {}
        for name in METRIC_NAMES:
            if name in metrics:
                lines.append(f"- {name}: {metrics[name]}")
        project_metrics = dataset.get("project_metrics") if isinstance(dataset.get("project_metrics"), dict) else {}
        for name in PROJECT_METRIC_NAMES:
            if name in project_metrics:
                lines.append(f"- {name}: {project_metrics[name]}")
        dataset_gate = dataset.get("gate") if isinstance(dataset.get("gate"), dict) else {}
        if dataset_gate:
            lines.append(f"- gate_passed: {dataset_gate.get('passed', False)}")
        if dataset.get("markdown_report_path"):
            lines.append(f"- report: `{dataset['markdown_report_path']}`")
        lines.append("")
    lines.extend(measurement_lines(report.get("measurement_coverage", {})))
    return "\n".join(lines).strip() + "\n"


def _build_coverage_markdown_lines(coverage_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section_name in ("route", "scope", "citation", "payload", "research"):
        section = coverage_summary.get(section_name)
        if not isinstance(section, dict):
            continue
        lines.append(f"### {section_name.title()}")
        lines.append("")
        for key, value in section.items():
            if isinstance(value, dict) and "count" in value and "rate" in value:
                lines.append(f"- {key}: {value['count']} ({value['rate']})")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        lines.append("")
    return lines


def _build_project_metric_group_lines(report: dict[str, Any]) -> list[str]:
    project_metrics = report.get("project_metrics") if isinstance(report.get("project_metrics"), dict) else {}
    lines = ["", "## Project Contract Metrics", ""]
    lines.extend(_build_metric_lines(project_metrics, CONTRACT_METRIC_NAMES, empty_text="no project contract metrics available"))
    lines.extend(["", "## Project Debug Metrics", ""])
    lines.extend(_build_metric_lines(project_metrics, DEBUG_METRIC_NAMES, empty_text="no project debug metrics available"))
    return lines


def _build_metric_lines(metrics: dict[str, Any], names: tuple[str, ...], *, empty_text: str) -> list[str]:
    lines = [f"- {name}: {metrics[name]}" for name in names if name in metrics]
    return lines or [f"- {empty_text}"]


def _build_gate_markdown_lines(gate: dict[str, Any]) -> list[str]:
    lines = [
        f"- mode: `{gate.get('mode', '')}`",
        f"- passed: {bool(gate.get('passed'))}",
        f"- regression_tolerance: {gate.get('regression_tolerance', 0.0)}",
    ]
    checks = gate.get("checks")
    if isinstance(checks, list):
        lines.append("- checks:")
        for item in checks:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  - {item.get('name', '')}: passed={bool(item.get('passed'))} details={json.dumps(item.get('details'), ensure_ascii=False)}"
            )
    warnings = gate.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append("- warnings:")
        for item in warnings:
            lines.append(f"  - {item}")
    return lines


def _build_baseline_markdown_lines(baseline: dict[str, Any]) -> list[str]:
    lines = [
        f"- baseline_report_path: `{baseline.get('baseline_report_path', '')}`",
    ]
    if baseline.get("baseline_run_id"):
        lines.append(f"- baseline_run_id: `{baseline['baseline_run_id']}`")
    for section_name in ("metrics", "project_metrics"):
        section = baseline.get(section_name)
        if not isinstance(section, dict) or not section:
            continue
        lines.append(f"- {section_name}:")
        for metric_name, payload in section.items():
            if not isinstance(payload, dict):
                continue
            lines.append(
                "  - "
                f"{metric_name}: current={payload.get('current')} baseline={payload.get('baseline')} "
                f"delta={payload.get('delta')} direction={payload.get('direction')}"
            )
    dataset_section = baseline.get("datasets")
    if isinstance(dataset_section, dict) and dataset_section:
        lines.append("- datasets:")
        for dataset_name, payload in dataset_section.items():
            if not isinstance(payload, dict):
                continue
            lines.append(f"  - {dataset_name}:")
            for section_name in ("metrics", "project_metrics"):
                section = payload.get(section_name)
                if not isinstance(section, dict):
                    continue
                for metric_name, metric_payload in section.items():
                    if not isinstance(metric_payload, dict):
                        continue
                    lines.append(
                        "    - "
                        f"{section_name}.{metric_name}: current={metric_payload.get('current')} "
                        f"baseline={metric_payload.get('baseline')} delta={metric_payload.get('delta')} "
                        f"direction={metric_payload.get('direction')}"
                    )
    return lines


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def describe_adapter(adapter: ReplayRAGAdapter | LiveRAGAdapter) -> dict[str, Any]:
    describe = getattr(adapter, "describe", None)
    if callable(describe):
        payload = describe()
        if isinstance(payload, dict):
            return payload
    return {"adapter": getattr(adapter, "adapter_name", adapter.__class__.__name__)}


def build_adapter(*, adapter_name: str = "replay", mode: str = "auto") -> ReplayRAGAdapter | LiveRAGAdapter:
    normalized_adapter = str(adapter_name or "replay").strip().lower() or "replay"
    normalized_mode = str(mode or "auto").strip().lower() or "auto"
    if normalized_adapter not in ADAPTER_CHOICES:
        raise ValueError(f"不支持的 adapter: {adapter_name}")
    if normalized_mode not in LIVE_MODE_CHOICES:
        raise ValueError(f"不支持的 mode: {mode}")
    if normalized_adapter == "replay":
        if normalized_mode != "auto":
            raise ValueError("`--mode` 仅在 `--adapter live` 时可用")
        return ReplayRAGAdapter()
    return LiveRAGAdapter(mode=normalized_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StudyAssistant Ragas offline evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to JSONL dataset")
    parser.add_argument("--manifest", default="", help="Path to dataset manifest JSON for suite runs")
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPERIMENTS_DIR), help="Directory for reports")
    parser.add_argument("--adapter", choices=ADAPTER_CHOICES, default="replay", help="Evaluation adapter")
    parser.add_argument("--mode", choices=LIVE_MODE_CHOICES, default="auto", help="Live retrieval mode override")
    parser.add_argument("--baseline-report", default="", help="Previous report.json or suite-report.json for regression compare")
    parser.add_argument("--gate", choices=GATE_MODE_CHOICES, default="smoke", help="Gate mode")
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.03,
        help="Allowed absolute regression before gate fails",
    )
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit non-zero when gate fails")
    args = parser.parse_args()
    if args.manifest:
        result = run_evaluation_suite(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            baseline_report_path=args.baseline_report or None,
            gate_mode=args.gate,
            regression_tolerance=args.regression_tolerance,
        )
    else:
        adapter = build_adapter(adapter_name=args.adapter, mode=args.mode)
        result = run_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            adapter=adapter,
            baseline_report_path=args.baseline_report or None,
            gate_mode=args.gate,
            regression_tolerance=args.regression_tolerance,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    gate = result.get("gate") if isinstance(result, dict) else None
    if args.fail_on_gate and isinstance(gate, dict) and not bool(gate.get("passed")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
