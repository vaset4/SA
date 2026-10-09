from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from app.services import retrieval_service
from evals.ragas import rag_adapter as ragas_adapter
from evals.ragas import evals as ragas_evals


def _live_record(
    question: str,
    *,
    response: str,
    metadata: dict[str, object],
    contexts: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question": question,
        "retrieved_contexts": contexts or ["ctx"],
        "response": response,
        "reference_answer": f"Reference for {question}",
        "metadata": metadata,
    }


def test_run_evaluation_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What changed in the pipeline?",
                "retrieved_contexts": ["The pipeline now routes PDF parsing through MinerU markdown artifacts."],
                "response": "It routes PDF parsing through MinerU and chunks markdown.",
                "reference_answer": "The pipeline now parses PDFs through MinerU-generated markdown before chunking.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_evaluator(records):
        assert len(records) == 1
        return {
            "metrics": {
                "faithfulness": 0.95,
                "context_precision": 0.9,
                "context_recall": 0.92,
                "answer_relevancy": 0.94,
            },
            "records": [
                {
                    "faithfulness": 0.95,
                    "context_precision": 0.9,
                    "context_recall": 0.92,
                    "answer_relevancy": 0.94,
                }
            ],
        }

    result = ragas_evals.run_evaluation(
        dataset_path=dataset_path,
        output_dir=tmp_path / "experiments",
        evaluator=fake_evaluator,
    )

    json_report = Path(result["json_report_path"])
    markdown_report = Path(result["markdown_report_path"])

    assert json_report.exists()
    assert markdown_report.exists()
    assert result["metrics"]["faithfulness"] == 0.95
    report_text = markdown_report.read_text(encoding="utf-8")
    assert "Aggregate Metrics" in report_text
    assert "Project Contract Metrics" in report_text
    assert "Project Debug Metrics" in report_text
    assert result["quality_metrics"]["faithfulness"] == 0.95
    assert "placeholder_citation_rate" in result["contract_metrics"]
    assert "answer_payload_present" in result["debug_metrics"]


def test_normalize_eval_record_preserves_fact_contract_fields() -> None:
    record = ragas_adapter.normalize_eval_record(
        {
            "question": "Which task does PDFormer target?",
            "retrieved_contexts": ["PDFormer targets traffic flow prediction."],
            "response": "PDFormer targets traffic flow prediction.",
            "reference_answer": "It targets traffic flow prediction.",
            "golden_evidence": ["chunk-pdformer-task"],
            "critical_facts": [
                {
                    "name": "target_task",
                    "must_include": [{"any_of": ["traffic flow prediction", "traffic-flow prediction"]}],
                    "must_not_include": ["traffic speed prediction"],
                }
            ],
            "rubric": {"requires_citation": True},
        }
    )

    assert record["golden_evidence"] == ["chunk-pdformer-task"]
    assert record["critical_facts"][0]["name"] == "target_task"
    assert record["rubric"] == {"requires_citation": True}
    assert not ({"golden_evidence", "critical_facts", "rubric"} & record.get("metadata", {}).keys())


def test_rubric_can_disable_citation_requirement_for_hypothesis_text() -> None:
    record = {
        "response": "A possible hypothesis is that delay-aware attention may help nearby roads.",
        "metadata": {"rubric": {"requires_citation": False, "allows_hypothesis": True}},
    }

    assert ragas_evals._compute_citation_presence_rate(record, record["metadata"]) is None


def test_bootstrap_openai_compat_env_falls_back_to_project_settings(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    info = ragas_evals._bootstrap_openai_compat_env(
        settings=SimpleNamespace(
            llm_api_key="project-key",
            llm_base_url="https://llm.example/v1",
        )
    )

    assert info["used_project_key"] is True
    assert info["used_project_base_url"] is True
    assert ragas_evals.os.environ["OPENAI_API_KEY"] == "project-key"
    assert ragas_evals.os.environ["OPENAI_BASE_URL"] == "https://llm.example/v1"


def test_bootstrap_openai_compat_env_requires_key_when_no_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    try:
        ragas_evals._bootstrap_openai_compat_env(
            settings=SimpleNamespace(
                llm_api_key="",
                llm_base_url="",
            )
        )
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when no OpenAI-compatible key is available")


def test_run_evaluation_fails_when_ragas_returns_only_nan_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Q",
                "retrieved_contexts": ["ctx"],
                "response": "A",
                "reference_answer": "B",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_evaluator(records):
        assert len(records) == 1
        return {
            "metrics": {
                "faithfulness": math.nan,
                "context_precision": math.nan,
                "context_recall": math.nan,
                "answer_relevancy": math.nan,
            },
            "records": [
                {
                    "faithfulness": math.nan,
                    "context_precision": math.nan,
                    "context_recall": math.nan,
                    "answer_relevancy": math.nan,
                }
            ],
        }

    try:
        ragas_evals.run_evaluation(
            dataset_path=dataset_path,
            output_dir=tmp_path / "experiments",
            evaluator=fake_evaluator,
        )
    except RuntimeError as exc:
        assert "没有产出任何有限指标值" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when evaluator returns only NaN metrics")


def test_default_ragas_evaluator_explicitly_injects_project_llm_and_embeddings(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dataset = object()
    llm = SimpleNamespace(set_run_config=lambda config: captured.setdefault("configured_run_config", config))
    embeddings = object()
    run_config = object()
    raw_result = {"faithfulness": 0.88}

    def fake_evaluate(input_dataset, **kwargs):
        captured["dataset"] = input_dataset
        captured["kwargs"] = kwargs
        return raw_result

    monkeypatch.setattr(ragas_evals, "build_ragas_dataset", lambda records: dataset)
    monkeypatch.setattr(ragas_evals, "_build_project_ragas_llm", lambda settings=None: llm)
    monkeypatch.setattr(
        ragas_evals,
        "_build_project_ragas_embeddings",
        lambda settings=None: embeddings,
    )
    monkeypatch.setattr(ragas_evals, "_build_ragas_run_config", lambda settings=None: run_config)
    monkeypatch.setattr(
        ragas_evals,
        "_load_project_settings",
        lambda: SimpleNamespace(
            llm_model_name="project-model",
            llm_api_key="project-key",
            llm_base_url="https://llm.example/v1",
            llm_temperature=0.0,
            llm_timeout=240,
            embedding_model="bge-m3",
        ),
    )
    monkeypatch.setattr(ragas_evals, "_is_upstream_provider_error", lambda exc: False)
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=fake_evaluate))

    result = ragas_evals.default_ragas_evaluator(
        [
            {
                "question": "Q",
                "retrieved_contexts": ["ctx"],
                "response": "A",
                "reference_answer": "B",
            }
        ]
    )

    assert result == {"raw_result": raw_result}
    assert captured["configured_run_config"] is run_config
    assert captured["dataset"] is dataset
    assert captured["kwargs"] == {
        "llm": llm,
        "embeddings": embeddings,
        "run_config": run_config,
        "raise_exceptions": True,
    }


def test_project_ragas_llm_throttles_between_requests(monkeypatch) -> None:
    sleeps: list[float] = []
    now = {"value": 50.0}

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["value"] += seconds

    def _fake_monotonic() -> float:
        return now["value"]

    llm = ragas_evals._ProjectRagasLLM(
        model="project-model",
        api_key="project-key",
        base_url="https://llm.example/v1",
        temperature=0.0,
        timeout=240,
        min_request_interval_seconds=4.5,
    )
    ragas_evals._ProjectRagasLLM._host_next_request_at.clear()

    monkeypatch.setattr(ragas_evals.time, "sleep", _fake_sleep)
    monkeypatch.setattr(ragas_evals.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(
        llm,
        "_build_client",
        lambda temperature=0.0: SimpleNamespace(
            invoke=lambda payload, use_cache=False: SimpleNamespace(content=f"ok:{payload}")  # noqa: ARG005
        ),
    )

    first = llm._invoke_once(payload="prompt-1", temperature=0.0)
    second = llm._invoke_once(payload="prompt-2", temperature=0.0)

    assert first == "ok:prompt-1"
    assert second == "ok:prompt-2"
    assert sleeps == [4.5]


def test_run_evaluation_persists_adapter_metadata(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q",
                    "retrieved_contexts": ["ctx"],
                    "response": "A",
                    "reference_answer": "B",
                }
            ]

        def describe(self):
            return {
                "adapter": "live",
                "mode": "fusion",
                "graph_mode": "hybrid",
                "requested_mode": "fusion",
                "requested_graph_mode": "hybrid",
            }

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 1.0},
            "records": [{"faithfulness": 1.0}],
        },
    )

    assert result["adapter"] == "live"
    assert result["mode"] == "fusion"
    assert result["graph_mode"] == "hybrid"
    assert result["requested_mode"] == "fusion"
    assert "project_metrics" in result


def test_build_adapter_supports_live_mode() -> None:
    adapter = ragas_evals.build_adapter(adapter_name="live", mode="graph")

    assert isinstance(adapter, ragas_adapter.LiveRAGAdapter)
    assert adapter.describe()["mode"] == "graph"


def test_build_adapter_rejects_mode_for_replay() -> None:
    try:
        ragas_evals.build_adapter(adapter_name="replay", mode="fusion")
    except ValueError as exc:
        assert "--adapter live" in str(exc)
    else:
        raise AssertionError("expected replay adapter to reject explicit live mode")


def test_live_rag_adapter_uses_graph_context_and_mode_override(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "How are the methods related?",
                "reference_answer": "They are connected through a shared block-aware retrieval pipeline.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_state: dict[str, object] = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "answer": "They are linked through the same graph pipeline.",
                "graph_context": "Graph says both methods share a retrieval pipeline.",
                "search_results": [],
                "retrieval_mode": "graph",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "skipped", "graph": "ok"},
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="graph",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert captured_state["retrieval_mode_override"] == "graph"
    assert records[0]["retrieved_contexts"] == ["Graph says both methods share a retrieval pipeline."]
    assert records[0]["metadata"]["requested_mode"] == "graph"
    assert records[0]["metadata"]["resolved_mode"] == "graph"
    assert records[0]["metadata"]["retrieval_mode"] == "graph"


def test_live_rag_adapter_reads_full_text_for_fusion_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Summarize the evidence",
                "reference_answer": "The evidence combines vector chunks with graph summary.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        async def ainvoke(self, state):
            assert state["retrieval_mode_override"] == "fusion"
            return {
                "answer": "The answer uses both vector and graph evidence.",
                "graph_context": "Graph summary evidence.",
                "search_results": [{"chunk_id": "chunk-1", "preview": "preview only"}],
                "retrieval_mode": "fusion",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "ok"},
            }

    class FakeStore:
        def read_chunk(self, chunk_id, already_read):
            assert chunk_id == "chunk-1"
            assert already_read == set()
            already_read.add(chunk_id)
            return {
                "chunk_id": chunk_id,
                "full_text": "Full chunk text from Milvus.",
                "paper_id": "paper-1",
                "section": "Intro",
                "page_num": 1,
                "skipped": False,
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="fusion",
        rag_node_factory=lambda: FakeGraph(),
        vector_provider_factory=lambda: FakeStore(),
    )

    records = adapter.load(dataset_path)

    assert records[0]["retrieved_contexts"] == [
        "Full chunk text from Milvus.",
        "Graph summary evidence.",
    ]
    assert records[0]["metadata"]["requested_mode"] == "fusion"
    assert records[0]["metadata"]["resolved_mode"] == "fusion"


def test_retrieval_service_supports_eval_mode_override() -> None:
    result = retrieval_service.resolve_retrieval_mode_override(
        {
            "retrieval_mode_override": "fusion",
            "graph_mode_override": "local",
        }
    )

    assert result is not None
    assert result.mode == "fusion"
    assert result.graph_mode == "local"


def test_live_rag_adapter_wraps_milvus_health_error_with_actionable_hint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What is the main contribution of the paper?",
                "reference_answer": "The main contribution is the markdown-based ingestion pipeline.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        async def ainvoke(self, state):  # noqa: ARG002
            raise RuntimeError("milvus failed: Milvus health probe failed: connect timeout")

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    try:
        adapter.load(dataset_path)
    except RuntimeError as exc:
        text = str(exc)
        assert "当前路由需要 Milvus" in text
        assert "--mode graph" in text
    else:
        raise AssertionError("expected actionable RuntimeError for Milvus health probe failure")


def test_live_rag_adapter_passes_scope_constraints_from_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What is the main contribution of the paper?",
                "reference_answer": "The paper introduces markdown-based ingestion.",
                "paper_ids": ["paper-123"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_state: dict[str, object] = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "answer": "The main contribution is markdown-based ingestion.",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "milvus",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "skipped"},
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    adapter.load(dataset_path)

    assert captured_state["structured_query"] == {
        "paper_ids": ["paper-123"],
        "resolved_scope": {"paper_ids": ["paper-123"]},
    }


def test_live_rag_adapter_preserves_dialogue_mode_and_synthesis_metadata(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Compare the two papers in this project.",
                "reference_answer": "The answer should stay in research mode and keep the synthesis state.",
                "project_id": "project-123",
                "paper_ids": ["paper-a", "paper-b"],
                "dialogue_mode": "research",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_state: dict[str, object] = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "answer": "The two papers differ in spatial modeling assumptions.",
                "search_results": [],
                "graph_context": "graph hint",
                "retrieval_mode": "fusion",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "ok"},
                "dialogue_mode": "research",
                "research_objects": [{"key": "methods"}],
                "note_actions": [{"key": "copy_summary_note"}],
                "synthesis_state": {
                    "status": "ready",
                    "dialogue_mode": "research",
                    "project_scope": {"project_id": "project-123", "paper_ids": ["paper-a", "paper-b"]},
                    "paper_focus": {"mode": "multi_paper"},
                    "evidence_summary": {"paper_count": 2, "citation_count": 1},
                    "dimension_status": {"methods": "supported"},
                    "supported_dimensions": ["methods"],
                    "unsupported_dimensions": ["metrics"],
                    "open_questions": ["还需要补指标对比。"],
                    "evidence_layers": {"milvus": "原文证据", "lightrag": "全局线索"},
                    "citations_count": 1,
                },
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )
    monkeypatch.setattr(
        ragas_adapter,
        "_resolve_live_project_context",
        lambda **kwargs: {
            "project_id": kwargs["project_id"],
            "paper_ids": ["paper-a", "paper-b"],
            "resolved_scope": {"project_id": kwargs["project_id"], "paper_ids": ["paper-a", "paper-b"]},
        },
    )

    records = adapter.load(dataset_path)

    assert captured_state["dialogue_mode"] == "research"
    assert captured_state["structured_query"]["dialogue_mode"] == "research"
    assert records[0]["metadata"]["dialogue_mode"] == "research"
    assert records[0]["metadata"]["requested_dialogue_mode"] == "research"
    assert records[0]["metadata"]["synthesis_status"] == "ready"
    assert records[0]["metadata"]["research_objects_count"] == 1
    assert records[0]["metadata"]["note_actions_count"] == 1
    assert records[0]["metadata"]["paper_focus"] == {"mode": "multi_paper"}


def test_live_rag_adapter_resolves_paper_title_into_scope_constraints(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What is the main contribution of the paper?",
                "reference_answer": "The paper introduces markdown-based ingestion.",
                "paper_title": "MinerU Markdown Pipeline Paper",
                "library_id": "default",
                "user_id": "anonymous",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_state: dict[str, object] = {}
    captured_resolver_args: dict[str, object] = {}

    def fake_resolver(paper_titles, *, library_id=None, user_id=None):
        captured_resolver_args["paper_titles"] = list(paper_titles)
        captured_resolver_args["library_id"] = library_id
        captured_resolver_args["user_id"] = user_id
        return ["paper-by-title-1"]

    monkeypatch.setattr(ragas_adapter, "_resolve_live_paper_ids_from_titles", fake_resolver)

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "answer": "The main contribution is markdown-based ingestion.",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "milvus",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "skipped"},
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert captured_resolver_args == {
        "paper_titles": ["MinerU Markdown Pipeline Paper"],
        "library_id": "default",
        "user_id": "anonymous",
    }
    assert captured_state["structured_query"] == {
        "paper_ids": ["paper-by-title-1"],
        "paper_titles": ["MinerU Markdown Pipeline Paper"],
        "resolved_scope": {"paper_ids": ["paper-by-title-1"]},
    }
    assert records[0]["metadata"]["paper_titles"] == ["MinerU Markdown Pipeline Paper"]


def test_load_live_jsonl_dataset_supports_paper_titles_in_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Q",
                "reference_answer": "A",
                "metadata": {
                    "paper_title": "Stored Paper Title",
                    "library_id": "default",
                    "user_id": "anonymous",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = ragas_adapter.load_live_jsonl_dataset(dataset_path)

    assert records == [
        {
            "question": "Q",
            "reference_answer": "A",
            "paper_titles": ["Stored Paper Title"],
            "library_id": "default",
            "user_id": "anonymous",
            "metadata": {
                "paper_title": "Stored Paper Title",
                "library_id": "default",
                "user_id": "anonymous",
            },
        }
    ]


def test_load_live_jsonl_dataset_supports_project_id_in_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Q",
                "reference_answer": "A",
                "metadata": {
                    "project_id": "project-123",
                    "user_id": "anonymous",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = ragas_adapter.load_live_jsonl_dataset(dataset_path)

    assert records == [
        {
            "question": "Q",
            "reference_answer": "A",
            "project_id": "project-123",
            "user_id": "anonymous",
            "metadata": {
                "project_id": "project-123",
                "user_id": "anonymous",
            },
        }
    ]


def test_load_live_jsonl_dataset_supports_direct_scope_and_dialogue_mode_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "比较这两篇论文的方法差异",
                "reference_answer": "A",
                "project_id": "proj-1",
                "paper_ids": ["paper-1", "paper-2"],
                "dialogue_mode": "research",
                "library_id": "default",
                "user_id": "anonymous",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = ragas_adapter.load_live_jsonl_dataset(dataset_path)

    assert records == [
        {
            "question": "比较这两篇论文的方法差异",
            "reference_answer": "A",
            "project_id": "proj-1",
            "paper_ids": ["paper-1", "paper-2"],
            "dialogue_mode": "research",
            "library_id": "default",
            "user_id": "anonymous",
        }
    ]


def test_live_rag_adapter_tracks_requested_and_resolved_modes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What is the main contribution of the paper?",
                "reference_answer": "The paper introduces markdown-based ingestion.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        async def ainvoke(self, state):  # noqa: ARG002
            return {
                "answer": "The main contribution is markdown-based ingestion.",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "milvus",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "skipped"},
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert records[0]["metadata"]["requested_mode"] == "auto"
    assert records[0]["metadata"]["resolved_mode"] == "milvus"


def test_live_rag_adapter_resolves_project_scope_into_filter_expr(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Summarize the project scope",
                "reference_answer": "The answer should stay inside the project scope.",
                "project_id": "project-123",
                "user_id": "anonymous",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_state: dict[str, object] = {}

    monkeypatch.setattr(
        ragas_adapter,
        "_resolve_live_project_context",
        lambda **kwargs: {
            "project_id": kwargs["project_id"],
            "paper_ids": ["paper-a", "paper-b"],
            "filter_expr": 'paper_id == "paper-a" or paper_id == "paper-b"',
            "resolved_scope": {
                "project_id": kwargs["project_id"],
                "paper_ids": ["paper-a", "paper-b"],
            },
        },
    )

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                "answer": "The answer stays inside the project scope.",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "milvus",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "skipped"},
                "citations": [{"paper_id": "paper-a"}],
                "answer_confidence": "medium",
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert captured_state["filter_expr"] == 'paper_id == "paper-a" or paper_id == "paper-b"'
    assert captured_state["structured_query"] == {
        "project_id": "project-123",
        "paper_ids": ["paper-a", "paper-b"],
        "resolved_scope": {
            "project_id": "project-123",
            "paper_ids": ["paper-a", "paper-b"],
        },
    }
    assert records[0]["metadata"]["project_id"] == "project-123"
    assert records[0]["metadata"]["resolved_scope"] == {
        "project_id": "project-123",
        "paper_ids": ["paper-a", "paper-b"],
    }
    assert records[0]["metadata"]["citations_count"] == 1
    assert records[0]["metadata"]["answer_confidence"] == "medium"


def test_live_rag_adapter_uses_inventory_contexts_for_project_paper_inventory(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "当前有哪些论文",
                "reference_answer": "回答应列出项目范围内论文清单。",
                "paper_ids": ["paper-a"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        async def ainvoke(self, state):  # noqa: ARG002
            return {
                "answer": "当前项目范围内有一篇论文：Paper A。",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "unused", "graph": "unused", "inventory": "ok"},
                "paper_sources": [
                    {
                        "paper_id": "paper-a",
                        "title": "Paper A",
                        "authors": ["Author A"],
                        "abstract": "Abstract A",
                    }
                ],
                "inventory_contexts": [
                    "title: Paper A\npaper_id: paper-a\nauthors: Author A\nabstract: Abstract A"
                ],
                "inventory_source_ids": ["paper-a"],
                "retrieval_reasoning": "当前问题是在项目范围内列出论文清单，已直接读取项目作用域内的论文元数据作为回答依据。",
                "answer_confidence": "medium",
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert records[0]["retrieved_contexts"] == [
        "title: Paper A\npaper_id: paper-a\nauthors: Author A\nabstract: Abstract A"
    ]
    assert records[0]["metadata"]["resolved_mode"] == "inventory"
    assert records[0]["metadata"]["paper_source_ids"] == ["paper-a"]
    assert records[0]["metadata"]["inventory_contexts_count"] == 1
    assert "项目作用域内的论文元数据" in records[0]["metadata"]["retrieval_reasoning"]


def test_live_rag_adapter_uses_chunk_citations_as_fallback_structured_citations(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "把方法部分的原文找出来",
                "reference_answer": "回答应带出块级原文证据。",
                "paper_ids": ["paper-1"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        async def ainvoke(self, state):  # noqa: ARG002
            return {
                "answer": "方法部分提到了时空卷积结构。",
                "search_results": [],
                "graph_context": "",
                "retrieval_mode": "milvus",
                "graph_mode": "hybrid",
                "source_status": {"milvus": "ok", "graph": "unused"},
                "citations": [],
                "chunk_citations": [
                    {
                        "evidence_id": "E6",
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "section": "Method",
                        "page_num": 4,
                        "excerpt": "时空卷积结构",
                    }
                ],
                "answer_confidence": "medium",
            }

    adapter = ragas_adapter.LiveRAGAdapter(
        mode="auto",
        rag_node_factory=lambda: FakeGraph(),
    )

    records = adapter.load(dataset_path)

    assert records[0]["metadata"]["citations_count"] == 1
    assert records[0]["metadata"]["citations"][0]["evidence_id"] == "E6"
    assert records[0]["metadata"]["citations"][0]["paper_id"] == "paper-1"
    assert records[0]["metadata"]["citations"][0]["chunk_id"] == "chunk-1"


def test_run_evaluation_accepts_list_evidence_layers_and_unused_graph_status(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["ctx"],
                    "response": "A1",
                    "reference_answer": "R1",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "unused"},
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"paper_ids": ["paper-a"]},
                        "dialogue_mode": "lookup",
                        "requested_dialogue_mode": "lookup",
                        "citations": [{"paper_id": "paper-a", "chunk_id": "chunk-1"}],
                        "citations_count": 1,
                        "chunk_citations_count": 1,
                        "answer_confidence": "high",
                        "synthesis_status": "ready",
                        "synthesis_state": {
                            "status": "ready",
                            "dialogue_mode": "lookup",
                            "project_scope": {"paper_ids": ["paper-a"]},
                            "paper_focus": {"mode": "single_paper"},
                            "evidence_summary": {"paper_count": 1, "citation_count": 1},
                            "dimension_status": {"methods": "supported"},
                            "supported_dimensions": ["methods"],
                            "unsupported_dimensions": [],
                            "open_questions": [],
                            "evidence_layers": [
                                {
                                    "layer": "milvus",
                                    "role": "primary_evidence",
                                    "output": "parent_chunk_evidence",
                                    "notes": "负责原文证据。",
                                }
                            ],
                            "citations_count": 1,
                        },
                        "evidence_layers": [
                            {
                                "layer": "milvus",
                                "role": "primary_evidence",
                                "output": "parent_chunk_evidence",
                                "notes": "负责原文证据。",
                            }
                        ],
                        "supported_dimensions": ["methods"],
                        "open_questions": [],
                    },
                }
            ]

        def describe(self):
            return {
                "adapter": "live",
                "mode": "auto",
                "graph_mode": "hybrid",
                "requested_mode": "auto",
                "requested_graph_mode": "hybrid",
            }

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.9, "context_precision": 0.8},
            "records": [{"faithfulness": 0.9, "context_precision": 0.8} for _ in records],
        },
        gate_mode="smoke",
    )

    assert result["project_metrics"]["route_correctness"] == 1.0
    assert result["project_metrics"]["synthesis_payload_present"] == 1.0
    assert result["coverage_summary"]["research"]["with_evidence_layers_field"]["count"] == 1
    assert result["coverage_summary"]["research"]["research_metadata_complete"]["count"] == 1


def test_run_evaluation_computes_project_specific_metrics(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["ctx"],
                    "response": "Scoped answer with real citation [paper-a].",
                    "reference_answer": "A1",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "skipped"},
                        "project_id": "project-123",
                        "filter_expr": 'paper_id == "paper-a"',
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"project_id": "project-123", "paper_ids": ["paper-a"]},
                        "citations": [{"paper_id": "paper-a"}],
                        "citations_count": 1,
                        "answer_confidence": "high",
                    },
                },
                {
                    "question": "Q2",
                    "retrieved_contexts": ["ctx"],
                    "response": "Answer without usable citation [paper_id].",
                    "reference_answer": "A2",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "graph",
                        "resolved_mode": "graph",
                        "retrieval_mode": "graph",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "skipped", "graph": "ok"},
                        "paper_ids": ["paper-b"],
                        "resolved_scope": {"paper_ids": ["paper-b"]},
                        "citations": [],
                        "citations_count": 0,
                        "answer_confidence": "",
                    },
                },
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.5},
            "records": [{"faithfulness": 0.5}, {"faithfulness": 0.5}],
        },
    )

    assert result["project_metrics"]["citation_presence_rate"] == 0.5
    assert result["project_metrics"]["placeholder_citation_rate"] == 0.5
    assert result["project_metrics"]["route_correctness"] == 1.0
    assert result["project_metrics"]["scope_adherence"] == 1.0
    assert result["project_metrics"]["answer_payload_present"] == 0.9375


def test_run_evaluation_computes_critical_fact_contract_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")

    class FakeAdapter:
        adapter_name = "replay"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["chunk-pdformer-task says traffic flow prediction."],
                    "response": "PDFormer targets traffic-flow prediction.",
                    "reference_answer": "PDFormer targets traffic flow prediction.",
                    "golden_evidence": ["chunk-pdformer-task"],
                    "critical_facts": [
                        {
                            "name": "target_task",
                            "must_include": [{"any_of": ["traffic flow prediction", "traffic-flow prediction"]}],
                            "must_not_include": ["traffic speed prediction"],
                        }
                    ],
                },
                {
                    "question": "Q2",
                    "retrieved_contexts": ["chunk-pdformer-task says traffic flow prediction."],
                    "response": "PDFormer targets traffic speed prediction.",
                    "reference_answer": "PDFormer targets traffic flow prediction.",
                    "critical_facts": [
                        {
                            "name": "target_task",
                            "must_include": ["traffic flow prediction"],
                            "must_not_include": ["traffic speed prediction"],
                        }
                    ],
                },
            ]

        def describe(self):
            return {"adapter": "replay"}

    result = ragas_evals.run_evaluation(
        dataset_path=dataset_path,
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {"metrics": {"faithfulness": 0.9}, "records": [{}, {}]},
        gate_mode="smoke",
    )

    assert result["records"][0]["critical_fact_coverage"] == 1.0
    assert result["records"][0]["critical_fact_error_rate"] == 0.0
    assert result["records"][0]["golden_evidence_presence_rate"] == 1.0
    assert result["records"][1]["critical_fact_coverage"] == 0.0
    assert result["records"][1]["critical_fact_error_rate"] == 1.0
    assert result["project_metrics"]["fact_contract_score"] == 0.5
    checks = {item["name"]: item["passed"] for item in result["gate"]["checks"]}
    assert checks["smoke_fact_contract_intact"] is False


def test_route_correctness_does_not_infer_milvus_from_auto_scope() -> None:
    metadata = {
        "requested_mode": "auto",
        "resolved_mode": "graph",
        "source_status": {"milvus": "unused", "graph": "ok", "inventory": "unused"},
        "project_id": "project-1",
        "paper_ids": ["paper-a"],
        "resolved_scope": {"project_id": "project-1", "paper_ids": ["paper-a"]},
    }

    assert ragas_evals._compute_route_correctness(metadata) == 1.0


def test_route_correctness_accepts_explicit_fusion_with_scope() -> None:
    metadata = {
        "requested_mode": "fusion",
        "resolved_mode": "fusion",
        "source_status": {"milvus": "ok", "graph": "empty", "inventory": "unused"},
        "project_id": "project-1",
        "paper_ids": ["paper-a"],
        "resolved_scope": {"project_id": "project-1", "paper_ids": ["paper-a"]},
    }

    assert ragas_evals._compute_route_correctness(metadata) == 1.0


def test_run_evaluation_excludes_not_applicable_scope_from_project_metric_mean(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                _live_record(
                    "Q1",
                    response="Scoped answer with citation [paper-a].",
                    metadata={
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok"},
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"paper_ids": ["paper-a"]},
                        "citations": [{"paper_id": "paper-a"}],
                        "citations_count": 1,
                        "answer_confidence": "high",
                    },
                ),
                _live_record(
                    "Q2",
                    response="Unscoped answer.",
                    metadata={
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok"},
                        "citations": [],
                        "citations_count": 0,
                        "answer_confidence": "medium",
                    },
                ),
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.8},
            "records": [{"faithfulness": 0.8} for _ in records],
        },
    )

    assert result["project_metrics"]["scope_adherence"] == 1.0
    assert "scope_adherence" not in result["records"][1]


def test_run_evaluation_excludes_citation_not_applicable_from_project_metric_mean(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                _live_record(
                    "Q1",
                    response="Answer with citation [paper-a].",
                    metadata={
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok"},
                        "citations": [{"paper_id": "paper-a"}],
                        "citations_count": 1,
                        "answer_confidence": "high",
                    },
                ),
                _live_record(
                    "Q2",
                    response="Open-ended writing advice.",
                    metadata={
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "none",
                        "retrieval_mode": "none",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "skipped"},
                        "requires_citation": False,
                        "retrieval_policy": "should_not_use",
                        "citations": [],
                        "citations_count": 0,
                        "answer_confidence": "medium",
                    },
                ),
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.8},
            "records": [{"faithfulness": 0.8} for _ in records],
        },
    )

    assert result["project_metrics"]["citation_presence_rate"] == 1.0
    assert "citation_presence_rate" not in result["records"][1]


def test_run_evaluation_includes_coverage_summary_dataset_tags_and_gate(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["ctx"],
                    "response": "Scoped answer [paper-a].",
                    "reference_answer": "A1",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "skipped"},
                        "dialogue_mode": "lookup",
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"paper_ids": ["paper-a"]},
                        "citations": [{"paper_id": "paper-a"}],
                        "citations_count": 1,
                        "chunk_citations_count": 1,
                        "answer_confidence": "high",
                        "synthesis_status": "ready",
                        "synthesis_state": {
                            "status": "ready",
                            "dialogue_mode": "lookup",
                            "project_scope": {"paper_ids": ["paper-a"]},
                            "paper_focus": {"mode": "single_paper"},
                            "evidence_summary": {"paper_count": 1, "citation_count": 1},
                            "dimension_status": {"methods": "supported"},
                            "supported_dimensions": ["methods"],
                            "unsupported_dimensions": [],
                            "open_questions": [],
                            "evidence_layers": {"milvus": "原文证据"},
                            "citations_count": 1,
                        },
                        "evidence_layers": {"milvus": "原文证据"},
                        "supported_dimensions": ["methods"],
                        "open_questions": [],
                    },
                }
            ]

        def describe(self):
            return {
                "adapter": "live",
                "mode": "auto",
                "graph_mode": "hybrid",
                "requested_mode": "auto",
                "requested_graph_mode": "hybrid",
            }

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        dataset_name="scope-smoke",
        dataset_tags=["live", "scope", "smoke"],
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.9, "context_precision": 0.8},
            "records": [{"faithfulness": 0.9, "context_precision": 0.8} for _ in records],
        },
        gate_mode="smoke",
    )

    assert result["dataset_name"] == "scope-smoke"
    assert result["dataset_tags"] == ["live", "scope", "smoke"]
    assert result["coverage_summary"]["route"]["route_metadata_complete"]["count"] == 1
    assert result["coverage_summary"]["scope"]["scope_metadata_complete"]["count"] == 1
    assert result["coverage_summary"]["citation"]["citation_metadata_complete"]["count"] == 1
    assert result["coverage_summary"]["payload"]["payload_metadata_complete"]["count"] == 1
    assert result["coverage_summary"]["research"]["research_metadata_complete"]["count"] == 1
    assert result["gate"]["passed"] is True
    assert result["gate"]["mode"] == "smoke"


def test_smoke_gate_fails_when_quality_and_route_signals_collapse(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                _live_record(
                    "Q1",
                    response="Answer with no usable evidence.",
                    metadata={
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "unknown",
                        "retrieval_mode": "unknown",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "unknown", "graph": "unknown"},
                        "project_id": "project-1",
                        "filter_expr": 'paper_id == "paper-a"',
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"project_id": "project-1", "paper_ids": ["paper-a"]},
                        "citations": [],
                        "citations_count": 0,
                        "chunk_citations_count": 0,
                        "answer_confidence": "low",
                    },
                )
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
            },
            "records": [
                {
                    "faithfulness": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "answer_relevancy": 0.0,
                }
                for _ in records
            ],
        },
        gate_mode="smoke",
    )

    assert result["gate"]["passed"] is False
    checks = {item["name"]: item["passed"] for item in result["gate"]["checks"]}
    assert checks["smoke_ragas_metrics_not_collapsed"] is False
    assert checks["smoke_route_signal_not_collapsed"] is False
    warning_text = "\n".join(result["gate"]["warnings"])
    assert "unknown" in warning_text


def test_smoke_gate_fails_when_contract_metrics_break() -> None:
    report = {
        "sample_count": 1,
        "metrics": {
            "faithfulness": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.75,
            "answer_relevancy": 0.82,
        },
        "project_metrics": {
            "route_correctness": 1.0,
            "placeholder_citation_rate": 0.25,
            "scope_adherence": 0.5,
        },
    }

    gate = ragas_evals._build_gate_result(
        report,
        gate_mode="smoke",
        regression_tolerance=0.03,
    )

    checks = {item["name"]: item["passed"] for item in gate["checks"]}
    assert gate["passed"] is False
    assert checks["smoke_ragas_metrics_not_collapsed"] is True
    assert checks["smoke_route_signal_not_collapsed"] is True
    assert checks["smoke_placeholder_citations_absent"] is False
    assert checks["smoke_scope_contract_intact"] is False


def test_smoke_gate_fails_when_fact_contract_breaks() -> None:
    gate = ragas_evals._build_gate_result(
        {
            "sample_count": 1,
            "metrics": {"faithfulness": 0.8},
            "project_metrics": {"fact_contract_score": 0.5},
        },
        gate_mode="smoke",
        regression_tolerance=0.03,
    )

    checks = {item["name"]: item["passed"] for item in gate["checks"]}
    assert gate["passed"] is False
    assert checks["smoke_fact_contract_intact"] is False


def test_run_evaluation_computes_dialogue_and_synthesis_metrics(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["ctx"],
                    "response": "A1",
                    "reference_answer": "R1",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "skipped"},
                        "paper_ids": ["paper-a"],
                        "dialogue_mode": "lookup",
                        "requested_dialogue_mode": "lookup",
                        "synthesis_status": "ready",
                        "synthesis_state": {
                            "status": "ready",
                            "dialogue_mode": "lookup",
                            "project_scope": {"paper_ids": ["paper-a"]},
                            "paper_focus": {"mode": "single_paper"},
                            "evidence_summary": {"paper_count": 1, "citation_count": 1},
                            "dimension_status": {"methods": "supported"},
                            "supported_dimensions": ["methods"],
                            "unsupported_dimensions": [],
                            "open_questions": [],
                            "evidence_layers": {"milvus": "原文证据"},
                            "citations_count": 1,
                        },
                        "evidence_layers": {"milvus": "原文证据"},
                    },
                },
                {
                    "question": "Q2",
                    "retrieved_contexts": ["ctx"],
                    "response": "A2",
                    "reference_answer": "R2",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "skipped"},
                        "project_id": "project-1",
                        "dialogue_mode": "chat",
                    },
                },
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.8},
            "records": [{"faithfulness": 0.8} for _ in records],
        },
    )

    assert result["project_metrics"]["dialogue_mode_alignment"] == 1.0
    assert result["project_metrics"]["synthesis_payload_present"] == 1.0
    assert result["measurement_coverage"]["synthesis_payload_present"]["not_measured_count"] == 1


def test_dialogue_mode_alignment_does_not_infer_expected_mode_from_scope() -> None:
    metadata = {
        "project_id": "project-1",
        "paper_ids": ["paper-a"],
        "dialogue_mode": "chat",
    }

    assert ragas_evals._compute_dialogue_mode_alignment(metadata) == 1.0


def test_run_evaluation_accepts_inventory_scope_route(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "当前有哪些论文",
                    "retrieved_contexts": ["title: Paper A\npaper_id: paper-a\nabstract: Abstract A"],
                    "response": "当前项目范围内有一篇论文：Paper A。",
                    "reference_answer": "回答应列出项目范围内论文清单。",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "inventory",
                        "retrieval_mode": "inventory",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "unused", "graph": "unused", "inventory": "ok"},
                        "project_id": "project-1",
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"project_id": "project-1", "paper_ids": ["paper-a"]},
                        "paper_source_ids": ["paper-a"],
                        "paper_sources_count": 1,
                        "inventory_contexts_count": 1,
                        "retrieval_reasoning": "当前问题是在项目范围内列出论文清单，已直接读取项目作用域内的论文元数据作为回答依据。",
                        "answer_confidence": "medium",
                        "dialogue_mode": "research",
                        "requested_dialogue_mode": "research",
                        "citations": [],
                        "citations_count": 0,
                    },
                }
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.9},
            "records": [{"faithfulness": 0.9} for _ in records],
        },
    )

    assert result["project_metrics"]["route_correctness"] == 1.0
    assert result["project_metrics"]["scope_adherence"] == 1.0
    assert result["coverage_summary"]["payload"]["with_retrieval_reasoning_field"]["count"] == 1


def test_run_evaluation_prefers_milvus_route_when_inventory_was_only_a_prestep(tmp_path: Path) -> None:
    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "介绍下这两篇论文的方法部分",
                    "retrieved_contexts": ["ctx"],
                    "response": "回答基于原文方法段落。",
                    "reference_answer": "回答应覆盖两篇论文的方法。",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "unused", "inventory": "ok"},
                        "project_id": "project-1",
                        "paper_ids": ["paper-a", "paper-b"],
                        "resolved_scope": {"project_id": "project-1", "paper_ids": ["paper-a", "paper-b"]},
                        "paper_source_ids": ["paper-a", "paper-b"],
                        "inventory_contexts_count": 2,
                        "citations": [{"paper_id": "paper-a", "chunk_id": "chunk-1"}],
                        "citations_count": 1,
                        "chunk_citations_count": 1,
                        "retrieval_reasoning": "当前问题需要项目范围内的原文证据，已使用 Milvus 检索并展开父块全文。",
                        "answer_confidence": "medium",
                    },
                }
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        evaluator=lambda records: {
            "metrics": {"faithfulness": 0.9},
            "records": [{"faithfulness": 0.9} for _ in records],
        },
    )

    assert result["project_metrics"]["route_correctness"] == 1.0


def test_run_evaluation_builds_baseline_comparison_and_regression_gate(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline-report.json"
    baseline_path.write_text(
        json.dumps(
            {
                "run_id": "baseline-1",
                "metrics": {
                    "faithfulness": 0.95,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                    "answer_relevancy": 0.92,
                },
                "project_metrics": {
                    "scope_adherence": 1.0,
                    "citation_presence_rate": 1.0,
                    "placeholder_citation_rate": 0.0,
                    "route_correctness": 1.0,
                    "answer_payload_present": 1.0,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        adapter_name = "live"

        def load(self, dataset_path):
            del dataset_path
            return [
                {
                    "question": "Q1",
                    "retrieved_contexts": ["ctx"],
                    "response": "Answer with citation [paper-a].",
                    "reference_answer": "A1",
                    "metadata": {
                        "adapter": "live",
                        "requested_mode": "auto",
                        "resolved_mode": "milvus",
                        "retrieval_mode": "milvus",
                        "graph_mode": "hybrid",
                        "source_status": {"milvus": "ok", "graph": "skipped"},
                        "paper_ids": ["paper-a"],
                        "resolved_scope": {"paper_ids": ["paper-a"]},
                        "citations": [{"paper_id": "paper-a"}],
                        "citations_count": 1,
                        "chunk_citations_count": 1,
                        "answer_confidence": "medium",
                    },
                }
            ]

        def describe(self):
            return {"adapter": "live", "mode": "auto", "graph_mode": "hybrid"}

    result = ragas_evals.run_evaluation(
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "experiments",
        adapter=FakeAdapter(),
        baseline_report_path=baseline_path,
        gate_mode="regression",
        regression_tolerance=0.05,
        evaluator=lambda records: {
            "metrics": {
                "faithfulness": 0.7,
                "context_precision": 0.82,
                "context_recall": 0.81,
                "answer_relevancy": 0.85,
            },
            "records": [
                {
                    "faithfulness": 0.7,
                    "context_precision": 0.82,
                    "context_recall": 0.81,
                    "answer_relevancy": 0.85,
                }
                for _ in records
            ],
        },
    )

    assert result["baseline_comparison"]["baseline_run_id"] == "baseline-1"
    assert result["baseline_comparison"]["metrics"]["faithfulness"]["delta"] == -0.25
    assert result["gate"]["passed"] is False
    failed_checks = [item for item in result["gate"]["checks"] if not item["passed"]]
    assert any(item["name"] == "baseline_regression_within_tolerance" for item in failed_checks)


def test_run_evaluation_suite_writes_suite_report(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "wave8-live-regression",
                "datasets": [
                    {"name": "dataset-a", "path": "dataset-a.jsonl", "tags": ["live", "smoke"]},
                    {"name": "dataset-b", "path": "dataset-b.jsonl", "tags": ["live", "scope"]},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_run_evaluation(*, dataset_path, output_dir, adapter, dataset_name, dataset_tags, gate_mode, regression_tolerance, **kwargs):
        del output_dir, adapter, kwargs
        assert gate_mode == "smoke"
        assert regression_tolerance == 0.03
        warnings = [f"{dataset_name} quality warning"] if dataset_name == "dataset-b" else []
        return {
            "run_id": f"run-{dataset_name}",
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "dataset_tags": dataset_tags,
            "adapter": "live",
            "mode": "auto",
            "graph_mode": "hybrid",
            "sample_count": 1,
            "metrics": {
                "faithfulness": 0.8,
                "context_precision": 0.75,
                "context_recall": 0.78,
                "answer_relevancy": 0.81,
            },
            "project_metrics": {
                "scope_adherence": 1.0,
                "citation_presence_rate": 1.0,
                "placeholder_citation_rate": 0.0,
                "route_correctness": 1.0,
                "answer_payload_present": 1.0,
            },
            "coverage_summary": {
                "sample_count": 1,
                "route": {
                    "requested_mode_counts": {"auto": 1},
                    "resolved_mode_counts": {"milvus": 1},
                    "graph_mode_counts": {"hybrid": 1},
                    "route_metadata_complete": {"count": 1, "rate": 1.0},
                },
                "scope": {
                    "with_scope_constraint": {"count": 1, "rate": 1.0},
                    "with_project_scope": {"count": 0, "rate": 0.0},
                    "with_filter_scope": {"count": 0, "rate": 0.0},
                    "with_paper_scope": {"count": 1, "rate": 1.0},
                    "scope_metadata_complete": {"count": 1, "rate": 1.0},
                },
                "citation": {
                    "with_citations_field": {"count": 1, "rate": 1.0},
                    "with_citations_count_field": {"count": 1, "rate": 1.0},
                    "with_chunk_citations_count_field": {"count": 1, "rate": 1.0},
                    "with_structured_citations": {"count": 1, "rate": 1.0},
                    "with_non_placeholder_citation_signal": {"count": 1, "rate": 1.0},
                    "citation_metadata_complete": {"count": 1, "rate": 1.0},
                },
                "payload": {
                    "with_answer_confidence_field": {"count": 1, "rate": 1.0},
                    "with_answer_confidence_value": {"count": 1, "rate": 1.0},
                    "with_source_status_field": {"count": 1, "rate": 1.0},
                    "with_retrieval_reasoning_field": {"count": 0, "rate": 0.0},
                    "payload_metadata_complete": {"count": 1, "rate": 1.0},
                },
            },
            "gate": {"mode": "smoke", "passed": True, "checks": [], "warnings": warnings},
            "json_report_path": str(Path(tmp_path / f"{dataset_name}.json")),
            "markdown_report_path": str(Path(tmp_path / f"{dataset_name}.md")),
            "output_dir": str(tmp_path / dataset_name),
        }

    monkeypatch.setattr(ragas_evals, "run_evaluation", fake_run_evaluation)

    result = ragas_evals.run_evaluation_suite(
        manifest_path=manifest_path,
        output_dir=tmp_path / "experiments",
        gate_mode="smoke",
    )

    assert result["suite_name"] == "wave8-live-regression"
    assert result["dataset_count"] == 2
    assert result["sample_count"] == 2
    assert result["metrics"]["faithfulness"] == 0.8
    assert result["gate"]["passed"] is True
    assert any("dataset-b" in warning for warning in result["gate"]["warnings"])
    assert Path(result["json_report_path"]).exists()
    assert result["coverage_summary"]["route"]["route_metadata_complete"]["count"] == 2


def test_suite_smoke_gate_fails_when_aggregate_quality_signals_collapse() -> None:
    report = {
        "dataset_count": 1,
        "datasets": [{"dataset_name": "legacy-dataset", "gate": {"passed": True}}],
        "metrics": {
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_relevancy": 0.0,
        },
        "project_metrics": {"route_correctness": 0.0},
        "coverage_summary": {"route": {"resolved_mode_counts": {"unknown": 1}}},
    }

    gate = ragas_evals._build_suite_gate_result(
        report,
        gate_mode="smoke",
        regression_tolerance=0.03,
    )

    checks = {item["name"]: item["passed"] for item in gate["checks"]}
    assert gate["passed"] is False
    assert checks["dataset_gates_passed"] is True
    assert checks["smoke_ragas_metrics_not_collapsed"] is False
    assert checks["smoke_route_signal_not_collapsed"] is False
    assert "unknown" in "\n".join(gate["warnings"])
