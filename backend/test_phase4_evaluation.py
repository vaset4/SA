from __future__ import annotations

import asyncio

import pytest

import app.evaluation.eval_pipeline as eval_pipeline
import app.evaluation.report as report_module


def test_build_evaluation_rows_collects_pipeline_output(monkeypatch) -> None:
    dataset = [{"question": "What is RAG?", "ground_truth": "retrieval augmented generation"}]

    async def fake_run_rag_pipeline(question: str) -> dict[str, object]:
        return {
            "question": question,
            "answer": "A generation answer",
            "contexts": ["ctx-1", "ctx-2"],
        }

    monkeypatch.setattr(eval_pipeline, "load_golden_dataset", lambda dataset_path=None: dataset)
    monkeypatch.setattr(eval_pipeline, "run_rag_pipeline", fake_run_rag_pipeline)

    rows = asyncio.run(eval_pipeline.build_evaluation_rows())

    assert rows == [
        {
            "question": "What is RAG?",
            "ground_truth": "retrieval augmented generation",
            "answer": "A generation answer",
            "contexts": ["ctx-1", "ctx-2"],
        }
    ]


def test_render_evaluation_report_fails_when_metric_missing() -> None:
    with pytest.raises(ValueError, match="faithfulness"):
        report_module.render_evaluation_report(
            {
                "rows": [{"question": "q1"}],
                "scores": {
                    "context_precision": [0.8],
                    "context_recall": [0.7],
                    "answer_relevancy": [0.9],
                },
            }
        )


def test_run_evaluation_can_include_report(monkeypatch) -> None:
    rows = [
        {
            "question": "What is RAG?",
            "ground_truth": "retrieval augmented generation",
            "answer": "A generation answer",
            "contexts": ["ctx-1"],
        }
    ]
    score_dict = {
        "context_precision": [0.9],
        "context_recall": [0.8],
        "faithfulness": [0.85],
        "answer_relevancy": [0.95],
    }

    class _FakeDataset:
        @staticmethod
        def from_list(items: list[dict[str, object]]) -> list[dict[str, object]]:
            return items

    class _FakeFrame:
        def __init__(self, data: dict[str, list[float]]) -> None:
            self._data = data

        def to_dict(self, orient: str) -> dict[str, list[float]]:
            assert orient == "list"
            return self._data

    class _FakeScores:
        def __init__(self, data: dict[str, list[float]]) -> None:
            self._data = data

        def to_pandas(self) -> _FakeFrame:
            return _FakeFrame(self._data)

    def fake_require_ragas():
        def fake_evaluate(dataset: list[dict[str, object]], metrics: list[object]) -> _FakeScores:
            assert dataset == rows
            assert len(metrics) == 4
            return _FakeScores(score_dict)

        return (
            _FakeDataset,
            fake_evaluate,
            "context_precision_metric",
            "context_recall_metric",
            "faithfulness_metric",
            "answer_relevancy_metric",
        )

    async def fake_build_rows(dataset_path: str | None = None) -> list[dict[str, object]]:
        return rows

    monkeypatch.setattr(eval_pipeline, "_require_ragas", fake_require_ragas)
    monkeypatch.setattr(eval_pipeline, "build_evaluation_rows", fake_build_rows)

    result = asyncio.run(eval_pipeline.run_evaluation(include_report=True))

    assert result["rows"] == rows
    assert result["scores"] == score_dict
    assert "样本数: 1" in result["report"]
    assert "faithfulness: 0.8500" in result["report"]
