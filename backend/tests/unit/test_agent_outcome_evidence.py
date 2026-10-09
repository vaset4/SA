"""验证实际原文覆盖、最终出处及诚实失败的任务评分。作者：xxx。"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from evals.agent_outcome.contracts import agent_record_contracts, load_agent_case
from evals.agent_outcome.evidence import collect_episode_facts
from evals.agent_outcome.runner import run_episode_sync
from evals.agent_outcome.scoring import grade_agent_record
from evals.contracts import snapshot
from test_agent_outcome_runner import _SuccessGraph, _environment, _record


def _captured() -> dict:
    """无参数；运行固定测试图，返回独立实际记录。"""
    return run_episode_sync(load_agent_case(_record()), _environment(_SuccessGraph))


def _changed_values(record: dict, patch: dict) -> dict:
    """参数为测试记录及实际载荷变化；同步通用和 Agent 观察以形成完整测试输入。"""
    changed = deepcopy(record)
    observed = json.loads(changed["agent_run"]["observation_snapshot"])
    values = json.loads(observed["turns"][0]["values_snapshot"])
    values.update(patch)
    observed["turns"][0]["values_snapshot"] = snapshot(values)
    for key in ("agent_run", "run_record"):
        changed[key]["observation_snapshot"] = snapshot(observed)
    return changed


def _metrics(record: dict) -> dict:
    """参数为测试记录；返回按指标名称索引的实际评分。"""
    return {item["name"]: item for item in grade_agent_record(record)["measurements"]}


def test_partial_original_text_is_measured_failure() -> None:
    """无参数；实际只取得比例而没有理由时，完整任务不能成功。"""
    record = _changed_values(_captured(), {"evidence_records": [
        {"evidence_id": "e1", "paper_id": "lsttn", "chunk_id": "chunk-1", "excerpt": "75% random masking"},
    ]})
    facts = collect_episode_facts(*agent_record_contracts(record))
    assert facts.evidence.status == "measured"
    assert facts.missing_spans == {"reason"}
    assert _metrics(record)["task_success"]["value"] == 0.0


@pytest.mark.parametrize("citation", [{"evidence_id": "not-returned"}, {"evidence_id": "e1", "paper_id": "another-paper"}, {"evidence_id": "e1", "page_num": 99}])
def test_nonempty_citation_requires_actual_binding(citation: dict) -> None:
    """参数为不存在或错论文的引用；非空引用必须通过真实出处关联。"""
    scores = _metrics(_changed_values(_captured(), {"citations": [citation]}))
    assert scores["citation_verifiability_rate"]["value"] == 0.0
    assert scores["task_success"]["value"] == 0.0


def test_complete_empty_observation_differs_from_missing_observation() -> None:
    """无参数；成功空证据是零分，没有采集到证据字段仍是 missing。"""
    empty = _changed_values(_captured(), {"evidence_records": [], "citations": []})
    assert _metrics(empty)["task_success"]["value"] == 0.0
    missing = _changed_values(_captured(), {"evidence_records": None, "citations": []})
    assert _metrics(missing)["task_success"]["status"] == "missing"


def test_unknown_mapping_is_not_a_measured_zero() -> None:
    """无参数；实际片段缺少原文映射时显示缺口，不能替换成检索零分。"""
    record = _changed_values(_captured(), {"evidence_records": [
        {"evidence_id": "e1", "paper_id": "lsttn", "chunk_id": "unknown", "excerpt": "some text"},
    ]})
    assert _metrics(record)["task_success"]["status"] == "missing"


def test_bare_exception_is_not_honest_failure_success() -> None:
    """无参数；进程失败且没有向用户说明限制，不能自动给失败处理满分。"""
    from test_agent_outcome_runner import _FailureGraph

    record = run_episode_sync(load_agent_case(_record()), _environment(_FailureGraph))
    assert _metrics(record)["failure_handling_correctness"]["value"] == 0.0


@pytest.mark.parametrize("linked_unit,expected", [("failed-unit", 1.0), ("unrelated-unit", 0.0)])
def test_recovery_requires_actual_failed_unit_relationship(linked_unit: str, expected: float) -> None:
    """参数为实际修复关联和期望；相同补查动作只有连回失败单元才构成恢复。"""
    from app.middleware.trace_context import evidence_scope, record_evidence_observation
    from test_agent_outcome_runner import _ACTUAL_TEXT

    class RecoveryGraph(_SuccessGraph):
        """真正执行失败及修复事件的可控图，不使用测试专属 related_to_failure 标签。"""

        async def astream(self, payload, *, stream_mode):
            """参数为实际调用；记录失败、补查及重执行后再输出真实形状的证据。"""
            try:
                with evidence_scope("planner_unit", unit_id="failed-unit"):
                    raise TimeoutError("transient source failure")
            except TimeoutError:
                pass
            records = [{"evidence_id": "e1", "paper_id": "lsttn", "chunk_id": "chunk-1", "excerpt": _ACTUAL_TEXT}]
            record_evidence_observation("repair_evidence_pool", {"records": records, "linked_unit_ids": [linked_unit]})
            with evidence_scope("repair_rerun", unit_id="failed-unit"):
                record_evidence_observation("rerun_evidence_pool", {"records": records})
            async for event in super().astream(payload, stream_mode=stream_mode):
                yield event

    raw = _record()
    raw["fault_injection"] = {"recoverable": True}
    record = run_episode_sync(load_agent_case(raw), _environment(RecoveryGraph))
    scores = _metrics(record)
    assert scores["task_success"]["value"] == 1.0
    assert scores["recovery_success"]["value"] == expected


def test_missing_second_paper_cannot_complete_cross_paper_task() -> None:
    """无参数；两篇目标只取得第一篇时，不能用任一论文交集判定全部完成。"""
    from evals.agent_outcome.environment import AgentEnvironment
    from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan, SourceBlock, make_source_snapshot
    from evals.evidence.corpus import build_index_from_chunks
    from evals.contracts import snapshot_hash
    from test_agent_outcome_runner import _ACTUAL_TEXT

    raw = _record()
    benchmark = EvidenceBenchmark.model_validate(raw["evidence_benchmark"])
    source = make_source_snapshot(paper_id="second", pdf_sha256=snapshot_hash("second-pdf"), artifact_manifest_snapshot="{}",
                                  blocks=(SourceBlock(block_id="body", page_num=1, section_path="", text="Second required evidence"),))
    span = EvidenceSpan(span_id="second-span", source_snapshot_id=source.snapshot_id, block_id="body", start=0,
                        end=len(source.blocks[0].text), quote=source.blocks[0].text, quote_sha256=snapshot_hash(source.blocks[0].text))
    sources = (*benchmark.sources, source)
    index, issues = build_index_from_chunks(sources, [
        {"paper_id": "lsttn", "chunk_id": "chunk-1", "text": _ACTUAL_TEXT},
        {"paper_id": "second", "chunk_id": "chunk-2", "text": source.blocks[0].text},
    ], retrieval_backend="fixture")
    assert not issues
    body = benchmark.model_dump(mode="json")
    body.update(sources=[item.model_dump(mode="json") for item in sources], index_snapshots=[index.model_dump(mode="json")],
                spans=[*body["spans"], span.model_dump(mode="json")], target_paper_ids=["lsttn", "second"])
    body["groups"].append({"group_id": "second-group", "importance": "critical", "proofs": [{"span_ids": ["second-span"]}]})
    raw["evidence_benchmark"] = body
    raw["evidence_targets"] = {"required_span_ids": ["ratio", "reason", "second-span"], "required_group_ids": ["masking", "second-group"], "required_paper_ids": ["lsttn", "second"]}
    raw["input"]["paper_keys"] = ["lsttn", "second"]
    raw["input"]["project_scope"]["paper_ids"] = ["lsttn", "second"]
    environment = AgentEnvironment(graph_factory=_SuccessGraph, declared={"index_snapshot_id": index.index_snapshot_id})
    record = run_episode_sync(load_agent_case(raw), environment)
    scores = _metrics(record)
    assert scores["task_success"]["status"] == "measured" and scores["task_success"]["value"] == 0.0
