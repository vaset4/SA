"""Offline shape tests for research planner replay eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evals.research_planner import run_eval  # noqa: E402
from evals.research_planner.live_fixture import build_live_records  # noqa: E402
from evals.research_planner.metrics import (  # noqa: E402
    CATEGORY_NAMES,
    DEPTH_ORDER,
    FAILURE_STAGES,
    METRIC_NAMES,
    MINIMAL_DEPTH,
    NOT_MEASURED,
    score_record,
)
from app.models.research_contracts import VALID_TURN_DEPTHS  # noqa: E402

LEGACY_RAGAS_METRICS = {
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
}
REQUIRED_FAILURE_STAGES = {
    "planning",
    "validation",
    "execution",
    "synthesis",
    "evidence_absence",
}
# 本单元新增的五项指标；现有 20 条样本都没有它们要的标注与运行时字段
DEPTH_AND_MEMORY_METRICS = (
    "depth_judgment_accuracy",
    "depth_escalation_recall",
    "unnecessary_retrieval_rate",
    "memory_injection_compliance_rate",
    "stale_memory_leak_rate",
)
# 四道资格门全过的记忆条目形状，用例按需覆盖其中一个轴来造不合格条目
QUALIFIED_MEMORY_ITEM = {
    "memory_id": "m1",
    "user_id": "owner-1",
    "review_state": "accepted",
    "status": "confirmed",
    "stale_state": "fresh",
}
MEMORY_OWNER = "owner-1"
# 现造一条没有深度标注、也不带记忆的样本，用来锁「无数据不许出分数」的判法
UNANNOTATED_RECORD = {
    "id": "unannotated-001",
    "category": "single_paper_lookup",
    "question": "这条样本没有深度标注。",
    "reference_answer": "无标注样本，只用来验哨兵取值。",
    "golden_evidence": ["unannotated-evidence"],
    "critical_facts": [{"name": "none", "must_include": [], "must_not_include": []}],
    "rubric": {"requires_citation": True},
    "expected": {"failure_stage": "none"},
    "replay": {"answer": "", "research_plan": {"units": []}, "research_unit_results": []},
}


def test_eval_dataset_shape() -> None:
    records = run_eval.load_dataset()

    assert len(records) >= run_eval.MIN_RECORDS
    assert {record["category"] for record in records} == set(CATEGORY_NAMES)
    for record in records:
        # minimal 回合本轮不检索论文，金标证据与关键事实无从给起；其余深度必须给全，
        # 否则「答得对不对」没有可对照的标准
        if str(record["expected"].get("depth") or "") != MINIMAL_DEPTH:
            assert record["golden_evidence"], record["id"]
            assert record["critical_facts"], record["id"]
        assert isinstance(record["golden_evidence"], list)
        assert isinstance(record["critical_facts"], list)
        assert isinstance(record["rubric"], dict)
        assert isinstance(record["expected"], dict)
        assert isinstance(record["replay"], dict)


def test_eval_report_has_all_metrics(tmp_path: Path) -> None:
    report = run_eval.run_evaluation(output_dir=tmp_path, write_report=True)

    assert set(METRIC_NAMES).issubset(report["metrics"])
    assert set(METRIC_NAMES).issubset(report["planner_metrics"])
    assert Path(report["json_report_path"]).exists()
    assert Path(report["markdown_report_path"]).exists()
    assert "Research Planner Replay Eval" in Path(report["markdown_report_path"]).read_text(encoding="utf-8")


def test_eval_metadata_failure_stage() -> None:
    """无参数；保留实际阶段和预期阶段差异，不能用预期失败伪造覆盖。"""
    report = run_eval.run_evaluation()
    stages = {record["failure_stage"] for record in report["records"]}

    assert stages.issubset(set(FAILURE_STAGES))
    assert REQUIRED_FAILURE_STAGES.issubset({record["expected_failure_stage"] for record in report["records"]})
    mismatches = []
    for record in report["records"]:
        if record["failure_stage"] != record["expected_failure_stage"]:
            mismatches.append(record)
            measurement = next(item for item in record["grading_result"]["measurements"]
                               if item["name"] == "planner_failure_visibility")
            assert measurement["value"] != 1.0
            assert measurement["value"] == 0.0 or measurement["status"] == "missing"
    assert {"validation", "synthesis"} <= {record["expected_failure_stage"] for record in mismatches}


def test_eval_report_stays_separate_from_legacy_ragas_metrics() -> None:
    report = run_eval.run_evaluation()

    assert report["adapter"] == "planner_live_fixture"
    assert "feature_flag" not in report
    assert not (set(report["metrics"]) & LEGACY_RAGAS_METRICS)


def test_eval_runner_uses_live_planner_branch(monkeypatch) -> None:
    calls = []

    def fake_live_records(records):
        calls.append(len(records))
        return [{**record, "replay": record["replay"]} for record in records]

    monkeypatch.setattr(run_eval, "build_live_records", fake_live_records)
    report = run_eval.run_evaluation()

    assert calls == [report["sample_count"]]


def _scored(expected: dict, replay: dict) -> dict:
    """按给定标注与运行时载荷打一条样本的分，返回指标名 → 取值。"""
    return score_record({"id": "case", "expected": expected, "replay": replay})


def test_depth_and_memory_metrics_reach_report(tmp_path: Path) -> None:
    """五项新指标进注册表，并且在 json 与 markdown 报告里都取得到。"""
    report = run_eval.run_evaluation(output_dir=tmp_path, write_report=True)
    markdown = Path(report["markdown_report_path"]).read_text(encoding="utf-8")

    for name in DEPTH_AND_MEMORY_METRICS:
        assert name in METRIC_NAMES
        assert name in report["metrics"]
        assert f"- {name}: " in markdown


def test_unannotated_metrics_report_not_measured(tmp_path: Path) -> None:
    """
    样本没有深度标注、本轮也没读到记忆时，五项指标必须报「没测」而不是分数。

    这条锁的是「无数据不许伪装成结论」：回 0.0 是假红、回 1.0 是假绿，
    报告里都会被当成深度策略或记忆门的真实表现。正式数据集已经标注过，
    所以这里现造一条没标注的样本来锁判法。
    """
    dataset = tmp_path / "unannotated.jsonl"
    dataset.write_text(json.dumps(UNANNOTATED_RECORD, ensure_ascii=False) + "\n", encoding="utf-8")
    report = run_eval.run_evaluation(dataset_path=dataset, output_dir=tmp_path, write_report=True)
    markdown = Path(report["markdown_report_path"]).read_text(encoding="utf-8")

    # 1. 哨兵取值本身不能是数，否则聚合与阈值判断会把它当成绩看
    assert not isinstance(NOT_MEASURED, (int, float))
    for name in DEPTH_AND_MEMORY_METRICS:
        # 2. 聚合值与逐条取值都得是哨兵，不能有任何一条悄悄给出 0.0 / 1.0
        assert report["metrics"][name] == NOT_MEASURED, name
        assert {record[name] for record in report["records"]} == {NOT_MEASURED}
        # 3. markdown 里也得看得见「没测」，不能渲染成 0.0
        assert f"- {name}: {NOT_MEASURED}" in markdown


def test_annotated_dataset_measures_depth_and_memory_metrics(tmp_path: Path) -> None:
    """标注齐了以后四项指标真出数：这就是 V4 加硬闸门的开工前置信号。

    深度升级召回率不在其中：产品目前没有「循环里把深度调深」这个行为，
    没有行为就没有可召回的东西，见任务交接 §6 的 S 段。
    """
    report = run_eval.run_evaluation(output_dir=tmp_path)

    assert report["metrics"]["depth_judgment_accuracy"] != NOT_MEASURED
    # minimal 样本一次论文检索都不该发生：V1 的直答短路在评测里也得成立
    assert report["metrics"]["unnecessary_retrieval_rate"] == 0.0
    # 合规率能出数本身就意味着进上下文的条目带齐了四道门的轴
    assert report["metrics"]["memory_injection_compliance_rate"] == 1.0
    # 0.0 而不是「没测」：样本里放了失效条目，是门把它挡住了，不是没数据
    assert report["metrics"]["stale_memory_leak_rate"] == 0.0


def test_memory_gates_keep_unqualified_rows_out_of_context() -> None:
    """失效、未接纳、他人的记忆行都进不了上下文，进去的每条都带着四道门的取值。"""
    record = next(item for item in run_eval.load_dataset() if item["id"] == "rp-memory-001")

    live = build_live_records([record])
    items = live[0]["replay"]["memory_read_result"]["items"]

    assert [item["memory_id"] for item in items] == ["mem-pdformer-attn", "mem-pdformer-transfer"]
    for item in items:
        assert item["user_id"] == record["expected"]["memory_owner_user_id"]
        assert (item["review_state"], item["status"], item["stale_state"]) == ("accepted", "confirmed", "fresh")


def _paper_claim(evidence_ids: list[str], *, validation_status: str = "validated") -> dict:
    """一条论文事实主张。"""
    return {
        "claim_type": "paper_supported",
        "text": "fixture paper fact",
        "supporting_evidence_ids": evidence_ids,
        "validation_status": validation_status,
    }


def _replay_with_claim(claim: dict, *, usable_as_citation: bool = True) -> dict:
    """带一条本轮 Evidence 与一条主张的运行时载荷。"""
    return {
        "evidence_records": [{"evidence_id": "e1", "usable_as_citation": usable_as_citation}],
        "research_unit_results": [{"claims": [claim]}],
    }


def test_citation_verifiability_counts_only_paper_fact_claims() -> None:
    """分母只认必须带论文引用的那类主张；本回合没有这类主张时报「没测」而不是满分。"""
    verifiable = _scored({}, _replay_with_claim(_paper_claim(["e1"])))
    inference_only = _scored({}, _replay_with_claim({
        "claim_type": "evidence_grounded_inference",
        "source_unit_ids": ["u1"],
        "reasoning_summary": "compared two units",
        "validation_status": "validated",
    }))

    assert verifiable["citation_verifiability_rate"] == 1.0
    # 一条引用都没核过就给 1.0 是假绿，硬闸门会因此在这类回合永远绿着
    assert inference_only["citation_verifiability_rate"] == NOT_MEASURED


def test_citation_verifiability_rejects_the_three_unverifiable_shapes() -> None:
    """空引用、引用不在本轮 Evidence、命中不可作引用，三种都判 0。"""
    empty = _scored({}, _replay_with_claim(_paper_claim([])))
    ghost = _scored({}, _replay_with_claim(_paper_claim(["ghost-evidence"])))
    not_citable = _scored({}, _replay_with_claim(_paper_claim(["e1"]), usable_as_citation=False))
    unvalidated = _scored({}, _replay_with_claim(_paper_claim(["e1"], validation_status="needs_validation")))

    assert empty["citation_verifiability_rate"] == 0.0
    assert ghost["citation_verifiability_rate"] == 0.0
    assert not_citable["citation_verifiability_rate"] == 0.0
    assert unvalidated["citation_verifiability_rate"] == 0.0


def test_compliance_gate_passes_only_when_all_three_hold() -> None:
    """三项合规指标各自达标才放行；任一越界即不通过，并指出越界的是哪一项。"""
    green = {
        "citation_verifiability_rate": 1.0,
        "memory_injection_compliance_rate": 1.0,
        "stale_memory_leak_rate": 0.0,
    }

    assert run_eval.evaluate_compliance_gate(green)["passed"] is True
    for name, bad_value in (
        ("citation_verifiability_rate", 0.9),
        ("memory_injection_compliance_rate", 0.5),
        ("stale_memory_leak_rate", 0.25),
    ):
        result = run_eval.evaluate_compliance_gate({**green, name: bad_value})
        assert result["passed"] is False, name
        assert [violation["metric"] for violation in result["violations"]] == [name]
        assert result["violations"][0]["reason"] == "threshold"


def test_compliance_gate_treats_not_measured_as_failure() -> None:
    """合规类测不出一律判不通过：把「没测」当合格放过，闸门会永远绿着。"""
    result = run_eval.evaluate_compliance_gate({
        "citation_verifiability_rate": NOT_MEASURED,
        "memory_injection_compliance_rate": 1.0,
        "stale_memory_leak_rate": 0.0,
    })

    assert result["passed"] is False
    assert result["violations"][0]["reason"] == "not_measured"


def test_accuracy_metrics_are_not_gated() -> None:
    """准确率类不进闸门（prd.md §13.3 只记基线），全 0 也不许因此判失败。"""
    metrics = {
        "citation_verifiability_rate": 1.0,
        "memory_injection_compliance_rate": 1.0,
        "stale_memory_leak_rate": 0.0,
        "depth_judgment_accuracy": 0.0,
        "plan_valid_rate": 0.0,
        "depth_escalation_recall": NOT_MEASURED,
    }

    assert run_eval.evaluate_compliance_gate(metrics)["passed"] is True


def test_current_dataset_reports_incomplete_compliance_observations(tmp_path: Path) -> None:
    """参数为临时目录；现有样本的局部数据缺口不能被其他样本满分掩盖。"""
    report = run_eval.run_evaluation(output_dir=tmp_path)

    assert report["compliance_gate"]["passed"] is False
    assert report["compliance_gate"]["violations"]
    for violation in report["compliance_gate"]["violations"]:
        assert violation["reason"] == "incomplete_measurement"
        assert violation["actual"]["missing_count"] > 0


def test_annotated_depth_metrics_produce_numbers() -> None:
    """有了 expected.depth 与运行时深度字段，三项深度指标立刻开始出数，不用改代码。"""
    plan = {"research_plan": {"units": [{"evidence_policy": {"source_type": "local_direct_text"}}]}}

    hit = _scored({"depth": "lookup"}, {"turn_depth": "lookup"})
    miss = _scored({"depth": "minimal"}, {"turn_outcome": {"final_depth": "research"}})
    minimal_with_retrieval = _scored({"depth": "minimal"}, plan)

    assert hit["depth_judgment_accuracy"] == 1.0
    assert miss["depth_judgment_accuracy"] == 0.0
    assert minimal_with_retrieval["unnecessary_retrieval_rate"] == 1.0
    # 标注不是 minimal 的样本不进这条的分母
    assert hit["unnecessary_retrieval_rate"] == NOT_MEASURED


def test_depth_escalation_separates_no_record_from_no_escalation() -> None:
    """
    depth_transitions 键不在＝没数据，键在而为空＝本回合确实没升级，两者不许归并。

    当前实现里没有任何代码在循环内改深度，所以空数组这一态是要能被看见的真红。
    """
    escalated = {"depth_transitions": [{"from_depth": "lookup", "to_depth": "research"}]}
    downgraded = {"depth_transitions": [{"from_depth": "research", "to_depth": "lookup"}]}
    annotation = {"should_escalate_depth": True}

    assert _scored(annotation, escalated)["depth_escalation_recall"] == 1.0
    assert _scored(annotation, downgraded)["depth_escalation_recall"] == 0.0
    assert _scored(annotation, {"depth_transitions": []})["depth_escalation_recall"] == 0.0
    assert _scored(annotation, {})["depth_escalation_recall"] == NOT_MEASURED
    # 没标注「应当升级」的样本不进分母
    assert _scored({}, escalated)["depth_escalation_recall"] == NOT_MEASURED


def test_memory_metrics_need_per_item_gate_axes() -> None:
    """
    记忆两项指标按进上下文的条目逐条核验，缺轴或一条都没进上下文时报「没测」。

    读空项目报 0 泄漏、报 100% 合规都是假绿：一条都没核验过的硬闸门等于没人看守。
    """
    stale_item = {**QUALIFIED_MEMORY_ITEM, "stale_state": "stale"}
    annotation = {"memory_owner_user_id": MEMORY_OWNER}
    qualified = _scored(annotation, {"memory_read_result": {"items": [QUALIFIED_MEMORY_ITEM]}})
    leaked = _scored(annotation, {"memory_read_result": {"items": [stale_item]}})
    projected = _scored(annotation, {"memory_read_result": {"items": [{"memory_id": "m1"}]}})
    empty_read = _scored(annotation, {"memory_read_result": {"status": "empty", "items": []}})

    assert qualified["memory_injection_compliance_rate"] == 1.0
    assert qualified["stale_memory_leak_rate"] == 0.0
    assert leaked["memory_injection_compliance_rate"] == 0.0
    assert leaked["stale_memory_leak_rate"] == 1.0
    # 只带 memory_id 的投影条目核验不了四道门，不能算它合规
    assert projected["memory_injection_compliance_rate"] == NOT_MEASURED
    assert projected["stale_memory_leak_rate"] == NOT_MEASURED
    assert empty_read["memory_injection_compliance_rate"] == NOT_MEASURED
    assert empty_read["stale_memory_leak_rate"] == NOT_MEASURED


def test_memory_compliance_checks_owner_gate() -> None:
    """归属人那道门也逐条核：同项目里别人的合格记忆进了上下文就是不合规。"""
    intruder = {**QUALIFIED_MEMORY_ITEM, "user_id": "other-user"}
    replay = {"memory_read_result": {"items": [QUALIFIED_MEMORY_ITEM, intruder]}}

    scored = _scored({"memory_owner_user_id": MEMORY_OWNER}, replay)

    assert scored["memory_injection_compliance_rate"] == 0.5
    # 归属人期望值没标注时无从核验第四道门，不许拿三道门的结果冒充合规率
    assert _scored({}, replay)["memory_injection_compliance_rate"] == NOT_MEASURED


def test_depth_vocabulary_matches_runtime_contract() -> None:
    """
    评测侧的深度档位词表与运行时契约同源。

    词表漂了会静默出事：运行时新增一档深度，评测这边照旧只认三档，
    合法标注反而在读样本时被拒。
    """
    assert DEPTH_ORDER == VALID_TURN_DEPTHS


def test_invalid_expected_depth_annotation_is_rejected(tmp_path: Path) -> None:
    """预期深度写错档位在读样本时就报错，不让它变成整列 0.0 的假红。"""
    record = run_eval.load_dataset()[0]
    record["expected"] = {**record["expected"], "depth": "deep"}
    dataset_path = tmp_path / "bad_depth.jsonl"
    dataset_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid expected depth"):
        run_eval.load_dataset(dataset_path)
