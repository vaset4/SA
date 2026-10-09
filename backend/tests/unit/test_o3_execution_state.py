"""O3 执行接续与共享取证的离线回归，作者：xxx。"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
from app.config import get_settings
from app.models.research_planner_contracts import ResearchClaimDraft, ResearchPlan, ResearchUnit
from app.services.internal_research.research_planner_executor import (
    PlannerExecutionContext, ResearchPlannerExecutor, read_evidence,
)
from app.services.internal_research.research_planner_execution_state import ExecutionIdentity, PlannerExecutionState
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from o2_evidence_test_utils import direct_evidence, read_result
from research_planner_test_utils import make_draft, make_unit, supporting_spans, explicit_claim_provider


def test_completed_read_survives_reasoning_failure() -> None:
    """无参数；推理异常保留原身份，并携带先前已完成的读取快照。"""
    failure = RuntimeError("reasoner failed")
    record = direct_evidence()

    async def adapter(unit, context):
        """参数为单元和上下文；返回已经成功的正文读取。"""
        return read_result([record])

    async def provider(*args, **kwargs):
        """参数为推理输入；抛出固定故障以检查失败身份。"""
        raise failure

    executor = ResearchPlannerExecutor(
        adapters={"local_direct_text": adapter},
        reasoner=ResearchUnitReasoner(draft_provider=provider),
    )
    plan = ResearchPlan.model_validate(make_draft())
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(executor.execute_with_evidence(plan, PlannerExecutionContext()))
    assert caught.value is failure
    snapshot = caught.value.planner_execution_report
    assert snapshot.evidence_records == [record]
    assert snapshot.read_receipts[0]["evidence_ids"] == [record["evidence_id"]]
    assert snapshot.results == []


def test_legacy_plan_cannot_execute_or_call_an_adapter() -> None:
    """无参数；旧合同只能回读历史，公共执行入口必须在任何取证前明确拒绝。"""
    calls = []

    async def adapter(unit, context):
        """参数为单元和上下文；若意外被调用则留下可核验记录。"""
        calls.append(unit.unit_id)
        return read_result([direct_evidence()])

    plan = ResearchPlan.model_validate({**make_draft(), "contract_version": "research-planner-v1"})
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
    with pytest.raises(ValueError, match="旧研究计划仅允许历史读取"):
        asyncio.run(executor.execute_with_evidence(plan))
    assert calls == []


def _context() -> PlannerExecutionContext:
    """无参数；创建拥有明确运行、范围、版本与离线请求解析器的上下文。"""
    identity = ExecutionIdentity(run_id="run-1", user_id="user-1", scope={"paper_ids": ["paper-a"]},
                                 publication_snapshot={"paper_batches": {"paper-a": "b" * 32}})
    return PlannerExecutionContext(
        execution_state=PlannerExecutionState(identity), request_resolver=_request,
        result_context={"model": "offline", "prompt_version": "test-v1", "memory_allowed": False},
    )


def _request(unit, context):
    """参数为单元与上下文；离线端口已经核验固定范围，返回影响读取的明确条件。"""
    return unit.evidence_policy.model_dump(mode="json")


def test_same_request_shared_when_one_waiter_is_cancelled() -> None:
    """无参数；单个等待者取消时，其他消费者仍拿到同一次真实读取。"""
    async def scenario():
        """无参数；按事件顺序触发共享和取消，返回调用次数及最终关联。"""
        context, calls = _context(), []
        started, finish = asyncio.Event(), asyncio.Event()

        async def adapter(unit, ctx):
            """参数为读取请求；等待测试放行后返回真实材料。"""
            calls.append(unit.unit_id)
            started.set()
            await finish.wait()
            return read_result([direct_evidence()])

        first = asyncio.create_task(read_evidence(adapter, ResearchUnit.model_validate(make_unit("a")), context))
        await started.wait()
        second = asyncio.create_task(read_evidence(adapter, ResearchUnit.model_validate(make_unit("b")), context))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()
        finish.set()
        result = await second
        assert len(result.records) == 1
        assert calls == ["a"]
        association = context.execution_state.read_associations("b")[-1]
        assert association["reused"] is True
        assert association["actual_call_count"] == 0
        assert context.execution_state.snapshot().read_receipts[0]["actual_call_count"] == 1

    asyncio.run(scenario())


def test_last_waiter_cancellation_waits_for_real_cleanup() -> None:
    """无参数；最后一个消费者退出时不能比底层资源清理更早返回。"""
    async def scenario():
        """无参数；用事件控制底层取消清理，断言所有者仍在等待。"""
        context = _context()
        started, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def adapter(unit, ctx):
            """参数为读取请求；收到取消后等待实际清理完成，再重抛原取消。"""
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleaning.set()
                await finish.wait()
                raise

        task = asyncio.create_task(read_evidence(adapter, ResearchUnit.model_validate(make_unit()), context))
        await started.wait()
        task.cancel()
        await cleaning.wait()
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshot = context.execution_state.snapshot()
        assert snapshot.read_receipts[0]["outcome"] == "cancelled"
        assert snapshot.evidence_records == []

    asyncio.run(scenario())


def test_partial_receipt_restores_materials_without_completing_remaining_range() -> None:
    """无参数；checkpoint 接续免读同请求，同时完整保留尚未读取的真实范围。"""
    async def scenario():
        """无参数；保存再恢复凭据，断言正文不重复读取且投影无法修改权威事实。"""
        context, calls = _context(), []
        unit = ResearchUnit.model_validate(make_unit())
        report = KnowledgeReadReport(operation="sections", status="partial", remaining=({"page": 2},))

        async def adapter(requested_unit, ctx):
            """参数为请求单元；返回有未读部分的正文材料。"""
            calls.append(requested_unit.unit_id)
            return EvidenceReadResult((direct_evidence(),), report)

        await read_evidence(adapter, unit, context)
        snapshot = context.execution_state.snapshot()
        restored = PlannerExecutionState(context.execution_state.identity,
                                         evidence_records=snapshot.evidence_records, read_receipts=snapshot.read_receipts)
        current = replace(context, execution_state=restored)
        result = await read_evidence(adapter, unit, current)
        assert result.read_report.status == "partial"
        assert result.read_report.remaining == ({"page": 2},)
        projection = restored.read_associations(unit.unit_id)[0]
        projection["read_report"]["remaining"][0]["page"] = 999
        assert restored.read_associations(unit.unit_id)[0]["read_report"]["remaining"][0]["page"] == 2
        assert len(calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_noncacheable_report_never_becomes_success_cache(status) -> None:
    """参数为非冻结读取状态；同请求再次出现必须仍调用实际能力。"""
    async def scenario():
        """无参数；运行两次同请求并验证实际调用数。"""
        context, calls = _context(), []

        async def adapter(unit, ctx):
            """参数为读取请求；返回明确不可复用的报告。"""
            calls.append(unit.unit_id)
            return EvidenceReadResult((), KnowledgeReadReport(operation="test", status=status))

        unit = ResearchUnit.model_validate(make_unit())
        await read_evidence(adapter, unit, context)
        await read_evidence(adapter, unit, context)
        assert len(calls) == 2

    asyncio.run(scenario())


def test_receipt_cannot_restore_into_another_run_or_missing_material_pool() -> None:
    """无参数；运行身份不匹配或材料缺失的成功凭据在恢复入口明确失败。"""
    async def scenario():
        """无参数；生成正式凭据后尝试不一致恢复。"""
        context = _context()

        async def adapter(unit, ctx):
            """参数为读取请求；返回一条固定正文。"""
            return read_result([direct_evidence()])

        await read_evidence(adapter, ResearchUnit.model_validate(make_unit()), context)
        snapshot = context.execution_state.snapshot()
        identity = context.execution_state.identity
        with pytest.raises(ValueError, match="不在同一快照"):
            PlannerExecutionState(identity, read_receipts=snapshot.read_receipts)
        other = identity.model_copy(update={"run_id": "run-2"})
        with pytest.raises(ValueError, match="不一致"):
            PlannerExecutionState(other, evidence_records=snapshot.evidence_records, read_receipts=snapshot.read_receipts)

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_version", [None, True])
def test_receipt_rejects_body_without_current_source_contract(invalid_version) -> None:
    """参数为缺失或非整数正文版本；新版凭据不得把历史材料升级为免读取成果。"""
    async def scenario():
        """无参数；损坏真实快照的正文合同，恢复入口必须先失败。"""
        context = _context()

        async def adapter(unit, ctx):
            """参数为读取请求；返回有完整来源承诺的正文。"""
            return read_result([direct_evidence()])

        await read_evidence(adapter, ResearchUnit.model_validate(make_unit()), context)
        snapshot = context.execution_state.snapshot()
        record = snapshot.evidence_records[0]
        if invalid_version is None:
            for name in ("read_contract_version", "source_spans", "material_group_id", "source_location_status"):
                record.pop(name)
        else:
            record["read_contract_version"] = invalid_version
        with pytest.raises(ValueError, match="当前正文读取合同"):
            PlannerExecutionState(context.execution_state.identity, evidence_records=snapshot.evidence_records,
                                  read_receipts=snapshot.read_receipts)

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_version", [None, True, 1.0, "1"])
def test_receipt_restore_rejects_noninteger_contract_version(invalid_version) -> None:
    """参数为缺失或非整数凭据版本；恢复入口不得把畸形新版识别为当前读取合同。"""
    async def scenario():
        """无参数；只损坏成功凭据的版本，确认正文仍完整时也不能进入活跃缓存。"""
        context = _context()

        async def adapter(unit, ctx):
            """参数为读取请求；返回完整且可复用的正文材料。"""
            return read_result([direct_evidence()])

        await read_evidence(adapter, ResearchUnit.model_validate(make_unit()), context)
        snapshot = context.execution_state.snapshot()
        receipt = snapshot.read_receipts[0]
        if invalid_version is None:
            receipt.pop("contract_version")
        else:
            receipt["contract_version"] = invalid_version
        with pytest.raises(ValueError, match="当前执行合同版本"):
            PlannerExecutionState(context.execution_state.identity, evidence_records=snapshot.evidence_records,
                                  read_receipts=snapshot.read_receipts)

    asyncio.run(scenario())


def test_renamed_unit_reuses_complete_result_but_new_goal_reasons_again() -> None:
    """无参数；只改编号可复用并重绑主张；改目标只复用材料，重新生成结论。"""
    async def scenario():
        """无参数；走正式执行入口记录实际取证及推理次数。"""
        context, reads, reasons = _context(), [], []
        record = direct_evidence()

        async def adapter(unit, ctx):
            """参数为读取请求；返回固定正文并记录 IO。"""
            reads.append(unit.unit_id)
            return read_result([record])

        async def provider(unit, records, dependencies, **kwargs):
            """参数为实际推理输入；返回关联当前要求和单元的合法主张。"""
            reasons.append(unit.unit_id)
            return [ResearchClaimDraft(text="Attention is used", claim_type="paper_supported",
                                       supporting_evidence_ids=[record["evidence_id"]], supporting_spans=supporting_spans([record]),
                                       addressed_facet_ids=["f1"])]

        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter},
                                            reasoner=ResearchUnitReasoner(draft_provider=provider))
        original = ResearchPlan.model_validate(make_draft([make_unit("old")]))
        first = await executor.execute_with_evidence(original, context)
        renamed = ResearchPlan.model_validate(make_draft([make_unit("new")]))
        second = await executor.execute_with_evidence(renamed, context, previous_results=first.results)
        assert second.results[0].metadata["result_reused"] is True
        assert second.results[0].claims[0].claim_id == "new.c1"
        assert first.results[0].claims[0].claim_id == "old.c1"
        assert reads == ["old"]
        assert reasons == ["old"]
        changed = ResearchPlan.model_validate(make_draft([make_unit("new", goal="Compare the findings")]))
        await executor.execute_with_evidence(changed, context, previous_results=second.results)
        assert reads == ["old"]
        assert reasons == ["old", "new"]

    asyncio.run(scenario())


def test_renamed_dependency_rebinds_claim_sources_without_reasoning_again() -> None:
    """无参数；真实上游主张依赖改编号时，复用结果必须重绑全部依赖引用并保留历史。"""
    async def scenario():
        """无参数；通过两次完整计划执行核对读取、推理次数与下游来源编号。"""
        context, reads, reasons = _context(), [], []
        record = direct_evidence()

        async def adapter(unit, ctx):
            """参数为读取请求；返回同一固定版本的正文。"""
            reads.append(unit.unit_id)
            return read_result([record])

        async def provider(unit, records, dependencies, **kwargs):
            """参数为实际推理输入；上游产论文事实，下游只引用当前有效上游。"""
            reasons.append(unit.unit_id)
            if not unit.dependencies:
                return [ResearchClaimDraft(text="fact", claim_type="paper_supported",
                                           supporting_evidence_ids=[record["evidence_id"]], supporting_spans=supporting_spans([record]), addressed_facet_ids=["f1"])]
            return [ResearchClaimDraft(text="inference", claim_type="evidence_grounded_inference",
                                       source_claim_ids=[dependencies[0].claims[0].claim_id], reasoning_summary="uses the fact",
                                       addressed_facet_ids=["f1"])]

        upstream = make_unit("old-source")
        downstream = make_unit("old-derived", source_type="inference_from_units",
                               allowed_claims=["evidence_grounded_inference"], deps=[{"unit_id": "old-source"}])
        downstream["claim_policy"]["citation_required"] = False
        original = ResearchPlan.model_validate(make_draft([upstream, downstream]))
        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter},
                                            reasoner=ResearchUnitReasoner(draft_provider=provider))
        first = await executor.execute_with_evidence(original, context)
        new_source = original.units[0].model_copy(update={"unit_id": "new-source"})
        new_derived = original.units[1].model_copy(update={
            "unit_id": "new-derived",
            "dependencies": [original.units[1].dependencies[0].model_copy(update={"unit_id": "new-source"})],
        })
        renamed = original.model_copy(update={"units": [new_source, new_derived]})
        second = await executor.execute_with_evidence(renamed, context, previous_results=first.results)
        assert [result.status for result in second.results] == ["evidence_found", "inferred"]
        assert all(result.metadata["result_reused"] for result in second.results)
        assert second.results[1].claims[0].source_unit_ids == ["new-source"]
        assert first.results[1].claims[0].source_unit_ids == ["old-source"]
        assert second.results[1].claims[0].claim_id == "new-derived.c1"
        assert reads == ["old-source"]
        assert reasons == ["old-source", "old-derived"]

    asyncio.run(scenario())


def test_structure_verification_keeps_separate_dependency_materials() -> None:
    """无参数；结构核实再次调用推理器时仍要带上游正文，不能只传本单元新取到的正文。"""
    source = direct_evidence("upstream", text="The upstream result is supported.")
    verification = direct_evidence("verification", text="The graph relation is supported.")
    hint = {"evidence_id": "graph-hint", "paper_id": "paper-a", "source": "neo4j",
            "layer": "structure_hint", "usable_as_citation": False, "verification_status": "unverified"}

    async def direct_adapter(unit, context):
        """参数为单元和上下文；上游与结构核实返回不同正文来源。"""
        return read_result([source if unit.unit_id == "source" else verification])

    async def hint_adapter(unit, context):
        """参数为结构请求；只返回不可引用的图线索。"""
        return read_result([hint])

    unit = make_unit("verified", source_type="local_structure_hint", allowed_claims=["paper_supported"],
                     deps=[{"unit_id": "source", "required_claim_types": ["paper_supported"]}])
    unit["evidence_policy"]["verify_with_direct_text"] = True
    plan = ResearchPlan.model_validate(make_draft([make_unit("source"), unit]))
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": direct_adapter,
                                                 "local_structure_hint": hint_adapter},
                                        reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider))
    report = asyncio.run(executor.execute_with_evidence(plan, _context()))
    assert [result.status for result in report.results] == ["evidence_found", "evidence_found"]
    result = report.results[1]
    assert result.claims[0].supporting_evidence_ids == [verification["evidence_id"]]
    assert result.metadata["structure_verification"]["status"] == "verified"
    assert len(result.metadata["read_reports"]) == 2
    assert len(report.read_receipts) == 3


def test_cancellation_after_adapter_finishes_preserves_successful_read() -> None:
    """无参数；底层取消清理期间实际完成的读取仍留痕，调用方保持取消身份。"""
    async def scenario():
        """无参数；底层完成清理后返回材料，外层仍重抛取消。"""
        context = _context()
        started = asyncio.Event()

        async def adapter(unit, ctx):
            """参数为请求；模拟不可中断 IO 在取消清理期间完成的真实结果。"""
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                return read_result([direct_evidence()])

        task = asyncio.create_task(read_evidence(adapter, ResearchUnit.model_validate(make_unit()), context))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshot = context.execution_state.snapshot()
        assert len(snapshot.evidence_records) == 1
        assert any(receipt["outcome"] == "success" for receipt in snapshot.read_receipts)
        assert any(receipt["outcome"] == "cancelled" for receipt in snapshot.read_receipts)

    asyncio.run(scenario())


def test_executor_cancellation_keeps_adapter_cleanup_failure_cause() -> None:
    """无参数；整层取消保留用户取消身份，并让底层真实清理失败仍可追溯。"""
    async def scenario():
        """无参数；经正式执行入口取消真实工作，核对异常链与失败快照。"""
        started = asyncio.Event()
        failure = RuntimeError("cleanup failed")

        async def adapter(unit, ctx):
            """参数为读取请求；取消后抛出资源清理的真实故障。"""
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                raise failure

        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
        task = asyncio.create_task(executor.execute_with_evidence(ResearchPlan.model_validate(make_draft()), _context()))
        await started.wait()
        task.cancel("user cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert str(caught.value) == "user cancellation"
        cause = caught.value.__cause__
        while cause is not None and cause is not failure:
            cause = cause.__cause__
        assert cause is failure
        assert caught.value.planner_execution_report.read_receipts[0]["outcome"] == "cancelled"

    asyncio.run(scenario())


def test_layer_cancellation_keeps_earlier_sibling_cleanup_failure() -> None:
    """无参数；同层较早结束的清理故障不能被最后一个正常取消的兄弟覆盖。"""
    async def scenario():
        """无参数；按事件控制两个独立读取的收尾顺序，核对原取消和真实故障链。"""
        context = PlannerExecutionContext()
        started = {name: asyncio.Event() for name in ("failed", "slow")}
        release = asyncio.Event()
        failure = RuntimeError("earlier sibling cleanup failed")

        async def adapter(unit, ctx):
            """参数为读取请求；一个单元清理失败，另一个延后完成正常取消。"""
            started[unit.unit_id].set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if unit.unit_id == "failed":
                    raise failure
                await release.wait()
                raise

        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
        plan = ResearchPlan.model_validate(make_draft([make_unit("failed"), make_unit("slow")]))
        task = asyncio.create_task(executor.execute_with_evidence(plan, context))
        await asyncio.gather(*(event.wait() for event in started.values()))
        task.cancel("cancel both")
        while not context.execution_state.snapshot().read_receipts:
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert str(caught.value) == "cancel both"
        cause = caught.value.__cause__
        while cause is not None and cause is not failure:
            cause = cause.__cause__
        assert cause is failure
        assert len(caught.value.planner_execution_report.read_receipts) == 2

    asyncio.run(scenario())


def test_failed_request_is_retried_as_real_work() -> None:
    """无参数；失败尝试保留，但不会命中成功缓存或阻止后来的合法读取。"""
    async def scenario():
        """无参数；先制造不可重试故障，再验证相同请求成功调用。"""
        context, calls = _context(), []

        async def adapter(unit, ctx):
            """参数为读取请求；首次明确失败，之后才提供正文。"""
            calls.append(unit.unit_id)
            if len(calls) == 1:
                raise ValueError("invalid request")
            return read_result([direct_evidence()])

        unit = ResearchUnit.model_validate(make_unit())
        with pytest.raises(Exception, match="invalid request"):
            await read_evidence(adapter, unit, context)
        await read_evidence(adapter, unit, context)
        assert len(calls) == 2
        assert [receipt["outcome"] for receipt in context.execution_state.snapshot().read_receipts] == ["failed", "success"]

    asyncio.run(scenario())


def test_same_layer_failure_keeps_successful_sibling_materials() -> None:
    """无参数；同层第一个单元失败不能丢失后续兄弟单元已完成的材料。"""
    failure = ValueError("first unit failed")

    async def adapter(unit, context):
        """参数为单元；第一个失败，第二个照常取得实际材料。"""
        if unit.unit_id == "first":
            raise failure
        return read_result([direct_evidence()])

    plan = ResearchPlan.model_validate(make_draft([make_unit("first"), make_unit("second")]))
    with pytest.raises(Exception, match="first unit failed") as caught:
        asyncio.run(ResearchPlannerExecutor(adapters={"local_direct_text": adapter},
                    reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider)).execute_with_evidence(plan))
    report = caught.value.planner_execution_report
    assert len(report.evidence_records) == 1
    assert report.results[0].unit_id == "second"
    assert {receipt["outcome"] for receipt in report.read_receipts} == {"failed", "success"}


def test_timeout_retry_retains_completed_reads_without_caching_failed_attempts(monkeypatch) -> None:
    """参数为配置替身；超时后完成的正文保留，但失败请求不会变成免读成功。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_planner_evidence_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "rag_planner_evidence_max_retries", 1)
    monkeypatch.setattr(settings, "rag_planner_evidence_retry_backoff_seconds", 0.0)

    async def scenario():
        """无参数；先耗尽真实超时重试，再验证后续相同请求仍调用能力。"""
        context, calls = _context(), []

        async def adapter(unit, ctx):
            """参数为读取请求；前两次在超时清理中完成，第三次正常返回。"""
            calls.append(unit.unit_id)
            if len(calls) < 3:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    return read_result([direct_evidence()])
            return read_result([direct_evidence()])

        unit = ResearchUnit.model_validate(make_unit())
        with pytest.raises(Exception, match="TimeoutError"):
            await read_evidence(adapter, unit, context)
        snapshot = context.execution_state.snapshot()
        assert len(snapshot.evidence_records) == 1
        assert sum(receipt["actual_call_count"] for receipt in snapshot.read_receipts) == 2
        await read_evidence(adapter, unit, context)
        assert len(calls) == 3
        assert context.execution_state.read_associations(unit.unit_id)[-1]["reused"] is False

    asyncio.run(scenario())


def test_changed_dependency_read_report_invalidates_derived_result() -> None:
    """无参数；上游主张和材料相同而读取完整性变化时，下游仍必须重新推理。"""
    async def scenario():
        """无参数；两次实际读取只改变报告，用正式依赖执行检查派生结论失效。"""
        context, reasons, reads = _context(), [], []
        record = direct_evidence()

        async def adapter(unit, ctx):
            """参数为请求；返回同一材料及先部分、后完整的真实读取说明。"""
            reads.append(unit.unit_id)
            return EvidenceReadResult((record,), KnowledgeReadReport(
                operation="sections", status="partial" if len(reads) == 1 else "complete",
            ))

        async def provider(unit, records, dependencies, **kwargs):
            """参数为推理输入；上游产正文事实，下游基于同一事实生成推断。"""
            reasons.append(unit.unit_id)
            if unit.unit_id == "a":
                return [ResearchClaimDraft(text="fact", claim_type="paper_supported",
                                           supporting_evidence_ids=[record["evidence_id"]], supporting_spans=supporting_spans([record]), addressed_facet_ids=["f1"])]
            return [ResearchClaimDraft(text="inference", claim_type="evidence_grounded_inference",
                                       source_claim_ids=[dependencies[0].claims[0].claim_id], reasoning_summary="uses the fact", addressed_facet_ids=["f1"])]

        downstream = make_unit("b", source_type="inference_from_units",
                               allowed_claims=["evidence_grounded_inference"], deps=[{"unit_id": "a"}])
        downstream["claim_policy"]["citation_required"] = False
        plan = ResearchPlan.model_validate(make_draft([make_unit("a"), downstream]))
        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter},
                                            reasoner=ResearchUnitReasoner(draft_provider=provider))
        first = await executor.execute_with_evidence(plan, context)
        restored = replace(context, execution_state=PlannerExecutionState(context.execution_state.identity,
                                                                          evidence_records=first.evidence_records))
        second = await executor.execute_with_evidence(plan, restored, previous_results=first.results)
        assert [result.status for result in second.results] == ["evidence_found", "inferred"]
        assert reasons == ["a", "b", "a", "b"]

    asyncio.run(scenario())
