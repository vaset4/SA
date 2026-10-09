"""上传任务唤醒及检索发布合同回归，作者：xxx。"""
from __future__ import annotations

import asyncio

import pytest


def test_research_projection_keeps_publication_snapshot():
    """RAG 范围投影必须保留原发布快照，不能丢失已核验的版本身份。"""
    from app.api.facades.research_contract_facade import build_dialogue_structured_query

    snapshot = {"lease_id": "read-lease", "user_id": "a", "batch_ids": ["published"]}
    context = {"project_id": "project", "user_id": "a", "paper_ids": ["paper"],
        "filter_expr": 'paper_id == "paper" and batch_id in ["published"]',
        "local_retrieval_allowed": True, "publication": snapshot, "resolved_scope": None}
    query = build_dialogue_structured_query(query="论文方法是什么", project_context=context,
        project_id="project", explicit_paper_ids=["paper"])
    assert query["project_scope"]["publication"] == snapshot
    assert query["project_scope"]["filter_expr"] == context["filter_expr"]


@pytest.mark.asyncio
async def test_embedded_worker_does_not_lose_enqueue_during_idle_exit(monkeypatch):
    """队列刚被判空时的新受理仍能唤醒消费，无需再次上传或重启。"""
    from app.services import embedded_worker_service as workers

    idle_checked, finish_idle = asyncio.Event(), asyncio.Event()
    queued, processed = [], []

    async def queue_round(**kwargs):
        """模拟判空与返回之间的新入队；返回本轮处理数量。"""
        if not idle_checked.is_set():
            idle_checked.set()
            await finish_idle.wait()
            return 0
        processed.extend(queued)
        queued.clear()
        return len(processed)

    monkeypatch.setattr(workers, "run_worker_loop", queue_round)
    worker = workers.ensure_embedded_worker_running(task_types=["upload_ingest"])
    await idle_checked.wait()
    queued.append("new-upload")
    workers.ensure_embedded_worker_running(task_types=["upload_ingest"])
    finish_idle.set()
    try:
        await asyncio.wait_for(worker, timeout=1)
        assert processed == ["new-upload"]
    finally:
        await workers.close_embedded_workers()


def test_startup_keeps_upload_consumer_alive_for_expiring_leases(monkeypatch):
    """启动后的消费者持续等候失租任务，仍由队列排除 paused 任务。"""
    from unittest.mock import Mock
    from app.runtime import lifespan
    from app.services import embedded_worker_service

    start = Mock()
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", start)
    lifespan.start_upload_worker()
    start.assert_called_once_with(task_types=["upload_ingest"], stop_when_idle=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("graph failed"), asyncio.CancelledError()])
async def test_graph_completion_failure_and_cancellation_release_same_snapshot(monkeypatch, failure):
    """问答成功、异常和取消都释放原快照，清理失败不得覆盖原执行结果。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.api.facades import graph_facade
    from app.services import upload_read_service
    from app.services.ingest import upload_pipeline

    snapshot = {"lease_id": "lease", "user_id": "a", "batch_ids": ["batch"], "paper_batches": {"paper": "batch"}}
    scope = {"user_id": "a", "publication": snapshot}
    release = AsyncMock()
    cleanup = AsyncMock()
    invoke = AsyncMock(return_value=({"answer": "ok"}, "session"), side_effect=failure)
    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository", lambda: SimpleNamespace(release_snapshot=release, renew_snapshot=AsyncMock()))
    monkeypatch.setattr(upload_pipeline, "cleanup_retired_batches", cleanup)
    monkeypatch.setattr(graph_facade, "_invoke_graph", invoke)
    if failure is None:
        assert await graph_facade.invoke_graph("query", {"project_scope": scope}) == ({"answer": "ok"}, "session")
    else:
        with pytest.raises(type(failure)):
            await graph_facade.invoke_graph("query", {"project_scope": scope})
    release.assert_awaited_once_with(snapshot)
    cleanup.assert_awaited_once_with("paper", user_id="a")


def test_deletion_removes_only_selected_papers_version_directory(tmp_path):
    """删除目标论文全部原文、解析和批次文件，其他论文及目录外文件保持存在。"""
    from app.services.paper_file_cleanup_service import delete_managed_paper_files

    root = tmp_path / "uploads"
    paths = [root / "paper" / "source" / "original.pdf", root / "paper" / "batches" / "old" / "index.json",
        root / "paper" / "parse" / "document.json", root / "other" / "source.pdf", tmp_path / "outside.txt"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    result = delete_managed_paper_files("paper", pdf_path=str(paths[0]), data_dir=tmp_path)
    assert "upload_versions" in result["deleted_kinds"]
    assert not (root / "paper").exists()
    assert paths[-1].exists() and paths[-2].exists()
    assert delete_managed_paper_files("paper", pdf_path=str(paths[0]), data_dir=tmp_path)["deleted_count"] == 0


@pytest.mark.asyncio
async def test_interrupted_research_uses_fresh_scope_without_resetting_plan():
    """研究澄清后续跑必须换成本轮授权快照，同时保留已完成的研究计划和查询语义。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.api.facades.graph_facade import _graph_input

    scope = {"paper_ids": ["paper"], "user_id": "a", "filter_expr": "new-filter",
        "publication": {"lease_id": "new-lease"}}
    previous = {"structured_query": {"intent": "research", "project_scope": {"publication": {"lease_id": "expired"}}},
        "research_plan": {"units": ["existing"]}}
    agent = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(interrupts=["question"], values=previous)))
    command = await _graph_input(agent=agent, user_query="继续", state={"project_scope": scope},
        session_id="session", conversation_id=None, thread_id="session", execution_kind="resume")
    assert command.resume == "继续"
    assert command.update["project_scope"] == scope
    assert command.update["structured_query"]["intent"] == "research"
    assert "research_plan" not in command.update


def test_project_context_reaches_milvus_with_same_publication(monkeypatch):
    """请求范围经过 RAG 投影与检索适配后，仍按原快照续租并过滤同一批次。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from app.api.facades.research_contract_facade import build_project_scope_contract
    from app.services import upload_read_service
    from app.services.milvus_store import MilvusStore
    from app.services.retrieval_service import search_chunks

    snapshot = {"lease_id": "read", "user_id": "a", "batch_ids": ["published"],
        "paper_batches": {"paper": "published"}, "collection_names": ["isolated"], "embedding_fingerprints": ["model"]}
    scope = build_project_scope_contract({"user_id": "a", "paper_ids": ["paper"], "filter_expr": 'paper_id == "paper"',
        "local_retrieval_allowed": True, "publication": snapshot})
    renewal = AsyncMock()
    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository", lambda: SimpleNamespace(renew_snapshot=renewal))
    monkeypatch.setattr(upload_read_service, "embedding_identity", lambda: "model")
    monkeypatch.setattr(upload_read_service.paper_db, "_run", asyncio.run)
    store = MilvusStore(collection_name="isolated")
    query = Mock(return_value=[{"chunk_id": "published-chunk"}])
    monkeypatch.setattr(store, "hybrid_search", query)
    assert search_chunks(store, "method", 2, scope=scope) == [{"chunk_id": "published-chunk"}]
    renewal.assert_awaited_once_with(snapshot)
    assert 'batch_id in ["published"]' in query.call_args.kwargs["filter_expr"]
