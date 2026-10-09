"""评测仓储的实际持久化、读取资格和重复隔离回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evals.agent_outcome.environment import EnvironmentPortUnavailable
from evals.agent_outcome.runtime_storage import EvaluationStorage
from evals.contracts import snapshot


def initial_snapshot() -> str:
    """无参数；返回显式测试初始记忆，包含正常及不合格候选。"""
    base = {"project_id": "project", "user_id": "user", "review_state": "accepted", "status": "confirmed", "stale_state": "fresh",
            "memory_type": "method_insight", "title": "test", "body": "original", "version": 1, "maturity": "validated", "source_kind": "manual"}
    rows = [{**base, "memory_id": "accepted"}, {**base, "memory_id": "draft", "review_state": "pending_review"},
            {**base, "memory_id": "other", "user_id": "another"}, {**base, "memory_id": "stale", "stale_state": "stale"}]
    return snapshot({"schema_version": "eval-storage-v1", "user_id": "user", "project_id": "project", "papers": [],
                     "items": rows, "questions": [], "evidence_refs": [], "links": [], "question_links": [], "versions": []})


def test_existing_memory_service_keeps_all_qualification_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录及依赖替换；使用原业务读取服务验证 owner/review/status/fresh 四道资格门。"""
    from app.services import research_memory_injector_service

    storage = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "storage")
    monkeypatch.setattr(research_memory_injector_service, "get_research_memory_repository", storage.repository)
    result = asyncio.run(research_memory_injector_service.load_injectable_memory("project", "user"))
    assert result["status"] == "success"
    assert [item["memory_id"] for item in result["items"]] == ["accepted"]
    storage.persist_snapshot()
    assert (tmp_path / "storage" / "io.jsonl").is_file()


def test_written_ids_are_real_and_second_episode_starts_from_fixed_state(tmp_path: Path) -> None:
    """参数为目录；写回实际对象可读，第二次运行不会继承第一次候选。"""
    first = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "first")
    payload = {"title": "new", "body": "new body", "memory_type": "method_insight", "user_id": "user", "review_state": "pending_review"}
    detail = asyncio.run(first.create_item("project", payload))
    identity = detail["item"]["memory_id"]
    assert asyncio.run(first.get_item(identity))["title"] == "new"
    assert payload == {"title": "new", "body": "new body", "memory_type": "method_insight", "user_id": "user", "review_state": "pending_review"}
    first.persist_snapshot()
    written = json.loads((tmp_path / "first" / "state.json").read_text(encoding="utf-8"))
    assert identity in {row["memory_id"] for row in written["items"]}
    second = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "second")
    assert asyncio.run(second.get_item(identity)) is None


def test_unconfigured_storage_port_is_a_visible_environment_error(tmp_path: Path) -> None:
    """参数为目录；产品捕获异常后，隔离仓储仍保留可见的失败访问。"""
    storage = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "storage")
    with pytest.raises(EnvironmentPortUnavailable, match="not_implemented"):
        storage.not_implemented()
    assert storage.errors[0]["port"] == "not_implemented"


def test_fault_declaration_only_fires_on_actual_port_attempt(tmp_path: Path) -> None:
    """参数为目录；只有实际调用才出现故障事件，且读取端口的集合参数可真实记录。"""
    from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
    from evals.suites.runtime import SourceFailure

    storage = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "storage")
    fault = SourceFailure(storage=storage, message="declared timeout")
    trace = TraceContext(request_id="fault-test")
    token = set_trace_ctx(trace)
    try:
        assert export_evidence_observations(trace)["evidence_observations"] == []
        with pytest.raises(TimeoutError, match="declared timeout"):
            fault.raise_on_attempt("read_chunk", "actual-child", {"already-read"})
    finally:
        reset_trace_ctx(token)
    events = export_evidence_observations(trace)["evidence_observations"]
    assert events[0]["stage"] == "source_port_failed"


def test_actual_model_request_parameters_are_recorded_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为端口替换器；原请求照常发送，观察仅保存模型、温度及去凭证地址。"""
    from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
    from evals.suites.runtime import ModelRequestObserver

    class Client:
        """受控 HTTP 客户端，只用于验证实际调用透传。"""

        base_url = "https://test-user:test-password@example.com/v1?token=test-token"

        def send(self, headers, payload):
            """参数为原请求；验证鉴权头仍透传给真正发送端口。"""
            assert headers == {"Authorization": "test-bearer"}
            return payload

    observer = ModelRequestObserver(send=Client.send)
    monkeypatch.setattr(Client, "send", observer)
    trace = TraceContext(request_id="actual-parameter-test")
    token = set_trace_ctx(trace)
    try:
        result = Client().send({"Authorization": "test-bearer"}, {"model": "test-model", "temperature": 0.2})
    finally:
        reset_trace_ctx(token)
    assert result["model"] == "test-model"
    observed = export_evidence_observations(trace)
    parameters = observed["evidence_observations"][0]["payload"]
    assert parameters == {"model": "test-model", "temperature": 0.2, "endpoint": "https://example.com/v1"}
    assert "test-bearer" not in snapshot(observed) and "test-password" not in snapshot(observed)


def test_sanitized_storage_writes_keep_original_business_state(tmp_path: Path) -> None:
    """参数为目录；实际trace/日志/状态落盘前脱敏，传入载荷和后续业务读取仍保持原对象内容。"""
    secret = "fixture-storage-credential"

    def sanitize(value: Any) -> Any:
        """参数为持久化副本；递归替换夹具凭证，返回同形安全副本。"""
        if isinstance(value, dict):
            return {key: sanitize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [sanitize(child) for child in value]
        return value.replace(secret, "[REDACTED]") if isinstance(value, str) else value

    output = tmp_path / "storage"
    storage = EvaluationStorage(initial_snapshot(), output_dir=output, sanitize=sanitize)
    trace = {"error_type": "ConnectionError", "error": f"credential rejected: {secret}"}
    storage.write_log_sync("failed", trace)
    asyncio.run(storage.write_rag_trace(trace_data=trace))
    assert secret not in (output / "io.jsonl").read_text(encoding="utf-8")
    detail = asyncio.run(storage.create_item("project", {
        "title": "recorded failure", "body": trace["error"], "user_id": "user", "memory_type": "method_insight",
    }))
    assert detail["item"]["body"] == trace["error"] and secret in trace["error"]
    storage.persist_snapshot()
    for filename in ("io.jsonl", "state.json", "access.json"):
        text = (output / filename).read_text(encoding="utf-8")
        assert secret not in text and "credential rejected: [REDACTED]" in text
    assert "ConnectionError" in (output / "access.json").read_text(encoding="utf-8")
    assert asyncio.run(storage.get_item(detail["item"]["memory_id"]))["body"] == trace["error"]
