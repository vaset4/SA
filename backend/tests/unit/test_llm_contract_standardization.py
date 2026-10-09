"""【模型合同】【标准边界】复现围栏、消息及完整响应丢失；作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app import config as app_config
from app.middleware import trace_context
from app.services import llm_service
from app.services.llm_service import (
    _structured_output_native_enabled as configured_native_enabled,
)


class TextDraft(BaseModel):
    """完整保留原文的最小严格合同，不是真实历史响应重放。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str
    score: int = 1


@pytest.fixture
def runtime(monkeypatch):
    """参数为替身管理器；返回真实传输入口与离线响应替身，禁止实际网络。"""
    cache = Mock()
    cache.get_llm_response.return_value = None
    post = Mock(side_effect=AssertionError("unexpected HTTP"))
    monkeypatch.setattr(llm_service, "get_cache", lambda: cache)
    monkeypatch.setattr(llm_service.requests, "post", post)
    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 1)
    queue = Mock(policy={"enabled": True})
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    client = llm_service.OpenAICompatibleLLM("fixture", "fixture-key", "https://fixture.invalid", 0, 1,
                                             request_exception_max_retries=0, empty_response_max_retries=0)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: False)
    trace = trace_context.TraceContext(request_id="contract")
    token = trace_context.set_trace_ctx(trace)
    try:
        yield client, post, cache, trace
    finally:
        trace_context.reset_trace_ctx(token)


def response(content: str, status: int = 200) -> requests.Response:
    """参数为可见响应正文与状态；返回本地 HTTP 响应，不打开连接。"""
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    return result


def test_json_embedded_fence_preserves_all_characters():
    """JSON 内部算法围栏、中文及首尾空白必须逐字保留。"""
    source = ' \n```python\nprint("中文")\n```\n '
    result = llm_service.parse_structured_response(json.dumps({"answer": source}), TextDraft)
    assert result.answer == source


@pytest.mark.parametrize("content", [
    '```json\n{"answer":"ok"}\n```', '{"answer":"ok"} trailing',
    '{"answer":"ok","score":"1"}', '{"answer":"ok","extra":true}',
])
def test_entire_json_must_satisfy_the_model(content):
    """外层围栏、尾随文字、错误类型和多余键不得被清洗成成功。"""
    with pytest.raises(ValidationError):
        llm_service.parse_structured_response(content, TextDraft)


def test_dictionary_and_base_messages_share_standard_conversion(runtime):
    """初次请求的字典和标准消息保留相同角色及真实换行。"""
    client, post, _cache, _trace = runtime
    post.side_effect = None
    post.return_value = response('{"answer":"ok"}')
    original = [{"role": "system", "content": "规则\n第二行"}, HumanMessage(content="问题\n原文"),
                AIMessage(content="之前的回答")]
    client.with_structured_output(TextDraft).invoke(original)
    assert post.call_args.kwargs["json"]["messages"][1:] == [
        {"role": "system", "content": "规则\n第二行"}, {"role": "user", "content": "问题\n原文"},
        {"role": "assistant", "content": "之前的回答"},
    ]


@pytest.mark.parametrize("message", [object(), {"role": "user", "content": 123},
                                      {"role": "user", "content": None}])
def test_unknown_messages_fail_before_http(runtime, message):
    """消息边界错误不发送 HTTP，也不消耗 schema 回修。"""
    client, post, _cache, _trace = runtime
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        client.with_structured_output(TextDraft).invoke([message])
    post.assert_not_called()


@pytest.mark.parametrize("content", [123, {"unexpected": "text"}, None])
def test_invalid_provider_content_is_envelope_error_without_schema_repair(runtime, content):
    """供应商内容字段类型错误发生在信封层，不强转字符串后消耗模型回修。"""
    client, post, _cache, trace = runtime
    post.side_effect = None
    post.return_value = response(content)
    with pytest.raises(TypeError):
        client.with_structured_output(TextDraft).invoke("question")
    assert post.call_count == 1
    observations = [json.loads(item) for item in trace.evidence_observations]
    assert not any(item["stage"] == "structured_response_validation" for item in observations)
    usage = next(item["payload"] for item in observations if item["stage"] == "model_response_usage")
    assert usage["error_boundary"] == "api_envelope"


def test_cancelled_before_admission_summary_records_zero_actual_sends(runtime, monkeypatch):
    """排队取消没有实际 HTTP，正式 trace 汇总不能按旧缺省公式虚构一次发送。"""
    client, post, _cache, trace = runtime
    queue = Mock(policy={"enabled": True})
    queue.acquire.side_effect = asyncio.CancelledError()
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    with pytest.raises(asyncio.CancelledError):
        client.with_structured_output(TextDraft).invoke("question")
    post.assert_not_called()
    assert trace_context.build_trace_summary(trace, {})["model_http_attempt_count"] == 0


def test_repair_keeps_full_previous_assistant_and_field_errors(runtime):
    """超过旧预览长度的失败正文完整进入 assistant 消息，字段路径保留。"""
    client, post, _cache, _trace = runtime
    raw = json.dumps({"answer": "中文\n" * 1000, "score": "wrong"}, ensure_ascii=False)
    post.side_effect = [response(raw), response('{"answer":"fixed"}')]
    client.with_structured_output(TextDraft).invoke([SystemMessage(content="系统\n规则"), HumanMessage(content="问题")])
    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[1:3] == [{"role": "system", "content": "系统\n规则"}, {"role": "user", "content": "问题"}]
    assert messages[-2] == {"role": "assistant", "content": raw}
    assert '"score"' in messages[-1]["content"] and "int_type" in messages[-1]["content"]


def test_native_rejection_never_removes_requested_format(runtime):
    """供应商拒绝所选格式时保持真实错误，不能额外发送降级请求。"""
    client, post, _cache, _trace = runtime
    post.side_effect = [response("response_format unsupported", 400), response('{"answer":"hidden"}')]
    with pytest.raises(requests.HTTPError):
        client.with_structured_output(TextDraft).invoke("question")
    assert post.call_count == 1


def test_program_exception_is_not_model_repair(runtime):
    """校验器中的程序异常原样抛出，不伪装成模型字段错误再次调用。"""
    class BrokenDraft(TextDraft):
        """用于证明程序错误分类的测试合同。"""

        @field_validator("answer")
        @classmethod
        def broken(cls, value):
            """参数为字段值；抛出人为程序异常以验证传播边界。"""
            raise RuntimeError("program defect")

    client, post, _cache, _trace = runtime
    post.side_effect = None
    post.return_value = response('{"answer":"ok"}')
    with pytest.raises(RuntimeError, match="program defect"):
        client.with_structured_output(BrokenDraft).invoke("question")
    assert post.call_count == 1


def test_bad_cache_consumes_only_existing_repair_budget(runtime):
    """坏缓存失效后仅使用一次回修，不产生额外刷新尝试。"""
    client, post, cache, _trace = runtime
    cache.get_llm_response.return_value = "bad cache"
    post.side_effect = [response("bad repair"), response('{"answer":"hidden"}')]
    with pytest.raises(llm_service.StructuredOutputError):
        client.with_structured_output(TextDraft).invoke("question")
    assert post.call_count == 1
    cache.delete_llm_response.assert_called_once()


def test_schema_and_prompt_versions_isolate_plain_text_cache(runtime, monkeypatch):
    """原生格式关闭后不同 schema 及提示版本仍使用不同缓存身份，同合同命中原键。"""
    class OtherDraft(TextDraft):
        """输出多一个字段时，缓存不能复用旧模型语义。"""

        note: str = ""

    client, post, cache, _trace = runtime
    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: False)
    post.side_effect = None
    post.return_value = response('{"answer":"ok"}')
    for schema in (TextDraft, TextDraft, OtherDraft):
        client.with_structured_output(schema).invoke("question")
    monkeypatch.setattr(llm_service, "STRUCTURED_OUTPUT_PROMPT_VERSION", "test-next-version")
    client.with_structured_output(TextDraft).invoke("question")
    keys = [call.args[0] for call in cache.get_llm_response.call_args_list]
    assert keys[0] == keys[1]
    assert len(set(keys)) == 3


@pytest.mark.parametrize(
    "env_value,native", [(None, True), ("true", True), ("false", False)]
)
def test_native_configuration_controls_payload_and_preserves_schema(
    runtime, monkeypatch, env_value, native
):
    """参数为运行夹具、环境替身、配置值和预期；验证请求格式及严格校验，无返回值。"""
    client, post, _cache, _trace = runtime
    if env_value is None:
        monkeypatch.delenv("STRUCTURED_OUTPUT_NATIVE_DEFAULT", raising=False)
    else:
        monkeypatch.setenv("STRUCTURED_OUTPUT_NATIVE_DEFAULT", env_value)
    settings = app_config.Settings(_env_file=None)
    monkeypatch.setattr(app_config, "get_settings", lambda: settings)
    monkeypatch.setattr(
        llm_service, "_structured_output_native_enabled", configured_native_enabled
    )
    post.side_effect = [response("bad json"), response('{"answer":"ok"}')]
    result = client.with_structured_output(TextDraft).invoke("question")
    assert result.answer == "ok"
    assert post.call_count == 2
    contract = json.dumps(TextDraft.model_json_schema(), ensure_ascii=False, indent=2)
    for call in post.call_args_list:
        payload = call.kwargs["json"]
        assert ("response_format" in payload) is native
        if native:
            assert payload["response_format"]["type"] == "json_schema"
        contents = [message["content"] for message in call.kwargs["json"]["messages"]]
        assert sum(content.count(contract) for content in contents) == 1


def test_schema_construction_error_is_visible_before_http(runtime, monkeypatch):
    """schema 构造失败不降为 json_object，也不发送请求。"""
    client, post, _cache, trace = runtime
    monkeypatch.setattr(TextDraft, "model_json_schema", Mock(side_effect=RuntimeError("schema defect")))
    with pytest.raises(RuntimeError, match="schema defect"):
        client.with_structured_output(TextDraft).invoke("question")
    post.assert_not_called()
    end = json.loads(trace.evidence_observations[-1])["payload"]
    assert end["failure_stage"] == "schema_configuration"


def test_response_trace_redacts_secret_without_changing_returned_quote(runtime):
    """原响应不修改，持久记录去除当前密钥并保留原文 URL 和其他字符。"""
    client, post, _cache, trace = runtime
    text = "quote fixture-key https://paper.test/source?version=1\n"
    post.side_effect = None
    post.return_value = response(json.dumps({"answer": text}))
    assert client.with_structured_output(TextDraft).invoke("question").answer == text
    exported = json.dumps(trace_context.export_evidence_observations(trace))
    assert "fixture-key" not in exported
    assert "https://paper.test/source?version=1" in exported


def test_interrupted_stream_retains_partial_visible_response(runtime, monkeypatch):
    """SSE 中断留存已到达正文并保持传输失败，不能拿半截正文通过 schema。"""
    client, post, _cache, trace = runtime
    streamed = response("")
    streamed.headers["Content-Type"] = "text/event-stream"
    streamed._content_consumed = True
    monkeypatch.setattr(client, "_iter_sse_data_lines", lambda _: iter([
        'data: {"choices":[{"delta":{"content":"partial visible text"}}]}',
    ]))
    post.side_effect = None
    post.return_value = streamed
    with pytest.raises(requests.ConnectionError):
        client.with_structured_output(TextDraft).invoke("question")
    full = trace_context.build_full_trace(trace, {})
    capture = full["model_response_observations"][0]["payload"]
    assert capture["response_text"] == "partial visible text"
    assert capture["content_complete"] is False
    assert capture["http_attempt"] == 1 and post.call_count == 1


def test_full_output_survives_export_and_actual_trace_write(runtime, monkeypatch):
    """大响应经过正式冻结导出及写库边界后仍完整，输入对象不被裁剪修改。"""
    client, post, _cache, trace = runtime
    raw = json.dumps({"answer": "中文算法\n" * 20000}, ensure_ascii=False)
    post.side_effect = None
    post.return_value = response(raw)
    client.with_structured_output(TextDraft).invoke("question")
    exported = trace_context.export_evidence_observations(trace)
    assert exported["model_call_observations"][0]["output_text"] == raw
    full = trace_context.build_full_trace(trace, {})
    before = json.dumps(full, ensure_ascii=False)
    saved = []

    class Repository:
        """替换唯一数据库写入点，核对传入序列化结果。"""

        async def write_rag_trace(self, **kwargs):
            """参数为正式持久化参数；保存独立副本供断言。"""
            saved.append(json.loads(json.dumps(kwargs["trace_data"])))

    monkeypatch.setattr("app.repositories.observability_repository.get_observability_repository", lambda: Repository())
    monkeypatch.setattr(trace_context, "OBS_TRACE_MAX_BYTES", 100)
    asyncio.run(trace_context.safe_write_trace(full, "contract", "session", "question", "owner"))
    assert json.dumps(full, ensure_ascii=False) == before
    assert saved[0]["model_responses"][0]["output_text"] == raw


def test_real_episode_freezing_preserves_complete_response(runtime, tmp_path):
    """正式回合采集、snapshot 和 frozen.jsonl 写入后仍能回读完整模型响应。"""
    from types import SimpleNamespace

    from evals.agent_outcome.report import write_agent_report
    from evals.agent_outcome.runner import collect_graph_turn

    client, post, _cache, _trace = runtime
    raw = json.dumps({"answer": "算法原文\n" * 20000}, ensure_ascii=False)
    post.side_effect = None
    post.return_value = response(raw)

    class Graph:
        """用固定模型响应替代研究动作，执行真实回合采集边界。"""

        async def astream(self, _input, *, stream_mode):
            """参数为执行输入和流模式；通过真实客户端产出终态。"""
            assert stream_mode == ["updates", "values"]
            answer = client.with_structured_output(TextDraft).invoke("question")
            yield "values", {"answer": answer.answer}

    environment = SimpleNamespace(
        create_turn_trace=lambda _: trace_context.TraceContext(request_id="frozen-response"),
        record_access=lambda *_args, **_kwargs: None,
        record_index_declaration=lambda: None,
    )
    captured = asyncio.run(collect_graph_turn(Graph(), {"question": "question"}, turn=1, env=environment))
    write_agent_report(tmp_path, {"grader_version": "not_assessed", "rows": []}, [captured])
    frozen = json.loads((tmp_path / "frozen.jsonl").read_text(encoding="utf-8"))
    evidence = json.loads(frozen["evidence_observation_snapshot"])
    assert evidence["model_call_observations"][0]["output_text"] == raw
    validation = next(item for item in evidence["evidence_observations"] if item["stage"] == "structured_response_validation")
    assert validation["payload"]["response_text"] == raw


def test_validation_error_fields_exclude_known_credentials(runtime, monkeypatch):
    """字段错误中的自定义消息也脱敏，不能从错误数组泄漏已知凭证。"""
    class RejectedDraft(TextDraft):
        """固定拒绝测试值的严格合同。"""

        @field_validator("answer")
        @classmethod
        def reject(cls, value):
            """参数为测试字段；在错误中回显测试凭证以核对采集边界。"""
            raise ValueError("fixture-key rejected")

    client, post, _cache, trace = runtime
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 0)
    post.side_effect = None
    post.return_value = response('{"answer":"ok"}')
    with pytest.raises(llm_service.StructuredOutputError):
        client.with_structured_output(RejectedDraft).invoke("question")
    assert "fixture-key" not in json.dumps(trace_context.export_evidence_observations(trace))

