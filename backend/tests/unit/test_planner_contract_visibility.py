"""【研究规划】【规则可见性】验证提示中的原话可通过正式合同。作者：xxx。"""

import pytest

from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
from app.services.internal_research.research_planner_plan_prompt import build_planner_prompt
from test_research_planner_strategy_contract import _strategy, _validate


@pytest.mark.parametrize("text", ["  解释  方法\n保留条件\t", "解释方法\n" + "中" * 510])
def test_visible_history_is_valid_user_basis(text: str) -> None:
    """参数为用户历史原文；返回无，验证可见引文保持字符并通过正式领域入口。"""
    prompt = build_planner_prompt({"user_query": "请解释方法", "chat_history": [
        {"role": "user", "content": text}, {"role": "assistant", "content": "仅供  参考\n总结"}]})
    visible = prompt.split("user (verbatim excerpt):\n", 1)[1].split("\n[end user excerpt]", 1)[0]
    assert visible == text[:500]
    assert visible in text
    assert "assistant: 仅供 参考 总结" in prompt
    assert ("[user excerpt truncated]" in prompt) == (len(text) > 500)
    payload = _strategy()
    payload["requested_facets"][0]["user_text_span"] = visible
    assert _validate(payload, user_texts=(text,)).is_valid


@pytest.mark.parametrize("span,reason", [("", "empty"), ("助手编造的文字", "does not match")])
def test_repair_feedback_identifies_field_without_fabricating_basis(span: str, reason: str) -> None:
    """参数为空或伪造依据；返回无，验证反馈定位字段且修正原话后才能通过。"""
    payload = _strategy()
    payload["requested_facets"][0]["user_text_span"] = span
    failure = _validate(payload).failure
    assert failure.code == "unverified_facet_basis"
    errors = failure.trace["validation_errors"]
    message = next(item["message"] for item in errors if item["code"] == failure.code)
    assert "method.user_text_span" in message and reason in message
    prompt = build_planner_prompt({"user_query": "请解释方法", "planner_observation": {
        "error_code": failure.code, "summary": message}})
    assert message in prompt
    assert payload["requested_facets"][0]["user_text_span"] == span
    payload["requested_facets"][0]["user_text_span"] = "解释方法"
    assert _validate(payload).is_valid


def test_schema_explains_conditional_obligations_without_changing_defaults() -> None:
    """无参数；返回无，验证说明进入正式schema且不把条件义务改成全局必填。"""
    schema = ResearchPlanStrategyDraft.model_json_schema()
    fields = schema["$defs"]["RequestedFacetDraft"]["properties"]
    assert "explicit" in fields["source"]["description"]
    assert "whitespace" in fields["user_text_span"]["description"]
    assert fields["user_text_span"]["default"] == ""
    assert "minLength" not in fields["user_text_span"]
    assert "clarification" in schema["properties"]["requested_facets"]["description"]
    payload = _strategy()
    payload["requested_facets"][0].update(source="inferred", user_text_span="", rationale="解释需要方法依据")
    result = _validate(payload)
    assert result.is_valid and result.plan.coverage_review.inferred_facets[0].required


def test_initial_clarification_and_action_keep_distinct_requirements() -> None:
    """无参数；返回无，验证纯澄清允许空要求，行动计划仍拒绝空要求。"""
    payload = {"units": [], "requested_facets": [], "answer_sections": [], "clarification": {
        "needed": True, "question": "你想解释哪种方法？", "reason_code": "unclear_goal"}}
    assert ResearchPlanStrategyDraft.model_validate(payload).clarification.needed
    payload["units"] = _strategy()["units"]
    assert _validate(payload).failure.code == "invalid_plan_format"


def test_history_window_and_untrusted_sources_do_not_expand_basis() -> None:
    """无参数；返回无，验证六条边界与助手、补全问题不自动成为可信原话。"""
    history = [{"role": "user", "content": "旧消息不可见"}] + [
        {"role": "user", "content": f"近期消息{i}"} for i in range(6)]
    prompt = build_planner_prompt({"user_query": "  请解释方法\n", "chat_history": history,
        "turn_depth_assessment": {"depth": "research", "resolved_query": "模型补全的要求"}})
    assert "旧消息不可见" not in prompt
    assert prompt.count("[end user excerpt]") == 6
    assert "Question:   请解释方法\n" in prompt
    for span in ("模型补全的要求", "助手的文字", "[user excerpt truncated]"):
        payload = _strategy()
        payload["requested_facets"][0]["user_text_span"] = span
        assert _validate(payload).failure.code == "unverified_facet_basis"
