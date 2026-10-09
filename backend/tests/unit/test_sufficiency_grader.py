"""AC6: sufficiency_grader 节点单测。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.exceptions import OutputParserException

from app.nodes.sufficiency_grader import sufficiency_grader_node, SufficiencyDecision


class _FakeSettings:
    rag_sufficiency_grader_enabled = True
    rag_sufficiency_grader_model = ""


def _patch_settings():
    return patch("app.nodes.sufficiency_grader.get_settings", return_value=_FakeSettings())


def _patch_log():
    return patch("app.nodes.sufficiency_grader.log_structured_event")


def _patch_trace():
    return patch("app.nodes.sufficiency_grader.trace_label")


def _make_state(**overrides):
    base = {
        "user_query": "test question",
        "search_results": [{"text": "some evidence", "score": 0.8}],
        "graph_context": "entity: X",
        "iteration": 2,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_disabled_returns_sufficient():
    """开关关闭 → 返回 sufficient。"""

    class _Disabled(_FakeSettings):
        rag_sufficiency_grader_enabled = False

    with patch("app.nodes.sufficiency_grader.get_settings", return_value=_Disabled()):
        result = await sufficiency_grader_node(_make_state())
    assert result["sufficiency_result"] == "sufficient"


@pytest.mark.asyncio
async def test_grader_returns_sufficient():
    """mock grader 返回 sufficient=True → sufficient + should_continue=False。"""
    decision = SufficiencyDecision(sufficient=True, reason="evidence covers core points")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=decision)

    with _patch_settings(), _patch_log() as mock_log, _patch_trace(), \
         patch("app.nodes.sufficiency_grader.get_llm", return_value=mock_llm):
        result = await sufficiency_grader_node(_make_state())

    assert result["sufficiency_result"] == "sufficient"
    assert result["should_continue"] is False
    log_calls = [c for c in mock_log.call_args_list if c[0][1] == "rag_sufficiency_graded"]
    assert len(log_calls) == 1
    assert log_calls[0][1]["sufficient"] is True


@pytest.mark.asyncio
async def test_grader_returns_insufficient():
    """mock grader 返回 sufficient=False → insufficient + should_continue=True。"""
    decision = SufficiencyDecision(sufficient=False, reason="missing key info")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=decision)

    with _patch_settings(), _patch_log(), _patch_trace(), \
         patch("app.nodes.sufficiency_grader.get_llm", return_value=mock_llm):
        result = await sufficiency_grader_node(_make_state())

    assert result["sufficiency_result"] == "insufficient"
    assert result["should_continue"] is True


@pytest.mark.asyncio
async def test_grader_exception_returns_insufficient():
    """grader 抛异常 → insufficient + 继续取证。"""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=OutputParserException("parse error")
    )

    with _patch_settings(), _patch_log() as mock_log, _patch_trace(), \
         patch("app.nodes.sufficiency_grader.get_llm", return_value=mock_llm):
        result = await sufficiency_grader_node(_make_state())

    assert result["sufficiency_result"] == "insufficient"
    assert result["should_continue"] is True
    log_calls = [c for c in mock_log.call_args_list if c[0][1] == "rag_sufficiency_graded"]
    assert len(log_calls) == 1
    assert log_calls[0][1]["sufficient"] is False
    assert "grader_error_default_insufficient" in log_calls[0][1]["reason"]
