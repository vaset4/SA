"""【研究交付】【独立复核】验证正文必需条件与授权范围恢复。作者：xxx。"""

from types import SimpleNamespace

import pytest

from app.api.facades import graph_facade
from app.services.result_delivery_service import assemble_delivery


def test_report_does_not_replace_missing_delivery_answer():
    """无参数；存在交付报告时缺少同次正文仍被拒绝，修复不得引入 report 回退。"""
    payload = graph_facade._cross_paper_done_payload({
        "report": "不能自动认作本次正文",
        "final_delivery_report": {"attempt_id": "same-attempt", "status": "failed"},
        "final_generation_attempt": {"attempt_id": "same-attempt"},
    }, "session")
    assert payload["answer"] is None
    with pytest.raises(ValueError, match="最终产物缺少同次正文"):
        assemble_delivery(run={"run_id": "run"}, result=payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["resume", "retry"])
async def test_scope_refresh_does_not_clear_checkpoint_answer(kind):
    """参数为恢复命令；重新授权范围仅刷新范围字段，不带新问题的空正文覆盖检查点。"""
    saved = {"answer": "已保存正文", "report": "已保存正文", "structured_query": {"query": "原问题"}}
    changes = []

    async def get_state(config):
        """参数为检查点配置；返回明确的隔离快照。"""
        return SimpleNamespace(values=saved, interrupts=({"question": "补充"},) if kind == "resume" else ())

    async def update_state(config, update):
        """参数为恢复更新；记录正式入口实际提交的更新。"""
        changes.append(update)

    current, session_id, conversation_id = graph_facade._build_agent_state("继续", {
        "session_id": "session", "project_scope": {"paper_ids": ["paper-a"], "filter_expr": "authorized"}})
    command = await graph_facade._graph_input(
        agent=SimpleNamespace(aget_state=get_state, aupdate_state=update_state), user_query="继续", state=current,
        session_id=session_id, conversation_id=conversation_id, thread_id=current["thread_id"], execution_kind=kind)
    update = command.update if kind == "resume" else changes[0]
    assert update["project_scope"] == current["project_scope"]
    assert update["structured_query"]["query"] == "原问题"
    assert "answer" not in update and "report" not in update
    assert saved["answer"] == "已保存正文"
