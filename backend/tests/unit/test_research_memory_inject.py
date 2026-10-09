"""研究记忆读取动作的排序、投影、渲染与四态返回测试。

覆盖点：
- 注入次序：established 先于 leaning 先于 intuition，按条数上限截断，截断数进 excluded_counts
- 四态返回：记忆存储不可用 → failed + error_kind；项目内无合格记忆 → empty。两者不归并
- 渲染文案：不再产出「待重绑」提示（门收紧后 needs_rebind 到不了渲染层）

原先这里还有「记忆注入 planner prompt」与「记忆穿过 LangGraph 状态通道」两组用例，
对应入口每轮无条件注入记忆的旧通路。该通路已删除：读记忆改为 Planner 循环内按需
调用的动作（切片 2 的记忆读取动作），不再是进图前的全量注入，所以那些用例连同
被测代码一起去掉。injector 本身仍是读取动作要复用的加载与渲染实现，保留在这里。

四道门本身是一句 SQL WHERE，假 repo 只能证「参数传进去了」，证不了「真的筛对了」，
所以五轴组合的合规断言在 test_research_memory_gates.py 用真库验证，这里不重复。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services import research_memory_injector_service as injector


def _row(
    memory_id: str,
    title: str,
    maturity: str,
    *,
    body: str = "",
    updated_at: str = "2026-08-01T00:00:00Z",
) -> dict:
    """参数为条目标识、标题、成熟度及展示字段；返回当前测试项目内可复核归属的仓储行。"""
    # 1. 【研究记忆】【读取回归】模拟真实仓储的归属字段，与调用方身份保持一致
    # 四道门那四个轴也在行上（真库列 SELECT i.* 一起回来），投影要把它们带给调用方复核
    return {
        "memory_id": memory_id,
        "version": 1,
        "title": title,
        "body": body,
        "memory_type": "insight",
        "maturity": maturity,
        "source_kind": "manual",
        "updated_at": updated_at,
        "user_id": "user-1",
        "project_id": "proj-x",
        "review_state": "accepted",
        "status": "confirmed",
        "stale_state": "fresh",
    }


class _FakeSettings:
    research_memory_inject_max_items = 3
    research_memory_inject_max_chars = 3600


def _item(title: str, maturity: str, *, body_summary: str = "") -> dict:
    # 模拟投影后的 MemoryReadItem，直接喂给渲染函数
    label = injector._MATURITY_LABEL[maturity]
    return {
        "memory_id": "mem-1",
        "version": 1,
        "title": title,
        "body_summary": body_summary,
        "memory_type": "insight",
        "maturity": maturity,
        "source_kind": "manual",
        "provenance_label": label,
    }


def test_render_never_marks_needs_rebind() -> None:
    """渲染层不再产出「待重绑」提示：失效记忆由四道门在 SQL 层排除，到不了这里。

    旧实现给 needs_rebind 记忆加「谨慎引用」标注，与新文案「不得作为引用来源」自相矛盾，
    整行已删。门把失效记忆排除掉这件事由 test_research_memory_gates.py 的真库组保证。
    """
    block = injector.render_memory_prompt_block([_item("ARES 支持该结论", "leaning")])
    assert "待重绑" not in block
    assert "ARES 支持该结论" in block


def test_prompt_block_declares_memory_is_not_citable() -> None:
    """段首必须交代记忆身份：本项目判断、非论文原文、不得作为引用来源。"""
    with patch.object(injector, "get_settings", return_value=_FakeSettings()):
        block = injector.render_memory_prompt_block([_item("检索增强更适合长文档", "intuition")])

    header = block.splitlines()[0]
    assert "非论文原文" in header
    assert "不得作为引用来源" in header
    # 旧文案把记忆描述成「可用、可引用」，这个说法已作废，不能残留
    assert "可引用" not in header
    # intuition 必须以「假设」身份出现，不能是中性的「直觉」
    assert "[假设]" in block
    assert "直觉" not in block


def test_injector_orders_by_maturity_and_truncates() -> None:
    """注入次序 established → leaning → intuition，超出条数上限的进 excluded_counts。"""
    captured: dict = {}

    class _FakeRepo:
        async def ensure_schema(self):
            return None

        async def list_items(self, project_id, *, user_id, review_state, status, stale_state):
            # 签名不收 maturity：加载函数若还想按成熟度分次查会直接 TypeError
            captured.update(
                project_id=project_id,
                user_id=user_id,
                review_state=review_state,
                status=status,
                stale_state=stale_state,
            )
            # repo 按 updated_at DESC 返回，这里照真实次序给
            return [
                _row("i1", "直觉判断1", "intuition", updated_at="2026-08-04T00:00:00Z"),
                _row("i2", "直觉判断2", "intuition", updated_at="2026-08-03T00:00:00Z"),
                _row("l1", "倾向判断1", "leaning", updated_at="2026-08-02T00:00:00Z"),
                _row("e1", "已确立判断1", "established", updated_at="2026-08-01T00:00:00Z"),
            ]

    with patch.object(injector, "get_settings", return_value=_FakeSettings()), \
         patch.object(injector, "get_research_memory_repository", return_value=_FakeRepo()):
        result = asyncio.run(injector.load_injectable_memory("proj-x", "user-1"))

    assert captured == {
        "project_id": "proj-x",
        "user_id": "user-1",
        "review_state": "accepted",
        "status": "confirmed",
        "stale_state": "fresh",
    }
    assert result["status"] == "success"
    # max_items=3，四条里最后一档直觉级被挤掉一条
    assert [item["maturity"] for item in result["items"]] == ["established", "leaning", "intuition"]
    assert result["items"][-1]["memory_id"] == "i1"
    assert result["excluded_counts"] == {"over_max_items": 1}
    # 身份标签在投影阶段派生，直觉级渲染为「假设」
    assert result["items"][-1]["provenance_label"] == "假设"


def test_injector_reports_failed_when_repo_unavailable() -> None:
    """记忆存储不可用 → status=failed + error_kind，items 为空但不冒充成功。"""
    class _BrokenRepo:
        async def ensure_schema(self):
            raise RuntimeError("db down")

        async def list_items(self, *args, **kwargs):
            raise RuntimeError("db down")

    with patch.object(injector, "get_settings", return_value=_FakeSettings()), \
         patch.object(injector, "get_research_memory_repository", return_value=_BrokenRepo()):
        result = asyncio.run(injector.load_injectable_memory("proj-x", "user-1"))

    assert result["status"] == "failed"
    assert result["error_kind"] == "RuntimeError"
    assert result["items"] == []
    assert result["excluded_counts"] == {}


def test_empty_is_not_merged_with_failed() -> None:
    """项目内无合格记忆是 empty，存储不可用是 failed，两者必须可分辨。"""
    class _EmptyRepo:
        async def ensure_schema(self):
            return None

        async def list_items(self, project_id, **kwargs):
            return []

    class _BrokenRepo:
        async def ensure_schema(self):
            return None

        async def list_items(self, project_id, **kwargs):
            raise ConnectionError("pool exhausted")

    with patch.object(injector, "get_settings", return_value=_FakeSettings()):
        with patch.object(injector, "get_research_memory_repository", return_value=_EmptyRepo()):
            empty = asyncio.run(injector.load_injectable_memory("proj-x", "user-1"))
        with patch.object(injector, "get_research_memory_repository", return_value=_BrokenRepo()):
            failed = asyncio.run(injector.load_injectable_memory("proj-x", "user-1"))

    assert empty["status"] == "empty"
    assert "error_kind" not in empty
    assert failed["status"] == "failed"
    assert failed["error_kind"] == "ConnectionError"
    # 两态都给空 items，唯一区分点是 status/error_kind，所以它们不能被归并
    assert empty["items"] == failed["items"] == []
    assert empty["status"] != failed["status"]
