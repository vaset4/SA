"""【研究交付】【旧客户端边界】防止消息批量替换破坏服务端答案，作者：xxx，2026-10-04 18:00:00。"""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.models.result_delivery_contracts import DeliveryConflict
from app.repositories.conversation_repository import ConversationRepository
from app.services import database
from result_delivery_test_utils import Connection, Pool


class LegacyMessageConnection(Connection):
    """只允许历史替换需要的语句；记录是否触碰已有答案。"""

    def __init__(self, *, committed=False, fail_insert=False):
        """参数为已提交与插入故障标记；初始化可回滚离线连接。"""
        super().__init__()
        self.committed = committed
        self.fail_insert = fail_insert
        self.data["messages"] = {"original": {"content": "原答案"}}

    async def fetchrow(self, query, *args):
        """参数为查询；仅识别对话锁和服务端交付检查。"""
        if "FROM conversations" in query:
            assert "FOR UPDATE" in query
            return {"id": "conversation"}
        if "FROM messages" in query:
            return {"id": "original"} if self.committed else None
        raise AssertionError(query)

    async def execute(self, query, *args):
        """参数为写语句；模拟历史替换失败及回滚，不连接真实数据库。"""
        if "INSERT INTO conversations" in query or "UPDATE conversations" in query:
            return
        if query.startswith("DELETE FROM messages"):
            self.data["messages"] = {}
            return
        if "INSERT INTO messages" in query:
            if self.fail_insert:
                raise OSError("离线消息写入失败")
            self.data["messages"][args[0]] = {"content": args[3]}
            return
        raise AssertionError(query)


@pytest.mark.asyncio
@pytest.mark.parametrize("committed,metadata", [(True, {}), (False, {"persistence_status": "saved"})])
async def test_legacy_replace_cannot_overwrite_or_forge_delivery(monkeypatch, committed, metadata):
    """参数为旧页面回写和伪造元数据；拒绝并完整保留已保存列表。"""
    connection = LegacyMessageConnection(committed=committed)
    monkeypatch.setattr(database, "get_pool", AsyncMock(return_value=Pool(connection)))
    with pytest.raises(DeliveryConflict):
        await ConversationRepository().set_messages("user", "conversation", [
            {"role": "assistant", "content": "迟到正文", "metadata": metadata}])
    assert connection.data["messages"] == {"original": {"content": "原答案"}}


@pytest.mark.asyncio
async def test_legacy_replace_failure_rolls_back_original_messages(monkeypatch):
    """参数为数据库端口；合法历史替换中途失败不能留下空对话。"""
    connection = LegacyMessageConnection(fail_insert=True)
    monkeypatch.setattr(database, "get_pool", AsyncMock(return_value=Pool(connection)))
    with pytest.raises(OSError):
        await ConversationRepository().set_messages("user", "conversation", [{"role": "user", "content": "新问题"}])
    assert connection.data["messages"] == {"original": {"content": "原答案"}}


@pytest.mark.asyncio
async def test_legacy_replace_route_exposes_conflict(monkeypatch):
    """参数为服务端口；正式路由明确返回409，不能把冲突包装成成功。"""
    from app.api.routes import conversation_api
    from app.api.schemas.conversation import SetMessagesRequest

    monkeypatch.setattr(conversation_api.conversation_facade, "set_messages_or_raise",
                        AsyncMock(side_effect=DeliveryConflict("请刷新")))
    with pytest.raises(HTTPException) as failure:
        await conversation_api.set_messages("conversation", SetMessagesRequest(messages=[]), user={"user_id": "user"})
    assert failure.value.status_code == 409
