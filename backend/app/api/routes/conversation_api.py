"""
前端会话管理路由。

这个文件服务的是“前端展示层会话库”，不是 LangGraph checkpoint。
因此它关注的是：
- 会话标题、置顶、标签、关联论文
- 持久化消息列表
- UI 需要的 CRUD 操作

理解时要把它和 `chat_api.py` 区分开：
`chat_api.py` 负责驱动智能体回答，
这里负责把前端看到的会话列表和消息记录保存下来。
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.result_delivery_contracts import DeliveryConflict

from app.api.facades import conversation_facade
from app.api.schemas.request_limits import MAX_PAGE_OFFSET, MAX_PAGE_SIZE
from app.api.schemas.conversation import (
    AddMessageRequest,
    AddPaperRequest,
    CreateConversationRequest,
    SetMessagesRequest,
    UpdateConversationRequest,
)
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id

router = APIRouter()


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0, le=MAX_PAGE_OFFSET),
    sort: Literal["updated_at", "created_at", "title", "message_count"] = "updated_at",
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """获取前端会话库里的对话列表。"""
    return await conversation_facade.list_conversations(get_user_id(user), limit, offset, sort)


@router.post("/conversations")
async def create_conversation(
    req: CreateConversationRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """创建前端会话记录。"""
    return await conversation_facade.create_conversation(
        get_user_id(user),
        req.title,
        req.associated_papers,
        req.session_id,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """获取对话详情与持久化消息。"""
    conversation = await conversation_facade.get_conversation_or_none(get_user_id(user), conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """更新对话元数据。"""
    try:
        return await conversation_facade.update_conversation_or_raise(
            user_id=get_user_id(user),
            conversation_id=conversation_id,
            title=req.title,
            is_pinned=req.is_pinned,
            tags=req.tags,
            associated_papers=req.associated_papers,
            session_id=req.session_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """删除前端会话记录。"""
    try:
        return await conversation_facade.delete_conversation_or_raise(get_user_id(user), conversation_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    req: AddMessageRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """向前端会话库追加一条消息。"""
    try:
        return await conversation_facade.add_message(
            user_id=get_user_id(user),
            conversation_id=conversation_id,
            role=req.role,
            content=req.content,
            title=req.title,
            tone=req.tone,
            render_mode=req.render_mode,
            metadata=req.metadata,
        )
    except DeliveryConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc



@router.put("/conversations/{conversation_id}/messages")
async def set_messages(
    conversation_id: str,
    req: SetMessagesRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    批量覆盖某个对话的消息列表。

    这个接口常用于前端把本地完整消息状态一次性回写后端，
    因而仓储层采用的是“先清空再重建”的覆盖语义。
    """
    try:
        return await conversation_facade.set_messages_or_raise(
            get_user_id(user),
            conversation_id,
            req.messages,
            session_id=req.session_id,
        )
    except DeliveryConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/papers")
async def add_paper_to_conversation(
    conversation_id: str,
    req: AddPaperRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """给对话增加关联论文。"""
    try:
        return await conversation_facade.add_paper_to_conversation_or_raise(
            get_user_id(user),
            conversation_id,
            req.paper_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}/papers/{paper_id}")
async def remove_paper_from_conversation(
    conversation_id: str,
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """从对话中移除关联论文。"""
    try:
        return await conversation_facade.remove_paper_from_conversation_or_raise(
            get_user_id(user),
            conversation_id,
            paper_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
