"""
用户请求的共享长度、集合和分页边界。

作者: xxx
时间: 2026-08-19 13:03:18
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, StringConstraints

MAX_QUERY_LENGTH = 4000
MAX_CONTEXT_SUMMARY_LENGTH = 12000
MAX_CHAT_MESSAGE_LENGTH = 12000
MAX_CHAT_HISTORY_ITEMS = 100
MAX_PAPER_IDS = 100
MAX_PAPER_ID_LENGTH = 256
MAX_PAGE_SIZE = 200
MAX_PAGE_OFFSET = 100000

UserQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH),
]
ContextSummary = Annotated[
    str,
    StringConstraints(max_length=MAX_CONTEXT_SUMMARY_LENGTH),
]
PaperId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PAPER_ID_LENGTH),
]
PaperIdList = Annotated[list[PaperId], Field(max_length=MAX_PAPER_IDS)]
ChatHistory = Annotated[list[dict[str, Any]], Field(max_length=MAX_CHAT_HISTORY_ITEMS)]


def validate_chat_history_content(history: ChatHistory) -> ChatHistory:
    """拒绝单条 content 超限的聊天历史，并保持原始列表结构。"""
    for item in history:
        content = item.get("content")
        if content is not None and len(str(content)) > MAX_CHAT_MESSAGE_LENGTH:
            raise ValueError(f"chat_history 单条 content 不能超过 {MAX_CHAT_MESSAGE_LENGTH} 字符")
    return history
