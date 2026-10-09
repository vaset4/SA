"""通知仓储。"""

from __future__ import annotations

from typing import Any

from app.repositories.common import normalize_record
from app.services.database import fetch, fetchval

NOTIFICATION_LIMIT_DEFAULT = 50
PUSH_NOTIFICATION_EVENT = "push_notification"
_repository: "NotificationRepository | None" = None

# 通知目前是“从 logs 表里筛事件类型”得到的，并没有独立 notifications 表。
# 也就是说，通知本质上是 observability 事件的一种特殊视图。


def _to_notification(row: Any) -> dict[str, Any]:
    data = normalize_record(row) or {}
    data["read"] = bool(data.pop("is_read", False))
    return data


class NotificationRepository:
    """封装通知读取与状态更新。"""

    async def list_notifications(self, user_id: str, limit: int = NOTIFICATION_LIMIT_DEFAULT) -> list[dict[str, Any]]:
        """列出当前用户的 push_notification 事件，并映射为前端通知结构。"""
        rows = await fetch(
            """
            SELECT log_id, session_id, event_type, node_name, query, result_count, latency_ms, created_at, is_read
            FROM logs WHERE event_type=$1 AND user_id=$2 ORDER BY created_at DESC LIMIT $3
            """,
            PUSH_NOTIFICATION_EVENT,
            user_id,
            limit,
        )
        return [_to_notification(row) for row in rows]

    async def set_read_state(self, log_id: str, user_id: str, read: bool) -> bool:
        """更新当前用户通知的已读状态；跨用户或不存在返回 False。"""
        return bool(
            await fetchval(
                "UPDATE logs SET is_read=$1 WHERE log_id=$2 AND user_id=$3 RETURNING 1",
                bool(read),
                log_id,
                user_id,
            )
        )


def get_notification_repository() -> NotificationRepository:
    """返回通知仓储单例。"""
    global _repository
    if _repository is None:
        _repository = NotificationRepository()
    return _repository
