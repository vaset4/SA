"""系统接口相关 schema。"""

from pydantic import BaseModel


class NotificationPatch(BaseModel):
    read: bool = True
