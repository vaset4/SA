"""知识库治理层模型。"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

DEFAULT_LIBRARY_ID = "default"
DEFAULT_LIBRARY_NAME = "默认知识库"
DEFAULT_LIBRARY_OWNER = "system"
DEFAULT_LIBRARY_VISIBILITY = "private"
DEFAULT_LIBRARY_STATUS = "active"

LibraryVisibility = Literal["private", "shared", "public"]
LibraryStatus = Literal["active", "archived", "disabled"]


class Library(TypedDict):
    """最小知识库实体。"""

    library_id: str
    name: str
    owner: str
    visibility: LibraryVisibility
    status: LibraryStatus
    description: str
    created_at: str
    updated_at: str


class LibraryCreatePayload(TypedDict, total=False):
    """创建知识库时的输入。"""

    library_id: str
    name: str
    owner: str
    visibility: LibraryVisibility
    status: LibraryStatus
    description: str


class LibraryUpdatePayload(TypedDict, total=False):
    """更新知识库时的输入。"""

    name: str
    owner: str
    visibility: LibraryVisibility
    status: LibraryStatus
    description: str
    updated_at: NotRequired[str]
