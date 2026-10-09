"""项目研究空间模型。"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from app.models.library_models import DEFAULT_LIBRARY_ID

DEFAULT_PROJECT_STATUS = "active"
DEFAULT_PROJECT_OWNER = "system"
DEFAULT_PROJECT_PAPER_ROLE = "core"

ProjectStatus = Literal["active", "archived"]
ProjectPaperRole = Literal["core", "reference", "candidate"]


class Project(TypedDict):
    project_id: str
    user_id: str
    name: str
    description: str
    owner: str
    status: ProjectStatus
    default_library_id: str
    created_at: str
    updated_at: str


class ProjectScope(TypedDict):
    project_id: str
    library_ids: list[str]
    paper_ids: list[str]
    excluded_paper_ids: list[str]
    year_from: int | None
    year_to: int | None
    task_filters: list[str]
    method_filters: list[str]
    created_at: str
    updated_at: str


class ProjectCreatePayload(TypedDict, total=False):
    project_id: str
    user_id: str
    name: str
    description: str
    owner: str
    status: ProjectStatus
    default_library_id: str


class ProjectUpdatePayload(TypedDict, total=False):
    user_id: str
    name: str
    description: str
    owner: str
    status: ProjectStatus
    default_library_id: str
    updated_at: NotRequired[str]


class ProjectScopePayload(TypedDict, total=False):
    library_ids: list[str]
    paper_ids: list[str]
    excluded_paper_ids: list[str]
    year_from: int | None
    year_to: int | None
    task_filters: list[str]
    method_filters: list[str]


class ResolvedProjectScope(TypedDict):
    project_id: str
    library_ids: list[str]
    paper_ids: list[str]
    excluded_paper_ids: list[str]
    year_from: int | None
    year_to: int | None
    task_filters: list[str]
    method_filters: list[str]


def empty_scope(default_library_id: str = DEFAULT_LIBRARY_ID) -> ProjectScopePayload:
    return {
        "library_ids": [default_library_id],
        "paper_ids": [],
        "excluded_paper_ids": [],
        "year_from": None,
        "year_to": None,
        "task_filters": [],
        "method_filters": [],
    }
