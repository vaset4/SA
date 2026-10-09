"""项目研究空间 schema。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    name: str
    description: str = ""
    owner: str = "system"
    default_library_id: str = "default"


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    status: Optional[str] = None
    default_library_id: Optional[str] = None


class ProjectScopeRequest(BaseModel):
    library_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    excluded_paper_ids: list[str] = Field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    task_filters: list[str] = Field(default_factory=list)
    method_filters: list[str] = Field(default_factory=list)


class ProjectPaperLinkRequest(BaseModel):
    paper_id: str
    role: str = "core"
    note: str = ""
