"""Library API schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateLibraryRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    visibility: str = "private"
    status: str = "active"
    description: str = ""
