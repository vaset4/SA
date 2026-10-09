"""Neo4j 运维管理 schema。"""

from __future__ import annotations

from pydantic import BaseModel


class Neo4jRebuildPaperRequest(BaseModel):
    paper_id: str


class Neo4jBackfillLibraryRequest(BaseModel):
    library_id: str
