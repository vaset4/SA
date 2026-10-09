"""来源发布前已经核验的内容版本事实，作者：xxx。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceVersionChange:
    """绑定旧指针及双方不可变清单；事务消费者只处理事实，不读取文件或调用模型。"""

    paper_id: str
    user_id: str
    previous_batch_id: str | None
    batch_id: str
    previous_manifest_sha256: str | None
    manifest_sha256: str
    classification: Literal["initial", "unchanged", "changed", "unknown"]
    source_sha256: str
    document_sha256: str
    normalized_sha256: str | None
