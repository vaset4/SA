"""图谱任务只能消费已核验原文的边界回归，作者：xxx。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.tasks import LightRAGTaskRequest
from app.tasks import graph_task


@pytest.mark.asyncio
async def test_legacy_graph_task_cannot_bypass_published_source():
    """历史正文直写任务显式拒绝，不能跳过用户归属或核验批次。"""
    with pytest.raises(ValueError, match="发布版本"):
        await graph_task.run_lightrag_build_task({"task_id": "old", "user_id": "a",
            "payload": {"paper_id": "paper", "full_text": "unverified text"}})


@pytest.mark.parametrize("extra", [{"full_text": "forged"}, {"library_id": "other-library"}])
def test_graph_request_rejects_caller_supplied_source(extra):
    """HTTP 合同只接受论文 ID，客户端不能替换建图原文或业务范围。"""
    with pytest.raises(ValidationError):
        LightRAGTaskRequest.model_validate({"paper_id": "paper", **extra})
