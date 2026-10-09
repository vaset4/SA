"""【研究记忆】【候选整理】圈定来源并保存可审核草稿，作者：xxx。"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.models.research_memory_contracts import RefRole, VALID_MEMORY_TYPES
from app.repositories.conversation_repository import get_conversation_repository
from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.llm_service import OpenAICompatibleLLM, get_llm
from app.services.memory_evidence_service import MemoryEvidenceError, _manifest, validate_memory_evidence
from app.repositories.memory_source_repository import read_source_state


class CuratorEvidenceCandidate(BaseModel):
    """模型只提议原句；实际来源、归属、区间和支持关系由服务重新核对。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    paper_id: str = Field(min_length=1)
    chunk_id: str = ""
    quote: str = ""
    source_field: str = "full_text"
    source_start: int | None = None
    source_end: int | None = None
    ref_role: RefRole = "primary_support"
    support: dict[str, Any] = Field(default_factory=dict)


class CuratorMemoryCandidate(BaseModel):
    """候选内容与明确来源支持；不会自动获得用户确认。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1)
    body: str = ""
    memory_type: str = Field(min_length=1)
    is_synthesis: bool = False
    evidence: list[CuratorEvidenceCandidate] = Field(default_factory=list)
    source_claim_id: str = ""
    source_attempt_id: str = ""


class CuratorCandidateBatch(BaseModel):
    """一次整理实际提出的候选集合。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    candidates: list[CuratorMemoryCandidate] = Field(default_factory=list)


def _curator_llm() -> Any:
    """无参数；返回既有配置的整理模型，不改变共享模型实例。"""
    settings = get_settings()
    model = (settings.research_memory_curator_model or "").strip()
    if not model or model == settings.llm_model_name:
        return get_llm()
    if not settings.llm_api_key or not settings.llm_base_url:
        raise ValueError("curator 独立模型需要 LLM_API_KEY 与 LLM_BASE_URL")
    return OpenAICompatibleLLM(model=model, api_key=settings.llm_api_key, base_url=settings.llm_base_url,
        temperature=settings.llm_temperature, timeout=settings.llm_timeout,
        min_request_interval_seconds=settings.llm_min_request_interval_seconds)


async def curate(project_id: str, scope: dict[str, Any], *, user_id: str) -> list[dict[str, Any]]:
    """参数为可信项目、圈定材料与账号；核验材料后请求整理，返回待审候选详情。"""
    from app.services.project_service import require_project

    await require_project(project_id, user_id=user_id)
    if not any(scope.get(key) for key in ("conversation_id", "paper_ids", "question_id")):
        raise ValueError("整理必须圈定对话、论文或研究问题")
    if scope.get("question_id"):
        question = await get_research_memory_repository().get_question(scope["question_id"], user_id=user_id, project_id=project_id)
        if not question or question.get("project_id") != project_id or question.get("user_id") != user_id:
            raise LookupError("研究问题不存在或无权访问")
    paper_ids = list(dict.fromkeys(scope.get("paper_ids") or []))
    if len(paper_ids) > get_settings().research_memory_curate_max_papers:
        raise ValueError("所选论文超过本次整理范围，请缩小选择")
    material = await _gather_material(project_id, scope, paper_ids, user_id=user_id)
    # 1. 【研究记忆】【发送前授权】收集原文后再次核对，撤权或新发布不能沿用旧输入继续整理
    for paper in material["papers"]:
        current = await read_source_state(user_id=user_id, project_id=project_id, paper_id=paper["paper_id"])
        if current["current_batch_id"] != paper["batch_id"]:
            raise MemoryEvidenceError("整理期间来源已更新，请重新选择材料")
    batch = await _invoke_curator_llm(material)
    # 1. 【研究记忆】【整理来源】模型没有权利把手动整理冒充已提交主张或另换输入论文
    allowed = set(paper_ids)
    for candidate in batch.candidates:
        if candidate.source_claim_id or candidate.source_attempt_id:
            raise MemoryEvidenceError("整理提议不能伪造研究主张身份")
        if any(ref.paper_id not in allowed for ref in candidate.evidence):
            raise MemoryEvidenceError("整理提议引用了本次圈定范围之外的论文")
    return await persist_candidates(project_id, batch.candidates, user_id=user_id,
        source_answer_id="curate:" + uuid.uuid4().hex, source_conversation_id=scope.get("conversation_id"))


async def _gather_material(project_id: str, scope: dict[str, Any], paper_ids: list[str], *, user_id: str) -> dict[str, Any]:
    """参数为可信范围；仅收集有权限的会话和论文原文，返回模型可见材料。"""
    conversation_text = ""
    if scope.get("conversation_id"):
        conversation = await get_conversation_repository().get_conversation(user_id, str(scope["conversation_id"]))
        if not conversation or conversation.get("project_id") != project_id:
            raise LookupError("对话不属于当前项目")
        conversation_text = "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in conversation.get("messages", []))
    existing = await get_research_memory_repository().list_items(project_id, user_id=user_id, review_state="accepted")
    papers = []
    budget = get_settings().research_memory_curate_paper_char_budget
    for paper_id in paper_ids:
        state = await read_source_state(user_id=user_id, project_id=project_id, paper_id=paper_id)
        manifest = await _manifest(state["current_batch_id"], paper_id=paper_id, user_id=user_id)
        chunks = [{"chunk_id": chunk["chunk_id"], "text": chunk.get("text") or chunk.get("full_text") or ""}
                  for chunk in manifest["chunks"]]
        text = "\n".join(f"[{chunk['chunk_id']}] {chunk['text']}" for chunk in chunks)
        papers.append({"paper_id": paper_id, "batch_id": state["current_batch_id"], "full_text": text[:budget], "truncated": len(text) > budget})
    return {"conversation_text": conversation_text, "existing_titles": [item["title"] for item in existing], "papers": papers}


async def _invoke_curator_llm(material: dict[str, Any]) -> CuratorCandidateBatch:
    """参数为已核验材料；返回结构化提议，模型或格式错误直接暴露。"""
    import json

    prompt = ("提炼项目中有持续价值的判断为待审核记忆。只输出 CuratorCandidateBatch JSON。"
              "evidence 的 paper_id/chunk_id/quote 必须来自给定原文，source_field=text。"
              "没有文献依据时明确保留个人判断身份。不能修改来源身份，不输出已确认状态。\n"
              + json.dumps(material, ensure_ascii=False))
    return await _curator_llm().with_structured_output(CuratorCandidateBatch).ainvoke([HumanMessage(content=prompt)])


async def persist_candidates(project_id: str, candidates: list[CuratorMemoryCandidate], *, user_id: str,
    source_answer_id: str, source_conversation_id: str | None = None, source_session_id: str | None = None,
    source_delivery_revision: int | None = None) -> list[dict[str, Any]]:
    """参数为提议与已提交来源身份；原子保存草稿、支持及初始版本，按来源主张幂等。"""
    if not source_answer_id:
        raise ValueError("候选保存需要明确的来源回答或整理操作身份")
    repo = get_research_memory_repository()
    await repo.ensure_schema()
    drafts = []
    for candidate in candidates:
        if candidate.memory_type not in VALID_MEMORY_TYPES:
            raise ValueError("候选记忆类型无效")
        refs, errors = [], []
        for evidence in candidate.evidence:
            try:
                refs.append(await validate_memory_evidence(evidence.model_dump(exclude_none=True), user_id=user_id, project_id=project_id))
            except (MemoryEvidenceError, LookupError) as error:
                errors.append(str(error))
        payload: dict[str, Any] = {"memory_type": candidate.memory_type, "title": candidate.title, "body": candidate.body,
            "status": "draft", "review_state": "pending_review", "source_kind": "agent_draft", "user_id": user_id,
            "evidence_mode": "evidence_backed" if candidate.evidence else "user_note_only",
            "stale_state": "needs_rebind" if errors else "fresh", "stale_reasons": errors,
            "maturity": "intuition" if errors or not refs else "leaning",
            "source_answer_id": source_answer_id, "source_attempt_id": candidate.source_attempt_id or None,
            "source_claim_id": candidate.source_claim_id or None, "source_conversation_id": source_conversation_id,
            "source_session_id": source_session_id, "confidence_signals": {"evidence_count": len(refs),
                "validation_errors": errors, "is_synthesis": candidate.is_synthesis,
                "proposed_evidence": [ref.model_dump(mode="json") for ref in candidate.evidence] if errors else []}}
        if source_delivery_revision is not None:
            payload["source_delivery_revision"] = source_delivery_revision
        normalized = " ".join(candidate.title.lower().split()) + "\0" + " ".join(candidate.body.lower().split())
        identity = candidate.source_claim_id or hashlib.sha256(normalized.encode()).hexdigest()
        key = hashlib.sha256((source_answer_id + "\0" + candidate.source_attempt_id + "\0" + identity).encode()).hexdigest()
        drafts.append(await repo.create_candidate(project_id, payload, refs, candidate_key=key))
    return drafts
