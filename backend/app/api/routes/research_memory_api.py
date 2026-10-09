"""
项目研究记忆层 HTTP 路由。

挂在 project 之下，覆盖研究记忆的手工治理端点（非 AI 部分）：
- 研究问题的增删改查
- 记忆条目的增删改查、确认、标记失效、成熟度调整
- 证据引用的绑定与解绑
- 记忆间关系的建立与删除
- 记忆对研究问题的归属与取消
- 版本历史、字面搜索、成熟度计数

异常映射沿用项目约定：LookupError → 404（不存在/越权查不到），
ValueError → 422（业务规则违反，如 evidence gate 未过、枚举非法）。
"""

from __future__ import annotations

from typing import Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from app.models.research_memory_contracts import MemoryVersionConflict
from app.services.memory_evidence_service import MemoryEvidenceError

from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import project_service, research_memory_service
from app.services import research_memory_curator_service
from app.services import research_memory_stale_service

router = APIRouter()


# ─── 请求模型 ───


class MemoryRequest(BaseModel):
    """记忆治理请求拒绝未知字段，不能把未生效命令当成功。"""
    model_config = ConfigDict(extra="forbid")


class VersionRequest(MemoryRequest):
    """用户操作必须基于其实际看到的资源版本。"""
    expected_version: int = Field(ge=1)


class CreateQuestionRequest(MemoryRequest):
    title: str = Field(..., description="研究问题标题")
    body: str = Field(default="", description="问题描述")
    is_purpose: bool = Field(default=False, description="是否为研究目的置顶项（每项目至多一条）")
    position: int = Field(default=0, description="排序位置")


class UpdateQuestionRequest(VersionRequest):
    title: str | None = Field(default=None, description="标题")
    body: str | None = Field(default=None, description="描述")
    is_purpose: bool | None = Field(default=None, description="是否研究目的")
    position: int | None = Field(default=None, description="排序位置")


class CreateMemoryRequest(MemoryRequest):
    memory_type: str = Field(..., description="记忆类型（受控枚举）")
    title: str = Field(..., description="记忆标题（一条判断的凝练）")
    body: str = Field(default="", description="正文 markdown")
    maturity: str | None = Field(default=None, description="认知成熟度 intuition/leaning/established")
    evidence_mode: str | None = Field(default=None, description="证据模式 evidence_backed/user_note_only/mixed")
    related_paper_ids: list[str] | None = Field(default=None, description="关联论文")
    tags: list[str] | None = Field(default=None, description="标签")
    question_id: str | None = Field(default=None, description="创建后归属到的研究问题")


class UpdateMemoryRequest(VersionRequest):
    status: Literal["draft"] | None = None
    confirm: bool = False
    title: str | None = Field(default=None, description="标题")
    body: str | None = Field(default=None, description="正文")
    memory_type: str | None = Field(default=None, description="记忆类型")
    evidence_mode: str | None = Field(default=None, description="证据模式")
    related_paper_ids: list[str] | None = Field(default=None, description="关联论文")
    tags: list[str] | None = Field(default=None, description="标签")


class SetMaturityRequest(VersionRequest):
    maturity: str = Field(..., description="目标成熟度 intuition/leaning/established")


class MarkStaleRequest(VersionRequest):
    reasons: list[str] = Field(default_factory=list, description="失效原因，可读展示")


class AddEvidenceRequest(VersionRequest):
    confirm: bool = False
    source_field: Literal["text", "full_text"] | None = None
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)
    paper_id: str = Field(..., description="证据所属论文")
    quote_text: str = Field(..., description="证据原句")
    chunk_id: str | None = Field(default=None, description="来源 chunk")
    parent_id: str | None = Field(default=None, description="父块")
    section: str | None = Field(default=None, description="章节")
    section_path: str | None = Field(default=None, description="章节路径")
    page_num: int | None = Field(default=None, description="页码")
    ref_role: str | None = Field(default=None, description="证据角色")


class AddLinkRequest(VersionRequest):
    dst_expected_version: int = Field(ge=1)
    dst_memory_id: str = Field(..., description="关系目标记忆")
    link_type: str = Field(..., description="关系类型 based_on/conflicts_with/supersedes")
    note: str = Field(default="", description="关系备注")


class CurateRequest(MemoryRequest):
    conversation_id: str | None = Field(default=None, description="圈定的对话")
    paper_ids: list[str] | None = Field(default=None, description="圈定的论文范围")
    question_id: str | None = Field(default=None, description="圈定的研究问题")


class AcceptDraftRequest(VersionRequest):
    title: str | None = Field(default=None, description="接纳前可编辑标题")
    body: str | None = Field(default=None, description="接纳前可编辑正文")
    maturity: str | None = Field(default=None, description="接纳前可调成熟度")
    memory_type: str | None = Field(default=None, description="接纳前可改类型")


# ─── 异常映射辅助 ───


def _raise_http(exc: Exception) -> NoReturn:
    if isinstance(exc, MemoryVersionConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, MemoryEvidenceError):
        raise HTTPException(status_code=422, detail={"detail": str(exc), "candidates": exc.candidates}) from exc
    # 业务异常统一转 HTTP：LookupError→404，ValueError→422
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


async def _guard_project(project_id: str, user: dict) -> None:
    # 归属校验：项目不属当前用户时按「查不到即 404」处理（对齐项目约定，不返回 403）
    try:
        await project_service.require_project(project_id, user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



# ─── 研究问题 ───


@router.get("/projects/{project_id}/questions")
async def list_questions(
    project_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；查询研究问题，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    return await research_memory_service.list_questions(project_id, user_id=get_user_id(user))



@router.post("/projects/{project_id}/questions")
async def create_question(
    project_id: str,
    req: CreateQuestionRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；创建研究问题，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.create_question(project_id, req.model_dump(), user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.patch("/projects/{project_id}/questions/{question_id}")
async def update_question(
    project_id: str,
    question_id: str,
    req: UpdateQuestionRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；编辑研究问题，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    patch = {k: v for k, v in req.model_dump(exclude={"expected_version", "confirm"}).items() if v is not None}
    try:
        return await research_memory_service.update_question(project_id, question_id, patch, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.delete("/projects/{project_id}/questions/{question_id}")
async def delete_question(
    project_id: str,
    question_id: str,
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, str]:
    """参数为可信项目、用户与资源请求；删除研究问题，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        await research_memory_service.delete_question(project_id, question_id, user_id=get_user_id(user), expected_version=expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)
    return {"status": "deleted"}



# ─── 记忆条目 ───


@router.get("/projects/{project_id}/memory")
async def list_memory(
    project_id: str,
    question_id: str | None = Query(default=None, description="按研究问题过滤"),
    memory_type: str | None = Query(default=None, description="按类型过滤"),
    paper_id: str | None = Query(default=None, description="按关联论文过滤"),
    tag: str | None = Query(default=None, description="按标签过滤"),
    status: str | None = Query(default=None, description="按生命周期过滤"),
    stale_state: str | None = Query(default=None, description="按新鲜度过滤"),
    maturity: str | None = Query(default=None, description="按成熟度过滤"),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；查询记忆管理列表，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    filters = {
        "question_id": question_id,
        "memory_type": memory_type,
        "paper_id": paper_id,
        "tag": tag,
        "status": status,
        "stale_state": stale_state,
        "maturity": maturity,
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    try:
        return await research_memory_service.list_items(project_id, user_id=get_user_id(user), **filters)
    except ValueError as exc:
        _raise_http(exc)



@router.get("/projects/{project_id}/memory/search")
async def search_memory(
    project_id: str,
    q: str = Query(..., description="字面搜索关键词"),
    stale_state: str | None = Query(default=None, description="按新鲜度过滤"),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；搜索记忆，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.search_memories_literal(project_id, q, stale_state=stale_state, user_id=get_user_id(user))
    except ValueError as exc:
        _raise_http(exc)



@router.get("/projects/{project_id}/memory/maturity_counts")
async def maturity_counts(
    project_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, int]:
    """参数为可信项目、用户与资源请求；统计成熟度，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    return await research_memory_service.count_by_maturity(project_id, user_id=get_user_id(user))



@router.post("/projects/{project_id}/memory")
async def create_memory(
    project_id: str,
    req: CreateMemoryRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；创建手工判断，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        return await research_memory_service.create_item(project_id, payload, user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.post("/projects/{project_id}/memory/curate")
async def curate_memory(
    project_id: str,
    req: CurateRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；整理待审候选，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    scope = {k: v for k, v in req.model_dump(exclude={"expected_version", "confirm"}).items() if v is not None}
    try:
        return await research_memory_curator_service.curate(project_id, scope, user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        _raise_http(exc)


@router.get("/projects/{project_id}/memory/review")
async def list_review_queue(
    project_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；查询待审记忆，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    return await research_memory_service.list_review_queue(project_id, user_id=get_user_id(user))


@router.post("/projects/{project_id}/memory/check_stale")
async def check_project_stale(
    project_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    # 校验项目下全部 evidence_backed 记忆的证据有效性，失效标 needs_rebind
    """参数为可信项目、用户与资源请求；复核项目记忆来源，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    return await research_memory_stale_service.check_evidence_staleness(project_id, user_id=get_user_id(user))


@router.get("/projects/{project_id}/memory/{memory_id}")
async def get_memory(
    project_id: str,
    memory_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；读取具体记忆详情，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.get_detail(memory_id, project_id=project_id, user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.patch("/projects/{project_id}/memory/{memory_id}")
async def update_memory(
    project_id: str,
    memory_id: str,
    req: UpdateMemoryRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；编辑并按显式意图确认记忆，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    patch = {k: v for k, v in req.model_dump(exclude={"expected_version", "confirm"}).items() if v is not None}
    try:
        return await research_memory_service.update_item(memory_id, patch, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version, confirm=req.confirm)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.delete("/projects/{project_id}/memory/{memory_id}")
async def delete_memory(
    project_id: str,
    memory_id: str,
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, str]:
    """参数为可信项目、用户与资源请求；删除具体记忆，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        await research_memory_service.delete_item(memory_id, project_id=project_id, user_id=get_user_id(user), expected_version=expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)
    return {"status": "deleted"}



@router.post("/projects/{project_id}/memory/{memory_id}/confirm")
async def confirm_memory(
    project_id: str,
    memory_id: str,
    req: VersionRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；确认当前记忆支持，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.confirm_item(memory_id, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.post("/projects/{project_id}/memory/{memory_id}/maturity")
async def set_maturity(
    project_id: str,
    memory_id: str,
    req: SetMaturityRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；调整记忆成熟度，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.set_maturity(memory_id, req.maturity, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.post("/projects/{project_id}/memory/{memory_id}/stale")
async def mark_stale(
    project_id: str,
    memory_id: str,
    req: MarkStaleRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；标记当前依据失效，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.mark_stale(memory_id, req.reasons, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.get("/projects/{project_id}/memory/{memory_id}/versions")
async def list_versions(
    project_id: str,
    memory_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；读取版本历史，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.list_versions(memory_id, project_id=project_id, user_id=get_user_id(user))
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



# ─── 证据 ───


@router.post("/projects/{project_id}/memory/{memory_id}/evidence")
async def add_evidence(
    project_id: str,
    memory_id: str,
    req: AddEvidenceRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；绑定原文并可同时确认，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    payload = {k: v for k, v in req.model_dump(exclude={"expected_version", "confirm"}).items() if v is not None}
    try:
        return await research_memory_service.add_evidence(memory_id, payload, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version, confirm=req.confirm)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.delete("/projects/{project_id}/memory/{memory_id}/evidence/{ref_id}")
async def remove_evidence(
    project_id: str,
    memory_id: str,
    ref_id: str,
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, str]:
    """参数为可信项目、用户与资源请求；移除指定依据，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.remove_evidence(memory_id, ref_id, project_id=project_id, user_id=get_user_id(user), expected_version=expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



# ─── 记忆关系 ───


@router.get("/projects/{project_id}/memory_links")
async def list_links(
    project_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> list[dict[str, Any]]:
    """参数为可信项目、用户与资源请求；查询同归属记忆关系，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    return await research_memory_service.list_links(project_id, user_id=get_user_id(user))



@router.post("/projects/{project_id}/memory/{memory_id}/links")
async def add_link(
    project_id: str,
    memory_id: str,
    req: AddLinkRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；建立两端关系，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.add_link(
            project_id, memory_id, req.dst_memory_id, req.link_type, note=req.note, user_id=get_user_id(user), expected_version=req.expected_version, dst_expected_version=req.dst_expected_version
        )
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.delete("/projects/{project_id}/memory_links/{link_id}")
async def remove_link(
    project_id: str,
    link_id: str,
    dst_expected_version: int = Query(..., ge=1),
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, str]:
    """参数为可信项目、用户与资源请求；移除两端关系，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        await research_memory_service.remove_link(link_id, project_id=project_id, user_id=get_user_id(user), expected_version=expected_version, dst_expected_version=dst_expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)
    return {"status": "deleted"}



# ─── 归属研究问题 ───


@router.post("/projects/{project_id}/memory/{memory_id}/questions/{question_id}")
async def attach_question(
    project_id: str,
    memory_id: str,
    question_id: str,
    req: VersionRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；关联研究问题，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.attach_question(memory_id, question_id, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.delete("/projects/{project_id}/memory/{memory_id}/questions/{question_id}")
async def detach_question(
    project_id: str,
    memory_id: str,
    question_id: str,
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；取消问题关联，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.detach_question(memory_id, question_id, project_id=project_id, user_id=get_user_id(user), expected_version=expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



# ─── 待审区（AI 草稿 curate / review / accept / dismiss，B6 置信度） ───


@router.post("/projects/{project_id}/memory/{memory_id}/accept")
async def accept_draft(
    project_id: str,
    memory_id: str,
    req: AcceptDraftRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；接纳已核验草稿，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    patch = {k: v for k, v in req.model_dump(exclude={"expected_version", "confirm"}).items() if v is not None}
    try:
        return await research_memory_service.accept_draft(memory_id, patch, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



@router.post("/projects/{project_id}/memory/{memory_id}/dismiss")
async def dismiss_draft(
    project_id: str,
    memory_id: str,
    req: VersionRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """参数为可信项目、用户与资源请求；拒绝待审草稿，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_service.dismiss_draft(memory_id, project_id=project_id, user_id=get_user_id(user), expected_version=req.expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)



# ─── 证据失效检测（B4） ───


@router.post("/projects/{project_id}/memory/{memory_id}/check_stale")
async def check_memory_stale(
    project_id: str,
    memory_id: str,
    expected_version: int = Query(..., ge=1),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    # 校验单条记忆的证据有效性
    """参数为可信项目、用户与资源请求；复核具体记忆来源，返回真实结果或明确错误。"""
    await _guard_project(project_id, user)
    try:
        return await research_memory_stale_service.check_evidence_staleness(project_id, memory_id,
            user_id=get_user_id(user), expected_version=expected_version)
    except (LookupError, ValueError) as exc:
        _raise_http(exc)
