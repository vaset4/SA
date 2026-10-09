"""
论文与笔记相关路由。

这组接口面向论文库管理，而不是 graph/chat 主执行链：
上传、列表、搜索、元数据维护、删除、笔记读写都从这里进入。
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.facades import paper_facade
from app.api.facades.upload_facade import UploadTooLargeError, read_upload_bytes
from app.api.schemas.papers import NoteRequest, PaperMetadataUpdateRequest, SearchPapersRequest
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.models.upload_models import UploadConflictError
from app.services import embedded_worker_service, task_service

router = APIRouter()


@router.post("/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    library_id: str | None = Form(None),
    project_id: str | None = Form(None),
    request_id: str = Form(...),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """上传论文或兼容文档，并进入统一入库流水线。"""
    try:
        paper_facade.validate_upload(file.filename or "", file.content_type)
        raw = await read_upload_bytes(file)
        task = await task_service.create_upload_task(
            file.filename or "",
            raw,
            library_id=library_id,
            project_id=project_id,
            user_id=get_user_id(user),
            request_id=request_id,
        )
        embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
        return task
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = 415 if "MIME" in message else 422
        raise HTTPException(status_code=status_code, detail=message) from exc


@router.get("/papers")
async def list_papers(
    limit: int = 20,
    offset: int = 0,
    library_id: str | None = None,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """从业务论文库读取分页结果。"""
    return paper_facade.list_papers_page(limit, offset, library_id, user_id=get_user_id(user))


@router.post("/papers/search")
async def search_papers(
    req: SearchPapersRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """直接走 Milvus 的论文检索接口。"""
    return paper_facade.search_papers(
        query=req.query,
        top_k=req.top_k,
        library_id=req.library_id,
        year_min=req.year_min,
        year_max=req.year_max,
        category=req.category,
        authors=req.authors,
        user_id=get_user_id(user),
    )


@router.get("/papers/{paper_id}")
async def get_paper(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按 paper_id 读取论文详情。"""
    paper = paper_facade.get_paper_or_none(paper_id, user_id=get_user_id(user))
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    return paper


@router.patch("/papers/{paper_id}")
async def update_paper_metadata(
    paper_id: str,
    req: PaperMetadataUpdateRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """手动更新论文元数据。"""
    try:
        payload = req.model_dump(exclude_unset=True)
        return paper_facade.update_paper_metadata_or_raise(paper_id, payload, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/papers/{paper_id}/analysis")
async def create_paper_analysis_task(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """手动触发论文 AI 分析后台任务。"""
    try:
        paper = paper_facade.get_paper_or_none(paper_id, user_id=get_user_id(user))
        if not paper:
            raise LookupError("论文不存在")
        task = await task_service.create_paper_analysis_task(paper_id, user_id=get_user_id(user))
        embedded_worker_service.ensure_embedded_worker_running(task_types=["paper_analysis"])
        return task
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/papers/{paper_id}/metadata/autofill")
async def autofill_paper_metadata(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """排队独立元信息补全任务，返回可查回的任务而非提前更新论文。"""
    try:
        return await paper_facade.autofill_paper_metadata_or_raise(paper_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/papers/{paper_id}", status_code=202)
async def delete_paper(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """创建论文删除任务；跨存储清理由后台 worker 执行。"""
    try:
        return await paper_facade.request_paper_deletion_or_raise(paper_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}") from exc


@router.post("/notes")
async def add_note(req: NoteRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """创建研究笔记；兼容旧的论文单笔记写法。"""
    try:
        return paper_facade.create_note(req.model_dump(), user_id=get_user_id(user))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/notes")
async def list_notes(
    paper_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    note_type: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
    include_project_papers: bool = Query(default=False),
    include_project_sessions: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=500),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按对象、来源和类型查询研究笔记。"""
    return paper_facade.list_notes(
        user_id=get_user_id(user),
        paper_id=paper_id,
        project_id=project_id,
        conversation_id=conversation_id,
        session_id=session_id,
        note_type=note_type,
        source_type=source_type,
        include_project_papers=include_project_papers,
        include_project_sessions=include_project_sessions,
        limit=limit,
    )


@router.get("/notes/{paper_id}")
async def get_notes(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """读取某篇论文关联的全部笔记。"""
    return paper_facade.get_notes(paper_id, user_id=get_user_id(user))
