"""
后台任务路由。

这组接口是异步任务体系的 HTTP 入口，适合上传入库、深度研究、建图等长耗时流程。
它们大多只负责创建 task / 查询状态，真正执行由 worker 完成。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile

from app.api.facades import paper_facade, session_identity_facade
from app.api.facades.upload_facade import UploadTooLargeError, read_upload_bytes
from app.api.schemas.request_limits import MAX_PAGE_OFFSET, MAX_PAGE_SIZE
from app.api.schemas.tasks import LightRAGTaskRequest, ResearchTaskRequest
from app.models.upload_models import UploadConflictError
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import embedded_worker_service, task_service, upload_service

router = APIRouter()


class UploadSourceResponse(Response):
    """实际响应体消费持有来源占用，不能把文件路径返回当作下载已经完成。"""

    def __init__(self, paper_id: str, user_id: str) -> None:
        """参数为可信论文与账号；创建延迟取得文件占用的下载响应。"""
        super().__init__()
        self.paper_id = paper_id
        self.user_id = user_id

    async def __call__(self, scope, receive, send) -> None:
        """参数为 ASGI 请求与消费端口；下载完成或断开后释放实际文件占用。"""
        from urllib.parse import quote
        from app.services.upload_read_service import open_upload_source

        try:
            async with open_upload_source(self.paper_id, user_id=self.user_id) as source:
                response = Response(source["content"], media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(source["filename"])})
                await response(scope, receive, send)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/papers/{paper_id}/source")
async def download_upload_source(paper_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """【上传】【原文下载】按 owner 核对持久来源，返回与原指纹一致的文件字节。"""
    return UploadSourceResponse(paper_id, get_user_id(user))


@router.get("/papers/{paper_id}/upload")
async def get_upload_status(paper_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """刷新后读取上传、元信息和可选图谱的持久状态。"""
    try:
        return await upload_service.get_paper_upload_status(paper_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/papers/{paper_id}/chunks/{chunk_id}")
async def get_published_chunk(paper_id: str, chunk_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """读取用户有权访问的原引用全文，切换索引后仍固定在原发布版本。"""
    from app.services.upload_read_service import read_published_chunk_async

    try:
        chunk = await read_published_chunk_async(chunk_id, user_id=get_user_id(user), paper_id=paper_id)
        return chunk
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/upload")
async def create_upload_task(
    file: UploadFile = File(...),
    library_id: str | None = Form(None),
    project_id: str | None = Form(None),
    request_id: str = Form(...),
    graph_requested: bool = Form(False),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    创建上传入库后台任务。

    这里会在任务入队后确保至少有一个内嵌 worker 起来消费 `upload_ingest` 任务，
    适合没有独立 worker 进程的轻量部署模式。
    """
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
            graph_requested=graph_requested,
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


@router.get("/tasks/upload/requests/{request_id}")
async def get_upload_receipt(request_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """通过稳定受理标识查回原任务，响应丢失后不生成第二次上传。"""
    try:
        return await upload_service.lookup_upload_request(request_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/resume")
async def resume_upload_task(task_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """继续本账号的已暂停上传，连续点击仍复用原任务。"""
    try:
        task = await upload_service.continue_upload(task_id, user_id=get_user_id(user))
        embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
        return task
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/papers/{paper_id}/upload/{operation}")
async def request_upload_operation(paper_id: str, operation: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """请求重新处理、元信息补全或可选图谱任务；基本可用状态与增强状态分开。"""
    try:
        task = await upload_service.request_upload_operation(paper_id, operation, user_id=get_user_id(user))
        embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
        return task
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/tasks/research")
async def create_research_task(
    req: ResearchTaskRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """创建 deep research 后台任务，并校验会话归属后写入任务 owner。"""
    user_id = get_user_id(user)
    try:
        identity = await session_identity_facade.resolve_conversation_identity(
            user_id=user_id,
            conversation_id=None,
            session_id=req.session_id,
            allow_missing_conversation=True,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    return await task_service.create_research_task(req.query, identity.session_id, user_id=user_id)


@router.post("/tasks/lightrag_build")
async def create_lightrag_task(
    req: LightRAGTaskRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """【上传】【可选图谱】兼容原建图地址，统一从已授权发布版本建立独立任务。"""
    try:
        task = await task_service.create_lightrag_task(req.paper_id, user_id=get_user_id(user))
        embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
        return task
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/tasks")
async def list_tasks(
    statuses: list[str] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0, le=MAX_PAGE_OFFSET),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """分页列出当前用户的后台任务。"""
    return await task_service.list_tasks(statuses, user_id=get_user_id(user), limit=limit, offset=offset)


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """读取单条后台任务；越权或不存在统一 404。"""
    try:
        return await task_service.get_task_or_raise(task_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """取消后台任务；越权或不存在统一 404。"""
    try:
        return await task_service.cancel_task(task_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/tasks/{task_id}/events")
async def get_task_events(
    task_id: str,
    limit: int = Query(default=80, ge=1, le=200),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """读取当前用户任务的事件流；越权或不存在统一 404。"""
    try:
        await task_service.get_task_or_raise(task_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await task_service.list_task_events(task_id, user_id=get_user_id(user), limit=limit)
