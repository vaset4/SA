"""
项目研究空间路由。

`project` 是本项目里很重要的一层抽象：它不是单纯的资料夹，
而是“一个可复用的研究范围定义”。这个范围会影响：

- 允许访问哪些 library / paper
- RAG 检索时生成什么 filter
- 哪些 session 属于这个研究空间

因此这里的接口本质上是在管理“研究上下文边界”，而不只是普通 CRUD。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas.project import (
    ProjectCreateRequest,
    ProjectPaperLinkRequest,
    ProjectScopeRequest,
    ProjectUpdateRequest,
)
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import project_service

router = APIRouter()


@router.get("/projects")
async def list_projects(user: dict = Depends(get_current_user_with_rate_limit)):
    # 列表接口只返回当前用户可见项目，用户隔离由 service/repository 共同保证。
    return {"projects": await project_service.list_projects(user_id=get_user_id(user))}


@router.post("/projects")
async def create_project(req: ProjectCreateRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return await project_service.create_project(req.model_dump(), user_id=get_user_id(user))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}")
async def get_project(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    project = await project_service.get_project_or_none(project_id, user_id=get_user_id(user))
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, req: ProjectUpdateRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return await project_service.update_project(
            project_id,
            req.model_dump(exclude_none=True),
            user_id=get_user_id(user),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        await project_service.delete_project(project_id, user_id=get_user_id(user))
        return {"ok": True}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{project_id}/delete")
async def delete_project_compat(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """
    删除项目的兼容入口。

    某些本地开发链路或代理环境会把 `DELETE` 方法拦掉，前端可在收到 405 后退回这个入口。
    """
    try:
        await project_service.delete_project(project_id, user_id=get_user_id(user))
        return {"ok": True}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}/scope")
async def get_project_scope(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    # scope 是后续 build_project_context / Milvus filter 的直接输入。
    try:
        return await project_service.get_project_scope(project_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/projects/{project_id}/scope")
async def put_project_scope(project_id: str, req: ProjectScopeRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return await project_service.update_project_scope(project_id, req.model_dump(), user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/projects/{project_id}/papers")
async def list_project_papers(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return {"papers": await project_service.list_project_papers(project_id, user_id=get_user_id(user))}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{project_id}/papers")
async def add_project_paper(project_id: str, req: ProjectPaperLinkRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return await project_service.add_project_paper(
            project_id,
            req.paper_id,
            role=req.role,
            note=req.note,
            user_id=get_user_id(user),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/projects/{project_id}/papers/{paper_id}")
async def delete_project_paper(project_id: str, paper_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        await project_service.remove_project_paper(project_id, paper_id, user_id=get_user_id(user))
        return {"ok": True}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{project_id}/sessions/{session_id}")
async def attach_project_session(project_id: str, session_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    # 把对话 session 挂到 project 下，便于按研究空间聚合历史会话。
    try:
        return await project_service.attach_session(project_id, session_id, user_id=get_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}/sessions")
async def list_project_sessions(project_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return {"sessions": await project_service.list_project_sessions(project_id, user_id=get_user_id(user))}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}/activities")
async def list_project_activities(project_id: str, limit: int = 50, user: dict = Depends(get_current_user_with_rate_limit)):
    try:
        return {"activities": await project_service.list_project_activities(project_id, limit=limit, user_id=get_user_id(user))}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
