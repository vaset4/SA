"""文献库上传的轻薄受理入口，作者：xxx。"""
from __future__ import annotations

from app.models.task_models import TaskRun
from app.services import embedded_worker_service, task_service


async def add_uploaded_paper(
    filename: str, content: bytes, library_id: str | None = None, *,
    user_id: str, request_id: str, project_id: str | None = None,
    graph_requested: bool = False,
) -> TaskRun:
    """【上传】【文献库受理】保存原文件并排队；返回可查回的任务，不提前声明论文可用。

    Args:
        filename/content: 客户端文件名及原始内容
        library_id/project_id: 目标文献库及需补充的项目关系
        user_id/request_id: 服务端账号和客户端稳定受理标识
        graph_requested: 是否在基础发布后单独构建图谱
    Returns:
        TaskRun: 持久任务及原论文身份
    """
    task = await task_service.create_upload_task(filename, content, library_id=library_id,
        project_id=project_id, user_id=user_id, request_id=request_id, graph_requested=graph_requested)
    embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
    return task
