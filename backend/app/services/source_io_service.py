"""来源文件线程的实际退出等待，作者：xxx。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar

from app.models.upload_models import UploadExecution
logger = logging.getLogger(__name__)
Result = TypeVar("Result")


async def run_source_io(function: Callable[..., Result], *args: Any, **kwargs: Any) -> Result:
    """参数为文件操作及参数；返回真实结果，取消时等待已启动线程退出再交还占用。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # 1. 【来源】【取消文件工作】重复取消也不能让仍在访问文件的线程脱离读删临界区
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled() and task.exception() is not None:
            logger.error("【来源】【取消文件工作】线程退出时发生错误", exc_info=task.exception())
        raise


async def run_upload_file_io(execution: UploadExecution, function: Callable[..., Result],
                             *args: Any, **kwargs: Any) -> Result:
    """参数为上传执行和文件操作；占用内重验 writer 后执行，删除撤权后不能再创建版本文件。"""
    from app.repositories.source_access_repository import get_source_access_repository
    from app.repositories.upload_repository import require_upload_execution

    async with get_source_access_repository().file_access((execution.paper_id,)) as connection:
        # 1. 【上传】【文件占用】只在实际文件工作期间持锁，模型等待和供应商轮询不占用连接
        await require_upload_execution(connection, execution)
        return await run_source_io(function, *args, **kwargs)
