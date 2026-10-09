"""研究取证已启动工作的收尾边界。

作者：xxx
创建时间：2026-09-24 10:36:32
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

_T = TypeVar("_T")


async def settle_owned_work(
    work: asyncio.Future[_T],
) -> tuple[_T | BaseException, asyncio.CancelledError | None]:
    """【规划取证】【执行收尾】等待已启动工作完成，分开返回工作结果与调用方取消。

    Args:
        work: 调用方持有的真实任务或查询批次；本函数不会取消它
    Returns:
        tuple: 第一项为完成结果或工作原异常，第二项为等待期间首次收到的调用方取消
            调用方必须检查两项并重抛终止信号，不能将异常结果视为成功
    """
    # 1. 【规划取证】【执行收尾】工作自身的取消作为结果回收，避免与调用方新收到的取消混淆
    completion = asyncio.gather(work, return_exceptions=True)
    cancellation: asyncio.CancelledError | None = None
    # 2. 【规划取证】【执行收尾】再次取消只登记，不中断同步线程的真实执行生命周期
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    # 3. 【规划取证】【执行收尾】从原任务读取异常，保留 gather 合成取消结果时会丢失的原因链
    try:
        outcome = work.result()
    except BaseException as exc:
        return exc, cancellation
    return outcome, cancellation
