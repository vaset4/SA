"""
上传流程步骤日志与汇总工具。

这个模块不是上传主链本身，而是给 `library_ingest.py` 这种编排层提供统一的
步骤记录助手。它只负责：
  - 定义阶段名常量
  - 生成 `start / done / error` 三类步骤记录
  - 统计每个阶段的耗时并打印简要日志

需要特别区分：
  - 真正的解析主链在 `document_parser.py`
  - 真正的上传编排主链在 `library_ingest.py`
  - 文档底层读取由统一解析器负责，这里只管理业务步骤
"""

from __future__ import annotations

import time

MS_PER_SEC = 1000.0
STEP_PARSE = "parse"
STEP_ANALYZE = "analyze"
STEP_INDEX = "index_write"
STEP_DB = "db_write"
STEP_VECTOR = "vector_write"
STEP_GRAPH = "graph_write"
STEP_DONE = "done"


def append_step(steps: list[dict], step: dict) -> list[dict]:
    """以不可变方式追加一步，返回新的步骤列表。"""
    return [*steps, step]


def start_step(steps: list[dict], stage: str) -> tuple[list[dict], float]:
    """
    记录某个上传阶段开始，并返回计时起点。

    编排层通过这一组 helper 统一输出步骤日志，避免每个阶段手写字典结构。
    """
    print(f"[Upload] {stage} start")
    return append_step(steps, {"stage": stage, "status": "start", "duration_ms": 0}), time.monotonic()


def finish_step(
    steps: list[dict],
    stage: str,
    started_at: float,
    detail: str | None = None,
) -> list[dict]:
    """记录阶段成功结束，写入耗时毫秒数。"""
    duration_ms = int((time.monotonic() - started_at) * MS_PER_SEC)
    print(f"[Upload] {stage} done ({duration_ms}ms)")
    step = {"stage": stage, "status": "done", "duration_ms": duration_ms}
    if detail:
        step["detail"] = detail
    return append_step(steps, step)


def fail_step(steps: list[dict], stage: str, started_at: float, error: Exception) -> list[dict]:
    """记录阶段失败，保留耗时与错误摘要，供 API/UI 返回上传明细。"""
    duration_ms = int((time.monotonic() - started_at) * MS_PER_SEC)
    detail = str(error)
    print(f"[Upload] {stage} error ({duration_ms}ms): {detail}")
    return append_step(
        steps,
        {"stage": stage, "status": "error", "duration_ms": duration_ms, "detail": detail},
    )
