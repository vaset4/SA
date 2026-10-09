"""业务数据库兼容 service 外观。"""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import TimeoutError as FutureTimeoutError
import logging
import threading
import uuid
from typing import Any, List, Optional

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.models import Paper, PaperAnalysis
from app.models.task_models import TaskEvent, TaskRun
from app.repositories.note_repository import get_note_repository
from app.repositories.notification_repository import (
    NOTIFICATION_LIMIT_DEFAULT,
    PUSH_NOTIFICATION_EVENT,
    get_notification_repository,
)
from app.repositories.observability_repository import get_observability_repository
from app.repositories.paper_repository import get_paper_repository
from app.repositories.project_repository import get_project_repository
from app.repositories.session_repository import get_session_repository
from app.repositories.task_repository import get_task_repository
from app.repositories.user_repository import get_user_repository
from app.services.database import close_pool

logger = logging.getLogger(__name__)

settings = get_settings()
SQLITE_FALSE = False
SQLITE_TRUE = True
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_started = threading.Event()
_loop_lock = threading.Lock()
_init_db_lock = threading.Lock()
_initialized = False

# 这个模块是“旧同步调用风格”与“新 async repository 层”之间的兼容桥：
#   - 上层大量代码仍按同步函数使用 `paper_db.*`
#   - 底层 repository 已经是 async/await 风格
# 所以这里通过后台事件循环线程把异步仓储包装成同步 facade。


def _to_paper(data: dict[str, Any] | None) -> Optional[Paper]:
    if data is None:
        return None
    return Paper(
        library_id=data.get("library_id") or DEFAULT_LIBRARY_ID,
        paper_id=data["paper_id"],
        arxiv_id=data.get("arxiv_id"),
        doi=data.get("doi"),
        title=data["title"],
        authors=data.get("authors", []),
        abstract=data["abstract"],
        pdf_url=data.get("pdf_url", ""),
        pdf_path=data.get("pdf_path"),
        categories=data.get("categories", []),
        published_date=data.get("published_date", ""),
        source=data.get("source", "arxiv"),
        analysis=data.get("analysis"),
        tags=data.get("tags", []),
        user_id=data.get("user_id", ""),
        is_vectorized=bool(data.get("is_vectorized")),
        index_batch_id=data.get("index_batch_id"),
        metadata_status=data.get("metadata_status", "not_requested"),
        metadata_error=data.get("metadata_error", ""),
        metadata_missing=data.get("metadata_missing", []),
    )


def _run_loop() -> None:
    """在后台线程里托管一个长期事件循环，专供同步 facade 提交协程。"""
    global _loop
    loop = asyncio.new_event_loop()
    _loop = loop
    asyncio.set_event_loop(loop)
    _started.set()
    loop.run_forever()
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """惰性启动后台事件循环线程，并返回 loop。"""
    global _thread
    if _loop is not None:
        return _loop
    with _loop_lock:
        if _loop is None:
            _started.clear()
            _thread = threading.Thread(target=_run_loop, daemon=True)
            _thread.start()
            _started.wait()
    return _loop


def _run(coro):
    """把协程提交到兼容循环，并按数据库命令预算同步等待结果。"""
    future = asyncio.run_coroutine_threadsafe(coro, _ensure_loop())
    try:
        return future.result(timeout=float(settings.database_command_timeout))
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"paper_db 兼容调用超过 {settings.database_command_timeout} 秒"
        ) from exc


def init_db() -> None:
    """
    初始化最基础的仓储 schema。

    这里只初始化当前同步 facade 明确依赖的核心仓储；更细分的 schema 仍可在各自服务里
    按需初始化。
    """
    global _initialized
    if _initialized:
        return
    with _init_db_lock:
        if not _initialized:
            _run(get_user_repository().ensure_schema())
            _run(get_paper_repository().ensure_schema())
            _initialized = True


def add_paper(paper: Paper) -> str:
    init_db()
    return _run(get_paper_repository().add_or_update_paper(paper))


def get_paper(paper_id: str, user_id: str | None = None) -> Optional[Paper]:
    init_db()
    return _to_paper(_run(get_paper_repository().get_paper(paper_id, user_id)))


def get_by_arxiv_id(arxiv_id: str, user_id: str | None = None) -> Optional[Paper]:
    init_db()
    return _to_paper(_run(get_paper_repository().get_by_arxiv_id(arxiv_id, user_id)))


def search_by_exact_title(
    title: str,
    library_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Paper]:
    init_db()
    return [
        _to_paper(row) or {}
        for row in _run(get_paper_repository().search_by_exact_title(title, library_id, user_id=user_id))
    ]


def exists(arxiv_id: Optional[str] = None, doi: Optional[str] = None, *, user_id: str | None = None) -> bool:
    init_db()
    return bool(_run(get_paper_repository().exists(arxiv_id, doi, user_id=user_id)))


def update_analysis(paper_id: str, analysis: PaperAnalysis, user_id: str | None = None) -> None:
    init_db()
    _run(get_paper_repository().update_analysis(paper_id, analysis, user_id=user_id))


def update_analysis_status(
    paper_id: str,
    status: str,
    *,
    error: str = "",
    clear_analysis: bool = False,
    user_id: str | None = None,
) -> None:
    init_db()
    _run(
        get_paper_repository().update_analysis_status(
            paper_id,
            status,
            error=error,
            clear_analysis=clear_analysis,
            user_id=user_id,
        )
    )


def update_paper_metadata(
    paper_id: str,
    patch: dict[str, Any],
    user_id: str | None = None,
) -> Optional[Paper]:
    init_db()
    return _to_paper(_run(get_paper_repository().update_paper_metadata(paper_id, patch, user_id=user_id)))


def mark_vectorized(paper_id: str) -> None:
    init_db()
    _run(get_paper_repository().mark_vectorized(paper_id))


def count_papers(
    tags: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    library_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> int:
    init_db()
    if library_id:
        return int(_run(get_paper_repository().count_papers_by_library(library_id, tags, categories, user_id=user_id)))
    return int(_run(get_paper_repository().count_papers(tags, categories, user_id=user_id)))


def list_papers(
    tags: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    limit: int = 50,
    offset: int = 0,
    library_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Paper]:
    init_db()
    return [
        _to_paper(row) or {}
        for row in _run(get_paper_repository().list_papers(tags, categories, limit, offset, library_id, user_id=user_id))
    ]


def search_by_keyword(keyword: str, library_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Paper]:
    init_db()
    return [
        _to_paper(row) or {}
        for row in _run(get_paper_repository().search_by_keyword(keyword, library_id, user_id=user_id))
    ]


def list_paper_ids(library_id: Optional[str] = None, user_id: Optional[str] = None) -> List[str]:
    init_db()
    return _run(get_paper_repository().list_paper_ids_by_library(library_id, user_id=user_id))


def delete_paper(paper_id: str, user_id: Optional[str] = None) -> bool:
    init_db()
    _run(get_note_repository().delete_notes_by_paper(paper_id, user_id=user_id))
    return bool(_run(get_paper_repository().delete_paper(paper_id, user_id=user_id)))


def add_note(
    paper_id: str | None,
    content: str,
    user_id: str = "anonymous",
    *,
    project_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    note_type: str = "reading_note",
    source_type: str = "manual",
    citation: str = "",
    excerpt: str = "",
    evidence_reference: dict[str, Any] | None = None,
) -> str:
    init_db()
    return _run(
        get_note_repository().add_note(
            user_id,
            paper_id,
            content,
            project_id=project_id,
            conversation_id=conversation_id,
            session_id=session_id,
            note_type=note_type,
            source_type=source_type,
            citation=citation,
            excerpt=excerpt,
            evidence_reference=evidence_reference,
        )
    )


def get_notes(paper_id: str, user_id: str = "anonymous") -> List[dict]:
    init_db()
    return _run(get_note_repository().list_notes(user_id, paper_id, order="ASC"))


def list_notes(
    *,
    user_id: str = "anonymous",
    paper_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    note_type: str | None = None,
    source_type: str | None = None,
    include_project_papers: bool = False,
    include_project_sessions: bool = False,
    limit: int = 200,
    order: str = "DESC",
) -> List[dict]:
    init_db()
    if project_id or include_project_papers or include_project_sessions:
        _run(get_project_repository().ensure_schema())
    return _run(
        get_note_repository().list_notes(
            user_id,
            paper_id,
            project_id=project_id,
            conversation_id=conversation_id,
            session_id=session_id,
            note_type=note_type,
            source_type=source_type,
            include_project_papers=include_project_papers,
            include_project_sessions=include_project_sessions,
            limit=limit,
            order=order,
        )
    )


def create_session(topic: str, user_id: str = "anonymous") -> str:
    """创建会话并返回 UUID；LangGraph checkpoint/UI conversation 会围绕它继续挂接。"""
    init_db()
    session_id = str(uuid.uuid4())
    _run(
        get_session_repository().create_session(
            user_id,
            session_id,
            topic,
            thread_id=session_id,
        )
    )
    return session_id


def save_checkpoint(session_id: str, step_name: str, state_json: str) -> None:
    init_db()
    _run(get_session_repository().save_checkpoint(session_id, step_name, state_json))


def load_checkpoint(session_id: str) -> Optional[dict]:
    init_db()
    row = _run(get_session_repository().load_checkpoint(session_id))
    if row is None or not row.get("state_json"):
        return None
    return {"step_name": row["step_name"], "state_json": row["state_json"]}


def update_session_papers(session_id: str, paper_ids: List[str], user_id: str | None = None) -> None:
    init_db()
    _run(get_session_repository().update_session_papers(session_id, paper_ids, user_id=user_id))


def get_session(session_id: str, user_id: str | None = None) -> Optional[dict]:
    init_db()
    return _run(get_session_repository().get_session(session_id, user_id=user_id))


def list_sessions(user_id: str | None = None, limit: int = 20, offset: int = 0) -> List[dict]:
    init_db()
    return _run(get_session_repository().list_sessions(user_id=user_id, limit=limit, offset=offset))


def remove_paper_from_sessions(paper_id: str, user_id: str | None = None) -> int:
    init_db()
    return int(_run(get_session_repository().remove_paper_from_all_sessions(paper_id, user_id=user_id)))


def write_log(event_type: str, node_name: str = "", query: str = "", result_count: int = 0, latency_ms: int = 0, session_id: Optional[str] = None, user_id: Optional[str] = None) -> None:
    """写观测日志；失败不影响主链，但必须留下 warning 证据。user_id 可选，面向用户通知必须显式传。"""
    if event_type == PUSH_NOTIFICATION_EVENT:
        if not user_id or not user_id.strip():
            raise ValueError("push_notification 必须显式传入 user_id")
    try:
        init_db()
        _run(get_observability_repository().write_log(event_type, node_name, query, result_count, latency_ms, session_id, user_id=user_id))
    except Exception as exc:
        if event_type == PUSH_NOTIFICATION_EVENT:
            logger.error(
                "【任务中心】【用户隔离】通知写入失败 event_type=%s node_name=%s session_id=%s: %s",
                event_type,
                node_name,
                session_id,
                exc,
                exc_info=True,
            )
            raise
        logger.error(
            "【业务观测】【日志写入】写入失败 event_type=%s node_name=%s session_id=%s: %s",
            event_type,
            node_name,
            session_id,
            exc,
            exc_info=True,
        )


def write_rag_trace(request_id: str, session_id: str, query: str, trace_data: dict[str, Any], user_id: str = "unknown") -> None:
    """写 RAG trace；同样采取 best-effort 策略。"""
    try:
        init_db()
        _run(get_observability_repository().write_rag_trace(request_id, session_id or None, query, trace_data, user_id=user_id))
    except Exception as exc:
        logger.warning("write_rag_trace failed for %s: %s", request_id, exc, exc_info=True)


def create_task_run(
    task_type: str,
    payload: dict[str, Any] | None = None,
    message: str = "",
    *,
    task_id: str | None = None,
    user_id: str,
) -> TaskRun:
    """创建任务主记录；user_id 为服务端归属，必填，禁止匿名默认。"""
    init_db()
    return _run(
        get_task_repository().create_task_run(
            task_type,
            payload or {},
            message,
            task_id=task_id,
            user_id=user_id,
        )
    )


def add_task_event(task_id: str, event_type: str, message: str, payload: dict[str, Any] | None = None) -> TaskEvent:
    init_db()
    return _run(get_task_repository().add_task_event(task_id, event_type, message, payload or {}))


def get_task_run(task_id: str, *, user_id: str) -> Optional[TaskRun]:
    """按归属读取单条任务。"""
    init_db()
    return _run(get_task_repository().get_task_run(task_id, user_id=user_id))


def list_task_runs(statuses: Optional[List[str]] = None, *, user_id: str, limit: int = 50, offset: int = 0) -> List[TaskRun]:
    """按当前用户分页列出任务；owner 必填。"""
    init_db()
    return _run(
        get_task_repository().list_task_runs(
            statuses,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
    )


def list_task_events(
    task_id: str,
    *,
    user_id: str,
    limit: int = 200,
) -> List[TaskEvent]:
    """按任务归属列出事件。"""
    init_db()
    return _run(get_task_repository().list_task_events(task_id, user_id=user_id, limit=limit))


def update_task_progress(task_id: str, progress: int, message: str) -> Optional[TaskRun]:
    init_db()
    return _run(get_task_repository().update_task_progress(task_id, progress, message))


def complete_task_run(
    task_id: str,
    result: Any,
    message: str,
    *,
    worker_id: str,
) -> Optional[TaskRun]:
    """使用 worker_id 持租校验写入完成终态；失租时返回 None。"""
    init_db()
    return _run(get_task_repository().complete_task_run(task_id, result, message, worker_id=worker_id))


def fail_task_run(
    task_id: str,
    error: str,
    message: str,
    *,
    worker_id: str,
) -> Optional[TaskRun]:
    """使用 worker_id 持租校验写入失败终态；失租时返回 None。"""
    init_db()
    return _run(get_task_repository().fail_task_run(task_id, error, message, worker_id=worker_id))


def mark_task_cancelled(task_id: str, message: str, *, worker_id: str) -> Optional[TaskRun]:
    """使用 worker_id 持租校验写入取消终态；失租时返回 None。"""
    init_db()
    return _run(get_task_repository().mark_task_cancelled(task_id, message, worker_id=worker_id))


def request_task_cancel(task_id: str, message: str, *, user_id: str) -> Optional[TaskRun]:
    """按归属请求取消任务；越权返回 None。"""
    init_db()
    return _run(get_task_repository().request_task_cancel(task_id, message, user_id=user_id))


def claim_next_task(worker_id: str, lease_seconds: int = 300, task_types: Optional[List[str]] = None) -> Optional[TaskRun]:
    init_db()
    return _run(get_task_repository().claim_next_task(worker_id, lease_seconds, task_types))


def heartbeat_task(task_id: str, worker_id: str, lease_seconds: int = 300) -> Optional[TaskRun]:
    init_db()
    return _run(get_task_repository().heartbeat_task(task_id, worker_id, lease_seconds))


def is_task_cancel_requested(task_id: str) -> bool:
    init_db()
    return bool(_run(get_task_repository().is_task_cancel_requested(task_id)))


def list_notifications(user_id: str, limit: int = NOTIFICATION_LIMIT_DEFAULT) -> List[dict]:
    """按当前用户列出通知。"""
    init_db()
    return _run(get_notification_repository().list_notifications(user_id, limit))


def set_notification_read_state(log_id: str, user_id: str, read: bool) -> bool:
    """按 (log_id, user_id) 更新通知已读状态。"""
    init_db()
    return bool(_run(get_notification_repository().set_read_state(log_id, user_id, read)))


def close_db() -> None:
    _shutdown()


def _shutdown() -> None:
    """关闭连接池和后台事件循环，供进程退出时清理。"""
    global _loop, _thread, _initialized
    if _loop is None:
        _initialized = False
        return
    try:
        _run(close_pool())
    except Exception as exc:
        logger.warning("close_pool failed during paper_db shutdown: %s", exc, exc_info=True)
    _loop.call_soon_threadsafe(_loop.stop)
    if _thread is not None:
        _thread.join(timeout=1)
    _loop = None
    _thread = None
    _initialized = False
    _started.clear()


atexit.register(_shutdown)
