"""评测 episode 的独立仓储与真实 IO 落盘，保留业务筛选语义。作者：xxx。"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from evals.contracts import read_snapshot, snapshot

from .environment import EnvironmentPortUnavailable

STORAGE_VERSION = "eval-storage-v1"
_COLLECTIONS = ("papers", "questions", "items", "evidence_refs", "links", "question_links", "versions")
_BEIJING_TZ = timezone(timedelta(hours=8))


def _now() -> str:
    """无参数；返回北京时间的真实写入时间，不参与实验初始条件比较。"""
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


class EvaluationStorage:
    """每次运行独立复制固定初始数据；所有写入仅进入本次目录。"""

    def __init__(
        self, initial_snapshot: str, *, output_dir: Path, sanitize: Callable[[Any], Any] | None = None,
    ) -> None:
        """参数为初始快照、目录及可选副本脱敏函数；建立独立业务状态，脱敏只作用于落盘观察。"""
        state = read_snapshot(initial_snapshot)
        if state.get("schema_version") != STORAGE_VERSION:
            raise ValueError("unsupported_evaluation_storage_snapshot")
        if any(not isinstance(state.get(key), list) for key in _COLLECTIONS):
            raise ValueError("evaluation_storage_collections_missing")
        self._state = state
        self._output_dir = output_dir
        self._sanitize = sanitize
        self._lock = threading.RLock()
        self.errors: list[dict[str, str]] = []
        self._access_events: list[dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=False)
        self.record_io("initial_state", state)

    def __getattr__(self, name: str) -> Any:
        """参数为未配置的端口名；实际访问立即记错并抛出，禁止假装返回空结果。"""
        error = EnvironmentPortUnavailable(f"evaluation storage port unavailable: {name}")
        self.errors.append({"port": name, "error": str(error)})
        self.record_io("unsupported_port", {"port": name, "error": str(error)})
        raise error

    def repository(self) -> EvaluationStorage:
        """无参数；返回本次隔离仓储，供现有 repository 工厂注入。"""
        return self

    def record_io(self, action: str, payload: dict[str, Any]) -> None:
        """参数为实际 IO 动作和载荷；追加真实持久化记录，写入失败保留错误并抛出。"""
        event = {"action": action, "at": _now(), "payload": deepcopy(payload)}
        with self._lock:
            try:
                # 1. 【评测仓储】【观察落盘】真实IO发生前处理副本，业务载荷及后续读取不被脱敏改写
                persisted = self._sanitize(deepcopy(event)) if self._sanitize is not None else event
                with (self._output_dir / "io.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(snapshot(persisted) + "\n")
            except (OSError, ValueError) as exc:
                self.errors.append({"port": action, "error": str(exc)})
                raise
            self._access_events.append(event)

    def persist_snapshot(self) -> None:
        """无参数；保存本次最终状态和访问清单，拒绝覆盖已经封存的结果。"""
        for filename, payload in (("state.json", self._state), ("access.json", self._access_events)):
            persisted = self._sanitize(deepcopy(payload)) if self._sanitize is not None else payload
            with (self._output_dir / filename).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(persisted, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    async def ensure_schema(self) -> None:
        """无参数；验证本次介质已具备实际集合，沿用仓储初始化契约并记录访问。"""
        if any(not isinstance(self._state.get(key), list) for key in _COLLECTIONS):
            raise ValueError("evaluation_storage_schema_corrupted")
        self.record_io("ensure_schema", {"schema_version": STORAGE_VERSION})

    def _rows(self, collection: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """参数为集合和等值条件；读取独立副本并记录实际命中身份。"""
        rows = [row for row in self._state[collection] if all(value is None or row.get(key) == value for key, value in filters.items())]
        self.record_io(f"read_{collection}", {"filters": filters, "result_count": len(rows)})
        return deepcopy(rows)

    def _append(self, collection: str, row: dict[str, Any]) -> None:
        """参数为目标集合和新对象；先持久化真实写入，再向后续读取公开对象。"""
        with self._lock:
            self.record_io(f"write_{collection}", row)
            self._state = {**self._state, collection: [*self._state[collection], deepcopy(row)]}

    async def list_items(self, project_id: str, *, user_id: str | None = None,
                         review_state: str | None = "accepted", memory_type: str | None = None,
                         evidence_mode: str | None = None, status: str | None = None,
                         stale_state: str | None = None, maturity: str | None = None,
                         paper_id: str | None = None, question_id: str | None = None,
                         tag: str | None = None) -> list[dict[str, Any]]:
        """参数为可信项目/用户、条目资格及关系筛选；返回实际范围内记忆副本，不吞未知字段。"""
        equal = {"user_id": user_id, "memory_type": memory_type, "evidence_mode": evidence_mode,
                 "status": status, "stale_state": stale_state, "maturity": maturity}
        filters = {"paper_id": paper_id, "question_id": question_id, "tag": tag}
        rows = self._rows("items", {"project_id": project_id, "review_state": review_state, **equal})
        selected = [row for row in rows if self._related_item(row, filters)]
        return sorted(selected, key=lambda row: str(row.get("updated_at", "")), reverse=True)

    def _related_item(self, row: dict[str, Any], filters: dict[str, Any]) -> bool:
        """参数为实际条目及收窄条件；核对论文、标签和研究问题的真实关联。"""
        for key, column in (("paper_id", "related_paper_ids"), ("tag", "tags")):
            value = filters.get(key)
            if value is not None and value not in row.get(column, []):
                return False
        question = filters.get("question_id")
        return question is None or any(link["question_id"] == question and link["memory_id"] == row["memory_id"] for link in self._state["question_links"])

    async def list_items_by_source_answer(self, project_id: str, *, user_id: str, source_answer_id: str) -> list[dict[str, Any]]:
        """参数为项目、归属人和实际回答 ID；返回本次已写入候选，支持正常幂等去重。"""
        rows = self._rows("items", {"project_id": project_id, "user_id": user_id, "source_answer_id": source_answer_id})
        return sorted(rows, key=lambda row: str(row.get("created_at", "")))

    async def get_item(self, memory_id: str, *, user_id: str | None = None,
                       project_id: str | None = None) -> dict[str, Any] | None:
        """参数为记忆 ID 和可信归属；只返回范围内实际对象副本，无权访问时返回 None。"""
        rows = self._rows("items", {"memory_id": memory_id, "user_id": user_id, "project_id": project_id})
        return rows[0] if rows else None

    async def get_detail(self, memory_id: str, *, user_id: str | None = None,
                         project_id: str | None = None) -> dict[str, Any] | None:
        """参数为记忆 ID 和可信归属；聚合范围内条目及其实际证据、关系、问题和版本。"""
        item = await self.get_item(memory_id, user_id=user_id, project_id=project_id)
        if item is None:
            return None
        return {
            "item": item, "evidence_refs": await self.list_evidence_refs(memory_id),
            "links": await self.list_links_of_memory(memory_id, user_id=item["user_id"], project_id=item["project_id"]),
            "question_ids": await self.list_question_ids(memory_id, user_id=item["user_id"], project_id=item["project_id"]),
            "versions": await self.list_versions(memory_id),
        }

    async def create_item(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为项目和业务服务产出的候选；实际创建对象和首版记录，返回已落盘详情。"""
        if payload.get("user_id") != self._state["user_id"] or project_id != self._state["project_id"]:
            raise ValueError("evaluation_memory_write_owner_or_project_mismatch")
        identity, now = f"mem_{uuid4().hex}", _now()
        row = {
            "memory_id": identity, "project_id": project_id, "version": 1,
            "created_at": now, "updated_at": now, "stale_state": "fresh", "status": "draft",
            "review_state": "accepted", "maturity": "intuition", "source_kind": "manual",
            "evidence_mode": "user_note_only", "related_paper_ids": [], "tags": [], "body": "",
            **deepcopy(payload),
        }
        if not row.get("title") or not row.get("memory_type"):
            raise ValueError("memory_title_and_type_required")
        self._append("items", row)
        self._append("versions", {"memory_id": identity, "version": 1, "change_kind": "create", "snapshot": deepcopy(row)})
        detail = await self.get_detail(identity)
        if detail is None:
            raise RuntimeError("created_evaluation_memory_not_readable")
        return detail

    async def add_evidence_ref(self, memory_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为已写入记忆及真实证据引用；追加引用并返回对应的实际 ID。"""
        if await self.get_item(memory_id) is None:
            raise LookupError(f"memory_not_found:{memory_id}")
        for key in ("paper_id", "quote_text", "quote_hash"):
            if not payload.get(key):
                raise ValueError(f"memory_evidence_field_required:{key}")
        row = {**deepcopy(payload), "ref_id": f"mref_{uuid4().hex}", "memory_id": memory_id, "created_at": _now()}
        self._append("evidence_refs", row)
        return deepcopy(row)

    async def list_evidence_refs(self, memory_id: str) -> list[dict[str, Any]]:
        """参数为记忆 ID；返回该对象实际持有的证据引用。"""
        return self._rows("evidence_refs", {"memory_id": memory_id})

    async def list_versions(self, memory_id: str) -> list[dict[str, Any]]:
        """参数为记忆 ID；返回已写入的版本记录。"""
        return self._rows("versions", {"memory_id": memory_id})

    async def list_question_ids(self, memory_id: str, *, user_id: str | None = None,
                                project_id: str | None = None) -> list[str]:
        """参数为记忆与归属；关联的问题也核对范围，排除错误的跨用户历史关系。"""
        allowed = {row["question_id"] for row in self._rows("questions", {"user_id": user_id, "project_id": project_id})}
        return [str(row["question_id"]) for row in self._rows("question_links", {"memory_id": memory_id})
                if row["question_id"] in allowed]

    async def list_questions(self, project_id: str, *, user_id: str | None = None,
                             review_state: str | None = "accepted") -> list[dict[str, Any]]:
        """参数为项目、归属人和审核过滤；返回真实问题，沿用目的置顶和position排序。"""
        rows = self._rows("questions", {"project_id": project_id, "user_id": user_id, "review_state": review_state})
        return sorted(rows, key=lambda row: (not row.get("is_purpose", False), row.get("position", 0), str(row.get("created_at", ""))))

    async def get_question(self, question_id: str, *, user_id: str | None = None,
                           project_id: str | None = None) -> dict[str, Any] | None:
        """参数为问题 ID 和可信归属；返回范围内实际问题，不从案例预期补造。"""
        rows = self._rows("questions", {"question_id": question_id, "user_id": user_id, "project_id": project_id})
        return rows[0] if rows else None

    async def list_links(self, project_id: str, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """参数为项目与归属人；关系两端都必须处于当前范围内。"""
        return self._scoped_links(project_id=project_id, user_id=user_id, filters={"project_id": project_id})

    async def list_links_of_memory(self, memory_id: str, *, user_id: str | None = None,
                                   project_id: str | None = None) -> list[dict[str, Any]]:
        """参数为记忆与可信归属；返回范围内两端都存在的真实双向关联。"""
        rows = self._scoped_links(project_id=project_id, user_id=user_id, filters={})
        return [row for row in rows if memory_id in (row["src_memory_id"], row["dst_memory_id"])]

    def _scoped_links(self, *, project_id: str | None, user_id: str | None,
                      filters: dict[str, Any]) -> list[dict[str, Any]]:
        """参数为归属条件和关系过滤；按生产双端关联语义返回按创建时间排序的实际关系。"""
        allowed = {row["memory_id"] for row in self._rows("items", {"project_id": project_id, "user_id": user_id})}
        rows = [row for row in self._rows("links", filters)
                if row["src_memory_id"] in allowed and row["dst_memory_id"] in allowed]
        return sorted(rows, key=lambda row: str(row.get("created_at", "")))

    async def mark_checked_stale(self, item: dict[str, Any], reasons: list[str]) -> None:
        """参数为读取版本及真实失效原因；在本episode原子维护失效版本，不访问业务写库。"""
        self._commit_source_check(item, patch={"stale_state": "needs_rebind", "stale_reasons": reasons},
                                  evidence_refs=None, summary="；".join(reasons))

    async def record_source_checks(self, item: dict[str, Any], evidence_refs: list[dict[str, Any]]) -> dict[str, Any]:
        """参数为读取版本和真实核验引用；复查当前来源，再将核验事实及版本只写入episode。"""
        from app.repositories import memory_source_repository

        # 1. 【评测记忆】【来源复核】沿用生产授权及发布规则，核验后的新发布不能继续提交旧支持
        pool = await memory_source_repository.get_pool()
        async with pool.acquire() as connection, connection.transaction():
            await memory_source_repository.revalidate_memory_sources(connection, user_id=item["user_id"],
                                                                       project_id=item["project_id"], evidence_refs=evidence_refs)
            # 2. 【评测记忆】【隔离提交】来源共享锁覆盖本地CAS和落盘，业务连接只读取来源，不更新记忆
            return self._commit_source_check(item, patch={}, evidence_refs=evidence_refs, summary="记录本次实际原文核验")

    def _commit_source_check(self, item: dict[str, Any], *, patch: dict[str, Any],
                             evidence_refs: list[dict[str, Any]] | None, summary: str) -> dict[str, Any]:
        """参数为读取快照和来源变更；持锁核对owner/version，原子公开条目、引用及同版历史。"""
        with self._lock:
            rows = self._rows("items", {"memory_id": item["memory_id"], "user_id": item["user_id"], "project_id": item["project_id"]})
            if (item["user_id"], item["project_id"]) != (self._state["user_id"], self._state["project_id"]) or not rows:
                raise LookupError("记忆不存在或无权访问")
            if rows[0]["version"] != item["version"]:
                raise ValueError("记忆在来源核验期间已更新")
            refs = self._merge_source_checks(item["memory_id"], evidence_refs or [])
            if not patch and refs == self._state["evidence_refs"]:
                return deepcopy(rows[0])
            updated = {**rows[0], **deepcopy(patch), "version": item["version"] + 1, "updated_at": _now()}
            version = {"memory_id": item["memory_id"], "version": updated["version"], "change_kind": "source_checked",
                       "change_summary": summary, "created_by": item["user_id"], "snapshot": deepcopy(updated)}
            # 2. 【评测记忆】【维护落盘】IO失败直接暴露，内存状态保持提交前版本
            self.record_io("source_checked", {"item": updated, "evidence_refs": refs, "version": version})
            self._state = {**self._state, "items": [updated if row["memory_id"] == item["memory_id"] else row for row in self._state["items"]],
                           "evidence_refs": refs, "versions": [*self._state["versions"], version]}
            return deepcopy(updated)

    def _merge_source_checks(self, memory_id: str, evidence_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """参数为记忆和核验引用；仅更新当前条目的来源事实，返回新集合供事务整体提交。"""
        existing = {ref["ref_id"]: ref for ref in self._state["evidence_refs"] if ref["memory_id"] == memory_id}
        replacements = {}
        for verified in evidence_refs:
            if verified["ref_id"] not in existing:
                raise LookupError("记忆核验引用不存在")
            updates = {key: deepcopy(verified[key]) for key in ("verification", "source_ref", "document_ref")}
            replacements[verified["ref_id"]] = {**existing[verified["ref_id"]], **updates}
        return [replacements.get(ref["ref_id"], ref) for ref in self._state["evidence_refs"]]

    def get_paper(self, paper_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """参数为论文 ID 与可选归属人；读取准备时经真实归属校验的元数据。"""
        rows = self._rows("papers", {"paper_id": paper_id, "user_id": user_id})
        return rows[0] if rows else None

    def list_papers(self, tags: list[str] | None = None, categories: list[str] | None = None, **options: Any) -> list[dict[str, Any]]:
        """参数为论文筛选和分页；为正常推荐链提供固定元数据，结果来自真实初始快照。"""
        unknown = options.keys() - {"user_id", "library_id", "limit", "offset"}
        if unknown:
            raise ValueError(f"unsupported_paper_filters:{sorted(unknown)}")
        rows = self._rows("papers", {key: options.get(key) for key in ("user_id", "library_id")})
        selected = [row for row in rows if (not tags or set(tags) & set(row.get("tags", []))) and (not categories or set(categories) & set(row.get("categories", [])))]
        offset = int(options.get("offset", 0))
        return selected[offset:offset + int(options.get("limit", len(selected)))]

    def list_paper_ids(self, library_id: str | None = None, user_id: str | None = None) -> list[str]:
        """参数为知识库和归属人；返回固定可访问论文中的实际身份。"""
        return [str(row["paper_id"]) for row in self._rows("papers", {"library_id": library_id, "user_id": user_id})]

    def write_log_sync(self, *args: Any, **kwargs: Any) -> None:
        """参数为现役同步日志端口实参；真实保存日志，不触碰业务日志库。"""
        self.record_io("log", {"args": list(args), "kwargs": kwargs})

    async def write_log(self, *args: Any, **kwargs: Any) -> None:
        """参数为异步日志仓储实参；保存本次实际日志。"""
        self.write_log_sync(*args, **kwargs)

    async def write_rag_trace(self, **payload: Any) -> None:
        """参数为实际 trace 仓储字段；将完整载荷写入本次介质。"""
        self.record_io("rag_trace", payload)

    def write_rag_trace_sync(self, *args: Any, **kwargs: Any) -> None:
        """参数为旧同步 trace 端口实参；保存实际载荷供兼容写入路径使用。"""
        self.record_io("rag_trace", {"args": list(args), "kwargs": kwargs})

    def push_notification(self, paper: dict[str, Any], score: float, reason: str) -> None:
        """参数为实际推荐结果；将待通知内容写入本次目录，不发送外部消息。"""
        self.record_io("notification", {"paper": paper, "score": score, "reason": reason})
