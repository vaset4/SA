"""Agent 评测的集中环境适配与 IO 访问留痕。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol

from .contracts import AgentCase, agent_case_payload, validate_agent_execution_input


@dataclass(frozen=True, kw_only=True)
class AccessEvent:
    """记录一次具名外部能力访问及其结果。"""

    port: str
    action: str
    status: str
    error_type: str | None = None
    at: str = ""


class GraphFactory(Protocol):
    """构造现役研究子图的可替换端口。"""

    def __call__(self) -> Any:
        """返回可执行的 LangGraph 子图。"""


StateBuilder = Callable[[AgentCase, str, list[dict[str, Any]]], dict[str, Any]]
_BEIJING_TZ = timezone(timedelta(hours=8))


class EnvironmentPortUnavailable(RuntimeError):
    """声明的外部能力未配置时抛出的显式错误。"""


@dataclass
class AgentEnvironment:
    """集中提供图构造、输入装配和外部 IO 访问记录。"""

    graph_factory: GraphFactory
    declared: Mapping[str, Any] = field(default_factory=dict)
    state_builder: StateBuilder | None = None
    access_events: list[AccessEvent] = field(default_factory=list)
    environment_issues: list[dict[str, Any]] = field(default_factory=list)

    def create_graph(self) -> Any:
        """无参数；构造一次真实子图并记录图端口访问。"""
        self.record_access("research_graph", "construct", status="started")
        try:
            graph = self.graph_factory()
        except Exception as exc:
            self.record_access("research_graph", "construct", status="failed", error=exc)
            raise
        self.record_access("research_graph", "construct", status="succeeded")
        return graph

    def build_turn_input(
        self, case: AgentCase, *, user_message: str, history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """参数为案例、当前用户消息和已发生历史；返回不含评分真值的图输入。"""
        if self.state_builder is not None:
            payload = self.state_builder(case, user_message, deepcopy(history))
        else:
            case_data = agent_case_payload(case)
            base = case_data["input"]
            payload = {
                **{key: deepcopy(base[key]) for key in (
                    "project_scope", "structured_query", "user_id", "project_id", "paper_ids",
                ) if key in base},
                "user_query": user_message,
                "chat_history": deepcopy(history),
                "paper_keys": list(base.get("paper_keys", [])),
                "paper_focus": base.get("focused_paper_key"),
            }
        validate_agent_execution_input(payload)
        return deepcopy(payload)

    def record_access(
        self, port: str, action: str, *, status: str, error: Exception | None = None,
    ) -> None:
        """参数为外部端口、动作、结果和可选异常；追加不可变访问事实。"""
        self.access_events.append(AccessEvent(
            port=port, action=action, status=status,
            error_type=type(error).__name__ if error is not None else None,
            at=datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        ))

    def require_port(self, port: str) -> Any:
        """参数为具名端口；未配置时记录环境问题并显式失败。"""
        if port not in self.declared or self.declared[port] is None:
            error = EnvironmentPortUnavailable(f"agent evaluation port is not configured: {port}")
            self.record_access(port, "require", status="failed", error=error)
            self.environment_issues.append({
                "port": port, "error_type": type(error).__name__, "message": str(error),
            })
            raise error
        self.record_access(port, "require", status="succeeded")
        return self.declared[port]

    def create_turn_trace(self, turn_input: dict[str, Any]) -> Any:
        """参数为实际本轮输入；返回独立请求 trace，不向模型传递评测真值。"""
        from uuid import uuid4

        from app.middleware.trace_context import TraceContext

        owner = self.declared.get("user_id")
        return TraceContext(
            request_id=str(uuid4()), user_id=owner if isinstance(owner, str) else "",
            session_id=str(turn_input.get("session_id") or ""),
        )

    def record_index_declaration(self) -> None:
        """无参数；在实际取证发生前记录本环境已验证的索引身份。"""
        from app.middleware.trace_context import record_evidence_observation

        identity = self.declared.get("index_snapshot_id")
        if identity is not None:
            if not isinstance(identity, str) or not identity:
                raise ValueError("agent environment index identity must be non-empty text")
            record_evidence_observation("index_snapshot_declared", {
                "index_snapshot_id": identity, "source": "eval_declared_before_execution",
            })

    def snapshot(self) -> dict[str, Any]:
        """无参数；返回可序列化的环境声明、访问记录和环境问题。"""
        return {
            "declared": dict(self.declared),
            "access_events": [event.__dict__.copy() for event in self.access_events],
            "environment_issues": deepcopy(self.environment_issues),
        }

    def validate_after_run(self) -> tuple[str, ...]:
        """无参数；返回运行后必须在报告中保留的环境问题字段。"""
        if self.environment_issues:
            return ("environment_access",)
        return ()
