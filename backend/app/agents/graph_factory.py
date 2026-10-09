"""
主图 compile 入口。

这个文件故意保持很薄，只做一件事：
把“定义 workflow”与“compile 成可执行 graph”拆开。

这样分层的好处是：
- `agent.py` 专注描述节点和边
- API/运行时层只关心“拿到一个可执行 graph”
- 测试时更容易分别替换 workflow 或 checkpointer
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.agent import build_agent_workflow


def create_compiled_graph(checkpointer: BaseCheckpointSaver | None = None):
    """
    统一编译主图，并按需挂载 checkpointer。

    `checkpointer` 来自运行时层；这里不决定其具体实现，只依赖 LangGraph 抽象接口。

    Args:
        checkpointer: 可选 LangGraph checkpoint 抽象实现

    Returns:
        Any: 已 compile 的主图实例
    """
    # 1. 先从 agent.py 取到尚未 compile 的 workflow 定义
    workflow = build_agent_workflow()
    # 2. 再按需挂载运行时传入的 checkpointer，并返回可执行 graph
    return workflow.compile(checkpointer=checkpointer)
