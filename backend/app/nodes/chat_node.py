"""
对话入口节点。

这个文件只实现“自由聊天”路径，对应 `agent.py` 里的普通对话模式。

这里强调两种模式，是为了说明职责边界：
  - `rag_chat`：检索知识库后回答，必须附 `[paper_id]` 引用
  - `free_chat`：纯 LLM 对话，携带历史，不自动入库

当前文件实际只负责 `free_chat`。
它不会：
  - 调 `vector_store` 检索论文内容
  - 把消息写成长期知识
  - 管理 checkpoint 底层存储

它做的事只有：
  1. 读取 `chat_history`
  2. 历史超长时压缩旧消息为摘要
  3. 调 LLM 生成当前轮回复并追加回历史
"""

from typing import List

# LangChain 消息类型，用于把前端 chat_history 转成 LLM 可消费的消息序列。
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# AgentState 是主图顶层状态结构。
from app.models.models import AgentState
# 导入统一 LLM 服务。
from app.services.llm_service import get_llm
# 导入全局配置，读取聊天历史压缩阈值等参数。
from app.config import get_settings


def _llm():
    """保留可 monkeypatch 的 LLM 入口，方便测试替换。"""
    return get_llm()


def _settings():
    """按需读取配置，避免模块导入期过早初始化。"""
    return get_settings()

# 自由聊天路径的 system prompt。
_FREE_CHAT_SYSTEM = (
    "You are a helpful AI research assistant. "
    "Engage in thoughtful conversation about research topics, "
    "methodology, and academic concepts. "
    "Be concise and accurate. "
    "The frontend currently renders plain text only. "
    "When answering math or formulas, use plain text expressions like "
    "`999 * 999 = 998001` instead of LaTeX delimiters such as \\( \\), \\[ \\], or $$, "
    "unless the user explicitly asks for LaTeX."
)

_COMPRESS_PROMPT = """Summarize the following conversation into a concise structured summary.
Preserve: key topics discussed, decisions made, paper IDs mentioned, any context needed for future turns.

Existing summary:
{existing}

New messages:
{messages}

Output a concise summary (max 400 words)."""


def _estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数，用于历史压缩阈值判断。"""
    # 当前只是按字符数粗略近似，不追求 tokenizer 级精度。
    return len(text) // 3


def _history_tokens(history: List[dict]) -> int:
    """统计聊天历史的大致 token 总量。"""
    # 对每条消息按内容长度做粗略累加。
    return sum(_estimate_tokens(m.get("content", "")) for m in history)


def _compress_history(history: List[dict], existing_summary: str) -> str:
    """
    压缩旧对话历史，生成新的摘要。

    这样可以在不完全丢失上下文的前提下，避免自由聊天历史无限增长。
    """
    # 把历史消息压成简短文本，供摘要提示词使用。
    msgs_text = "\n".join(
        f'{m["role"].upper()}: {m["content"][:500]}'
        for m in history
    )
    # 拼装摘要压缩提示词。
    prompt = _COMPRESS_PROMPT.format(
        existing=existing_summary or "(none)",
        messages=msgs_text,
    )
    try:
        # 用 LLM 生成新的上下文摘要。
        resp = _llm().invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception:
        # 摘要失败时保留旧摘要，避免自由聊天链路直接报错。
        return existing_summary


def _build_lc_messages(
    chat_history: List[dict],
    user_query: str,
    system: str,
) -> list:
    """把前端样式的 `chat_history` 转成 LangChain messages。"""
    # 先放 system prompt，定义自由聊天行为边界。
    msgs = [SystemMessage(content=system)]
    for m in chat_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        # 前端 history 里的 user 消息映射成 HumanMessage。
        if role == "user":
            msgs.append(HumanMessage(content=content))
        else:
            # 其余统一当作 assistant 消息处理。
            msgs.append(AIMessage(content=content))
    # 最后一条永远是当前轮用户消息。
    msgs.append(HumanMessage(content=user_query))
    return msgs


async def free_chat_node(state: AgentState) -> dict:
    """
    自由对话节点：LLM 直接回答，携带历史，不强制引用，不自动入库。

    与 `rag_node` 的关键区别：
      - 没有检索循环
      - 没有 `[paper_id]` 引用要求
      - 没有长期知识写回
    """
    # 读取当前轮用户问题。
    user_query = state.get("user_query", "")
    # 复制一份聊天历史，避免原地修改输入状态对象。
    chat_history: List[dict] = list(state.get("chat_history", []))
    # 读取已有上下文摘要。
    context_summary = state.get("context_summary", "")

    # 历史超限时，只压缩较早的一半消息，尽量保留最近轮次的连续性。
    # 这说明自由聊天链虽然不做检索，但仍维护一套“短期历史 + 长期摘要”的上下文控制策略。
    max_tokens = _settings().chat_max_history_tokens
    if _history_tokens(chat_history) > max_tokens and len(chat_history) > 4:
        half = len(chat_history) // 2
        old_part = chat_history[:half]
        keep_part = chat_history[half:]
        context_summary = _compress_history(old_part, context_summary)
        chat_history = keep_part

    # 摘要作为 system 背景注入，而不是普通消息追加，
    # 这样既能保留上下文，又不污染显式对话轮次。
    # 先从基础自由聊天 system prompt 开始。
    system = _FREE_CHAT_SYSTEM
    if context_summary:
        # 已有摘要时，把它作为对话背景拼进 system prompt。
        system += f"\n\nConversation summary so far:\n{context_summary}"

    # 把当前会话状态转换成 LangChain 消息序列。
    # 到这里为止，free chat 链路都还停留在“会话态”，没有进入任何知识库/RAG 组件。
    messages = _build_lc_messages(chat_history, user_query, system)

    try:
        # 自由聊天路径直接调用 LLM，不经过检索。
        resp = _llm().invoke(messages)
        answer = resp.content.strip()
    except Exception as e:
        # 失败时返回错误文本，不抛异常打断主图。
        answer = f"对话失败: {e}"

    # 更新后的 `chat_history` 仍属于会话展示层状态，
    # 不是论文业务库，也不是向量知识记忆。
    # 先追加当前轮用户消息。
    chat_history.append({"role": "user",      "content": user_query})
    # 再追加 assistant 回复。
    chat_history.append({"role": "assistant", "content": answer})

    return {
        # 主图统一使用 `report` 承载可直接展示的文本输出。
        "report":          answer,
        # 返回更新后的会话短期历史。
        "chat_history":    chat_history,
        # 返回可能更新过的上下文摘要。
        "context_summary": context_summary,
    }
