"""minimal 深度的直答生成：不产研究计划、不发论文检索、不产带引用的答案块。

判为 minimal 的回合本来就不需要论文事实（闲聊、澄清任务、整理研究问题、改写文字、
回忆项目已有结论），走完整 planner 链只是白等一遍草案与检索。这里是那条短路的终点：
拿问题、对话上下文与本项目历史判断直接生成正文。

载荷形状对齐 research_planner_conversion.build_evidence_not_found_response，
下游写回与前端读的是同一套键。差别在于直答是正常成功路径，所以 error_code /
error_summary 留空；而 source_status.planner 落 "skipped"，让 trace 的
retrieve_paper_evidence 明确投出「本轮没取论文证据」，而不是从四态表里整条消失、
被评测误读成「取过且为空」。

作者：xxx
创建时间：2026-08-31 00:00:00
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from app.middleware.trace_context import trace_label
from app.models.research_planner_contracts import DirectAnswerDraft

# 来源说明按本轮实际注入的上下文写死。系统说系统的话：不经过模型生成，因此不会漏、
# 不会每次说法不一样，也不会编出没发生过的取证过程；两句话的分支只在「本轮提示词里
# 有没有真带项目记忆段」这一件系统事实之间切换（口径 3）
_PROVENANCE_LINE = "本轮没有检索你项目里的论文，以上内容来自模型的通用知识。"
_PROVENANCE_WITH_MEMORY_LINE = (
    "本轮没有检索你项目里的论文；以上内容来自模型的通用知识，并结合了本轮提供的项目研究记忆（记忆不是论文原文）。"
)

# 论文取证在本轮压根没发生。这个值会被 research_memory_writeback 的
# _PAPER_RETRIEVAL_CAPABILITY 换算表读走并映射成四态里的 skipped；
# 换成表外取值，retrieve_paper_evidence 会整条从四态表消失
_PLANNER_STATUS_SKIPPED = "skipped"
# 本地证据执行的词表只有 not_run / attempted 两个取值（research_planner_contracts
# ResearchPlanMetadata），没跑过用 not_run，不能借用 planner 那套的 skipped
_LOCAL_EVIDENCE_NOT_RUN = "not_run"
_RUNTIME_CHAIN = "planner"
# 无历史时调用方渲染进来的占位串，用于判断要不要在 prompt 里摆对话上下文段
_EMPTY_CONVERSATION_CONTEXT = "(none)"


async def build_direct_answer_response(
    *,
    question: str,
    conversation_context: str,
    memory_block: str,
    scoped_paper_count: int,
    llm: Any,
) -> dict[str, Any]:
    """
    生成 minimal 深度的直答载荷：调模型出正文，不取论文证据、不产引用。

    参数:
        question: 用户本轮问题
        conversation_context: 调用方已渲染好的对话上下文文本，"(none)" 表示无历史
        memory_block: 调用方已渲染好的项目记忆段（段首自带身份约束句），
            空串表示本轮没读到记忆
        scoped_paper_count: 本轮授权论文数，0 表示这个项目里还没有任何论文。
            不给默认值：漏传会当场报错，比默默按「项目里有论文」答一轮好找
        llm: 支持 ainvoke 的模型实例
    返回:
        直答响应字典，键集合与 build_evidence_not_found_response 对齐
    抛出:
        ValueError: 模型返回空正文时显式失败，不退回模板正文
    """
    prompt = _direct_answer_prompt(
        question=question,
        conversation_context=conversation_context,
        memory_block=memory_block,
        scoped_paper_count=scoped_paper_count,
    )
    draft = await _generate_answer_draft(prompt, llm)
    # 来源说明跟随本轮实际注入的记忆状态：提示词里真带了记忆段才在说明里提到记忆，
    # 没带（含空、读取失败而没渲染出来）就维持原始说明
    memory_provided = bool(str(memory_block or "").strip())
    return _direct_answer_payload(
        _with_provenance(
            draft.answer,
            knowledge_bearing=draft.knowledge_bearing,
            memory_provided=memory_provided,
        )
    )


async def _generate_answer_draft(prompt: str, llm: Any) -> DirectAnswerDraft:
    """
    调模型产出直答草案：正文 + 这一轮要不要附来源说明的判断。

    模型报错按原样抛出，由调用方转成结构化失败。

    参数:
        prompt: 已拼好的直答提示词
        llm: 支持 with_structured_output 的模型实例
    返回:
        已通过契约校验的直答草案
    抛出:
        ValueError: 模型返回空正文
    """
    with trace_label("planner_synthesize", "direct_answer"):
        draft = await llm.with_structured_output(DirectAnswerDraft).ainvoke([HumanMessage(content=prompt)])
    # 空正文是模型这一轮没干活，直答又没有第二个出口，如实失败比返回一句
    # 「暂时无法回答」有用：后者会把模型故障伪装成正常回答留在会话里
    if not str(draft.answer or "").strip():
        raise ValueError("minimal 直答模型返回空正文")
    return draft


def _with_provenance(answer: str, *, knowledge_bearing: bool, memory_provided: bool) -> str:
    """
    按模型的判断与本轮实际注入的记忆状态决定要不要附来源说明、附哪一句。

    分工：正文与「这段是不是知识性内容」归模型，来源说明归系统。这句话由系统按本轮
    真实情况写死，不让模型复述——实测四种提示词写法（开头「说明来源」1/1、开头「必须」
    2/3、只放末尾 0/4、开头加末尾 0/3）都做不到每轮都出现，而复述一句系统已知的事实
    本来也不该指望模型。

    参数:
        answer: 模型生成的正文
        knowledge_bearing: 模型判定的「正文含知识性内容」
        memory_provided: 本轮提示词里是否真的带了项目记忆段
    返回:
        正文；含知识性内容时另起一段附上对应来源说明，纯寒暄原样返回
    """
    if not knowledge_bearing:
        return str(answer).strip()
    provenance = _PROVENANCE_WITH_MEMORY_LINE if memory_provided else _PROVENANCE_LINE
    return f"{str(answer).strip()}\n\n> {provenance}"


def _direct_answer_payload(answer: str) -> dict[str, Any]:
    """
    组装直答载荷，承载引用与证据的键一律留空。

    参数:
        answer: 模型生成的用户可见正文
    返回:
        直答响应字典
    """
    return {
        "answer": answer,
        # 1. 直答是正常成功路径：不是硬失败、也不是查过没找到的诚实收尾，两个错误位留空
        "error_code": "",
        "error_summary": "",
        "research_plan": None,
        "research_unit_results": [],
        "coverage_review": None,
        "planner_trace": {},
        "runtime_chain": _RUNTIME_CHAIN,
        "source_status": {
            "planner": _PLANNER_STATUS_SKIPPED,
            "planner_repair": _PLANNER_STATUS_SKIPPED,
            "planner_final": _PLANNER_STATUS_SKIPPED,
            "local_evidence_execution": _LOCAL_EVIDENCE_NOT_RUN,
        },
        "source_status_reasons": {
            "planner_final": "本轮按 minimal 直答，未取论文证据、未走定稿引用校验",
        },
        # 2. 引用与证据相关的键全留空：本轮没有任何论文证据可挂，宁可空着也不给
        #    前端一个查不到出处的引用
        "planner_final_blocks": [],
        "planner_final_validation_errors": [],
        "repair_query_plans": [],
        "planner_repair_trace": {},
        "planner_repair_merge_report": {},
        "final_answer_eligibility_report": {},
        "evidence_sufficiency_report": {},
        "evidence_records": [],
        "missing_evidence": [],
        "attempted_queries": [],
        "scoped_papers": [],
    }


def _direct_answer_prompt(
    *,
    question: str,
    conversation_context: str,
    memory_block: str,
    scoped_paper_count: int,
) -> str:
    """
    拼直答提示词：规则 + 问题 + 对话上下文 +（读到记忆时才有的）记忆段。

    参数:
        question: 用户本轮问题
        conversation_context: 已渲染的对话上下文文本
        memory_block: 已渲染的项目记忆段，空串表示本轮没读到记忆
        scoped_paper_count: 本轮授权论文数，决定论文事实那条规则用哪种口径
    返回:
        交给模型的完整提示词
    """
    lines = [
        *_DIRECT_ANSWER_HEAD_RULE_LINES,
        _paper_fact_rule(scoped_paper_count),
        *_DIRECT_ANSWER_TAIL_RULE_LINES,
        "",
        f"用户问题：{question}",
    ]
    # 1. 无历史时不摆空的对话上下文段，省得模型把「(none)」当成一轮真实对话来接
    context = str(conversation_context or "").strip()
    if context and context != _EMPTY_CONVERSATION_CONTEXT:
        lines.extend(["", "对话上下文：", conversation_context])
    # 2. 本轮没读到记忆就整段不出现：补一句「暂无历史判断」等于替项目下了个没验证过的
    #    结论，模型会顺着这句说这个项目还没想法
    if memory_block.strip():
        lines.extend(["", memory_block])
    # 3. 结构化输出的第二个字段是模型唯一要做的判断：这段正文是不是给了知识性内容。
    #    判断留在提示词末尾，紧接着就是问题，避免混进上面那堆边界规则里被忽略
    lines.extend(["", _KNOWLEDGE_FLAG_RULE_LINE])
    return "\n".join(lines)


def _paper_fact_rule(scoped_paper_count: int) -> str:
    """
    按项目里有没有论文，选「问题得靠论文事实才能答准」时的说法。

    参数:
        scoped_paper_count: 本轮授权论文数
    返回:
        论文事实那条规则的单行文本
    """
    # 1. 项目里有论文时，让用户去查项目里的论文是条真出路，用原有措辞
    if scoped_paper_count > 0:
        return _PAPER_FACT_RULE_WITH_PAPERS
    # 2. 空项目里压根没有论文可查，那句「去查项目里的论文」会让用户白跑一趟：
    #    这一档改口说清项目里还没有可核对的材料，而不是含糊拒绝、也不是凭印象给个像事实的答案
    return _PAPER_FACT_RULE_EMPTY_PROJECT


# 直答的边界：本轮没检索论文，所以不能出现任何论文出处；记忆可以延续但不是文献。
# 这段就是 minimal 直答的产品口径，改它等于改这一档回答的行为。
# 分工写在规则里：正文归模型，来源说明归系统（见 _with_provenance），
# 所以这里不要求模型在正文里交代来源，只要求它判断正文是不是知识性的
_DIRECT_ANSWER_HEAD_RULE_LINES = (
    "你是这个项目的研究助手。本轮判定为不需要论文事实的回合，请直接用中文回答用户。",
    "本轮系统没有为你做检索规划，没有检索项目论文，也没有做引用核对；"
    "你手上只有自己的通用知识（外加下面给出时才算的项目研究记忆）。",
    "因此不要在正文里编造论文标题、作者、出处、页码或引用标记，也不要声称某个结论出自"
    "某篇文献或项目里的论文。本轮回答的来源说明由系统另行标注，你不必在正文里交代来源。",
    "系统具备、但本轮没用上的能力：检索项目里的论文全文、按章节定位正文、"
    "列出论文元数据、检索 arXiv 上的外部论文、翻查本项目的既往判断。"
    "用户问起能力时照这份清单如实回答，清单以外的能力不要编。",
)
_DIRECT_ANSWER_TAIL_RULE_LINES = (
    "下面给出的项目研究记忆是本项目此前已形成的判断，可以直接用来理解背景、延续既有思路，"
    "也可以说明这是项目先前的判断；但它不是论文原文，不得写成引用来源或文献依据。",
    "按用户的问题自然作答，不要套固定小标题或固定段落结构。",
)
# 结构化输出的第二个字段由这段说明；判断标准写成「正文里有没有给用户新的事实或概念」，
# 比「是不是知识性内容」好判——后者容易被模型读成「这轮算不算研究性回合」
_KNOWLEDGE_FLAG_RULE_LINE = (
    "另外判断这段正文有没有给用户新的事实或概念（解释、方法、原理、数据这类都算）："
    "有就把 knowledge_bearing 置为 true；只是寒暄、承接对话、复述用户说过的话则为 false。"
)
# 项目里有论文：本轮虽然没检索，但材料确实在项目里，指回去查是有效出路
_PAPER_FACT_RULE_WITH_PAPERS = (
    "如果这个问题其实必须靠论文事实才能答准，就直说需要去查项目里的论文，"
    "不要凭印象编一个看起来像事实的答案。"
)
# 空项目：一篇论文都没上传，问论文事实时既不能编、也不能把人指回一个空书架，
# 而是说清缺的是可核对的论文材料，并告诉用户先把相关论文加进项目
_PAPER_FACT_RULE_EMPTY_PROJECT = (
    "这个项目里目前没有任何论文，所以本轮没有任何可核对的论文材料。"
    "如果这个问题必须靠论文事实才能答准，就明确说明本项目缺少可验证材料、现在无法核对，"
    "并请用户先把相关论文加进项目；"
    "不要凭印象编一个看起来像事实的答案，也不要让用户去核对一份并不存在的材料。"
    "问题里不依赖论文原文的部分照常正常回答，不要因此把整个问题挡回去。"
)
