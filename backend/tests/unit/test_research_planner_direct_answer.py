"""minimal 深度直答 provider 测试（单元 V1）。

覆盖点：
- 载荷键集合与 build_evidence_not_found_response 对齐，source_status.planner 落 skipped
- 直答是成功路径：error_code / error_summary 留空，计划与引用类键全空
- 记忆段透传：读到记忆时 prompt 含那段（含身份约束句），空串时整段不出现
- 来源说明按本轮实际注入的记忆状态切换：有记忆的知识性回答点到记忆，无记忆维持原句
- 模型故障不吞：异常向上传播、空正文显式失败，都不退回模板正文
- 空项目口径（单元 V5）：论文数为 0 时说清缺少可验证材料，不把用户指回一个空书架；
  论文数 > 0 时保持「去查项目里的论文」的原措辞

作者：xxx
创建时间：2026-08-31 00:00:00
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.models.research_planner_contracts import DirectAnswerDraft
from app.services.internal_research.research_planner_conversion import (
    build_evidence_not_found_response,
)
from app.services.internal_research.research_planner_direct_answer import (
    _PROVENANCE_LINE,
    _PROVENANCE_WITH_MEMORY_LINE,
    build_direct_answer_response,
)
from app.services.internal_research.research_planner_plan_prompt import (
    render_project_memory_block,
)
from app.services.research_memory_injector_service import render_memory_prompt_block

# 记忆段段首的身份约束原句片段，直答 prompt 必须原样带上它
_IDENTITY_CONSTRAINT = "非论文原文，不得作为引用来源"
# 空项目那一档必须说清缺的是可核对的论文材料
_MISSING_MATERIAL_WORDING = "缺少可验证材料"
# 项目里有论文那一档的原措辞：材料确实在项目里，指回去查是条真出路
_GO_CHECK_PAPERS_WORDING = "去查项目里的论文"
# 缺材料不等于整轮拒答：不依赖论文原文的部分照常作答
_STILL_ANSWERS_WORDING = "照常正常回答"
# 分工的第一半：模型要知道本轮系统跑了什么、没跑什么，才知道自己手上有什么
_SYSTEM_STATE_WORDING = "本轮系统没有为你做检索规划，没有检索项目论文，也没有做引用核对"
# 分工的第二半：来源说明由系统标注，模型不用在正文里复述（复述实测压不住）
_DISCLOSURE_OWNED_BY_SYSTEM_WORDING = "本轮回答的来源说明由系统另行标注，你不必在正文里交代来源"
# 模型要照实说系统能干什么；这份清单是本轮没用上的真实能力
_CAPABILITY_LIST_WORDING = "检索项目里的论文全文、按章节定位正文、列出论文元数据、检索 arXiv 上的外部论文"
# 结构化输出的第二个字段由这句说明，判断标准是「正文里有没有给用户新的事实或概念」
_KNOWLEDGE_FLAG_WORDING = "有就把 knowledge_bearing 置为 true"
# 承载论文引用与证据的键：直答不取论文证据，这些键必须留空
_CITATION_BEARING_KEYS = (
    "planner_final_blocks",
    "evidence_records",
    "missing_evidence",
    "attempted_queries",
    "scoped_papers",
    "research_unit_results",
    "repair_query_plans",
)


class _StubDirectAnswerLLM:
    """直答用的结构化模型替身：按契约造草案，顺带记下收到的 prompt。

    正文为空时照真实路径那样被契约挡下（`DirectAnswerDraft` 的 `min_length=1`），
    而不是让替身自己抛错——要测的是「空正文进不了正常回答」这件事本身。
    """

    def __init__(self, content: str = "这个项目先前倾向用 A 作骨干，本轮先把问题拆清楚。",
                 knowledge_bearing: bool = False) -> None:
        self._content = content
        self._knowledge_bearing = knowledge_bearing
        self.prompts: list[str] = []

    def with_structured_output(self, _schema: type) -> "_StubDirectAnswerLLM":
        return self

    async def ainvoke(self, messages: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return DirectAnswerDraft(answer=self._content, knowledge_bearing=self._knowledge_bearing)


class _RaisingLLM:
    """调用即抛错的 LLM 替身，用于验证模型故障不被吞成模板正文。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def with_structured_output(self, _schema: type) -> "_RaisingLLM":
        return self

    async def ainvoke(self, _messages: Any) -> Any:
        raise self._error


def _memory_block() -> str:
    """用真实渲染器造一段项目记忆，确保测的是线上那句身份约束。"""
    return render_memory_prompt_block([
        {
            "memory_id": "m1",
            "version": 1,
            "title": "时空预测主干先用 A",
            "body_summary": "A 的骨干在本项目数据上更稳。",
            "memory_type": "decision_note",
            "maturity": "leaning",
            "source_kind": "user_decision",
            "provenance_label": "倾向",
        }
    ])


def _run_direct_answer(
    llm: Any,
    *,
    memory_block: str = "",
    question: str = "我们之前定的主干是哪个",
    conversation_context: str = "(none)",
    scoped_paper_count: int = 2,
) -> dict[str, Any]:
    """跑一次直答生成，返回载荷。

    参数:
        llm: LLM 替身
        memory_block: 已渲染的记忆段，空串表示本轮没读到记忆
        question: 用户本轮问题
        conversation_context: 已渲染的对话上下文，"(none)" 表示无历史
        scoped_paper_count: 本轮授权论文数，缺省走「项目里有论文」这一档
    返回:
        直答载荷
    """
    return asyncio.run(
        build_direct_answer_response(
            question=question,
            conversation_context=conversation_context,
            memory_block=memory_block,
            scoped_paper_count=scoped_paper_count,
            llm=llm,
        )
    )


def test_payload_keys_align_with_evidence_not_found() -> None:
    """键集合逐键对齐诚实收尾载荷，下游 write_back 读的是同一套键。"""
    payload = _run_direct_answer(_StubDirectAnswerLLM())
    reference = build_evidence_not_found_response(
        question="q",
        scoped_paper_ids=["p1"],
        authorized_paper_count=1,
        attempted_queries=["q1"],
        missing_evidence=[{"dimension": "d1"}],
    )
    assert set(payload) == set(reference)
    assert set(payload["source_status"]) == set(reference["source_status"])


def test_planner_status_is_skipped_for_capability_table() -> None:
    """planner 落 skipped，四态表里 retrieve_paper_evidence 才会投出 skipped。"""
    payload = _run_direct_answer(_StubDirectAnswerLLM())
    assert payload["source_status"]["planner"] == "skipped"
    assert payload["source_status"]["local_evidence_execution"] == "not_run"
    assert "planner_final" in payload["source_status_reasons"]


def test_direct_answer_is_success_path_without_plan_or_citations() -> None:
    """直答是成功路径：错误位留空，计划与引用类键全空。"""
    llm = _StubDirectAnswerLLM("本项目先前倾向用 A 作骨干。")
    payload = _run_direct_answer(llm)
    assert payload["answer"] == "本项目先前倾向用 A 作骨干。"
    assert payload["error_code"] == ""
    assert payload["error_summary"] == ""
    assert payload["research_plan"] is None
    assert payload["coverage_review"] is None
    assert payload["planner_final_blocks"] == []
    for key in _CITATION_BEARING_KEYS:
        assert payload[key] == [], f"{key} 必须留空，直答不挂论文引用"


def test_memory_block_reaches_prompt_with_identity_constraint() -> None:
    """读到记忆时整段进 prompt，身份约束句原样带着。"""
    llm = _StubDirectAnswerLLM()
    block = _memory_block()
    _run_direct_answer(llm, memory_block=block)
    prompt = llm.prompts[0]
    assert block in prompt
    assert _IDENTITY_CONSTRAINT in prompt


def test_empty_memory_block_leaves_no_memory_section() -> None:
    """本轮没读到记忆时 prompt 里不出现记忆段，也不编「暂无记忆」的占位。"""
    llm = _StubDirectAnswerLLM()
    _run_direct_answer(llm, memory_block="")
    prompt = llm.prompts[0]
    assert "【项目研究记忆】" not in prompt
    assert _IDENTITY_CONSTRAINT not in prompt


def test_empty_conversation_context_leaves_no_context_section() -> None:
    """无历史的 "(none)" 不摆成对话上下文段，免得模型当真实一轮来接。"""
    llm = _StubDirectAnswerLLM()
    _run_direct_answer(llm, conversation_context="(none)")
    assert "对话上下文：" not in llm.prompts[0]


def test_empty_project_prompt_states_missing_verifiable_material() -> None:
    """空项目问论文事实：prompt 要求说清缺少可验证材料，且不把用户指回一个空书架。

    项目里一篇论文都没有时，「去查项目里的论文」是让用户白跑一趟；
    这一档的口径是说清缺的是可核对的论文材料，同时不依赖论文原文的部分照常作答，
    既不凭印象编一个像事实的答案，也不把整个问题挡回去。
    """
    llm = _StubDirectAnswerLLM()
    _run_direct_answer(llm, question="这篇论文的实验用了多少数据", scoped_paper_count=0)
    prompt = llm.prompts[0]
    assert _MISSING_MATERIAL_WORDING in prompt
    assert _GO_CHECK_PAPERS_WORDING not in prompt
    assert _STILL_ANSWERS_WORDING in prompt


def test_project_with_papers_keeps_go_check_papers_wording() -> None:
    """项目里有论文时保持原措辞：材料在项目里，指回去查确实是条出路。"""
    llm = _StubDirectAnswerLLM()
    _run_direct_answer(llm, scoped_paper_count=1)
    prompt = llm.prompts[0]
    assert _GO_CHECK_PAPERS_WORDING in prompt
    assert _MISSING_MATERIAL_WORDING not in prompt


def test_prompt_splits_content_from_provenance() -> None:
    """直答 prompt 把分工讲清：系统跑了什么、系统有什么能力、模型负责什么。

    来源说明归系统，所以 prompt 要明确写「你不必在正文里交代来源」——实测让模型
    每轮复述那句话做不到（四种写法分别 1/1、2/3、0/4、0/3），这句要求留着只会
    让模型在正文里自由发挥一个来源描述。模型这边只剩内容与一个判断。
    """
    llm = _StubDirectAnswerLLM()
    _run_direct_answer(llm, question="什么是注意力机制")
    prompt = llm.prompts[0]

    assert _SYSTEM_STATE_WORDING in prompt
    assert _DISCLOSURE_OWNED_BY_SYSTEM_WORDING in prompt
    assert _CAPABILITY_LIST_WORDING in prompt
    assert _KNOWLEDGE_FLAG_WORDING in prompt


def test_provenance_line_is_appended_by_system_for_knowledge_answers() -> None:
    """模型说这段是知识性内容，系统就把来源说明另起一段附上。

    这句话由系统写死：措辞不会每次不同、也不会漏，更不会编出没发生过的取证过程。
    用引用块另起一段，是让「系统说的话」和「模型说的话」在正文里就分得开。
    """
    llm = _StubDirectAnswerLLM("注意力机制是……", knowledge_bearing=True)
    payload = _run_direct_answer(llm, question="什么是注意力机制")

    assert payload["answer"] == f"注意力机制是……\n\n> {_PROVENANCE_LINE}"


def test_provenance_line_stays_out_of_chitchat() -> None:
    """纯寒暄不附来源说明：「你好」底下挂一句「本轮未检索论文」是噪声。"""
    llm = _StubDirectAnswerLLM("你好哇，今天想聊点什么？", knowledge_bearing=False)
    payload = _run_direct_answer(llm, question="你好")

    assert payload["answer"] == "你好哇，今天想聊点什么？"
    assert _PROVENANCE_LINE not in payload["answer"]


def test_provenance_line_mentions_memory_when_injected() -> None:
    """本轮提示词真带了项目记忆时，来源说明要提到记忆，并点明记忆不是论文原文。"""
    llm = _StubDirectAnswerLLM("结合项目先前判断……", knowledge_bearing=True)
    payload = _run_direct_answer(llm, question="我们之前定的主干是哪个", memory_block=_memory_block())

    assert payload["answer"] == f"结合项目先前判断……\n\n> {_PROVENANCE_WITH_MEMORY_LINE}"


def test_provenance_line_keeps_original_text_when_memory_absent() -> None:
    """没读到记忆（空或读取失败没渲染出来）时维持原始说明，不谎称结合了记忆。"""
    llm = _StubDirectAnswerLLM("注意力机制是……", knowledge_bearing=True)
    payload = _run_direct_answer(llm, question="什么是注意力机制", memory_block="")

    assert payload["answer"] == f"注意力机制是……\n\n> {_PROVENANCE_LINE}"
    assert _PROVENANCE_WITH_MEMORY_LINE not in payload["answer"]


def test_memory_present_chitchat_gets_no_provenance() -> None:
    """带了记忆但这一轮只是寒暄：仍不附来源说明，记忆存在不改变纯寒暄的分支。"""
    llm = _StubDirectAnswerLLM("你好哇，今天想聊点什么？", knowledge_bearing=False)
    payload = _run_direct_answer(llm, question="你好", memory_block=_memory_block())

    assert payload["answer"] == "你好哇，今天想聊点什么？"
    assert _PROVENANCE_LINE not in payload["answer"]
    assert _PROVENANCE_WITH_MEMORY_LINE not in payload["answer"]


def _memory_read_state(status: str) -> dict[str, Any]:
    """按真实状态形状造一份「本轮没读到记忆」的 state，交给真实渲染器取值。

    参数:
        status: 记忆读取终态（empty / failed 之类）
    返回:
        含 memory_read_result 的状态字典
    """
    result: dict[str, Any] = {"status": status, "items": [], "excluded_counts": {}}
    if status == "failed":
        result["error_kind"] = "RuntimeError"
    return {"memory_read_result": result}


@pytest.mark.parametrize("status", ["empty", "failed"])
def test_provenance_keeps_original_text_when_memory_read_empty_or_failed(status: str) -> None:
    """记忆读取为空或失败的真实链路：渲染器给空串，来源说明维持原句、不谎称结合了记忆。"""
    memory_block = render_project_memory_block(_memory_read_state(status))
    assert memory_block == ""
    llm = _StubDirectAnswerLLM("注意力机制是……", knowledge_bearing=True)

    payload = _run_direct_answer(llm, question="什么是注意力机制", memory_block=memory_block)

    assert "【项目研究记忆】" not in llm.prompts[0]
    assert payload["answer"] == f"注意力机制是……\n\n> {_PROVENANCE_LINE}"
    assert _PROVENANCE_WITH_MEMORY_LINE not in payload["answer"]


def test_llm_error_propagates_instead_of_template_answer() -> None:
    """模型报错向上传播，不被吞成一句「暂时无法回答」的模板正文。"""
    llm = _RaisingLLM(RuntimeError("upstream model timeout"))
    with pytest.raises(RuntimeError, match="upstream model timeout"):
        _run_direct_answer(llm)


def test_empty_answer_cannot_pass_the_contract() -> None:
    """空正文在契约层就进不来，不会被当成一条正常回答写进会话。

    改成结构化输出后，空正文由 `DirectAnswerDraft` 的 min_length 挡下：解析失败走
    结构化失败路径，而不是退回一句模板正文把模型故障伪装成答上了。
    """
    with pytest.raises(ValidationError):
        DirectAnswerDraft(answer="   ", knowledge_bearing=True)
    with pytest.raises(ValidationError):
        _run_direct_answer(_StubDirectAnswerLLM("   "))
