"""项目研究记忆的回合写回：把本轮已验证判断落成待审候选，并组装回合终态。

写回在同次答案成功提交后执行，只消费目标 run 保存的已有材料：
最终答案、已验证 claim、本轮 Evidence 登记表、项目身份、来源回答身份。
它只核对明确支持片段的不可变原文，不重新整理整篇论文，也不调用手动 curate()。

四态语义按 MemoryWritebackResult 的约定各自独立：
skipped=深度或终态压根没触发评估；empty=评估过但本轮没够格的判断；
failed=存储/模型侧出问题，答案照常返回、原因码与 error_kind 如实记账。

作者：xxx
2026-08-30 16:40:00
"""

from __future__ import annotations

from app.services.internal_research.research_planner_final_delivery import (
    delivery_report_from_state,
    project_final_delivery,
)

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.middleware.trace_context import trace_label
from app.models.research_contracts import (
    VALID_CAPABILITY_STATUSES,
    MemoryWritebackResult,
    ResearchTurnOutcome,
)
from app.models.research_planner_contracts import ResearchClaim, ResearchUnitResult
from app.services.internal_research.research_planner_support import resolve_support
from app.services.research_memory_curator_service import (
    CuratorEvidenceCandidate,
    CuratorMemoryCandidate,
    _curator_llm,
    persist_candidates,
)

logger = logging.getLogger(__name__)


def prepare_committed_memory_writeback(state: dict[str, Any]) -> MemoryWritebackResult:
    """【研究记忆】【提交初态】参数为待保存产物；返回未执行候选处理事实，不调用模型或存储。"""
    return {"status": "pending", "candidate_ids": [], "reason_codes": ["awaiting_candidate_processing"],
            "retryable": False, "error_kind": ""}


async def after_result_committed(committed: dict[str, Any], *, processing_source: Literal["delivery_run", "http_retry"] = "delivery_run") -> MemoryWritebackResult:
    """参数为已成功提交的完整结果；只按该版本产物写回候选，并保存独立处理状态供重试。"""
    from app.repositories.rag_run_repository import get_rag_run_repository
    from app.repositories.result_delivery_repository import ResultDeliveryRepository

    run = committed["run"]
    revision = committed.get("delivery_revision") or run["metadata"]["delivery_revision"]
    current = await get_rag_run_repository().get_run(user_id=run["user_id"], run_id=run["run_id"])
    if current is None or current["metadata"].get("delivery_revision") != revision:
        raise ValueError("已提交答案版本已变化，不能用旧结果写回记忆")
    metadata = current["metadata"]
    previous = metadata.get("memory_writeback") or {}
    if previous.get("status") in {"success", "empty"} or (
            previous.get("status") == "skipped" and "awaiting_result_commit" not in previous.get("reason_codes", [])):
        return previous
    state = {**metadata, "source_delivery_revision": revision, "answer": current["answer"], "conversation_id": current.get("conversation_id"),
             "session_id": current.get("session_id")}
    if not state.get("project_scope") and metadata.get("project_id"):
        state["project_scope"] = {"project_id": metadata["project_id"], "user_id": run["user_id"], "paper_ids": metadata.get("paper_ids", [])}
    repository = ResultDeliveryRepository()
    # 1. 【研究记忆】【处理归属】HTTP重试没有持久执行owner，不能继承已经停止的原答案执行事实
    owner_run_id = committed.get("operation_run_id") if processing_source == "delivery_run" else None
    pending = {"status": "pending", "candidate_ids": [], "reason_codes": ["processing_committed_result"],
               "retryable": True, "error_kind": "", "processing_source": processing_source, "processing_run_id": owner_run_id}
    claimed = await repository.record_memory_writeback(user_id=run["user_id"], run_id=run["run_id"], delivery_revision=revision, outcome=pending)
    if claimed is False:
        latest = await get_rag_run_repository().get_run(user_id=run["user_id"], run_id=run["run_id"])
        if latest is None or latest["metadata"].get("delivery_revision") != revision:
            raise ValueError("候选处理对应的答案版本已变化")
        return latest["metadata"]["memory_writeback"]
    try:
        outcome = await write_back_turn_memory(state, request_id=run["run_id"])
    except asyncio.CancelledError:
        from app.services.internal_research.research_planner_lifecycle import settle_owned_work

        cleanup = asyncio.create_task(repository.record_memory_writeback(user_id=run["user_id"], run_id=run["run_id"],
            delivery_revision=revision, outcome=_failed("CancelledError")))
        result, _cancellation = await settle_owned_work(cleanup)
        if isinstance(result, BaseException):
            raise result
        raise
    await repository.record_memory_writeback(user_id=run["user_id"], run_id=run["run_id"], delivery_revision=revision, outcome=outcome)
    return outcome

# 只有 research 深度会评估写回：minimal 不检索、lookup 只做单次取证，两者都产不出可复用的项目判断
_WRITEBACK_DEPTH = "research"
# 一轮最多落几条候选，对齐 prd「生成少量 pending_review 候选」
_MAX_WRITEBACK_CANDIDATES = 3
# 送进提示的单条判断正文上限，避免把整段答案塞进模型
_MAX_CLAIM_CHARS = 600
# 答案正文进提示的上限：写回只需要判断的语境，不需要完整长文
_MAX_ANSWER_CHARS = 1500
# 循环终止决策里代表「本轮没能正常收敛」的两种，写回不在这种回合上评估
_BLOCKING_LOOP_DECISIONS = ("terminate_fail", "terminate_honest")
# planner 证据不足的终态取值，与 research_planner_synthesis._source_status 一致
_INSUFFICIENT_PLANNER_STATUS = "insufficient"

# ─── 稳定原因码：进 trace 供「写回触发准确率」取数，禁止改字面量 ───

# skipped：深度没触发，后缀带上本轮真实深度
_REASON_DEPTH_PREFIX = "skipped_depth_"
# skipped：终态没触发，后缀带上真实终态标签
_REASON_TERMINAL_PREFIX = "skipped_terminal_"
_TERMINAL_NO_ANSWER = "no_final_answer"
_TERMINAL_NO_VALIDATED_CLAIM = "no_validated_claim"
_TERMINAL_EVIDENCE_INSUFFICIENT = "evidence_insufficient"
_TERMINAL_ERROR_PREFIX = "error_"
# empty：评估过但没有够格的判断
_REASON_NO_PROPOSAL = "no_candidate_proposed"
_REASON_NO_DURABLE_VALUE = "no_durable_project_value"
_REASON_NOT_REUSABLE = "no_reusable_judgment"
_REASON_TRANSIENT = "transient_wording_or_lookup_detail"
_REASON_UNLINKED_CLAIM = "proposal_not_linked_to_validated_claim"
_REASON_MEMORY_TYPE_REJECTED = "candidate_rejected_by_memory_type"
# success
_REASON_SUCCESS = "validated_claim_with_evidence"
# failed：身份不全时不重试，重试也不会变全
_ERROR_MISSING_IDENTITY = "MissingWritebackIdentity"

# 候选政策照抄 prd §7.2，逐条进提示：模型只负责判断这几条特征，成熟度与置信度仍由 curator 推导
_POLICY_RULE_LINES = (
    "满足以下特征时可以生成少量 pending_review 候选：",
    "- 对当前项目后续研究有持续价值",
    "- 具有清晰、可复用的认识或决定",
    "- 能追溯到本轮回答和真实 Evidence，或明确标为用户项目决策",
    "- 不是寒暄、临时措辞、单次 lookup 细节或尚未验证的猜测",
    "不满足条件时应记录「本轮不适合形成记忆」的原因，而不是创建空泛候选。",
)

# 循环决策 → 回合终态与终止原因的换算表，两套词表显式对齐，不直接把决策塞进终态
_OUTCOME_BY_LOOP_DECISION = {
    "terminate_fail": ("failed", "terminate_fail"),
    "terminate_honest": ("incomplete", "terminate_honest"),
}
_DEFAULT_OUTCOME = ("completed", "converged")
# 取证部分完成时的终态覆盖：只答了一部分（有的子问题查到、有的没查清）不能被记成
# 正常完成，否则用户看到的与统计里的都是「答完了」，部分交付这件事在留痕里消失
_PARTIAL_EVIDENCE_OUTCOME = ("incomplete", "partial_evidence")
# planner 自己的论文取证终态取值，取这个值代表本轮交付的是部分结果
_PARTIAL_PLANNER_STATUS = "partial"
# planner 论文取证终态 → 四态能力结果，两套词表不同源，必须显式换算。
# 部分取证（有单元查到、有单元没查清）换算成 empty 而不是 success：按 CapabilityStatus
# 词表，这不是「四态里的部分」，按空值走，留痕里才不会把半份取证读成完整成功。
# 部分完成档由回合终态 status=incomplete 承担（见 _OUTCOME_BY_PARTIAL_EVIDENCE）
_PAPER_RETRIEVAL_CAPABILITY = {
    "ok": "success",
    "partial": "empty",
    "insufficient": "empty",
    "no_evidence": "empty",
    "failed": "failed",
    "skipped": "skipped",
    "not_run": "skipped",
}
_CAPABILITY_MEMORY_READ = "read_project_memory"
_CAPABILITY_PAPER_RETRIEVAL = "retrieve_paper_evidence"
# 动作名取 design.md §6.2 循环里的那一个，别另起一个名字
_CAPABILITY_ASK_CLARIFICATION = "ask_clarification"
_CAPABILITY_MEMORY_WRITEBACK = "write_back_project_memory"


class TurnMemoryProposal(BaseModel):
    """模型对一条已验证判断的写回提议；四个政策标记分别对应 prd §7.2 的四条特征。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    # 提议所依据的已验证判断 id，必须来自提示里给出的清单，服务层据此回连真实 claim
    source_claim_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = ""
    memory_type: str = Field(min_length=1)
    is_synthesis: bool = False
    # 政策一：对当前项目后续研究有持续价值
    durable_project_value: bool = False
    # 政策二：具有清晰、可复用的认识或决定
    reusable_judgment: bool = False
    # 政策四：是寒暄、临时措辞、单次 lookup 细节或未验证猜测（命中即淘汰）
    transient_or_lookup_detail: bool = True


class TurnMemoryProposalBatch(BaseModel):
    """一次回合写回的提议集合。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    proposals: list[TurnMemoryProposal] = Field(default_factory=list)
    # 模型认为本轮不适合形成记忆时给出的说明，进日志便于排查空轮
    not_suitable_reason: str = ""


async def write_back_turn_memory(
    state: dict[str, Any],
    *,
    request_id: str,
    llm: Any = None,
) -> MemoryWritebackResult:
    """
    在回合终结处评估并写回项目研究记忆候选。

    参数:
        state: 终结节点看到的图状态，读 turn_depth / project_scope / 已验证判断 / Evidence
        request_id: 本次回答身份（rag_traces 主键），落成候选的 source_answer_id 供幂等
        llm: 可注入的模型实例，默认用 curator 配置的模型
    返回:
        MemoryWritebackResult：四态终态 + 候选 id + 稳定原因码
    """
    # 1. 深度闸门：非 research 深度压根不评估，原因码带上本轮真实深度
    depth = str(state.get("turn_depth") or "").strip()
    if depth != _WRITEBACK_DEPTH:
        return _skipped(f"{_REASON_DEPTH_PREFIX}{depth or 'unset'}")
    # 2. 终态闸门：没正常收敛的回合不评估，原因码带上真实终态而不是报 empty
    blocking = _blocking_terminal_label(state)
    if blocking:
        return _skipped(f"{_REASON_TERMINAL_PREFIX}{blocking}")
    try:
        return await _write_back_research_turn(state, request_id=request_id, llm=llm)
    except Exception as exc:
        # 3. 写回是回答之后的动作：失败如实记 failed + 真实 error_kind，答案照常返回
        logger.error(
            "【记忆写回】【候选落库】写回失败 request_id=%s error_kind=%s",
            request_id,
            type(exc).__name__,
            exc_info=True,
        )
        return _failed(type(exc).__name__)


async def _write_back_research_turn(
    state: dict[str, Any],
    *,
    request_id: str,
    llm: Any,
) -> MemoryWritebackResult:
    """
    research 回合的写回主流程：身份校验 → 模型提议 → 政策筛选 → 落待审草稿。

    参数:
        state: 终结节点的图状态
        request_id: 来源回答身份
        llm: 可注入模型实例
    返回:
        MemoryWritebackResult 终态
    """
    # 1. 身份闸门：项目/用户/来源回答缺任一都无法归属候选，重试也补不回来
    identity = _writeback_identity(state, request_id=request_id)
    if identity is None:
        return _failed(_ERROR_MISSING_IDENTITY, retryable=False)
    # 2. 模型只在本轮已验证判断上提议，未验证的猜测进不了提示
    claims = _turn_validated_claims(state)
    await _verify_writeback_sources(state, claims=claims)
    batch = await _propose_candidates(state, claims=claims, llm=llm)
    accepted, rejections = _accept_proposals(
        batch.proposals,
        claims={claim.claim_id: claim for claim in claims},
    )
    if not accepted:
        logger.info(
            "【记忆写回】【候选评估】本轮不适合形成记忆 request_id=%s model_reason=%s reason_codes=%s",
            identity.answer_id,
            batch.not_suitable_reason,
            rejections,
        )
        return _empty(rejections or [_REASON_NO_PROPOSAL])
    # 3. 落 pending_review 草稿：三轴形状与同一次回答的指纹去重都由 curator 统一保证
    return await _persist_accepted(accepted, identity=identity, state=state)


@dataclass(frozen=True)
class _WritebackIdentity:
    """候选归属身份：项目 + 归属人 + 来源回答，三者缺一不可落库。"""

    project_id: str
    user_id: str
    answer_id: str
    conversation_id: str
    session_id: str


def _writeback_identity(state: dict[str, Any], *, request_id: str) -> _WritebackIdentity | None:
    """
    从本轮项目范围与来源回答身份取候选归属，缺项即视为接线错误。

    参数:
        state: 终结节点的图状态，项目身份取自 project_scope（与记忆读取同源）
        request_id: 来源回答身份
    返回:
        三项齐全时返回身份对象；缺任一项返回 None，由调用方记 failed
    """
    scope = _dict_value(state, "project_scope")
    identity = _WritebackIdentity(
        project_id=str(scope.get("project_id") or "").strip(),
        user_id=str(scope.get("user_id") or "").strip(),
        answer_id=str(request_id or "").strip(),
        conversation_id=str(state.get("conversation_id") or "").strip(),
        session_id=str(state.get("session_id") or "").strip(),
    )
    if not identity.project_id or not identity.user_id or not identity.answer_id:
        return None
    return identity


async def _persist_accepted(
    accepted: list[TurnMemoryProposal],
    *,
    identity: _WritebackIdentity,
    state: dict[str, Any],
) -> MemoryWritebackResult:
    """
    把过了政策的提议落成待审草稿，并按真实落库结果给终态。

    参数:
        accepted: 已过政策筛选的提议
        identity: 候选归属身份
    返回
        落库成功返回 success + 候选 id；一条都没落住时返回 empty 并记原因码
    """
    drafts = await persist_candidates(
        identity.project_id,
        [_to_curator_candidate(proposal, state=state) for proposal in accepted],
        user_id=identity.user_id,
        source_answer_id=identity.answer_id,
        source_conversation_id=identity.conversation_id,
        source_session_id=identity.session_id,
        source_delivery_revision=state.get("source_delivery_revision"),
    )
    candidate_ids = [str(detail["item"]["memory_id"]) for detail in drafts if detail.get("item")]
    # 候选被 curator 的 memory_type 枚举挡掉时列表为空，空列表绝不冒充 success
    if not candidate_ids:
        return _empty([_REASON_MEMORY_TYPE_REJECTED])
    logger.info(
        "【记忆写回】【候选落库】request_id=%s project_id=%s candidate_count=%s",
        identity.answer_id,
        identity.project_id,
        len(candidate_ids),
    )
    return {
        "status": "success",
        "candidate_ids": candidate_ids,
        "reason_codes": [_REASON_SUCCESS],
        "retryable": False,
        "error_kind": "",
    }


def _blocking_terminal_label(state: dict[str, Any]) -> str:
    """
    判断本回合终态是否挡住写回，并给出可读的终态标签。

    终态不合格与「评估过但没东西可记」是两件事：这里返回非空即走 skipped，
    绝不并进 empty，否则分不清「写回没跑」和「跑过且结论是不该记」。

    参数:
        state: 终结节点的图状态
    返回:
        挡住写回的终态标签；正常收敛返回空串
    """
    # 1. 【研究交付】【记忆准入】失败与不足正文不能作为成功回合进入候选评估
    delivery = delivery_report_from_state(state)
    if delivery is not None and delivery.status in {"failed", "insufficient"}:
        return f"final_delivery_{delivery.status}"
    # 2. partial 继续使用已有准入政策，不扩大自动接纳范围
    decision = str(state.get("planner_loop_decision") or "").strip()
    if decision in _BLOCKING_LOOP_DECISIONS:
        return decision
    # 2. 主链已记错误码：错误码本身就是终态标签
    error_code = str(state.get("error_code") or "").strip()
    if error_code:
        return f"{_TERMINAL_ERROR_PREFIX}{error_code}"
    # 3. planner 论文取证判为不足：答案不足以支撑可复用判断
    if str(_dict_value(state, "source_status").get("planner") or "") == _INSUFFICIENT_PLANNER_STATUS:
        return _TERMINAL_EVIDENCE_INSUFFICIENT
    # 4. 没有定稿答案 / 没有任何已验证判断：没有可写回的材料
    if not str(state.get("answer") or "").strip():
        return _TERMINAL_NO_ANSWER
    if not _turn_validated_claims(state):
        return _TERMINAL_NO_VALIDATED_CLAIM
    return ""


def _accept_proposals(
    proposals: list[TurnMemoryProposal],
    *,
    claims: dict[str, ResearchClaim],
) -> tuple[list[TurnMemoryProposal], list[str]]:
    """
    按 prd §7.2 的四条特征逐条筛选提议，淘汰的记原因码。

    参数:
        proposals: 模型提议
        claims: 本轮已验证判断，按 claim_id 索引
    返回:
        (通过筛选的提议, 淘汰原因码去重后的列表)
    """
    accepted: list[TurnMemoryProposal] = []
    rejections: list[str] = []
    for proposal in proposals:
        reason = _rejection_reason(proposal, claims=claims)
        if reason:
            if reason not in rejections:
                rejections.append(reason)
            continue
        accepted.append(proposal)
    return accepted[:_MAX_WRITEBACK_CANDIDATES], rejections


def _rejection_reason(
    proposal: TurnMemoryProposal,
    *,
    claims: dict[str, ResearchClaim],
) -> str:
    """
    给一条提议判「淘汰原因码」，空串表示可以形成候选。

    参数:
        proposal: 模型提议
        claims: 本轮已验证判断，按 claim_id 索引
    返回:
        淘汰原因码；通过全部政策时返回空串
    """
    claim = claims.get(proposal.source_claim_id)
    # 1. 提议必须挂在提示里给出的已验证判断上，模型自造 claim_id 一律不采信
    if claim is None:
        return _REASON_UNLINKED_CLAIM
    # 2. 政策一：对当前项目后续研究有持续价值
    if not proposal.durable_project_value:
        return _REASON_NO_DURABLE_VALUE
    # 3. 政策二：具有清晰、可复用的认识或决定
    if not proposal.reusable_judgment:
        return _REASON_NOT_REUSABLE
    # 4. 政策四：寒暄、临时措辞、单次 lookup 细节或未验证猜测，命中即淘汰
    if proposal.transient_or_lookup_detail:
        return _REASON_TRANSIENT
    # 5. 政策三：判断要么能回连本轮真实 Evidence，要么明确标为用户项目决策
    # 5. 【研究记忆】【支持闭包】claims 已由 O4 统一支持资格筛选，纯继承不再按直接引用重复拒绝
    return ""


async def _propose_candidates(
    state: dict[str, Any],
    *,
    claims: list[ResearchClaim],
    llm: Any,
) -> TurnMemoryProposalBatch:
    """
    让模型在本轮已验证判断上提议记忆候选，结构化校验失败由 llm_service 显式抛错。

    参数:
        state: 终结节点的图状态，取问题与定稿答案作为语境
        claims: 本轮已验证判断
        llm: 可注入模型实例；为空时用 curator 配置的模型
    返回:
        TurnMemoryProposalBatch：提议列表 + 模型自述的「不适合形成记忆」原因
    """
    from app.services.research_memory_stale_service import MemoryCheckedModel

    drafter = MemoryCheckedModel(llm or _curator_llm(), state).with_structured_output(TurnMemoryProposalBatch)
    prompt = _build_writeback_prompt(state, claims=claims)
    with trace_label("write_back", "memory_writeback"):
        return await drafter.ainvoke([HumanMessage(content=prompt)])


async def _verify_writeback_sources(state: dict[str, Any], *, claims: list[ResearchClaim]) -> None:
    """参数为已提交产物及合格主张；发送候选模型前重验每个明确支持，撤权内容不能继续消费。"""
    from app.models.research_planner_contracts import ResearchPlan
    from app.services.memory_evidence_service import validate_memory_evidence

    scope = state["project_scope"]
    supports = resolve_support(ResearchPlan.model_validate(state["research_plan"]),
        [ResearchUnitResult.model_validate(row) for row in state["research_unit_results"]], state["evidence_records"])
    registry = {row["evidence_id"]: row for row in state["evidence_records"]}
    checked = set()
    for claim in claims:
        for ref in _candidate_support_evidence(supports[claim.claim_id], supports=supports, registry=registry):
            identity = (ref.paper_id, ref.chunk_id, ref.source_field, ref.source_start, ref.source_end)
            if identity in checked:
                continue
            await validate_memory_evidence(ref.model_dump(exclude_none=True), user_id=scope["user_id"], project_id=scope["project_id"])
            checked.add(identity)


def _build_writeback_prompt(state: dict[str, Any], *, claims: list[ResearchClaim]) -> str:
    """
    拼写回提示：政策逐条给出，判断清单只含已验证 claim，答案与论文原文都不重扫。

    参数:
        state: 终结节点的图状态
        claims: 本轮已验证判断
    返回:
        提示正文
    """
    return "\n".join([
        "你是项目研究记忆的写回助手。判断本轮问答里有没有值得长期留在项目里的「判断」。",
        "只输出 JSON，验证为 TurnMemoryProposalBatch。禁止 Markdown、注释或 JSON 外的散文。",
        *_POLICY_RULE_LINES,
        "",
        "字段规则：",
        f"- source_claim_id 必须取自下面的已验证判断清单，不得自造；最多提议 {_MAX_WRITEBACK_CANDIDATES} 条。",
        "- title 写一句可复用的判断，body 写这条判断的成立依据。",
        "- memory_type 从受控枚举里取；用户的项目决策用 decision_note，研究目的用 research_purpose。",
        "- durable_project_value / reusable_judgment / transient_or_lookup_detail 按上面四条特征逐条如实判断。",
        "- 不要输出 maturity 或 confidence，这些由系统按证据校验结果判定。",
        "- 一条都不够格时 proposals 给空数组，并在 not_suitable_reason 说明原因。",
        "",
        f"本轮问题：{str(state.get('user_query') or state.get('query') or '')}",
        "",
        "本轮定稿答案：",
        str(state.get("answer") or "")[:_MAX_ANSWER_CHARS],
        "",
        "本轮已验证判断清单：",
        *_claim_prompt_lines(claims),
    ])


def _claim_prompt_lines(claims: list[ResearchClaim]) -> list[str]:
    """
    把已验证判断渲染成提示行，带 claim_id 与证据 id 供模型回连。

    参数:
        claims: 本轮已验证判断
    返回:
        每条判断一行的提示文本；无判断时给出显式占位
    """
    if not claims:
        return ["（无）"]
    return [
        f"- claim_id={claim.claim_id} type={claim.claim_type} "
        f"evidence={','.join(claim.supporting_evidence_ids) or '(none)'} "
        f"text={claim.text[:_MAX_CLAIM_CHARS]}"
        for claim in claims
    ]


def _to_curator_candidate(proposal: TurnMemoryProposal, *, state: dict[str, Any]) -> CuratorMemoryCandidate:
    """参数为提议及已提交产物；保留原claim/attempt与精确支持范围，禁止降级成无依据笔记。"""
    from app.models.research_planner_contracts import ResearchPlan

    supports = resolve_support(ResearchPlan.model_validate(state["research_plan"]),
        [ResearchUnitResult.model_validate(row) for row in state["research_unit_results"]], state["evidence_records"])
    support = supports[proposal.source_claim_id]
    registry = {row["evidence_id"]: row for row in state["evidence_records"]}
    evidence = _candidate_support_evidence(support, supports=supports, registry=registry)
    if not evidence:
        raise ValueError("自动候选缺少可核验的原文支持，不能改名为个人判断")
    return CuratorMemoryCandidate(title=proposal.title, body=proposal.body, memory_type=proposal.memory_type,
        is_synthesis=proposal.is_synthesis, evidence=evidence, source_claim_id=proposal.source_claim_id,
        source_attempt_id=state["final_generation_attempt"]["attempt_id"])


def _candidate_support_evidence(support: Any, *, supports: dict[str, Any], registry: dict[str, Any]) -> list[CuratorEvidenceCandidate]:
    """参数为统一支持闭包与原材料；把实际主张支持区间映回原文，保留继承路径且不扩整材料。"""
    evidence = []
    for claim_id in (support.claim.claim_id, *support.claim_ids):
        for selected in supports[claim_id].claim.supporting_spans:
            if selected.evidence_id not in support.citable_evidence_ids:
                continue
            record = registry[selected.evidence_id]
            field = "expanded_text" if record.get("expanded_text") else "excerpt"
            if selected.field != field or not record.get("source_spans"):
                raise ValueError("已提交支持缺少与阅读正文对应的精确原文区间")
            text = record[field]
            matched = False
            for span in record["source_spans"]:
                start, end = max(selected.start, span["text_start"]), min(selected.end, span["text_end"])
                if start >= end:
                    continue
                matched = True
                source_start = span["start"] + start - span["text_start"]
                evidence.append(CuratorEvidenceCandidate(paper_id=span["paper_id"], chunk_id=span["chunk_id"],
                    quote=text[start:end], source_field=span["field"], source_start=source_start,
                    source_end=source_start + end - start,
                    ref_role="primary_support" if claim_id == support.claim.claim_id else "derived_basis",
                    support={"source_claim_id": support.claim.claim_id, "supporting_claim_id": claim_id,
                        "evidence_id": selected.evidence_id, "inherited_claim_ids": list(support.claim_ids),
                        "selected_support": selected.model_dump(mode="json"), "source_span": span}))
            if not matched:
                raise ValueError("主张支持范围没有实际原文对应")
    return evidence


def build_turn_outcome(
    state: dict[str, Any],
    *,
    memory_writeback: MemoryWritebackResult,
) -> ResearchTurnOutcome:
    """
    组装回合终态，字段全部取自本轮真实产出，缺来源的不编默认值。

    参数:
        state: 终结节点的图状态
        memory_writeback: 本轮写回终态，作为写回能力的四态结果来源
    返回:
        ResearchTurnOutcome：终态 + 深度 + 四态能力表 + 记忆/证据/缺口 + 循环步数
    """
    # 1. 循环决策换算成终态与终止原因：两套词表分开，正常收敛才是 completed。
    #    决策收敛但取证只完成了一部分时改判 incomplete：终态由最终产出决定，不看决策动作
    decision = str(state.get("planner_loop_decision") or "").strip()
    status, termination_reason = _OUTCOME_BY_LOOP_DECISION.get(decision, _DEFAULT_OUTCOME)
    if status == "completed" and _dict_value(state, "source_status").get("planner") == _PARTIAL_PLANNER_STATUS:
        status, termination_reason = _PARTIAL_EVIDENCE_OUTCOME
    # 2. 【研究交付】【回合投影】研究交付是完成度权威，循环动作与取证能力保持各自含义
    delivery = delivery_report_from_state(state)
    if delivery is not None:
        projection = project_final_delivery(delivery)
        status, termination_reason = projection["turn_status"], projection["termination_reason"]
    assessment = _dict_value(state, "turn_depth_assessment")
    memory_read = _dict_value(state, "memory_read_result")
    return {
        "status": status,
        "final_depth": str(state.get("turn_depth") or ""),
        "depth_reason_code": str(assessment.get("reason_code") or ""),
        # 2. 循环内没有任何代码改深度，本轮就没有深度变更轨迹，留空数组不造生产方
        "depth_transitions": [],
        "capability_results": _capability_results(state, memory_writeback=memory_writeback),
        "memory_item_ids": [
            str(item.get("memory_id"))
            for item in memory_read.get("items") or []
            if isinstance(item, dict) and item.get("memory_id")
        ],
        # 3. 最终被引用的证据以定稿答案块为准，不拿证据池冒充「被引用」
        "evidence_ids": _final_evidence_ids(state),
        "missing_evidence": state.get("missing_evidence") or [],
        # 4. 循环实际走了几步：只作观察记录，不参与任何额度判断（额度已取消）
        "loop_steps": int(_dict_value(state, "planner_budget").get("total_steps") or 0),
        "termination_reason": termination_reason,
        "error_kind": str(state.get("error_kind") or ""),
    }


def _capability_results(
    state: dict[str, Any],
    *,
    memory_writeback: MemoryWritebackResult,
) -> dict[str, Any]:
    """
    把各能力的原生状态换算成统一四态表，没跑过的能力不进表。

    参数:
        state: 终结节点的图状态
        memory_writeback: 本轮写回终态
    返回:
        能力名 → CapabilityStatus；来源缺失或取值不在词表内的能力整条不出现
    """
    results: dict[str, Any] = {}
    # 1. 记忆读取：读过就用读取结果的四态；模型判定本轮不必翻记忆才算 skipped
    memory_read_status = str(_dict_value(state, "memory_read_result").get("status") or "")
    assessment = _dict_value(state, "turn_depth_assessment")
    if memory_read_status:
        results[_CAPABILITY_MEMORY_READ] = memory_read_status
    elif assessment.get("needs_project_memory") is False:
        results[_CAPABILITY_MEMORY_READ] = "skipped"
    # 2. 论文取证：planner 自己的终态词表与四态不同源，只收换算表里有的取值
    planner_status = str(_dict_value(state, "source_status").get("planner") or "")
    if planner_status in _PAPER_RETRIEVAL_CAPABILITY:
        results[_CAPABILITY_PAPER_RETRIEVAL] = _PAPER_RETRIEVAL_CAPABILITY[planner_status]
    # 3. 向用户澄清：判要问就是真问了（那句问题即本轮回答），判过不用问才算 skipped。
    #    压根没判过（状态里没有这一项）时整条不进表，免得把「没判过」读成「判过不用问」
    asked = assessment.get("needs_clarification")
    if asked is not None:
        results[_CAPABILITY_ASK_CLARIFICATION] = "success" if asked else "skipped"
    # 4. 候选写回：写回结果本身就是四态
    writeback_status = str((memory_writeback or {}).get("status") or "")
    if writeback_status:
        results[_CAPABILITY_MEMORY_WRITEBACK] = writeback_status
    # 5. 取证工具箱：execute 阶段按 source_type 如实产出的四态原样并入；
    #    取值不在 CapabilityStatus 词表内的整条不出现，不把未知状态带进终态
    for capability, status in _dict_value(state, "planner_evidence_capabilities").items():
        status_text = str(status or "").strip()
        if status_text in VALID_CAPABILITY_STATUSES:
            results[str(capability)] = status_text
    return results


def _final_evidence_ids(state: dict[str, Any]) -> list[str]:
    """
    取定稿答案块真实引用到的证据 id，按出现顺序去重。

    参数:
        state: 终结节点的图状态
    返回:
        evidence_id 列表
    """
    delivery = delivery_report_from_state(state)
    if delivery is not None:
        return list(dict.fromkeys(binding.evidence_id for binding in delivery.citation_bindings))
    ordered: list[str] = []
    for block in state.get("planner_final_blocks") or []:
        if not isinstance(block, dict):
            continue
        for evidence_id in block.get("evidence_ids") or []:
            text = str(evidence_id or "").strip()
            if text and text not in ordered:
                ordered.append(text)
    return ordered


def _turn_validated_claims(state: dict[str, Any]) -> list[ResearchClaim]:
    """
    取本轮通过验证的判断，未验证的猜测在这一步就被挡住。

    参数:
        state: 终结节点的图状态，判断来自 research_unit_results
    返回:
        validation_status 为 validated 的 ResearchClaim 列表
    """
    results = [
        ResearchUnitResult.model_validate(result)
        for result in state.get("research_unit_results") or []
        if isinstance(result, dict)
    ]
    # 1. 【研究交付】【候选来源】新版回合沿统一支持资格选主张，旧回合保持其原有读取语义
    if state.get("final_delivery_report") is not None and state.get("research_plan"):
        from app.models.research_planner_contracts import ResearchPlan

        support = resolve_support(ResearchPlan.model_validate(state["research_plan"]), results,
                                  state.get("evidence_records") or [])
        return [item.claim for item in support.values() if item.qualified and not item.hypothesis]
    # 2. 【研究记忆】【历史成果】旧回合缺统一支持事实时不自动生成新的文献候选
    return []


def _dict_value(state: dict[str, Any], key: str) -> dict[str, Any]:
    """
    读一个只在是字典时才有意义的 state 键。

    参数:
        state: 图状态
        key: 键名
    返回:
        该键的字典值；缺失或类型不符时返回空字典
    """
    value = state.get(key)
    return value if isinstance(value, dict) else {}


def _skipped(reason_code: str) -> MemoryWritebackResult:
    """深度或终态没触发评估：skipped 不与 empty 归并，原因码带上真实深度/终态。"""
    return {
        "status": "skipped",
        "candidate_ids": [],
        "reason_codes": [reason_code],
        "retryable": False,
        "error_kind": "",
    }


def _empty(reason_codes: list[str]) -> MemoryWritebackResult:
    """评估过但本轮没有够格形成记忆的判断：原因码如实记账，不落空泛候选。"""
    return {
        "status": "empty",
        "candidate_ids": [],
        "reason_codes": reason_codes,
        "retryable": False,
        "error_kind": "",
    }


def _failed(error_kind: str, *, retryable: bool = True) -> MemoryWritebackResult:
    """模型或存储侧出问题：如实落 failed 与真实 error_kind，答案照常返回。

    参数:
        error_kind: 真实异常类型名
        retryable: 该失败是否值得重试；身份缺失这类接线错误重试也补不回来
    返回:
        failed 终态
    """
    return {
        "status": "failed",
        "candidate_ids": [],
        "reason_codes": [],
        "retryable": retryable,
        "error_kind": error_kind,
    }
