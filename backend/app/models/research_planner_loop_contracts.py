"""Planner ReAct 循环的观察/决策契约（步数只作观察记录）。

这些模型是循环控制器（`research_planner_loop_controller`）与图节点之间的
数据契约：观察层把校验/执行结果归一成 `PlannerObservation`，纯函数决策器据
观察与留痕 `PlannerLoopBudget` 产出 `PlannerDecision`，每步转移留痕进 `PlannerLoopStep`。

`PlannerLoopBudget` 已不再承载任何研究额度：它只记本回合实际走了几步、各类动作
各发生几次，供留痕与用户可见的进度细节读取。是否继续由「这一步是否带来新东西」
判定（判据见 `research_planner_loop_controller` 的无进展比较），收敛兜底是图自身的
递归上限（配置 `rag_planner_graph_recursion_limit`），不是这里的计数。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import Field, model_validator

from app.models.evidence_layers import (
    PLANNER_SOURCE_LOCAL_CHUNK_METADATA,
    PLANNER_SOURCE_LOCAL_METADATA,
    PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE,
    PLANNER_SOURCE_LOCAL_SECTION_TEXT,
    PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
)
from app.models.research_contracts import CapabilityStatus
from app.models.research_planner_contracts import (
    PlannerContractModel,
    PreferredBlockType,
)


# 决策器输出的封闭动作枚举：有界 ReAct，动作集不可扩展
PlannerDecisionAction = Literal[
    "continue",          # 计划自洽 / 证据充分 → 进下一阶段（execute 或 synthesize）
    "replan",            # 观察回流驱动重规划 → 回 analyze（带反馈）
    "revise_answer",     # 正文问题直接回综合，不重新规划研究动作
    "repair_evidence",   # 证据硬缺口 → 再进一次 execute 做定向补检索
    "probe_context",     # 出计划前申请补充上下文 → 回 analyze（带探测结果）
    "terminate_honest",  # 范围内查过但无证据 → 诚实收尾（非硬失败）
    "terminate_fail",    # 计划始终不自洽 / 系统性失败 → 硬失败收尾
]

# 计划前探测与计划后执行共享累计事实，校验失败保持独立的合同诊断
PlannerObservationPhase = Literal["validate", "execute", "probe"]


class PlannerLoopStep(PlannerContractModel):
    """单轮「思考-行动-观察-决策」的留痕，进 planner_recovery_trace。"""

    step_index: int = Field(ge=0)
    stage: str = Field(min_length=1)
    action: PlannerDecisionAction
    reason: str = ""
    observation: dict[str, Any] = Field(default_factory=dict)


class PlannerLoopBudget(PlannerContractModel):
    """循环步数与动作计数的观察快照：跨 analyze/validate/execute 回边流转。

    这里只有观察，没有任何上界：计数的用途是留痕与进度展示，不参与「还能不能继续」
    的判断。无 reducer 的 Dict 键，计数只能靠节点显式「读旧值 → advance → 整体回写」
    推进（design §3.1.1），漏写则本回合步数统计失真。
    """

    total_steps: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)
    evidence_repair_count: int = Field(default=0, ge=0)
    probe_count: int = Field(default=0, ge=0)
    history: list[PlannerLoopStep] = Field(default_factory=list)


# 探测可申请的取证来源：只有不经向量检索的能力。
# 字面量与 app.models.evidence_layers 的 PLANNER_SOURCE_* 同源（Literal 语法只接受
# 字面量，无法直接引用常量），文件末尾的一致性断言防两处漂移。
# 故意不含 local_direct_text：探测是「出计划前补上下文」，不该在出计划前就烧一次向量检索
PlannerProbeSourceType = Literal[
    "local_section_structure",  # 列章节结构
    "local_section_text",       # 按章节路径取正文
    "local_chunk_metadata",     # 按元数据条件筛块（章节/块类型/页码）
    "local_structure_hint",     # 取结构线索
    # 论文详情：只要这篇在库里，未索引也能读到真实摘要与已存分析。
    # 与 local_chunk_metadata 分开：那个按块类型/页码筛块，这个读整篇的元数据
    "local_metadata",
]


class PlannerProbeRequest(PlannerContractModel):
    """模型出计划前申请的一份补充上下文。

    四个筛选字段与 `EvidencePolicy` 逐字对齐（类型与下限都一致）：这份申请下游会被
    合成成一个 ResearchUnit 去调同一批取证适配器，口径不一致就得在中间做字段转换。
    """

    source_type: PlannerProbeSourceType
    # 为什么要这份信息，进 trace，让「模型在探什么」可追查
    purpose: str = ""
    target_paper_ids: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    required_block_types: list[PreferredBlockType] = Field(default_factory=list)
    # 页码精确匹配、不做范围解析；元素下限 1，0 和负数在信任边界即被拒
    required_pages: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)


class PlannerProbeResult(PlannerContractModel):
    """一次探测的执行结果，回灌给模型再出计划。

    status 直接用取证能力的四态 `CapabilityStatus`，不新造状态词表：`empty`
    表示本次没有返回材料，`failed` 表示调用失败，两者不合并。是否实际读过正文、
    还有哪些内容未读由 read_report 说明；零返回不能单独证明目标内容不存在。
    """

    source_type: str
    status: CapabilityStatus
    # empty / failed 的人可读原因，让探测没拿到东西时失败清晰暴露
    reason: str = ""
    # 取回内容的人可读摘要，供回灌提示词
    findings: list[str] = Field(default_factory=list)
    # 【科研取证】【探测报告】工具读取事实独立于发现文本，旧保存结果没有该字段时保持未知
    read_report: dict[str, Any] | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    receipt_id: str = ""
    reused: bool = False
    actual_call_count: int = Field(default=0, ge=0)
    action_basis_keys: list[str] = Field(default_factory=list)


class PlannerProbeDecision(PlannerContractModel):
    """模型「要不要先补上下文再出计划」的结构化输出根对象。"""

    needs_probe: bool
    requests: list[PlannerProbeRequest] = Field(default_factory=list)
    # 判要探 / 不探的理由，进 trace
    reason: str = ""

    @model_validator(mode="after")
    def _validate_probe_decision(self) -> "PlannerProbeDecision":
        # 判要探却没给任何申请，这一轮就没有任何可执行的探测动作，白烧一步预算；
        # 在信任边界拒掉，不在下游造一份默认申请把模型的空输出兜圆
        if self.needs_probe and not self.requests:
            raise ValueError("needs_probe requires at least one probe request")
        return self


# 探测来源枚举必须是那几种非向量取证能力的常量取值：任一常量改名或本枚举漏改都在导入期炸
assert set(get_args(PlannerProbeSourceType)) == {
    PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE,
    PLANNER_SOURCE_LOCAL_SECTION_TEXT,
    PLANNER_SOURCE_LOCAL_CHUNK_METADATA,
    PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
    PLANNER_SOURCE_LOCAL_METADATA,
}


class PlannerObservation(PlannerContractModel):
    """阶段观察归一化载体：校验/执行结果压成决策器可读的确定性信号。

    决策器是纯函数、不做 I/O；一切需要查库（如 scoped 论文是否在索引里有内容）
    的判断都在观察层（observe_*）完成后落到本模型的布尔量上（design §4 注）。
    """

    phase: PlannerObservationPhase
    facts_version: str = "research-progress-v2"
    requirements_version: str = ""
    fulfilled_requirement_keys: list[str] = Field(default_factory=list)
    material_fact_keys: list[str] = Field(default_factory=list)
    locator_fact_keys: list[str] = Field(default_factory=list)
    pending_work_keys: list[str] = Field(default_factory=list)
    attempted_work_keys: list[str] = Field(default_factory=list)
    gap_keys: list[str] = Field(default_factory=list)
    # 阶段实际满足事实：validate=计划自洽；execute=必需要求与依赖满足
    ok: bool
    # validate 失败错误码（如 invalid_facet_id / explicit_facet_uncovered）
    error_code: str = ""
    # 观察到的问题码集合（校验 issue codes / 执行缺口类型）
    issue_codes: list[str] = Field(default_factory=list)
    # execute：是否存在可定向补检索的硬缺口（决定能否 repair_evidence）
    has_repairable_gap: bool = False
    # execute：本步被判受阻的单元 id（排序去重），是「缺口集是否变了」的一半判据：
    # 编号只用于当前执行定位；进展判定只消费稳定要求和实际来源事实
    blocked_unit_ids: list[str] = Field(default_factory=list)
    # execute：是否已拿到任何可引用证据（决定收敛时能综合还是诚实收尾）
    has_answerable_evidence: bool = False
    # execute：scoped 论文在向量库里是否确有可检索内容（false → 无需 replan 直接诚实收尾）
    scoped_in_index: bool | None = True
    # 诚实收尾用：实际尝试过的检索式（复用 derive_missing_evidence 的 attempted_queries）
    attempted_queries: list[str] = Field(default_factory=list)
    # 诚实收尾用：本轮 scoped 论文范围
    scoped_paper_ids: list[str] = Field(default_factory=list)
    # 观察摘要（进 trace，人可读）
    summary: str = ""


class PlannerDecision(PlannerContractModel):
    """决策器输出：下一步动作 + 理由。"""

    action: PlannerDecisionAction
    reason: str = ""
