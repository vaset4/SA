"""内部研究链路的正式合同模型。"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from app.models.evidence_layers import EvidenceLayer, EvidenceSource, VerificationStatus
from app.models.knowledge_access_contracts import SourceSpan
from app.models.upload_models import PublicationSnapshot


DIALOGUE_MODE_CHAT = "chat"
DIALOGUE_MODE_LOOKUP = "lookup"
DIALOGUE_MODE_RESEARCH = "research"
VALID_DIALOGUE_MODES = (
    DIALOGUE_MODE_CHAT,
    DIALOGUE_MODE_LOOKUP,
    DIALOGUE_MODE_RESEARCH,
)

ENTRY_PROJECT_SCOPE = "project_scope"
ENTRY_PAPER_FOCUS = "paper_focus"
ENTRY_CONVERSATION = "conversation"
ENTRY_WEB_SCOPE = "web_scope"


class ScopeOwnershipError(LookupError):
    """请求引用的项目、知识库或论文不属于当前用户。"""


class RetrievalScopeError(ValueError):
    """本地检索缺少经过归属校验的非空范围。"""

    def __init__(self, message: str = "本地检索需要至少一篇已授权论文") -> None:
        super().__init__(message)


class ProjectScopeContract(TypedDict):
    """项目级长期研究边界。"""

    user_id: str
    project_id: str
    library_ids: list[str]
    paper_ids: list[str]
    filter_expr: str
    local_retrieval_allowed: bool
    resolved_scope: dict[str, Any] | None
    has_project_scope: bool
    summary: str
    publication: NotRequired[PublicationSnapshot]


class PaperFocusContract(TypedDict):
    """当前问题的临时论文焦点。"""

    mode: Literal["none", "single_paper", "multi_paper", "project_scope"]
    paper_ids: list[str]
    is_explicit: bool
    source: str
    summary: str


class EvidenceLayerRole(TypedDict):
    """证据层角色定义。"""

    layer: str
    role: str
    output: str
    notes: str


class SynthesisState(TypedDict):
    """研究整理层合同。"""

    status: Literal["collecting", "ready", "insufficient"]
    dialogue_mode: Literal["chat", "lookup", "research"]
    question_type: str
    project_scope: ProjectScopeContract
    paper_focus: PaperFocusContract
    evidence_summary: dict[str, Any]
    dimension_status: dict[str, str]
    supported_dimensions: list[str]
    unsupported_dimensions: list[str]
    open_questions: list[str]
    evidence_layers: list[EvidenceLayerRole]
    notes: list[str]
    citations_count: int


class EvidenceRecord(TypedDict, total=False):
    """统一证据记录。"""

    evidence_id: str
    source: EvidenceSource
    layer: EvidenceLayer
    paper_id: str
    chunk_id: str
    parent_id: str
    section: str
    page_num: int
    excerpt: str
    expanded_text: str
    summary: str
    dimensions: list[str]
    confidence: float
    provenance: str
    verification_status: VerificationStatus
    usable_as_citation: bool
    read_contract_version: int
    material_group_id: str
    source_spans: list[SourceSpan]
    source_location_status: Literal["exact", "legacy_chunk", "no_physical_location"]


class SelectedEvidence(TypedDict, total=False):
    """Generation-facing direct-text evidence selected by the system."""

    evidence_id: str
    paper_id: str
    chunk_id: str
    parent_id: str
    section: str
    page_num: int
    title: str
    excerpt: str
    expanded_text: str
    score: float
    source_channel: str
    token_count: int


class EvidenceSelectionStats(TypedDict, total=False):
    """Observable counters for selected evidence budgeting."""

    used_tokens: int
    context_budget: int
    dropped_count: int
    dropped_chunk_ids: list[str]


class MissingEvidence(TypedDict, total=False):
    """缺失证据记录。"""

    paper_id: str
    dimension: str
    reason: str
    attempted_queries: list[str]


class AnswerBrief(TypedDict, total=False):
    """回答简报。"""

    intent: str
    focus: str
    usable_evidence: list[EvidenceRecord]
    missing_evidence: list[MissingEvidence]
    uncertainties: list[str]
    answer_guidance: str
    forbidden_claims: list[str]


# ─── 二阶段：意图识别与焦点解析 ───

Intent = Literal["lookup", "compare", "explore", "inventory"]

VALID_INTENTS: tuple[str, ...] = ("lookup", "compare", "explore", "inventory")

class IntentResult(TypedDict, total=False):
    """意图识别结果。"""

    intent: Intent
    confidence: float
    source: Literal["rule"]
    signals: list[str]


class FocusResolution(TypedDict, total=False):
    """焦点解析结果。"""

    project_scope: list[str]
    conversation_scope: list[str]
    turn_focus: list[str]
    fallback_reason: str


# ─── 项目科研回合：处理深度与终态合同 ───

# 服务端派生的回合处理深度：minimal 不检索论文，lookup 有界取证，research 多轮检索与综合。
TurnDepth = Literal["minimal", "lookup", "research"]

VALID_TURN_DEPTHS: tuple[str, ...] = ("minimal", "lookup", "research")

# 单个能力（记忆读取 / 论文检索 / 候选提取 / 候选保存）的执行状态。
# empty 与 failed 不得归并：「项目内没有相关记忆」和「记忆存储不可用」是两件事。
CapabilityStatus = Literal["success", "empty", "skipped", "failed"]

VALID_CAPABILITY_STATUSES: tuple[str, ...] = ("success", "empty", "skipped", "failed")


class ProjectResearchTurnContext(TypedDict):
    """项目科研回合的服务端只读输入。

    授权范围由服务端解析，不接受客户端提交的 paper_ids 直接作为授权依据。
    项目没有论文时 authorized_paper_ids 为空是合法状态，不在入口层提前拒绝。
    """

    user_id: str
    project_id: str
    # 经归属校验后本回合可用的论文全集。
    authorized_paper_ids: list[str]
    # 本回合用户实际聚焦的论文子集，为空表示未指定具体焦点。
    turn_focus_paper_ids: list[str]
    session_id: str
    conversation_id: str
    query: str
    history_summary: str
    # 仅当本回合挂在某个研究问题下时携带。
    research_question_id: NotRequired[str]


class TurnDepthAssessment(TypedDict, total=False):
    """Planner 首次分析产出的处理深度判断、本回合记忆读取判断与澄清判断，不含检索细节。

    确定性硬约束优先于模型偏好：明确的论文事实问题不能判 minimal；空论文范围
    不能要求论文 Evidence；明确的多论文比较不能降为无证据直答。越界时由调用方
    返回可观测的决策失败，不静默退回默认深度。
    """

    depth: TurnDepth
    # 选择该深度的稳定原因码，进 trace 供深度判断准确率评测取数。
    reason_code: str
    # 该深度下回答是否必须有已验证的论文 Evidence 支撑。
    requires_paper_evidence: bool
    # 归一化后的本回合检索问题，焦点解析结果已并入。
    resolved_query: str
    # 本轮要不要翻项目研究记忆，由模型在同一次深度评估里判断，三档深度都可能为真。
    needs_project_memory: bool
    # 读取收窄条件，键与记忆层收窄契约一致（paper_id / question_id / memory_type /
    # follow_links）。只放模型真给了值的键：空串、None、False 一律不入表，
    # 否则记忆层会把「模型没指定」当成「指定为空」，按空值过滤直接读空。
    # 这里用 dict[str, Any] 而不是具体 TypedDict，是因为记忆层契约住在
    # research_memory_contracts，而那个模块 import 本模块，反向 import 会成环；
    # 松的只是这段中间通道，两端仍是强类型：上游 pydantic 已校验过取值，
    # 下游记忆服务按自己的 TypedDict 收键。
    memory_narrowing: dict[str, Any]
    # 判断要不要翻记忆的稳定原因码，与深度 reason_code 分开记，两个判断各自可评测。
    memory_read_reason_code: str
    # 本轮是否必须先向用户澄清目标。目标无法确认且不同理解会导致明显不同动作时才为真，
    # 判得准就不许问 —— 澄清是 Planner 的一个动作，不是拿不准时的安全默认值。
    needs_clarification: bool
    # 要问用户的那一句话，澄清回合直接作为本轮回答返回。
    clarification_question: str
    # 为什么非问不可的稳定原因码，与深度、记忆两个原因码分开记，各自可追查。
    clarification_reason_code: str


class DepthTransition(TypedDict, total=False):
    """深度变更记录。深度是循环内的可变策略参数，每次变更追加一条。

    升级与降级都要记录（例如 research 执行后发现单篇即可回答），
    这是「深度升级召回率」的数据来源。
    """

    from_depth: TurnDepth
    to_depth: TurnDepth
    # 触发变更的那次观察结论，说明为什么在这一步改深度。
    trigger_observation: str
    reason_code: str
    # 发生变更时的循环步序号。
    loop_step: int


class ResearchTurnOutcome(TypedDict, total=False):
    """回合终态。统一终结节点产出，同时进 trace 与 API/SSE done payload。

    撞上步数或时长上限时 status 为 incomplete，并交出已取得证据与剩余缺口，
    不表现为正常完成。只答了一部分（有已查清的子问题、也有没查清的）同属这一档：
    部分交付必须能被读出来，不能混进 completed，否则「只答了一部分」与全答对
    在统计与评测里无从区分。
    """

    status: Literal["completed", "incomplete", "failed"]
    # 最终生效的处理深度，与 depth_transitions 的末条 to_depth 一致。
    final_depth: TurnDepth
    depth_reason_code: str
    depth_transitions: list[DepthTransition]
    # 本回合调用过的能力及其四态结果，能力名 → 状态。
    capability_results: dict[str, CapabilityStatus]
    # 本回合命中的项目记忆条目标识。
    memory_item_ids: list[str]
    # 本回合最终被引用的 Evidence 标识。
    evidence_ids: list[str]
    missing_evidence: list[MissingEvidence]
    # 循环实际步数，供进度展示与留痕追溯；不是上界、也不参与继续/停止判断。
    loop_steps: int
    termination_reason: str
    error_kind: str


class MemoryWritebackResult(TypedDict, total=False):
    """记忆候选写回状态；pending 表示已提交答案正在处理候选，终态保持原四态语义。

    skipped = 本回合深度没触发写回，压根没评估过有没有值得记的判断；
    empty = 评估过了，但本回合没有够格形成记忆的判断，并把原因记进 reason_codes。
    这两态不得归并：一个说明写回没跑，一个说明跑过且结论是「不该记」，
    归并后就分不清「功能没接上」和「本轮确实没东西可记」。
    同理 empty 与 failed 也不得归并，只有 failed 才代表存储侧出问题可重试。
    候选列表为空绝不许报 success：空列表冒充成功会让写回失效永远看不出来。
    """

    status: CapabilityStatus | Literal["pending"]
    processing_source: Literal["delivery_run", "http_retry"]
    processing_run_id: str | None
    # 本回合归属该次回答的候选记忆标识，含命中幂等后复用的已有候选。
    candidate_ids: list[str]
    # 得出该终态的稳定原因码，进 trace 供写回触发准确率评测取数。
    reason_codes: list[str]
    # 仅 failed 时有意义：该失败是否值得重试。
    retryable: bool
    error_kind: str


def default_evidence_layer_roles() -> list[EvidenceLayerRole]:
    """返回当前内部研究主链的证据层角色合同。"""
    return [
        {
            "layer": "milvus",
            "role": "primary_evidence",
            "output": "parent_chunk_evidence",
            "notes": "负责项目范围内的原文证据召回与父块全文回读。",
        },
        {
            "layer": "lightrag",
            "role": "global_hint",
            "output": "graph_hint_text",
            "notes": "负责主题脉络、关系线索和可能相关论文提示，不直接替代原文证据。",
        },
        {
            "layer": "neo4j",
            "role": "structural_explanation",
            "output": "evidence_paths",
            "notes": "负责研究对象关系、解释路径和结构化比较，不承担主检索。",
        },
    ]
