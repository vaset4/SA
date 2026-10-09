"""项目研究记忆层的正式合同模型与受控枚举。

一条研究记忆 = 用户在某个研究项目里逐渐形成的一条「判断」，
它带着四个正交的状态轴（生命周期 status / 新鲜度 stale_state /
是否被用户接纳 review_state / 认知成熟度 maturity），
以及证据链（回指论文原句）、记忆互联（based_on/conflicts_with/supersedes）、
全量版本快照。

枚举集中在本模块，service 层做校验、不下推 DB CHECK；
新增枚举值只改这一处（对齐无硬编码约定）。
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from app.models.research_contracts import CapabilityStatus


class MemoryVersionConflict(ValueError):
    """记忆或研究问题已有新版本，迟到编辑必须刷新后重试。"""


# ─── 受控枚举（service 层校验，不下推 DB CHECK） ───

# 记忆类型：从纯笔记到推断综合，覆盖研究过程里的各类判断
MemoryType = Literal[
    "research_purpose",
    "paper_note",
    "method_note",
    "dataset_note",
    "metric_note",
    "comparison_note",
    "evidence_clip",
    "question_note",
    "gap_note",
    "decision_note",
    "synthesis_note",
]
VALID_MEMORY_TYPES: tuple[str, ...] = (
    "research_purpose",
    "paper_note",
    "method_note",
    "dataset_note",
    "metric_note",
    "comparison_note",
    "evidence_clip",
    "question_note",
    "gap_note",
    "decision_note",
    "synthesis_note",
)

# 生命周期状态：草稿→已确认→归档；编辑可回到草稿
MemoryStatus = Literal["draft", "confirmed", "archived"]
VALID_MEMORY_STATUSES: tuple[str, ...] = ("draft", "confirmed", "archived")

# 新鲜度：证据是否仍然可信；needs_rebind 表示原证据失效待重绑
StaleState = Literal["fresh", "stale", "needs_rebind"]
VALID_STALE_STATES: tuple[str, ...] = ("fresh", "stale", "needs_rebind")

# 是否被用户接纳：AI 草稿落 pending_review，用户 accept 后才进正式库
ReviewState = Literal["pending_review", "accepted", "dismissed"]
VALID_REVIEW_STATES: tuple[str, ...] = ("pending_review", "accepted", "dismissed")

# 认知成熟度：直觉→倾向→已确立，与上面三轴正交，是台账主轴
Maturity = Literal["intuition", "leaning", "established"]
VALID_MATURITIES: tuple[str, ...] = ("intuition", "leaning", "established")

# 证据模式：有文献背书 / 纯用户笔记 / 混合
EvidenceMode = Literal["evidence_backed", "user_note_only", "mixed"]
VALID_EVIDENCE_MODES: tuple[str, ...] = ("evidence_backed", "user_note_only", "mixed")

# 证据角色：正/辅/反证据、上下文、推断依据
RefRole = Literal[
    "primary_support",
    "secondary_support",
    "counter_evidence",
    "context_only",
    "derived_basis",
]
VALID_REF_ROLES: tuple[str, ...] = (
    "primary_support",
    "secondary_support",
    "counter_evidence",
    "context_only",
    "derived_basis",
)

# 来源：人工/从回答保存/从chunk查看器保存/AI草稿/系统草稿
SourceKind = Literal[
    "manual",
    "save_from_answer",
    "save_from_chunk_inspector",
    "agent_draft",
    "system_draft",
]
VALID_SOURCE_KINDS: tuple[str, ...] = (
    "manual",
    "save_from_answer",
    "save_from_chunk_inspector",
    "agent_draft",
    "system_draft",
)

# 用户亲手落笔的来源：手写、从回答保存、从 chunk 查看器保存。
# 这三种来源的判断新建即生效（status=confirmed），draft 那个人审位是给 AI 草稿留的。
# 用白名单而非「AI 来源黑名单」：将来新增来源若漏改这里，默认落 draft 多一道人审，
# 而不是默认 confirmed 绕过人审。
USER_AUTHORED_SOURCE_KINDS: tuple[str, ...] = (
    "manual",
    "save_from_answer",
    "save_from_chunk_inspector",
)

# 记忆互联类型：基于/冲突(双向)/修正(旧条目保留)
LinkType = Literal["based_on", "conflicts_with", "supersedes"]
VALID_LINK_TYPES: tuple[str, ...] = ("based_on", "conflicts_with", "supersedes")

# 变更类型：驱动版本快照的动作；含 B2 新增的 change_maturity
ChangeKind = Literal[
    "create",
    "edit_body",
    "attach_evidence",
    "detach_evidence",
    "mark_stale",
    "rebind_evidence",
    "confirm",
    "archive",
    "attach_question",
    "detach_question",
    "add_link",
    "remove_link",
    "accept_draft",
    "dismiss_draft",
    "change_maturity",
]
VALID_CHANGE_KINDS: tuple[str, ...] = (
    "create",
    "edit_body",
    "attach_evidence",
    "detach_evidence",
    "mark_stale",
    "rebind_evidence",
    "confirm",
    "archive",
    "attach_question",
    "detach_question",
    "add_link",
    "remove_link",
    "accept_draft",
    "dismiss_draft",
    "change_maturity",
)

# AI 草稿置信度信号里 quote 匹配强度的取值
QuoteMatchStrength = Literal["exact", "fuzzy", "none"]
VALID_QUOTE_MATCH_STRENGTHS: tuple[str, ...] = ("exact", "fuzzy", "none")


# ─── 各枚举默认值（建表默认与 service 兜底共用一处） ───

DEFAULT_MEMORY_STATUS = "draft"
DEFAULT_STALE_STATE = "fresh"
DEFAULT_REVIEW_STATE = "accepted"
DEFAULT_MATURITY = "intuition"
DEFAULT_EVIDENCE_MODE = "user_note_only"
DEFAULT_REF_ROLE = "primary_support"
DEFAULT_SOURCE_KIND = "manual"


# ─── 合同模型（TypedDict） ───


class ResearchQuestion(TypedDict, total=False):
    """研究问题：作数据归属 + 研究目的置顶；前端只作过滤标签，不作分组主轴。"""

    question_id: str
    project_id: str
    user_id: str
    title: str
    body: str
    is_purpose: bool
    position: int
    source_kind: SourceKind
    review_state: ReviewState
    created_at: str
    updated_at: str


class MemoryEvidenceRef(TypedDict, total=False):
    """记忆的一条证据引用：回指某篇论文的具体 chunk 与原句。"""

    ref_id: str
    memory_id: str
    paper_id: str
    chunk_id: str
    parent_id: str
    evidence_id: str
    section: str
    section_path: str
    page_num: int
    quote_text: str
    quote_hash: str
    excerpt: str
    claim_key: str
    ref_role: RefRole
    source_ref: dict[str, Any]
    document_ref: dict[str, Any]
    support: dict[str, Any]
    verification: dict[str, Any]
    created_at: str


class MemoryLink(TypedDict, total=False):
    """记忆与记忆之间的有向关系。"""

    link_id: str
    project_id: str
    src_memory_id: str
    dst_memory_id: str
    link_type: LinkType
    note: str
    created_at: str


class ConfidenceSignals(TypedDict, total=False):
    """AI 草稿置信度的构成信号，用于待审区排序与低置信度高亮。"""

    evidence_count: int
    quote_match_strength: QuoteMatchStrength
    is_synthesis: bool


class ResearchMemoryItem(TypedDict, total=False):
    """一条研究记忆条目：用户的一条判断 + 四轴状态 + 证据模式 + 来源溯源。"""

    memory_id: str
    project_id: str
    user_id: str
    memory_type: MemoryType
    title: str
    body: str
    status: MemoryStatus
    stale_state: StaleState
    review_state: ReviewState
    maturity: Maturity
    stale_reasons: list[str]
    evidence_mode: EvidenceMode
    related_paper_ids: list[str]
    tags: list[str]
    source_kind: SourceKind
    source_session_id: str
    source_conversation_id: str
    source_answer_id: str
    confidence_score: float
    confidence_signals: ConfidenceSignals
    possible_duplicate: bool
    version: int
    created_at: str
    updated_at: str
    confirmed_at: str
    archived_at: str


class MemoryVersion(TypedDict, total=False):
    """记忆的一个版本快照，含条目本体 + 证据 + 归属问题 + 关系。"""

    version_id: str
    memory_id: str
    project_id: str
    version: int
    snapshot: dict[str, Any]
    change_kind: ChangeKind
    change_summary: str
    created_by: str
    created_at: str


class ResearchMemoryDetail(TypedDict, total=False):
    """记忆详情聚合：条目 + 证据 + 关系 + 归属问题 + 版本摘要。"""

    item: ResearchMemoryItem
    evidence_refs: list[MemoryEvidenceRef]
    links: list[MemoryLink]
    question_ids: list[str]
    versions: list[MemoryVersion]


# ─── 记忆读取结果合同 ───


class MemoryReadItem(TypedDict, total=False):
    """一次记忆读取里返回的单条记忆，字段裁剪到注入提示词所需的最小集。"""

    memory_id: str
    version: int
    title: str
    # 正文摘要（已截断），不是全文
    body_summary: str
    memory_type: MemoryType
    maturity: Maturity
    source_kind: SourceKind
    # 归属研究问题。注入侧不查这一维，逐条回表查归属会退化成 N+1，
    # 所以这里缺键是正常状态、不是漏填；只有按问题收窄的读取动作才会填上。
    question_ids: list[str]
    # 渲染给 LLM 的身份标签，由 injector 按 maturity 派生：
    # established→已确立，leaning→倾向，intuition→假设。
    # intuition 取「假设」而不是中性的「直觉」，为的是让 LLM 看清这条判断还没被证据坐实。
    provenance_label: str
    # 四道资格门在这条记忆上的取值：归属人 + 三个状态轴。门本身写在 SQL 里，
    # 这四个键是「这条为什么被放行」的凭据，供合规核查与评测逐条复核，不参与渲染
    user_id: str
    review_state: ReviewState
    status: MemoryStatus
    stale_state: StaleState


class MemoryReadResult(TypedDict):
    """一次记忆读取动作的终态：读到什么、排除了什么、失败在哪。"""

    # 复用能力级状态，empty 与 failed 不得归并：
    # 「项目内没有符合条件的记忆」与「记忆存储不可用」是两件事，
    # 读取失败时禁止返回空列表冒充成功，调用方要能分辨该不该重试、该不该提示用户。
    status: CapabilityStatus
    items: list[MemoryReadItem]
    # 本次读取中实际测到的排除计数，键是排除原因。
    # 只允许放真正统计过的维度：四道资格门写在 SQL WHERE 里，
    # 被门挡掉的行根本不会回到应用层，因此不许编造 gate_* 之类的假计数。
    excluded_counts: dict[str, int]
    # 仅 status=failed 时出现，承载可观测的失败类型
    error_kind: NotRequired[str]


class MemoryReadNarrowing(TypedDict, total=False):
    """Planner 发起一次记忆读取时能给的收窄条件，这四个键就是全集。

    键位本身就是「只能收窄不能放宽」的结构保证。
    review_state / status / stale_state / user_id / project_id 五项是资格门，
    门写死在读取服务的查询函数体内：不合格的记忆对 Planner 是「不存在」而不是
    「低分」，所以这几个键故意不出现在这里 —— Planner 既给不了也改不了，
    想放宽连参数都没有。
    maturity 同样故意不给：三档成熟度都有资格进上下文，成熟度只决定身份标签与
    预算截断次序，把它做成过滤条件等于让 Planner 悄悄丢掉整档直觉级判断。
    想在这四个键之外再加维度前，先确认那个维度不是资格门。
    """

    # 只看与某篇论文相关的判断，走 related_paper_ids 的 JSONB 包含过滤
    paper_id: str
    # 只看挂在某个研究问题下的判断，走 memory_question_links 关联
    question_id: str
    # 只看某一类判断，例如只要方法笔记或只要空白点
    memory_type: MemoryType
    # 顺着记忆互联（based_on / conflicts_with / supersedes）再扩一跳；
    # 扩进来的邻居同样要自己过四道门，不会因为「与种子有关系」就获得豁免
    follow_links: bool
