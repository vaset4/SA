"""
配置管理模块

使用 Pydantic Settings 管理应用配置。

这里统一收拢当前系统的几类运行参数：
  - LLM / 搜索 / deep research 旧兼容配置
  - RAG 主链配置（Embedding、Rerank、Query Rewrite、Self-RAG、压缩）
  - 基础设施配置（PostgreSQL、Milvus、LightRAG、Redis）
  - API 与上传相关配置

需要注意：
  - 当前默认业务库后端是 PostgreSQL
  - 当前默认向量后端是 Milvus，`faiss` 主要作为兼容/离线选项保留
  - `graph_rag_enabled` 控制的是 LightRAG 图检索增强层
"""

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from pydantic_settings import BaseSettings
from pydantic import Field
# 1. 【配置】【文件边界】只读取当前副本的配置，禁止向上查找其它仓库的私有文件
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = "resources/tokenizers/bge-m3/tokenizer.json"
DEFAULT_TOKENIZER_SHA256 = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"
_PROJECT_ROOT = _BACKEND_ROOT.parent if _BACKEND_ROOT.name == "backend" else _BACKEND_ROOT
_ENV_FILES = tuple(dict.fromkeys((_BACKEND_ROOT / ".env", _PROJECT_ROOT / ".env")))
_LLM_PROFILE_FIELDS = frozenset({
    "llm_model_name", "llm_api_key", "llm_base_url", "llm_temperature", "llm_timeout",
    "llm_requests_per_minute_limit", "llm_http_max_retries", "llm_min_request_interval_seconds",
    "llm_retry_backoff_seconds",
})


class Settings(BaseSettings):
    """
    应用配置类

    使用 Pydantic Settings 进行配置管理，支持：
    1. 从环境变量读取
    2. 类型验证
    3. 默认值设置

    历史上部分字段按“Phase X”分组命名，这些标签主要是演进痕迹，
    当前应按真实能力域来理解，而不是把它们视为功能是否仍停留在某个阶段。
    """

    # ============ LLM 配置 ============
    llm_model_name: str = Field(
        default="gemini-3-flash-preview",
        description="LLM模型名称"
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API密钥"
    )
    llm_base_url: str = Field(
        default="",
        description="LLM API端点URL"
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM温度参数（0-2）"
    )
    llm_timeout: int = Field(
        default=360,
        ge=5,
        le=600,
        description="LLM 请求超时（秒）"
    )
    llm_streaming_enabled: bool = Field(
        default=True,
        description="LLM 请求走 SSE 流式（客户端聚合成完整结果）；中转站对非流式请求排慢车道，关闭即回退旧的非流式行为",
    )
    structured_output_native_default: bool = Field(
        default=True,
        description="是否向模型接口发送原生 JSON Schema；关闭后仍保留 JSON 提示词和严格校验",
    )
    llm_connection_test_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="模型服务连通性探活请求超时（秒）；只发 max_tokens=1 的 ping，慢速免费上游（如 NVIDIA 网关）握手偏慢时可调大",
    )
    paper_analysis_llm_timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=600,
        description="论文结构化分析单次 LLM 请求超时（秒），默认低于 Cloudflare 524 常见 200s 边界",
    )
    paper_analysis_llm_max_retries: int = Field(
        default=0,
        ge=0,
        le=5,
        description="论文结构化分析 LLM 请求失败后的重试次数，默认 0 避免重复提交长耗时请求",
    )
    llm_min_request_interval_seconds: float = Field(
        default=2,
        ge=0.0,
        le=60.0,
        description="同一上游 LLM 主机两次请求之间的最小间隔（秒）；NVIDIA 免费上游实测 RPM=30，2s 间隔即贴合额度上限，过大会浪费额度",
    )
    llm_requests_per_minute_limit: int = Field(
        default=30,
        ge=0,
        le=600,
        description="同一上游 LLM 主机每分钟请求上限；实际生效为 max(显式最小间隔, 60/rpm)，NVIDIA 免费上游 RPM=30；0 表示只使用显式最小间隔",
    )
    llm_queue_enabled: bool = Field(default=False, description="启用跨进程供应商共享额度队列")
    llm_queue_database_url: str = Field(default="", description="显式共享协调库；不回落到业务或评测库")
    llm_queue_group: str = Field(default="", description="同凭证共享额度的非敏感组标识")
    llm_queue_requests_per_window: int = Field(default=5, ge=1, description="同组滚动窗口请求上限")
    llm_queue_window_seconds: float = Field(default=60.0, gt=0, description="同组滚动窗口秒数")
    llm_queue_min_interval_seconds: float = Field(default=12.0, ge=0, description="共享队列最小发送间隔秒数")
    llm_queue_cooldown_seconds: float = Field(default=60.0, gt=0, description="429缺少合法等待头时的明确冷却秒数")
    llm_retry_delay_cap_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="非429重试等待上限（秒）；429遵守共享冷却，不受此值截断",
    )
    llm_http_max_retries: int = Field(
        default=8,
        ge=0,
        le=15,
        description="主问答/RAG 链 LLM 请求的 HTTP/连接错误重试次数；NVIDIA 免费上游实测 503 高发但错误 ~1.6s 快速返回、重试 7 次内必成，故取 8 配合短退避",
    )
    llm_retry_backoff_seconds: float = Field(
        default=1.5,
        ge=0.0,
        le=60.0,
        description="HTTP/连接错误的固定重试等待秒数；不稳定但快速失败的上游取短值(如1.5)，低 RPM 上游取长值；激活预设时可按供应商热切换",
    )
    ragas_llm_min_request_interval_seconds: float = Field(
        default=5,
        ge=0.0,
        le=120.0,
        description="ragas 评估专用的最小 LLM 请求间隔（秒），默认比主运行时更保守",
    )
    ragas_max_workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="ragas 评估阶段的最大并发 worker 数，默认单线程避免触发频率限制",
    )
    graph_timeout_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=600.0,
        description="普通 graph 总执行时限（秒）；0 表示无总时限，仍保留取消和单次网络请求超时"
    )
    deep_research_graph_timeout_seconds: float = Field(
        default=600.0,
        ge=0.0,
        le=1200.0,
        description="DeepResearch 总执行时限（秒）；0 表示无总时限，仍保留取消和单次网络请求超时"
    )

    # ============ 搜索配置 ============
    tavily_api_key: str = Field(
        default="",
        description="Tavily API密钥"
    )
    search_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="每次搜索的最大结果数（1-10）"
    )
    search_content_preview_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="搜索内容预览长度（50-500字符）"
    )
    deep_research_allow_clarification: bool = Field(
        default=True,
        description="deep research 是否在执行前先做澄清判断"
    )
    deep_research_max_parallel_units: int = Field(
        default=2,
        ge=1,
        le=5,
        description="deep research 并发 research unit 上限"
    )
    deep_research_max_researcher_iterations: int = Field(
        default=2,
        ge=1,
        le=8,
        description="deep research supervisor 最大委派轮次"
    )
    deep_research_max_react_tool_calls: int = Field(
        default=1,
        ge=1,
        le=6,
        description="deep research 单个 researcher 最大搜索轮次"
    )
    deep_research_max_search_queries_per_round: int = Field(
        default=1,
        ge=1,
        le=5,
        description="deep research 单个 researcher 每轮最大搜索 query 数"
    )
    deep_research_raw_note_max_chars: int = Field(
        default=2400,
        ge=500,
        le=8000,
        description="deep research 单个 raw note 最大保留字符数"
    )
    deep_research_final_report_max_note_chars: int = Field(
        default=1200,
        ge=200,
        le=5000,
        description="deep research final report 阶段单条 note 最大保留字符数"
    )
    deep_research_final_report_max_findings_chars: int = Field(
        default=6000,
        ge=1000,
        le=20000,
        description="deep research final report 阶段 findings 总字符预算"
    )
    deep_research_final_report_raw_note_preview_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="deep research final report fallback 中保留的 raw note 预览条数"
    )
    deep_research_final_report_raw_note_max_chars: int = Field(
        default=700,
        ge=100,
        le=4000,
        description="deep research final report fallback 中单条 raw note 预览最大字符数"
    )

    # ============ 任务配置 ============
    planner_task_count: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Planner生成的任务数量（2-5）"
    )
    executor_max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Executor并发执行的最大任务数（1-10）"
    )
    executor_show_search_details: bool = Field(
        default=True,
        description="是否显示搜索详情（URL和内容摘要）"
    )

    # ============ ReAct配置 ============
    react_enabled: bool = Field(
        default=True,
        description="是否启用ReAct循环"
    )
    react_max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="ReAct最大迭代次数（1-5）"
    )

    # ============ 证据合同 Feature Flag ============
    rag_intent_llm_fallback: bool = Field(
        default=True,
        description="意图识别 LLM 兜底总开关，关闭时只用规则版"
    )
    rag_expose_unified_contract: bool = Field(
        default=True,
        description="API 响应体暴露 evidence_records / answer_brief / missing_evidence 三字段；关闭时退回旧响应"
    )
    rag_research_planner_max_units: int = Field(
        default=8,
        ge=0,
        le=20,
        description="Research Planner 每批动作或单次计划的单元数上限；0 表示不限制数量"
    )
    rag_planner_lightrag_locator_enabled: bool = Field(
        default=False,
        description="EXPERIMENTAL: 是否允许 planner 用 LightRAG 仅扩展 locator queries；默认关闭，且不得生成证据记录"
    )
    rag_planner_loop_total_timeout_seconds: float = Field(
        default=180.0,
        ge=0,
        le=600,
        description="规划循环整图兜底总超时（秒）；replan×多轮检索最坏耗时的显式护栏，超时返回带 error_code 的显式失败载荷让失败清晰暴露、不静默；设 0=禁用兜底"
    )
    rag_planner_graph_recursion_limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="规划子图自身的递归上限（LangGraph recursion_limit），基础设施护栏不是研究额度：取消研究计数上界后回边可以合法地多走几步，取足余量；超过时留下可读失败原因而不是框架错误"
    )

    # ============ 报告配置 ============
    report_min_length: int = Field(
        default=500,
        ge=200,
        description="报告最小字数"
    )
    report_target_length: int = Field(
        default=2000,
        ge=500,
        description="报告目标字数"
    )
    report_style: Literal["academic", "business", "casual"] = Field(
        default="academic",
        description="报告风格：academic（学术）、business（商务）、casual（轻松）"
    )

    # ============ arXiv 配置 ============
    arxiv_max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="arXiv 搜索最大返回数（1-100）"
    )
    arxiv_default_categories: list = Field(
        default=["cs.AI", "cs.LG"],
        description="arXiv 默认分类过滤列表"
    )
    arxiv_request_interval: float = Field(
        default=3.0,
        ge=1.0,
        description="arXiv API 请求间隔（秒），官方建议不少于3秒"
    )

    # ============ RAG：Embedding 配置 ============
    hf_home: str = Field(default="", description="本地 HuggingFace 缓存主目录")
    huggingface_hub_cache: str = Field(default="", description="本地 HuggingFace Hub 缓存目录")
    embedding_mode: Literal["api", "local"] = Field(default="api")
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str = Field(
        default="cpu",
        description="本地 embedding 运行设备，例如 cpu / cuda"
    )
    embedding_api_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="Embedding API 基地址，客户端追加 /embeddings"
    )
    embedding_api_key: str = Field(default="", description="独立的 Embedding API 密钥", repr=False)
    embedding_api_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        description="Embedding API 请求超时（秒）"
    )
    embedding_dimension: int = Field(
        default=1024,
        ge=1,
        description="Embedding 向量维度"
    )
    lightrag_embedding_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="LightRAG 专用 embedding batch size，默认比主向量链更保守",
    )

    # ============ RAG：Rerank 配置 ============
    rerank_mode: Literal["api", "local"] = Field(default="api")
    rerank_enabled: bool = Field(default=True)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        description="Reranker 输出 top_k"
    )
    vector_rerank_enabled: bool | None = Field(
        default=None,
        description="应用层向量 rerank 开关；未设置时回落到 legacy rerank_enabled",
    )
    vector_rerank_model: str | None = Field(
        default=None,
        description="应用层向量 rerank 模型；未设置时回落到 legacy rerank_model",
    )
    vector_rerank_top_k: int | None = Field(
        default=None,
        ge=1,
        description="应用层向量 rerank 输出 top_k；未设置时回落到 legacy rerank_top_k",
    )

    # ============ 解析链：Parent-Child 分块配置（单位：token） ============
    chunk_tokenizer_path: str = Field(default=DEFAULT_TOKENIZER_PATH, min_length=1,
        description="API 模式独立分词文件，相对路径从 backend 目录解析")
    chunk_tokenizer_sha256: str = Field(default=DEFAULT_TOKENIZER_SHA256, pattern=r"^[0-9a-f]{64}$",
        description="API 分词文件的实际 SHA-256，更换后需重启并显式重新处理资料")
    chunk_merge_floor: int = Field(
        default=150,
        ge=1,
        description="结构保真合并只吸收低于该 token 数的同结构碎片",
    )
    parent_chunk_max_size: int = Field(
        default=1500,
        ge=1,
        description="父块最大 token 数；超限后仅按结构/段落/句子边界切分，单句或原子块超限会失败",
    )
    structure_merge_enabled: bool = Field(
        default=True,
        description="启用按 section_path 结构边界合并父块",
    )
    parent_chunk_overlap_tokens: int = Field(
        default=150,
        ge=0,
        description="父块之间保留的重叠 token 数",
    )
    child_chunk_target_size: int = Field(
        default=320,
        ge=1,
        description="子块目标 token 数，用于控制检索粒度",
    )
    child_chunk_overlap_tokens: int = Field(
        default=40,
        ge=0,
        description="子块之间保留的重叠 token 数",
    )

    # ============ RAG：查询改写与质量控制配置 ============
    query_rewrite_enabled: bool = Field(
        default=True,
        description="是否启用查询改写与 Multi-Query"
    )
    query_filter_enabled: bool = Field(
        default=True,
        description="是否启用查询过滤条件自动解析（年份、分类、作者）"
    )
    multi_query_count: int = Field(
        default=2,
        ge=1,
        le=3,
        description="补充查询数量"
    )
    document_grading_enabled: bool = Field(
        default=True,
        description="是否启用检索文档相关性评分"
    )
    self_rag_enabled: bool = Field(
        default=True,
        description="已废弃，仅为兼容旧配置保留；运行时主链不再使用 Self-RAG"
    )
    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="已废弃，仅为兼容旧配置保留；运行时主链不再使用 Self-RAG"
    )
    self_rag_max_iterations: int = Field(
        default=2,
        ge=0,
        le=3,
        description="已废弃，仅为兼容旧配置保留；运行时主链不再使用 Self-RAG"
    )

    # ============ RAG：检索循环配置 ============
    rag_top_k: int = Field(default=5)
    rag_score_threshold: float = Field(default=0.75)
    rag_similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="向量相似度阈值（第一道门，粗筛）"
    )
    rag_rerank_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Rerank 分数阈值（第二道门，精筛）"
    )
    rag_max_iterations: int = Field(default=4)
    rag_graph_query_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=300.0,
        description="LightRAG 图检索单次查询超时（秒）"
    )
    rag_vector_query_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=300.0,
        description="Milvus 向量检索单次查询超时（秒）"
    )

    # ============ Phase 1（05-23-cross-paper-qa-retrieval-fix）：研究问答证据守门 ============
    # R1 — ENOUGH 守门（multi_paper + research 强制证据组合）
    rag_enough_guard_enabled: bool = Field(
        default=True,
        description="[DEPRECATED by 05-24] 多论文研究问答的 ENOUGH 运行时守门总开关；已被 sufficiency_grader 替代"
    )
    rag_enough_guard_min_vector_calls: int = Field(
        default=1,
        ge=0,
        le=5,
        description="[DEPRECATED by 05-24] 允许 ENOUGH 之前最少向量类工具调用次数；不再有计数逻辑"
    )
    rag_enough_guard_min_graph_calls: int = Field(
        default=1,
        ge=0,
        le=5,
        description="[DEPRECATED by 05-24] 允许 ENOUGH 之前最少图谱类工具调用次数；graph 已前置"
    )

    # ============ guardrails 安全底线拒绝文案 ============
    rag_guardrails_rejection_template: str = Field(
        default="这次输入无法处理，请换一个问题。若为空输入或内容过长，请调整后重试。",
        description="guardrails 安全底线拒绝（空输入/超长/敏感主题）时返回给用户的文案模板"
    )

    # ============ 05-24 R2 — graph prefetch + sufficiency grader ============
    rag_graph_prefetch_enabled: bool = Field(
        default=False,
        description="LEGACY DEBUG: 是否在 ReAct 循环前执行 LightRAG graph_prefetch；默认关闭，避免图谱提示污染首轮决策"
    )
    rag_sufficiency_grader_enabled: bool = Field(
        default=False,
        description="DEPRECATED: sufficiency_grader 已从 RAG 正常路径移除；保留该字段仅用于旧配置兼容"
    )
    rag_sufficiency_grader_model: str = Field(
        default="",
        description="sufficiency_grader 使用的模型；空字符串表示使用主 RAG 模型"
    )

    # ============ 05-24 R5 — 观测预算 ============
    rag_short_circuit_max_latency_budget_ms: int = Field(
        default=10000,
        ge=0,
        description="guardrails 短路场景端到端延迟预算，仅用于 AC 验证断言"
    )

    # R3 — 未索引论文提示模板
    rag_unindexed_paper_notice_template: str = Field(
        default="⚠ 以下论文尚未完成索引，本次未参与检索：{paper_titles_csv}",
        description="未索引论文提示模板，占位符 {paper_titles_csv}"
    )

    # R4 — Graph hint 段渲染模板
    rag_graph_hint_unavailable_template: str = Field(
        default="图谱线索本次不可用：{reason}",
        description="graph 调用失败时回答中插入的固定段，占位符 {reason}"
    )
    rag_graph_hint_skipped_template: str = Field(
        default="图谱线索本次未生成（已达检索步数上限）",
        description="守门放弃强制时回答中插入的固定段"
    )

    # R5 — Graph 证据引用质量门
    rag_graph_quality_gate_enabled: bool = Field(
        default=True,
        description="graph 证据引用质量门开关；关闭后调用成功即判 ok（旧行为）"
    )
    rag_graph_hint_ok_but_unused_template: str = Field(
        default="图谱线索本次调用成功，但未被回答正文引用（可能因证据与正文重叠或未命中关键路径）",
        description="graph 调用成功但 evidence 未被引用时插入的固定段"
    )
    rag_citation_regex_patterns: list[str] = Field(
        default=[
            r"\[chunk_([a-zA-Z0-9_]+)\]",
            r'<cite\s+chunk_id="([^"]+)"',
            r"paper://upload_[a-zA-Z0-9_]+#chunk_([a-zA-Z0-9_]+)",
        ],
        description="从 final_answer 提取引用 chunk_id 的正则集合；每条第 1 个 group 必须是 chunk_id"
    )

    # ============ Phase 1.5（05-23-rag-recall-coverage-guard-fix）：多论文召回深度与守门 ============
    rag_scoped_per_paper_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="多论文 scoped_paper_search 每篇论文最多贡献的 chunk 数（rerank 后）"
    )
    rag_scoped_total_chunk_cap: int = Field(
        default=30,
        ge=5,
        le=100,
        description="多论文 scoped_paper_search 单次工具调用总输出 chunk 数硬上限（防上下文爆炸）"
    )
    rag_scoped_per_paper_min_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="多论文 scoped_paper_search 每篇期望最低 chunk 数；不达标记入 source_status_reasons（soft min）"
    )
    rag_auto_expand_parent_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="研究单元声明父块取证粒度时，单单元最多回读父块全文的命中条数（超出者保持子块片段形态）"
    )
    rag_planner_section_text_max_chars: int = Field(
        default=6000,
        ge=0,
        le=40000,
        description="研究规划器按章节读取的字符预算；0为完整读取，正数预算保留未读范围以供续读"
    )
    rag_planner_chunk_metadata_max_rows: int = Field(
        default=50,
        ge=1,
        le=200,
        description="研究规划器按元数据条件筛块的单单元条数上限（超出取前 N 条并产截断概览）"
    )
    rag_planner_graph_relation_max_rows: int = Field(
        default=50,
        ge=1,
        le=200,
        description="研究规划器查研究对象图关系三元组的单单元条数上限（超出取前 N 条并产截断概览）"
    )
    rag_planner_consumer_text_max_chars: int = Field(
        default=8000,
        ge=0,
        le=40000,
        description="研究规划器所选材料的送模字符预算；0为完整提供，正数预算记录未提供范围"
    )

    # ============ Phase 4：认证与限流配置 ============
    app_env: Literal["local", "staging", "prod"] = Field(
        default="local",
        description="部署环境，用于认证保护规则"
    )
    auth_enabled: bool = Field(
        default=True,
        description="是否启用 API 认证（开发环境可关闭）"
    )
    developer_usernames: str = Field(
        default="",
        description="逗号分隔的 developer 用户名；命中后可访问开发者观测页"
    )
    admin_usernames: str = Field(
        default="",
        description="逗号分隔的 admin 用户名；admin 拥有 developer 观测权限"
    )
    jwt_secret_key: str = Field(
        default="change-me-in-production",
        description="JWT 签名密钥"
    )
    jwt_expire_hours: int = Field(
        default=24,
        ge=1,
        description="JWT token 过期时间（小时）"
    )
    jwt_refresh_expire_days: int = Field(
        default=7,
        ge=1,
        description="JWT refresh token 过期时间（天）"
    )
    api_key: str = Field(
        default="",
        description="静态 API Key"
    )
    rate_limit: str = Field(
        default="60/minute",
        description="默认速率限制"
    )

    # ============ 基础设施：缓存配置 ============
    cache_enabled: bool = Field(
        default=True,
        description="是否启用 L1 内存缓存"
    )
    redis_cache_enabled: bool = Field(
        default=False,
        description="是否启用 Redis 缓存探针/二级缓存"
    )
    cache_embedding_ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Embedding 缓存 TTL（秒）"
    )
    cache_search_ttl_seconds: int = Field(
        default=120,
        ge=1,
        description="检索结果缓存 TTL（秒）"
    )
    cache_llm_ttl_seconds: int = Field(
        default=1800,
        ge=1,
        description="LLM 响应缓存 TTL（秒）"
    )
    cache_embedding_maxsize: int = Field(
        default=1000,
        ge=1,
        description="Embedding 缓存最大条目数"
    )
    cache_search_maxsize: int = Field(
        default=200,
        ge=1,
        description="检索缓存最大条目数"
    )
    cache_llm_maxsize: int = Field(
        default=100,
        ge=1,
        description="LLM 缓存最大条目数"
    )

    rerank_max_candidates: int = Field(
        default=20,
        ge=1,
        description="Rerank 前召回放大量"
    )
    rerank_api_key: str = Field(
        default="",
        description="远程 Reranker API Key",
        repr=False,
    )
    rerank_endpoint: str = Field(
        default="https://api.siliconflow.cn/v1/rerank",
        description="Reranker API 完整接口地址"
    )
    rerank_api_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        description="Reranker API 请求超时（秒）"
    )

    tracing_enabled: bool = Field(
        default=True,
        description="是否启用 RAG 追踪日志"
    )
    obs_trace_max_bytes: int = Field(default=1024 * 1024)
    obs_call_input_max: int = Field(default=128 * 1024)
    obs_call_output_max: int = Field(default=64 * 1024)

    # ============ RAG：上下文压缩配置 ============
    compress_base_threshold: int = Field(default=6000)
    compress_growth_factor: float = Field(default=0.3)

    # ============ 推荐：推送配置 ============
    push_score_threshold: float = Field(default=0.6)
    push_mode: str = Field(default="terminal")

    # ============ 引用链配置 ============
    citation_cache_enabled: bool = Field(default=True)
    citation_max_depth: int = Field(default=1)

    # ============ 对话配置 ============
    chat_max_history_tokens: int = Field(default=8192)

    # ============ 文件与上传配置 ============
    data_dir: str = Field(default="./data")
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        description="最大上传文件大小（MB）"
    )
    mineru_enabled: bool = Field(
        default=True,
        description="PDF 上传是否优先走 MinerU 在线解析"
    )
    mineru_api_base_url: str = Field(
        default="https://mineru.net/api/v4",
        description="MinerU 官方 API 基地址"
    )
    mineru_api_token: str = Field(
        default="",
        description="MinerU API token，必须通过环境变量注入"
    )
    mineru_poll_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="MinerU 任务轮询间隔（秒）"
    )
    mineru_timeout_seconds: int = Field(
        default=600,
        ge=30,
        le=1800,
        description="MinerU 单个 PDF 解析超时（秒）"
    )
    mineru_result_download_timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="MinerU 结果包下载超时（秒）"
    )
    # ============ 基础设施：业务数据库配置 ============
    db_backend: Literal["sqlite", "postgresql"] = Field(
        default="postgresql",
        description="业务数据库后端；当前默认与推荐为 postgresql"
    )
    database_url: str = Field(
        default="",
        description="PostgreSQL 业务库连接串，必须通过环境变量显式注入"
    )
    database_pool_min_size: int = Field(default=2, ge=1)
    database_pool_max_size: int = Field(default=10, ge=1)
    database_command_timeout: int = Field(default=30, ge=1)

    # ============ 基础设施：检索后端配置 ============
    milvus_uri: str = Field(
        default="http://localhost:19530",
        description="Milvus 连接地址"
    )
    milvus_collection: str = Field(
        default="paper_chunks",
        description="Milvus 旧 collection 名称（仅供迁移脚本读取）"
    )
    milvus_collection_v2: str = Field(
        default="study_assistant_chunks_v2",
        description="Milvus v2 collection 名称（当前主链使用）"
    )
    research_memory_curator_model: str = Field(
        default="",
        description="研究记忆 AI 整理助手使用的模型名；空字符串表示复用主 LLM 模型"
    )
    research_memory_curate_max_papers: int = Field(
        default=5,
        ge=1,
        le=50,
        description="单次 curate 圈定的论文数上限，超出需分批，避免全库扫"
    )
    research_memory_curate_paper_char_budget: int = Field(
        default=24000,
        ge=2000,
        le=200000,
        description="单篇论文全文拼接进 curator prompt 的字符预算，超出截断并标注未覆盖"
    )
    research_memory_inject_max_items: int = Field(
        default=8,
        ge=1,
        le=50,
        description="A 面注入 planner 的记忆条数上限（B1）"
    )
    research_memory_inject_max_chars: int = Field(
        default=3600,
        ge=500,
        le=40000,
        description="A 面注入 planner 的记忆总字符预算，超出截断并标注（B1）"
    )
    planner_paper_context_enabled: bool = Field(
        default=True,
        description="是否在 planner 规划阶段注入论文元数据上下文"
    )
    planner_paper_context_max_papers: int = Field(
        default=10,
        ge=1,
        le=50,
        description="单轮 planner 查询论文元数据的数量上限"
    )
    planner_graph_context_enabled: bool = Field(
        default=True,
        description="是否在 planner 规划阶段注入研究对象图谱上下文"
    )
    planner_context_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="planner 上下文装载（论文库/图谱查询）的基础设施故障重试次数"
    )
    planner_context_retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="planner 上下文装载重试的固定等待秒数"
    )
    rag_planner_evidence_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=120.0,
        description="planner 单次取证调用的时间上限（秒），到点按基础设施失败处理"
    )
    rag_planner_evidence_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="planner 取证调用遇基础设施故障后的额外重试次数，总尝试数 = 次数 + 1"
    )
    rag_planner_evidence_retry_backoff_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="planner 取证调用重试前的固定等待秒数"
    )
    rag_planner_max_parallel_units: int = Field(
        default=4,
        ge=0,
        le=8,
        description="planner 同层无依赖单元的最大并行数；0 表示不额外限并发，依赖层仍按序执行"
    )
    vector_backend: Literal["faiss", "milvus"] = Field(
        default="milvus",
        description="向量存储后端；当前默认主链为 milvus，faiss 为兼容选项"
    )
    graph_rag_enabled: bool = Field(
        default=True,
        description="是否启用 LightRAG 图检索增强层"
    )
    graph_rerank_enabled: bool = Field(
        default=False,
        description="是否启用 LightRAG 内部 rerank；默认关闭，避免与应用层 vector_rerank 混淆",
    )
    graph_extraction_mode: Literal["llm", "regex"] = Field(
        default="llm",
        description="Neo4j 研究对象抽取模式：llm 为默认领域自适应路径，regex 仅作紧急回滚",
    )
    lightrag_use_domain_prompt: bool = Field(
        default=True,
        description="是否给 LightRAG 注入学术论文领域 prompt 与 entity_types",
    )
    graph_llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Neo4j LLM 抽取失败后的额外重试次数；超出后抛 ExtractionFailure",
    )
    neo4j_enabled: bool = Field(
        default=True,
        description="是否启用 Neo4j 研究对象图谱"
    )
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j 连接地址"
    )
    neo4j_username: str = Field(
        default="neo4j",
        description="Neo4j 用户名"
    )
    neo4j_password: str = Field(
        default="",
        description="Neo4j 密码，必须通过环境变量注入，禁止写入源码"
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database 名称"
    )
    # Phase 1（R2）— Neo4j 优雅降级超时配置
    neo4j_connect_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Neo4j driver 连接超时（秒）；触发 SocketDeadlineExceededError 后由上层降级"
    )
    neo4j_read_timeout_seconds: float = Field(
        default=10.0,
        ge=0.5,
        le=120.0,
        description="Neo4j 单次只读查询超时（秒）；到点由服务端终止事务并以 ClientError 上抛，"
        "该异常不在降级白名单内，由调用方按失败处理"
    )
    lightrag_insert_max_concurrency: int = Field(
        default=1,
        ge=1,
        le=4,
        description="LightRAG 后台建图最大并发数，CPU 环境下建议保持 1",
    )
    multimodal_rag_enabled: bool = Field(
        default=False,
        description="是否启用多模态 RAG（预留开关）"
    )

    # ============ 应用配置 ============
    app_name: str = Field(
        default="Study Assistant",
        description="应用名称"
    )
    app_version: str = Field(
        default="2.0.0-subgraph",
        description="应用版本"
    )
    debug: bool = Field(
        default=False,
        description="是否开启调试模式"
    )
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")

    class Config:
        """Pydantic Settings 配置。"""
        env_file = _ENV_FILES
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量
        # 环境变量前缀（可选）
        # env_prefix = "APP_"


# 全局配置实例（单例）。
_settings_instance = None


def get_settings() -> Settings:
    """
    获取配置实例（单例模式）。

    Returns:
        Settings: 配置实例
    """
    global _settings_instance

    if _settings_instance is None:
        _settings_instance = Settings()

    return _settings_instance


def get_settings_env_path() -> Path:
    """无参数；返回当前副本的权威配置文件路径，供预设持久化使用。"""
    return _PROJECT_ROOT / ".env"


def reset_settings(*, llm_overrides: dict[str, object] | None = None) -> "Settings":
    """
    从当前副本配置重新加载单例，返回验证后的 Settings。

    参数:
        llm_overrides: 用户明确激活的 LLM 连接及重试字段；省略时按普通启动优先级读取

    返回:
        重新加载后的 Settings 实例
    """
    global _settings_instance
    if llm_overrides is not None and set(llm_overrides) - _LLM_PROFILE_FIELDS:
        raise ValueError("LLM 预设只能覆盖模型连接和重试配置")
    # 1. 【配置】【预设激活】明确选中的 LLM 参数优先；其余字段继续遵守启动配置优先级
    _settings_instance = Settings(**(llm_overrides or {}))
    return _settings_instance


def is_vector_rerank_enabled(settings: object) -> bool:
    """兼容新旧配置名，统一读取应用层向量 rerank 开关。"""
    value = getattr(settings, "vector_rerank_enabled", None)
    if value is None:
        value = getattr(settings, "rerank_enabled", True)
    return bool(value)


def get_vector_rerank_top_k(settings: object) -> int:
    """兼容新旧配置名，统一读取应用层向量 rerank top_k。"""
    value = getattr(settings, "vector_rerank_top_k", None)
    if value is None:
        value = getattr(settings, "rerank_top_k", getattr(settings, "rag_top_k", 5))
    return max(1, int(value))


def get_vector_rerank_model(settings: object) -> str:
    """兼容新旧配置名，统一读取应用层向量 rerank 模型名。"""
    value = getattr(settings, "vector_rerank_model", None)
    if value:
        return str(value)
    return str(getattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3"))


def normalize_model_api_url(value: str, *, setting_name: str) -> str:
    """接收服务地址与配置名，返回无凭证的规范 URL；非法地址只报告配置名。"""
    message = f"{setting_name} 必须是有效的 HTTP(S) 地址，且不能包含凭证、查询参数或片段"
    try:
        parts = urlsplit(value.strip())
        port = parts.port
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None
                or parts.query or parts.fragment or any(char.isspace() for char in value)):
            raise ValueError(message)
        host = parts.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if port is not None and (parts.scheme, port) not in {("http", 80), ("https", 443)}:
            host = f"{host}:{port}"
        return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), "", ""))
    except (ValueError, AttributeError):
        raise ValueError(message) from None


def get_embedding_identity(settings: object) -> str:
    """接收模型配置，返回不含密钥的向量指纹；local 保留原发布数据的摘要算法。"""
    payload = {"model": getattr(settings, "embedding_model"),
        "dimension": getattr(settings, "embedding_dimension"), "normalized": True}
    mode = getattr(settings, "embedding_mode", "api")
    if mode == "api":
        payload.update(mode="api", service=normalize_model_api_url(
            getattr(settings, "embedding_api_base_url"), setting_name="EMBEDDING_API_BASE_URL",
        ))
    elif mode != "local":
        raise ValueError("EMBEDDING_MODE 必须是 api 或 local")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def print_settings():
    """打印当前配置（用于调试）"""
    settings = get_settings()

    print("\n" + "=" * 70)
    print("  当前配置")
    print("=" * 70)

    print("\n【LLM配置】")
    print(f"  模型: {settings.llm_model_name}")
    print(f"  温度: {settings.llm_temperature}")

    print("\n【搜索配置】")
    print(f"  最大结果数: {settings.search_max_results}")

    print("\n【任务配置】")
    print(f"  计划任务数: {settings.planner_task_count}")
    print(f"  最大并发数: {settings.executor_max_concurrent}")
    print(f"  显示搜索详情: {settings.executor_show_search_details}")

    print("\n【报告配置】")
    print(f"  目标字数: {settings.report_target_length}")
    print(f"  报告风格: {settings.report_style}")

    print("\n【基础设施配置】")
    print(f"  DB后端: {settings.db_backend}")
    print(f"  向量后端: {settings.vector_backend}")
    print(f"  Milvus: {settings.milvus_uri}")
    print(f"  GraphRAG: {settings.graph_rag_enabled}")
    print(f"  GraphRAG Rerank: {settings.graph_rerank_enabled}")
    print(f"  Vector Rerank: {is_vector_rerank_enabled(settings)}")
    print(f"  Neo4j: {settings.neo4j_enabled} @ {settings.neo4j_uri}")

    print("\n【应用配置】")
    print(f"  应用名称: {settings.app_name}")
    print(f"  版本: {settings.app_version}")
    print(f"  调试模式: {settings.debug}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 测试配置
    print_settings()
