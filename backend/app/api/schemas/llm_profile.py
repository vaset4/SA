"""LLM 供应商预设相关 schema。

每个预设收拢一个上游供应商的连接三要素与其专属的重试/限流规则。
之所以把重试规则并入预设而非全局配置：不同供应商脾气不同——
RPM 低的上游要慢、要长间隔；RPM 高但不稳定的免费上游要快速失败、
短等待、多重试。切换预设时连接与重试策略必须一起换。
"""

from pydantic import BaseModel, Field


class LLMProfileRetryRule(BaseModel):
    """单个供应商的重试与限流规则。"""

    requests_per_minute: int = Field(
        default=0,
        ge=0,
        le=600,
        description="每分钟请求上限；0 表示只用显式最小间隔。用于反推最小请求间隔(60/rpm)",
    )
    max_retries: int = Field(
        default=5,
        ge=0,
        le=15,
        description="HTTP/连接错误重试次数；不稳定上游调高(如8)靠多试扛抖动",
    )
    retry_backoff_seconds: float = Field(
        default=1.5,
        ge=0.0,
        le=60.0,
        description="单次重试等待秒数；快速失败的不稳定上游取短值(1.5)，RPM 低的取长值",
    )
    timeout_seconds: int = Field(
        default=240,
        ge=5,
        le=600,
        description="单次请求超时秒数",
    )


class LLMProfileBase(BaseModel):
    """预设的可编辑字段。"""

    name: str = Field(min_length=1, max_length=60, description="预设显示名，如 NVIDIA免费")
    base_url: str = Field(min_length=1, max_length=500, description="OpenAI 兼容 Base URL")
    model_name: str = Field(min_length=1, max_length=120, description="模型名")
    api_key: str = Field(default="", max_length=500, description="API Key；编辑时留空表示不修改")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度")
    retry_rule: LLMProfileRetryRule = Field(default_factory=LLMProfileRetryRule)


class LLMProfileCreate(LLMProfileBase):
    """新增预设请求体。"""


class LLMProfileUpdate(LLMProfileBase):
    """编辑预设请求体；语义同新增，api_key 留空表示保留原值。"""


class LLMProfileView(LLMProfileBase):
    """预设对外视图：带 id 与激活标记，api_key 脱敏。"""

    id: str
    is_active: bool = False
    api_key_masked: str = Field(default="", description="脱敏后的 Key，仅供展示")


class LLMTestRequest(BaseModel):
    """测试连接请求；不传字段则测当前激活预设。"""

    base_url: str = Field(default="", max_length=500)
    model_name: str = Field(default="", max_length=120)
    api_key: str = Field(default="", max_length=500)
    profile_id: str = Field(default="", description="传入则用该预设的 Key/URL 测试")


class LLMTestResult(BaseModel):
    """测试连接结果。"""

    ok: bool
    latency_ms: int = Field(default=0, description="请求往返耗时毫秒")
    status_code: int = Field(default=0, description="上游 HTTP 状态码，0 表示未拿到响应")
    message: str = Field(default="", description="成功或失败的可读说明")
