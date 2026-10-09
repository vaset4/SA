"""上传恢复、产物与发布的模块边界，作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

UploadOperation = Literal["ingest", "rebuild", "metadata", "graph"]
UploadStage = Literal["receive", "parse", "metadata", "embedding", "index", "publish", "cleanup"]


class ArtifactRef(TypedDict):
    """不可变产物的物理定位和完整性证据。"""

    path: str
    sha256: str
    size: int


class PublicationSnapshot(TypedDict):
    """请求固定的授权发布版本和读取占用，供 RAG 与向量存储完整透传。"""

    lease_id: str
    user_id: str
    batch_ids: list[str]
    paper_batches: dict[str, str]
    embedding_fingerprints: list[str]
    collection_names: list[str]


@dataclass(frozen=True, slots=True)
class UploadRequest:
    """已核验原文件的受理意图；业务归属由服务端提供。"""

    user_id: str
    request_id: str
    request_fingerprint: str
    paper_id: str
    task_id: str
    parse_id: str
    library_id: str
    project_id: str | None
    filename: str
    source_ref: ArtifactRef
    graph_requested: bool = False


@dataclass(frozen=True, slots=True)
class UploadExecution:
    """一次持租执行的身份；后续执行不能复用该批次。"""

    task_id: str
    worker_id: str
    paper_id: str
    user_id: str
    parse_id: str
    batch_id: str
    previous_batch_id: str | None
    operation: UploadOperation = "ingest"


@dataclass(frozen=True, slots=True)
class UploadFailure:
    """可以持久保存并直接说明给用户的失败事实。"""

    stage: str
    kind: str
    detail: str
    can_resume: bool = True
    retryable: bool = False
    retry_after: float | None = None


class UploadPausedError(RuntimeError):
    """当前上传必须保留进度并等待用户继续。"""

    def __init__(self, failure: UploadFailure, *, execution: UploadExecution | None = None) -> None:
        """接收结构化原因；异常文本保留原始故障说明。"""
        super().__init__(failure.detail)
        self.failure = failure
        self.execution = execution


class UploadLeaseLostError(RuntimeError):
    """当前执行已失去写入上传状态和发布结果的权限。"""


class UploadCancelledError(RuntimeError):
    """用户已取消本次执行，保留来源且不再开始新的外部步骤。"""


class UploadConflictError(ValueError):
    """上传意图、原文或当前执行状态发生冲突。"""


class ArtifactIntegrityError(ValueError):
    """原文件、缓存或供应商产物未通过完整性核验。"""


class UploadCheckpoint(TypedDict):
    """可恢复步骤的输入身份、远端操作身份与已确认结果。"""

    parse_id: str
    stage: str
    operation_id: str
    input_hash: str
    state: str
    payload: dict[str, Any]
