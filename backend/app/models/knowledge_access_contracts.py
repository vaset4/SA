"""知识正文读取、来源范围及完整性合同，作者：xxx。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, NotRequired, TypedDict

from app.models.upload_models import ArtifactRef

if TYPE_CHECKING:
    from app.models.research_contracts import EvidenceRecord

CONTENT_SCHEMA_VERSION = 1
READ_CONTRACT_VERSION = 1


class SourceItem(TypedDict):
    """规范阅读文本及其原始项目定位；字符坐标均为 Unicode 半开区间。"""

    source_item: str
    text: str
    source_kind: str
    source_index: int
    block_type: str
    page_num: NotRequired[int | None]
    region: NotRequired[dict[str, Any] | None]
    artifact_ref: NotRequired[ArtifactRef]
    source_start: NotRequired[int]
    source_end: NotRequired[int]


class SourceFragment(TypedDict):
    """规范来源区间与块内实际正文区间的一一对应，不含合成分隔符。"""

    source_item: str
    source_start: int
    source_end: int
    text_start: int
    text_end: int
    page_num: NotRequired[int | None]
    region: NotRequired[dict[str, Any] | None]


class SourceSpan(TypedDict):
    """发布字段实际读取区间与返回材料正文区间，锚点不能代替全部范围。"""

    paper_id: str
    batch_id: str
    chunk_id: str
    field: Literal["text", "full_text"]
    start: int
    end: int
    text_start: int
    text_end: int
    source_fragments: NotRequired[list[SourceFragment]]


@dataclass(frozen=True, kw_only=True)
class KnowledgeReadRequest:
    """授权范围内的明确正文定位；额度沿用调用方现有配置。"""

    operation: Literal["chunks", "sections"]
    scope: Mapping[str, Any]
    paper_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...] = ()
    section_paths: tuple[str, ...] = ()
    granularity: Literal["child", "parent"] = "child"
    max_chars: int | None = None
    continuations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, kw_only=True)
class KnowledgeReadReport:
    """本次读取事实；不判断研究目标是否充分，也不作为可引用材料。"""

    operation: str
    status: Literal["complete", "partial", "not_applicable", "unavailable"]
    requested: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    remaining: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    used_chars: int = 0
    read_spans: tuple[SourceSpan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """无参数；返回独立的可序列化读取说明，供既有 metadata 与提示投影。"""
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class KnowledgeMaterial:
    """一次读取的一组真实正文，可以包含多个可核验来源跨度。"""

    paper_id: str
    batch_id: str
    group_kind: Literal["parent", "section"]
    group_locator: str
    material_group_id: str
    text: str
    source_spans: tuple[SourceSpan, ...]
    chunk_id: str
    parent_id: str
    section_path: str
    section: str
    page_num: int
    source_location_status: Literal["exact", "legacy_chunk", "no_physical_location"]
    block_type: str


@dataclass(frozen=True)
class KnowledgeReadResult:
    """正文材料与独立读取说明；空材料不等同于读取成功。"""

    materials: tuple[KnowledgeMaterial, ...]
    read_report: KnowledgeReadReport


@dataclass(frozen=True)
class SectionResolution:
    """单个章节请求的明确匹配结果；歧义保留候选，不擅自选择。"""

    requested: str
    status: Literal["matched", "not_found", "ambiguous"]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceReadResult:
    """所有生产取证适配器唯一返回合同，说明不能混进证据池。"""

    records: tuple[EvidenceRecord, ...]
    read_report: KnowledgeReadReport


class KnowledgeReadBudgetError(ValueError):
    """首个原子结构无法放入已有额度，没有任何可返回正文。"""

    def __init__(self, reason: str, *, remaining: tuple[dict[str, Any], ...], minimum_chars: int):
        """参数为额度冲突、剩余定位及最低空间；保存事实并返回明确异常。"""
        super().__init__(reason)
        self.remaining = remaining
        self.minimum_chars = minimum_chars
