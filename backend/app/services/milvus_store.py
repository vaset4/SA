"""
Milvus 向量存储服务。

这是当前默认的向量检索主实现。上传入库主链会把 `document_parser` 产出的
`Chunk` 写入这里，RAG 主链也默认通过这里完成 dense + BM25 混合检索、
全文展开和健康检查。

与 `vector_store.py` 的关系：
  - `milvus_store.py`：当前默认主链，面向在线检索
  - `vector_store.py`：本地 FAISS 兼容实现，主要用于兼容/离线场景

从调用链上看，最重要的几个入口是：
  - 上传链：`upload_steps.py / document_index_service.py` -> `add_paper()`
  - 检索链：`retrieval_service.py` -> `hybrid_search() / local_rag_search()`
  - 阅读链：`rag_node.py` -> `read_chunk()`
"""

import time
from typing import Dict, List, Optional, Set
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker
except ImportError as exc:
    AnnSearchRequest = DataType = Function = FunctionType = MilvusClient = RRFRanker = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import app.services.paper_db as paper_db
from app.config import get_embedding_identity, get_settings
from app.models.research_contracts import RetrievalScopeError
from app.models.models import Chunk
from app.services.cache import get_cache

settings = get_settings()
_STORE: Optional["MilvusStore"] = None
# 这些上限对应 collection schema 的 VARCHAR 长度约束，入库前必须主动截断。
_FULL_TEXT_LIMIT = 65535
_TEXT_LIMIT = 8192
_SECTION_LIMIT = 512
MS_PER_SEC = 1000.0
HEALTH_PROBE_TIMEOUT_SECONDS = 2.0
HEALTH_FAILURE_COOLDOWN_SECONDS = 30.0
_COUNT_BATCH_SIZE = 1000
_WRITE_BATCH_SIZE = 128
_MILVUS_IO_TIMEOUT_SECONDS = 60
_FIELD_OUTPUTS = [
    "chunk_id", "paper_id", "batch_id", "parent_id", "embedding_text", "display_text",
    "full_text", "section", "page_num", "block_type", "section_path",
    "asset_ref", "parser_mode", "authors", "venue", "category", "file_type", "year",
]
# 按章节路径取正文只需这些字段：正文字段取 display_text，不投影 embedding_text / full_text
# 等大字段，避免为几节正文把整篇论文的全文拉进内存
_SECTION_ROW_OUTPUT_FIELDS = [
    "chunk_id", "parent_id", "section", "section_path", "block_type", "page_num", "display_text",
]
# 按元数据条件筛块只要定位字段：本能力只回答「哪篇/哪节/第几页/什么类型/哪一块」，
# 不投影 display_text，要正文是 read_section_chunks 的活
_METADATA_ROW_OUTPUT_FIELDS = [
    "chunk_id", "parent_id", "section", "section_path", "block_type", "page_num",
]


def _require_dependencies() -> None:
    """确保 Milvus 依赖已安装。"""
    if _IMPORT_ERROR is not None:
        raise RuntimeError("缺少 pymilvus 依赖，无法使用 Milvus 存储") from _IMPORT_ERROR


def get_embedding_client():
    """按需取得编码端口；仅检查发布状态和隔离索引 IO 时无需加载本地模型。"""
    from app.services.embedding_client import get_embedding_client as get_client

    return get_client()


def _truncate(text: str, limit: int) -> str:
    """按字段上限截断文本，避免超过 collection schema 限制。"""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _quote(value: str) -> str:
    """转义 Milvus filter 表达式中的字符串值。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _string_in_clause(values: List[str]) -> str:
    """把字符串序列拼成 Milvus filter 里 `in [...]` 的取值片段，逐项转义。"""
    return ", ".join(f'"{_quote(value)}"' for value in values)


def _ordered_unique(values: List[str]) -> List[str]:
    """
    去重并保持首次出现顺序，丢弃空值。

    章节路径按 chunk 写入顺序重复出现（同一章节切了多块），
    清单要的是章节本身，顺序则代表论文里的先后。

    Args:
        values: 可能含空值和重复项的字符串序列
    Returns:
        List[str]: 首次出现顺序的非空唯一值
    """
    seen: Set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


class MilvusStore:
    """当前默认主检索链的 Milvus 封装。"""
    def __init__(self, *, collection_name: str | None = None) -> None:
        """接收显式集合配置，迁移可独立建立影子集合。"""
        self._collection_name = collection_name or settings.milvus_collection_v2
        self._client = None
        self._last_health_check_at = 0.0
        self._last_health_status: bool | None = None
        self._last_health_error = ""
        self._schema_checked = False
        self._reader_user_id: str | None = None
        self._reader_snapshot: dict | None = None

    def for_user(self, user_id: str) -> "MilvusStore":
        """为一次已认证读取绑定用户，避免把用户状态写进共享单例。"""
        from copy import copy

        bound = copy(self)
        bound._reader_user_id = user_id
        return bound

    def for_snapshot(self, snapshot: dict | None) -> "MilvusStore":
        """为一次已授权读取绑定本轮发布快照，让按论文直读沿用同一版本。

        向量检索在请求创建时就用快照固定了批次，按论文直读（列章节、取正文、按元数据筛块）
        若仍读当前批次，同一次研究会在新旧版本间混用证据，甚至用旧章节路径筛新批次产生假空结果。
        绑定后副本沿用既有用户身份，两者可叠加使用。

        Args:
            snapshot: 请求级发布快照，字段取自 attach_publication_snapshot 的产出
        Returns:
            MilvusStore: 绑定了快照的副本；快照为 None 或空字典时不绑定，行为与原来一致
        """
        from copy import copy

        bound = copy(self)
        bound._reader_snapshot = snapshot or None
        return bound

    def _current_batch(self, paper_id: str) -> str | None:
        """诊断与存在性检查只读取当前已发布批次，不统计未核验的私有写入。

        绑定过发布快照的副本只认快照里的批次，保证一次读取内的版本稳定；
        论文不在快照映射里说明本轮没有授权这篇，如实返回空，不回退去读当前批次。
        """
        if self._reader_snapshot is not None:
            from app.services.upload_read_service import authorize_publication_read

            paper_db._run(authorize_publication_read(self._reader_snapshot,
                user_id=self._reader_user_id or "", paper_ids=(paper_id,)))
            return self._reader_snapshot.get("paper_batches", {}).get(paper_id)

        from app.repositories.upload_publication_repository import get_upload_publication_repository

        return paper_db._run(get_upload_publication_repository().current_batch(paper_id, user_id=self._reader_user_id))

    @property
    def collection_name(self) -> str:
        """返回本实例使用的显式集合配置，迁移期间不混读旧集合。"""
        return self._collection_name

    def _get_client(self):
        """延迟创建 MilvusClient；建链等待也带上限，服务端不可达时挂在这一步，重试根本轮不到。

        构造参数 timeout 既是建链等待上限，也是该客户端后续调用的默认上限，取值与单次 IO 上限一致。
        """
        _require_dependencies()
        if self._client is None:
            self._client = MilvusClient(uri=settings.milvus_uri, timeout=_MILVUS_IO_TIMEOUT_SECONDS)
        return self._client

    def _health_url(self) -> str:
        """
        把 Milvus gRPC/REST 端口地址映射到 healthz 管理端口。

        主链不直接靠一次真实检索做健康检查，而是先走更轻量的 `/healthz`，这样启动期
        和请求前探针的成本更低，也更容易区分"服务不可达"和"查询逻辑异常"。
        """
        parsed = urlsplit(settings.milvus_uri)
        scheme = parsed.scheme or "http"
        hostname = parsed.hostname or "localhost"
        netloc = hostname
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        if parsed.port is not None:
            netloc = f"{netloc}:9091"
        return urlunsplit((scheme, netloc, "/healthz", "", ""))

    def probe_health(self, *, force: bool = False) -> bool:
        """
        使用 Milvus 管理端口做轻量健康探针。

        失败后会在短时间内熔断，避免每次请求都重复走长阻塞链。
        """
        now = time.monotonic()
        if (
            not force
            and self._last_health_status is False
            and (now - self._last_health_check_at) < HEALTH_FAILURE_COOLDOWN_SECONDS
        ):
            return False
        health_url = self._health_url()
        try:
            response = requests.get(health_url, timeout=HEALTH_PROBE_TIMEOUT_SECONDS)
            healthy = response.ok and "ok" in (response.text or "").strip().lower()
            self._last_health_status = healthy
            self._last_health_error = "" if healthy else f"health endpoint returned {response.status_code}"
            self._last_health_check_at = now
            return healthy
        except Exception as exc:
            self._last_health_status = False
            self._last_health_error = str(exc)
            self._last_health_check_at = now
            return False

    def last_health_error(self) -> str:
        """返回最近一次健康探针失败原因。"""
        return self._last_health_error

    def warmup(self) -> None:
        """启动期预热：先做健康探针，再触发 collection 确认。"""
        if not self.probe_health(force=True):
            detail = self.last_health_error() or "health probe failed"
            raise RuntimeError(f"Milvus health probe failed: {detail}")
        self.ensure_collection()

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本，并复用 embedding 缓存减少重复开销。"""
        cache = get_cache()
        model_key = get_embedding_identity(settings)
        vectors: List[List[float] | None] = [None] * len(texts)
        missing_positions: Dict[str, List[int]] = {}
        for index, text in enumerate(texts):
            cached_vector = cache.get_embedding(model_key + ":" + text)
            if cached_vector is not None:
                vectors[index] = cached_vector
                continue
            missing_positions.setdefault(text, []).append(index)
        if missing_positions:
            missing_texts = list(missing_positions)
            encoded_list = get_embedding_client().embed_texts(missing_texts)
            for text, vector in zip(missing_texts, encoded_list):
                cache.set_embedding(model_key + ":" + text, vector)
                for index in missing_positions[text]:
                    vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise RuntimeError("Embedding 编码结果不完整")
        return [vector for vector in vectors if vector is not None]

    def ensure_collection(self) -> None:
        """核对批次字段和向量维度；无参数，不匹配时保留原集合并明确报错。"""
        client = self._get_client()
        if client.has_collection(collection_name=self.collection_name, timeout=_MILVUS_IO_TIMEOUT_SECONDS):
            if not self._schema_checked:
                fields = client.describe_collection(
                    collection_name=self.collection_name, timeout=_MILVUS_IO_TIMEOUT_SECONDS)["fields"]
                if "batch_id" not in {field["name"] for field in fields}:
                    raise RuntimeError("当前 Milvus 集合尚未迁移到批次发布结构，请先执行有界集合迁移")
                # 1. 【检索】【集合维度】已有数据不因模型切换而删除或自动重建
                dense_field = next((field for field in fields if field["name"] == "dense_vector"), None)
                if dense_field is None:
                    raise RuntimeError("当前 Milvus 集合缺少 dense_vector 字段，请使用匹配的集合并重新处理资料")
                actual_dimension = dense_field.get("params", {}).get("dim")
                if str(actual_dimension) != str(int(settings.embedding_dimension)):
                    raise RuntimeError(
                        f"当前 Milvus 集合向量维度 {actual_dimension} 与配置 {settings.embedding_dimension} 不一致，"
                        "请选择新集合并重新处理资料"
                    )
                self._schema_checked = True
            return

        schema = self._upload_schema()

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="chunk_id", index_type="AUTOINDEX")
        index_params.add_index(field_name="paper_id", index_type="AUTOINDEX")
        index_params.add_index(field_name="batch_id", index_type="AUTOINDEX")
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="sparse_vector", index_type="AUTOINDEX", metric_type="BM25")

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        self._schema_checked = True

    def _upload_schema(self):
        """构造批次隔离的完整索引 schema；返回当前模型维度对应的字段和 BM25 函数。"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="batch_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="embedding_text", datatype=DataType.VARCHAR, max_length=_TEXT_LIMIT, enable_analyzer=True)
        schema.add_field(field_name="display_text", datatype=DataType.VARCHAR, max_length=_TEXT_LIMIT)
        schema.add_field(field_name="full_text", datatype=DataType.VARCHAR, max_length=_FULL_TEXT_LIMIT)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=_SECTION_LIMIT)
        schema.add_field(field_name="page_num", datatype=DataType.INT64)
        schema.add_field(field_name="block_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="asset_ref", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="parser_mode", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="year", datatype=DataType.INT64)
        schema.add_field(field_name="authors", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="venue", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=int(settings.embedding_dimension),
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["embedding_text"],
                output_field_names=["sparse_vector"],
            )
        )
        return schema


    def write_upload_batch(self, batch_id: str, rows: list[dict]) -> list[dict]:
        """【上传】【私有索引写入】只写当前独立批次，回读完整载荷交给发布服务核验。"""
        if not rows or any(row.get("batch_id") != batch_id for row in rows):
            raise ValueError("索引载荷缺少一致的独立批次身份")
        self.ensure_collection()
        client = self._get_client()
        for start in range(0, len(rows), _WRITE_BATCH_SIZE):
            batch = rows[start:start + _WRITE_BATCH_SIZE]
            result = client.upsert(collection_name=self.collection_name, data=batch, timeout=_MILVUS_IO_TIMEOUT_SECONDS)
            if result.get("upsert_count") != len(batch):
                raise RuntimeError("Milvus 未确认完整批次写入")
        client.flush(collection_name=self.collection_name, timeout=_MILVUS_IO_TIMEOUT_SECONDS)
        iterator = client.query_iterator(collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"', output_fields=[*_FIELD_OUTPUTS, "dense_vector"],
            batch_size=_COUNT_BATCH_SIZE, consistency_level="Strong", timeout=_MILVUS_IO_TIMEOUT_SECONDS)
        actual = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    return actual
                actual.extend(dict(row) for row in batch)
        finally:
            iterator.close()

    def delete_upload_batch(self, batch_id: str) -> None:
        """清理已经过发布仓储确认无读取占用的旧批次；保留其他版本。"""
        self.ensure_collection()
        self._get_client().delete(collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"', timeout=_MILVUS_IO_TIMEOUT_SECONDS)
        self._get_client().flush(collection_name=self.collection_name, timeout=_MILVUS_IO_TIMEOUT_SECONDS)

    def add_paper(self, paper_id: str, chunks: List[Chunk], metadata: dict | None = None) -> int:
        """拒绝绕过受理、核验和事务发布的旧写入方式，避免破坏已有可用结果。"""
        raise RuntimeError("论文索引必须通过上传任务的独立批次发布，请使用上传或重新处理入口")

    def exists(self, paper_id: str) -> bool:
        """判断某篇论文是否已写入当前 Milvus collection。"""
        batch_id = self._current_batch(paper_id)
        if not batch_id:
            return False
        self.ensure_collection()
        rows = self._get_client().query(
            collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"',
            output_fields=["paper_id"],
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        return bool(rows)

    def count_paper_chunks(self, paper_id: str) -> int:
        """统计某篇论文当前写入 Milvus 的 chunk 数。"""
        batch_id = self._current_batch(paper_id)
        if not batch_id:
            return 0
        self.ensure_collection()
        iterator = self._get_client().query_iterator(
            collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"',
            output_fields=["chunk_id"],
            batch_size=_COUNT_BATCH_SIZE,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        count = 0
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    return count
                count += len(rows)
        finally:
            iterator.close()

    def list_section_paths(self, paper_id: str) -> List[str]:
        """
        列出某篇论文已索引的章节路径，只取章节字段。

        规划提示词要的是「这篇论文有哪些章节」这份名单，正文由工具按需再取。
        因此这里只投影 `section_path`，不能改用 `list_chunk_rows_by_paper`——
        那个方法会把 `full_text` 等正文字段整篇拉进内存。

        Args:
            paper_id: 论文 ID
        Returns:
            List[str]: 按论文内先后顺序去重的章节路径；未发布或零行时为空列表
        """
        batch_id = self._current_batch(paper_id)
        if not batch_id:
            return []
        self.ensure_collection()
        iterator = self._get_client().query_iterator(
            collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"',
            output_fields=["section_path"],
            batch_size=_COUNT_BATCH_SIZE,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        paths: list[str] = []
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    return _ordered_unique(paths)
                paths.extend(str(row.get("section_path") or "") for row in rows)
        finally:
            iterator.close()

    def read_section_chunks(self, paper_id: str, section_paths: List[str]) -> List[dict]:
        """
        取回某篇论文指定章节的块行，只投影拼正文与定位需要的字段。

        按章节路径取正文的取证能力（planner 的 local_section_text）走这里：
        调用方已用章节名单把模型点名的短词落到真实路径上，这里只按批次 + 路径
        精确取块，不把整篇论文的 `full_text` 拉回内存。

        Args:
            paper_id: 论文 ID
            section_paths: 要取正文的章节路径，取自 `list_section_paths` 的真实路径
        Returns:
            List[dict]: 该论文命中章节的块行；未发布或路径都不存在时为空列表
        """
        batch_id = self._current_batch(paper_id)
        if not batch_id or not section_paths:
            return []
        self.ensure_collection()
        path_clause = ", ".join(f'"{_quote(path)}"' for path in section_paths)
        rows = self._get_client().query(
            collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}" and section_path in [{path_clause}]',
            output_fields=_SECTION_ROW_OUTPUT_FIELDS,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        return [dict(row) for row in rows]

    def list_chunk_rows_by_metadata(
        self,
        paper_id: str,
        *,
        section_paths: List[str] | None = None,
        block_types: List[str] | None = None,
        page_nums: List[int] | None = None,
    ) -> List[dict]:
        """
        按元数据条件筛出某篇论文的块行，只回定位字段不回正文。

        规划器「按元数据条件筛块」的取证能力（local_chunk_metadata）走这里。调用方给的
        章节路径是已落位的真实路径，块类型与页码按精确值匹配；条件之间是 AND，同一条件
        多个值是 OR。这里只投影定位字段，不取 `display_text`——要读正文是
        `read_section_chunks` 的活，两条路不混。

        Args:
            paper_id: 论文 ID
            section_paths: 章节路径条件，取自 `list_section_paths` 的真实路径；空表示不限
            block_types: 块类型条件；空表示不限
            page_nums: 页码条件，精确匹配不做范围解析；空表示不限
        Returns:
            List[dict]: 命中块行；未发布或零命中时为空列表
        """
        batch_id = self._current_batch(paper_id)
        if not batch_id:
            return []
        clauses = [f'batch_id == "{_quote(batch_id)}"']
        if section_paths:
            clauses.append(f"section_path in [{_string_in_clause(section_paths)}]")
        if block_types:
            clauses.append(f"block_type in [{_string_in_clause(block_types)}]")
        if page_nums:
            page_clause = ", ".join(str(int(page)) for page in page_nums)
            clauses.append(f"page_num in [{page_clause}]")
        self.ensure_collection()
        rows = self._get_client().query(
            collection_name=self.collection_name,
            filter=" and ".join(clauses),
            output_fields=_METADATA_ROW_OUTPUT_FIELDS,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        return [dict(row) for row in rows]

    def get_chunk_row(self, chunk_id: str) -> dict | None:
        """按 chunk_id 读取当前 Milvus v2 行，供诊断接口使用。"""
        if self._reader_user_id:
            from app.services.upload_read_service import read_published_chunk
            read_published_chunk(chunk_id, user_id=self._reader_user_id, publication=self._reader_snapshot)
        self.ensure_collection()
        rows = self._get_client().query(
            collection_name=self.collection_name,
            ids=[chunk_id],
            output_fields=_FIELD_OUTPUTS,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        return dict(rows[0]) if rows else None

    def list_chunk_rows_by_paper(self, paper_id: str, batch_size: int = _COUNT_BATCH_SIZE) -> List[dict]:
        """按 paper_id 遍历当前 Milvus v2 行，供 chunk map 和 audit 使用。"""
        batch_id = self._current_batch(paper_id)
        if not batch_id:
            return []
        self.ensure_collection()
        iterator = self._get_client().query_iterator(
            collection_name=self.collection_name,
            filter=f'batch_id == "{_quote(batch_id)}"',
            output_fields=_FIELD_OUTPUTS,
            batch_size=batch_size,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        rows: list[dict] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    return rows
                rows.extend(dict(row) for row in batch)
        finally:
            iterator.close()

    def flush_collection(self) -> None:
        """等待当前 collection 的写入对后续标量查询可见。"""
        self.ensure_collection()
        self._get_client().flush(collection_name=self.collection_name, timeout=_MILVUS_IO_TIMEOUT_SECONDS)

    def delete_paper(self, paper_id: str) -> None:
        """按 `paper_id` 删除整篇论文在 Milvus 中的全部 chunk。"""
        self.ensure_collection()
        self._get_client().delete(
            collection_name=self.collection_name,
            filter=f'paper_id == "{_quote(paper_id)}"',
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )

    def list_paper_ids(self, batch_size: int = 256) -> List[str]:
        """遍历当前 collection 中出现过的全部 `paper_id`。"""
        self.ensure_collection()
        iterator = self._get_client().query_iterator(
            collection_name=self.collection_name,
            filter="paper_id != \"\"",
            output_fields=["paper_id"],
            batch_size=batch_size,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        paper_ids: list[str] = []
        seen: set[str] = set()
        while True:
            rows = iterator.next()
            if not rows:
                break
            for row in rows:
                paper_id = str(row.get("paper_id") or "").strip()
                if not paper_id or paper_id in seen:
                    continue
                seen.add(paper_id)
                paper_ids.append(paper_id)
        iterator.close()
        return paper_ids

    def _format_hit(self, hit: dict) -> dict:
        """把 Milvus 命中结果统一转换成上层 RAG 可消费的命中结构。"""
        entity = hit.get("entity", {}) if hasattr(hit, "get") else {}
        if not isinstance(entity, dict):
            entity = {}
        chunk_id = (
            (hit.get("chunk_id") if hasattr(hit, "get") else None)
            or getattr(hit, "id", None)
            or (hit.get("id") if hasattr(hit, "get") else None)
            or entity.get("chunk_id")
        )
        preview = (entity.get("display_text") or entity.get("embedding_text") or "").strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        return {
            "chunk_id": chunk_id,
            "batch_id": entity.get("batch_id", ""),
            "parent_id": entity.get("parent_id", ""),
            "preview": f'[{entity.get("section", "")} p.{entity.get("page_num", 0)}] {preview}'.strip(),
            "paper_id": entity.get("paper_id", ""),
            "section": entity.get("section", ""),
            "section_path": entity.get("section_path", ""),
            "page_num": entity.get("page_num", 0),
            "block_type": entity.get("block_type", "paragraph"),
            "parser_mode": entity.get("parser_mode", ""),
            "display_text": entity.get("display_text", ""),
            "score": float(hit.get("distance", 0.0)),
        }

    def local_rag_search(self, query: str, top_k: int = None, filter_paper_ids: Optional[List[str]] = None, *, scope: dict | None = None) -> List[dict]:
        """
        仅执行 dense 向量检索。

        该方法主要用于兼容旧接口或需要仅语义检索的场景；
        当前主 RAG 冷启动更常用的是 `hybrid_search()`。
        """
        scoped_paper_ids = [
            str(item or "").strip()
            for item in filter_paper_ids or []
            if str(item or "").strip()
        ]
        if not scoped_paper_ids:
            raise RetrievalScopeError()
        if scope is None:
            raise RetrievalScopeError("本地检索缺少授权发布快照")
        from app.services.upload_read_service import require_publication_scope

        publication_filter = require_publication_scope(scope, collection_name=self.collection_name)
        self.ensure_collection()
        top_k = top_k or settings.rag_top_k
        ids = ", ".join(f'"{_quote(item)}"' for item in scoped_paper_ids)
        filter_expr = f"(paper_id in [{ids}]) and ({publication_filter})"
        result = self._get_client().search(
            collection_name=self.collection_name,
            data=self._encode([query]),
            anns_field="dense_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=_FIELD_OUTPUTS,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
        )
        return [self._format_hit(hit) for hit in (result[0] if result else [])]

    def search_scope(self, query: str, *, top_k: int, scope: dict) -> List[dict]:
        """【检索】【固定批次】验证授权发布快照后查询，同一请求不混用新旧索引。"""
        from app.services.upload_read_service import require_publication_scope

        filter_expr = require_publication_scope(scope, collection_name=self.collection_name)
        return self.hybrid_search(query, top_k=top_k, filter_expr=filter_expr)

    def hybrid_search(self, query: str, top_k: int = 10, filter_expr: str = "") -> List[dict]:
        """
        执行当前默认的 Milvus 混合检索。

        检索流程是：
          dense_vector 语义检索 + sparse_vector BM25 检索
          -> RRFRanker 融合
          -> 返回统一命中结构
        """
        scoped_filter_expr = str(filter_expr or "").strip()
        if not scoped_filter_expr or "paper_id" not in scoped_filter_expr or "batch_id" not in scoped_filter_expr:
            raise RetrievalScopeError()
        self.ensure_collection()
        dense_req = AnnSearchRequest(
            data=self._encode([query]),
            anns_field="dense_vector",
            param={"metric_type": "IP"},
            limit=top_k * 4,
            expr=scoped_filter_expr,
        )
        sparse_req = AnnSearchRequest(
            # sparse 检索直接吃原始 query 字符串，让 BM25 函数在 Milvus 侧自行分词。
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k * 4,
            expr=scoped_filter_expr,
        )
        result = self._get_client().hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=top_k,
            output_fields=_FIELD_OUTPUTS,
            timeout=_MILVUS_IO_TIMEOUT_SECONDS,
            consistency_level="Strong",
        )
        return [self._format_hit(hit) for hit in (result[0] if result else [])]

    def read_chunk(self, chunk_id: str, already_read: Set[str], *, user_id: str | None = None) -> dict:
        """按用户读取不可变引用的完整父块；向量退役后仍可回读原版本正文。"""
        from app.services.upload_read_service import read_published_chunk

        owner = user_id or self._reader_user_id
        if not owner:
            raise LookupError("读取引用正文需要当前用户身份")
        row = read_published_chunk(chunk_id, user_id=owner, publication=self._reader_snapshot)
        parent_key = "parent:" + row["parent_id"]
        skipped = chunk_id in already_read or parent_key in already_read
        return {**row, "skipped": skipped, "read_keys": [chunk_id, parent_key],
            "full_text": "(already read parent chunk, skipped to save tokens)" if skipped else row["full_text"]}

    def is_healthy(self) -> bool:
        """检查 Milvus 管理端口是否健康响应。"""
        return self.probe_health()


def get_milvus_store() -> MilvusStore:
    """返回 MilvusStore 单例，供当前主检索链复用。"""
    global _STORE
    if _STORE is None:
        _STORE = MilvusStore()
    return _STORE
