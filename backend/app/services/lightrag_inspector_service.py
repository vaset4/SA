"""图谱数据诊断查看器 · LightRAG 侧只读读取服务。

只读解析 LightRAG 工作目录里的真实图数据，产出实体/关系明细与质量信号，
供诊断页"看清现状"。权威源是 graphml（节点含 entity_type/description，边含
description/keywords/weight）；不写、不删、不重抽，也不依赖 LightRAG 运行时
（即使 graph_rag 关闭、依赖未装，只要文件在就能读）。

数据源真相（步骤 0 实测，见 task design.md §0）：
- 实体类型在 graph_chunk_entity_relation.graphml 的节点属性，不在 kv_store_full_entities.json。
- kv_store_full_entities.json 是 {上传ID: {entity_names, count}}，不含类型，本服务不据它判类型。
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.services.graph.academic_schema import ACADEMIC_ENTITY_TYPES

# graphml 命名空间与文件名（LightRAG 写出的图结构权威文件）
_GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
_GRAPHML_FILE = "graph_chunk_entity_relation.graphml"
_DOC_STATUS_FILE = "kv_store_doc_status.json"
_TEXT_CHUNKS_FILE = "kv_store_text_chunks.json"
_LIGHTRAG_SUBDIR = "lightrag_index"

# 分页与采样默认值（不硬编码散落各处）
_DEFAULT_LIMIT = 200
_MAX_DUP_SAMPLE = 50

# 关系不可读时的统一原因文案
_REASON_NO_GRAPH = "LightRAG 图文件不存在或不可解析（该资料库可能尚未建图）"


def _data_root() -> Path:
    """返回数据根目录，单独抽出便于测试替换。"""
    return Path(get_settings().data_dir)


def _lib(library_id: str | None) -> str:
    """归一化 library_id，空值落到默认库。"""
    return (library_id or "").strip() or DEFAULT_LIBRARY_ID


def _working_dir(library_id: str | None) -> Path:
    """
    解析某个 library 的 LightRAG 工作目录。

    与 LightRAGService._working_dir_for 行为一致（默认库用根目录，其余库用子目录），
    这里独立解析是为了让诊断查看器不依赖 LightRAG 运行时即可定位文件。

    Args:
        library_id: 资料库 ID，空表示默认库

    Returns:
        Path: 该 library 的 LightRAG 工作目录
    """
    root = _data_root() / _LIGHTRAG_SUBDIR
    lib = _lib(library_id)
    return root if lib == DEFAULT_LIBRARY_ID else root / lib


def _graphml_path(library_id: str | None) -> Path:
    """返回某个 library 的 graphml 文件路径。"""
    return _working_dir(library_id) / _GRAPHML_FILE


def _is_suspect(entity_type: str) -> bool:
    """判定实体类型是否"可疑"：不在学术类型白名单内（大小写不敏感）。"""
    allowed = {item.lower() for item in ACADEMIC_ENTITY_TYPES}
    return entity_type.strip().lower() not in allowed


def _norm_name(name: str) -> str:
    """实体名规范化（仅用于重名分组的启发式：去空白 + 小写 + 折叠内部空格）。"""
    return " ".join(name.strip().lower().split())


def _build_key_map(root: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    """
    从 graphml 的 <key> 声明建"属性名→key id"映射（节点、边分开）。

    graphml 里 <data key="dN"> 的 dN 由写出方分配、跨版本可能变，稳定契约是
    <key attr.name="entity_type" .../> 的属性名，所以按名字而非 dN 取值。

    Args:
        root: graphml 根元素

    Returns:
        tuple: (节点属性名→keyid, 边属性名→keyid)
    """
    node_keys: dict[str, str] = {}
    edge_keys: dict[str, str] = {}
    for key in root.findall("g:key", _GRAPHML_NS):
        name = key.get("attr.name")
        key_id = key.get("id")
        if not name or not key_id:
            continue
        if key.get("for") == "node":
            node_keys[name] = key_id
        elif key.get("for") == "edge":
            edge_keys[name] = key_id
    return node_keys, edge_keys


def _data_value(elem: ET.Element, key_id: str | None) -> str:
    """按 key id 取某个元素下 <data> 的文本，缺失返回空串。"""
    if not key_id:
        return ""
    for data in elem.findall("g:data", _GRAPHML_NS):
        if data.get("key") == key_id:
            return (data.text or "").strip()
    return ""


def _parse_graph(library_id: str | None) -> dict[str, list] | None:
    """
    解析 graphml，产出标准化的节点与边。

    Args:
        library_id: 资料库 ID

    Returns:
        dict {"nodes": [...], "edges": [...]}；文件缺失/解析失败返回 None（上层据此报不可用）
    """
    path = _graphml_path(library_id)
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    # 1. 先建属性名映射，再按名字读节点/边的具体属性
    node_keys, edge_keys = _build_key_map(root)
    graph = root.find("g:graph", _GRAPHML_NS)
    if graph is None:
        return {"nodes": [], "edges": []}

    # 2. 节点：名称取 id（兜底取 entity_id），带类型/可疑标记/描述/来源 chunk
    nodes = [
        {
            "name": node.get("id") or _data_value(node, node_keys.get("entity_id")),
            "entity_type": _data_value(node, node_keys.get("entity_type")),
            "is_suspect": _is_suspect(_data_value(node, node_keys.get("entity_type"))),
            "description": _data_value(node, node_keys.get("description")),
            "source_id": _data_value(node, node_keys.get("source_id")),
        }
        for node in graph.findall("g:node", _GRAPHML_NS)
    ]

    # 3. 边：source/target 是 <edge> 的属性；描述/关键词/权重在 <data> 里
    edges = [
        {
            "src": edge.get("source") or "",
            "dst": edge.get("target") or "",
            "description": _data_value(edge, edge_keys.get("description")),
            "keywords": _data_value(edge, edge_keys.get("keywords")),
            "weight": _data_value(edge, edge_keys.get("weight")),
        }
        for edge in graph.findall("g:edge", _GRAPHML_NS)
    ]
    return {"nodes": nodes, "edges": edges}


def _unavailable(library_id: str | None, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """统一的"不可用"返回：显式标注，不静默空白。"""
    payload = {
        "available": False,
        "reason": _REASON_NO_GRAPH,
        "library_id": _lib(library_id),
        "source_path": str(_graphml_path(library_id)),
    }
    if extra:
        payload.update(extra)
    return payload


def _compute_signals(nodes: list[dict], edges: list[dict]) -> dict[str, Any]:
    """
    计算只读质量信号：类型分布 / 可疑数 / 孤儿数 / 重名分组。

    Args:
        nodes: 标准化节点列表
        edges: 标准化边列表

    Returns:
        dict: by_type / suspect_count / orphan_count / dup_group_count / dup_groups(采样)
    """
    by_type: dict[str, int] = defaultdict(int)
    for node in nodes:
        by_type[node["entity_type"] or "(未标类型)"] += 1

    # 孤儿：名字没出现在任何边端点上的实体
    connected: set[str] = set()
    for edge in edges:
        connected.add(edge["src"])
        connected.add(edge["dst"])
    orphan_count = sum(1 for node in nodes if node["name"] not in connected)

    # 重名：规范化名相同且超过一个的实体分组
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[_norm_name(node["name"])].append(node["name"])
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    dup_sample = dict(list(dup_groups.items())[:_MAX_DUP_SAMPLE])

    return {
        "by_type": dict(by_type),
        "suspect_count": sum(1 for node in nodes if node["is_suspect"]),
        "orphan_count": orphan_count,
        "dup_group_count": len(dup_groups),
        "dup_groups": dup_sample,
    }


def read_overview(library_id: str | None = None) -> dict[str, Any]:
    """
    LightRAG 图概览：实体/关系总数 + 类型分布 + 可疑/孤儿/重名信号。

    Args:
        library_id: 资料库 ID，空表示默认库

    Returns:
        dict: available + 统计；图不可读时 available=false 并带 reason
    """
    graph = _parse_graph(library_id)
    if graph is None:
        return _unavailable(library_id, extra={"entity_total": 0, "relation_total": 0})
    nodes, edges = graph["nodes"], graph["edges"]
    signals = _compute_signals(nodes, edges)
    return {
        "available": True,
        "library_id": _lib(library_id),
        "source_path": str(_graphml_path(library_id)),
        "entity_total": len(nodes),
        "relation_total": len(edges),
        **signals,
    }


def read_entities(
    library_id: str | None = None,
    *,
    suspect: bool | None = None,
    entity_type: str | None = None,
    q: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """
    LightRAG 实体明细，支持按可疑/类型/关键词过滤与分页。

    Args:
        library_id: 资料库 ID
        suspect: True 仅看可疑(非学术类型)，False 仅看正常，None 不过滤
        entity_type: 仅看某类型(大小写不敏感)
        q: 名称或描述包含的关键词
        limit: 分页大小(<=0 表示不分页)
        offset: 分页偏移

    Returns:
        dict: available + total + items(本页)
    """
    graph = _parse_graph(library_id)
    if graph is None:
        return _unavailable(library_id, extra={"total": 0, "items": []})
    items = graph["nodes"]
    if suspect is not None:
        items = [n for n in items if n["is_suspect"] is suspect]
    if entity_type:
        target = entity_type.strip().lower()
        items = [n for n in items if n["entity_type"].lower() == target]
    if q:
        needle = q.strip().lower()
        items = [n for n in items if needle in n["name"].lower() or needle in n["description"].lower()]
    total = len(items)
    page = items[offset : offset + limit] if limit and limit > 0 else items
    return {"available": True, "library_id": _lib(library_id), "total": total, "items": page}


def read_relations(
    library_id: str | None = None,
    *,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """
    LightRAG 关系明细（源/目标/描述/关键词/权重），分页。

    Args:
        library_id: 资料库 ID
        limit: 分页大小(<=0 表示不分页)
        offset: 分页偏移

    Returns:
        dict: available + total + items(本页)
    """
    graph = _parse_graph(library_id)
    if graph is None:
        return _unavailable(library_id, extra={"total": 0, "items": []})
    items = graph["edges"]
    total = len(items)
    page = items[offset : offset + limit] if limit and limit > 0 else items
    return {"available": True, "library_id": _lib(library_id), "total": total, "items": page}


def _doc_status_path(library_id: str | None) -> Path:
    """返回某个 library 的文档摄入状态文件路径。"""
    return _working_dir(library_id) / _DOC_STATUS_FILE


def _doc_title(content_summary: str) -> str:
    """从 content_summary 的 markdown H1 抽论文标题；取不到返回空串。"""
    for line in (content_summary or "").splitlines():
        text = line.strip()
        if text.startswith("# ") and not text.startswith("## "):
            return text[2:].strip()
    return ""


def read_documents(library_id: str | None = None) -> dict[str, Any]:
    """
    LightRAG 文档/上传摄入状态：每篇论文是否成功进图 + 失败原因。

    这是"为什么图里东西少"的第一诊断面——很多时候不是抽取质量差，而是摄入/抽取失败。

    Args:
        library_id: 资料库 ID，空表示默认库

    Returns:
        dict: available + total + by_status + items(标题/状态/chunk数/长度/错误/来源)
    """
    path = _doc_status_path(library_id)
    base = {"library_id": _lib(library_id), "source_path": str(path), "total": 0, "by_status": {}, "items": []}
    if not path.exists():
        return {**base, "available": False, "reason": "文档状态文件不存在（该资料库可能尚未摄入）"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**base, "available": False, "reason": "文档状态文件无法解析"}
    if not isinstance(payload, dict):
        payload = {}

    # 1. 逐条整理每篇上传的摄入状态，并按 status 聚合计数
    by_status: dict[str, int] = defaultdict(int)
    items: list[dict[str, Any]] = []
    for paper_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        status = str(value.get("status") or "")
        by_status[status or "(unknown)"] += 1
        chunks = value.get("chunks_count")
        if chunks is None:
            chunks = len(value.get("chunks_list") or [])
        items.append(
            {
                "paper_id": paper_id,
                "title": _doc_title(str(value.get("content_summary") or "")),
                "status": status,
                "chunks_count": chunks,
                "content_length": value.get("content_length", 0),
                "error_msg": str(value.get("error_msg") or "")[:300],
                "file_path": str(value.get("file_path") or ""),
                "created_at": str(value.get("created_at") or ""),
            }
        )
    return {**base, "available": True, "total": len(items), "by_status": dict(by_status), "items": items}


def _text_chunks_path(library_id: str | None) -> Path:
    """返回某个 library 的文本 chunk 文件路径。"""
    return _working_dir(library_id) / _TEXT_CHUNKS_FILE


def read_chunks(
    library_id: str | None = None,
    *,
    full_doc_id: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """
    LightRAG 文本 chunk 明细：每个 chunk 的原文内容 + 归属论文 + 顺序 + token 数。

    这是 LightRAG "实际摄入了什么内容"的真相面——by_doc 覆盖全库，能看出哪些论文有
    chunk（可能比图/doc_status 多，暴露"有内容却没进图"的不一致）。

    Args:
        library_id: 资料库 ID
        full_doc_id: 仅看某篇论文（upload_*）的 chunk
        limit: 分页大小(<=0 不分页)
        offset: 分页偏移

    Returns:
        dict: available + total(过滤后) + doc_count + by_doc(全库) + items(本页, 按论文+顺序排)
    """
    path = _text_chunks_path(library_id)
    base = {"library_id": _lib(library_id), "source_path": str(path), "total": 0, "doc_count": 0, "by_doc": {}, "items": []}
    if not path.exists():
        return {**base, "available": False, "reason": "chunk 文件不存在（该资料库可能尚未摄入）"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**base, "available": False, "reason": "chunk 文件无法解析"}
    if not isinstance(payload, dict):
        payload = {}

    # 1. 逐个 chunk 整理内容+归属论文，by_doc 统计全库每篇论文的 chunk 数
    by_doc: dict[str, int] = defaultdict(int)
    items: list[dict[str, Any]] = []
    for chunk_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        doc_id = str(value.get("full_doc_id") or "")
        by_doc[doc_id or "(unknown)"] += 1
        items.append(
            {
                "chunk_id": chunk_id,
                "full_doc_id": doc_id,
                "chunk_order_index": value.get("chunk_order_index", 0),
                "tokens": value.get("tokens", 0),
                "content": str(value.get("content") or ""),
            }
        )

    # 2. 按论文 + chunk 顺序排，便于按原文顺序阅读
    items.sort(key=lambda c: (c["full_doc_id"], c["chunk_order_index"]))
    if full_doc_id:
        items = [c for c in items if c["full_doc_id"] == full_doc_id]
    total = len(items)
    page = items[offset : offset + limit] if limit and limit > 0 else items
    return {**base, "available": True, "total": total, "doc_count": len(by_doc), "by_doc": dict(by_doc), "items": page}
