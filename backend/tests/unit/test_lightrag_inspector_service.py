"""图谱数据诊断查看器 · LightRAG 只读服务单测。

用临时 graphml 夹具验证：类型计数 / 可疑筛选 / 类型+关键词过滤 / 关系读取 / 文件缺失降级。
不依赖真实 LightRAG 运行时，仅替换数据根目录。
"""

from __future__ import annotations

import json
from pathlib import Path

import app.services.lightrag_inspector_service as inspector

# 4 节点(method×2 / dataset×1 / person×1) + 1 边(STGformer→LargeST)
_GRAPHML = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
<key id="d0" for="node" attr.name="entity_id" attr.type="string"/>
<key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
<key id="d2" for="node" attr.name="description" attr.type="string"/>
<key id="d7" for="edge" attr.name="weight" attr.type="double"/>
<key id="d8" for="edge" attr.name="description" attr.type="string"/>
<key id="d9" for="edge" attr.name="keywords" attr.type="string"/>
<graph edgedefault="undirected">
<node id="STGformer"><data key="d0">STGformer</data><data key="d1">method</data><data key="d2">时空图Transformer</data></node>
<node id="LargeST"><data key="d0">LargeST</data><data key="d1">dataset</data><data key="d2">大规模交通数据集</data></node>
<node id="SomePerson"><data key="d0">SomePerson</data><data key="d1">person</data><data key="d2">非学术类型</data></node>
<node id="Orphan"><data key="d0">Orphan</data><data key="d1">method</data><data key="d2">孤儿节点</data></node>
<edge source="STGformer" target="LargeST"><data key="d7">1.0</data><data key="d8">在该数据集上评测</data><data key="d9">benchmark</data></edge>
</graph></graphml>
"""


def _seed(tmp_path: Path, monkeypatch) -> None:
    """把临时目录写成默认库的 lightrag_index 并把服务数据根指过去。"""
    idx = tmp_path / "lightrag_index"
    idx.mkdir()
    (idx / "graph_chunk_entity_relation.graphml").write_text(_GRAPHML, encoding="utf-8")
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)


def test_overview_counts_types_and_signals(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    ov = inspector.read_overview()
    assert ov["available"] is True
    assert ov["entity_total"] == 4
    assert ov["relation_total"] == 1
    assert ov["by_type"] == {"method": 2, "dataset": 1, "person": 1}
    assert ov["suspect_count"] == 1            # person 非学术类型
    assert ov["orphan_count"] == 2             # SomePerson / Orphan 不在任何边上
    assert ov["dup_group_count"] == 0


def test_entities_suspect_filter(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    res = inspector.read_entities(suspect=True)
    assert res["total"] == 1
    assert res["items"][0]["name"] == "SomePerson"
    assert res["items"][0]["is_suspect"] is True


def test_entities_type_and_query_filter(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert inspector.read_entities(entity_type="method")["total"] == 2
    assert inspector.read_entities(q="交通")["total"] == 1   # 仅 LargeST 描述含"交通"


def test_relations_read(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    rel = inspector.read_relations()
    assert rel["available"] is True
    assert rel["total"] == 1
    item = rel["items"][0]
    assert item["src"] == "STGformer" and item["dst"] == "LargeST"
    assert item["keywords"] == "benchmark"


def test_missing_graph_is_unavailable(tmp_path, monkeypatch):
    # 不写 graphml，仅指向空目录：应显式不可用，不静默空白
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)
    ov = inspector.read_overview()
    assert ov["available"] is False
    assert ov["reason"]
    ent = inspector.read_entities()
    assert ent["available"] is False and ent["items"] == []


# 文档摄入状态：1 篇 processed + 1 篇 failed（524 超时）
_DOC_STATUS = {
    "upload_ok": {
        "status": "processed",
        "chunks_count": 1,
        "content_length": 6051,
        "content_summary": "[Library ID: default]\n[Paper ID: upload_ok]\n\n# STGformer: Efficient Transformer\n## Abstract\n...",
    },
    "upload_bad": {
        "status": "failed",
        "chunks_list": [],
        "content_length": 6051,
        "error_msg": "C[1/2]: 524 Server Error: A timeout occurred",
        "content_summary": "[Library ID: default]\n\n# LSTTN: long-short term\n## Abstract\n...",
    },
}


def _seed_docs(tmp_path, monkeypatch):
    """把临时目录写成默认库并放入文档状态文件。"""
    idx = tmp_path / "lightrag_index"
    idx.mkdir(exist_ok=True)
    (idx / "kv_store_doc_status.json").write_text(json.dumps(_DOC_STATUS, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)


def test_documents_status_and_title(tmp_path, monkeypatch):
    _seed_docs(tmp_path, monkeypatch)
    docs = inspector.read_documents()
    assert docs["available"] is True
    assert docs["total"] == 2
    assert docs["by_status"] == {"processed": 1, "failed": 1}
    titles = {d["paper_id"]: d["title"] for d in docs["items"]}
    assert titles["upload_ok"] == "STGformer: Efficient Transformer"
    bad = next(d for d in docs["items"] if d["paper_id"] == "upload_bad")
    assert bad["status"] == "failed"
    assert "524" in bad["error_msg"]


def test_documents_missing_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)
    docs = inspector.read_documents()
    assert docs["available"] is False


# 文本 chunk：upload_x 两段（乱序）+ upload_y 一段，验证分组/排序/过滤
_TEXT_CHUNKS = {
    "chunk-b": {"tokens": 468, "content": "second part", "chunk_order_index": 1, "full_doc_id": "upload_x"},
    "chunk-a": {"tokens": 1200, "content": "first part", "chunk_order_index": 0, "full_doc_id": "upload_x"},
    "chunk-c": {"tokens": 900, "content": "other paper", "chunk_order_index": 0, "full_doc_id": "upload_y"},
}


def test_chunks_grouped_sorted_and_filtered(tmp_path, monkeypatch):
    idx = tmp_path / "lightrag_index"
    idx.mkdir(exist_ok=True)
    (idx / "kv_store_text_chunks.json").write_text(json.dumps(_TEXT_CHUNKS, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)

    res = inspector.read_chunks()
    assert res["available"] is True
    assert res["total"] == 3
    assert res["doc_count"] == 2
    assert res["by_doc"] == {"upload_x": 2, "upload_y": 1}
    # 同一论文按 chunk_order 升序
    xs = [c for c in res["items"] if c["full_doc_id"] == "upload_x"]
    assert [c["chunk_order_index"] for c in xs] == [0, 1]
    # 按 full_doc_id 过滤
    only_y = inspector.read_chunks(full_doc_id="upload_y")
    assert only_y["total"] == 1 and only_y["items"][0]["content"] == "other paper"


def test_chunks_missing_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector, "_data_root", lambda: tmp_path)
    res = inspector.read_chunks()
    assert res["available"] is False
