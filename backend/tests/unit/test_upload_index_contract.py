"""索引完整集合与真实正文边界回归，作者：xxx。"""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.models.upload_models import ArtifactIntegrityError
from app.services.ingest.document_analysis_service import build_upload_paper
from app.services.ingest.upload_index_service import build_index_rows, normalize_vectors, verify_index_rows, version_chunks


def _rows():
    """构造包含完整正文、短预览和确定向量的真实索引载荷。"""
    chunks = [{"chunk_id": "source-0", "parent_id": "parent-0", "text": "原文" * 9000,
        "full_text": "完整父块" * 20000, "section": "Method", "page_num": 1, "parser_mode": "local"}]
    versioned = version_chunks(chunks, batch_id="a" * 32, paper_id="paper")
    rows = build_index_rows(versioned, [[0.25, 0.5]], metadata={"year": 2026})
    return versioned, rows


@pytest.mark.parametrize("field,value", [("chunk_id", "wrong-id"), ("display_text", "wrong text"), ("dense_vector", [0.25, 0.75])])
def test_same_count_wrong_id_text_or_vector_never_passes(field, value):
    """数量相同但 ID、正文或向量不一致必须拒绝发布。"""
    _, expected = _rows()
    actual = deepcopy(expected)
    actual[0][field] = value
    with pytest.raises(ArtifactIntegrityError):
        verify_index_rows(expected, actual)


def test_index_preview_limits_do_not_truncate_immutable_content():
    """索引字段按 UTF-8 大小受限，而引用清单仍保留完整父块。"""
    chunks, rows = _rows()
    assert len(rows[0]["embedding_text"].encode("utf-8")) <= 8192
    assert len(rows[0]["full_text"].encode("utf-8")) <= 65535
    assert chunks[0]["full_text"] == "完整父块" * 20000
    assert verify_index_rows(rows, deepcopy(rows))


@pytest.mark.parametrize("vectors", [[[float("nan"), 0.0]], [[1.0]], []])
def test_incomplete_or_invalid_vectors_fail(vectors):
    """向量数量、维度和数值必须同时有效。"""
    with pytest.raises(ArtifactIntegrityError):
        normalize_vectors(vectors, count=1, dimension=2)


def test_missing_abstract_is_not_fabricated_from_document_prefix():
    """没有识别出的摘要时保持空值，全文片段不会冒充论文摘要。"""
    paper = build_upload_paper("source.pdf", {"metadata": {}, "full_text": "not an abstract", "file_path": "source.pdf"}, "paper")
    assert paper["abstract"] == ""
