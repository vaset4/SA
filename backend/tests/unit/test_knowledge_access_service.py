"""统一知识正文读取的离线契约回归，作者：xxx。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.models.knowledge_access_contracts import KnowledgeReadBudgetError, KnowledgeReadRequest
from app.models.research_contracts import RetrievalScopeError
from app.models.upload_models import ArtifactIntegrityError
from app.services import knowledge_access_service as service, upload_read_service
from app.services.ingest.upload_artifacts import write_artifact, write_json_artifact
from source_file_test_ports import install_file_ports

_BATCH = "a" * 32


def _item(key, text, *, index=0, block_type="text", page_num=1):
    """参数为项目身份、正文及真实结构位置；返回测试用规范来源项目。"""
    return {"source_item": key, "text": text, "source_kind": "markdown", "source_index": index,
            "block_type": block_type, "page_num": page_num}


def _chunk(index, text, *, parent="p0", start=0, source="s0", path="Paper / 1 Method", page=1):
    """参数为发布顺序、文本、来源区间与父块；返回有完整来源映射的发布块。"""
    fragment = {"source_item": source, "source_start": start, "source_end": start + len(text),
                "text_start": 0, "text_end": len(text), "page_num": page}
    return {"chunk_id": f"v3:{_BATCH}:{index}", "parent_id": parent, "batch_id": _BATCH,
            "paper_id": "paper", "text": text, "display_text": text, "full_text": text,
            "section_path": path, "section": path.rsplit(" / ", 1)[-1], "page_num": page or 0,
            "block_type": "text", "source_fragments": [fragment], "parent_source_fragments": [fragment]}


@pytest.fixture
def publication(tmp_path, monkeypatch):
    """参数为临时目录与替换端口；返回真实产物读写和隔离授权仓储，绝不访问业务库。"""
    install_file_ports(monkeypatch)
    snapshot = {"user_id": "owner", "lease_id": "lease", "batch_ids": [_BATCH],
                "paper_batches": {"paper": _BATCH}}
    repository = SimpleNamespace(content_reference=AsyncMock(), renew_snapshot=AsyncMock())
    monkeypatch.setattr(upload_read_service, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path)))
    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository", lambda: repository)
    monkeypatch.setattr(upload_read_service.paper_db, "_run", asyncio.run)
    return {"root": tmp_path, "repository": repository, "snapshot": snapshot}


def _publish(publication, chunks, items, *, legacy=False):
    """参数为隔离发布上下文、块与来源表；保存真实 SHA 产物，返回正式服务读取请求。"""
    source_ref = write_artifact(publication["root"], b"original-source", suffix=".md")
    manifest = {"format": 3, "paper_id": "paper", "batch_id": _BATCH, "user_id": "owner", "chunks": chunks}
    if not legacy:
        manifest.update(content_schema_version=1, source_items={
            key: {**item, "artifact_ref": source_ref} for key, item in items.items()
        })
    reference = write_json_artifact(publication["root"], manifest)
    publication["repository"].content_reference.return_value = {"paper_id": "paper", "manifest_ref": reference}
    publication["manifest"] = manifest
    publication["source_ref"] = source_ref
    publication["manifest_ref"] = reference
    scope = {"user_id": "owner", "paper_ids": ["paper"], "publication": publication["snapshot"]}
    return KnowledgeReadRequest(operation="chunks", scope=scope, paper_ids=("paper",),
                                chunk_ids=tuple(chunk["chunk_id"] for chunk in chunks))


def _read(request):
    """参数为正式读取请求；运行异步知识入口并返回真实材料结果。"""
    return asyncio.run(service.read_knowledge(request))


def test_multiple_children_load_manifest_once_and_preserve_authoritative_text(publication, monkeypatch):
    """同父多个命中共享一次清单核验，子块读取不使用索引预览或父正文替代。"""
    text = "甲😀乙丙丁" * 2000
    first = _chunk(0, text[:5000])
    second = _chunk(1, text[4000:], start=4000)
    for chunk in (first, second):
        chunk["full_text"] = text
        chunk["parent_source_fragments"] = _chunk(0, text)["parent_source_fragments"]
    request = _publish(publication, [first, second], {"s0": _item("s0", text)})
    load = Mock(wraps=upload_read_service.read_json_artifact)
    monkeypatch.setattr(upload_read_service, "read_json_artifact", load)
    result = _read(request)
    assert len(result.materials) == 1
    assert result.materials[0].text == text
    assert result.read_report.used_chars == len(text)
    assert load.call_count == 1
    publication["repository"].content_reference.assert_awaited_once_with(_BATCH, user_id="owner", connection="isolated-file-connection")
    # 1. 进入读取与等待文件锁后分别核验同一快照，清单仍只加载一次
    assert publication["repository"].renew_snapshot.await_args_list == [
        call(publication["snapshot"]), call(publication["snapshot"])]


def test_parent_read_uses_full_text_and_keeps_all_page_fragments(publication):
    """父块跨页时所有来源片段保留，不能把首子块页号显示为整父块位置。"""
    first, second = "第一页正文", "第二页正文"
    chunk = _chunk(0, first)
    full = first + "\n\n" + second
    second_fragment = {**_chunk(1, second, source="s1", page=2)["source_fragments"][0],
                       "text_start": len(first) + 2, "text_end": len(full)}
    chunk.update(full_text=full, parent_source_fragments=[*chunk["source_fragments"], second_fragment])
    request = _publish(publication, [chunk], {"s0": _item("s0", first), "s1": _item("s1", second, index=1, page_num=2)})
    child = _read(request).materials[0]
    parent = _read(replace(request, granularity="parent")).materials[0]
    assert child.text == first
    assert parent.text == full and parent.page_num == 0
    assert {span["source_fragments"][0]["page_num"] for span in parent.source_spans} == {1, 2}
    assert child.material_group_id == parent.material_group_id


def test_sibling_hits_use_same_published_parent_anchor_and_evidence_identity(publication):
    """同父兄弟命中读取相同父正文时固定使用发布首锚点，证据身份不随本轮命中变化。"""
    from app.services.internal_research.evidence_identity import make_read_evidence_id

    text = "First child. Second child. Third child."
    positions = (0, 13, 27, len(text))
    chunks = [_chunk(index, text[start:end], start=start)
              for index, (start, end) in enumerate(zip(positions, positions[1:]))]
    for chunk in chunks:
        chunk.update(full_text=text, parent_source_fragments=_chunk(0, text)["source_fragments"])
    request = _publish(publication, chunks, {"s0": _item("s0", text)})
    first = _read(replace(request, chunk_ids=(chunks[1]["chunk_id"],), granularity="parent")).materials[0]
    second = _read(replace(request, chunk_ids=(chunks[2]["chunk_id"],), granularity="parent")).materials[0]
    assert first.text == second.text == text
    assert first.source_spans == second.source_spans
    assert first.chunk_id == second.chunk_id == chunks[0]["chunk_id"]
    identities = {make_read_evidence_id(paper_id=material.paper_id, material_group_id=material.material_group_id,
        source_spans=material.source_spans, text=material.text) for material in (first, second)}
    assert len(identities) == 1
    child = _read(replace(request, chunk_ids=(chunks[2]["chunk_id"],), granularity="child")).materials[0]
    assert child.chunk_id == chunks[2]["chunk_id"] and child.text == chunks[2]["text"]


def test_section_removes_parent_overlap_by_source_interval_not_repeated_text(publication):
    """相同句子在不同原始位置仍保留，只去掉已覆盖的来源区间。"""
    text = "重复句子。重复句子。结尾。"
    first = _chunk(0, text[:10])
    second = _chunk(1, text[5:], parent="p1", start=5)
    request = _publish(publication, [first, second], {"s0": _item("s0", text)})
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("1 Method",))
    result = _read(request)
    assert result.materials[0].text == text
    spans = result.materials[0].source_spans
    assert spans[1]["start"] == 5
    assert spans[1]["source_fragments"][0]["source_start"] == 10
    assert result.read_report.status == "complete"


def test_partial_table_budget_retains_prior_material_and_remaining_location(publication):
    """文字之后的大表格超额度时保留已读文字，并报告整表未读。"""
    intro, table = "已读正文。", "| 列一 | 列二 |\n|---|---|\n| 甲 | 乙 |"
    chunks = [_chunk(0, intro), _chunk(1, table, parent="p1", source="s1")]
    items = {"s0": _item("s0", intro), "s1": _item("s1", table, index=1, block_type="table")}
    request = _publish(publication, chunks, items)
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("1",), max_chars=len(intro) + 8)
    result = _read(request)
    assert result.materials[0].text == intro
    assert result.read_report.status == "partial"
    assert result.read_report.remaining[0]["reason"] == "atomic_structure_budget"
    assert result.read_report.remaining[0]["minimum_chars"] == len(table)


@pytest.mark.parametrize("block_type", ["table", "equation", "formula"])
def test_first_atomic_structure_too_large_is_explicit_error(publication, block_type):
    """首个原子结构装不下时明确失败，不能把任意半张表或半个公式作为成功。"""
    text = "不可拆分结构" * 20
    request = _publish(publication, [_chunk(0, text)], {"s0": _item("s0", text, block_type=block_type)})
    with pytest.raises(KnowledgeReadBudgetError) as error:
        _read(replace(request, max_chars=10))
    assert error.value.minimum_chars == len(text)
    assert error.value.remaining[0]["chunk_id"] == request.chunk_ids[0]


def test_prose_partial_uses_real_unicode_ranges(publication):
    """非原子正文按自然边界部分读取，所有字段区间按字符而非 UTF-8 字节计算。"""
    text = "😀第一句。 第二句还有正文。"
    request = _publish(publication, [_chunk(0, text)], {"s0": _item("s0", text)})
    result = _read(replace(request, max_chars=8))
    material = result.materials[0]
    assert material.text == "😀第一句。 "
    assert result.read_report.status == "partial"
    span = material.source_spans[0]
    assert text[span["start"]:span["end"]] == material.text[span["text_start"]:span["text_end"]]
    assert span["source_fragments"][0]["source_end"] == len(material.text)


@pytest.mark.parametrize("formula", ["$$x = a. b + c$$", "$x = a. b + c$"])
@pytest.mark.parametrize("legacy", [False, True])
def test_paragraph_formula_first_requires_space_for_whole_expression(publication, formula, legacy):
    """段落类型中的首个公式放不下时明确报错，新旧资料均不能返回半个公式。"""
    text = formula + " suffix text."
    request = _publish(publication, [_chunk(0, text)], {"s0": _item("s0", text)}, legacy=legacy)
    with pytest.raises(KnowledgeReadBudgetError) as error:
        _read(replace(request, max_chars=len(formula) - 1))
    assert error.value.minimum_chars == len(formula)
    remaining = error.value.remaining[0]
    assert remaining["reason"] == "atomic_structure_budget"
    assert remaining["start"] == 0 and remaining["end"] == len(formula)


@pytest.mark.parametrize("formula", ["$$x = a. b + c$$", "$x = a. b + c$"])
def test_paragraph_prefix_survives_unread_formula_budget(publication, formula):
    """公式前已读正文保留，未读公式独立报告准确位置和最低空间。"""
    prefix = "Read introduction. "
    text = prefix + formula + " suffix text."
    request = _publish(publication, [_chunk(0, text)], {"s0": _item("s0", text)})
    result = _read(replace(request, max_chars=len(prefix) + len(formula) - 1))
    assert result.materials[0].text == prefix
    assert result.read_report.status == "partial"
    remaining = result.read_report.remaining[0]
    assert remaining["reason"] == "atomic_structure_budget"
    assert remaining["minimum_chars"] == len(formula)
    assert remaining["start"] == len(prefix)
    assert remaining["end"] == len(prefix) + len(formula)


@pytest.mark.parametrize("formula", ["$$x = a. b + c$$", "$x = a. b + c$"])
def test_complete_formula_is_kept_when_budget_stops_in_suffix(publication, formula):
    """公式已完整读取后可停止在后续正文，不能把整段当原子结构拒绝。"""
    prefix = "Intro. "
    text = prefix + formula + " trailing further text."
    request = _publish(publication, [_chunk(0, text)], {"s0": _item("s0", text)})
    result = _read(replace(request, max_chars=len(prefix + formula) + 10))
    assert result.materials[0].text.startswith(prefix + formula)
    assert len(result.materials[0].text) > len(prefix + formula)
    assert result.read_report.status == "partial"
    assert result.read_report.remaining[0]["reason"] == "read_budget"


def test_one_formula_across_source_fragments_has_one_atomic_budget(publication):
    """同一公式跨规范来源项目时仍整体读取，剩余定位保存两个真实跨度。"""
    prefix, left, right = "Intro. ", "$$x = a.", " b + c$$"
    text = prefix + left + right + " suffix."
    first = _chunk(0, prefix + left)["source_fragments"][0]
    second_text = right + " suffix."
    second = {**_chunk(1, second_text, source="s1")["source_fragments"][0],
              "text_start": len(prefix + left), "text_end": len(text)}
    chunk = _chunk(0, text)
    chunk.update(source_fragments=[first, second], parent_source_fragments=[first, second])
    request = _publish(publication, [chunk], {"s0": _item("s0", prefix + left),
        "s1": _item("s1", second_text, index=1)})
    result = _read(replace(request, max_chars=len(prefix + left + right) - 1))
    assert result.materials[0].text == prefix
    remaining = result.read_report.remaining[0]
    assert remaining["minimum_chars"] == len(left + right)
    assert len(remaining["spans"]) == 2
    assert _read(replace(request, max_chars=len(prefix + left + right))).materials[0].text == prefix + left + right


def test_one_formula_across_published_chunks_keeps_each_unread_address(publication):
    """公式跨块时不能把第二块结束坐标拼到首块地址，未读范围须分别可回读。"""
    formula = "$x = a. b + c$"
    split = 7
    chunks = [_chunk(0, formula[:split]), _chunk(1, formula[split:], parent="p1", start=split)]
    request = _publish(publication, chunks, {"s0": _item("s0", formula)})
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("1",), max_chars=len(formula) - 1)
    with pytest.raises(KnowledgeReadBudgetError) as error:
        _read(request)
    assert error.value.minimum_chars == len(formula)
    spans = error.value.remaining[0]["spans"]
    assert [span["chunk_id"] for span in spans] == [chunk["chunk_id"] for chunk in chunks]
    assert [chunk["text"][span["start"]:span["end"]] for chunk, span in zip(chunks, spans)] == [formula[:split], formula[split:]]


def test_atomic_table_caption_and_body_share_budget_unit(publication):
    """表格标题来自独立来源项目时仍与表正文整体读取，不能只返回标题假装读到表格。"""
    caption, table = "表 1 实验结果", "| Method | Score |\n|---|---|\n| A | 0.9 |"
    chunk = _chunk(0, caption)
    text = caption + "\n\n" + table
    body = {**_chunk(1, table, source="s1")["source_fragments"][0],
            "text_start": len(caption) + 2, "text_end": len(text)}
    fragments = [*chunk["source_fragments"], body]
    chunk.update(text=text, full_text=text, display_text=text, block_type="table",
                 source_fragments=fragments, parent_source_fragments=fragments)
    request = _publish(publication, [chunk], {"s0": _item("s0", caption),
                                            "s1": _item("s1", table, index=1, block_type="table")})
    with pytest.raises(KnowledgeReadBudgetError) as error:
        _read(replace(request, max_chars=len(caption) + 3))
    assert error.value.minimum_chars == len(text)
    assert error.value.remaining[0]["minimum_chars"] == len(text)
    assert _read(replace(request, max_chars=len(text))).materials[0].text == text


def test_legacy_read_retains_text_and_exposes_overlap_limit(publication):
    """旧发布内容可读且不伪造来源；重复文字保留并明确报告未做精确消重。"""
    chunks = [_chunk(0, "旧表 A B"), _chunk(1, "B C", parent="p1")]
    request = _publish(publication, chunks, {}, legacy=True)
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("1",))
    result = _read(request)
    assert result.materials[0].text == "旧表 A B\n\nB C"
    assert result.materials[0].source_location_status == "legacy_chunk"
    assert result.read_report.limitations
    assert all("source_fragments" not in span for span in result.materials[0].source_spans)


def test_markdown_without_physical_page_does_not_claim_page_one(publication):
    """没有物理页的来源保留字符定位，但不猜测第一页。"""
    request = _publish(publication, [_chunk(0, "正文", page=None)], {"s0": _item("s0", "正文", page_num=None)})
    material = _read(request).materials[0]
    assert material.source_location_status == "no_physical_location"
    assert material.page_num == 0


@pytest.mark.parametrize("original_range", [
    {"source_start": 0}, {"source_end": 2},
    {"source_start": -1, "source_end": 2}, {"source_start": 2, "source_end": 2},
    {"source_start": 3, "source_end": 2}, {"source_start": True, "source_end": 2},
    {"source_start": 0, "source_end": 2.0}, {"source_start": None, "source_end": 2},
])
def test_original_artifact_ranges_reject_incomplete_or_invalid_coordinates(publication, original_range):
    """参数为发布替身和非法原始区间；正式读取必须拒绝缺半边、非整数和倒序坐标。"""
    item = {**_item("s0", "正文"), **original_range}
    request = _publish(publication, [_chunk(0, "正文")], {"s0": item})
    with pytest.raises(ArtifactIntegrityError, match="原始.*区间"):
        _read(request)


def test_original_artifact_range_uses_raw_text_coordinates(publication):
    """参数为发布替身；原始产物坐标可长于规范正文，不能误用清洗后的正文长度校验。"""
    item = {**_item("s0", "正文"), "source_start": 2, "source_end": 10}
    request = _publish(publication, [_chunk(0, "正文")], {"s0": item})
    assert _read(request).materials[0].text == "正文"


@pytest.mark.parametrize("region", [{}, {"bbox": [0, 0, 10, 10]}, "page-box"])
def test_undefined_region_contract_is_rejected_before_returning_body(publication, region):
    """参数为发布替身和未定义区域；没有坐标口径的区域不能被当作已核验物理定位。"""
    chunk = _chunk(0, "正文")
    for key in ("source_fragments", "parent_source_fragments"):
        chunk[key][0]["region"] = region
    request = _publish(publication, [chunk], {"s0": {**_item("s0", "正文"), "region": region}})
    with pytest.raises(ArtifactIntegrityError, match="区域"):
        _read(request)


def test_explicitly_absent_region_keeps_valid_page_location(publication):
    """参数为发布替身；区域缺省或显式为空均不影响已保存的实际页码。"""
    chunk = _chunk(0, "正文")
    chunk["source_fragments"][0]["region"] = None
    request = _publish(publication, [chunk], {"s0": {**_item("s0", "正文"), "region": None}})
    material = _read(request).materials[0]
    assert material.page_num == 1 and material.source_location_status == "exact"


@pytest.mark.parametrize("page_num", [True, 1.0, "1"])
def test_fragment_page_number_requires_an_integer(publication, page_num):
    """参数为发布替身和非法页号；即使值与来源页号比较相等，也不能绕过整数页码合同。"""
    chunk = _chunk(0, "正文")
    for key in ("source_fragments", "parent_source_fragments"):
        chunk[key][0]["page_num"] = page_num
    request = _publish(publication, [chunk], {"s0": _item("s0", "正文")})
    with pytest.raises(ArtifactIntegrityError):
        _read(request)


def test_historical_v3_citation_reads_saved_legacy_body_without_new_location(publication):
    """参数为发布替身；历史引用仍读取当时保存正文，不申请当前快照或补造新版来源字段。"""
    chunk = _chunk(0, "历史正文")
    for key in ("source_fragments", "parent_source_fragments"):
        del chunk[key]
    request = _publish(publication, [chunk], {}, legacy=True)
    assert upload_read_service.read_published_chunk(request.chunk_ids[0], user_id="owner") == chunk
    publication["repository"].renew_snapshot.assert_not_called()


@pytest.mark.parametrize("mutation", ["missing_fragments", "missing_display", "wrong_text", "outside", "missing_items", "unknown_version"])
def test_malformed_new_schema_never_falls_back_to_legacy(publication, mutation):
    """新版产物缺字段、错范围或内容不一致时全部明确失败。"""
    request = _publish(publication, [_chunk(0, "正文")], {"s0": _item("s0", "正文")})
    manifest = deepcopy(publication["manifest"])
    if mutation == "missing_fragments":
        del manifest["chunks"][0]["source_fragments"]
    elif mutation == "missing_display":
        del manifest["chunks"][0]["display_text"]
    elif mutation == "wrong_text":
        manifest["source_items"]["s0"]["text"] = "错误"
    elif mutation == "outside":
        manifest["chunks"][0]["source_fragments"][0]["source_end"] = 9
    elif mutation == "missing_items":
        del manifest["source_items"]
    else:
        manifest["content_schema_version"] = 2
    reference = write_json_artifact(publication["root"], manifest)
    publication["repository"].content_reference.return_value = {"manifest_ref": reference}
    with pytest.raises(ArtifactIntegrityError):
        _read(request)


@pytest.mark.parametrize("target", ["manifest", "source"])
def test_corrupted_artifact_fails_without_another_version(publication, target):
    """原始产物或发布清单损坏时失败，不回到索引预览或当前批次。"""
    request = _publish(publication, [_chunk(0, "正文")], {"s0": _item("s0", "正文")})
    Path(publication[f"{target}_ref"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ArtifactIntegrityError):
        _read(request)


@pytest.mark.parametrize("kind", ["missing_snapshot", "foreign_owner", "foreign_paper", "foreign_chunk"])
def test_scope_violations_fail_explicitly(publication, kind):
    """缺快照、账号不符或目标越权明确失败，不跨版本搜索或读取。"""
    request = _publish(publication, [_chunk(0, "正文")], {"s0": _item("s0", "正文")})
    scope = deepcopy(dict(request.scope))
    if kind == "missing_snapshot":
        del scope["publication"]
    elif kind == "foreign_owner":
        scope["publication"]["user_id"] = "other"
    elif kind == "foreign_paper":
        request = replace(request, paper_ids=("foreign",))
    else:
        request = replace(request, chunk_ids=("v3:" + "b" * 32 + ":0",))
    with pytest.raises(RetrievalScopeError):
        _read(replace(request, scope=scope))


def test_sections_preserve_requested_order_and_ambiguity(publication):
    """完整路径按请求顺序读取，短标题歧义和未找到分别保留定位结果。"""
    paths = ("Paper / 1 Method", "Paper / 2 Results", "Paper / 3 Method")
    chunks = [_chunk(i, str(i), parent=f"p{i}", source=f"s{i}", path=path) for i, path in enumerate(paths)]
    items = {f"s{i}": _item(f"s{i}", str(i), index=i) for i in range(3)}
    request = _publish(publication, chunks, items)
    request = replace(request, operation="sections", chunk_ids=(), section_paths=(paths[1], "Method", paths[0], "missing"))
    result = _read(request)
    assert [material.text for material in result.materials] == ["1", "0"]
    assert [row["reason"] for row in result.read_report.remaining] == ["ambiguous", "not_found"]
    assert result.read_report.status == "partial"


def test_authorized_unpublished_paper_remains_unavailable_without_loading_current(publication):
    """授权但不在本轮发布快照的论文保留不可用原因，不能临时查当前版本。"""
    request = _publish(publication, [_chunk(0, "正文")], {"s0": _item("s0", "正文")})
    scope = {**request.scope, "paper_ids": ["unpublished"]}
    request = replace(request, scope=scope, paper_ids=("unpublished",), operation="sections", chunk_ids=(), section_paths=("1",))
    result = _read(request)
    assert result.materials == () and result.read_report.status == "unavailable"
    assert result.read_report.remaining[0]["reason"] == "not_in_publication_snapshot"
    publication["repository"].content_reference.assert_not_called()


def test_group_identity_includes_paper_batch_kind_and_locator():
    """组身份必须隔离不同论文、批次和分组类型，实际读取范围不参与组键。"""
    base = {"paper_id": "paper", "batch_id": _BATCH, "group_kind": "section", "group_locator": "Method"}
    identities = {service.material_group_id(**base)}
    for key in base:
        identities.add(service.material_group_id(**{**base, key: "other"}))
    assert len(identities) == 5


def test_cancelled_read_waits_for_owned_file_work_before_return(publication):
    """取消不会遗留脱离请求的后台文件读取，外层清理可在真正结束后进行。"""
    request = _publish(publication, [_chunk(0, "正文")], {"s0": _item("s0", "正文")})
    started, finish, ended = threading.Event(), threading.Event(), threading.Event()

    def reader(*args, **kwargs):
        """参数为发布读取参数；用事件模拟请求持有的同步 IO，返回已保存清单。"""
        started.set()
        assert finish.wait(5)
        ended.set()
        return {_BATCH: publication["manifest"]}

    async def exercise():
        """无参数；取消进行中的正式入口，验证任务等待线程结束后才传播取消。"""
        task = asyncio.create_task(service.read_knowledge(request, publication_reader=reader))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ended.is_set()

    asyncio.run(exercise())


class _WordCounter:
    """离线词计数端口，用固定小块触发真实父子 overlap。"""

    def count(self, text):
        """参数为正文；返回空白分词数量，不加载 tokenizer 或模型。"""
        return len(text.split())


async def _run_local_stage(stage, action, *, inputs, **kwargs):
    """参数为上传阶段、动作和既有输入；仅运行传入本地动作，不访问数据库检查点。"""
    return await action()


def test_real_parse_index_manifest_and_knowledge_read_share_source_contract(publication, monkeypatch):
    """正式解析、批次存档与正文读取贯通，真实切分不丢字、不重复并保留表格行列。"""
    from app.services import document_parser as parser
    from app.services.ingest.upload_index_service import UploadBatchIndexer
    from app.services.ingest.upload_parse_service import UploadDocumentParser

    monkeypatch.setattr(parser, "_get_token_counter", _WordCounter)
    for key, value in {"structure_merge_enabled": True, "chunk_merge_floor": 5,
                       "parent_chunk_max_size": 14, "parent_chunk_overlap_tokens": 2,
                       "child_chunk_target_size": 5, "child_chunk_overlap_tokens": 1,
                       "embedding_dimension": 2}.items():
        monkeypatch.setattr(parser.settings, key, value, raising=False)
    body = "alpha beta gamma delta. epsilon zeta eta theta. " * 6
    body = body.rstrip()
    markdown = "# 1 Method\n\n" + body + "\n\n# 2 Results\n\nTable 1 Results\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    root = publication["root"]
    reference = write_artifact(root / "source", markdown.encode("utf-8"), suffix=".md")
    runner = SimpleNamespace(execution=SimpleNamespace(paper_id="paper", batch_id=_BATCH, parse_id="parse", user_id="owner"),
                             run=_run_local_stage, repository=SimpleNamespace(check_execution=AsyncMock()))
    uploader = UploadDocumentParser(runner=runner, client=SimpleNamespace(base_url="offline"),
                                     source=reference, filename="paper.md", root=root)
    parsed = asyncio.run(uploader.parse())
    store = SimpleNamespace(collection_name="offline", write_upload_batch=lambda batch, rows: rows)
    encoder = SimpleNamespace(embed_texts=lambda texts: [[0.25, 0.5] for text in texts])
    indexer = UploadBatchIndexer(runner=runner, store=store, encoder=encoder,
                                publication=SimpleNamespace(stage_batch=AsyncMock()), root=root)
    indexed = asyncio.run(indexer.prepare({**parsed, "source_ref": reference}, paper={"year": 2026}, metadata_missing=[]))
    publication["repository"].content_reference.return_value = {"manifest_ref": indexed["manifest_ref"]}
    scope = {"user_id": "owner", "paper_ids": ["paper"], "publication": publication["snapshot"]}
    result = _read(KnowledgeReadRequest(operation="sections", scope=scope, paper_ids=("paper",),
                                       section_paths=("1 Method", "2 Results")))
    assert len(result.materials) == 2
    assert result.materials[0].text == body
    assert "| 1 | 2 |" in result.materials[1].text
    assert result.materials[0].source_location_status == "no_physical_location"
    for material in result.materials:
        for span in material.source_spans:
            chunk = parsed["document"]["chunks"][int(span["chunk_id"].rsplit(":", 1)[-1])]
            assert chunk[span["field"]][span["start"]:span["end"]] == material.text[span["text_start"]:span["text_end"]]
