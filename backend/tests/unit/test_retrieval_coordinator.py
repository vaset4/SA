"""权威材料转证据、来源身份及图谱转换离线测试，作者：xxx。"""
from dataclasses import replace
from copy import deepcopy

import pytest

from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials, convert_neo4j_evidence
from app.services.internal_research.evidence_identity import (
    count_material_groups, register_evidence_records, validate_read_evidence,
)
from o2_evidence_test_utils import material


def test_authoritative_text_not_preview_and_no_hidden_truncation():
    """无参数；验证表格结构及长父正文完整转换，排名预览不能替换正文。"""
    body = "| a | b |\n| 1 | 2 |\n" + "x" * 5000
    source = material(body, field="full_text")
    record = convert_knowledge_materials((source,), hits=[{"chunk_id": source.chunk_id,
                                                        "preview": "clipped", "score": 0.8}])[0]
    assert record["excerpt"] == body
    assert record["expanded_text"] == ""
    assert record["confidence"] == 0.8
    assert record["source_spans"][0]["field"] == "full_text"
    validate_read_evidence(record)


def test_same_group_extended_range_has_new_identity_but_one_count():
    """无参数；同一材料范围扩展获得新 ID，数量仍只有一个材料组。"""
    first = convert_knowledge_materials((material("first"),))[0]
    larger = convert_knowledge_materials((material("first and more"),))[0]
    assert first["evidence_id"] != larger["evidence_id"]
    assert first["material_group_id"] == larger["material_group_id"]
    assert count_material_groups([first, larger]) == 1
    assert len(register_evidence_records({}, [first, larger])) == 2


def test_query_and_ranking_do_not_change_source_identity():
    """无参数；正文来源不变时不同查询排名不能改变证据身份。"""
    source = material()
    first = convert_knowledge_materials((source,), hits=[{"chunk_id": source.chunk_id, "score": 0.1}])[0]
    second = convert_knowledge_materials((source,), hits=[{"chunk_id": source.chunk_id, "score": 0.9}])[0]
    assert first["evidence_id"] == second["evidence_id"]
    assert len(register_evidence_records({}, [first, second])) == 1


def test_canonical_parent_anchor_keeps_score_from_another_hit_child():
    """无参数；父块固定回读锚点不等于检索命中子块时，仍保留该父材料真实排名分数。"""
    source = material(field="full_text")
    record = convert_knowledge_materials((source,), hits=[{
        "paper_id": source.paper_id, "chunk_id": f"v3:{source.batch_id}:2",
        "parent_id": source.parent_id, "score": 0.75,
    }])[0]
    assert record["confidence"] == 0.75


def test_paper_batch_and_group_kind_are_distinct():
    """无参数；不同论文、批次及分组类型的同名定位不能碰撞。"""
    from app.services.knowledge_access_service import material_group_id
    sources = [material(), material(paper_id="paper-b"), material(batch_id="c")]
    section_group = material_group_id(paper_id="paper-a", batch_id="b", group_kind="section", group_locator="parent-a")
    sources.append(replace(sources[0], group_kind="section", material_group_id=section_group))
    records = convert_knowledge_materials(tuple(sources))
    assert len({record["evidence_id"] for record in records}) == 4
    assert count_material_groups(records) == 4


def test_changed_content_cannot_reuse_a_saved_new_identity():
    """无参数；同 ID 篡改正文或跨度明确失败，不被首次登记吞掉。"""
    record = convert_knowledge_materials((material(),))[0]
    changed = deepcopy(record)
    changed["excerpt"] = "x" * len(record["excerpt"])
    with pytest.raises(ValueError, match="承诺"):
        register_evidence_records({record["evidence_id"]: record}, [changed])
    broken = {**record, "source_spans": []}
    with pytest.raises(ValueError, match="跨度"):
        count_material_groups([broken])


def test_legacy_saved_id_remains_opaque():
    """无参数；历史保存身份不重算，仍可通过历史消费者读取。"""
    record = {"evidence_id": "old:saved-id", "excerpt": "saved body"}
    assert register_evidence_records({}, [record]) == {"old:saved-id": record}
    assert count_material_groups([record, record]) == 1


class TestConvertNeo4jEvidence:
    def test_basic_conversion(self):
        evidence = [
            {"summary": "Paper A uses method X", "paper_id": "paper_a"},
            {"summary": "Paper B extends Paper A", "paper_id": "paper_b"},
        ]
        records = convert_neo4j_evidence(evidence)
        assert len(records) == 2
        for r in records:
            assert r["source"] == "neo4j"
            assert r["layer"] == "structure_hint"
            assert r["provenance"] == "ai_extracted_graph"
            assert r["verification_status"] == "unverified"
            assert r["usable_as_citation"] is False

    def test_empty_summary_skipped(self):
        evidence = [{"summary": "", "paper_id": "a"}, {"paper_id": "b"}]
        assert convert_neo4j_evidence(evidence) == []

    def test_none_input(self):
        assert convert_neo4j_evidence(None) == []
