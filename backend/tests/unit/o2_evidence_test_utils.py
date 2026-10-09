"""O2 取证离线夹具：真实知识访问入口仅替换发布存储端口，作者：xxx。"""
from __future__ import annotations

from functools import partial
from typing import Any
from types import SimpleNamespace

from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeMaterial, KnowledgeReadReport
from app.services.knowledge_access_service import material_group_id, read_knowledge
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials


def read_result(records: list[dict[str, Any]]) -> EvidenceReadResult:
    """参数为显式构造的测试材料；仅封装正式返回合同，不修改版本、内容或身份。"""
    return EvidenceReadResult(tuple(records), KnowledgeReadReport(operation="test", status="complete"))


def direct_evidence(label: str = "test", *, paper_id: str = "paper-a",
                    text: str = "The paper uses attention.") -> dict[str, Any]:
    """参数为来源定位、论文与正文；经正式材料转换构造新版证据，不将旧记录自动升级。"""
    return dict(convert_knowledge_materials((material(text, paper_id=paper_id, locator=label),))[0])


def material(text: str = "The paper uses attention.", *, paper_id: str = "paper-a",
             batch_id: str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", locator: str = "parent-a", field: str = "text") -> KnowledgeMaterial:
    """参数为正文、论文、批次、父定位和字段；返回具备实际范围的离线材料组。"""
    chunk_id = f"v3:{batch_id}:0"
    return KnowledgeMaterial(
        paper_id=paper_id, batch_id=batch_id, group_kind="parent", group_locator=locator,
        material_group_id=material_group_id(paper_id=paper_id, batch_id=batch_id,
                                            group_kind="parent", group_locator=locator),
        text=text, source_spans=({"paper_id": paper_id, "batch_id": batch_id, "chunk_id": chunk_id,
                                "field": field, "start": 0, "end": len(text),
                                "text_start": 0, "text_end": len(text)},),
        chunk_id=chunk_id, parent_id=locator, section_path="Paper / Method", section="Method",
        page_num=0, source_location_status="legacy_chunk", block_type="paragraph",
    )


def scope(paper_id: str = "paper-a", *, batch_id: str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") -> dict[str, Any]:
    """参数为论文与批次；返回明确用户与租约的离线发布快照。"""
    return {"user_id": "u1", "paper_ids": [paper_id], "local_retrieval_allowed": True,
            "publication": {"user_id": "u1", "lease_id": "test-lease", "batch_ids": [batch_id],
                            "paper_batches": {paper_id: batch_id}}}


def bind_publication(monkeypatch, rows: list[dict[str, Any]], *, paper_id: str = "paper-a",
                     batch_id: str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", failure: Exception | None = None):
    """参数为测试 patch、发布块及可选存储故障；注入真实知识读取并返回存储调用观察。"""
    from app.nodes import rag_node
    from app.services import upload_read_service

    calls: list[tuple[Any, dict[str, Any]]] = []
    chunks = sorted(rows, key=lambda row: int(row["chunk_id"].rsplit(":", 1)[-1]))
    manifest = {"batch_id": batch_id, "paper_id": paper_id, "chunks": chunks}

    def reader(batch_ids, **kwargs):
        """参数为批次和授权上下文；返回固定离线清单或真实故障替身。"""
        calls.append((batch_ids, kwargs))
        if failure is not None:
            raise failure
        return {batch_id: manifest}

    monkeypatch.setattr(rag_node, "read_knowledge", partial(read_knowledge, publication_reader=reader))
    async def renew_snapshot(snapshot):
        """参数为正式授权入口检查后的快照；隔离租约数据库并核对固定批次。"""
        assert snapshot["paper_batches"] == {paper_id: batch_id}

    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository",
                        lambda: SimpleNamespace(renew_snapshot=renew_snapshot))
    return calls
