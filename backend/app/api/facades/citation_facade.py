"""引用相关 facade。"""

from __future__ import annotations

from app.api.facades import paper_facade
from app.tools.citation_tool import get_citation_tree
from app.services.graph import get_explainable_answer_service

# facade 的作用是把 route 层和底层 tool/service 解耦：
#   - route 不直接理解 citation tree 的拼装细节
#   - tool/service 可以独立演进，而 HTTP 层保持薄


async def get_citation_tree_payload(
    paper_id: str,
    depth: int = 1,
    *,
    user_id: str = "anonymous",
    include_graph_paths: bool = False,
):
    """
    读取某篇论文的引用树。

    `citation_tool` 负责基础引用结构；如果请求方希望看到“这些论文之间在图谱里
    是怎么连起来的”，这里再补一层 explainable answer 证据包。
    """
    clean_paper_id = str(paper_id or "").strip()
    if not clean_paper_id:
        raise ValueError("paper_id 不能为空")
    owned_paper = paper_facade.get_paper_or_none(clean_paper_id, user_id=user_id)
    if not owned_paper:
        raise LookupError("论文不存在")
    payload = get_citation_tree(clean_paper_id, depth=depth)
    if not include_graph_paths:
        return payload
    related_ids: list[str] = [clean_paper_id]
    for section in ("references", "cited_by"):
        for item in payload.get(section, []) or []:
            ref_id = str(item.get("paper_id") or item.get("arxiv_id") or "").strip()
            if ref_id and ref_id not in related_ids:
                related_ids.append(ref_id)
    evidence_bundle = await get_explainable_answer_service().build_bundle(
        paper_ids=related_ids,
        query=f"citation graph for {clean_paper_id}",
        path_type="citation_bridge",
    )
    return {
        **payload,
        "graph_evidence": evidence_bundle["graph_evidence"],
        "evidence_summary": evidence_bundle["evidence_summary"],
    }


async def format_citation_payload(paper_id: str, doi: str, arxiv_id: str, style: str) -> dict:
    """格式化单条引用；把不同身份标识统一收敛给 citation formatter。"""
    if not any((paper_id.strip(), doi.strip(), arxiv_id.strip())):
        raise ValueError("paper_id、doi、arxiv_id 至少提供一个")
    from app.services.citation_formatter import get_citation_formatter

    citation = await get_citation_formatter().format(
        paper_id=paper_id.strip(),
        doi=doi.strip(),
        arxiv_id=arxiv_id.strip(),
        style=style.strip().lower(),
    )
    return {"citation": citation, "style": style.strip().lower()}


async def batch_format_citations_payload(paper_ids: list[str], style: str) -> dict:
    """批量格式化引用；这里会先清洗空字符串，避免下游 formatter 处理脏输入。"""
    clean_ids = [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]
    if not clean_ids:
        raise ValueError("paper_ids 不能为空")
    from app.services.citation_formatter import get_citation_formatter

    citations = await get_citation_formatter().format_batch(
        paper_ids=clean_ids,
        style=style.strip().lower(),
    )
    return {"citations": citations, "style": style.strip().lower(), "count": len(citations)}
