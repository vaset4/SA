from __future__ import annotations

from app.api.facades import research_contract_facade


def test_resolve_dialogue_mode_prefers_lookup_for_single_explicit_paper() -> None:
    mode = research_contract_facade.resolve_dialogue_mode(
        "这篇论文的贡献是什么",
        project_id="proj-1",
        paper_ids=["paper-1"],
    )

    assert mode == "lookup"


def test_resolve_dialogue_mode_prefers_research_for_multi_paper_scope() -> None:
    mode = research_contract_facade.resolve_dialogue_mode(
        "比较这些论文的方法差异",
        project_id="proj-1",
        paper_ids=["paper-1", "paper-2"],
    )

    assert mode == "research"


def test_build_dialogue_structured_query_contains_scope_focus_and_evidence_roles() -> None:
    payload = research_contract_facade.build_dialogue_structured_query(
        query="比较这两篇论文的方法",
        project_context={
            "project_id": "proj-1",
            "resolved_scope": {"paper_ids": ["paper-1", "paper-2"]},
            "filter_expr": 'paper_id == "paper-1" or paper_id == "paper-2"',
            "paper_ids": ["paper-1", "paper-2"],
        },
        project_id="proj-1",
        explicit_paper_ids=["paper-1", "paper-2"],
    )

    assert payload["dialogue_mode"] == "research"
    assert payload["project_scope"]["project_id"] == "proj-1"
    assert payload["paper_focus"]["mode"] == "multi_paper"
    assert payload["paper_ids"] == ["paper-1", "paper-2"]
    assert [item["layer"] for item in payload["evidence_layers"]] == ["milvus", "lightrag", "neo4j"]
