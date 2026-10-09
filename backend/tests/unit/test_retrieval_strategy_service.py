from app.services.retrieval_strategy_service import build_retrieval_plan


def test_internal_project_question_prefers_milvus_and_neo4j() -> None:
    plan = build_retrieval_plan(
        query="这两篇论文的方法差异是什么",
        project_id="proj_1",
        paper_ids=["p1", "p2"],
        dialogue_mode="research",
        paper_focus_mode="multi_paper",
    )

    assert plan.primary_store == "milvus"
    assert plan.enable_graph_evidence is True
    assert plan.enable_library_graph_hint is True


def test_library_overview_question_uses_lightrag_as_hint_layer() -> None:
    plan = build_retrieval_plan(
        query="这个方向近三年的演化脉络",
        project_id=None,
        paper_ids=[],
    )

    assert plan.primary_store == "milvus"
    assert plan.enable_graph_evidence is False
    assert plan.enable_library_graph_hint is True


def test_single_paper_lookup_prefers_milvus_only() -> None:
    plan = build_retrieval_plan(
        query="这篇论文的贡献是什么",
        project_id="proj_1",
        paper_ids=["p1"],
        dialogue_mode="lookup",
        paper_focus_mode="single_paper",
    )

    assert plan.primary_store == "milvus"
    assert plan.enable_graph_evidence is False
    assert plan.enable_library_graph_hint is False
