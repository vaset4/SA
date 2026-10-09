from __future__ import annotations

import app.services.internal_research_service as internal_research_service


def test_build_internal_research_contract_returns_cross_paper_sections(monkeypatch) -> None:
    papers = {
        "paper-1": {
            "paper_id": "paper-1",
            "title": "STGCN",
            "authors": ["A"],
            "abstract": "traffic forecasting",
            "analysis": {
                "method": "Spatio-temporal graph convolution",
                "limitation": "只在固定图结构上验证。",
                "methodology_detail": "Datasets: METR-LA. Metrics: MAE, RMSE.",
                "contribution": ["建模时空依赖"],
                "problem": "traffic forecasting",
            },
        },
        "paper-2": {
            "paper_id": "paper-2",
            "title": "DCRNN",
            "authors": ["B"],
            "abstract": "traffic forecasting",
            "analysis": {
                "method": "Diffusion convolution recurrent network",
                "limitation": "对异常事件鲁棒性不足。",
                "methodology_detail": "Datasets: METR-LA, PEMS-BAY. Metrics: MAE, RMSE.",
                "contribution": ["引入扩散卷积"],
                "problem": "traffic forecasting",
            },
        },
    }

    monkeypatch.setattr(
        internal_research_service.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: papers.get(paper_id),
    )

    contract = internal_research_service.build_internal_research_contract(
        query="比较 STGCN 和 DCRNN 的数据集、指标和局限",
        answer="两篇方法都在交通预测上有效，但 DCRNN 覆盖的数据集更广。",
        citations=[{"paper_id": "paper-1"}, {"paper_id": "paper-2"}],
        structured_query={"paper_ids": ["paper-1", "paper-2"]},
        answer_confidence="high",
    )

    assert contract["research_scope"]["mode"] == "cross_paper"
    assert contract["dialogue_mode"] == "research"
    assert contract["synthesis_state"]["paper_focus"]["mode"] == "multi_paper"
    section_keys = [section["key"] for section in contract["research_objects"]]
    assert "comparisons" in section_keys
    assert "datasets" in section_keys
    assert "metrics" in section_keys
    assert "limitations" in section_keys
    assert any(action["type"] == "copy_note" for action in contract["note_actions"])


def test_build_internal_research_contract_adds_save_note_action_for_single_paper(monkeypatch) -> None:
    paper = {
        "paper_id": "paper-1",
        "title": "Retrieval-Augmented Generation",
        "authors": ["A"],
        "abstract": "rag",
        "analysis": {
            "method": "Retrieval-Augmented Generation",
            "limitation": "依赖检索质量。",
            "methodology_detail": "Datasets: NQ. Metrics: EM, F1.",
            "contribution": ["把检索和生成结合"],
            "problem": "open-domain qa",
        },
    }

    monkeypatch.setattr(
        internal_research_service.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: paper if paper_id == "paper-1" else None,
    )

    contract = internal_research_service.build_internal_research_contract(
        query="这篇论文的方法和局限是什么",
        answer="核心方法是 RAG，但性能依赖检索器。",
        citations=[{"paper_id": "paper-1"}],
        structured_query={"paper_ids": ["paper-1"]},
        answer_confidence="medium",
    )

    assert contract["research_scope"]["mode"] == "single_paper"
    assert contract["dialogue_mode"] == "lookup"
    assert contract["synthesis_state"]["paper_focus"]["mode"] == "single_paper"
    assert any(
        action["type"] == "save_note"
        and action["target_paper_id"] == "paper-1"
        and action["note_type"] == "reading_note"
        and action["source_type"] == "local_rag"
        for action in contract["note_actions"]
    )


def test_build_internal_research_contract_marks_cross_paper_note_type(monkeypatch) -> None:
    papers = {
        "paper-1": {
            "paper_id": "paper-1",
            "title": "A",
            "authors": ["A"],
            "abstract": "a",
            "analysis": {"method": "Method A"},
        },
        "paper-2": {
            "paper_id": "paper-2",
            "title": "B",
            "authors": ["B"],
            "abstract": "b",
            "analysis": {"method": "Method B"},
        },
    }

    monkeypatch.setattr(
        internal_research_service.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: papers.get(paper_id),
    )

    contract = internal_research_service.build_internal_research_contract(
        query="比较两篇论文方法差异",
        answer="两篇方法差异明显。",
        citations=[{"paper_id": "paper-1"}, {"paper_id": "paper-2"}],
        structured_query={"paper_ids": ["paper-1", "paper-2"]},
        answer_confidence="high",
    )

    assert any(
        action["type"] == "copy_note"
        for action in contract["note_actions"]
    )
    assert not any(action["type"] == "save_note" for action in contract["note_actions"])


def test_build_internal_research_contract_emits_project_scope_and_open_questions(monkeypatch) -> None:
    monkeypatch.setattr(
        internal_research_service.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: None,
    )

    contract = internal_research_service.build_internal_research_contract(
        query="总结当前项目里的论文",
        answer="当前证据不足。",
        citations=[],
        structured_query={
            "project_scope": {
                "project_id": "proj-1",
                "paper_ids": ["paper-1", "paper-2"],
                "filter_expr": 'paper_id == "paper-1" or paper_id == "paper-2"',
                "resolved_scope": {"paper_ids": ["paper-1", "paper-2"]},
                "has_project_scope": True,
                "summary": "当前项目范围覆盖 2 篇论文。",
            },
            "paper_focus": {
                "mode": "project_scope",
                "paper_ids": ["paper-1", "paper-2"],
                "is_explicit": False,
                "source": "project_scope",
                "summary": "当前问题默认使用范围内的 2 篇论文作为候选焦点。",
            },
            "dialogue_mode": "research",
            "project_id": "proj-1",
            "paper_ids": ["paper-1", "paper-2"],
        },
        answer_confidence="low",
        failure_reason="analysis failed",
    )

    assert contract["synthesis_state"]["project_scope"]["project_id"] == "proj-1"
    assert contract["synthesis_state"]["status"] == "insufficient"
    assert contract["synthesis_state"]["unsupported_dimensions"]
    assert contract["synthesis_state"]["open_questions"]
