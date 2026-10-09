from app.services.graph.gap_detection_service import get_gap_detection_service


def test_detect_limitation_unresolved_gap() -> None:
    service = get_gap_detection_service()
    papers = [
        {
            "paper_id": "p1",
            "analysis": {
                "limitation": "Still struggles with long-context reasoning.",
                "problem": "question answering",
                "method": "Method A",
            },
        },
        {
            "paper_id": "p2",
            "analysis": {
                "limitation": "Still struggles with long-context reasoning.",
                "problem": "question answering",
                "method": "Method B",
            },
        },
    ]

    gaps = service.detect_gaps(papers, "question answering")
    assert any(item["gap_type"] == "limitation_unresolved" for item in gaps)


def test_detect_dataset_or_evaluation_gap() -> None:
    service = get_gap_detection_service()
    papers = [
        {
            "paper_id": "p3",
            "analysis": {
                "problem": "question answering",
                "method": "Method A",
                "contribution": ["Improved accuracy on SQuAD."],
                "methodology_detail": "SQuAD accuracy",
            },
        },
        {
            "paper_id": "p4",
            "analysis": {
                "problem": "question answering",
                "method": "Method B",
                "contribution": ["Improved accuracy on SQuAD."],
                "methodology_detail": "SQuAD accuracy",
            },
        },
        {
            "paper_id": "p5",
            "analysis": {
                "problem": "question answering",
                "method": "Method C",
                "contribution": ["Improved accuracy on SQuAD."],
                "methodology_detail": "SQuAD accuracy",
            },
        },
    ]

    gaps = service.detect_gaps(papers, "question answering")
    assert any(item["gap_type"] in {"dataset_gap", "evaluation_gap"} for item in gaps)
