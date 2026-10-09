from app.services.graph.claim_conflict_service import get_claim_conflict_service


def test_detect_metric_direction_conflict() -> None:
    service = get_claim_conflict_service()
    papers = [
        {
            "paper_id": "p1",
            "analysis": {
                "problem": "question answering",
                "method": "Method A",
                "contribution": ["Improved accuracy on SQuAD."],
                "tldr": "Better results.",
                "methodology_detail": "SQuAD accuracy",
            },
        },
        {
            "paper_id": "p2",
            "analysis": {
                "problem": "question answering",
                "method": "Method B",
                "contribution": ["Lower accuracy on SQuAD under the same metric."],
                "tldr": "Worse results.",
                "methodology_detail": "SQuAD accuracy",
            },
        },
    ]

    triples = service.build_claim_triples(papers)
    conflicts = service.detect_conflicts(triples)

    assert any(item["conflict_type"] == "metric_direction_conflict" for item in conflicts)


def test_detect_method_tradeoff_conflict() -> None:
    service = get_claim_conflict_service()
    papers = [
        {
            "paper_id": "p3",
            "analysis": {
                "problem": "question answering",
                "method": "Method C",
                "contribution": ["Improved accuracy on SQuAD."],
                "methodology_detail": "SQuAD accuracy",
            },
        },
        {
            "paper_id": "p4",
            "analysis": {
                "problem": "question answering",
                "method": "Method D",
                "contribution": ["More efficient with lower latency and memory cost."],
                "methodology_detail": "SQuAD accuracy",
            },
        },
    ]

    conflicts = service.detect_conflicts(service.build_claim_triples(papers))
    assert any(item["conflict_type"] == "method_tradeoff_conflict" for item in conflicts)
