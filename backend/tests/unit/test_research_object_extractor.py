from app.services.graph.research_object_extractor import extract_research_objects


def test_extract_research_objects_prefers_analysis_and_full_text_patterns() -> None:
    paper = {
        "paper_id": "upload_1",
        "library_id": "default",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "categories": ["cs.CL"],
        "source": "upload",
        "analysis": {
            "problem": "This paper studies machine translation with attention.",
            "method": "Transformer",
            "contribution": ["Improves translation quality"],
            "limitation": "Needs large-scale training",
            "keywords": ["machine translation", "attention"],
            "related_work": [],
            "tldr": "A transformer model for sequence transduction.",
            "methodology_type": "experimental",
            "methodology_detail": "Evaluated on WMT with BLEU and ImageNet is not used.",
        },
    }
    result = extract_research_objects(
        paper,
        "We report BLEU on WMT and compare on SQuAD-style benchmarks for question answering.",
    )
    assert result["methods"] == ["Transformer"]
    assert "bleu" in [item.lower() for item in result["metrics"]]
    assert "machine translation" in [item.lower() for item in result["tasks"]]
    assert result["limitations"] == ["Needs large-scale training"]


def test_extract_research_objects_drops_affiliation_and_placeholder_noise() -> None:
    paper = {
        "paper_id": "upload_2",
        "library_id": "default",
        "title": "Graph WaveNet",
        "authors": [
            "Centre for Artificial Intelligence, FEIT, University of Technology Sydney, Australia",
            "Guodong Long",
        ],
        "categories": ["cs.LG"],
        "source": "upload",
        "analysis": {
            "problem": "Traffic forecasting.",
            "method": "Graph WaveNet, a novel graph neural network architecture that captures hidden spatial dependencies.",
            "contribution": ["Improves traffic forecasting accuracy."],
            "limitation": "Not explicitly stated",
            "keywords": ["traffic forecasting"],
            "related_work": [],
            "tldr": "Graph WaveNet for traffic forecasting.",
            "methodology_type": "experimental",
            "methodology_detail": "Evaluated with accuracy and METR-LA.",
        },
    }

    result = extract_research_objects(paper, "")

    assert result["authors"] == ["Guodong Long"]
    assert result["methods"] == ["Graph WaveNet"]
    assert result["limitations"] == []
