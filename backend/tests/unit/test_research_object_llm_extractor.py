import pytest

from app.services.graph.research_object_llm_extractor import ExtractionFailure, extract_with_llm


class FakeResponse:
    def __init__(self, content: str, raw: dict | None = None) -> None:
        self.content = content
        self.raw = raw or {}


class FakeLlm:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        self.prompts.append(messages[-1].content)
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return FakeResponse(response, raw={"usage": {"prompt_tokens": 12, "completion_tokens": 5}})


def _paper() -> dict:
    return {
        "paper_id": "paper_1",
        "library_id": "lib_1",
        "title": "STGCN for Traffic Forecasting",
        "authors": ["Alice Zhang"],
        "analysis": {
            "contribution": ["Improves traffic forecasting."],
            "limitation": "Needs sensor graph.",
            "methodology_type": "experimental",
        },
    }


@pytest.mark.asyncio
async def test_extract_returns_typed_extraction() -> None:
    llm = FakeLlm([
        '{"entities":[{"canonical_name":"STGCN","type":"Method","aliases":["Spatio-Temporal Graph Convolutional Network"],"evidence_excerpt":"STGCN is proposed."},{"canonical_name":"PEMS-BAY","type":"Dataset","aliases":[],"evidence_excerpt":"PEMS-BAY is used."},{"canonical_name":"traffic forecasting","type":"Task","aliases":[],"evidence_excerpt":"traffic forecasting task."}]}'
    ])

    extraction, alias_map = await extract_with_llm(_paper(), "STGCN uses PEMS-BAY for traffic forecasting.", llm=llm)

    assert extraction["methods"] == ["STGCN"]
    assert extraction["datasets"] == ["PEMS-BAY"]
    assert extraction["tasks"] == ["traffic forecasting"]
    assert alias_map == {"STGCN": ["Spatio-Temporal Graph Convolutional Network"], "PEMS-BAY": [], "traffic forecasting": []}


@pytest.mark.asyncio
async def test_extract_invalid_json_retries_until_success() -> None:
    llm = FakeLlm(["garbage", "garbage", "garbage", '{"entities": []}'])

    extraction, alias_map = await extract_with_llm(_paper(), "Valid paper text.", llm=llm, max_retries=3)

    assert extraction["methods"] == []
    assert alias_map == {}
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_extract_max_retries_raises_failure() -> None:
    llm = FakeLlm(["garbage"])

    with pytest.raises(ExtractionFailure) as exc_info:
        await extract_with_llm(_paper(), "Valid paper text.", llm=llm, max_retries=3)

    assert exc_info.value.attempts == 4
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_extract_filters_noise_entities_and_invalid_types() -> None:
    llm = FakeLlm([
        '{"entities":[{"canonical_name":"Beijing","type":"Method","aliases":[],"evidence_excerpt":"Beijing University."},{"canonical_name":"Tang","type":"Method","aliases":[],"evidence_excerpt":"Tang et al."},{"canonical_name":"Person X","type":"Person","aliases":[],"evidence_excerpt":"Person."},{"canonical_name":"BERT","type":"Method","aliases":["deep learning"],"evidence_excerpt":"BERT is fine-tuned."}]}'
    ])

    extraction, alias_map = await extract_with_llm(_paper(), "BERT is fine-tuned. BERT is evaluated.", llm=llm)

    assert extraction["methods"] == ["BERT"]
    assert alias_map == {"BERT": []}


@pytest.mark.asyncio
async def test_extract_does_not_pass_filename_to_llm() -> None:
    llm = FakeLlm(['{"entities": []}'])
    paper = {**_paper(), "file_path": "D:/tmp/private-file-name.pdf"}

    await extract_with_llm(paper, "Valid paper text.", llm=llm)

    assert "private-file-name.pdf" not in llm.prompts[0]
