import pytest

from app.services import document_parser as parser


class FakeTokenCounter:
    def _tokens(self, text: str) -> list[str]:
        return text.split()

    def count(self, text: str) -> int:
        return len(self._tokens(text))

    def trim(self, text: str, max_tokens: int) -> str:
        return " ".join(self._tokens(text)[:max_tokens]).strip()

    def tail(self, text: str, max_tokens: int) -> str:
        return " ".join(self._tokens(text)[-max_tokens:]).strip()

    def slice(self, text: str, max_tokens: int) -> list[str]:
        tokens = self._tokens(text)
        return [
            " ".join(tokens[index: index + max_tokens]).strip()
            for index in range(0, len(tokens), max_tokens)
            if tokens[index: index + max_tokens]
        ]


@pytest.fixture(autouse=True)
def fake_token_counter(monkeypatch):
    monkeypatch.setattr(parser, "_get_token_counter", lambda: FakeTokenCounter())


def test_structure_merge_preserves_formal_numbered_stub(monkeypatch):
    monkeypatch.setattr(parser.settings, "structure_merge_enabled", True, raising=False)
    monkeypatch.setattr(parser.settings, "chunk_merge_floor", 2, raising=False)
    sections = [
        parser._section("3.1 Setup", "stub", 1, section_path="3.1 Setup"),
        parser._section("3.2 Model", "main model text", 2, section_path="3.2 Model"),
        parser._section("4 Result", "result evidence", 3, section_path="4 Result"),
    ]

    merged = parser._merge_sections_for_parenting(sections, "paper")

    assert [item["section_path"] for item in merged] == ["3.1 Setup", "3.2 Model", "4 Result"]
    assert merged[0]["text"] == "stub"


def test_split_large_section_applies_token_overlap():
    text = "a b c d\n\ne f g h\n\ni j k l"

    parts = parser._split_large_section(text, max_size=6, overlap_tokens=2)

    assert len(parts) == 3
    assert parts[1].split()[:2] == parts[0].split()[-2:]
    assert parts[2].split()[:2] == parts[1].split()[-2:]
    assert all(parser._count_tokens(item) <= 6 for item in parts)


def test_split_to_child_chunks_sets_token_metadata():
    parent_text = "alpha beta gamma. delta epsilon zeta. eta theta iota. kappa lambda mu."

    chunks = parser._split_to_child_chunks(
        parent_text=parent_text,
        parent_id="paper_0",
        page_num=1,
        section="method",
        target_size=4,
        overlap_tokens=1,
    )

    assert len(chunks) >= 2
    assert all(chunk["full_text_token_count"] == 12 for chunk in chunks)
    assert all(chunk["chunk_token_count"] == parser._count_tokens(chunk["text"]) for chunk in chunks)
    assert all(parser._count_tokens(chunk["text"]) <= 6 for chunk in chunks)


def test_parse_markdown_text_preserves_structured_blocks_and_filters_references(monkeypatch):
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", 256, raising=False)
    monkeypatch.setattr(parser.settings, "child_chunk_target_size", 128, raising=False)
    markdown = """
# Paper Title

## Abstract

This work routes PDF parsing through MinerU and chunks the normalized markdown.

## Method

$$
L = x + y
$$

| metric | value |
| --- | --- |
| F1 | 0.91 |

## References

[1] Some cited work.
""".strip()

    chunks = parser.parse_markdown_text(markdown, "paper")

    assert chunks
    assert any("Abstract" in chunk["section"] for chunk in chunks)
    assert any("Method" in chunk["section"] for chunk in chunks)
    assert all("References" not in chunk["section"] for chunk in chunks)
    assert any("$$" in chunk["full_text"] for chunk in chunks)
    assert any("| metric | value |" in chunk["full_text"] for chunk in chunks)


def test_parse_markdown_text_filters_discarded_noise_from_content_list(monkeypatch):
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", 256, raising=False)
    monkeypatch.setattr(parser.settings, "child_chunk_target_size", 128, raising=False)

    markdown = """
# Paper Title

Author One, Author Two

## Abstract

Structured abstract text.
""".strip()
    content_list = [
        {"type": "text", "text": "Paper Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Author One, Author Two", "page_idx": 0},
        {"type": "discarded", "text": "∗Equal contributions. †Corresponding author.", "page_idx": 0},
        {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Structured abstract text.", "page_idx": 0},
    ]

    chunks = parser.parse_markdown_text(markdown, "paper", content_list=content_list)

    assert chunks
    assert all("Equal contributions" not in chunk["full_text"] for chunk in chunks)
