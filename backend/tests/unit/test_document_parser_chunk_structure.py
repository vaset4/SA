from __future__ import annotations

import pytest

from app.services import document_parser as parser
from app.services.upload_steps import STEP_PARSE, fail_step


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
            " ".join(tokens[index:index + max_tokens]).strip()
            for index in range(0, len(tokens), max_tokens)
            if tokens[index:index + max_tokens]
        ]


@pytest.fixture(autouse=True)
def fake_token_counter(monkeypatch):
    monkeypatch.setattr(parser, "_get_token_counter", lambda: FakeTokenCounter())
    monkeypatch.setattr(parser.settings, "structure_merge_enabled", True, raising=False)
    monkeypatch.setattr(parser.settings, "chunk_merge_floor", 5, raising=False)
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", 20, raising=False)
    monkeypatch.setattr(parser.settings, "parent_chunk_overlap_tokens", 0, raising=False)
    monkeypatch.setattr(parser.settings, "child_chunk_target_size", 20, raising=False)
    monkeypatch.setattr(parser.settings, "child_chunk_overlap_tokens", 0, raising=False)


def test_chunk_structure_merge_never_crosses_top_level_key() -> None:
    sections = [
        parser._section("1 Intro", "intro tiny", 1, section_path="1 Intro"),
        parser._section("2 Method", "method tiny", 2, section_path="2 Method"),
    ]

    merged = parser._merge_sections_for_parenting(sections, "paper")

    assert len(merged) == 2
    assert all("intro tiny" not in item["text"] or "method tiny" not in item["text"] for item in merged)


def test_chunk_structure_merge_never_crosses_formal_numbered_siblings() -> None:
    sections = [
        parser._section("3.1 Setup", "stub", 1, section_path="3.1 Setup"),
        parser._section("3.2 Model", "main model text", 1, section_path="3.2 Model"),
    ]

    merged = parser._merge_sections_for_parenting(sections, "paper")

    assert [item["section_path"] for item in merged] == ["3.1 Setup", "3.2 Model"]
    assert all(" + " not in item["section_path"] for item in merged)


def test_chunk_structure_merge_never_crosses_local_numbered_subsections() -> None:
    sections = [
        parser._section("(1) Masking strategy", "masking stub", 1, section_path="(1) Masking strategy"),
        parser._section(
            "(2) Model for learning representations",
            "transformer model text",
            1,
            section_path="(2) Model for learning representations",
        ),
    ]

    merged = parser._merge_sections_for_parenting(sections, "paper")

    assert [item["section_path"] for item in merged] == [
        "(1) Masking strategy",
        "(2) Model for learning representations",
    ]


def test_missing_section_path_fails_fast() -> None:
    sections = [parser._section("untitled", "alpha", 1, section_path="")]

    with pytest.raises(parser.StructuredChunkingError) as exc_info:
        parser._merge_sections_for_parenting(sections, "paper-missing")

    error = exc_info.value
    assert error.stage == "parse"
    assert error.code == "missing_section_path"
    assert error.paper_id == "paper-missing"
    assert "stage=parse" in str(error)
    assert "code=missing_section_path" in str(error)


def test_single_unusable_section_key_fails_fast() -> None:
    sections = [
        parser._section("content", "alpha", 1, section_path="content"),
        parser._section("content", "beta", 1, section_path="content"),
    ]

    with pytest.raises(parser.StructuredChunkingError) as exc_info:
        parser._merge_sections_for_parenting(sections, "paper-single-key")

    assert exc_info.value.code == "single_section_key"
    assert "single unusable section key" in exc_info.value.detail


def test_atomic_block_stays_separate_and_keeps_block_type() -> None:
    content_list = [
        {"type": "text", "text": "Method", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "prose before table", "page_idx": 0},
        {"type": "table", "table_body": "<table><tr><td>metric</td><td>0.91</td></tr></table>", "page_idx": 0},
        {"type": "text", "text": "prose after table", "page_idx": 0},
    ]

    chunks = parser.parse_markdown_text("", "paper", content_list=content_list)

    assert [chunk["block_type"] for chunk in chunks] == ["paragraph", "table", "paragraph"]
    assert chunks[1]["text"] == chunks[1]["full_text"]
    assert "prose before table" not in chunks[1]["full_text"]


def test_metric_label_stays_attached_to_display_formula() -> None:
    markdown = """
# Paper

## 4.2.1. Evaluation metrics

Intro text.

• Root mean squared error (RMSE):

$$
RMSE = value
$$
""".strip()

    chunks = parser.parse_markdown_text(markdown, "paper-formula")

    formula_chunks = [chunk for chunk in chunks if chunk["block_type"] == "formula"]
    assert len(formula_chunks) == 1
    assert "Root mean squared error (RMSE)" in formula_chunks[0]["full_text"]
    assert "$$" in formula_chunks[0]["full_text"]
    assert "Intro text" not in formula_chunks[0]["full_text"]


def test_adjacent_metric_formulas_in_same_section_share_retrieval_parent(monkeypatch) -> None:
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", 80, raising=False)
    markdown = """
# Paper

## 4.2.1. Evaluation metrics

• Root mean squared error (RMSE):

$$
RMSE = value
$$

• Mean absolute error (MAE):

$$
MAE = value
$$

• Mean absolute percentage error (MAPE):

$$
MAPE = value
$$
""".strip()

    chunks = parser.parse_markdown_text(markdown, "paper-formulas")

    formula_chunks = [chunk for chunk in chunks if chunk["block_type"] == "formula"]
    assert len(formula_chunks) == 1
    assert "RMSE = value" in formula_chunks[0]["full_text"]
    assert "MAE = value" in formula_chunks[0]["full_text"]
    assert "MAPE = value" in formula_chunks[0]["full_text"]


def test_table_caption_stays_attached_to_table_block() -> None:
    markdown = """
# Paper

## 4.3.2. Ablation studies

Lead paragraph.

Table 3 Ablation studies.

| Model | RMSE |
| --- | --- |
| Ours | 4.42 |
""".strip()

    chunks = parser.parse_markdown_text(markdown, "paper-table-caption")

    table_chunks = [chunk for chunk in chunks if chunk["block_type"] == "table"]
    assert len(table_chunks) == 1
    assert "Table 3 Ablation studies." in table_chunks[0]["full_text"]
    assert "| Model | RMSE |" in table_chunks[0]["full_text"]
    assert "Lead paragraph" not in table_chunks[0]["full_text"]


def test_masking_strategy_splits_enumerated_child_chunks_without_overlap() -> None:
    parent_text = "\n\n".join([
        "Two important elements are considered.",
        "(a) The basic unit for masking uses equal-length subseries.",
        "(b) Mask ratio is set to 75 percent for low-density traffic data.",
    ])

    chunks = parser._split_to_child_chunks(
        parent_text=parent_text,
        parent_id="paper_0",
        page_num=1,
        section="(1) Masking strategy",
        target_size=100,
        overlap_tokens=2,
        section_path="(1) Masking strategy",
    )

    assert [chunk["display_text"] for chunk in chunks] == [
        "Two important elements are considered. (a) The basic unit for masking uses equal-length subseries.",
        "(b) Mask ratio is set to 75 percent for low-density traffic data.",
    ]
    assert "(a)" not in chunks[1]["display_text"]


def test_inline_enumerated_claims_are_child_boundaries() -> None:
    parent_text = (
        "As seen from the table, (1) long-term trend helps larger horizons. "
        "(2) periodicity improves performance. "
        "(3) short-term trend is necessary."
    )

    chunks = parser._split_to_child_chunks(
        parent_text=parent_text,
        parent_id="paper_0",
        page_num=1,
        section="4.3.2. Ablation studies",
        target_size=100,
        section_path="4.3.2. Ablation studies",
    )

    assert len(chunks) == 3
    assert chunks[0]["display_text"].startswith("As seen from the table, (1)")
    assert chunks[1]["display_text"].startswith("(2)")
    assert chunks[2]["display_text"].startswith("(3)")


def test_formula_split_never_places_boundary_inside_display_formula() -> None:
    text = (
        "before formula words. "
        "$$ a b $$ "
        "after formula words."
    )

    parts = parser._split_oversize_paragraph(text, max_size=6)

    assert any(part.startswith("$$") and part.endswith("$$") for part in parts)
    assert all(part.count("$$") in (0, 2) for part in parts)


def test_oversized_non_table_atomic_block_fails_instead_of_splitting() -> None:
    sections = [
        parser._section(
            "4 Results",
            " ".join(f"metric{i}" for i in range(25)),
            1,
            block_type="code",
            section_path="4 Results",
        )
    ]

    with pytest.raises(parser.StructuredChunkingError) as exc_info:
        parser._sections_to_chunks(sections, "paper-table")

    assert exc_info.value.code == "structured_block_too_large"
    assert "code too large" in exc_info.value.detail


def test_oversized_table_stays_atomic_with_visible_warning() -> None:
    sections = [
        parser._section(
            "4 Results",
            " ".join(f"cell{i}" for i in range(25)),
            1,
            block_type="table",
            section_path="4 Results",
        )
    ]

    chunks = parser._sections_to_chunks(sections, "paper-table")

    assert len(chunks) == 1
    assert chunks[0]["block_type"] == "table"
    assert chunks[0]["full_text"] == sections[0]["text"]
    assert chunks[0]["chunk_warnings"] == ["atomic_block_exceeds_parent_limit"]


def test_oversized_sentence_fails_without_token_slice(monkeypatch) -> None:
    class NoSliceTokenCounter(FakeTokenCounter):
        def slice(self, text: str, max_tokens: int) -> list[str]:
            raise AssertionError("token slicing must not be used")

    monkeypatch.setattr(parser, "_get_token_counter", lambda: NoSliceTokenCounter())
    text = " ".join(f"word{i}" for i in range(25))

    with pytest.raises(parser.StructuredChunkingError) as exc_info:
        parser._split_oversize_paragraph(text, max_size=6, paper_id="paper-sentence", section_path="3 Method")

    assert exc_info.value.code == "sentence_too_large"
    assert "tokens=25" in exc_info.value.detail


def test_parse_step_detail_preserves_strict_chunking_fields() -> None:
    error = parser.StructuredChunkingError(
        "missing_section_path",
        paper_id="paper-upload",
        file_id="paper-upload",
        detail="no usable section_path",
    )

    steps = fail_step([], STEP_PARSE, 0.0, error)
    detail = steps[-1]["detail"]

    assert steps[-1]["stage"] == "parse"
    assert steps[-1]["status"] == "error"
    assert "stage=parse" in detail
    assert "code=missing_section_path" in detail
    assert "paper_id=paper-upload" in detail
    assert "detail=no usable section_path" in detail


def test_embedding_text_prefixes_section_path_without_changing_display_text() -> None:
    chunk = parser._split_to_child_chunks(
        parent_text="main body text",
        parent_id="paper_0",
        page_num=1,
        section="3 Method",
        target_size=20,
        section_path="Paper Title / 3 Method",
    )[0]

    assert chunk["embedding_text"].startswith("Paper Title / 3 Method\n")
    assert chunk["display_text"] == "main body text"
    assert chunk["full_text"] == "main body text"


def test_table_embedding_text_keeps_rows_from_original_html() -> None:
    """表格的索引文本按原始 HTML 还原行边界，显示文本保持原样。"""
    chunks = parser.parse_markdown_text(
        "## 4.1 Datasets\n\nTable 1 | caption\n\nplaceholder\n",
        "paper-table",
        content_list=[
            {"type": "text", "text_level": 2, "text": "4.1 Datasets", "page_idx": 0},
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": ["Table 1", "Statistics of datasets."],
                "table_body": (
                    "<table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td></tr>"
                    "<tr><td>METR-LA</td><td>207</td><td>1722</td></tr>"
                    "<tr><td>PEMS-BAY</td><td>325</td><td>2694</td></tr></table>"
                ),
            },
        ],
    )
    table = next(chunk for chunk in chunks if chunk["block_type"] == "table")

    assert "| METR-LA | 207 | 1722 |" in table["embedding_text"]
    assert "| PEMS-BAY | 325 | 2694 |" in table["embedding_text"]
    # 表名的词项不能因为被摘成标签而离开索引文本
    assert "Statistics of datasets." in table["embedding_text"]
    assert "\n" in table["embedding_text"]
    # 显示文本仍然是原先那一条，未被改写
    assert table["display_text"] == table["text"]


def test_native_markdown_table_embedding_text_keeps_rows_without_source_text() -> None:
    """原生 Markdown 表格没有替代来源文本时，索引文本仍保留表头、分隔与行边界。"""
    markdown = (
        "# Paper\n\n## 4.3.2. Ablation studies\n\n"
        "| Model | RMSE |\n| --- | --- |\n| Ours | 4.42 |\n| Base | 5.10 |\n"
    )

    chunks = parser.parse_markdown_text(markdown, "paper-md-table")
    table = next(chunk for chunk in chunks if chunk["block_type"] == "table")

    assert "| Model | RMSE |" in table["embedding_text"]
    assert "| Ours | 4.42 |" in table["embedding_text"]
    assert "| Base | 5.10 |" in table["embedding_text"]
    # 整表不能被压成一行：表头、分隔线与两行数据各自成行
    assert len(table["embedding_text"].splitlines()) >= 4
    assert table["display_text"] == table["text"]


def test_native_markdown_paragraph_embedding_text_still_flattens_table_pipes() -> None:
    """非表格块不受本次修复影响，正文里的竖线仍按通用清洗语义处理。"""
    markdown = "# Paper\n\n## 3 Method\n\nplain text with | pipe and *marker*.\n"

    chunks = parser.parse_markdown_text(markdown, "paper-md-paragraph")
    paragraph = next(chunk for chunk in chunks if chunk["block_type"] == "paragraph")

    assert "|" not in paragraph["embedding_text"]
    assert "*" not in paragraph["embedding_text"]


def test_table_embedding_text_marks_merged_cells_without_duplicating_values() -> None:
    """合并单元格只标注跨度：列对应关系可辨，同一数值不会被复制成多列。"""
    chunks = parser.parse_markdown_text(
        "## 4.1 Datasets\n\nplaceholder\n",
        "paper-merge",
        content_list=[
            {"type": "text", "text_level": 2, "text": "4.1 Datasets", "page_idx": 0},
            {
                "type": "table",
                "page_idx": 0,
                "table_body": (
                    "<table><tr><td rowspan=\"2\">Model</td><td colspan=\"2\">PEMS03</td></tr>"
                    "<tr><td>MAE</td><td>RMSE</td></tr>"
                    "<tr><td>LSTM</td><td>20.62</td><td>33.54</td></tr></table>"
                ),
            },
        ],
    )
    table = next(chunk for chunk in chunks if chunk["block_type"] == "table")

    assert "Model [rowspan=2] | PEMS03 [colspan=2]" in table["embedding_text"]
    assert table["embedding_text"].count("20.62") == 1
    assert table["embedding_text"].count("PEMS03") == 1


def test_formula_embedding_text_keeps_subscript_and_superscript() -> None:
    """公式块的索引文本保留 `_` / `^`，不改动显示文本。"""
    chunks = parser.parse_markdown_text(
        "## 3 Method\n\n$$L = \\alpha L_1 + \\beta L_2$$\n",
        "paper-formula",
    )
    formula = next(chunk for chunk in chunks if chunk["block_type"] == "formula")

    assert "L_1" in formula["embedding_text"]
    assert "L_2" in formula["embedding_text"]
    assert formula["embedding_text"].endswith(formula["display_text"])


def test_inline_formula_inside_paragraph_keeps_subscript() -> None:
    """段落里的行内公式同样受保护，公式外的 Markdown 标记照旧剥掉。"""
    chunks = parser.parse_markdown_text(
        "## 3 Method\n\nThe objective is $L_1 + L_2$ in **training** and _nothing_ more.\n",
        "paper-inline",
    )
    paragraph = next(chunk for chunk in chunks if chunk["block_type"] == "paragraph")

    assert "$L_1 + L_2$" in paragraph["embedding_text"]
    assert "training" in paragraph["embedding_text"]
    assert "**" not in paragraph["embedding_text"]
    assert "L1" not in paragraph["embedding_text"]


def test_paragraph_embedding_text_still_strips_underscore_outside_formula() -> None:
    """公式外的下划线仍是 Markdown 强调符，正文口径不因本次修复改变。"""
    assert parser._strip_markdown_for_embedding("a_b and *c*") == "ab and c"
