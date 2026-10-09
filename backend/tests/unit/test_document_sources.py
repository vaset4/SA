"""规范正文、跨页来源和发布存档回归，作者：xxx。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from app.models.upload_models import (
    ArtifactIntegrityError, UploadConflictError, UploadExecution, UploadPausedError,
)
from app.repositories.upload_repository import UploadRepository
from app.services import document_parser as parser
from app.services.ingest import document_parse_service, mineru_markdown_service, upload_parse_service, upload_pipeline
from app.services.ingest.upload_artifacts import fingerprint, read_json_artifact, write_artifact, write_json_artifact
from app.services.ingest.upload_content_identity import CONTENT_IDENTITY_VERSION
from app.services.ingest.upload_index_service import UploadBatchIndexer
from app.services.ingest.upload_parse_service import PARSER_FORMAT_VERSION, UploadDocumentParser
from app.services.ingest.upload_recovery import UploadStageRunner
from app.services.upload_read_service import validate_content_schema
from source_file_test_ports import install_file_ports


class WordCounter:
    """仅计算离线测试的词数量，不加载模型或 tokenizer。"""

    def count(self, text: str) -> int:
        """参数为正文；返回空白分隔的词数量。"""
        return len(text.split())

    def identity(self) -> dict:
        """无参数；返回该测试明确采用的计数身份，不冒充真实模型分词器。"""
        return {"engine": "fixture-word-counter", "version": 1}


@pytest.fixture(autouse=True)
def source_chunk_settings(monkeypatch):
    """参数为测试替身；固定小额度，强制覆盖来源传播和父/子重叠路径。"""
    install_file_ports(monkeypatch)
    monkeypatch.setattr(parser, "_get_token_counter", WordCounter)
    for key, value in {"structure_merge_enabled": True, "chunk_merge_floor": 5,
        "parent_chunk_max_size": 14, "parent_chunk_overlap_tokens": 2,
        "child_chunk_target_size": 5, "child_chunk_overlap_tokens": 1}.items():
        monkeypatch.setattr(parser.settings, key, value, raising=False)


def _source_coverage(document: dict, fragment_key: str) -> dict[str, set[int]]:
    """参数为解析文档和片段字段；返回各来源实际覆盖字符位置，用于检查原文没有被裁丢。"""
    coverage = {key: set() for key in document["source_items"]}
    for chunk in document["chunks"]:
        for fragment in chunk[fragment_key]:
            coverage[fragment["source_item"]].update(range(fragment["source_start"], fragment["source_end"]))
    return coverage


def test_repeated_cross_page_text_preserves_distinct_sources_and_both_overlaps():
    """相同句子跨页重复时，父/子重叠仍指向各自实际项目，不靠文本反查首个匹配。"""
    body = "重复 😀 alpha beta. gamma delta epsilon zeta. eta theta iota kappa. lambda mu nu xi."
    document = parser.parse_markdown_document("", "paper", content_list=[
        {"type": "text", "text_level": 1, "text": "3 Method", "page_idx": 0},
        {"type": "text", "text": body, "page_idx": 0},
        {"type": "text", "text": body, "page_idx": 1},
    ])
    validate_content_schema(document)
    assert len(document["source_items"]) == 2
    assert {item["page_num"] for item in document["source_items"].values()} == {1, 2}
    for key in ("source_fragments", "parent_source_fragments"):
        coverage = _source_coverage(document, key)
        for source_id, item in document["source_items"].items():
            assert {index for index, char in enumerate(item["text"]) if not char.isspace()} <= coverage[source_id]
    parents = {chunk["parent_id"]: chunk for chunk in document["chunks"]}
    assert len(parents) > 2
    ranges = [fragment for chunk in parents.values() for fragment in chunk["parent_source_fragments"]]
    assert any(a["source_item"] == b["source_item"] and a["source_start"] < b["source_end"]
        and b["source_start"] < a["source_end"] for index, a in enumerate(ranges) for b in ranges[index + 1:])
    assert any(chunk["source_fragments"] != chunk["parent_source_fragments"] for chunk in document["chunks"])


def test_markdown_records_original_offsets_without_inventing_physical_pages():
    """Markdown 项目记录原文件位置；没有显式页标记时不能伪装成第一页。"""
    markdown = "# Method\r\n\r\nalpha 😀 beta.\r\n\r\n<!-- page: 7 -->\r\n\r\ngamma delta.\r\n"
    document = parser.parse_markdown_document(markdown, "paper")
    validate_content_schema(document)
    items = list(document["source_items"].values())
    assert [item["page_num"] for item in items] == [None, 7]
    assert markdown[items[0]["source_start"]:items[0]["source_end"]].strip() == "alpha 😀 beta."
    assert "<!--" not in "".join(chunk["text"] for chunk in document["chunks"])
    assert all(chunk["page_num"] == 0 for chunk in document["chunks"] if len(chunk["source_fragments"]) > 1)


@pytest.mark.parametrize("page_idx", [True, -1, 0.0, "invalid", "-1"])
def test_mineru_invalid_page_index_is_not_converted_to_a_physical_page(tmp_path, page_idx):
    """参数为临时目录和非法页索引；转换入口明确拒绝坏位置，不能变成合法页号或缺省无页。"""
    artifacts = {"markdown_text": "# Method\n\nalpha beta.", "content_list": [
        {"type": "text", "text_level": 1, "text": "Method", "page_idx": 0},
        {"type": "text", "text": "alpha beta.", "page_idx": page_idx},
    ]}
    with pytest.raises(ValueError, match="页索引"):
        document_parse_service.parse_pdf_artifacts(tmp_path / "paper.pdf", "paper", artifacts)


def test_epub_is_rejected_before_parsing(tmp_path):
    """参数为临时目录；EPUB 扩展名和解析注册同时拒绝，不再接受不支持的文档。"""
    with pytest.raises(ValueError, match="不支持"):
        document_parse_service.validate_extension("paper.epub")
    with pytest.raises(parser.UnsupportedFileTypeError):
        parser.get_parser_registry().parse_document(str(tmp_path / "paper.epub"), "paper")


def test_tables_formulas_and_labels_remain_readable_and_atomic(monkeypatch):
    """表格行列与合并单元格说明进入阅读正文，公式和标签保留可核验来源。"""
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", 100)
    document = parser.parse_markdown_document("", "paper", content_list=[
        {"type": "text", "text_level": 1, "text": "4 Results", "page_idx": 2},
        {"type": "table", "page_idx": 2, "table_caption": "Table 1 Results", "table_footnote": "note",
            "table_body": '<table><tr><td rowspan="2">Model</td><td colspan="2">Score</td></tr>'
                '<tr><td>A</td><td>B</td></tr><tr><td>Ours</td><td>1</td><td>2</td></tr></table>'},
        {"type": "text", "text": "RMSE:", "page_idx": 3},
        {"type": "equation", "latex": r"L = x_1^2 + y_2", "page_idx": 3},
    ])
    validate_content_schema(document)
    table, formula = document["chunks"]
    assert "Model [rowspan=2] | Score [colspan=2]" in table["text"]
    assert "\n| Ours | 1 | 2 |" in table["text"]
    assert table["text"] == table["full_text"] == table["display_text"]
    assert "RMSE:" in formula["text"] and r"x_1^2 + y_2" in formula["text"]
    assert len(formula["source_fragments"]) == 2
    assert formula["text"] == formula["full_text"]


@pytest.mark.parametrize("body", ["<div>broken</div>", ""])
def test_invalid_html_table_does_not_publish_caption_as_complete_body(body):
    """存在表格标题但表格正文无行列结构时明确失败，不将标题当作成功表格。"""
    with pytest.raises(ValueError, match="行列结构"):
        parser.parse_markdown_document("", "paper", content_list=[
            {"type": "text", "text_level": 1, "text": "Results"},
            {"type": "table", "table_caption": "Table 1", "table_body": body},
        ])


def test_markup_normalization_retains_offsets_into_original_markdown(tmp_path):
    """清洗排版标签改变正文长度时，原始项目坐标仍指向未清洗的 Markdown 产物。"""
    markdown = "# Meth<sub>o</sub>d\n\nalpha <sub>beta.</sub> gamma.\n"
    document = document_parse_service.parse_pdf_artifacts(tmp_path / "paper.pdf", "paper", {
        "markdown_text": markdown, "content_list": [], "parser_mode": "mineru_online"})
    validate_content_schema(document)
    item = next(iter(document["source_items"].values()))
    assert item["text"] == "alpha beta. gamma."
    assert markdown[item["source_start"]:item["source_end"]].strip() == "alpha <sub>beta.</sub> gamma."


@pytest.mark.parametrize("formula", ["$x (1) y. z$", "$$x (1) y. z$$", "$x\n\n y. z$"])
def test_inline_formula_survives_parent_child_sentence_and_enumeration_boundaries(monkeypatch, formula):
    """段落内公式跨句点、枚举或空行时，父子切分也必须保留完整表达式。"""
    for key, value in {"parent_chunk_max_size": 6, "parent_chunk_overlap_tokens": 0,
        "child_chunk_target_size": 4, "child_chunk_overlap_tokens": 0}.items():
        monkeypatch.setattr(parser.settings, key, value)
    document = parser.parse_markdown_document("", "paper", content_list=[
        {"type": "text", "text_level": 1, "text": "Method"},
        {"type": "text", "text": "Before. " + formula + " After. More."},
    ])
    validate_content_schema(document)
    for chunk in document["chunks"]:
        if "$" in chunk["text"]:
            assert formula in chunk["text"]
        if "$" in chunk["full_text"]:
            assert formula in chunk["full_text"]


class LocalRunner:
    """只执行内存动作的上传端口，禁止数据库访问。"""

    def __init__(self):
        """无参数；构造固定上传身份与检查调用记录。"""
        self.execution = SimpleNamespace(paper_id="paper", batch_id="batch", parse_id="parse", user_id="owner")
        self.repository = self
        self.inputs = {}

    async def run(self, stage, action, *, inputs, **kwargs):
        """参数为阶段、动作和输入；执行动作并保存检查点输入用于格式身份断言。"""
        self.inputs[stage] = deepcopy(inputs)
        return await action()

    async def check_execution(self, execution):
        """参数为执行身份；离线测试只核对传入身份，无外部读写。"""
        assert execution == self.execution

    async def stage_batch(self, execution, evidence):
        """参数为执行身份和发布证据；只保存在测试内存中。"""
        self.evidence = evidence


@pytest.mark.asyncio
async def test_local_parse_and_publication_persist_sources_and_keep_v3_addresses(tmp_path, monkeypatch):
    """参数为临时目录与替身；验证本地来源存档、处理版本与 v3 发布地址，无返回值。"""
    monkeypatch.setattr(parser.settings, "embedding_dimension", 2)
    reference = write_artifact(tmp_path / "sources", b"# Method\n\nalpha beta gamma.\n", suffix=".md")
    runner = LocalRunner()
    uploader = UploadDocumentParser(runner=runner, client=SimpleNamespace(base_url="offline"),
        source=reference, filename="paper.md", root=tmp_path)
    parsed = await uploader.parse()
    assert PARSER_FORMAT_VERSION == runner.inputs["document"]["version"] == 4
    document = read_json_artifact(parsed["document_ref"], root=tmp_path)
    assert document["content_schema_version"] == 1
    assert all(item["artifact_ref"] == reference for item in document["source_items"].values())
    store = SimpleNamespace(collection_name="offline", write_upload_batch=lambda batch, rows: rows)
    encoder = SimpleNamespace(embed_texts=lambda texts: [[0.25, 0.5] for text in texts])
    indexer = UploadBatchIndexer(runner=runner, store=store, encoder=encoder, publication=runner, root=tmp_path)
    result = await indexer.prepare({**parsed, "source_ref": reference}, paper={"year": 2026}, metadata_missing=[])
    manifest = read_json_artifact(result["manifest_ref"], root=tmp_path)
    assert manifest["format"] == 3 and manifest["content_schema_version"] == 1
    assert manifest["content_identity"]["version"] == CONTENT_IDENTITY_VERSION == 1
    assert manifest["source_items"] == document["source_items"]
    assert manifest["chunks"][0]["chunk_id"] == "v3:batch:0"
    assert manifest["chunks"][0]["source_fragments"] == document["chunks"][0]["source_fragments"]
    validate_content_schema(manifest)


def test_mineru_conversion_binds_source_to_verified_content_artifact(tmp_path):
    """MinerU 正文项目回连其内容产物，不能只保存临时列表序号。"""
    content = [{"type": "text", "text_level": 1, "text": "Method", "page_idx": 0},
        {"type": "text", "text": "alpha beta gamma.", "page_idx": 1}]
    content_ref = write_json_artifact(tmp_path / "bundle", content)
    markdown_ref = write_artifact(tmp_path / "bundle", b"# Method\n\nalpha beta gamma.", suffix=".md")
    source = write_artifact(tmp_path / "source", b"%PDF-test", suffix=".pdf")
    uploader = UploadDocumentParser(runner=LocalRunner(), client=SimpleNamespace(base_url="offline"),
        source=source, filename="paper.pdf", root=tmp_path)
    document = uploader._convert_bundle({"markdown_ref": markdown_ref, "content_ref": content_ref,
        "artifact_dir": str(tmp_path / "bundle"), "normalized_page_count": 2, "resources": {}})
    validate_content_schema(document)
    assert list(document["source_items"].values())[0]["artifact_ref"] == content_ref


@pytest.mark.asyncio
async def test_cached_legacy_document_cannot_be_relabelled_as_new_parser_output(tmp_path):
    """新解析身份取到缺来源的旧缓存必须失败，不能将旧正文默认为新版。"""
    source = write_artifact(tmp_path / "source", b"# Method\n\nalpha.", suffix=".md")
    legacy = {"chunks": [{"text": "alpha.", "full_text": "alpha."}], "full_text": "alpha."}
    document_ref = write_json_artifact(tmp_path / "legacy", legacy)
    runner = LocalRunner()

    async def old_checkpoint(stage, action, **kwargs):
        """参数为检查点请求；返回固定旧产物以检查版本拒绝。"""
        return {"document_ref": document_ref}

    runner.run = old_checkpoint
    uploader = UploadDocumentParser(runner=runner, client=SimpleNamespace(base_url="offline"),
        source=source, filename="paper.md", root=tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="格式版本"):
        await uploader.parse()
    assert read_json_artifact(document_ref, root=tmp_path) == legacy


# 【来源测试】【冻结样例】CPiRi sidecar 第 83 项；完整原文来自 caption-parser-probe.json
# 原 sidecar SHA256：1dae62233cedd6aeecaa3c2ec7c8ed2a378f14e987c31834bcf5b499ede01893
# 实际 chart_footnote 为空；下方非空脚注用例为独立合成输入
CHART_TEST_PARENT_LIMIT = 512
LEGACY_PROCESSING_VERSION = 2


@pytest.fixture
def frozen_chart_item():
    """无参数；返回冻结原始 chart 项的新副本，保留原图题、空脚注、页码和资产路径。"""
    return {
        "type": "chart",
        "img_path": "images/4de307f1f866f7d649e13eab19f0896319d33643c2d38b56aa3a0a5de74d3d1e.jpg",
        "content": "",
        "chart_caption": [
            "Figure 4: Analysis of inductive generalization to unseen channels and the impact of permutationinvariant regularization strategy. Performance is evaluated with and without the strategy as the percentage of available training channels varies. The results show that the strategy consistently improves performance, with the benefits becoming more pronounced in low-data regimes. For clarity, WAPE (%) values are annotated directly on the corresponding data points."
        ],
        "chart_footnote": [],
        "bbox": [
            699,
            112,
            821,
            199
        ],
        "page_idx": 8
    }



@pytest.fixture
def chart_bundle(tmp_path, monkeypatch, frozen_chart_item):
    """参数为临时目录、替身和原始图表；返回微型文档及真实文件引用，不访问供应商。"""
    monkeypatch.setattr(parser.settings, "parent_chunk_max_size", CHART_TEST_PARENT_LIMIT)
    content = [{"type": "text", "text_level": 1, "text": "4 Results", "page_idx": 8}, frozen_chart_item]
    caption = "\n".join(frozen_chart_item["chart_caption"])
    markdown = "# 4 Results\n\n" + caption
    content_ref = write_json_artifact(tmp_path / "bundle", content)
    markdown_ref = write_artifact(tmp_path / "bundle", markdown.encode("utf-8"), suffix=".md")
    asset_ref = write_artifact(tmp_path / "assets", b"isolated-asset-reference", suffix=".jpg")
    source_ref = write_artifact(tmp_path / "source", b"isolated-pdf-source", suffix=".pdf")
    bundle = {"markdown_ref": markdown_ref, "content_ref": content_ref,
        "artifact_dir": str(tmp_path / "bundle"), "normalized_page_count": 9,
        "resources": {frozen_chart_item["img_path"]: asset_ref}}
    return SimpleNamespace(bundle=bundle, source_ref=source_ref, content=content, caption=caption)


def test_native_chart_caption_reaches_atomic_sources(tmp_path, chart_bundle, frozen_chart_item):
    """参数为隔离目录、图表产物和原项；验证规范化到结构块、来源和父子正文，无返回值。"""
    original = deepcopy(chart_bundle.content)
    normalized = mineru_markdown_service.normalize_content_list({"content_list": chart_bundle.content})
    cleaned = document_parse_service._normalize_mineru_markup({"markdown_text": "", "content_list": normalized})
    assert cleaned["content_list"][1] == frozen_chart_item
    assert frozen_chart_item["chart_footnote"] == []
    uploader = UploadDocumentParser(runner=LocalRunner(), client=SimpleNamespace(base_url="offline"),
        source=chart_bundle.source_ref, filename="paper.pdf", root=tmp_path)

    # 1. 【来源测试】【正式转换】调用实际 bundle 入口，保留规范化、结构解析和来源绑定
    document = uploader._convert_bundle(chart_bundle.bundle)
    validate_content_schema(document)
    assert document["content_schema_version"] == 1
    assert len(document["chunks"]) == len(document["source_items"]) == 1
    chunk = document["chunks"][0]
    assert chunk["block_type"] == "figure_caption"
    assert chunk["page_num"] == 9
    assert chunk["asset_ref"] == frozen_chart_item["img_path"]
    assert chunk["text"] == chunk["display_text"] == chunk["full_text"] == chart_bundle.caption
    assert chunk["embedding_text"] == "4 Results\n" + chart_bundle.caption
    source = next(iter(document["source_items"].values()))
    assert source["source_kind"] == "content_list" and source["source_index"] == 1
    assert source["text"] == chart_bundle.caption and source["page_num"] == 9
    assert source["artifact_ref"] == chart_bundle.bundle["content_ref"]

    # 2. 【来源测试】【逐字定位】微型列表保留实际输入序号，父子片段各自回到同一原文区间
    for field, fragment_key in (("text", "source_fragments"), ("full_text", "parent_source_fragments")):
        assert len(chunk[fragment_key]) == 1
        fragment = chunk[fragment_key][0]
        assert fragment["source_item"] == source["source_item"]
        assert fragment["page_num"] == 9
        assert (fragment["source_start"], fragment["source_end"]) == (0, len(chart_bundle.caption))
        assert chunk[field][fragment["text_start"]:fragment["text_end"]] == chart_bundle.caption
    assert chart_bundle.content == original


@pytest.mark.parametrize("block_type", ["chart", "image", "figure"])
def test_native_chart_fields_do_not_change_image_or_figure_aliases(block_type):
    """参数为图像类型；验证合成中文图题/脚注只对 chart 优先，旧 image/figure 保持，无返回值。"""
    item = {"type": block_type, "page_idx": 2, "img_path": "images/synthetic.png",
        "chart_caption": ["图 1：中文原生图题"], "chart_footnote": ["合成脚注，非原论文脚注"],
        "img_caption": ["旧图题"], "figure_caption": ["次级图题"], "caption": ["通用图题"],
        "img_footnote": ["旧脚注"], "near_text": ["邻近文字"], "footnote": ["通用脚注"]}
    original = deepcopy(item)
    block = parser._content_item_to_markdown_block(item)
    expected = "图 1：中文原生图题\n合成脚注，非原论文脚注" if block_type == "chart" else "旧图题\n旧脚注"
    assert block == {"type": "figure_caption", "text": expected,
        "page_num": 3, "asset_ref": "images/synthetic.png"}
    assert item == original


@pytest.mark.parametrize("empty_native", [None, "", [], [" \n "]])
@pytest.mark.parametrize("caption_key,footnote_key", [
    ("img_caption", "img_footnote"), ("figure_caption", "near_text"), ("caption", "footnote"),
])
def test_empty_native_chart_fields_keep_existing_aliases(empty_native, caption_key, footnote_key):
    """参数为原生空值和旧字段名；验证空图题/脚注仍使用已支持别名，无返回值。"""
    item = {"type": "chart", "chart_caption": empty_native, "chart_footnote": empty_native,
        caption_key: ["旧图题"], footnote_key: ["合成旧脚注"], "image_path": "images/legacy.png"}
    block = parser._content_item_to_markdown_block(item)
    assert block == {"type": "figure_caption", "text": "旧图题\n合成旧脚注",
        "page_num": 0, "asset_ref": "images/legacy.png"}


@pytest.mark.parametrize("native_caption,native_note,expected", [
    (["原生图题"], [], "原生图题\n旧脚注"),
    ([], ["合成原生脚注"], "旧图题\n合成原生脚注"),
    (["旧图题"], ["旧脚注"], "旧图题\n旧脚注"),
])
def test_chart_caption_and_footnote_choose_independently(native_caption, native_note, expected):
    """参数为合成原生字段及期望正文；验证各字段独立选择且同一信息不重复拼接，无返回值。"""
    block = parser._content_item_to_markdown_block({"type": "chart",
        "chart_caption": native_caption, "chart_footnote": native_note,
        "img_caption": ["旧图题"], "img_footnote": ["旧脚注"]})
    assert block["text"] == expected


def test_synthetic_chart_footnote_retains_source_ranges_after_normalization(tmp_path):
    """参数为临时目录；验证合成中文图题与非空脚注经过排版清洗和父子来源映射，无返回值。"""
    item = {"type": "chart", "chart_caption": ["图 2：温度<sub>变化</sub>"],
        "chart_footnote": ["合成脚注：每小时<sup>一次</sup>，并非原论文记录"],
        "page_idx": 1, "img_path": "images/synthetic.png"}
    original = deepcopy(item)
    document = document_parse_service.parse_pdf_artifacts(tmp_path / "paper.pdf", "paper", {
        "markdown_text": "# Results\n\n合成图表正文", "content_list": [
            {"type": "text", "text_level": 1, "text": "Results", "page_idx": 1}, item]})
    validate_content_schema(document)
    expected = "图 2：温度变化\n合成脚注：每小时一次，并非原论文记录"
    chunk = document["chunks"][0]
    assert len(document["chunks"]) == 1
    assert chunk["text"] == chunk["display_text"] == chunk["full_text"] == expected
    assert chunk["asset_ref"] == item["img_path"] and chunk["page_num"] == 2
    source = next(iter(document["source_items"].values()))
    assert source["text"] == expected
    for key in ("source_fragments", "parent_source_fragments"):
        assert len(chunk[key]) == 1
        fragment = chunk[key][0]
        assert fragment["source_item"] == source["source_item"]
        assert (fragment["source_start"], fragment["source_end"]) == (0, len(expected))
    assert item == original


def test_chart_without_caption_or_footnote_remains_absent():
    """无参数；确认原生和别名均无正文时仍不产生空图题结构，无返回值。"""
    assert parser._content_item_to_markdown_block({"type": "chart",
        "chart_caption": [], "chart_footnote": [], "img_path": "images/empty.png"}) is None


@pytest.fixture
def checkpoint_ports(tmp_path, monkeypatch, chart_bundle):
    """参数为隔离目录、替身和图表产物；返回真实检查点仓储及假连接，SQL 不触达数据库。"""
    connection = AsyncMock()
    connection.transaction = MagicMock()
    connection.transaction.return_value.__aenter__.return_value = connection
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = connection
    repository = UploadRepository(AsyncMock(return_value=pool), tasks=AsyncMock())
    execution = UploadExecution(task_id="task", worker_id="worker", paper_id="paper", user_id="owner",
        parse_id="old-parse", batch_id="new-batch", previous_batch_id="published-batch")
    runner = UploadStageRunner(repository, execution, sleeper=AsyncMock())
    uploader = UploadDocumentParser(runner=runner, client=SimpleNamespace(base_url="offline"),
        source=chart_bundle.source_ref, filename="paper.pdf", root=tmp_path)
    settings = SimpleNamespace(mineru_enabled=True,
        **{key: getattr(parser.settings, key) for key in upload_parse_service.CHUNK_PARAMETER_KEYS})
    monkeypatch.setattr(upload_parse_service, "get_settings", lambda: settings)
    old_artifact = write_json_artifact(tmp_path / "previous", {"saved_text": "旧检查点的不可变产物"})
    old_payload = {"document_ref": old_artifact}
    current = uploader._parameters()
    previous = {**current, "version": LEGACY_PROCESSING_VERSION}
    completed = {"parse_id": execution.parse_id, "stage": "document", "state": "completed",
        "input_hash": fingerprint(previous), "operation_id": "old-operation", "payload": old_payload}
    return SimpleNamespace(connection=connection, repository=repository, uploader=uploader,
        current=current, completed=completed, old_artifact=old_artifact)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["document", "mineru_slot"])
async def test_completed_v2_checkpoint_conflicts_before_action(checkpoint_ports, stage, tmp_path):
    """参数为隔离仓储、旧阶段和目录；验证旧 completed 输入冲突先于复用/动作且不写产物，无返回值。"""
    ports = checkpoint_ports
    completed = {**ports.completed, "stage": stage}
    ports.connection.fetchrow.return_value = completed
    before_row = deepcopy(completed)
    before_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    action = AsyncMock(side_effect=AssertionError("处理身份冲突不得执行阶段动作"))
    assert ports.current["version"] == 4 and LEGACY_PROCESSING_VERSION == 2
    assert ports.current["version"] == PARSER_FORMAT_VERSION

    with pytest.raises(UploadConflictError, match="需要明确重新处理"):
        await ports.uploader.runner.run(stage, action, inputs=ports.current, repeatable=stage == "document")
    action.assert_not_awaited()
    ports.connection.fetchrow.assert_awaited_once()
    assert ports.connection.fetchrow.await_args.args[1:] == ("old-parse", stage, "owner", "paper")
    ports.connection.fetchval.assert_not_awaited()
    ports.connection.execute.assert_not_awaited()
    assert completed == before_row
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before_files


@pytest.mark.asyncio
async def test_old_document_conflict_pauses_task_without_publication(checkpoint_ports, tmp_path, monkeypatch):
    """参数为隔离仓储、目录和替身；验证真实任务入口保留冲突原因并阻止解析/索引/发布，无返回值。"""
    ports = checkpoint_ports
    execution = ports.uploader.runner.execution
    ports.connection.fetchrow.return_value = ports.completed
    monkeypatch.setattr(ports.repository, "begin_attempt", AsyncMock(return_value=execution))
    monkeypatch.setattr(ports.repository, "progress", AsyncMock())
    monkeypatch.setattr(ports.repository, "get_document", AsyncMock(return_value={
        "source_ref": ports.uploader.source, "filename": "paper.pdf", "library_id": "library"}))
    monkeypatch.setattr(upload_pipeline, "get_upload_repository", lambda: ports.repository)
    monkeypatch.setattr(upload_pipeline, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path)))
    client = Mock(return_value=ports.uploader.client)
    monkeypatch.setattr(upload_pipeline, "MinerUClient", client)
    parse_action = AsyncMock(side_effect=AssertionError("旧文档冲突不得重算"))
    monkeypatch.setattr(UploadDocumentParser, "_parse_document", parse_action)
    indexer = Mock(side_effect=AssertionError("解析冲突不得建立索引"))
    monkeypatch.setattr(upload_pipeline, "UploadBatchIndexer", indexer)
    publication = Mock(side_effect=AssertionError("解析冲突不得发布"))
    monkeypatch.setattr(upload_pipeline, "get_upload_publication_repository", publication)
    before_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    before_row = deepcopy(ports.completed)

    with pytest.raises(UploadPausedError) as error:
        await upload_pipeline.prepare_upload_task({"task_id": execution.task_id})
    assert isinstance(error.value.__cause__, UploadConflictError)
    assert error.value.failure.kind == "invalid_result"
    assert error.value.failure.can_resume is False
    assert "需要明确重新处理" in error.value.failure.detail
    assert error.value.execution == execution
    ports.repository.begin_attempt.assert_awaited_once()
    parse_action.assert_not_awaited()
    indexer.assert_not_called()
    publication.assert_not_called()
    assert ports.completed == before_row
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before_files


@pytest.mark.asyncio
async def test_new_parse_generation_uses_current_chart_mapping_and_reuses_same_version(
    checkpoint_ports, chart_bundle, tmp_path, monkeypatch,
):
    """参数为隔离仓储、图表、目录和替身；验证新代次真实解析与同版复用，旧产物不变，无返回值。"""
    ports = checkpoint_ports
    old_row = deepcopy(ports.completed)
    old_bytes = Path(ports.old_artifact["path"]).read_bytes()
    execution = replace(ports.uploader.runner.execution, parse_id="new-parse")
    uploader = replace(ports.uploader, runner=UploadStageRunner(ports.repository, execution, sleeper=AsyncMock()))
    pending = {"parse_id": execution.parse_id, "stage": "document", "state": "pending",
        "input_hash": fingerprint(ports.current), "operation_id": "new-operation", "payload": None}
    ports.connection.fetchrow.side_effect = [None, pending, {"operation_id": "new-operation"}]
    reserve = AsyncMock(return_value=1)
    monkeypatch.setattr(ports.repository, "_reserve_attempt", reserve)
    bundle = AsyncMock(return_value=chart_bundle.bundle)
    monkeypatch.setattr(UploadDocumentParser, "_mineru_bundle", bundle)
    metadata = Mock(return_value={})
    monkeypatch.setattr(document_parse_service, "extract_document_metadata", metadata)
    action = AsyncMock(wraps=uploader._parse_document)
    monkeypatch.setattr(UploadDocumentParser, "_parse_document", action)

    # 1. 【来源测试】【新代次】SQL 以新 parse_id 查找；仅隔离远端产物取得和无关元信息提取
    parsed = await uploader.parse()
    assert ports.connection.fetchrow.await_args_list[0].args[1:] == ("new-parse", "document", "owner", "paper")
    inserted = ports.connection.fetchrow.await_args_list[1].args
    assert inserted[1] == "new-parse" and inserted[6] == fingerprint(ports.current)
    completed = ports.connection.fetchrow.await_args_list[2].args
    saved_payload = json.loads(completed[1])
    assert completed[2:5] == ("new-parse", "document", "new-operation")
    assert saved_payload["document_ref"] == parsed["document_ref"]
    document = read_json_artifact(parsed["document_ref"], root=tmp_path)
    validate_content_schema(document)
    assert document["chunks"][0]["text"] == chart_bundle.caption
    assert "new-parse" in Path(parsed["document_ref"]["path"]).parts
    assert parsed["document_ref"] != ports.old_artifact
    assert ports.completed == old_row and Path(ports.old_artifact["path"]).read_bytes() == old_bytes

    # 2. 【来源测试】【同版恢复】实际保存结果回读后命中 completed，不重复解析或调用供应商
    ports.connection.fetchrow.side_effect = None
    ports.connection.fetchrow.return_value = {**pending, "state": "completed", "payload": saved_payload}
    assert await uploader.parse() == parsed
    action.assert_awaited_once()
    bundle.assert_awaited_once_with(ports.current)
    reserve.assert_awaited_once()
    assert ports.connection.fetchrow.await_count == 4
