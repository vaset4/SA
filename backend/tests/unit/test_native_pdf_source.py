"""原生 PDF 新提取格式与证据真实性回归，作者：xxx。"""
from importlib.metadata import version

import pytest

from evals.contracts import read_snapshot, snapshot, snapshot_hash
from evals.evidence.alignment import EvidenceDataError, validate_benchmark
from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan, SourceBlock, make_source_snapshot
from evals.evidence.corpus import build_native_source_snapshot, validate_native_source
from pdf_fixtures import write_pdf


def rebuilt(source, **changes):
    """参数为快照与测试篡改字段；重新计算外层身份，以检查内容而非只检查外层哈希。"""
    values = source.model_dump(exclude={"snapshot_id", "schema_version", "locator_version"})
    return make_source_snapshot(**{**values, **changes})


def test_snapshot_keeps_exact_known_text_page_and_parser(tmp_path):
    """参数为目录；由独立生成器写已知内容，逐字校验页原文和物理页位置。"""
    pdf = write_pdf(tmp_path / "source.pdf", ("First evidence.", "", "Last evidence."))
    source = build_native_source_snapshot("paper", pdf, paper_dir=tmp_path)
    assert [(b.block_id, b.page_num, b.text) for b in source.blocks] == [
        ("pypdf-page-000001", 1, "First evidence.\n"),
        ("pypdf-page-000003", 3, "Last evidence.\n"),
    ]
    assert source.parser_engine_version == f"pypdf:{version('pypdf')}"
    assert validate_native_source(source, paper_dir=tmp_path) == pdf


@pytest.mark.parametrize("field,value", [("text", "Fabricated evidence."), ("page_num", 2), ("block_id", "pdf-p001-b000")])
def test_rehashed_but_fabricated_block_is_rejected(tmp_path, field, value):
    """参数为目录与篡改字段；即使重算快照哈希，假正文、页码和旧式块号仍被拒绝。"""
    pdf = write_pdf(tmp_path / "source.pdf")
    source = build_native_source_snapshot("paper", pdf, paper_dir=tmp_path)
    block = SourceBlock(**{**source.blocks[0].model_dump(), field: value})
    modified = rebuilt(source, blocks=(block,))
    with pytest.raises(EvidenceDataError, match="native_pdf_block_mismatch"):
        validate_native_source(modified, paper_dir=tmp_path)


def test_missing_page_and_modified_pdf_are_rejected(tmp_path):
    """参数为目录；快照少一页或磁盘原文件变化不能通过完整性核验。"""
    pdf = write_pdf(tmp_path / "source.pdf", ("First.", "Second."))
    source = build_native_source_snapshot("paper", pdf, paper_dir=tmp_path)
    with pytest.raises(EvidenceDataError, match="block_mismatch"):
        validate_native_source(rebuilt(source, blocks=(source.blocks[0],)), paper_dir=tmp_path)
    write_pdf(pdf, ("Changed evidence.",))
    with pytest.raises(EvidenceDataError, match="checksum_mismatch"):
        validate_native_source(source, paper_dir=tmp_path)


def test_old_extractor_and_parser_version_require_review(tmp_path):
    """参数为目录；旧提取格式和解析器版本不能伪装为当前结果，原快照不改写。"""
    pdf = write_pdf(tmp_path / "source.pdf")
    source = build_native_source_snapshot("paper", pdf, paper_dir=tmp_path)
    artifacts = read_snapshot(source.artifact_manifest_snapshot)
    legacy = rebuilt(source, artifact_manifest_snapshot=snapshot({**artifacts, "extraction_mode": "pymupdf_pdf_text_blocks"}))
    before = legacy.model_dump_json()
    with pytest.raises(EvidenceDataError, match="重新准备并复核"):
        validate_native_source(legacy, paper_dir=tmp_path)
    assert legacy.model_dump_json() == before
    with pytest.raises(EvidenceDataError, match="parser_version_mismatch"):
        validate_native_source(rebuilt(source, parser_engine_version="pypdf:old"), paper_dir=tmp_path)


def test_quote_and_source_identity_checks_still_apply(tmp_path):
    """参数为目录；新提取器不能绕过真实摘录及来源引用校验。"""
    source = build_native_source_snapshot("paper", write_pdf(tmp_path / "source.pdf"), paper_dir=tmp_path)
    quote = "Fabricated"
    span = EvidenceSpan(span_id="quote", source_snapshot_id=source.snapshot_id,
        block_id=source.blocks[0].block_id, start=0, end=len(quote), quote=quote, quote_sha256=snapshot_hash(quote))
    benchmark = EvidenceBenchmark(dataset_version="test-v1", review_status="candidate", sources=(source,),
        spans=(span,), groups=())
    with pytest.raises(EvidenceDataError):
        validate_benchmark(benchmark)


def test_nested_paper_directory_is_literal_and_escape_is_rejected(tmp_path):
    """参数为目录；合法名为 paper 的子目录不被截断，越界路径被明确拒绝。"""
    nested = tmp_path / "paper"
    nested.mkdir()
    pdf = write_pdf(nested / "source.pdf")
    source = build_native_source_snapshot("paper", pdf, paper_dir=tmp_path)
    assert validate_native_source(source, paper_dir=tmp_path) == pdf
    with pytest.raises(EvidenceDataError, match="outside_uploads"):
        build_native_source_snapshot("paper", tmp_path.parent / "outside.pdf", paper_dir=tmp_path)
