"""独立分词文件、完整计数与处理身份回归，作者：xxx。"""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from app.services import document_parser
from app.services.chunk_tokenizer import DEFAULT_TOKENIZER_PATH, DEFAULT_TOKENIZER_SHA256, load_api_token_counter
from app.services.ingest.upload_artifacts import fingerprint
from app.services.ingest.upload_parse_service import UploadDocumentParser


@pytest.fixture(autouse=True)
def clear_counter_cache():
    """无参数；每例隔离进程级缓存，配置替身恢复后不遗留测试计数器。"""
    document_parser._get_token_counter.cache_clear()
    yield
    document_parser._get_token_counter.cache_clear()


def tokenizer_file(path, *, padded=False):
    """参数为路径与推理配置开关；生成真正可加载的最小词表，返回字节哈希。"""
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if padded:
        tokenizer.enable_padding(length=8)
        tokenizer.enable_truncation(max_length=2)
    tokenizer.save(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loaded_file_cannot_truncate_or_pad_counting(tmp_path):
    """参数为目录；词表带推理截断/填充时仍统计完整正文，不把预算判断变成固定数。"""
    path = tmp_path / "tokenizer.json"
    digest = tokenizer_file(path, padded=True)
    counter = load_api_token_counter(path, digest)
    assert counter.count("alpha beta alpha beta") == 4
    assert counter.count("alpha") == 1
    assert counter.identity()["sha256"] == digest


def test_missing_and_modified_files_fail_before_counting(tmp_path):
    """参数为目录；缺文件和修改后的词表均不得退回字符数或联网补文件。"""
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="分词文件缺失"):
        load_api_token_counter(path, DEFAULT_TOKENIZER_SHA256)
    digest = tokenizer_file(path)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_api_token_counter(path, digest)


def test_real_bge_counter_works_without_transformers(monkeypatch):
    """参数为替身；使用准备好的真实 BGE 词表计数，API 入口不能触碰 transformers。"""
    monkeypatch.setattr(document_parser.settings, "embedding_mode", "api")
    monkeypatch.setitem(sys.modules, "transformers", None)
    counter = document_parser._get_token_counter()
    expected = load_api_token_counter(DEFAULT_TOKENIZER_PATH, DEFAULT_TOKENIZER_SHA256)
    assert counter.count("研究方法与 experimental evidence.") == expected.count("研究方法与 experimental evidence.")
    assert counter.count("alpha " * 2000) > 1000
    assert counter.identity()["sha256"] == DEFAULT_TOKENIZER_SHA256


def test_rule_change_changes_checkpoint_identity(tmp_path, monkeypatch):
    """参数为目录和配置替身；分词文件改变必须改变解析检查点身份，不能复用旧结果。"""
    path = tmp_path / "tokenizer.json"
    digest = tokenizer_file(path)
    settings = document_parser.settings
    monkeypatch.setattr(settings, "embedding_mode", "api")
    monkeypatch.setattr(settings, "chunk_tokenizer_path", str(path))
    monkeypatch.setattr(settings, "chunk_tokenizer_sha256", digest)
    uploader = UploadDocumentParser(runner=None, client=SimpleNamespace(base_url="offline"),
        source={"path": "unused", "sha256": "a" * 64, "size": 1}, filename="paper.pdf", root=tmp_path)
    before = uploader._parameters()
    digest = tokenizer_file(path, padded=True)
    monkeypatch.setattr(settings, "chunk_tokenizer_sha256", digest)
    document_parser._get_token_counter.cache_clear()
    after = uploader._parameters()
    assert fingerprint(before) != fingerprint(after)
    assert before["chunking"]["tokenizer"]["sha256"] != after["chunking"]["tokenizer"]["sha256"]


def test_local_mode_keeps_existing_model_tokenizer(tmp_path, monkeypatch):
    """参数为目录和依赖替身；local 继续从配置模型加载原计数器，不读取 API 词表。"""
    path = tmp_path / "tokenizer.json"
    tokenizer_file(path)
    backend = Tokenizer.from_file(str(path))
    class LocalTokenizer:
        """显式替代模型加载端口，底层仍执行真实词表编码。"""
        backend_tokenizer = backend

        def encode(self, text, *, add_special_tokens):
            """参数为原文和特殊标记开关；返回本地接口使用的 token ID 列表。"""
            return backend.encode(text, add_special_tokens=add_special_tokens).ids
    calls = []
    def load(source, **options):
        """参数为模型目录和加载选项；记录配置并返回真实计数能力。"""
        calls.append((source, options))
        return LocalTokenizer()
    monkeypatch.setattr(document_parser.settings, "embedding_mode", "local")
    monkeypatch.setattr(document_parser, "configure_offline_env", lambda: None)
    monkeypatch.setattr(document_parser, "resolve_local_model_source", lambda value: "configured-local-model")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    counter = document_parser._get_token_counter()
    assert counter.count("alpha beta") == 2
    assert calls == [("configured-local-model", {"local_files_only": True, "use_fast": True})]


def test_prepare_writes_verified_bytes_and_rejects_bad_existing_file(tmp_path, monkeypatch):
    """参数为目录和 HTTP 替身；准备命令保存完整核验文件，已有坏文件保持原样且明确失败。"""
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_chunk_tokenizer.py"
    spec = importlib.util.spec_from_file_location("prepare_test_tokenizer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = tmp_path / "fixture.json"
    digest = tokenizer_file(fixture)
    content = fixture.read_bytes()
    monkeypatch.setattr(module, "DEFAULT_TOKENIZER_SHA256", digest)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: BytesIO(content))
    target = tmp_path / "prepared/tokenizer.json"
    assert module.prepare_tokenizer(target) == target
    assert target.read_bytes() == content
    assert module.prepare_tokenizer(target) == target
    target.write_bytes(b"broken")
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare_tokenizer(target)
    assert target.read_bytes() == b"broken"
