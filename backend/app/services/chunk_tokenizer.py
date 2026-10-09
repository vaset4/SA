"""API 分块的独立分词文件与计数器，作者：xxx。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from app.config import DEFAULT_TOKENIZER_PATH as DEFAULT_TOKENIZER_PATH
from app.config import DEFAULT_TOKENIZER_SHA256 as DEFAULT_TOKENIZER_SHA256
TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
TOKENIZER_URL = f"https://huggingface.co/BAAI/bge-m3/resolve/{TOKENIZER_REVISION}/tokenizer.json"
BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApiTokenCounter:
    """【文档分块】【API 计数】只持有分词规则，不加载检索模型或执行网络请求。"""

    _tokenizer: Any
    sha256: str
    engine_version: str

    def count(self, text: str) -> int:
        """参数为原文；返回无填充、无截断、无额外特殊标记的完整 token 数。"""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def identity(self) -> dict[str, str | bool]:
        """无参数；返回实际加载规则的身份，供解析检查点判断能否复用。"""
        return {"engine": self.engine_version, "sha256": self.sha256,
                "add_special_tokens": False, "truncation": False, "padding": False}


def load_api_token_counter(path: str | Path, expected_sha256: str) -> ApiTokenCounter:
    """参数为分词文件路径和预期 SHA-256；核对同一份字节后返回离线计数器。"""
    from tokenizers import Tokenizer

    source = Path(path)
    if not source.is_absolute():
        source = BACKEND_ROOT / source
    if not source.is_file():
        raise ValueError("【文档分块】【分词文件缺失】请运行 python backend/scripts/prepare_chunk_tokenizer.py，"
                         "或检查 CHUNK_TOKENIZER_PATH")
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError("【文档分块】【分词文件损坏】SHA-256 与 CHUNK_TOKENIZER_SHA256 不符，不能继续分块")
    tokenizer = Tokenizer.from_str(content.decode("utf-8"))
    # 1. 【文档分块】【计数约定】必须统计完整正文，不让文件内的推理配置裁掉或补齐文字
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return ApiTokenCounter(tokenizer, digest, f"tokenizers:{version('tokenizers')}")
