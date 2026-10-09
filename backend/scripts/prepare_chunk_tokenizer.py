"""准备默认 BGE-M3 分词文件，不下载模型权重，作者：xxx。"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chunk_tokenizer import (  # noqa: E402
    DEFAULT_TOKENIZER_PATH, DEFAULT_TOKENIZER_SHA256, TOKENIZER_URL, load_api_token_counter,
)

DOWNLOAD_TIMEOUT_SECONDS = 60


def prepare_tokenizer(destination: Path) -> Path:
    """参数为目标文件；已有有效文件直接复用，否则下载固定版本并原子保存，返回路径。"""
    if destination.exists():
        load_api_token_counter(destination, DEFAULT_TOKENIZER_SHA256)
        return destination
    # 1. 【文档分块】【分词准备】仅访问固定分词 JSON，校验失败不得写出可用文件
    with urlopen(TOKENIZER_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != DEFAULT_TOKENIZER_SHA256:
        raise ValueError("【文档分块】【分词准备】下载文件 SHA-256 不匹配")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="tokenizer-", suffix=".part", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        load_api_token_counter(Path(temporary), DEFAULT_TOKENIZER_SHA256)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main() -> None:
    """无调用参数；读取 CLI 目标参数并输出实际准备位置，失败保留异常。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=BACKEND_ROOT / DEFAULT_TOKENIZER_PATH)
    arguments = parser.parse_args()
    print(f"BGE-M3 tokenizer ready: {prepare_tokenizer(arguments.destination.resolve())}")


if __name__ == "__main__":
    main()
