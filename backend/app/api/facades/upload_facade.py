"""上传请求体读取边界。"""

from __future__ import annotations

from fastapi import UploadFile

from app.config import get_settings

BYTES_PER_MIB = 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = BYTES_PER_MIB


class UploadTooLargeError(ValueError):
    """上传体超过配置大小限制。"""


def resolve_upload_limit_bytes(settings=None) -> tuple[int, int]:
    """返回上传大小限制的字节值和 MB 值。"""
    active_settings = settings or get_settings()
    max_size_mb = int(getattr(active_settings, "max_upload_size_mb", 50))
    return max_size_mb * BYTES_PER_MIB, max_size_mb


async def read_upload_bytes(file: UploadFile, settings=None) -> bytes:
    """按 chunk 读取上传内容，超过配置大小时立即停止。"""
    max_size, max_size_mb = resolve_upload_limit_bytes(settings)
    chunks: list[bytes] = []
    total_size = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            size_mb = total_size / BYTES_PER_MIB
            raise UploadTooLargeError(f"文件大小 {size_mb:.1f}MB 超过限制 {max_size_mb}MB")
        chunks.append(chunk)
    return b"".join(chunks)
