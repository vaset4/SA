"""正文格式单元测试的文件占用端口，作者：xxx。"""
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock


@asynccontextmanager
async def file_access(_paper_ids, *, exclusive=False):
    """参数为目标与占用方式；格式测试不验证数据库互斥，返回固定连接身份。"""
    yield "isolated-file-connection"


def install_file_ports(monkeypatch):
    """参数为测试替换器；隔离来源占用和 writer SQL，实际临时文件操作仍正常运行。"""
    from app.repositories import source_access_repository, upload_repository
    from app.services import upload_read_service

    gate = SimpleNamespace(file_access=file_access)
    monkeypatch.setattr(source_access_repository, "get_source_access_repository", lambda: gate)
    monkeypatch.setattr(upload_read_service, "get_source_access_repository", lambda: gate)
    monkeypatch.setattr(upload_repository, "require_upload_execution", AsyncMock())
