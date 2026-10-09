"""账号身份到持久上传任务的边界回归，作者：xxx。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import Depends, FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from app.services import auth_service, jwt_service, task_service, upload_service
from app.middleware.auth import get_current_user
from app.tasks import upload_task


def _task_row(task_type, payload, message, *, task_id, user_id) -> dict:
    """接收真实创建服务的数据库参数，返回同一归属的任务记录。"""
    return {
        "task_id": task_id, "task_type": task_type, "payload": payload,
        "message": message, "user_id": user_id, "status": "queued",
    }


@pytest.fixture
def identity_io(monkeypatch, tmp_path) -> dict:
    """隔离数据库和入库适配器；返回配置、调用记录及真实临时目录。"""
    settings = SimpleNamespace(
        app_env="local", auth_enabled=True, api_key="unit-test-api-key",
        jwt_secret_key="unit-test-identity-secret", jwt_expire_hours=24,
        host="127.0.0.1", developer_usernames="", admin_usernames="",
    )
    create = Mock(side_effect=_task_row)
    ingest = AsyncMock(side_effect=lambda task: {"paper_id": task["payload"]["paper_id"]})
    progress = AsyncMock(return_value={})
    monkeypatch.setattr(auth_service, "_settings", lambda: settings)
    monkeypatch.setattr(task_service, "get_settings", lambda: settings)
    monkeypatch.setattr(task_service.paper_db, "create_task_run", create)
    monkeypatch.setattr(task_service, "append_event", AsyncMock(return_value={}))
    monkeypatch.setattr(task_service, "report_progress", progress)
    async def receive(filename, content, **options):
        """记录实际控制层传递的 owner，受理事务的原子性由真实 PG 测试验证。"""
        return create("upload_ingest", {"filename": filename, "paper_id": "upload-test"}, "queued",
            task_id="task-test", user_id=options["user_id"])
    monkeypatch.setattr(upload_service, "receive_upload", receive)
    monkeypatch.setattr(upload_task, "prepare_upload_task", ingest)
    return {"settings": settings, "create": create, "ingest": ingest, "progress": progress, "directory": tmp_path}


@pytest.mark.parametrize("auth_enabled", [False, True])
def test_signed_upload_owner_survives_task_and_worker(identity_io, auth_enabled) -> None:
    """真实 JWT 经身份、暂存、任务创建和 worker 后保持同一用户与 paper_id。"""
    settings = identity_io["settings"]
    settings.auth_enabled = auth_enabled
    token = jwt_service.create_access_token("owner-a", settings=settings)
    user = auth_service.resolve_authenticated_user(token, None, settings=settings)
    task = asyncio.run(task_service.create_upload_task(
        "paper.pdf", b"%PDF-1.4\nbody", user_id=auth_service.get_user_id(user),
    ))
    assert task["user_id"] == "owner-a"
    assert "user_id" not in task["payload"]
    result = asyncio.run(upload_task.run_upload_ingest_task(task))
    assert result["paper_id"] == task["payload"]["paper_id"]
    assert identity_io["ingest"].await_args.args[0]["user_id"] == "owner-a"
    assert identity_io["create"].call_args.kwargs["user_id"] == "owner-a"


@pytest.mark.parametrize("bearer,api_key", [("bad-token", None), ("", None), (None, "bad-key"), (None, "")])
def test_invalid_explicit_credentials_never_become_developer(identity_io, bearer, api_key) -> None:
    """免登录开发模式也拒绝显式错误凭证，不能转为 anonymous。"""
    identity_io["settings"].auth_enabled = False
    with pytest.raises(HTTPException) as error:
        auth_service.resolve_authenticated_user(bearer, api_key)
    assert error.value.status_code == 401


def test_credentialless_development_and_explicit_api_key_are_distinct(identity_io) -> None:
    """开发免登录保留原身份，显式 API Key 则使用自己的合成身份。"""
    identity_io["settings"].auth_enabled = False
    assert auth_service.resolve_authenticated_user(None, None)["user_id"] == "anonymous"
    assert auth_service.resolve_authenticated_user(None, "unit-test-api-key")["user_id"] == "api_key_user"
    identity_io["settings"].auth_enabled = True
    with pytest.raises(HTTPException) as error:
        auth_service.resolve_authenticated_user(None, None)
    assert error.value.status_code == 401


@pytest.mark.parametrize("headers", [{"Authorization": "Basic invalid"}, {"Authorization": "Bearer "}, {"X-API-Key": ""}])
@pytest.mark.asyncio
async def test_http_adapter_does_not_erase_invalid_explicit_credentials(identity_io, headers) -> None:
    """HTTP 安全适配器丢弃格式错误凭证时，服务仍必须拒绝它而非启用开发身份。"""
    identity_io["settings"].auth_enabled = False
    application = FastAPI()

    @application.get("/owner")
    async def owner(user: dict = Depends(get_current_user)) -> dict:
        """返回真实认证依赖解析的用户。"""
        return user

    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.get("/owner", headers=headers)
        assert response.status_code == 401
        anonymous = await client.get("/owner")
        assert anonymous.status_code == 200
        assert anonymous.json()["user_id"] == "anonymous"


@pytest.mark.parametrize("identity", [
    None, {}, {"user_id": " "}, {"user_id": "owner-a", "sub": "owner-b"},
    {"user_id": 123}, {"sub": []}, {"user_id": "anonymous"},
])
def test_missing_or_conflicting_request_identity_is_rejected(identity_io, identity) -> None:
    """缺失、矛盾或账号模式 anonymous 身份不能用于业务归属。"""
    with pytest.raises(HTTPException) as error:
        auth_service.get_user_id(identity)
    assert error.value.status_code == 401


@pytest.mark.parametrize("owner", [None, "", " ", 123, "anonymous"])
def test_invalid_upload_owner_is_rejected_before_staging(identity_io, owner) -> None:
    """无效 owner 必须在文件和任务产生副作用之前失败。"""
    with pytest.raises(ValueError):
        asyncio.run(task_service.create_upload_task("paper.pdf", b"%PDF-1.4\nbody", user_id=owner))
    identity_io["create"].assert_not_called()
    assert list(identity_io["directory"].iterdir()) == []


@pytest.mark.parametrize("owner,payload_owner", [
    (None, "owner-a"), ("", "owner-a"), ("owner-a", "owner-b"),
    ("anonymous", "anonymous"), ("owner-a", ""),
])
def test_worker_rejects_owner_conflicts_before_reading(identity_io, owner, payload_owner) -> None:
    """持久任务身份缺失或冲突时不读取原文件、不调用入库且不删除 staging。"""
    staged = identity_io["directory"] / "paper.pdf"
    staged.write_bytes(b"%PDF-1.4\nbody")
    task = {"task_id": "task-a", "user_id": owner, "payload": {
        "user_id": payload_owner, "paper_id": "paper-a", "staging_file_path": str(staged),
    }}
    with pytest.raises(ValueError):
        asyncio.run(upload_task.run_upload_ingest_task(task))
    identity_io["progress"].assert_not_awaited()
    identity_io["ingest"].assert_not_awaited()
    assert staged.read_bytes() == b"%PDF-1.4\nbody"


@pytest.mark.parametrize("include_payload_owner", [False, True])
def test_worker_accepts_record_owner_and_matching_legacy_payload(identity_io, include_payload_owner) -> None:
    """新 payload 不重复保存身份；旧 payload 同 owner 时仍复用原论文。"""
    staged = identity_io["directory"] / "paper.pdf"
    staged.write_bytes(b"%PDF-1.4\nbody")
    payload = {"paper_id": "paper-a", "filename": "paper.pdf"}
    if include_payload_owner:
        payload["user_id"] = "owner-a"
    result = asyncio.run(upload_task.run_upload_ingest_task({
        "task_id": "task-a", "user_id": "owner-a", "payload": payload,
    }))
    assert result["paper_id"] == "paper-a"
    assert identity_io["ingest"].await_args.args[0]["user_id"] == "owner-a"
    assert Path(staged).exists()
