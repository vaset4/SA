"""MinerU 真实响应边界及错误分类回归，作者：xxx。"""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
import requests

from app.services.ingest.mineru_client import (
    MinerUAuthenticationError, MinerUClient, MinerUQuotaError,
    MinerUResultError, MinerUTemporaryError,
)


def _response(payload: dict, status: int = 200) -> requests.Response:
    """构造标准 requests 响应，测试真实客户端的解码和状态处理。"""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode("utf-8")
    return response


def _client(*responses: requests.Response) -> MinerUClient:
    """用可控 HTTP 端口驱动真实客户端；返回按顺序消费响应的实例。"""
    session = Mock()
    session.request.side_effect = responses
    return MinerUClient(api_token="unit-test-token", session=session, poll_interval_seconds=0, timeout_seconds=1)


@pytest.mark.parametrize("code,error_type", [
    (-60009, MinerUTemporaryError), (-60007, MinerUTemporaryError),
    (-60018, MinerUQuotaError), ("A0202", MinerUAuthenticationError), ("A0211", MinerUAuthenticationError),
])
def test_http_200_business_error_is_not_success(code, error_type) -> None:
    """HTTP 200 中的额度、凭证和临时故障仍分别失败。"""
    client = _client(_response({"code": code, "msg": "provider error"}))
    with pytest.raises(error_type) as error:
        client.request_upload_slot("paper.pdf")
    assert error.value.code == str(code)


def test_pending_result_url_does_not_mean_completed() -> None:
    """服务端未声明 done 时，结果 URL 的存在不能作为完成证据。"""
    client = _client(
        _response({"data": [{"file_name": "paper.pdf", "state": "running", "full_zip_url": "https://example.invalid/partial"}]}),
        _response({"data": [{"file_name": "paper.pdf", "state": "done", "full_zip_url": "https://example.invalid/result"}]}),
    )
    result = client.wait_for_result("batch", file_name="paper.pdf")
    assert result.state == "done"
    assert result.full_zip_url.endswith("/result")
    assert client._session.request.call_count == 2


def test_wrong_file_never_falls_back_to_first_result() -> None:
    """批次含另一文件时明确报错，不把首项冒充当前上传。"""
    client = _client(_response({"data": [{"file_name": "another.pdf", "state": "done", "full_zip_url": "https://example.invalid/other"}]}))
    with pytest.raises(MinerUResultError, match="身份"):
        client.get_result("batch", file_name="paper.pdf")


def test_data_id_is_checked_even_when_file_name_matches() -> None:
    """文件名相同但回传 data_id 不同仍拒绝使用。"""
    client = _client(_response({"data": [{"file_name": "paper.pdf", "data_id": "another", "state": "done", "full_zip_url": "https://example.invalid/other"}]}))
    with pytest.raises(MinerUResultError, match="身份"):
        client.get_result("batch", file_name="paper.pdf", data_id="expected")


def test_failed_result_keeps_official_error_message() -> None:
    """failed 使用官方 err_msg 原因，不能只返回 unknown error。"""
    client = _client(_response({"data": [{"file_name": "paper.pdf", "state": "failed", "err_msg": "page conversion failed"}]}))
    with pytest.raises(MinerUResultError, match="page conversion failed") as error:
        client.get_result("batch", file_name="paper.pdf")
    assert error.value.code == "REMOTE_PARSE_FAILED"


def test_done_without_result_address_is_invalid() -> None:
    """done 缺少产物地址时不能被当作完整可用结果。"""
    client = _client(_response({"data": [{"file_name": "paper.pdf", "state": "done"}]}))
    with pytest.raises(MinerUResultError, match="缺少"):
        client.get_result("batch", file_name="paper.pdf")


def test_download_signature_error_is_distinct_from_api_token_error() -> None:
    """下载地址失效不能误报账号 token 错误。"""
    from app.services.ingest.mineru_client import MinerUTransferError

    client = _client()
    client._session.get.return_value = _response({}, status=403)
    with pytest.raises(MinerUTransferError, match="传输地址"):
        client.download_result_zip("https://example.invalid/result")


def test_retry_after_http_date_preserves_long_wait_hint() -> None:
    """HTTP 日期格式的等待提示仍用于判断短暂重试是否应转暂停。"""
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from app.services.ingest.mineru_client import MinerURateLimitError

    response = _response({}, status=429)
    response.headers["Retry-After"] = format_datetime(datetime.now(timezone.utc) + timedelta(minutes=2))
    with pytest.raises(MinerURateLimitError) as error:
        _client(response).request_upload_slot("source.pdf")
    assert 110 <= error.value.retry_after <= 120
