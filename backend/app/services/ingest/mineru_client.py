"""MinerU 官方在线解析客户端。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from app.config import get_settings

HTTP_REQUEST_TIMEOUT_SECONDS = 60
SUCCESS_CODES = {"0", "200"}
AUTHENTICATION_CODES = {"A0202", "A0211"}
QUOTA_CODES = {"-60018"}
TEMPORARY_CODES = {"-60007", "-60009"}
ACTIVE_REMOTE_STATES = {"waiting-file", "pending", "running", "converting", "done"}


class MinerUError(RuntimeError):
    """MinerU 在线解析通用错误。"""

    def __init__(self, message: str, *, code: str = "", retry_after: float | None = None) -> None:
        """保存可追溯的供应商错误码及等待提示，不包含认证请求头。"""
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class MinerUConfigurationError(MinerUError):
    """MinerU 配置缺失或非法。"""


class MinerUAuthenticationError(MinerUError):
    """MinerU 鉴权失败。"""


class MinerURateLimitError(MinerUError):
    """MinerU 短暂限流，可由当前安全步骤有限重试。"""


class MinerUQuotaError(MinerURateLimitError):
    """日配额耗尽，必须暂停，不能当作短暂限流持续请求。"""


class MinerUTemporaryError(MinerUError):
    """供应商服务或队列暂不可用。"""


class MinerUTransferError(MinerUError):
    """文件传输签名地址失效，与解析 API 的账号凭证错误分开。"""


class MinerUTimeoutError(MinerUError):
    """MinerU 任务轮询超时。"""


class MinerUResultError(MinerUError):
    """MinerU 返回了不可用结果。"""


@dataclass(frozen=True, slots=True)
class MinerUBatchFile:
    """MinerU 批处理文件槽位。"""

    batch_id: str
    file_name: str
    file_url: str
    upload_url: str
    data_id: str = ""


@dataclass(frozen=True, slots=True)
class MinerUExtractResult:
    """MinerU 单文件解析结果。"""

    batch_id: str
    file_name: str
    state: str
    full_zip_url: str
    raw_payload: dict[str, Any]


def _normalize_base_url(raw_url: str) -> str:
    return raw_url.rstrip("/")


def _retry_after_seconds(value: str) -> float | None:
    """把数字或 HTTP 日期等待提示转换为秒；非法提示返回空值。"""
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class MinerUClient:
    """面向官方 HTTP API 的最小同步客户端。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_token: str | None = None,
        poll_interval_seconds: float | None = None,
        timeout_seconds: int | None = None,
        result_download_timeout_seconds: int | None = None,
        session: requests.Session | None = None,
        sleeper: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        settings = get_settings()
        self.base_url = _normalize_base_url(base_url or settings.mineru_api_base_url)
        self.api_token = str(api_token if api_token is not None else settings.mineru_api_token).strip()
        self.poll_interval_seconds = float(
            poll_interval_seconds if poll_interval_seconds is not None else settings.mineru_poll_interval_seconds
        )
        self.timeout_seconds = int(timeout_seconds if timeout_seconds is not None else settings.mineru_timeout_seconds)
        self.result_download_timeout_seconds = int(
            result_download_timeout_seconds
            if result_download_timeout_seconds is not None
            else settings.mineru_result_download_timeout_seconds
        )
        self._session = session or requests.Session()
        self._sleep = sleeper
        self._monotonic = monotonic

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_token:
            raise MinerUConfigurationError("MinerU 已启用，但未配置 mineru_api_token")
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, response: requests.Response, body: dict[str, Any] | None = None, *, transfer: bool = False) -> None:
        """【解析】【供应商响应】核对 HTTP 状态，区分凭证、额度和可重试服务故障。"""
        if response.status_code in (401, 403):
            if transfer:
                raise MinerUTransferError("MinerU 文件传输地址已过期或被拒绝", code="SIGNED_URL_REJECTED")
            raise MinerUAuthenticationError("MinerU token 无效或已过期")
        if isinstance(body, dict) and str(body.get("code")) in QUOTA_CODES:
            raise MinerUQuotaError(str(body.get("msg") or "MinerU 当日解析额度不足"), code=str(body["code"]))
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After", ""))
            raise MinerURateLimitError("MinerU 请求触发限流", retry_after=retry_after)
        if response.status_code >= 500:
            raise MinerUTemporaryError(f"MinerU 服务暂不可用: HTTP {response.status_code}")
        if not response.ok:
            detail = ""
            if isinstance(body, dict):
                detail = str(body.get("msg") or body.get("message") or body.get("detail") or "")
            raise MinerUError(f"MinerU 请求失败: status={response.status_code} detail={detail or response.text[:200]}")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = self._session.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            headers=self._auth_headers(),
            timeout=timeout or min(self.timeout_seconds, HTTP_REQUEST_TIMEOUT_SECONDS),
            **kwargs,
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        self._raise_for_status(response, body)
        if isinstance(body, dict):
            code = body.get("code")
            if code is not None and str(code) not in SUCCESS_CODES:
                self._raise_business_error(str(code), body)
            return body
        raise MinerUResultError("MinerU 返回了非 JSON 响应")

    def _raise_business_error(self, code: str, body: dict[str, Any]) -> None:
        """【解析】【业务错误】HTTP 200 中的业务失败仍明确暴露对应原因。"""
        message = str(body.get("err_msg") or body.get("msg") or body.get("message") or "unknown error")
        error_type: type[MinerUError] = MinerUResultError
        if code in AUTHENTICATION_CODES:
            error_type = MinerUAuthenticationError
        elif code in QUOTA_CODES:
            error_type = MinerUQuotaError
        elif code in TEMPORARY_CODES:
            error_type = MinerUTemporaryError
        raise error_type(f"MinerU 返回错误 code={code}: {message}", code=code)

    def request_upload_slot(self, file_name: str, *, data_id: str = "") -> MinerUBatchFile:
        """申请本地上传槽位；data_id 用于核对回传文件身份，不视为供应商幂等保证。"""
        body = self._request_json(
            "POST",
            "file-urls/batch",
            json={
                "enable_formula": True,
                "enable_table": True,
                "files": [{"name": file_name, **({"data_id": data_id} if data_id else {})}],
            },
        )
        data = body.get("data") or {}
        batch_id = str(data.get("batch_id") or "").strip()
        file_entries = data.get("file_urls") or []
        if not batch_id or not isinstance(file_entries, list) or len(file_entries) != 1:
            raise MinerUResultError("MinerU 未返回可用的 batch_id 或 file_urls")
        entry = file_entries[0]
        if isinstance(entry, str):
            upload_url = entry.strip()
            file_url = upload_url
            resolved_name = file_name
        elif isinstance(entry, dict):
            upload_url = str(entry.get("upload_url") or entry.get("url") or entry.get("file_url") or "").strip()
            file_url = str(entry.get("file_url") or entry.get("upload_url") or entry.get("url") or "").strip()
            resolved_name = str(entry.get("name") or file_name)
        else:
            raise MinerUResultError("MinerU 上传槽位结构异常")
        if not upload_url:
            raise MinerUResultError("MinerU 上传槽位缺少 upload_url")
        return MinerUBatchFile(
            batch_id=batch_id,
            file_name=resolved_name,
            file_url=file_url,
            upload_url=upload_url,
            data_id=data_id,
        )

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        """把本地 PDF PUT 到 MinerU 预签名地址。"""
        self.upload_bytes(upload_url, file_path.read_bytes())

    def upload_bytes(self, upload_url: str, content: bytes) -> None:
        """【解析】【上传原文】上传调用方已经核对过指纹的同一份原始字节。"""
        response = self._session.put(
            upload_url,
            data=content,
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response, transfer=True)

    def get_result(
        self, batch_id: str, *, file_name: str | None = None, data_id: str = "",
    ) -> MinerUExtractResult | None:
        """【解析】【查询原任务】查询一次指定批次，仅返回匹配文件的真实状态。"""
        body = self._request_json("GET", f"extract-results/batch/{batch_id}")
        entries = body.get("data") or []
        if isinstance(entries, dict):
            entries = entries.get("extract_result") or entries.get("results") or []
        if not isinstance(entries, list):
            raise MinerUResultError("MinerU 轮询结果结构异常")
        target = self._select_result(entries, file_name=file_name, data_id=data_id)
        if target is None:
            return None
        state = str(target.get("state") or "").strip().lower()
        if state in {"failed", "error", "canceled", "cancelled"}:
            message = str(target.get("err_msg") or target.get("msg") or target.get("error") or "unknown error")
            raise MinerUResultError(f"MinerU 解析失败: {message}", code="REMOTE_PARSE_FAILED")
        if state not in ACTIVE_REMOTE_STATES:
            raise MinerUResultError(f"MinerU 返回无法识别的文件状态: {state}")
        full_zip_url = str(target.get("full_zip_url") or "").strip()
        nested = target.get("extract_result") or []
        if not full_zip_url and isinstance(nested, list) and nested:
            if not isinstance(nested[0], dict):
                raise MinerUResultError("MinerU 结果地址结构异常")
            full_zip_url = str(nested[0].get("full_zip_url") or "").strip()
        if state == "done" and not full_zip_url:
            raise MinerUResultError("MinerU 已完成但缺少结果 ZIP 地址")
        return MinerUExtractResult(batch_id, str(target.get("file_name") or file_name or ""), state, full_zip_url, target)

    def wait_for_result(self, batch_id: str, *, file_name: str | None = None, data_id: str = "") -> MinerUExtractResult:
        """轮询指定原任务；只有显式 done 且结果地址有效时才完成。"""
        deadline = self._monotonic() + self.timeout_seconds
        last_state = ""
        while self._monotonic() < deadline:
            result = self.get_result(batch_id, file_name=file_name, data_id=data_id)
            if result is not None:
                last_state = result.state
                if result.state == "done":
                    return result
            self._sleep(self.poll_interval_seconds)
        raise MinerUTimeoutError(
            f"MinerU 解析超时: batch_id={batch_id} state={last_state or 'unknown'} timeout={self.timeout_seconds}s"
        )

    def download_result_zip(self, full_zip_url: str) -> bytes:
        """下载解析结果 ZIP。"""
        response = self._session.get(full_zip_url, timeout=self.result_download_timeout_seconds)
        self._raise_for_status(response, transfer=True)
        if not response.content:
            raise MinerUResultError("MinerU 返回了空结果包")
        return response.content

    def parse_local_file(self, file_path: Path) -> tuple[MinerUBatchFile, MinerUExtractResult, bytes]:
        """执行“申请上传地址 -> 上传文件 -> 轮询 -> 下载结果”的完整流程。"""
        batch_file = self.request_upload_slot(file_path.name)
        self.upload_file(batch_file.upload_url, file_path)
        extract_result = self.wait_for_result(batch_file.batch_id, file_name=batch_file.file_name)
        archive_bytes = self.download_result_zip(extract_result.full_zip_url)
        return batch_file, extract_result, archive_bytes

    @staticmethod
    def _select_result(
        entries: list[dict[str, Any]], *, file_name: str | None = None, data_id: str = "",
    ) -> dict[str, Any] | None:
        """核对 data_id 或明确文件名；不把列表首项当成未匹配文件的结果。"""
        if not entries:
            return None
        if any(not isinstance(item, dict) for item in entries):
            raise MinerUResultError("MinerU 文件结果必须是对象")
        if not file_name and not data_id and len(entries) == 1:
            return entries[0]
        for item in entries:
            if data_id and str(item.get("data_id") or "") == data_id:
                return item
            current_name = str(item.get("file_name") or item.get("name") or "").strip()
            if not data_id and current_name == file_name:
                return item
        raise MinerUResultError("MinerU 批次中的文件身份与本次上传不一致")
