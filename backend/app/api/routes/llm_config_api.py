"""LLM 供应商配置中心路由。

提供预设的增删改查、一键激活（热切换上游）与连接测试。
这些接口面向"运行时切换 LLM 供应商"场景：用户在前端管理多套上游预设，
激活某套即写回 .env 并热重载，无需重启服务。所有接口都要求登录。
"""

import logging
import threading
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas.llm_profile import (
    LLMProfileCreate,
    LLMProfileUpdate,
    LLMProfileView,
    LLMTestRequest,
    LLMTestResult,
)
from app.middleware.auth import get_current_user_with_rate_limit
from app.services import llm_profile_service
from app.config import get_settings
from app.services.provider_request_queue import provider_request_context, run_provider_thread

router = APIRouter(prefix="/llm/profiles", tags=["llm-config"])
logger = logging.getLogger(__name__)


@router.get("", response_model=list[LLMProfileView])
async def list_llm_profiles(user: dict = Depends(get_current_user_with_rate_limit)):
    """列出全部 LLM 预设，Key 已脱敏，当前激活项带 is_active 标记。"""
    return llm_profile_service.list_profiles()


@router.post("", response_model=LLMProfileView)
async def create_llm_profile(
    payload: LLMProfileCreate,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """新增一套 LLM 供应商预设。"""
    return llm_profile_service.create_profile(payload)


@router.put("/{profile_id}", response_model=LLMProfileView)
async def update_llm_profile(
    profile_id: str,
    payload: LLMProfileUpdate,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """编辑预设；api_key 留空表示保留原值。找不到返回 404。"""
    try:
        return llm_profile_service.update_profile(profile_id, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{profile_id}")
async def delete_llm_profile(
    profile_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """删除预设；不能删当前激活项。找不到返回 404，删激活项返回 409。"""
    try:
        llm_profile_service.delete_profile(profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/{profile_id}/activate", response_model=LLMProfileView)
async def activate_llm_profile(
    profile_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """激活预设：写回 .env + 热重载，运行时立即切换到该上游。找不到返回 404。"""
    try:
        return llm_profile_service.activate_profile(profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/test", response_model=LLMTestResult)
async def test_llm_connection(
    payload: LLMTestRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """测试连接：显式传 base_url/model/key 或传 profile_id 用已存预设测试。"""
    base_url, model_name, api_key = _resolve_test_target(payload)
    if not base_url or not model_name:
        raise HTTPException(status_code=422, detail="缺少 base_url 或 model_name，无法测试")
    deadline = time.monotonic() + get_settings().llm_connection_test_timeout_seconds
    with provider_request_context(cancel_event=threading.Event(), deadline=deadline):
        return await run_provider_thread(llm_profile_service.test_connection,
            base_url=base_url, model_name=model_name, api_key=api_key)


def _resolve_test_target(payload: LLMTestRequest) -> tuple[str, str, str]:
    """解析测试目标：优先用显式字段，其次按 profile_id 取存储中的真实 Key。"""
    if payload.profile_id:
        return llm_profile_service.resolve_test_target(
            profile_id=payload.profile_id,
            base_url=payload.base_url,
            model_name=payload.model_name,
            api_key=payload.api_key,
        )
    return payload.base_url, payload.model_name, payload.api_key
