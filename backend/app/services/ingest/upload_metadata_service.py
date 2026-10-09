"""独立论文元信息补全；失败由上传任务持久记录，作者：xxx。"""
from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.services.ingest import document_parse_service

METADATA_FIELDS = ("title", "authors", "published_date", "abstract")
METADATA_EXCERPT_CHARACTERS = 12000
METADATA_REQUEST_TIMEOUT_SECONDS = 60


def missing_metadata(metadata: dict[str, Any]) -> list[str]:
    """返回原文尚未识别的字段，用于页面提示及独立补全任务。"""
    return [field for field in METADATA_FIELDS if not metadata.get(field)]


def metadata_client() -> Any:
    """使用现役模型配置建立单次调用客户端，重试统一由持久阶段控制。"""
    from app.services.llm_service import OpenAICompatibleLLM

    settings = get_settings()
    if not settings.llm_api_key or not settings.llm_base_url:
        from app.models.upload_models import UploadFailure, UploadPausedError
        raise UploadPausedError(UploadFailure("metadata", "credentials", "元信息模型尚未配置凭证或服务地址"))
    return OpenAICompatibleLLM(model=settings.llm_model_name, api_key=settings.llm_api_key,
        base_url=settings.llm_base_url, temperature=settings.llm_temperature, timeout=min(settings.llm_timeout, METADATA_REQUEST_TIMEOUT_SECONDS),
        min_request_interval_seconds=settings.llm_min_request_interval_seconds,
        requests_per_minute_limit=settings.llm_requests_per_minute_limit,
        request_exception_max_retries=0, empty_response_max_retries=0)


def complete_metadata(full_text: str, metadata: dict[str, Any], *, client: Any) -> dict[str, Any]:
    """仅补原文有依据的缺项，返回合并后的元信息；调用和结构错误直接抛出。"""
    from app.services.llm_service import build_structured_output_prompt, parse_structured_response

    if not missing_metadata(metadata):
        return dict(metadata)
    schema = document_parse_service.UploadMetadataLLMResult
    prompt = build_structured_output_prompt(
        "Extract academic paper metadata from the supplied document. Fill only missing fields. "
        "Use the title and abstract printed in the document; do not invent an abstract or infer authors. "
        "Use YYYY-MM-DD for an explicit date, YYYY-01-01 for an explicit year. "
        "Return empty strings/lists when the document does not provide the field.\n"
        f"Known metadata: {metadata}\nDocument:\n{full_text[:METADATA_EXCERPT_CHARACTERS]}", schema)
    response = client.invoke(prompt, use_cache=False)
    extracted = parse_structured_response(response.content, schema)
    additions = {"title": document_parse_service._clean_text(extracted.title, 400),
        "authors": document_parse_service._split_authors("; ".join(extracted.authors)),
        "published_date": document_parse_service._parse_pdf_date(extracted.published_date),
        "abstract": document_parse_service._clean_text(extracted.abstract, 3000)}
    return {field: metadata.get(field) or additions[field] for field in METADATA_FIELDS}
