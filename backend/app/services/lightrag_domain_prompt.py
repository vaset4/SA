"""Domain prompt management and entity lint helpers for LightRAG."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.graph.academic_schema import (
    ACADEMIC_ENTITY_EXTRACTION_EXAMPLES,
    ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ACADEMIC_ENTITY_TYPES,
)

logger = logging.getLogger(__name__)
settings = get_settings()
PROMPT_SYSTEM_KEY = "entity_extraction_system_prompt"
PROMPT_EXAMPLES_KEY = "entity_extraction_examples"
ORIGINAL_DOMAIN_PROMPTS: dict[str, Any] = {}
DOMAIN_PROMPTS_INSTALLED = False


def reset_domain_prompt_flag() -> None:
    global DOMAIN_PROMPTS_INSTALLED
    DOMAIN_PROMPTS_INSTALLED = False


def install_domain_prompts(light_rag_prompts: dict[str, Any] | None) -> bool:
    """Install or restore process-global LightRAG academic extraction prompts."""
    global DOMAIN_PROMPTS_INSTALLED
    if light_rag_prompts is None:
        return False
    remember_original_domain_prompts(light_rag_prompts)
    if not settings.lightrag_use_domain_prompt:
        restore_domain_prompts(light_rag_prompts)
        return False
    if DOMAIN_PROMPTS_INSTALLED:
        return True
    light_rag_prompts[PROMPT_SYSTEM_KEY] = ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT
    light_rag_prompts[PROMPT_EXAMPLES_KEY] = list(ACADEMIC_ENTITY_EXTRACTION_EXAMPLES)
    DOMAIN_PROMPTS_INSTALLED = True
    logger.info(
        "LightRAG domain prompt installed: entity_types=%s language=Chinese",
        list(ACADEMIC_ENTITY_TYPES),
    )
    return True


def remember_original_domain_prompts(light_rag_prompts: dict[str, Any]) -> None:
    if ORIGINAL_DOMAIN_PROMPTS:
        return
    ORIGINAL_DOMAIN_PROMPTS[PROMPT_SYSTEM_KEY] = light_rag_prompts.get(PROMPT_SYSTEM_KEY)
    ORIGINAL_DOMAIN_PROMPTS[PROMPT_EXAMPLES_KEY] = light_rag_prompts.get(PROMPT_EXAMPLES_KEY)


def restore_domain_prompts(light_rag_prompts: dict[str, Any]) -> None:
    global DOMAIN_PROMPTS_INSTALLED
    remember_original_domain_prompts(light_rag_prompts)
    for key, value in ORIGINAL_DOMAIN_PROMPTS.items():
        light_rag_prompts[key] = value
    DOMAIN_PROMPTS_INSTALLED = False


def build_addon_params(domain_prompt_enabled: bool) -> dict[str, Any]:
    params: dict[str, Any] = {"language": "Chinese"}
    if domain_prompt_enabled:
        params["entity_types"] = list(ACADEMIC_ENTITY_TYPES)
    return params


def lint_lightrag_entities_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"deleted": 0, "kept": 0, "by_type": {}, "path": str(path)}
    payload = read_json_dict(path)
    kept_payload, stats = filter_academic_entities(payload)
    if stats["deleted"]:
        path.write_text(json.dumps(kept_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**stats, "path": str(path)}


def read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("读取 LightRAG entity 文件失败: %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def filter_academic_entities(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = {item.lower() for item in ACADEMIC_ENTITY_TYPES}
    kept_payload: dict[str, Any] = {}
    by_type: dict[str, int] = {}
    deleted = 0
    for key, value in payload.items():
        entity_type = entity_type_from_payload(value)
        if entity_type.lower() not in allowed:
            deleted += 1
            continue
        kept_payload[key] = value
        by_type[entity_type] = by_type.get(entity_type, 0) + 1
    return kept_payload, {"deleted": deleted, "kept": len(kept_payload), "by_type": by_type}


def entity_type_from_payload(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("entity_type") or value.get("type") or "").strip()
    return ""
