"""Agent 场景文件加载与版本校验。作者：xxx。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import AgentCase, load_agent_case


def load_agent_cases(path: Path) -> tuple[AgentCase, ...]:
    """参数为 JSON/JSONL 场景文件路径；返回去重且已冻结的案例。"""
    records = _read_records(path)
    cases = tuple(load_agent_case(record) for record in records)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("agent case IDs must be unique")
    return cases


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    """参数为场景文件路径；读取 JSON 数组或 JSONL 对象，拒绝混合格式。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("agent dataset must not be empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _read_jsonl(text)
    if not isinstance(payload, list) or any(not isinstance(item, Mapping) for item in payload):
        raise ValueError("agent dataset JSON must be an array of objects")
    return [dict(item) for item in payload]


def _read_jsonl(text: str) -> list[Mapping[str, Any]]:
    """参数为 JSONL 文本；按行解析对象并保留行号错误。"""
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid agent JSONL at line {line_number}") from exc
        if not isinstance(item, Mapping):
            raise ValueError(f"agent JSONL line {line_number} must contain an object")
        records.append(dict(item))
    if not records:
        raise ValueError("agent dataset must contain at least one record")
    return records
