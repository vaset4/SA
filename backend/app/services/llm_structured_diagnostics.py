"""【模型服务】【错误定位】解释合同中的校验路径，不校验或修改模型输出。作者：xxx。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

PathPart = str | int
SCALAR_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "null"})


@dataclass(frozen=True)
class _Location:
    """保存已证实的 JSON 路径、遇到的联合标签及无法确定的原因。"""

    path: tuple[PathPart, ...] | None = None
    branches: tuple[PathPart, ...] = ()
    reason: str = "当前合同无法唯一确定原 JSON 字段位置"

    def prepend(self, part: PathPart) -> _Location:
        """参数为确证的对象键或数组下标；返回增加该层的新位置，不修改已有诊断。"""
        return replace(self, path=(part, *self.path)) if self.path is not None else self


def _resolve_reference(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    """参数为当前位置合同和根合同；返回局部引用目标，未知或循环引用明确无法定位。"""
    seen: set[str] = set()
    while "$ref" in node:
        reference = node["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/$defs/") or reference in seen:
            return None
        seen.add(reference)
        name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
        target = root.get("$defs", {}).get(name)
        if not isinstance(target, dict):
            return None
        node = target
    return node


def _trace_union(
    node: dict[str, Any], value: Any, location: tuple[PathPart, ...],
    root: dict[str, Any], *, error_type: str,
) -> _Location:
    """参数为判别联合、输入、剩余 loc 和根合同；仅剥离与合同及输入同时吻合的标签。"""
    discriminator = node["discriminator"]
    if not isinstance(discriminator, dict) or not isinstance(value, dict):
        return _Location()
    property_name = discriminator.get("propertyName")
    mapping = discriminator.get("mapping")
    if not isinstance(property_name, str) or not isinstance(mapping, dict):
        return _Location()
    tag = value.get(property_name)
    if type(tag) not in (str, int) or tag != location[0]:
        return _Location(reason="输入判别值与校验分支无法对应，保留原校验位置")
    selected = mapping.get(str(tag))
    branch = {"$ref": selected} if isinstance(selected, str) else selected
    if not isinstance(branch, dict) or branch not in node.get("oneOf", []):
        return _Location()
    result = _trace_location(branch, value, location[1:], root, error_type=error_type)
    return replace(result, branches=(tag, *result.branches))


def _trace_object(
    node: dict[str, Any], value: Any, location: tuple[PathPart, ...],
    root: dict[str, Any], *, error_type: str,
) -> _Location:
    """参数为对象合同、输入及错误位置；返回真实键或缺失字段的位置，不跳过不存在的键。"""
    part, *remaining = location
    if not isinstance(part, str) or not isinstance(value, dict):
        return _Location()
    properties = node.get("properties", {})
    child = properties.get(part)
    if child is None:
        if error_type == "extra_forbidden" and not remaining and part in value and node.get("additionalProperties") is False:
            return _Location(path=(part,), reason="原 JSON 含合同禁止的额外字段")
        child = node.get("additionalProperties")
    if not isinstance(child, dict):
        return _Location()
    if part not in value:
        if error_type == "missing" and not remaining and part in properties:
            return _Location(path=(part,), reason="合同要求的字段缺失；此位置表示应填写的字段")
        return _Location()
    return _trace_location(child, value[part], tuple(remaining), root, error_type=error_type).prepend(part)


def _trace_location(
    node: dict[str, Any], value: Any, location: tuple[PathPart, ...],
    root: dict[str, Any], *, error_type: str,
) -> _Location:
    """参数为同一 validation 合同、输入、loc 与错误类型；返回确证位置，不重跑校验器。"""
    resolved = _resolve_reference(node, root)
    if resolved is None:
        return _Location()
    # 1. 【模型服务】【错误定位】整体对象或数组错误不指定某个字段，保留原始 loc
    if not location:
        if resolved.get("type") in SCALAR_JSON_TYPES:
            return _Location(path=(), reason="已由合同确认原 JSON 字段位置")
        return _Location(reason="错误作用于整个对象、数组或无法细分的合同；没有确定的字段位置")
    # 2. 【模型服务】【错误定位】只有 discriminator 显式证明的标签才从 JSON 键路径剥离
    if "discriminator" in resolved:
        return _trace_union(resolved, value, location, root, error_type=error_type)
    if any(key in resolved for key in ("anyOf", "oneOf", "allOf")):
        return _Location(reason="联合或组合合同未提供可唯一对应的判别路径，保留原校验位置")
    if resolved.get("type") == "object":
        return _trace_object(resolved, value, location, root, error_type=error_type)
    part, *remaining = location
    items = resolved.get("items")
    if resolved.get("type") == "array" and isinstance(value, list) and type(part) is int:
        if 0 <= part < len(value) and isinstance(items, dict):
            return _trace_location(items, value[part], tuple(remaining), root, error_type=error_type).prepend(part)
    return _Location()


def _json_pointer(path: tuple[PathPart, ...]) -> str:
    """参数为已证实的键与下标；返回按 RFC 6901 转义的 JSON Pointer。"""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def describe_validation_locations(
    raw_content: str, schema: dict[str, Any], errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """参数为原始响应、Pydantic validation JSON Schema 和原错误；返回独立定位说明，输入均不修改。"""
    # 1. 【模型服务】【错误定位】JSON 未解析成功时只展示原校验位置，不从坏文本猜字段
    try:
        value = json.loads(raw_content)
    except ValueError:
        value = None
        parsed = False
    else:
        parsed = True
    root = _resolve_reference(schema, schema)
    root_properties = root.get("properties", {}) if root is not None else {}
    descriptions = []
    for error in errors:
        location = tuple(error["loc"])
        result = _trace_location(schema, value, location, schema, error_type=error["type"]) if parsed else _Location(
            reason="原响应不是合法 JSON，无法确定原 JSON 字段位置",
        )
        description: dict[str, Any] = {
            "validation_location": list(location),
            "json_pointer": _json_pointer(result.path) if result.path else None,
            "location_status": "located" if result.path else "unresolved",
            "location_note": result.reason,
            "union_branches": list(result.branches),
        }
        # 2. 【模型服务】【错误定位】仅提示根合同确实声明的同名字段，不替模型搬移值
        if error["type"] == "extra_forbidden" and result.path and len(result.path) > 1:
            field = result.path[-1]
            if field in root_properties:
                description["declared_root_field_pointer"] = _json_pointer((field,))
        descriptions.append(description)
    return descriptions
