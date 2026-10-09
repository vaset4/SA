"""消费侧「按配置总额度取正文，超限写明省略」的公共工具。

工具侧负责描述自己取回了什么；消费侧（主张证据、探测发现、定稿摘录）负责模型可容纳性与
显式省略。三处共用正文额度工具，并把读取报告独立渲染，避免把未读提示变成证据。

作者：xxx
创建时间：2026-09-25 00:00:00
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.config import get_settings
from app.models.research_planner_contracts import (
    ClaimExposure, ConsumerExposure, EvidenceExposure, ResearchClaim, UnprovidedEvidenceRange,
)

_READ_REPORT_PROMPT_FIELDS = (
    "operation", "status", "requested", "resolved", "remaining", "limitations", "used_chars",
)

CLAIM_PROMPT_VERSION = "research-claim-v8"


def configured_consumer_budget() -> int | None:
    """无参数；把配置0解释为不限，消费函数显式传入0仍保持历史空预算语义。"""
    configured = get_settings().rag_planner_consumer_text_max_chars
    return None if configured == 0 else int(configured)


def unprovided_range(record: Mapping[str, Any], *, start: int, end: int) -> UnprovidedEvidenceRange:
    """参数为原材料及系统未提供区间；返回绑定同一正文版本的可选择身份，不修改证据。"""
    field, body = evidence_text_field(record)
    if not 0 <= start < end <= len(body):
        raise ValueError("未提供范围超出当前证据正文")
    values = {"evidence_id": str(record["evidence_id"]), "evidence_version": evidence_version(record),
              "field": field, "start": start, "end": end, "content_hash": content_hash(body[start:end])}
    return UnprovidedEvidenceRange(range_id="range-" + content_hash(values), **values)


def content_hash(value: Any) -> str:
    """参数为可序列化内容；返回稳定摘要，保留 Unicode 字符和原始空白。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evidence_text_field(record: Mapping[str, Any]) -> tuple[str, str]:
    """参数为权威材料；返回真实供模型读取的字段名与原文，不改写正文空白。"""
    for field in ("expanded_text", "excerpt", "summary"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return field, value
    return "excerpt", ""


def evidence_version(record: Mapping[str, Any]) -> str:
    """参数为原始材料；返回正文及已有定位身份的摘要，不重算 Evidence ID。"""
    keys = ("evidence_id", "paper_id", "chunk_id", "layer", "usable_as_citation", "read_contract_version",
            "material_group_id", "source_spans", "expanded_text", "excerpt", "summary")
    return content_hash({key: record.get(key) for key in keys})


def claim_content_hash(claim: ResearchClaim) -> str:
    """参数为具体主张；返回支持内容摘要，主张与单元编号由独立绑定表达。"""
    return content_hash(claim.model_dump(mode="json", exclude={"claim_id", "source_unit_ids"}))


@dataclass(frozen=True, kw_only=True)
class PreparedConsumerInput:
    """【研究输入】【一次准备】正文与不可变投影共同交给签名、模型及即时校验。"""

    evidence_text: str
    claims_text: str
    exposure: ConsumerExposure


def prepare_consumer_input(
    records: Sequence[Mapping[str, Any]], claims: Sequence[ResearchClaim], *,
    evidence_max_chars: int | None, prompt_version: str,
    selected_ranges: Mapping[str, UnprovidedEvidenceRange] | None = None,
) -> PreparedConsumerInput:
    """参数为本次合法材料、主张及配置；返回一次裁剪后的文本和实际可见性。"""
    remaining = evidence_max_chars
    blocks: list[str] = []
    exposures: list[EvidenceExposure] = []
    omitted: list[str] = []
    unprovided: list[UnprovidedEvidenceRange] = []
    for record in records:
        field, body = evidence_text_field(record)
        version = evidence_version(record)
        selection = (selected_ranges or {}).get(str(record["evidence_id"]))
        start, end = (selection.start, selection.end) if selection is not None else (0, len(body))
        if selection is not None and selection != unprovided_range(record, start=start, end=end):
            raise ValueError("选择的送模范围与当前材料版本不一致")
        prefix = "\n".join((f"id: {record['evidence_id']}", f"evidence_version: {version}",
                            f"layer: {record.get('layer')}", f"paper_id: {record.get('paper_id')}",
                            f"field: {field}", f"section: {record.get('section') or 'unknown'}",
                            evidence_source_prompt(record), "text: "))
        room = None if remaining is None else max(0, remaining - len(prefix))
        if not body or room == 0:
            omitted.append(str(record["evidence_id"]))
            if start < end:
                unprovided.append(unprovided_range(record, start=start, end=end))
            continue
        visible = body[start:end] if room is None else fit_within_budget(body[start:end], room)[0]
        visible_end = start + len(visible)
        display = render_exposed_text(body, start, visible_end)
        blocks.append(prefix + display)
        if remaining is not None:
            remaining = max(0, remaining - len(prefix) - len(visible))
        exposures.append(EvidenceExposure(evidence_id=str(record["evidence_id"]), evidence_version=version,
                         field=field, start=start, end=visible_end, content_hash=content_hash(visible)))
        if visible_end < end:
            unprovided.append(unprovided_range(record, start=visible_end, end=end))
    if omitted:
        blocks.append(f"[另有 {len(omitted)} 条证据记录因总额度上限或无可读正文未展开]")
    evidence_text = "\n\n".join(blocks) or "(none)"
    claims_text = _render_claims(claims)
    exposure = ConsumerExposure(
        schema_version="research-exposure-v2",
        evidence=tuple(exposures), claims=tuple(ClaimExposure(claim_id=claim.claim_id,
                       content_hash=claim_content_hash(claim)) for claim in claims),
        acquired_evidence_ids=tuple(str(record["evidence_id"]) for record in records),
        omitted_evidence_ids=tuple(omitted), evidence_max_chars=evidence_max_chars, prompt_version=prompt_version,
        unprovided_ranges=tuple(unprovided), rendering_version="research-consumer-v4",
        rendered_hash=content_hash({"evidence": evidence_text, "claims": claims_text}),
        identity_normalized_rendered_hash=content_hash({
            "evidence": evidence_text, "claims": _render_claims(claims, normalize_identity=True),
        }),
    )
    return PreparedConsumerInput(evidence_text=evidence_text, claims_text=claims_text, exposure=exposure)


def _render_claims(claims: Sequence[ResearchClaim], *, normalize_identity: bool = False) -> str:
    """参数为有序主张及身份规范化开关；返回同一渲染，判等只替换明确 ID 字段。"""
    payloads = [claim.model_dump(mode="json") for claim in claims]
    if normalize_identity:
        # 1. 【研究输入】【重绑判等】正文中的相似编号原样保留，具体来源身份另由规范支持链核对
        payloads = [{**payload, "claim_id": "<claim-id>",
                     "source_claim_ids": ["<claim-id>" for _ in payload["source_claim_ids"]],
                     "source_unit_ids": ["<unit-id>" for _ in payload["source_unit_ids"]]} for payload in payloads]
    return "\n\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads) or "(none)"


def render_read_reports(reports: Sequence[Mapping[str, Any]]) -> str:
    """参数为已保存的读取报告；返回独立提示段，无报告时不虚构完整性说明。"""
    if not reports:
        return ""
    # 1. 【科研取证】【读取说明】完整读取与任务完成分别表达，报告不能获得引用身份
    return "\n".join([
        "Reading reports (tool facts, not evidence and not task-completion judgments):",
        "Unreturned material has not been read. Partial reads do not prove absence or completeness.",
        "Consumer text omissions below are separate from tool reading completeness.",
        *(json.dumps(project_read_report(report), ensure_ascii=False, sort_keys=True) for report in reports),
    ])


def project_read_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """参数为完整读取报告；返回模型可行动的已读和未读事实，底层范围仍保存在原报告。"""
    projected = {key: report[key] for key in _READ_REPORT_PROMPT_FIELDS if key in report}
    if "remaining" in report:
        fields = ("paper_id", "locator", "section_path", "reason", "candidates", "minimum_chars",
                  "structure_type", "continuation_id")
        projected["remaining"] = [{key: item[key] for key in fields if key in item}
                                  for item in report["remaining"]]
    return projected

def evidence_source_prompt(record: Mapping[str, Any]) -> str:
    """参数为实际证据；返回新版范围统计及实际页码，完整坐标由证据 ID 关联保存。"""
    if record.get("read_contract_version") is None:
        return ""
    spans = record["source_spans"]
    fields = sorted({span["field"] for span in spans})
    pages = sorted({fragment["page_num"] for span in spans for fragment in span.get("source_fragments", [])
                    if type(fragment.get("page_num")) is int and fragment["page_num"] > 0})
    # 1. 【科研取证】【来源提示】共享字段只描述一次，避免短正文被大量存储地址挤出模型额度
    parts = [f"location={record['source_location_status']}", f"range_count={len(spans)}",
             f"read_fields={json.dumps(fields, ensure_ascii=False)}", f"pages={json.dumps(pages)}"]
    if record.get("chunk_id"):
        parts.append(f"anchor_chunk={record['chunk_id']} (anchor only)")
    return " ".join(parts)


def fit_within_budget(text: str, limit: int) -> tuple[str, str]:
    """把正文约束在 limit 字符内；放得下原样返回，放不下返回（可见段、带省略说明的展示文本）。

    参数:
        text: 本次要放的正文（保留原有结构空白、按 Unicode 字符计数）
        limit: 本段剩余的可用字符额度
    返回:
        (visible, display)：visible 是不带说明的可见段（可追溯原文前缀），
        display 是放进提示词的展示文本；未截断时两者相同
    """
    if len(text) <= limit:
        return text, text
    # 1. 只裁正文、不裁到中文字符中间：按字符切片即可，末尾空白去掉
    visible = text[:limit].rstrip()
    return visible, render_exposed_text(text, 0, len(visible))


def render_exposed_text(text: str, start: int, end: int) -> str:
    """参数为完整原文及已准备的 Unicode 半开区间；返回该范围的实际展示文本。"""
    visible = text[start:end]
    if start == 0 and end == len(text):
        return visible
    # 1. 【研究输入】【送模观察】沿既有范围复现截断说明，不重新计算预算或修改原文定位
    dropped = len(text) - len(visible)
    return f"{visible}...（已截断，完整 {len(text)} 字，此处省略 {dropped} 字）"
