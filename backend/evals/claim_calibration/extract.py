"""从冻结运行提取回答主张和 Evidence 关系。作者：xxx。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evals.contracts import grading_projection, read_snapshot, record_contracts, snapshot, snapshot_hash
from evals.evidence.alignment import covered_span_ids
from evals.evidence.contracts import EvidenceBenchmark

from .contracts import AnswerPosition, ClaimExtraction, ClaimReviewItem


def extract_claims(
    record: Mapping[str, Any], benchmark: EvidenceBenchmark | Mapping[str, Any] | None = None,
) -> ClaimExtraction:
    """参数为冻结运行和可选原文基准；返回实际 Claim、回答位置及证据映射。"""
    case, run = record_contracts(dict(record))
    observation = read_snapshot(run.observation_snapshot)
    review_status = _review_status(record, observation)
    eligible = run.run_mode == "agent_live" and review_status == "reviewed"
    raw_results = observation.get("research_unit_results")
    if not isinstance(raw_results, list):
        return _missing_extraction(
            dict(record), case.case_id, run.run_id, run.run_mode, review_status,
            eligible, "research_unit_results_missing",
        )
    blocks = _blocks(observation)
    answer = _answer(observation, blocks)
    evidence = _evidence_records(observation)
    span_ids, mappings = _evidence_links(benchmark)
    items: list[ClaimReviewItem] = []
    for result in raw_results:
        if not isinstance(result, Mapping):
            continue
        for raw_claim in _list(result.get("claims")):
            if isinstance(raw_claim, Mapping):
                items.append(_claim_item(raw_claim, blocks, evidence, span_ids, mappings))
    if not items:
        return _missing_extraction(
            dict(record), case.case_id, run.run_id, run.run_mode, review_status,
            eligible, "claims_missing",
        )
    complete = all(not item.missing_fields for item in items)
    reason = "claims_and_answer_positions_extracted" if complete else "claims_extracted_with_missing_links"
    return ClaimExtraction(
        case_id=case.case_id, run_id=run.run_id, run_mode=run.run_mode,
        review_status=review_status, human_quality_eligible=eligible, status="measured",
        reason=reason, items=tuple(items), answer=answer,
        projection_hash=snapshot_hash(snapshot(grading_projection(case, run))),
    )


def _missing_extraction(
    record: dict[str, Any], case_id: str, run_id: str, run_mode: str,
    review_status: str, eligible: bool, reason: str,
) -> ClaimExtraction:
    """参数为运行身份和缺失原因；返回不含猜测主张的缺失提取结果。"""
    case, run = record_contracts(record)
    return ClaimExtraction(
        case_id=case_id, run_id=run_id, run_mode=run_mode, review_status=review_status,
        human_quality_eligible=eligible, status="missing", reason=reason,
        projection_hash=snapshot_hash(snapshot(grading_projection(case, run))),
    )


def _claim_item(
    raw: Mapping[str, Any], blocks: list[dict[str, Any]], evidence: dict[str, dict[str, Any]],
    spans: set[str], mappings: dict[tuple[str, str, str | None], tuple[str, ...]],
) -> ClaimReviewItem:
    """参数为实际主张、回答块和证据索引；返回一条可审计评审项。"""
    claim_id = _text(raw.get("claim_id"))
    claim_type = _text(raw.get("claim_type"))
    supporting = _strings(raw.get("supporting_evidence_ids"))
    related = _strings(raw.get("related_evidence_ids"))
    positions = tuple(item for index, block in enumerate(blocks)
                      if claim_id in _strings(block.get("claim_ids"))
                      and (item := _positions(index, block)) is not None)
    cited = tuple(sorted({item for position in positions for item in _strings(blocks[position.block_index].get("evidence_ids"))}))
    evidence_ids = tuple(sorted(set(cited + tuple(supporting))))
    span_links = {
        item: tuple(span for span in _linked_spans(evidence.get(item, {}), mappings) if span in spans)
        for item in evidence_ids
    }
    linked_spans = tuple(sorted({span for values in span_links.values() for span in values}))
    observed_ids = tuple(sorted(evidence))
    citable_ids = tuple(sorted(item for item in cited if _is_citable(evidence.get(item, {}))))
    missing: list[str] = []
    if not positions:
        missing.append("answer_position")
    if cited and not evidence:
        missing.append("evidence_records")
    if supporting and not spans:
        missing.append("evidence_spans")
    return ClaimReviewItem(
        claim_id=claim_id, text=_text(raw.get("text")), claim_type=claim_type,
        validation_status=_text(raw.get("validation_status")),
        requires_citation=claim_type == "paper_supported", cited_evidence_ids=cited,
        supporting_evidence_ids=tuple(supporting), related_evidence_ids=tuple(related),
        citable_evidence_ids=citable_ids, observed_evidence_ids=observed_ids,
        evidence_span_ids=linked_spans, evidence_span_links=span_links,
        answer_positions=positions, missing_fields=tuple(sorted(set(missing))),
    )


def _is_citable(record: Mapping[str, Any]) -> bool:
    """参数为实际 Evidence 记录；返回是否声明为可引用直接文本。"""
    return bool(record.get("usable_as_citation")) and record.get("layer") == "direct_text"


def _positions(index: int, block: Mapping[str, Any]) -> AnswerPosition | None:
    """参数为回答块位置；返回主张在最终回答中的稳定定位。"""
    role = _text(block.get("role"))
    if not role:
        return None
    text = block.get("text")
    return AnswerPosition(block_index=index, block_role=role, text_span=text if isinstance(text, str) else None)


def _blocks(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """参数为运行观察；返回最终回答块，不把自然语言拆成猜测主张。"""
    raw = observation.get("planner_final_blocks")
    return [dict(item) for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def _answer(observation: Mapping[str, Any], blocks: list[dict[str, Any]]) -> str:
    """参数为运行观察和回答块；返回实际回答文本。"""
    response = observation.get("response") or observation.get("answer")
    if isinstance(response, str) and response.strip():
        return response
    return "\n".join(str(item.get("text")) for item in blocks if isinstance(item.get("text"), str)).strip()


def _evidence_records(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """参数为运行观察；按 Evidence ID 保存真实引用记录。"""
    raw = observation.get("evidence_records")
    if not isinstance(raw, list):
        return {}
    return {str(item["evidence_id"]): dict(item) for item in raw if isinstance(item, Mapping) and _text(item.get("evidence_id"))}


def _evidence_links(
    benchmark: EvidenceBenchmark | Mapping[str, Any] | None,
) -> tuple[set[str], dict[tuple[str, str, str | None], tuple[str, ...]]]:
    """参数为 Evidence 基准；返回 span 身份和 IndexMapping 到 span 的关系。"""
    if benchmark is None:
        return set(), {}
    model = benchmark if isinstance(benchmark, EvidenceBenchmark) else EvidenceBenchmark.model_validate(benchmark)
    spans = {span.span_id: span for span in model.spans}
    links: dict[tuple[str, str, str | None], tuple[str, ...]] = {}
    for index in model.index_snapshots:
        for mapping in index.mappings:
            key = (mapping.paper_id, mapping.chunk_id, mapping.parent_id)
            links[key] = tuple(sorted(covered_span_ids(model.spans, mapping.ranges)))
    return set(spans), links


def _linked_spans(
    record: Mapping[str, Any], mappings: dict[tuple[str, str, str | None], tuple[str, ...]],
) -> tuple[str, ...]:
    """参数为 Evidence 记录和 IndexMapping；返回显式或可验证映射的 span 身份。"""
    explicit = tuple(_strings(record.get("span_ids")))
    if explicit:
        return explicit
    key = (_text(record.get("paper_id")), _text(record.get("chunk_id")), _nullable_text(record.get("parent_id")))
    return mappings.get(key, ())


def _review_status(record: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    """参数为冻结记录和观察；返回显式审核状态，缺失时保持 candidate。"""
    agent_case = record.get("agent_case")
    for payload in (record, agent_case if isinstance(agent_case, Mapping) else {}, observation):
        value = payload.get("review_status")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "candidate"


def _list(value: Any) -> list[Any]:
    """参数为任意值；仅返回真实列表。"""
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    """参数为任意值；返回非空文本身份，不将自然语言转成身份。"""
    return [str(item).strip() for item in _list(value) if isinstance(item, str) and item.strip()]


def _text(value: Any) -> str:
    """参数为任意值；返回去空白文本，缺失保持空字符串。"""
    return value.strip() if isinstance(value, str) else ""


def _nullable_text(value: Any) -> str | None:
    """参数为任意值；返回非空文本或 None。"""
    text = _text(value)
    return text or None
