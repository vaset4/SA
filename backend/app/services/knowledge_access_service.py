"""固定发布版本的正文定位、范围拼装与完整性读取，作者：xxx。"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import re
from typing import Any, Callable, Literal, Sequence, cast

from app.models.knowledge_access_contracts import (
    KnowledgeMaterial, KnowledgeReadBudgetError, KnowledgeReadReport, KnowledgeReadRequest,
    KnowledgeReadResult, SectionResolution, SourceFragment, SourceSpan,
)
from app.models.research_contracts import RetrievalScopeError
from app.models.upload_models import ArtifactIntegrityError
from app.services.upload_read_service import batch_id_from_chunk, read_published_batches

_ATOMIC_TYPES = frozenset({"table", "equation", "formula", "code", "figure_caption"})
_SEPARATOR = "\n\n"
_LEGACY_LIMITATION = "legacy_overlap_not_deduplicated: 旧正文未保存来源项目区间，跨父块重复范围未推测消除"
_FORMULA_PATTERN = re.compile(r"(?<![\\$])(\$\$(?:\\.|(?!\$\$)[^\\])*?\$\$|\$(?!\$)(?:\\.|[^\\$])+\$)", re.DOTALL)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Piece:
    """一个可核验的发布字段区间，来源项目决定是否允许额度切分。"""

    chunk: dict[str, Any]
    field: Literal["text", "full_text"]
    start: int
    end: int
    fragment: SourceFragment | None
    atomic: bool
    source_text: str | None = None
    formula_span: tuple[int, int] | None = None

    @property
    def text(self) -> str:
        """无参数；返回该发布字段中实际承诺的文字。"""
        return self.chunk[self.field][self.start:self.end]


@dataclass(frozen=True)
class _Group:
    """材料归组及按发布顺序展开的真实片段。"""

    manifest: dict[str, Any]
    kind: Literal["parent", "section"]
    locator: str
    rows: tuple[dict[str, Any], ...]
    field: Literal["text", "full_text"]


async def read_knowledge(request: KnowledgeReadRequest, *,
                         publication_reader: Callable[..., dict[str, dict[str, Any]]] = read_published_batches,
                         ) -> KnowledgeReadResult:
    """参数为可信读取请求与发布端口；返回真实材料及说明，取消时等自己持有的 IO 结束。"""
    owned_request = replace(request, scope=deepcopy(dict(request.scope)))
    task = asyncio.create_task(asyncio.to_thread(_read_knowledge, owned_request, publication_reader))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # 1. 【知识访问】【取消读取】文件工作仍属于本请求，等待完成后才释放外层发布占用
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled() and task.exception() is not None:
            logger.error("【知识访问】【取消读取】已取消请求的正文工作失败", exc_info=task.exception())
        raise


def _read_knowledge(request: KnowledgeReadRequest, reader: Callable[..., dict[str, dict[str, Any]]]) -> KnowledgeReadResult:
    """参数为读取请求和同步发布端口；在同一工作线程完成授权、读取与范围拼装。"""
    snapshot = _validate_request(request)
    batch_ids = tuple(dict.fromkeys(snapshot["paper_batches"][paper_id]
                                   for paper_id in request.paper_ids if paper_id in snapshot["paper_batches"]))
    manifests = reader(batch_ids, user_id=request.scope["user_id"], publication=snapshot)
    groups, remaining = _resolve_groups(request, manifests)
    resolved = {(group.manifest["paper_id"], group.manifest["batch_id"], group.locator) for group in groups}
    if any((entry["paper_id"], entry["batch_id"], entry["locator"]) not in resolved for entry in request.continuations):
        raise ValueError("续读定位未在同一发布版本中解析为实际章节")
    materials, unread, used = _read_groups(groups, max_chars=request.max_chars, continuations=request.continuations)
    remaining.extend(unread)
    limitations = (_LEGACY_LIMITATION,) if any("content_schema_version" not in group.manifest for group in groups) else ()
    status: Literal["partial", "complete", "unavailable"] = "partial" if remaining and materials else "complete"
    if remaining and not materials:
        status = "unavailable"
    report = KnowledgeReadReport(
        operation=request.operation, status=status, requested=request.section_paths or request.chunk_ids,
        resolved=tuple(group.locator for group in groups), remaining=tuple(remaining),
        limitations=limitations, used_chars=used,
        read_spans=tuple(span for material in materials for span in material.source_spans),
    )
    return KnowledgeReadResult(tuple(materials), report)


def _validate_request(request: KnowledgeReadRequest) -> dict[str, Any]:
    """参数为正文请求；核对操作、定位及用户快照，返回固定快照，不读取当前版本。"""
    if request.operation not in ("chunks", "sections") or request.granularity not in ("child", "parent"):
        raise ValueError("知识读取操作或粒度无效")
    if not request.paper_ids or any(not isinstance(value, str) or not value.strip() for value in request.paper_ids):
        raise ValueError("知识读取需要明确的目标论文")
    if request.max_chars is not None and (type(request.max_chars) is not int or request.max_chars < 0):
        raise ValueError("知识读取额度必须为非负字符数")
    selectors = request.chunk_ids if request.operation == "chunks" else request.section_paths
    forbidden = request.section_paths if request.operation == "chunks" else request.chunk_ids
    if not selectors or forbidden or any(not isinstance(value, str) or not value.strip() for value in selectors):
        raise ValueError("知识读取定位为空或与操作不匹配")
    scope = request.scope
    if not scope.get("user_id") or not set(request.paper_ids).issubset(scope.get("paper_ids", [])):
        raise RetrievalScopeError("正文目标不在本轮授权论文范围内")
    snapshot = scope.get("publication")
    if not isinstance(snapshot, dict) or snapshot.get("user_id") != scope["user_id"] or not snapshot.get("lease_id"):
        raise RetrievalScopeError("正文读取必须绑定本轮已授权发布快照")
    if not isinstance(snapshot.get("paper_batches"), dict) or not isinstance(snapshot.get("batch_ids"), list):
        raise RetrievalScopeError("正文读取发布快照结构不完整")
    if not set(snapshot["paper_batches"].values()).issubset(snapshot["batch_ids"]):
        raise RetrievalScopeError("正文读取发布快照批次不一致")
    _validate_continuations(request, snapshot)
    return snapshot


def _validate_continuations(request: KnowledgeReadRequest, snapshot: dict[str, Any]) -> None:
    """参数为读取请求及已授权快照；检查续读身份、版本和章节一致性，不接受模型新坐标。"""
    for entry in request.continuations:
        if entry.get("continuation_id") != continuation_identity(entry):
            raise ValueError("续读定位身份与原始未读范围不一致")
        if (entry.get("paper_id") not in request.paper_ids
                or snapshot["paper_batches"].get(entry["paper_id"]) != entry.get("batch_id")):
            raise RetrievalScopeError("续读范围不属于本轮授权发布版本")
        if request.operation != "sections" or entry.get("locator") not in request.section_paths:
            raise ValueError("续读范围与本次章节定位不一致")


def resolve_section_paths(paths: Sequence[str], requested: Sequence[str]) -> tuple[SectionResolution, ...]:
    """参数为发布章节目录及按序定位；返回逐项唯一命中、未找到或歧义，完整路径优先。"""
    canonical = tuple(dict.fromkeys(path.strip() for path in paths if path.strip()))
    results: list[SectionResolution] = []
    for value in requested:
        needle = value.strip()
        if not needle:
            raise ValueError("章节定位不能为空")
        exact = tuple(path for path in canonical if path.casefold() == needle.casefold())
        candidates = exact or tuple(path for path in canonical if _section_matches(path, needle))
        status: Literal["matched", "ambiguous", "not_found"] = (
            "matched" if len(candidates) == 1 else "ambiguous" if candidates else "not_found"
        )
        results.append(SectionResolution(needle, status, candidates))
    return tuple(results)


def _section_matches(path: str, needle: str) -> bool:
    """参数为规范路径和短定位；按末段编号边界或标题文字匹配，避免 3.1 命中 3.10。"""
    compact = "".join(needle.split())
    segment = path.rsplit("/", 1)[-1].strip().casefold()
    if re.fullmatch(r"\d+(?:\.\d+)*\.?", compact):
        return segment.startswith(compact) and not segment[len(compact):len(compact) + 1].isdigit()
    return needle.casefold() in segment


def published_chunk_order(row: dict[str, Any]) -> int:
    """参数为已发布索引行；返回不可变地址的文档序号，非法地址明确失败。"""
    chunk_id = str(row.get("chunk_id") or "")
    batch_id_from_chunk(chunk_id)
    return int(chunk_id.rsplit(":", 1)[-1])


def _resolve_groups(request: KnowledgeReadRequest, manifests: dict[str, dict[str, Any]],
                    ) -> tuple[list[_Group], list[dict[str, Any]]]:
    """参数为请求和已核验清单；生成稳定材料组与各项未读取原因，不解释问题充分性。"""
    groups: list[_Group] = []
    remaining: list[dict[str, Any]] = []
    snapshot = request.scope["publication"]
    for paper_id in request.paper_ids:
        batch_id = snapshot["paper_batches"].get(paper_id)
        if batch_id is None:
            remaining.append({"paper_id": paper_id, "reason": "not_in_publication_snapshot"})
    if request.operation == "chunks":
        return _chunk_groups(request, manifests), remaining
    for paper_id in request.paper_ids:
        batch_id = snapshot["paper_batches"].get(paper_id)
        if batch_id is None:
            continue
        manifest = manifests[batch_id]
        by_path: dict[str, list[dict[str, Any]]] = {}
        for chunk in manifest["chunks"]:
            path = str(chunk.get("section_path") or chunk.get("section") or "").strip()
            if path:
                by_path.setdefault(path, []).append(chunk)
        seen: set[str] = set()
        for result in resolve_section_paths(tuple(by_path), request.section_paths):
            if result.status != "matched":
                remaining.append({"paper_id": paper_id, "locator": result.requested,
                                  "reason": result.status, "candidates": list(result.paths)})
                continue
            path = result.paths[0]
            if path not in seen:
                rows = _unique_parent_rows(by_path[path])
                groups.append(_Group(manifest, "section", path, rows, "full_text"))
                seen.add(path)
    return groups, remaining


def _chunk_groups(request: KnowledgeReadRequest, manifests: dict[str, dict[str, Any]]) -> list[_Group]:
    """参数为明确子块定位与发布正文；按父块分组，子粒度保留实际命中区间。"""
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for chunk_id in dict.fromkeys(request.chunk_ids):
        batch_id = batch_id_from_chunk(chunk_id)
        if batch_id not in manifests:
            raise RetrievalScopeError("命中分块不在请求目标的发布批次中")
        manifest = manifests[batch_id]
        ordinal = published_chunk_order({"chunk_id": chunk_id})
        if ordinal >= len(manifest["chunks"]) or manifest["chunks"][ordinal]["chunk_id"] != chunk_id:
            raise LookupError("正文定位不在已发布分块清单中")
        chunk = manifest["chunks"][ordinal]
        parent_id = chunk.get("parent_id")
        if not parent_id:
            raise ArtifactIntegrityError("正文分块缺少父块身份")
        grouped.setdefault((batch_id, parent_id), []).append(chunk)
    groups: list[_Group] = []
    parent_heads: dict[tuple[str, str], dict[str, Any]] = {}
    if request.granularity == "parent":
        # 1. 【知识访问】【父块身份】父正文固定使用发布时最早锚点，查询命中哪个兄弟子块不改变来源承诺
        for batch_id, manifest in manifests.items():
            for row in manifest["chunks"]:
                parent_heads.setdefault((batch_id, row.get("parent_id")), row)
    for (batch_id, parent_id), rows in grouped.items():
        ordered = sorted(rows, key=published_chunk_order)
        field: Literal["text", "full_text"] = "full_text" if request.granularity == "parent" else "text"
        selected = (parent_heads[(batch_id, parent_id)],) if field == "full_text" else tuple(ordered)
        groups.append(_Group(manifests[batch_id], "parent", parent_id, selected, field))
    return groups


def _unique_parent_rows(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """参数为文档顺序分块；返回每个父块首个可回读锚点，避免重复加载相同父正文。"""
    parents: dict[str, dict[str, Any]] = {}
    for row in rows:
        parent_id = row.get("parent_id")
        if not parent_id:
            raise ArtifactIntegrityError("章节分块缺少父块身份")
        parents.setdefault(parent_id, row)
    return tuple(parents.values())


def _group_pieces(group: _Group) -> list[_Piece]:
    """参数为材料组；按规范来源区间消除重叠，旧正文只采用真实保存文本。"""
    pieces: list[_Piece] = []
    covered: dict[str, list[tuple[int, int]]] = {}
    items: dict[str, Any] = group.manifest.get("source_items", {})
    for row in group.rows:
        text = row.get(group.field)
        if not isinstance(text, str):
            raise ArtifactIntegrityError("发布正文缺少请求的阅读字段")
        if not text.strip():
            continue
        if "content_schema_version" not in group.manifest:
            pieces.append(_Piece(row, group.field, 0, len(text), None, row.get("block_type") in _ATOMIC_TYPES))
            continue
        key = "source_fragments" if group.field == "text" else "parent_source_fragments"
        for fragment in row[key]:
            source_item = fragment["source_item"]
            intervals = covered.get(source_item, [])
            for start, end in _uncovered(fragment["source_start"], fragment["source_end"], intervals):
                offset = start - fragment["source_start"]
                part = {**fragment, "source_start": start, "source_end": end,
                        "text_start": fragment["text_start"] + offset,
                        "text_end": fragment["text_start"] + offset + end - start}
                pieces.append(_Piece(row, group.field, part["text_start"], part["text_end"],
                                     cast(SourceFragment, part), row.get("block_type") in _ATOMIC_TYPES
                                     or items[source_item]["block_type"] in _ATOMIC_TYPES, items[source_item]["text"]))
            covered[source_item] = _merge_intervals(intervals + [(fragment["source_start"], fragment["source_end"])])
    return _protect_formula_pieces(pieces)


def _protect_formula_pieces(pieces: list[_Piece]) -> list[_Piece]:
    """参数为已去重的真实片段；把完整公式交叠区间归为同一原子结构，前后正文仍可单独读取。"""
    text_parts: list[str] = []
    offsets = []
    length = 0
    previous = None
    for piece in pieces:
        separator = _piece_separator(previous, piece)
        text_parts.extend((separator, piece.text))
        length += len(separator)
        offsets.append(length)
        length += len(piece.text)
        previous = piece
    formulas = [match.span() for match in _FORMULA_PATTERN.finditer("".join(text_parts))]
    if not formulas:
        return pieces
    protected: list[_Piece] = []
    formula_index = 0
    for piece, offset in zip(pieces, offsets):
        while formula_index < len(formulas) and formulas[formula_index][1] <= offset:
            formula_index += 1
        if piece.atomic:
            protected.append(piece)
            continue
        cursor = 0
        # 1. 【知识访问】【公式范围】先固定完整表达式，再按实际字段交叠坐标切片，允许跨来源项目
        for index in range(formula_index, len(formulas)):
            start, end = formulas[index]
            if start >= offset + len(piece.text):
                break
            left, right = max(start - offset, 0), min(end - offset, len(piece.text))
            if cursor < left:
                protected.append(_slice_piece(piece, cursor, left))
            protected.append(replace(_slice_piece(piece, left, right), atomic=True, formula_span=(start, end)))
            cursor = right
        if cursor < len(piece.text):
            protected.append(_slice_piece(piece, cursor, len(piece.text)))
    return protected


def _uncovered(start: int, end: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """参数为源区间和已覆盖有序区间；返回仍未出现的部分，不按相似文本猜测。"""
    remaining = []
    cursor = start
    for left, right in covered:
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            remaining.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        remaining.append((cursor, end))
    return remaining


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """参数为同一来源项目的覆盖区间；返回合并后的有序区间，保留未读缺口。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _piece_separator(previous: _Piece | None, current: _Piece) -> str:
    """参数为相邻实际片段；保留原有空白，来源连续部分无额外分隔。"""
    if previous is None:
        return ""
    left, right = previous.fragment, current.fragment
    if left and right and left["source_item"] == right["source_item"] and current.source_text is not None:
        if left["source_end"] <= right["source_start"]:
            gap = current.source_text[left["source_end"]:right["source_start"]]
            if not gap.strip():
                return gap
    if previous.chunk["chunk_id"] == current.chunk["chunk_id"] and previous.field == current.field:
        gap = current.chunk[current.field][previous.end:current.start]
        if previous.end <= current.start and not gap.strip():
            return gap
    return _SEPARATOR


def _read_groups(groups: list[_Group], *, max_chars: int | None,
                 continuations: tuple[dict[str, Any], ...] = (),
                 ) -> tuple[list[KnowledgeMaterial], list[dict[str, Any]], int]:
    """参数为按请求排序的组和已有额度；返回材料、剩余定位及实际字符数，保留部分成果。"""
    materials: list[KnowledgeMaterial] = []
    remaining: list[dict[str, Any]] = []
    used = 0
    exhausted = False
    for group in groups:
        pieces = _group_pieces(group)
        if continuations:
            pieces = _continuation_pieces(group, pieces, continuations)
            if not pieces:
                continue
        allowance = None if max_chars is None else max_chars - used
        selected, unread = _fit_group(pieces, allowance=0 if exhausted else allowance)
        if selected and any(piece.text.strip() for piece in selected):
            material = _material(group, selected)
            materials.append(material)
            used += len(material.text)
        if unread:
            exhausted = True
            remaining.extend(_remaining_entries(group, unread))
            if not materials:
                minimum = _pieces_length(_atomic_unit(unread, 0), previous=None) if unread[0].atomic else 1
                raise KnowledgeReadBudgetError(
                    "读取额度无法容纳首个正文结构", remaining=tuple(remaining), minimum_chars=minimum,
                )
    return materials, remaining, used


def continuation_identity(entry: dict[str, Any]) -> str:
    """参数为系统未读定位；返回绑定论文、发布版本和精确范围的稳定身份，不包含模型输入偏移。"""
    keys = ("paper_id", "batch_id", "locator", "chunk_id", "field", "start", "end", "spans")
    identity = {key: entry[key] for key in keys if key in entry}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return "continue-" + digest


def _continuation_pieces(group: _Group, pieces: list[_Piece], entries: tuple[dict[str, Any], ...]) -> list[_Piece]:
    """参数为发布组、真实片段和已核验续读定位；返回准确未读正文，原子结构不能被切开。"""
    selected: list[_Piece] = []
    for entry in entries:
        if (entry["paper_id"], entry["batch_id"], entry["locator"]) != (
                group.manifest["paper_id"], group.manifest["batch_id"], group.locator):
            continue
        for span in entry.get("spans") or [entry]:
            start, end = span["start"], span["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("续读范围必须是有效的系统字符区间")
            matches = [piece for piece in pieces if piece.chunk["chunk_id"] == span["chunk_id"]
                       and piece.field == span["field"] and piece.start < end and start < piece.end]
            if sum(min(end, piece.end) - max(start, piece.start) for piece in matches) != end - start:
                raise ValueError("续读范围不属于当前发布正文")
            for piece in matches:
                left, right = max(start, piece.start), min(end, piece.end)
                if piece.atomic and (left != piece.start or right != piece.end):
                    raise ValueError("续读不得切开完整表格、公式或代码结构")
                selected.append(_slice_piece(piece, left - piece.start, right - piece.start))
    selected.sort(key=lambda piece: (published_chunk_order(piece.chunk), piece.start))
    for previous, current in zip(selected, selected[1:]):
        if previous.chunk["chunk_id"] == current.chunk["chunk_id"] and previous.field == current.field and current.start < previous.end:
            raise ValueError("续读范围重复或重叠")
    return selected


def _fit_group(pieces: list[_Piece], *, allowance: int | None) -> tuple[list[_Piece], list[_Piece]]:
    """参数为真实片段及本组剩余额度；在自然边界读取文本，表格公式保持原子结构。"""
    if allowance is None:
        return pieces, []
    selected: list[_Piece] = []
    used = 0
    consumed = 0
    for index, piece in enumerate(pieces):
        if index < consumed:
            continue
        if piece.atomic:
            unit = _atomic_unit(pieces, index)
            length = _pieces_length(unit, previous=selected[-1] if selected else None)
            if used + length > allowance:
                return selected, pieces[index:]
            selected.extend(unit)
            used += length
            consumed = index + len(unit)
            continue
        separator = _piece_separator(selected[-1] if selected else None, piece)
        available = allowance - used - len(separator)
        if len(piece.text) <= available:
            selected.append(piece)
            used += len(separator) + len(piece.text)
            continue
        if piece.atomic or available <= 0:
            return selected, pieces[index:]
        length = _natural_cut(piece.text, available)
        if length:
            selected.append(_slice_piece(piece, 0, length))
        return selected, [_slice_piece(piece, length, len(piece.text)), *pieces[index + 1:]]
    return selected, []


def _atomic_unit(pieces: list[_Piece], start: int) -> list[_Piece]:
    """参数为片段和起点；返回整张表或同一公式的实际片段，跨字段公式也共同占额度。"""
    first = pieces[start]
    end = start + 1
    while end < len(pieces):
        current = pieces[end]
        if not current.atomic or current.formula_span != first.formula_span:
            break
        if first.formula_span is None and (current.chunk["chunk_id"] != first.chunk["chunk_id"] or current.field != first.field):
            break
        end += 1
    return pieces[start:end]


def _pieces_length(pieces: list[_Piece], *, previous: _Piece | None) -> int:
    """参数为完整原子结构片段及先前片段；返回包括真实分隔空白的最低空间。"""
    length = 0
    for piece in pieces:
        length += len(_piece_separator(previous, piece)) + len(piece.text)
        previous = piece
    return length


def _natural_cut(text: str, limit: int) -> int:
    """参数为非原子正文和可用字符；优先段落、句子、词边界，返回真实读取终点。"""
    prefix = text[:limit]
    for pattern in (r"\n\s*\n", r"[。！？.!?](?:\s|$)", r"\s+"):
        boundaries = list(re.finditer(pattern, prefix))
        if boundaries:
            end = boundaries[-1].end()
            if prefix[:end].strip():
                return end
    return len(prefix.rstrip())


def _slice_piece(piece: _Piece, start: int, end: int) -> _Piece:
    """参数为原片段及相对裁剪范围；返回同步裁剪来源坐标的新片段。"""
    fragment = piece.fragment
    if fragment is not None:
        fragment = {**fragment, "source_start": fragment["source_start"] + start,
                    "source_end": fragment["source_start"] + end,
                    "text_start": piece.start + start, "text_end": piece.start + end}
    return replace(piece, start=piece.start + start, end=piece.start + end, fragment=fragment)


def _remaining_piece(group: _Group, piece: _Piece) -> dict[str, Any]:
    """参数为组和未读片段；返回可重定位的额度事实，不生成伪证据。"""
    return {"paper_id": group.manifest["paper_id"], "batch_id": group.manifest["batch_id"],
            "locator": group.locator, "chunk_id": piece.chunk["chunk_id"], "field": piece.field,
            "start": piece.start, "end": piece.end, "reason": "atomic_structure_budget" if piece.atomic else "read_budget",
            "minimum_chars": len(piece.text) if piece.atomic else 1}


def _remaining_entries(group: _Group, pieces: list[_Piece]) -> list[dict[str, Any]]:
    """参数为材料组和未读片段；同一原子结构作为完整剩余定位，最低空间包含标题与正文。"""
    entries = []
    consumed = 0
    for index, piece in enumerate(pieces):
        if index < consumed:
            continue
        entry = _remaining_piece(group, piece)
        if piece.atomic:
            unit = _atomic_unit(pieces, index)
            entry.update(minimum_chars=_pieces_length(unit, previous=None), spans=[
                {"chunk_id": part.chunk["chunk_id"], "field": part.field, "start": part.start, "end": part.end}
                for part in unit])
            if all(part.chunk["chunk_id"] == piece.chunk["chunk_id"] and part.field == piece.field for part in unit):
                entry["end"] = unit[-1].end
            if piece.formula_span is not None:
                entry["structure_type"] = "formula"
            consumed = index + len(unit)
        if group.kind == "section":
            entry["continuation_id"] = continuation_identity(entry)
        entries.append(entry)
    return entries


def material_group_id(*, paper_id: str, batch_id: str, group_kind: str, group_locator: str) -> str:
    """参数为论文、批次、分组类型和规范定位；返回不受查询或已读范围影响的材料组身份。"""
    payload = ["knowledge-material-group-v1", paper_id, batch_id, group_kind, group_locator]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return "mg-v1-" + digest


def _material(group: _Group, pieces: list[_Piece]) -> KnowledgeMaterial:
    """参数为稳定材料组和实际片段；组装正文及完整跨度，分隔符不冒充来源文字。"""
    text_parts: list[str] = []
    spans: list[SourceSpan] = []
    length = 0
    previous = None
    for piece in pieces:
        separator = _piece_separator(previous, piece)
        text_parts.extend((separator, piece.text))
        length += len(separator)
        span: SourceSpan = {"paper_id": group.manifest["paper_id"], "batch_id": group.manifest["batch_id"],
                            "chunk_id": piece.chunk["chunk_id"], "field": piece.field,
                            "start": piece.start, "end": piece.end,
                            "text_start": length, "text_end": length + len(piece.text)}
        if piece.fragment is not None:
            span["source_fragments"] = [cast(SourceFragment, dict(piece.fragment))]
        spans.append(span)
        length += len(piece.text)
        previous = piece
    first = pieces[0].chunk
    pages = {piece.fragment.get("page_num") for piece in pieces if piece.fragment}
    status: Literal["exact", "legacy_chunk", "no_physical_location"] = (
        "exact" if pages and None not in pages else "no_physical_location"
    )
    if "content_schema_version" not in group.manifest:
        status = "legacy_chunk"
    page_num = int(next(iter(pages)) or 0) if len(pages) == 1 else 0
    return KnowledgeMaterial(
        paper_id=group.manifest["paper_id"], batch_id=group.manifest["batch_id"],
        group_kind=group.kind, group_locator=group.locator,
        material_group_id=material_group_id(paper_id=group.manifest["paper_id"], batch_id=group.manifest["batch_id"],
                                            group_kind=group.kind, group_locator=group.locator),
        text="".join(text_parts), source_spans=tuple(spans), chunk_id=first["chunk_id"],
        parent_id=first["parent_id"], section_path=str(first.get("section_path") or ""),
        section=str(first.get("section") or ""), page_num=page_num, source_location_status=status,
        block_type=str(first.get("block_type") or "text"),
    )
