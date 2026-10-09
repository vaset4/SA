"""
引用链工具。

这一层负责从 Semantic Scholar 获取“论文之间谁引用谁”的关系网络，
补充系统对研究脉络的理解能力。

和其他核心模块的边界如下：
  - `paper_db`：保存系统内部论文元数据
  - `vector_store`：保存论文内容如何被检索
  - `citation_tool`：查询论文之间的外部引用关系

因此这里处理的是“关系网络”，不是“全文内容”，也不是“会话记忆”。

函数：
  - `get_references(paper_id)`：该论文引用了哪些论文
  - `get_citations(paper_id)`：哪些论文引用了该论文
  - `get_citation_tree(paper_id, depth=1)`：双向引用树

限制：
  - `depth` 默认 1，最大 2，防止树爆炸
  - 请求间隔 `sleep(1)`，主动遵守 Semantic Scholar 速率限制
  - 结果缓存到 SQLite，重复调用优先走缓存
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests

from app.config import get_settings
import app.services.paper_db as paper_db

settings = get_settings()

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "paperId,externalIds,title,authors,abstract,year,publicationTypes"

# 当前缓存是一个独立 SQLite 表，而不是走 PostgreSQL 主库。
# 这样做的动机是：引用关系属于“外部可重建数据”，适合本地轻量缓存，不必占主业务表结构。

# SQLite 路径与 paper_db 同目录
def _db_path() -> str:
    """返回引用缓存库路径，复用 `papers.db` 所在目录。"""
    p = Path(settings.data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / "papers.db")


def _ensure_citations_table():
    """确保引用缓存表存在。"""
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                cache_key   TEXT PRIMARY KEY,
                data_json   TEXT NOT NULL,
                cached_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


_citations_table_ready = False

def _init_once():
    """懒初始化缓存表，避免模块导入时立即触发数据库写入。"""
    global _citations_table_ready
    if not _citations_table_ready:
        _ensure_citations_table()
        _citations_table_ready = True


# ============================================================
# 缓存读写
# ============================================================

def _cache_get(key: str) -> Optional[list]:
    """读取引用缓存。"""
    _init_once()
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT data_json FROM citations WHERE cache_key = ?", (key,)
        ).fetchone()
    if row:
        return json.loads(row[0])
    return None


def _cache_set(key: str, data: list):
    """写入引用缓存。"""
    _init_once()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO citations (cache_key, data_json) VALUES (?, ?)",
            (key, json.dumps(data, ensure_ascii=False)),
        )
        conn.commit()


# ============================================================
# Semantic Scholar API 调用
# ============================================================

def _s2_paper_id(paper_id: str) -> str:
    """
    将系统内 `paper_id` 转换为 Semantic Scholar 可接受的 ID 格式。

    - arXiv ID（含数字/点） -> `ArXiv:XXXX.XXXXX`
    - 其他 -> 直接传递（可能本身就是 S2 paper_id）
    """
    import re
    if re.match(r'^\d{4}\.\d+', paper_id):
        return f"ArXiv:{paper_id}"
    return paper_id


def _s2_to_paper(item: dict) -> dict:
    """
    将 Semantic Scholar 返回的论文条目映射成系统 `Paper` 兼容结构。

    注意这里只是“形状兼容”的 dict，表示外部查询结果，
    不代表论文已经正式进入 `paper_db` 或向量库。
    """
    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    authors = [a.get("name", "") for a in (item.get("authors") or [])]
    year = item.get("year")
    return {
        "paper_id":       arxiv_id or item.get("paperId", ""),
        "arxiv_id":       arxiv_id,
        "doi":            ext.get("DOI"),
        "title":          item.get("title", ""),
        "authors":        authors,
        "abstract":       item.get("abstract") or "",
        "pdf_url":        f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        "pdf_path":       None,
        "categories":     [],
        "published_date": str(year) if year else "",
        "source":         "semantic_scholar",
        "analysis":       None,
        "tags":           [],
        "is_vectorized":  False,
        "read_count":     0,
        "added_at":       "",
        "last_accessed":  None,
    }


def _fetch_s2(url: str, params: dict) -> Optional[dict]:
    """执行一次 Semantic Scholar HTTP 请求，失败时显式打印错误。"""
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[CitationTool] S2 API 请求失败: {e}")
        return None


# ============================================================
# 公开接口
# ============================================================

def get_references(paper_id: str) -> List[dict]:
    """
    获取该论文引用了哪些论文（outgoing references）。
    结果缓存，重复调用走缓存。

    这是“向前看”的关系：当前论文依赖了哪些先前工作。
    """
    if not settings.citation_cache_enabled:
        return _fetch_references(paper_id)

    key = f"ref:{paper_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    result = _fetch_references(paper_id)
    _cache_set(key, result)
    return result


def _fetch_references(paper_id: str) -> List[dict]:
    """直接从 Semantic Scholar 拉取 references，不经过缓存判定。"""
    s2_id = _s2_paper_id(paper_id)
    url = f"{_S2_BASE}/paper/{s2_id}/references"
    data = _fetch_s2(url, {"fields": _S2_FIELDS, "limit": 50})
    time.sleep(1)

    if not data:
        return []

    papers = []
    for item in data.get("data", []):
        cited = item.get("citedPaper")
        if cited:
            papers.append(_s2_to_paper(cited))
    return papers


def get_citations(paper_id: str) -> List[dict]:
    """
    获取哪些论文引用了该论文（incoming citations）。
    结果缓存，重复调用走缓存。

    这是“向后看”的关系：后续哪些论文延续了当前工作。
    """
    if not settings.citation_cache_enabled:
        return _fetch_citations(paper_id)

    key = f"cit:{paper_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    result = _fetch_citations(paper_id)
    _cache_set(key, result)
    return result


def _fetch_citations(paper_id: str) -> List[dict]:
    """直接从 Semantic Scholar 拉取 citations，不经过缓存判定。"""
    s2_id = _s2_paper_id(paper_id)
    url = f"{_S2_BASE}/paper/{s2_id}/citations"
    data = _fetch_s2(url, {"fields": _S2_FIELDS, "limit": 50})
    time.sleep(1)

    if not data:
        return []

    papers = []
    for item in data.get("data", []):
        citing = item.get("citingPaper")
        if citing:
            papers.append(_s2_to_paper(citing))
    return papers


def get_citation_tree(paper_id: str, depth: int = 1) -> dict:
    """
    获取引用树（双向：references + citations），深度最大 2。

    这个结构更适合上层做“相关工作脉络”展示，而不是直接用于全文问答。

    返回格式：
    {
      "paper_id": str,
      "references": [Paper, ...],
      "cited_by":   [Paper, ...],
      "depth":      int,
      "children": {   # 仅 depth=2 时有内容
          "paper_id_x": {"references": [...], "cited_by": [...]},
          ...
      }
    }
    """
    # 双重限制：
    # 1. 遵循配置项
    # 2. 代码层面再限制到 2，避免递归树过大
    depth = min(depth, settings.citation_max_depth)
    depth = min(depth, 2)

    refs = get_references(paper_id)
    cits = get_citations(paper_id)

    tree: dict = {
        "paper_id":  paper_id,
        "references": refs,
        "cited_by":   cits,
        "depth":      depth,
        "children":   {},
    }

    if depth >= 2:
        # 二层树只沿 references 扩一层，当前实现没有继续向 cited_by 方向扩展，
        # 这是为了控制调用量和返回体积。
        # 这里只扩一层 references，且最多取前 5 篇，避免关系树爆炸。
        for ref in refs[:5]:
            child_id = ref.get("paper_id", "")
            if child_id and child_id != paper_id:
                child_refs = get_references(child_id)
                tree["children"][child_id] = {
                    "references": child_refs,
                    "cited_by":   [],
                }

    return tree
