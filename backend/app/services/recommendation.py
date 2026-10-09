"""
推荐与推送服务

这个模块负责“新论文入库后的推荐与通知”副作用逻辑，不属于主问答链、
主检索链或主入库链。

当前职责：
  - 基于知识库已有论文分析结果构建一个轻量兴趣画像
  - 对新论文做启发式相关度打分并生成推荐理由
  - 按配置把推荐结果打印到终端或落盘到待推送文件

需要注意：
  - 它不做定时任务调度
  - 它不决定论文是否入库，只处理入库后的推荐/推送
  - 它更接近 `write_back`/通知后处理，而不是核心 agent 执行路径
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.models.models import Paper
from app.config import get_settings
import app.services.paper_db as paper_db

settings = get_settings()
logger = logging.getLogger(__name__)

# 推荐服务不是主动触发的“主流程”，目前主要在 `rag_trace_service.execute_write_back()`
# 这类收尾阶段被顺手调用，用来做“答完之后是否要推一篇新论文”的副作用判断。


class UserProfile:
    """轻量用户兴趣画像，组合显式兴趣与从已有论文自动归纳出的兴趣词。"""

    def __init__(
        self,
        explicit_keywords: List[str] = None,
        explicit_categories: List[str] = None,
        auto_keywords: List[str] = None,
    ):
        self.explicit_keywords   = explicit_keywords or []
        self.explicit_categories = explicit_categories or []
        self.auto_keywords       = auto_keywords or []

    @property
    def all_keywords(self) -> List[str]:
        """返回显式关键词与自动关键词的去重并集。"""
        return list(set(self.explicit_keywords + self.auto_keywords))


def build_profile() -> UserProfile:
    """
    从业务库已有论文的 `analysis.keywords` 自动构建兴趣画像。

    这里读取的是当前知识库中已分析论文的关键词分布，不依赖会话态，也不读取
    向量库；高频词（出现 >= 2 次）会进入 `auto_keywords`。
    """
    papers = paper_db.list_papers(limit=200)
    freq: dict = {}
    for p in papers:
        if p.get('analysis') and p['analysis'].get('keywords'):
            for kw in p['analysis']['keywords']:
                kw = kw.strip().lower()
                if kw:
                    freq[kw] = freq.get(kw, 0) + 1

    # 这里的画像非常轻量，本质上是高频关键词统计，而不是复杂用户建模。
    auto_keywords = [kw for kw, cnt in freq.items() if cnt >= 2]
    return UserProfile(auto_keywords=auto_keywords)


def score_paper(paper: Paper, profile: Optional[UserProfile] = None) -> Tuple[float, str]:
    """
    对单篇论文做启发式推荐打分。

    打分维度：
      分类匹配  0.3   categories ∩ explicit_categories 占比
      关键词匹配 0.4  paper 摘要/关键词 vs profile 关键词 词频重叠
      兴趣词命中 0.3  paper keywords ∩ profile.auto_keywords 占比

    Returns:
        `(score 0-1, reason 推荐理由)`。

    这是后处理推荐分，不参与 RAG 召回排序，也不是 reranker 分数。
    """
    if profile is None:
        profile = build_profile()

    score = 0.0
    reasons = []

    # 1. 分类匹配（0.3）：更适合显式订阅类别场景。
    if profile.explicit_categories:
        cat_match = len(set(paper.get('categories', [])) & set(profile.explicit_categories))
        cat_score = min(cat_match / max(len(profile.explicit_categories), 1), 1.0)
        score += 0.3 * cat_score
        if cat_match:
            matched = list(set(paper.get('categories', [])) & set(profile.explicit_categories))
            reasons.append(f'分类匹配: {", ".join(matched)}')

    # 2. 关键词频率匹配（0.4）：用标题/摘要中的命中做粗粒度相关度估计。
    if profile.all_keywords:
        text = (paper.get('abstract', '') + ' ' + paper.get('title', '')).lower()
        hits = sum(1 for kw in profile.all_keywords if kw in text)
        kw_score = min(hits / max(len(profile.all_keywords), 1), 1.0) * 2  # 放大系数
        kw_score = min(kw_score, 1.0)
        score += 0.4 * kw_score
        if hits:
            reasons.append(f'关键词命中 {hits}/{len(profile.all_keywords)} 个')

    # 3. 兴趣词命中（0.3）：对 analysis.keywords 的直接交集做补充加权。
    paper_keywords = []
    if paper.get('analysis') and paper['analysis'].get('keywords'):
        paper_keywords = [kw.lower() for kw in paper['analysis']['keywords']]
    if profile.auto_keywords and paper_keywords:
        hits = len(set(paper_keywords) & set(profile.auto_keywords))
        interest_score = min(hits / max(len(profile.auto_keywords), 1) * 3, 1.0)
        score += 0.3 * interest_score
        if hits:
            reasons.append(f'兴趣词命中 {hits} 个')

    reason = '；'.join(reasons) if reasons else '一般相关'
    return round(score, 3), reason


def push_notification(paper: Paper, score: float, reason: str) -> None:
    """
    按当前配置输出推荐通知。

    `push_mode=terminal` 时直接打印到终端；
    `push_mode=file` 时追加到 `data/pending_notifications.json`。

    这里只负责生成待消费通知，不负责真正的外部消息发送。
    """
    mode = settings.push_mode

    if mode == 'terminal':
        print(
            f'\n[推送] 新论文推荐 (score={score:.2f})\n'
            f'  标题: {paper["title"]}\n'
            f'  arXiv: {paper.get("arxiv_id", "N/A")}\n'
            f'  理由: {reason}\n'
        )
    elif mode == 'file':
        # 文件模式下只是落“待消费通知”，后续是否真有外部推送器读取它，不在本模块职责里。
        p = Path(settings.data_dir) / 'pending_notifications.json'
        existing = []
        if p.exists():
            try:
                existing = json.loads(p.read_text(encoding='utf-8'))
            except Exception as exc:
                logger.warning(
                    "pending notification read failed path=%s: %s",
                    p,
                    exc,
                    exc_info=True,
                )
        existing.append({
            'paper_id': paper['paper_id'],
            'title':    paper['title'],
            'arxiv_id': paper.get('arxiv_id'),
            'score':    score,
            'reason':   reason,
        })
        p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
