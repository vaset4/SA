"""
arXiv 搜索工具模块。

这是系统接入外部论文元数据的入口层，负责把 arXiv SDK 返回的数据统一映射成
项目内部的 `Paper` 结构。

职责边界：
  - 只负责从 arXiv 获取论文元数据
  - 不触发 `PaperAnalysis`
  - 不写入 `paper_db` 或 `vector_store`
  - 不下载 PDF

因此它可以理解为“外部论文发现层”：
先把论文找回来，再由上层节点决定是否分析、入库、向量化。

arXiv API 限速说明：
  官方建议请求间隔不少于 3 秒，本模块通过 `ARXIV_REQUEST_INTERVAL`
  配置控制，在每次实际请求前等待，尽量避免触发限流。
"""

import time
import re
from typing import List, Optional
from datetime import datetime

import arxiv

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.models import Paper, PaperAnalysis
from app.config import get_settings

# 获取配置
settings = get_settings()

# 这是“外部论文发现入口”，常见调用链是：
#   route/node -> arxiv_tool.search()/get_by_id()
#   -> 返回标准 `Paper`
#   -> 上层再决定是否走 paper_analyzer / paper_db / ingest


def _normalize_arxiv_id(raw_id: str) -> str:
    """
    规范化 arXiv ID，去除 "arxiv:" 前缀和版本号后缀。

    arXiv ID 的几种常见形式：
      "arxiv:2301.00001"   → "2301.00001"
      "2301.00001v2"       → "2301.00001"
      "2301.00001"         → "2301.00001"（不变）

    Args:
        raw_id: 原始 arXiv ID 字符串

    Returns:
        规范化后的 ID
    """
    # 去除 "arxiv:" 前缀（大小写不敏感）
    normalized = re.sub(r'^arxiv:', '', raw_id, flags=re.IGNORECASE).strip()
    # 去除版本号后缀，如 "v2"、"v10"
    normalized = re.sub(r'v\d+$', '', normalized)
    return normalized


def _build_paper_from_result(result: arxiv.Result) -> Paper:
    """
    将 arxiv.Result 对象转换为标准 Paper 数据模型。

    arxiv.Result 的关键字段：
      entry_id:    完整 arXiv URL，如 "http://arxiv.org/abs/2301.00001v2"
      title:       论文标题
      authors:     作者列表（arxiv.Author 对象）
      summary:     摘要
      pdf_url:     PDF 下载链接（通常以 .pdf 结尾）
      categories:  分类标签列表
      published:   首次发布时间（datetime 对象）
      doi:         DOI 字符串，未注册时为 None

    Args:
        result: arXiv API 返回的单个论文结果对象

    Returns:
        标准 `Paper` 对象，analysis 字段保持 `None`

    这里完成的是“外部 SDK 对象 -> 内部业务实体”的第一层映射。
    后续 `paper_analyzer`、`paper_db`、`vector_store` 都基于这个统一结构工作。
    """
    # 从 entry_id URL 中提取纯 arXiv ID
    # entry_id 格式："http://arxiv.org/abs/2301.00001v2"
    raw_id = result.entry_id.split('/abs/')[-1]
    arxiv_id = _normalize_arxiv_id(raw_id)

    # 格式化作者列表（Author 对象 → 姓名字符串）
    authors = [author.name for author in result.authors]

    # 格式化发表日期
    published_date = result.published.strftime('%Y-%m-%d') if result.published else ''

    # 构造 PDF URL：统一为 https，方便后续下载或展示。
    pdf_url = result.pdf_url or ''
    if pdf_url and not pdf_url.startswith('https'):
        pdf_url = pdf_url.replace('http://', 'https://')

    return Paper(
        library_id=DEFAULT_LIBRARY_ID,
        paper_id=arxiv_id,           # 使用 arxiv_id 作为系统主键
        arxiv_id=arxiv_id,
        doi=result.doi,              # 部分论文有 DOI，None 表示未注册
        title=result.title.strip(),  # 去除标题首尾空白
        authors=authors,
        abstract=result.summary.strip(),
        pdf_url=pdf_url,
        pdf_path=None,               # 尚未下载，路径为空
        categories=list(result.categories),
        published_date=published_date,
        source='arxiv',
        analysis=None,               # 懒加载，不在此处触发分析
        tags=[],                     # 用户标签初始为空
    )


class ArxivTool:
    """
    arXiv 论文搜索工具。

    封装 arXiv Python SDK，提供三个搜索接口：
      - search:      按关键词/分类/时间范围搜索
      - get_by_id:   按精确 arXiv ID 获取单篇论文
      - get_latest:  获取指定分类的最新论文列表

    使用示例：
        tool = ArxivTool()
        papers = tool.search("retrieval augmented generation", max_results=5)
        for p in papers:
            print(p['title'], p['pdf_url'])
    """

    def __init__(self):
        """初始化搜索工具，从配置读取限流和默认返回数。"""
        # 请求间隔（秒），防止触发 arXiv API 限速
        self._request_interval = settings.arxiv_request_interval
        # 默认最大返回数
        self._default_max_results = settings.arxiv_max_results
        # 上次请求时间戳，用于计算等待时间
        self._last_request_time: float = 0.0

    def _wait_for_rate_limit(self):
        """
        等待直到满足请求间隔要求。

        arXiv 官方建议：相邻两次请求间隔不少于 3 秒。
        此方法在每次实际 API 调用前执行，自动计算并等待剩余时间。

        这层限流封装的好处是：上层节点不需要自己关心请求节奏，
        只要调用 `search/get_by_id/get_latest` 即可。
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            wait_time = self._request_interval - elapsed
            print(f"  [ArxivTool] 限速等待 {wait_time:.1f}s...")
            time.sleep(wait_time)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = None,
        categories: List[str] = None,
        date_from: str = None,
    ) -> List[Paper]:
        """
        按关键词搜索 arXiv 论文。

        查询语法支持布尔运算符：
          "RAG AND LLM"              两个词都包含
          "transformer OR attention"  任一词包含
          "BERT NOT classification"   包含 BERT 但不含 classification

        分类过滤：
          arXiv 分类系统：cs.AI（人工智能）、cs.LG（机器学习）、
          cs.CV（计算机视觉）、cs.CL（计算语言学）、stat.ML（统计机器学习）等。
          传入分类列表时，会用 OR 连接分类条件与关键词查询。

        时间过滤：
          arXiv SDK 不原生支持时间过滤，本方法通过 submittedDate 字段
          在结果层面过滤，因此 max_results 是搜索数量上限，
          实际返回数量可能少于 max_results。

        Args:
            query:       搜索关键词，英文效果最佳
            max_results: 最大返回论文数，None 时使用配置默认值
            categories:  分类过滤列表，如 ["cs.AI", "cs.LG"]，None 表示不过滤
            date_from:   时间过滤，格式 "YYYY-MM-DD"，只返回此日期之后的论文

        Returns:
            `Paper` 对象列表，按相关度排序，analysis 字段均为 None

        这是最常用的入口，通常会被上层路由用于“搜索论文”场景：
          用户问题 -> agent 路由 -> arxiv_tool.search()
          -> 结果交给 paper_analyzer / reporter / 入库链继续处理
        """
        if max_results is None:
            max_results = self._default_max_results

        # 构造 arXiv 查询字符串。
        # 如果指定分类，就把关键词查询和分类过滤组合起来。
        if categories:
            # 多分类用 OR 连接，例：cat:cs.AI OR cat:cs.LG
            cat_filter = ' OR '.join([f'cat:{c}' for c in categories])
            search_query = f'({query}) AND ({cat_filter})'
        else:
            search_query = query

        print(f"  [ArxivTool] 搜索: {search_query!r}（最多 {max_results} 篇）")

        self._wait_for_rate_limit()

        try:
            # 按相关度搜索，适合“用户给主题词找论文”的主路径。
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(client.results(search))

            # 将 arxiv.Result 转换为 Paper 对象
            papers = [_build_paper_from_result(r) for r in results]

            # 时间过滤在结果层面做，因为 arXiv SDK 查询本身不直接支持这个约束。
            if date_from:
                cutoff = datetime.strptime(date_from, '%Y-%m-%d')
                papers = [
                    p for p in papers
                    if p['published_date'] and
                    datetime.strptime(p['published_date'], '%Y-%m-%d') >= cutoff
                ]
                print(f"  [ArxivTool] 时间过滤后剩余 {len(papers)} 篇（{date_from} 之后）")

            print(f"  [ArxivTool] 搜索完成，返回 {len(papers)} 篇论文")
            return papers

        except Exception as e:
            print(f"  [ArxivTool] 搜索失败: {e}")
            return []

    def get_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        按精确 arXiv ID 获取单篇论文。

        用于已知论文 ID 时的精确查询，例如用户输入 "2301.00001"。
        会自动处理 ID 格式（去除前缀和版本号）。

        Args:
            arxiv_id: arXiv ID，支持多种格式：
                      "2301.00001"、"arxiv:2301.00001"、"2301.00001v2"

        Returns:
            `Paper` 对象，找不到时返回 `None`

        这个接口适合“已知精确论文 ID”场景，常见于用户直接给出 arXiv 编号，
        或系统已经从别处拿到了确定的 arXiv ID。
        """
        normalized_id = _normalize_arxiv_id(arxiv_id)
        print(f"  [ArxivTool] 按 ID 获取论文: {normalized_id}")

        self._wait_for_rate_limit()

        try:
            client = arxiv.Client()
            # `id_list` 走精确匹配，不做关键词检索。
            search = arxiv.Search(id_list=[normalized_id])
            results = list(client.results(search))

            if not results:
                print(f"  [ArxivTool] 未找到论文: {normalized_id}")
                return None

            paper = _build_paper_from_result(results[0])
            print(f"  [ArxivTool] 找到论文: {paper['title'][:60]}...")
            return paper

        except Exception as e:
            print(f"  [ArxivTool] 按 ID 获取失败: {e}")
            return None

    def get_latest(
        self,
        categories: List[str] = None,
        max_results: int = 20,
    ) -> List[Paper]:
        """
        获取指定分类的最新论文列表，按发表日期降序排列。

        主要用于“最新论文发现/推荐”场景，而不是用户主动关键词检索。
        它更偏向时间维度的拉取，而不是相关度搜索。

        Args:
            categories:  分类过滤列表，None 时使用配置中的默认分类
            max_results: 最大返回数，默认20

        Returns:
            `Paper` 对象列表，按发表日期降序排列，analysis 字段均为 None
        """
        if categories is None:
            categories = settings.arxiv_default_categories

        # 用 OR 连接所有分类，拉取多个方向的最新论文。
        cat_query = ' OR '.join([f'cat:{c}' for c in categories])
        print(f"  [ArxivTool] 获取最新论文: {categories}（最多 {max_results} 篇）")

        self._wait_for_rate_limit()

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=cat_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,  # 按提交日期降序
            )
            results = list(client.results(search))
            papers = [_build_paper_from_result(r) for r in results]

            print(f"  [ArxivTool] 获取最新论文完成，返回 {len(papers)} 篇")
            return papers

        except Exception as e:
            print(f"  [ArxivTool] 获取最新论文失败: {e}")
            return []


# ============================================================
# 模块级单例（供各节点直接 import 使用）
# ============================================================

# 使用模块级单例避免重复实例化。
# ArxivTool 只有限流时间戳等轻量状态，进程内复用即可。
_arxiv_tool_instance: Optional[ArxivTool] = None


def get_arxiv_tool() -> ArxivTool:
    """
    获取 ArxivTool 单例。

    Returns:
        `ArxivTool` 实例
    """
    global _arxiv_tool_instance
    if _arxiv_tool_instance is None:
        _arxiv_tool_instance = ArxivTool()
    return _arxiv_tool_instance


# ============================================================
# 独立测试
# ============================================================

def test():
    """
    ArxivTool 独立测试。
    运行：python -m app.tools.arxiv_tool
    """
    print("=" * 60)
    print("  ArxivTool 测试")
    print("=" * 60)

    tool = ArxivTool()

    # 测试1：关键词搜索
    print("\n[测试1] 关键词搜索：RAG")
    papers = tool.search("retrieval augmented generation", max_results=3)
    print(f"返回 {len(papers)} 篇论文：")
    for p in papers:
        print(f"  - [{p['arxiv_id']}] {p['title'][:60]}...")
        print(f"    作者: {', '.join(p['authors'][:2])}...")
        print(f"    日期: {p['published_date']}")
        print(f"    分类: {p['categories']}")
        print(f"    analysis: {p['analysis']}")  # 应为 None

    # 测试2：带分类过滤
    print("\n[测试2] 分类过滤：cs.AI")
    papers2 = tool.search("large language model", max_results=2, categories=["cs.AI"])
    print(f"返回 {len(papers2)} 篇：")
    for p in papers2:
        print(f"  - {p['title'][:50]}... | 分类: {p['categories']}")

    # 测试3：按 ID 获取
    print("\n[测试3] 按 ID 获取：arxiv:1706.03762")
    paper = tool.get_by_id("arxiv:1706.03762")
    if paper:
        print(f"  标题: {paper['title']}")
        print(f"  arxiv_id: {paper['arxiv_id']}")
        print(f"  doi: {paper['doi']}")

    print("\n测试完成")


if __name__ == "__main__":
    test()
