"""搜索工具模块"""
import re
import time
from typing import List, Dict, Any

import requests
from tavily import TavilyClient

SEARCH_QUERY_MAX_CHARS = 240
SEARCH_QUERY_PREVIEW_CHARS = 160
_INSTRUCTION_NOISE_PATTERN = re.compile(r"(?:^|\s)(?:\d+[\.\)]|[-*•]+)\s*")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# 这个工具主要服务“联网搜索补充上下文”场景，和 Milvus / LightRAG 不同：
#   - 它查的是外部网页
#   - 不返回 chunk_id / paper_id 体系
#   - 更适合调研、deep research、推荐外部资料

# 请求频率限制（秒）
_LAST_REQUEST_TIME = 0
_MIN_REQUEST_INTERVAL = 1.0  # 最小请求间隔 1 秒


def _collapse_whitespace(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _trim_query(value: str, max_chars: int) -> str:
    compact = _collapse_whitespace(value)
    if len(compact) <= max_chars:
        return compact
    truncated = compact[: max_chars - 3].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = compact[: max_chars - 3].strip()
    return truncated + "..."


def build_web_search_query(topic: str, instruction: str) -> str:
    """把 topic 与 instruction 收敛成适合 Tavily 的单条查询。"""
    topic_text = _collapse_whitespace(topic)
    instruction_text = _INSTRUCTION_NOISE_PATTERN.sub(" ", instruction or "")
    instruction_text = _collapse_whitespace(instruction_text)
    if not instruction_text:
        return _trim_query(topic_text, SEARCH_QUERY_MAX_CHARS)
    return _trim_query(f"{topic_text} {instruction_text}", SEARCH_QUERY_MAX_CHARS)


class SearchResult:
    """搜索结果数据类"""

    def __init__(self, title: str, content: str, url: str, score: float = 0.0):
        self.title = title
        self.content = content
        self.url = url
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "score": self.score
        }

    def format_simple(self) -> str:
        """简单格式化（不显示URL）"""
        return f"""
标题: {self.title}
内容: {self.content}
"""

    def format_detailed(self) -> str:
        """详细格式化（显示URL和评分）"""
        return f"""
标题: {self.title}
内容: {self.content}
来源: {self.url}
相关度: {self.score:.2f}
"""


class SearchTool:
    """
    Tavily搜索工具封装

    提供简单的搜索接口，返回格式化的搜索结果。
    """

    def __init__(self, api_key: str, max_results: int = 5):
        """
        初始化搜索工具

        Args:
            api_key (str): Tavily API密钥
            max_results (int): 默认最大结果数
        """
        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    def search(self, query: str, max_results: int = None, show_details: bool = False) -> tuple[str, List[SearchResult]]:
        """
        执行搜索并返回格式化结果

        Args:
            query (str): 搜索查询
            max_results (int): 最大结果数，None则使用默认值
            show_details (bool): 是否显示详细信息（URL等）

        Returns:
            tuple: (格式化的搜索结果字符串, SearchResult列表)
        """
        global _LAST_REQUEST_TIME

        if max_results is None:
            max_results = self.max_results

        # 验证查询
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace-only")

        # 清理查询（移除多余空格）
        cleaned_query = _collapse_whitespace(query)

        # 验证清理后的查询
        if not cleaned_query:
            raise ValueError(f"Search query is empty after cleaning: {query!r}")

        # 频率限制
        # 这里是进程内最小间隔控制，避免同一 worker 在短时间内把 Tavily 打爆。
        elapsed = time.time() - _LAST_REQUEST_TIME
        if elapsed < _MIN_REQUEST_INTERVAL:
            sleep_time = _MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)

        # 重试逻辑
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                _LAST_REQUEST_TIME = time.time()
                response = self.client.search(
                    query=cleaned_query,
                    max_results=max_results
                )
                break  # 成功则跳出重试循环

            except requests.exceptions.HTTPError as error:
                status_code = getattr(getattr(error, "response", None), "status_code", "unknown")
                response_text = getattr(getattr(error, "response", None), "text", "")
                preview = _trim_query(cleaned_query, SEARCH_QUERY_PREVIEW_CHARS)

                # 如果是最后一次尝试，抛出错误
                if attempt == max_retries - 1:
                    error_msg = (
                        f"Tavily search failed after {max_retries} attempts: status={status_code}, "
                        f"query_length={len(cleaned_query)}, query_preview={preview!r}"
                    )

                    # 添加响应详情（如果有）
                    if response_text:
                        error_msg += f", response={response_text[:200]}"

                    raise ValueError(error_msg) from error

                # 如果是 400 错误，不重试（参数问题）
                if status_code == 400:
                    error_msg = (
                        f"Tavily search bad request (400): "
                        f"query_length={len(cleaned_query)}, query_preview={preview!r}"
                    )
                    if response_text:
                        error_msg += f", response={response_text[:200]}"
                    raise ValueError(error_msg) from error

                # 其他错误，等待后重试
                print(f"[SearchTool] 搜索失败 (尝试 {attempt + 1}/{max_retries}): {status_code}, 等待 {retry_delay}s 后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避

            except Exception as error:
                # 其他异常，如果是最后一次尝试则抛出
                if attempt == max_retries - 1:
                    raise
                print(f"[SearchTool] 搜索异常 (尝试 {attempt + 1}/{max_retries}): {type(error).__name__}: {error}")
                time.sleep(retry_delay)
                retry_delay *= 2

        # 解析结果
        # Tavily 原始 payload 会被收敛成 SearchResult，方便上层统一格式化或直接序列化。
        search_results = []
        for item in response.get("results", []):
            result = SearchResult(
                title=item.get('title', ''),
                content=item.get('content', ''),
                url=item.get('url', ''),
                score=item.get('score', 0.0)
            )
            search_results.append(result)

        # 格式化结果
        if show_details:
            formatted = "\n---\n".join([r.format_detailed() for r in search_results])
        else:
            formatted = "\n---\n".join([r.format_simple() for r in search_results])

        return formatted, search_results
