"""Planner 上下文预装载测试（单元 1，09-12 卡）。

覆盖点：
- 索引状态四态：判据是章节行 + is_vectorized，不是「有没有摘要」；
  库记录已索引却查不到行是独立的第四态，不并进「未索引」
- 章节窄查询：只投影 section_path，一篇查不到不影响其余论文
- 装载失败：论文库/章节查询的基础设施故障落成显式失败载荷（带失败面与可读原因），
  绝不吞成「本轮没有论文」
- 空是空、不是失败：论文没章节、图库无关系照常返回空段
- 图谱面软降级：查不到只让该段写「无」，清单与章节段照常，失败原因进 source_status_reasons
- 单次读库：焦点解析读到的论文记录被装载器复用，同一回合不重复查库
- 清单顺序：走保序的 ID 列表，不是 frozenset 哈希序
- 首次读取（F05）：焦点解析那一次读库由装载层注入，动作包在既有重试里；
  瞬断一次后成功、重试用尽报论文库面；最终失败归因成上下文装载失败，不是深度评估失败；
  没有授权论文时不惊动读取

作者：xxx
创建时间：2026-09-13 00:00:00
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import get_settings
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research.focus_resolver import resolve_focus
from app.services.internal_research.turn_context_resolver import TURN_PAPERS_KEY, resolve_turn_context

# 预装载三段在状态里的键名
_PAPER_LIST_KEY = "_paper_list_lines"
_SECTION_KEY = "_section_structure_lines"
_GRAPH_KEY = "_graph_status_lines"

_USER_ID = "u1"


class _StubPaperRepo:
    """按 ID 返回预置论文记录的仓储替身，记录每次调用。"""

    def __init__(self, papers: list[dict[str, Any]] | Exception) -> None:
        self._papers = papers
        self.calls: list[tuple[list[str], str]] = []

    async def get_papers_by_ids(self, paper_ids: list[str], *, user_id: str) -> list[dict[str, Any]]:
        self.calls.append((list(paper_ids), user_id))
        if isinstance(self._papers, Exception):
            raise self._papers
        by_id = {str(p.get("paper_id")): p for p in self._papers}
        return [by_id[pid] for pid in paper_ids if pid in by_id]


class _StubMilvusStore:
    """按 paper_id 返回预置章节路径的 Milvus 替身，记录每次窄查询。"""

    def __init__(self, sections: dict[str, list[str] | Exception]) -> None:
        self._sections = sections
        self.queried: list[str] = []
        self.bound_user: str = ""
        self.bound_snapshots: list[dict | None] = []

    def for_user(self, user_id: str) -> "_StubMilvusStore":
        self.bound_user = user_id
        return self

    def for_snapshot(self, snapshot: dict | None) -> "_StubMilvusStore":
        # 留痕上下文装载盖章的发布快照，供断言章节名单与本轮版本一致
        self.bound_snapshots.append(snapshot)
        return self

    def list_section_paths(self, paper_id: str) -> list[str]:
        self.queried.append(paper_id)
        result = self._sections.get(paper_id, [])
        if isinstance(result, Exception):
            raise result
        return result


class _StubGraphService:
    """返回预置子图统计的图谱查询替身。"""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self._result = result
        self.calls: list[list[str]] = []

    async def query_project_subgraph(self, paper_ids: list[str]) -> dict[str, Any]:
        self.calls.append(list(paper_ids))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _paper(
    paper_id: str,
    *,
    title: str = "标题",
    analysis_status: str = "not_started",
    is_vectorized: bool = False,
    abstract: str = "",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": title,
        "analysis_status": analysis_status,
        "is_vectorized": is_vectorized,
        "abstract": abstract,
    }


def _state(paper_ids: list[str], **extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_query": "这几篇论文讲了什么",
        "project_scope": {"user_id": _USER_ID, "paper_ids": paper_ids, "publication": {
            "user_id": _USER_ID, "lease_id": "fixture-lease", "batch_ids": [f"fixture-{paper_id}" for paper_id in paper_ids],
            "paper_batches": {paper_id: f"fixture-{paper_id}" for paper_id in paper_ids},
        }},
        "structured_query": {"paper_ids": paper_ids},
        "focus": {"project_scope": paper_ids, "conversation_scope": [], "turn_focus": paper_ids},
    }
    state.update(extra)
    return state


def _graph_result(node_count: int = 0, edge_count: int = 0) -> dict[str, Any]:
    return {"meta": {"node_count": node_count, "edge_count": edge_count}}


def _load(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
    *,
    repo: _StubPaperRepo | None = None,
    store: _StubMilvusStore | None = None,
    graph: _StubGraphService | None = None,
) -> dict[str, Any]:
    """把三个外部依赖换成替身后跑一次装载，返回状态增量。"""
    repo = repo or _StubPaperRepo([])
    store = store or _StubMilvusStore({})
    graph = graph or _StubGraphService(_graph_result())
    monkeypatch.setattr(pipeline, "get_paper_repository", lambda: repo)
    monkeypatch.setattr(pipeline, "get_milvus_store", lambda: store)
    monkeypatch.setattr(pipeline, "get_research_object_query_service", lambda: graph)
    return asyncio.run(pipeline._load_context(state))


def _status_of(line: str) -> str:
    """从清单行 `- id | title | status` 里取状态列。"""
    return line.rsplit("|", 1)[-1].strip()


class TestIndexStatusJudgesOnSectionRows:
    """验索引状态按检索侧到底有没有章节行分档，不是按「有没有摘要」。"""

    def test_three_papers_with_abstracts_do_not_collapse_into_one_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """三篇都有摘要时仍要分出三种状态。

        按摘要分档会把真实库里三篇 is_vectorized=true 的论文全渲染成同一态，
        模型看到的是一份「大家都一样」的清单，没法据它决定该查哪篇。
        """
        papers = [
            _paper("p1", analysis_status="analysis_ready", is_vectorized=True, abstract="摘要一"),
            _paper("p2", is_vectorized=True, abstract="摘要二"),
            _paper("p3", is_vectorized=False, abstract="摘要三"),
        ]
        store = _StubMilvusStore({"p1": ["Method"], "p2": ["Intro"], "p3": []})

        delta = _load(
            monkeypatch,
            _state(["p1", "p2", "p3"]),
            repo=_StubPaperRepo(papers),
            store=store,
        )

        statuses = [_status_of(line) for line in delta[_PAPER_LIST_KEY]]
        assert len(set(statuses)) == 3

    def test_vectorized_without_rows_is_its_own_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """库说已向量化却查不到行时如实写「取不到内容」，不写「未索引」。

        两者的处置完全不同：未索引要重新入库，库记录已索引却取不到内容要查索引为什么丢了。
        并成一态会让用户按错的方向修。
        """
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1", is_vectorized=True, abstract="摘要")]),
            store=_StubMilvusStore({"p1": []}),
        )

        status = _status_of(delta[_PAPER_LIST_KEY][0])
        assert status == pipeline._INDEX_STATUS_VECTORIZED_NO_ROWS
        assert status != pipeline._INDEX_STATUS_NOT_INDEXED

    def test_no_rows_and_not_vectorized_reads_as_not_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """既没章节行、库也没标向量化：这篇确实没索引。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1")]),
            store=_StubMilvusStore({"p1": []}),
        )

        assert _status_of(delta[_PAPER_LIST_KEY][0]) == pipeline._INDEX_STATUS_NOT_INDEXED

    def test_rows_plus_analysis_ready_reads_as_analyzed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """有章节行且分析已完成：已分析。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1", analysis_status="analysis_ready", is_vectorized=True)]),
            store=_StubMilvusStore({"p1": ["Method", "Results"]}),
        )

        assert _status_of(delta[_PAPER_LIST_KEY][0]) == pipeline._INDEX_STATUS_ANALYZED

    def test_missing_title_never_shows_as_an_empty_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """标题缺失时写「无标题」，不用空串冒充有值。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1", title="")]),
            store=_StubMilvusStore({"p1": []}),
        )

        assert "无标题" in delta[_PAPER_LIST_KEY][0]


class TestSectionStructurePreload:
    """验章节结构段：预装载章节名单，一篇查不到不影响其余。"""

    def test_each_paper_gets_one_narrow_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """每篇论文查一次章节，绑定当前用户身份。

        不绑身份时 Milvus 替身按 owner 收窄的批次取不到，等于跨用户读别人的库。
        """
        store = _StubMilvusStore({"p1": ["Intro"], "p2": ["Method"]})

        _load(monkeypatch, _state(["p1", "p2"]), repo=_StubPaperRepo([_paper("p1"), _paper("p2")]), store=store)

        assert store.queried == ["p1", "p2"]
        assert store.bound_user == _USER_ID

    def test_section_narrow_query_carries_the_round_publication_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """章节名单是直读的第一跳：盖上本轮发布快照，后续读正文才不会落到新版本上。"""
        store = _StubMilvusStore({"p1": ["Intro"]})
        snapshot = {
            "lease_id": "lease-1",
            "user_id": _USER_ID,
            "batch_ids": ["batch-a"],
            "paper_batches": {"p1": "batch-a"},
        }

        _load(
            monkeypatch,
            _state(["p1"], project_scope={"user_id": _USER_ID, "paper_ids": ["p1"], "publication": snapshot}),
            repo=_StubPaperRepo([_paper("p1")]),
            store=store,
        )

        assert store.bound_snapshots == [snapshot]

    def test_paper_list_marks_which_details_are_readable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """清单要标出每篇还能读到什么：只有摘要、未索引的论文否则会被模型当成没料可读。"""
        papers = [
            _paper("p1", analysis_status="analysis_ready", abstract="摘要一"),
            _paper("p2", abstract="摘要二"),
            _paper("p3"),
        ]

        delta = _load(
            monkeypatch,
            _state(["p1", "p2", "p3"]),
            repo=_StubPaperRepo(papers),
            store=_StubMilvusStore({"p1": [], "p2": [], "p3": []}),
        )

        lines = delta[_PAPER_LIST_KEY]
        assert "可读详情: 摘要+分析" in lines[0]
        assert "可读详情: 摘要" in lines[1]
        # 两样都没有时明写「无」，不留空字段冒充有值
        assert "可读详情: 无" in lines[2]

    def test_paper_without_sections_writes_an_honest_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """一篇没章节时该篇如实写空，其余论文的章节照常列出。"""
        delta = _load(
            monkeypatch,
            _state(["p1", "p2"]),
            repo=_StubPaperRepo([_paper("p1"), _paper("p2")]),
            store=_StubMilvusStore({"p1": [], "p2": ["Method", "Results"]}),
        )

        lines = delta[_SECTION_KEY]
        assert pipeline._SECTION_EMPTY_TEXT in lines[0]
        assert "Method" in lines[1] and "Results" in lines[1]

    def test_no_papers_renders_no_section_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """本轮没有授权论文时三段全空，不查库也不查图。"""
        repo = _StubPaperRepo([])
        store = _StubMilvusStore({})

        delta = _load(monkeypatch, _state([]), repo=repo, store=store)

        assert delta[_SECTION_KEY] == []
        assert delta[_PAPER_LIST_KEY] == []
        assert repo.calls == [] and store.queried == []


class TestLoadFailureNeverBecomesAnEmptyProject:
    """验基础设施故障落成显式失败，不吞成「本轮没有论文」。"""

    def test_paper_db_failure_raises_with_face_and_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """论文库读不出来时抛 ContextLoadError，带失败面与可读原因。

        返回空列表等于告诉模型「这个项目一篇论文都没有」，模型据此铺出的计划全是错的，
        用户看到的也不是「库连不上」而是「你的项目是空的」。
        """
        monkeypatch.setattr(pipeline, "_retry_sleep_seconds", lambda _cfg: 0.0, raising=False)

        with pytest.raises(pipeline.ContextLoadError) as caught:
            _load(
                monkeypatch,
                _state(["p1"]),
                repo=_StubPaperRepo(RuntimeError("connection refused")),
            )

        assert caught.value.face == pipeline._CONTEXT_FACE_PAPER_DB
        assert "connection refused" in caught.value.detail

    def test_section_query_failure_carries_the_section_face(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """章节查询塌了要报章节面，不能混成论文库面。"""
        with pytest.raises(pipeline.ContextLoadError) as caught:
            _load(
                monkeypatch,
                _state(["p1"]),
                repo=_StubPaperRepo([_paper("p1")]),
                store=_StubMilvusStore({"p1": TimeoutError("milvus timeout")}),
            )

        assert caught.value.face == pipeline._CONTEXT_FACE_SECTION

    def test_retry_happens_before_giving_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """可重试故障先按配置重试，重试仍不成立才失败。"""
        repo = _StubPaperRepo(RuntimeError("boom"))

        with pytest.raises(pipeline.ContextLoadError):
            _load(monkeypatch, _state(["p1"]), repo=repo)

        assert len(repo.calls) > 1


class TestEmptyDataIsNotFailure:
    """验数据本身为空是如实的空，不当失败。"""

    def test_graph_without_relations_renders_the_empty_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """图库里没有关系时该段写「无」，不记失败原因。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1")]),
            graph=_StubGraphService(_graph_result(0, 0)),
        )

        assert delta[_GRAPH_KEY] == [pipeline._GRAPH_STATUS_EMPTY_TEXT]
        assert "source_status_reasons" not in delta

    def test_graph_with_relations_reports_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """图里有数据时给出节点数与关系数，供规划判断值不值得查图。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1")]),
            graph=_StubGraphService(_graph_result(7, 12)),
        )

        assert "7" in delta[_GRAPH_KEY][0] and "12" in delta[_GRAPH_KEY][0]


class TestGraphFailureDegradesSoftly:
    """验图谱面单独降级：查不到只让该段写「无」，其余段照常。"""

    def test_paper_list_survives_a_graph_outage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """图库塌了不影响论文清单与章节段。"""
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1", is_vectorized=True)]),
            store=_StubMilvusStore({"p1": ["Method"]}),
            graph=_StubGraphService(RuntimeError("neo4j down")),
        )

        assert delta[_PAPER_LIST_KEY] and delta[_SECTION_KEY]
        assert delta[_GRAPH_KEY] == [pipeline._GRAPH_STATUS_EMPTY_TEXT]

    def test_graph_failure_reason_is_recorded_not_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """失败原因要落到可见状态位上。

        段里只写「无」时，图库空和图库连不上长得一模一样，而这两件事的排查方向完全相反。
        """
        delta = _load(
            monkeypatch,
            _state(["p1"]),
            repo=_StubPaperRepo([_paper("p1")]),
            graph=_StubGraphService(RuntimeError("neo4j down")),
        )

        assert delta["source_status"]["neo4j"] == "failed"
        assert "neo4j down" in delta["source_status_reasons"]["neo4j"]

    def test_existing_source_status_entries_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """写 neo4j 状态位时不能抹掉同回合别处写的状态。

        source_status 没有 reducer，节点返回的是整份替换，只写自己那一条会把别人的丢掉。
        """
        state = _state(["p1"], source_status={"planner": "analyzed"})

        delta = _load(
            monkeypatch,
            state,
            repo=_StubPaperRepo([_paper("p1")]),
            graph=_StubGraphService(RuntimeError("neo4j down")),
        )

        assert delta["source_status"]["planner"] == "analyzed"
        assert delta["source_status"]["neo4j"] == "failed"


class TestPaperRecordsAreReadOnce:
    """验焦点解析已读到的论文记录被复用，同一回合不重复查库。"""

    def test_cached_turn_papers_skip_the_repository(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """状态里带着本轮论文记录时装载器不再查库。"""
        repo = _StubPaperRepo([_paper("p1")])
        state = _state(["p1"], **{TURN_PAPERS_KEY: [_paper("p1", title="已读到的标题")]})

        delta = _load(monkeypatch, state, repo=repo)

        assert repo.calls == []
        assert "已读到的标题" in delta[_PAPER_LIST_KEY][0]

    def test_cache_missing_a_paper_falls_back_to_one_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """已读记录覆盖不全本轮授权集合时补一次查询，不拿半份清单凑。"""
        repo = _StubPaperRepo([_paper("p1"), _paper("p2")])
        state = _state(["p1", "p2"], **{TURN_PAPERS_KEY: [_paper("p1")]})

        delta = _load(monkeypatch, state, repo=repo)

        assert len(repo.calls) == 1
        assert len(delta[_PAPER_LIST_KEY]) == 2


class TestPaperListOrderIsStable:
    """验清单顺序走保序 ID 列表，不是 frozenset 哈希序。"""

    def test_list_follows_the_structured_query_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """清单里的「第二篇」要和后端序数规则里的「第二篇」指同一篇。

        走 frozenset 时遍历顺序是哈希序，提示词里的第二篇和焦点解析认的第二篇会错位，
        用户说「第二篇」时两边指到不同论文上。
        """
        ordered = ["p3", "p1", "p2"]
        papers = [_paper(pid, title=f"标题{pid}") for pid in ordered]

        delta = _load(
            monkeypatch,
            _state(ordered),
            repo=_StubPaperRepo(papers),
            store=_StubMilvusStore({pid: [] for pid in ordered}),
        )

        rendered = [line.split("|")[0].strip().lstrip("- ") for line in delta[_PAPER_LIST_KEY]]
        assert rendered == ordered


class TestGraphSectionTakesTheSameAuthorizedScope:
    """F01/C1：焦点解析收窄后，预装载图谱段拿到的就是授权上界。

    锁的是两段接起来能不能挡住：焦点解析放行一个带越界候选的历史，预装载的图谱查询
    必须只发授权范围内的论文，否则下游按 ID 直查会读出本轮授权之外的图谱关系。
    """

    def test_graph_query_uses_authorized_scope_not_widened_focus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph = _StubGraphService(_graph_result(7, 12))
        # 走真实的焦点解析：历史里混进一篇授权范围外的论文，看它会不会被带进焦点
        focus = resolve_focus(
            "对比第一篇的方法",
            intent="compare",
            project_paper_ids=["own_a"],
            conversation_history=[{"paper_ids": ["foreign_b"]}],
        )
        state = _state(["own_a"], focus=focus, chat_history=[{"paper_ids": ["foreign_b"]}])

        delta = _load(monkeypatch, state, graph=graph)

        assert graph.calls == [["own_a"]]
        assert "foreign_b" not in str(delta)


class _FlakyPaperRepo:
    """前 N 次调用抛基础设施故障、之后返回预置记录的仓储替身。

    只抛固定次数的错是为了验「瞬断一次后成功」这条路径：替身每次都抛时，
    分不出整轮是走了重试，还是第一次就被别处兜住了。
    """

    def __init__(
        self,
        papers: list[dict[str, Any]],
        *,
        fail_times: int,
        error: Exception | None = None,
    ) -> None:
        self._papers = papers
        self._fail_times = fail_times
        self._error = error or RuntimeError("connection refused")
        self.calls: list[tuple[list[str], str]] = []

    async def get_papers_by_ids(self, paper_ids: list[str], *, user_id: str) -> list[dict[str, Any]]:
        self.calls.append((list(paper_ids), user_id))
        if len(self.calls) <= self._fail_times:
            raise self._error
        by_id = {str(p.get("paper_id")): p for p in self._papers}
        return [by_id[pid] for pid in paper_ids if pid in by_id]


class _UnreachableLLM:
    """本类用例不该走到深度评估：被调用即说明读取故障没能在装载层拦住。"""

    def with_structured_output(self, schema: type) -> "_UnreachableLLM":
        raise AssertionError(f"首次读取失败时不该走到深度评估：{schema.__name__}")

    async def ainvoke(self, _messages: Any) -> Any:
        raise AssertionError("首次读取失败时不该走到模型调用")


class _AnalyzeSettings:
    """最小配置对象：analyze 已不读循环额度，只保留计划单元数上限。"""

    rag_research_planner_max_units: int = 8


def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """把上下文装载的重试退避压成 0：用例验的是重试次数，不是真等配置里的秒数。"""
    monkeypatch.setattr(get_settings(), "planner_context_retry_backoff_seconds", 0.0)


def _analyze(monkeypatch: pytest.MonkeyPatch, *, repo: Any, state: dict[str, Any]) -> dict[str, Any]:
    """把外部依赖换成替身后跑一次 analyze 节点，返回节点增量。"""
    monkeypatch.setattr(pipeline, "get_paper_repository", lambda: repo)
    monkeypatch.setattr(pipeline, "get_milvus_store", lambda: _StubMilvusStore({}))
    monkeypatch.setattr(pipeline, "get_research_object_query_service", lambda: _StubGraphService(_graph_result()))
    _no_backoff(monkeypatch)
    return asyncio.run(pipeline.analyze_planner_state(state, _UnreachableLLM(), _AnalyzeSettings()))


class TestFirstReadGoesThroughTheRetryGate:
    """F05：本轮第一次读论文库由装载层注入焦点解析，读取动作必须在既有重试入口内部。

    焦点解析曾经自己调仓储，一次瞬时连接故障就直接打断整轮，还被归因成「深度评估失败」，
    把排查方向带偏。四条用例分别盯：瞬断后能恢复、重试用尽报论文库面、最终失败码是上下文
    装载、没有授权论文时不惊动读取。
    """

    def test_transient_failure_on_the_first_read_still_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """首次读取失败一次后成功：整份读取成功返回，且仓储确实被调了两次。"""
        repo = _FlakyPaperRepo([_paper("p1")], fail_times=1)
        monkeypatch.setattr(pipeline, "get_paper_repository", lambda: repo)
        _no_backoff(monkeypatch)

        papers = asyncio.run(pipeline._load_turn_papers_with_retry(_state(["p1"]))(["p1"]))

        assert [paper["paper_id"] for paper in papers] == ["p1"]
        assert len(repo.calls) == 2

    def test_first_read_exhausts_retries_with_the_paper_db_face(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """重试仍失败时抛 ContextLoadError，失败面是论文库并带上可读原因。"""
        repo = _FlakyPaperRepo([], fail_times=99)
        monkeypatch.setattr(pipeline, "get_paper_repository", lambda: repo)
        _no_backoff(monkeypatch)

        with pytest.raises(pipeline.ContextLoadError) as caught:
            asyncio.run(pipeline._load_turn_papers_with_retry(_state(["p1"]))(["p1"]))

        assert caught.value.face == pipeline._CONTEXT_FACE_PAPER_DB
        assert "connection refused" in caught.value.detail

    def test_first_read_failure_is_attributed_to_context_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """走真实 analyze：首次读取塌了要落成上下文装载失败，不是深度评估失败。

        这条盯的是归因本身：落错分支时用户看到的失败码会把排查引向深度评估，
        而真正塌掉的是论文库连接。
        """
        repo = _StubPaperRepo(RuntimeError("connection refused"))

        delta = _analyze(monkeypatch, repo=repo, state=_state(["p1"]))

        assert delta["error_code"] == pipeline._CONTEXT_LOAD_FAILED_CODE
        assert delta["error_code"] != pipeline._DEPTH_ASSESSMENT_FAILED_CODE
        assert delta["source_status"]["context_load"] == "failed"
        assert delta["source_status"].get("depth_assessment") != "failed"
        assert pipeline._CONTEXT_FACE_PAPER_DB in delta["source_status_reasons"]["context_load"]
        assert "connection refused" in delta["source_status_reasons"]["context_load"]
        assert len(repo.calls) > 1

    def test_no_authorized_papers_skips_the_injected_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """本轮没有授权论文时不调用注入的读取：短路保留，省掉一次注定空结果的库查询。"""
        repo = _FlakyPaperRepo([], fail_times=99)
        monkeypatch.setattr(pipeline, "get_paper_repository", lambda: repo)
        state = _state([])

        delta = asyncio.run(
            resolve_turn_context(state, load_papers=pipeline._load_turn_papers_with_retry(state))
        )

        assert delta[TURN_PAPERS_KEY] == []
        assert repo.calls == []
