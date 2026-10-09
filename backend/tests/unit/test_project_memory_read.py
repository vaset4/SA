"""Planner 记忆读取动作（read_project_memory）的合规硬闸门测试。

作者: xxx
创建时间: 2026-08-30 10:20:00

被验的产品约束：

1. 四道资格门（当前 user_id + 当前 project_id + review_state=accepted +
   status=confirmed + stale_state=fresh）由读取服务的共享查询函数写死，Planner 给的
   收窄条件只能让结果更少。门里不含 maturity，这是有意的——三档成熟度都有资格进
   上下文，maturity 只管排序与预算截断次序。
2. 一跳关系扩展不是权限旁路：邻居只能从「不带收窄的过门池」里取交集，不许按
   memory_id 单独直查。谁跟合格记忆有关系谁就能进上下文，等于门形同虚设。
3. 四态不归并：读到 success、门与收窄后确实没有 empty、记忆存储不可用 failed。
   Planner 靠 empty 与 failed 的区别决定要不要重试、要不要提示用户。
4. 空身份抛 ValueError 而不是落 failed：仓储的过滤循环只在取值非 None 时才拼
   WHERE，空 user_id 会让归属门整段消失，读到该项目下所有人的记忆。这是调用侧接线
   错误，用 failed 承载会被当成环境问题容忍过去。
5. 投影与截断：条数上限进 excluded_counts；question_ids 只回填给收窄直接命中的种子，
   一跳邻居未必挂在同一研究问题下，给它填收窄用的 question_id 就是编造归属。
6. load_injectable_memory 的既有行为不被本单元改动波及。

本文件分两组：
- 结构组：用按真实 SQL 语义筛内存行的假仓储验上述约束，不连库，任何环境都真跑。
  假仓储照抄真仓储那句 `if value is not None` 才拼 WHERE 的过滤循环，所以少传一道门
  时它会像真库一样把不合格的行放回来，用例才有机会变红。
- 真库组（TestOneHopOnRealDb）：一跳扩展叠在真实 SQL WHERE 上才能证明「过门池」是
  真筛出来的，本机库不可连时整组跳过（见 research_memory_db_fixture.can_connect_db）。
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

import research_memory_db_fixture as db_fixture
from app.repositories import research_memory_repository as repo_mod
from app.services import research_memory_injector_service as injector
from app.services.internal_research import project_memory_reader as reader

# 真库组的跳过条件：库不可连只跳真库用例，结构组照常跑
requires_db = pytest.mark.skipif(
    not db_fixture.can_connect_db(), reason="业务库不可连，跳过记忆读取一跳扩展真库测试"
)

# 被验的枚举字面量集中在此，避免测试里散落裸字符串
_STATUS_CONFIRMED = "confirmed"
_STATUS_DRAFT = "draft"
_REVIEW_ACCEPTED = "accepted"
_REVIEW_PENDING = "pending_review"
_STALE_FRESH = "fresh"
_STALE_STALE = "stale"
_STALE_NEEDS_REBIND = "needs_rebind"
_MATURITY_ESTABLISHED = "established"
_MATURITY_LEANING = "leaning"
_MATURITY_INTUITION = "intuition"
_TYPE_DECISION = "decision_note"
_TYPE_GAP = "gap_note"
_LINK_CONFLICTS = "conflicts_with"
_LINK_BASED_ON = "based_on"

# 记忆读取的终态取值：三态不得归并，所以断言要点到具体那一态
_READ_SUCCESS = "success"
_READ_EMPTY = "empty"
_READ_FAILED = "failed"

# 记忆归属：本人、同项目里的另一个人、另一个项目
_OWNER_USER = "reader_owner"
_OTHER_USER = "reader_intruder"
_PROJECT_ID = "proj-reader"
_OTHER_PROJECT = "proj-someone-else"

# 各种「空身份」写法：缺值、空串、纯空白、非字符串。四者都不是合法的记忆归属人
_BLANK_USER_IDS: tuple[Any, ...] = (None, "", "   ", 12345)
_BLANK_USER_ID_IDS = ("none", "empty", "whitespace", "non_str")

# 收窄条件的键名
_KEY_FOLLOW_LINKS = "follow_links"
_KEY_QUESTION_ID = "question_id"
_KEY_MEMORY_TYPE = "memory_type"
_KEY_PAPER_ID = "paper_id"

# 条数上限默认拨到远大于用例造的记忆数，避免截断混进资格门断言
_MAX_ITEMS = 20

# 真库与假库共用的标题口径：断言里靠标题认条目
_TITLE_QUALIFIED = "四道门全满足的判断"
_TITLE_DISQUALIFIED = "单点不合格的判断"
_TITLE_SEED = "收窄直接命中的判断"
_TITLE_NEIGHBOR = "与种子有关系的邻居判断"

# 记忆行 id：种子与邻居
_SEED_ID = "mem-seed"
_NEIGHBOR_ID = "mem-neighbor"

# 收窄到种子（邻居是另一类记忆，只能靠关系被带进来）并要求扩一跳
_NARROW_TO_SEED = MappingProxyType({_KEY_MEMORY_TYPE: _TYPE_DECISION, _KEY_FOLLOW_LINKS: True})

# 期望到达仓储的过门查询：四道门加项目归属，一个不多一个不少。
# 多出 paper_id=None 之类的键会破坏既有断言，少一项就是那道门在 SQL 里根本不存在
_EXPECTED_GATE_QUERY = MappingProxyType(
    {
        "project_id": _PROJECT_ID,
        "user_id": _OWNER_USER,
        "review_state": _REVIEW_ACCEPTED,
        "status": _STATUS_CONFIRMED,
        "stale_state": _STALE_FRESH,
    }
)

# 允许出现在一次记忆读取里的仓储方法。按 memory_id 直查条目的方法不在其中：
# 那条查询不带门，是一跳扩展最容易开出来的权限旁路
_ALLOWED_REPO_CALLS = frozenset({"ensure_schema", "list_items", "list_links_of_memory"})

# 真仓储 list_items 里走 `if value is not None` 过滤循环的等值列
_EQUALITY_FILTER_COLUMNS = (
    "user_id",
    "review_state",
    "memory_type",
    "status",
    "stale_state",
    "maturity",
)

# 单点不合格的轴：每条只打破一道门，用来验单道门的隔离效果
_DISQUALIFYING_AXES: tuple[dict[str, Any], ...] = (
    {"status": _STATUS_DRAFT},
    {"review_state": _REVIEW_PENDING},
    {"stale_state": _STALE_STALE},
    {"stale_state": _STALE_NEEDS_REBIND},
    {"user_id": _OTHER_USER},
    {"project_id": _OTHER_PROJECT},
)
_DISQUALIFYING_IDS = (
    "status_draft",
    "review_pending",
    "stale",
    "needs_rebind",
    "other_user",
    "other_project",
)

# Planner 试图放宽资格门的各种写法。这些键都不在 MemoryReadNarrowing 契约里，
# 用例故意越过契约传，模拟被改坏的上游或手写的收窄字典
_WIDENING_ATTEMPTS: tuple[dict[str, Any], ...] = (
    {"review_state": _REVIEW_PENDING},
    {"status": _STATUS_DRAFT},
    {"stale_state": _STALE_STALE},
    {"stale_state": _STALE_NEEDS_REBIND},
    {"maturity": _MATURITY_INTUITION},
    {"user_id": _OTHER_USER},
    {"project_id": _OTHER_PROJECT},
)
_WIDENING_IDS = (
    "review_state",
    "status",
    "stale_state",
    "needs_rebind",
    "maturity",
    "user_id",
    "project_id",
)

# 投影后 MemoryReadItem 的字段集。question_ids 不在其中：注入路径不查归属问题，
# 缺键是正常状态，只有按问题收窄的读取才补上。
# 四道门的四个轴在其中：门写在 SQL WHERE 里，被挡掉的行不回应用层，所以进了上下文的
# 条目要自带「凭什么放行」的取值，供合规核查与评测逐条复核
_EXPECTED_ITEM_KEYS = frozenset(
    {
        "memory_id",
        "version",
        "title",
        "body_summary",
        "memory_type",
        "maturity",
        "source_kind",
        "provenance_label",
        "user_id",
        "review_state",
        "status",
        "stale_state",
    }
)


class _FakeSettings:
    """只提供读取路径会碰到的条数上限，字符预算属渲染阶段、本文件不测渲染。"""

    def __init__(self, max_items: int = _MAX_ITEMS) -> None:
        self.research_memory_inject_max_items = max_items


class _FilteringRepo:
    """假仓储：照真实 SQL 的过滤语义筛内存行，并记下每一次查询。

    只记参数的替身证不了「门真的筛掉了谁」，所以这里抄了真仓储那句
    `if value is not None` 才拼 WHERE 的过滤循环：产品代码少传一道门时，这个替身会
    像真库一样把不合格的行放回来，用例才会变红而不是继续绿着。
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        links: tuple[dict[str, Any], ...] = (),
        question_links: dict[str, set[str]] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.links = list(links)
        self.question_links = dict(question_links or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def ensure_schema(self) -> None:
        self.calls.append(("ensure_schema", {}))

    async def list_items(self, project_id, /, **filters):
        # project_id 声明为位置形参并加 `/`：产品代码若改成按关键字传，这里直接 TypeError
        self.calls.append(("list_items", {"project_id": project_id, **filters}))
        matched = [row for row in self.rows if self._passes(row, project_id, filters)]
        ordered = sorted(matched, key=lambda row: row["updated_at"], reverse=True)
        return [dict(row) for row in ordered]

    async def list_links_of_memory(self, memory_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_links_of_memory", {"memory_id": memory_id}))
        endpoints = ("src_memory_id", "dst_memory_id")
        return [dict(link) for link in self.links if memory_id in [link[key] for key in endpoints]]

    async def get_item(self, memory_id: str) -> dict[str, Any] | None:
        # 按 memory_id 直查不带任何门，一旦一跳扩展走上这条路，邻居就绕过了资格判定
        self.calls.append(("get_item", {"memory_id": memory_id}))
        raise AssertionError("一跳邻居不许按 memory_id 直查：这条查询不过四道门")

    async def get_detail(self, memory_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_detail", {"memory_id": memory_id}))
        raise AssertionError("一跳邻居不许按 memory_id 直查详情：这条查询不过四道门")

    def _passes(self, row: dict[str, Any], project_id: Any, filters: dict[str, Any]) -> bool:
        """判断一行是否被本次查询的过滤条件放行，语义对齐真仓储的 WHERE 拼装。"""
        # 1. project_id 走基础 WHERE 恒定生效，传 None 等于 `= NULL`，匹配不到任何行
        if row["project_id"] != project_id:
            return False
        # 2. 等值过滤只在取值非 None 时才拼进 WHERE：这正是空身份让归属门消失的机理
        for column in _EQUALITY_FILTER_COLUMNS:
            wanted = filters.get(column)
            if wanted is not None and row.get(column) != wanted:
                return False
        # 3. paper_id 是 JSONB 数组包含过滤
        paper_id = filters.get(_KEY_PAPER_ID)
        if paper_id is not None and paper_id not in row.get("related_paper_ids", ()):
            return False
        # 4. question_id 走 memory_question_links 关联，归属表在条目行之外
        question_id = filters.get(_KEY_QUESTION_ID)
        if question_id is not None and row["memory_id"] not in self.question_links.get(question_id, set()):
            return False
        return True


class _BrokenRepo(_FilteringRepo):
    """假仓储：在指定入口抛错，用来验记忆存储不可用时落 failed 而不是空列表冒充成功。"""

    def __init__(self, fail_on: str, rows: list[dict[str, Any]], links: tuple[dict[str, Any], ...] = ()) -> None:
        super().__init__(rows, links)
        self.fail_on = fail_on

    async def ensure_schema(self) -> None:
        self._maybe_break("ensure_schema")
        await super().ensure_schema()

    async def list_items(self, project_id, /, **filters):
        self._maybe_break("list_items")
        return await super().list_items(project_id, **filters)

    async def list_links_of_memory(self, memory_id: str) -> list[dict[str, Any]]:
        self._maybe_break("list_links_of_memory")
        return await super().list_links_of_memory(memory_id)

    def _maybe_break(self, entry: str) -> None:
        if entry == self.fail_on:
            raise RuntimeError(f"记忆存储不可用：{entry}")


def _row(memory_id: str, title: str, **overrides: Any) -> dict[str, Any]:
    """
    造一条仓储风格的记忆行，默认四道门全满足。

    参数:
        memory_id: 记忆 id，断言里靠它认条目
        title: 记忆标题
        overrides: 要打破的轴取值，如 status=draft，或换 project_id / user_id 造越权样本
    返回:
        投影所需字段齐全的条目行；缺字段应当直接 KeyError 暴露而不是静默补默认值
    """
    return {
        "memory_id": memory_id,
        "project_id": _PROJECT_ID,
        "user_id": _OWNER_USER,
        "version": 1,
        "title": title,
        "body": f"{title}的正文",
        "memory_type": _TYPE_DECISION,
        "maturity": _MATURITY_LEANING,
        "source_kind": "manual",
        "status": _STATUS_CONFIRMED,
        "review_state": _REVIEW_ACCEPTED,
        "stale_state": _STALE_FRESH,
        "related_paper_ids": [],
        "updated_at": "2026-08-01T00:00:00Z",
        **overrides,
    }


def _link(src_memory_id: str, dst_memory_id: str, link_type: str = _LINK_CONFLICTS) -> dict[str, Any]:
    """造一条记忆间关系行；conflicts_with 单行存双向，所以两端都要能被查到。"""
    return {
        "link_id": f"lnk-{src_memory_id}-{dst_memory_id}",
        "project_id": _PROJECT_ID,
        "src_memory_id": src_memory_id,
        "dst_memory_id": dst_memory_id,
        "link_type": link_type,
    }


def _mixed_rows() -> list[dict[str, Any]]:
    """一条全合格记忆 + 每道门各一条单点不合格记忆，用来验只有全合格那条出得来。"""
    rows = [_row("mem-ok", _TITLE_QUALIFIED)]
    rows += [
        _row(f"mem-bad-{index}", _TITLE_DISQUALIFIED, **axis)
        for index, axis in enumerate(_DISQUALIFYING_AXES)
    ]
    return rows


def _read(
    repo: Any,
    narrowing: Any = None,
    *,
    user_id: Any = _OWNER_USER,
    project_id: Any = _PROJECT_ID,
    max_items: int = _MAX_ITEMS,
) -> dict[str, Any]:
    """
    用替身仓储跑一次 Planner 记忆读取。

    过门查询在 injector 模块里、条数上限与关系查询在 reader 模块里，两处各自持有
    模块级的仓储与配置获取函数，所以两边都要替换，漏一个就会打到真库或真配置。

    参数:
        repo: 替身仓储
        narrowing: Planner 给的收窄条件，None 表示不收窄
        user_id/project_id: 项目范围，传空值用于验入参校验
        max_items: 条数上限，验截断时才拨小
    返回:
        MemoryReadResult
    """
    settings = _FakeSettings(max_items)
    context = {"project_id": project_id, "user_id": user_id}
    with patch.object(injector, "get_research_memory_repository", return_value=repo), \
         patch.object(reader, "get_research_memory_repository", return_value=repo), \
         patch.object(injector, "get_settings", return_value=settings), \
         patch.object(reader, "get_settings", return_value=settings):
        return asyncio.run(
            reader.read_project_memory(
                context=context,
                query_narrowing=None if narrowing is None else dict(narrowing),
            )
        )


def _load(repo: Any) -> dict[str, Any]:
    """跑一次入口注入加载（load_injectable_memory），用于验既有行为未被本单元改动波及。"""
    with patch.object(injector, "get_research_memory_repository", return_value=repo), \
         patch.object(injector, "get_settings", return_value=_FakeSettings()):
        return asyncio.run(injector.load_injectable_memory(_PROJECT_ID, _OWNER_USER))


def _item_queries(repo: _FilteringRepo) -> list[dict[str, Any]]:
    """取出本次读取里所有条目查询实际到达仓储的过滤条件。"""
    return [payload for name, payload in repo.calls if name == "list_items"]


def _call_names(repo: _FilteringRepo) -> list[str]:
    """取出本次读取碰过的仓储方法名，按调用先后排列。"""
    return [name for name, _ in repo.calls]


def _ids(result: dict[str, Any]) -> list[str]:
    """取读取结果里各条记忆的 id，次序即注入次序。"""
    return [item["memory_id"] for item in result["items"]]


def _titles(result: dict[str, Any]) -> list[str]:
    """取读取结果里各条记忆的标题，用于核对到底哪些条目通过了门。"""
    return [item["title"] for item in result["items"]]


# ─── 一、四道资格门不可放宽 ───


class TestGatesCannotBeWidenedByNarrowing:
    """验 Planner 的收窄条件只能让结果更少，任何放宽写法都到不了 SQL。"""

    def test_no_narrowing_sends_exactly_the_four_gates(self) -> None:
        """
        不收窄时到达仓储的过滤条件恰好是四道门加项目归属，不多不少。

        少一项等于那道门在 SQL 里根本不存在；多出 paper_id=None 之类的空键会让
        「门就是这几项」的断言失效，日后有人往资格判定里塞新维度也看不出来。
        """
        repo = _FilteringRepo([_row("mem-ok", _TITLE_QUALIFIED)])
        _read(repo)

        assert _item_queries(repo) == [dict(_EXPECTED_GATE_QUERY)]

    def test_fully_qualified_memory_reaches_planner(self) -> None:
        """
        全合格记忆读得出来且 items 非空：这是所有「不合格进不来」用例的正对照。

        少了它，读取函数哪天永远返回 empty，下面每一条隔离用例都会绿着通过。
        """
        result = _read(_FilteringRepo(_mixed_rows()))

        assert result["status"] == _READ_SUCCESS
        assert _titles(result) == [_TITLE_QUALIFIED]
        assert result["items"]

    @pytest.mark.parametrize("widening", _WIDENING_ATTEMPTS, ids=_WIDENING_IDS)
    def test_planner_cannot_widen_any_gate(self, widening: dict[str, Any]) -> None:
        """
        Planner 把门的取值塞进收窄条件时一个都不生效，门取值仍是原值。

        防的是「Planner 自己决定读草稿 / 读失效判断 / 读别人的记忆」：只要放宽值能拼进
        SQL，待审草稿和他人判断就会喂回 LLM，且现场看起来完全正常。
        """
        repo = _FilteringRepo(_mixed_rows())
        result = _read(repo, widening)

        # 1. 放宽值到不了仓储：查询条件逐字仍是四道门
        assert _item_queries(repo) == [dict(_EXPECTED_GATE_QUERY)]
        # 2. 结果里也只有全合格那条，放宽请求没有换来任何额外条目
        assert _titles(result) == [_TITLE_QUALIFIED]

    @pytest.mark.parametrize("axis", _DISQUALIFYING_AXES, ids=_DISQUALIFYING_IDS)
    def test_single_axis_disqualification_blocks_the_memory(self, axis: dict[str, Any]) -> None:
        """
        任一道门单独不满足即出局，不存在「降权混入」。

        每条用例只打破一道门，所以哪道门被删掉都会有专属用例变红；合并成一条会导致
        漏掉某道门时它仍然是红的，反而证明不了单点隔离。
        """
        repo = _FilteringRepo([_row("mem-bad", _TITLE_DISQUALIFIED, **axis)])
        result = _read(repo)

        assert result["status"] == _READ_EMPTY
        assert result["items"] == []

    @pytest.mark.parametrize("stale_state", [_STALE_STALE, _STALE_NEEDS_REBIND])
    def test_stale_memory_stays_out_even_when_narrowing_asks_for_it(self, stale_state: str) -> None:
        """
        失效记忆在任何情况下都不进结果，包括 Planner 明确点名要它的时候。

        证据已过期或待重绑的判断混进上下文，模型会拿早已被推翻的结论继续推理。
        """
        repo = _FilteringRepo([_row("mem-stale", _TITLE_DISQUALIFIED, stale_state=stale_state)])
        result = _read(repo, {"stale_state": stale_state, _KEY_FOLLOW_LINKS: True})

        assert result["status"] == _READ_EMPTY
        assert result["items"] == []

    @pytest.mark.parametrize(
        "narrowing",
        [
            {_KEY_MEMORY_TYPE: _TYPE_GAP},
            {_KEY_PAPER_ID: "paper-9"},
            {_KEY_QUESTION_ID: "rq-9"},
        ],
        ids=["memory_type", "paper_id", "question_id"],
    )
    def test_whitelisted_narrowing_really_narrows(self, narrowing: dict[str, Any]) -> None:
        """
        白名单里的三个维度确实拼进查询并起到收窄作用。

        这条是上面各条「放宽不生效」的反面锚点：若收窄参数被整体丢弃，那些用例会因为
        「什么都没传进去」而假绿，收窄能力也就悄悄没了。
        """
        repo = _FilteringRepo([_row("mem-ok", _TITLE_QUALIFIED)])
        result = _read(repo, narrowing)

        assert _item_queries(repo) == [{**_EXPECTED_GATE_QUERY, **narrowing}]
        assert result["status"] == _READ_EMPTY


# ─── 二、一跳关系扩展不是权限旁路 ───

# 一跳扩展的读取形状：种子查询 + 逐条种子取关系 + 不带收窄的过门池查询。
# 中间不许多出任何按 memory_id 直查条目的调用
_ONE_HOP_CALL_SEQUENCE = [
    "ensure_schema",
    "list_items",
    "list_links_of_memory",
    "ensure_schema",
    "list_items",
]


def _seed_and_neighbor(**neighbor_overrides: Any) -> _FilteringRepo:
    """
    造「一条种子 + 一条只能靠关系被带进来的邻居」这个局面。

    邻居故意是另一类记忆，所以收窄查询捞不到它，它进结果的唯一途径就是一跳扩展；
    这样邻居有没有过门就成了唯一变量。

    参数:
        neighbor_overrides: 邻居要打破的门，不传则邻居全合格（正对照）
    返回:
        装好种子、邻居与一条关系行的替身仓储
    """
    seed = _row(_SEED_ID, _TITLE_SEED)
    neighbor = _row(_NEIGHBOR_ID, _TITLE_NEIGHBOR, memory_type=_TYPE_GAP, **neighbor_overrides)
    return _FilteringRepo([seed, neighbor], (_link(_SEED_ID, _NEIGHBOR_ID),))


class TestOneHopIsNotAPermissionBypass:
    """验一跳扩展进来的邻居必须自己也过四道门，「与合格记忆有关系」不是豁免理由。"""

    def test_qualified_neighbor_enters_through_link(self) -> None:
        """
        全合格邻居确实能被关系带进来，且带它进来的那次查询也带着四道门。

        这是下面各条隔离用例的正对照：扩展要是根本没跑起来，那些用例会全部假绿。
        """
        repo = _seed_and_neighbor()
        result = _read(repo, _NARROW_TO_SEED)

        assert _ids(result) == [_SEED_ID, _NEIGHBOR_ID]
        # 种子查询叠了收窄，取邻居的池查询只带门、不带收窄
        assert _item_queries(repo) == [
            {**_EXPECTED_GATE_QUERY, _KEY_MEMORY_TYPE: _TYPE_DECISION},
            dict(_EXPECTED_GATE_QUERY),
        ]

    @pytest.mark.parametrize("axis", _DISQUALIFYING_AXES, ids=_DISQUALIFYING_IDS)
    def test_neighbor_failing_one_gate_stays_out(self, axis: dict[str, Any]) -> None:
        """
        邻居单点不合格就进不来：待审草稿、未确认、已失效、他人的、别项目的都挡在外面。

        这是本单元最容易开出的洞——顺着关系多看一跳时，只要邻居不重新过门，
        同项目里别人的判断和没人审过的草稿就会跟着种子一起进 LLM 上下文。
        每道门一条用例，谁被删掉都有专属用例变红。
        """
        repo = _seed_and_neighbor(**axis)
        result = _read(repo, _NARROW_TO_SEED)

        assert _ids(result) == [_SEED_ID]
        assert _TITLE_NEIGHBOR not in _titles(result)

    def test_link_hits_on_either_endpoint(self) -> None:
        """
        关系两端都能命中：conflicts_with 单行存双向，只认 src 会漏掉一半邻居。

        漏掉的那半是「与当前判断冲突的旧判断」，恰恰是最该被看见的一类记忆。
        """
        seed = _row(_SEED_ID, _TITLE_SEED)
        as_source = _row("mem-dst", "种子指向的邻居", memory_type=_TYPE_GAP)
        as_target = _row("mem-src", "指向种子的邻居", memory_type=_TYPE_GAP)
        links = (
            _link(_SEED_ID, "mem-dst", _LINK_BASED_ON),
            _link("mem-src", _SEED_ID, _LINK_CONFLICTS),
        )
        repo = _FilteringRepo([seed, as_source, as_target], links)

        result = _read(repo, _NARROW_TO_SEED)

        assert set(_ids(result)) == {_SEED_ID, "mem-dst", "mem-src"}

    def test_neighbors_are_never_fetched_by_id(self) -> None:
        """
        取邻居只走「不带收窄的过门池」，全程没有按 memory_id 直查条目的调用。

        按 id 直查的那条 SQL 不带任何门，等于谁跟合格记忆有关系谁就能进上下文。
        替身仓储的 get_item / get_detail 一被碰就失败，调用序列也逐个核对。
        """
        repo = _seed_and_neighbor()
        result = _read(repo, _NARROW_TO_SEED)

        assert result["status"] == _READ_SUCCESS
        assert _call_names(repo) == _ONE_HOP_CALL_SEQUENCE
        assert set(_call_names(repo)) <= _ALLOWED_REPO_CALLS

    def test_follow_links_without_narrowing_makes_no_second_item_query(self) -> None:
        """
        不收窄时 follow_links 不该再查一次条目：种子已是全部过门条目，邻居必在其中。

        多跑一次全量过门查询不改变结果，只是白花一次数据库往返，还让读取路径多一处
        可能失败的地方。
        """
        rows = [_row(_SEED_ID, _TITLE_SEED), _row(_NEIGHBOR_ID, _TITLE_NEIGHBOR)]
        repo = _FilteringRepo(rows, (_link(_SEED_ID, _NEIGHBOR_ID),))

        result = _read(repo, {_KEY_FOLLOW_LINKS: True})

        assert len(_item_queries(repo)) == 1
        assert set(_ids(result)) == {_SEED_ID, _NEIGHBOR_ID}


# ─── 三、三态合同：empty 与 failed 绝不归并 ───


class TestReadStatusContract:
    """验读取终态点到具体那一态，调用方据此决定重试还是继续推进。"""

    def test_no_match_is_empty_not_failed(self) -> None:
        """
        门与收窄之后没命中是 empty：项目里没有相关判断，不是记忆存储坏了。

        两态归并会让 Planner 在「本项目还没攒下这方面判断」时反复重试，或者反过来在
        存储真出问题时当作「没有记忆」继续推进，把读不到当成事实上的空。
        """
        repo = _FilteringRepo(_mixed_rows())
        result = _read(repo, {_KEY_MEMORY_TYPE: _TYPE_GAP})

        assert result["status"] == _READ_EMPTY
        assert result["status"] != _READ_FAILED
        assert result["items"] == []
        assert "error_kind" not in result
        assert result["excluded_counts"] == {}

    @pytest.mark.parametrize(
        "fail_on",
        ["ensure_schema", "list_items", "list_links_of_memory"],
        ids=["ensure_schema", "list_items", "list_links_of_memory"],
    )
    def test_storage_failure_reports_failed_with_error_kind(self, fail_on: str) -> None:
        """
        三个入库口任一处炸掉都落 failed 并带上 error_kind，不拿空列表冒充成功。

        关系查询这一口最容易漏：它在扩展函数里、比种子查询晚发生，早期实现常把它的
        异常吞成「没有邻居」，于是存储半瘫时 Planner 看到的是一份静悄悄少了内容的记忆。
        """
        rows = [_row(_SEED_ID, _TITLE_SEED), _row(_NEIGHBOR_ID, _TITLE_NEIGHBOR, memory_type=_TYPE_GAP)]
        repo = _BrokenRepo(fail_on, rows, (_link(_SEED_ID, _NEIGHBOR_ID),))

        result = _read(repo, _NARROW_TO_SEED)

        assert result["status"] == _READ_FAILED
        assert result["error_kind"] == "RuntimeError"
        assert result["items"] == []
        assert result["excluded_counts"] == {}

    def test_success_never_carries_an_empty_item_list(self) -> None:
        """success 必须真有条目：空列表配 success 会让调用方以为读到了东西。"""
        result = _read(_FilteringRepo([_row("mem-ok", _TITLE_QUALIFIED)]))

        assert result["status"] == _READ_SUCCESS
        assert len(result["items"]) == 1


# ─── 四、空身份是接线错误，抛 ValueError 而不是落 failed ───


class TestBlankOwnerIsWiringError:
    """验空 user_id 在碰库之前就被拒，且不被包成看着像环境故障的 failed。"""

    @pytest.mark.parametrize("blank_user_id", _BLANK_USER_IDS, ids=_BLANK_USER_ID_IDS)
    def test_blank_owner_raises_before_touching_storage(self, blank_user_id: Any) -> None:
        """
        四种空身份写法一律 ValueError，且仓储零调用。

        仓储的过滤循环遇到 None 会跳过拼 WHERE，空身份传下去等于归属门整段消失，
        读到该项目下所有人的记忆。校验必须在 try 之外：一旦被 except Exception 收成
        failed，接错线就显示成「记忆存储不可用」，当成环境问题被容忍过去。
        """
        repo = _FilteringRepo(_mixed_rows())

        with pytest.raises(ValueError):
            _read(repo, user_id=blank_user_id)

        assert repo.calls == []

    def test_missing_project_id_fails_closed(self) -> None:
        """
        项目范围缺 project_id 时读不出任何东西，绝不退化成跨项目全捞。

        project_id 走基础 WHERE 而不是那个「非 None 才拼」的过滤循环，所以空值方向是
        关闭；这条把该方向钉住，防止日后有人把项目归属挪进条件过滤循环里。
        """
        repo = _FilteringRepo(_mixed_rows())
        result = _read(repo, project_id=None)

        assert result["status"] == _READ_EMPTY
        assert result["items"] == []


# ─── 五、投影、排序与截断 ───

# 四条全合格记忆，仓储按 updated_at 倒序给，用来验成熟度排序与同档内的稳定次序
_ORDERED_ROWS_EXPECTATION = ["mem-established", "mem-leaning-new", "mem-leaning-old", "mem-intuition"]


def _rows_across_maturities() -> list[dict[str, Any]]:
    """造跨三档成熟度的合格记忆，其中 leaning 两条用于验同档内保持最近更新在前。"""
    return [
        _row("mem-established", "已确立判断", maturity=_MATURITY_ESTABLISHED, updated_at="2026-08-01T00:00:00Z"),
        _row("mem-leaning-new", "较新的倾向判断", maturity=_MATURITY_LEANING, updated_at="2026-08-05T00:00:00Z"),
        _row("mem-leaning-old", "较旧的倾向判断", maturity=_MATURITY_LEANING, updated_at="2026-08-02T00:00:00Z"),
        _row("mem-intuition", "直觉判断", maturity=_MATURITY_INTUITION, updated_at="2026-08-09T00:00:00Z"),
    ]


class TestProjectionOrderAndTruncation:
    """验进上下文的条目次序与截断口径，以及归属问题只在确凿时回填。"""

    def test_order_is_maturity_first_then_recency(self) -> None:
        """
        次序是 established → leaning → intuition，同一档内保持最近更新在前。

        次序决定预算不够时先掉谁：先掉直觉级而不是先掉刚写下的判断。排序若不稳定，
        同一批记忆每次读出来的顺序都不同，模型的回答会跟着无缘无故漂。
        """
        result = _read(_FilteringRepo(_rows_across_maturities()))

        assert _ids(result) == _ORDERED_ROWS_EXPECTATION

    def test_over_max_items_is_counted_not_silently_dropped(self) -> None:
        """
        超出条数上限的条目进 excluded_counts，被截掉的是次序最后的直觉级判断。

        默默丢掉是最坏的：调用方以为拿到了项目里全部相关判断，实际只看到一部分，
        也没有任何迹象表明发生了截断。
        """
        result = _read(_FilteringRepo(_rows_across_maturities()), max_items=2)

        assert _ids(result) == _ORDERED_ROWS_EXPECTATION[:2]
        assert result["excluded_counts"] == {"over_max_items": 2}

    def test_question_ids_backfilled_only_for_seeds(self) -> None:
        """
        归属问题只回填给按问题收窄直接命中的种子，一跳邻居不填。

        邻居只是与种子有关系，未必挂在同一研究问题下；给它盖上这个 question_id 就是
        编造归属，Planner 会照着假归属组织回答，用户看到的是「这条判断属于那个问题」。
        """
        rows = [_row(_SEED_ID, _TITLE_SEED), _row(_NEIGHBOR_ID, _TITLE_NEIGHBOR, memory_type=_TYPE_GAP)]
        repo = _FilteringRepo(rows, (_link(_SEED_ID, _NEIGHBOR_ID),), {"rq-1": {_SEED_ID}})

        result = _read(repo, {_KEY_QUESTION_ID: "rq-1", _KEY_FOLLOW_LINKS: True})

        items = {item["memory_id"]: item for item in result["items"]}
        assert items[_SEED_ID]["question_ids"] == ["rq-1"]
        assert "question_ids" not in items[_NEIGHBOR_ID]


# ─── 六、既有入口注入行为未被本单元波及 ───


class TestExistingInjectionUnchanged:
    """验共享查询函数被抽出来复用之后，入口注入的查询与返回结构逐字不变。"""

    def test_load_injectable_memory_query_and_shape_unchanged(self) -> None:
        """
        入口注入仍然只发一次四道门查询，返回结构与字段集不变。

        本单元性质是「加一个读取动作」，不是改注入：入口注入多传一个键、少一个字段，
        都会顺着 prompt 拼装一路歪下去，而那条链路没有别的断言看着。
        """
        repo = _FilteringRepo(_mixed_rows())
        result = _load(repo)

        assert _item_queries(repo) == [dict(_EXPECTED_GATE_QUERY)]
        assert result["status"] == _READ_SUCCESS
        assert _titles(result) == [_TITLE_QUALIFIED]
        assert set(result["items"][0]) == _EXPECTED_ITEM_KEYS
        assert result["excluded_counts"] == {}


# ─── 七、真库组：一跳扩展叠在真实 SQL WHERE 上（库不可连时整组跳过） ───

# 真库里造记忆用的全合格轴取值，用例只打破其中一个轴来制造单点不合格的邻居
_REAL_DB_AXES = MappingProxyType(
    {
        "memory_type": _TYPE_DECISION,
        "user_id": _OWNER_USER,
        "status": _STATUS_CONFIRMED,
        "review_state": _REVIEW_ACCEPTED,
        "stale_state": _STALE_FRESH,
        "maturity": _MATURITY_LEANING,
    }
)

# 邻居在真库里要被挡住的门；跨项目关系不在此列，schema 不约束关系两端同项目，
# 那条路径由结构组覆盖
_REAL_DB_NEIGHBOR_AXES: tuple[dict[str, Any], ...] = (
    {"status": _STATUS_DRAFT},
    {"review_state": _REVIEW_PENDING},
    {"stale_state": _STALE_STALE},
    {"user_id": _OTHER_USER},
)
_REAL_DB_NEIGHBOR_IDS = ("status_draft", "review_pending", "stale", "other_user")


async def _seed_on_db(project_id: str, title: str, **axis_overrides: Any) -> str:
    """
    在真库里造一条记忆，默认四道门全满足。

    直连仓储而不走 service：service 会按来源改写 status，而这里要精确控制每个轴。

    参数:
        project_id: 记忆落在哪个项目
        title: 记忆标题，断言里靠它认条目
        axis_overrides: 要打破的轴取值，如 status=draft
    返回:
        新建记忆的 memory_id
    """
    detail = await repo_mod.get_research_memory_repository().create_item(
        project_id, {**_REAL_DB_AXES, "title": title, **axis_overrides}
    )
    return detail["item"]["memory_id"]


async def _read_on_db(project_id: str, narrowing: Any) -> dict[str, Any]:
    """在真库上跑一次 Planner 记忆读取，只替换配置、仓储用真的。"""
    settings = _FakeSettings()
    with patch.object(injector, "get_settings", return_value=settings), \
         patch.object(reader, "get_settings", return_value=settings):
        return await reader.read_project_memory(
            context={"project_id": project_id, "user_id": _OWNER_USER},
            query_narrowing=dict(narrowing),
        )


@requires_db
class TestOneHopOnRealDb:
    """在真实 Postgres 上验一跳扩展的邻居是被 SQL WHERE 真筛过的，而非只是参数递到位。"""

    def test_qualified_neighbor_enters_through_link(self) -> None:
        """
        真库里全合格的邻居能顺着关系进上下文，收窄查询本身捞不到它。

        没有这条正对照，下面的隔离用例可能只是因为扩展在真库上根本没跑起来才通过。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                seed_id = await _seed_on_db(project_id, _TITLE_SEED)
                neighbor_id = await _seed_on_db(project_id, _TITLE_NEIGHBOR, memory_type=_TYPE_GAP)
                await repo_mod.get_research_memory_repository().add_link(
                    project_id, seed_id, neighbor_id, _LINK_CONFLICTS
                )
                return await _read_on_db(project_id, _NARROW_TO_SEED)

        result = asyncio.run(scenario())
        assert result["status"] == _READ_SUCCESS
        assert sorted(_titles(result)) == sorted([_TITLE_SEED, _TITLE_NEIGHBOR])

    @pytest.mark.parametrize("axis", _REAL_DB_NEIGHBOR_AXES, ids=_REAL_DB_NEIGHBOR_IDS)
    def test_neighbor_failing_one_gate_stays_out(self, axis: dict[str, Any]) -> None:
        """
        真库里单点不合格的邻居进不来：草稿、待审、已失效、他人的判断都被 WHERE 挡住。

        这是假仓储证不了的那一半——过门池到底是不是真的按 SQL 筛出来的。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                seed_id = await _seed_on_db(project_id, _TITLE_SEED)
                neighbor_id = await _seed_on_db(
                    project_id, _TITLE_NEIGHBOR, memory_type=_TYPE_GAP, **axis
                )
                await repo_mod.get_research_memory_repository().add_link(
                    project_id, seed_id, neighbor_id, _LINK_CONFLICTS
                )
                return await _read_on_db(project_id, _NARROW_TO_SEED)

        result = asyncio.run(scenario())
        assert _titles(result) == [_TITLE_SEED]
