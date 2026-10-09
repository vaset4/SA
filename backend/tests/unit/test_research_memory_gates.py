"""
研究记忆四道资格门的合规硬闸门测试。

作者: xxx
创建时间: 2026-08-29 18:40:00

被验的产品约束：能进 LLM 上下文的记忆必须同时满足当前用户 + 当前项目 +
review_state=accepted + status=confirmed + stale_state=fresh。门写在仓储/服务层，
调用方只能收窄不能放宽，所以「注入合规率 100%」是结构保证而不是提示词约定：
不合格的记忆对 Planner 是「不存在」，不是「低分」。门里不含 maturity，这是有意的
（决策 A）——三档成熟度都有资格，maturity 只决定身份标签与预算截断次序。

本文件分两组：
- 结构组（TestGateStructureCannotBeWidened）：用假 repo 与签名反射验「调用方在结构上
  无法放宽」，不连库，任何环境都真跑。
- 真库组（TestGateComplianceOnRealDb）：四道门本质是一句 SQL WHERE，假 repo 只能证
  「参数传进去了」，证不了「真的筛对了」。五轴组合的合规断言只能在真实 Postgres 上验，
  本机库不可连时整个 class 跳过（见 research_memory_db_fixture.can_connect_db）。
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

import research_memory_db_fixture as db_fixture
from app.models import research_memory_contracts as contracts
from app.repositories import research_memory_repository as repo_mod
from app.services import research_memory_injector_service as injector

# 真库组的跳过条件：库不可连只跳真库用例，结构组照常跑
requires_db = pytest.mark.skipif(
    not db_fixture.can_connect_db(), reason="业务库不可连，跳过研究记忆资格门真库测试"
)

# 被验的枚举字面量集中在此，避免测试里散落裸字符串
_STATUS_DRAFT = "draft"
_STATUS_CONFIRMED = "confirmed"
_REVIEW_PENDING = "pending_review"
_REVIEW_ACCEPTED = "accepted"
_STALE_FRESH = "fresh"
_STALE_STALE = "stale"
_STALE_NEEDS_REBIND = "needs_rebind"
_MATURITY_ESTABLISHED = "established"
_MATURITY_LEANING = "leaning"
_MATURITY_INTUITION = "intuition"
_MEMORY_TYPE = "decision_note"

# 记忆读取的终态取值：empty 与 failed 不得归并，所以断言要点到具体那一态
_READ_SUCCESS = "success"
_READ_EMPTY = "empty"

# 记忆归属人：本人与同项目里的另一个人，用来验 user_id 这道门
_OWNER_USER = "gate_owner"
_OTHER_USER = "gate_intruder"
_PROJECT_ID = "proj-gate"

# 各种「空身份」写法：缺值、空串、纯空白。三者都不是合法的记忆归属人
_BLANK_USER_IDS: tuple[Any, ...] = (None, "", "   ")

# 加载函数递给仓储的四个关键字过滤条件（project_id 走位置参数，不在此列）
_GATE_KWARG_NAMES = ("user_id", "review_state", "status", "stale_state")
# 加载函数对外只暴露这两个参数，没有任何可放宽的口子
_LOADER_PARAM_NAMES = ("project_id", "user_id")

# 成熟度 → 注入给 LLM 的身份标签。这里写死期望值而不是读实现里的映射表：
# 独立声明才能在实现被改动时报警，intuition 必须是「假设」而不是中性的「直觉」
_EXPECTED_PROVENANCE_LABEL = MappingProxyType(
    {_MATURITY_ESTABLISHED: "已确立", _MATURITY_LEANING: "倾向", _MATURITY_INTUITION: "假设"}
)

# 条数上限拨到远大于用例造的记忆数，避免预算截断混进资格门断言
_INJECT_MAX_ITEMS = 20

# 真库用例里按标题认条目，标题即断言口径
_TITLE_QUALIFIED = "四道门全满足的判断"
_TITLE_OTHER_USER = "他人的合格判断"


def _run(coro):
    return asyncio.run(coro)


class _FakeSettings:
    # 只有条数上限会被加载路径读到；字符预算属渲染阶段，本文件不测渲染
    research_memory_inject_max_items = _INJECT_MAX_ITEMS


# ─── A 组：结构保证（不连库） ───


class _CapturingRepo:
    """假 repo：只记录加载函数递给仓储的过滤条件，不返回任何行。"""

    def __init__(self) -> None:
        self.project_id: str | None = None
        self.gates: dict[str, Any] = {}

    async def ensure_schema(self) -> None:
        return None

    async def list_items(self, project_id, /, **gates):
        # project_id 声明为位置形参并加 `/`：加载函数若改成按关键字传，这里直接 TypeError
        self.project_id = project_id
        self.gates = dict(gates)
        return []


class _ForbiddenRepo:
    """假 repo：一被碰到就记账并直接失败，用来验入参校验发生在碰库之前。"""

    def __init__(self) -> None:
        self.touched = False

    async def ensure_schema(self) -> None:
        self.touched = True
        raise AssertionError("空身份不该走到仓储：入参校验必须在碰库之前完成")

    async def list_items(self, project_id, /, **gates):
        self.touched = True
        raise AssertionError("空身份不该查库：user_id 门在这里已经形同虚设")


def _load_with_repo(
    repo: Any, overrides: dict[str, Any] | None = None, *, user_id: Any = _OWNER_USER
) -> dict[str, Any]:
    """
    用给定的假 repo 跑一次记忆读取。

    参数:
        repo: 替身仓储
        overrides: 额外塞给加载函数的关键字参数，用于验证调用方放宽不了门
        user_id: 记忆归属人，传空值用于验入参校验
    返回:
        记忆读取结果
    """
    with patch.object(injector, "get_settings", return_value=_FakeSettings()), \
         patch.object(injector, "get_research_memory_repository", return_value=repo):
        return asyncio.run(
            injector.load_injectable_memory(_PROJECT_ID, user_id, **(overrides or {}))
        )


class TestGateStructureCannotBeWidened:
    """验「Planner 只能收窄不能放宽」这件事在结构上成立，而不是靠约定。"""

    def test_five_gates_all_reach_repository(self) -> None:
        """
        五道门条件全部递到仓储：当前项目 + 当前用户 + accepted + confirmed + fresh。

        少递一道就等于那道门在 SQL 里根本不存在，不合格记忆会照常回到应用层。
        """
        repo = _CapturingRepo()
        _load_with_repo(repo)

        # 1. 项目归属必须走位置参数落进仓储，跨项目记忆不互相串
        assert repo.project_id == _PROJECT_ID
        # 2. 余下四道门逐个核对取值，不能只是「键在」而值被写错
        assert repo.gates["user_id"] == _OWNER_USER
        assert repo.gates["review_state"] == _REVIEW_ACCEPTED
        assert repo.gates["status"] == _STATUS_CONFIRMED
        assert repo.gates["stale_state"] == _STALE_FRESH

    def test_maturity_is_not_a_gate(self) -> None:
        """
        maturity 不在资格门里（决策 A）：三档成熟度都有资格进上下文。

        仓储明明支持按 maturity 过滤，加载函数偏偏不用——这是有意选择而非能力缺失，
        所以两件事一起断言：门里没有 maturity，且这个过滤位确实存在。
        """
        repo = _CapturingRepo()
        _load_with_repo(repo)

        assert "maturity" not in repo.gates
        # 门只由这四项构成，多一项就意味着有人往资格判定里塞了新维度
        assert set(repo.gates) == set(_GATE_KWARG_NAMES)
        repo_params = inspect.signature(repo_mod.ResearchMemoryRepository.list_items).parameters
        assert "maturity" in repo_params

    def test_loader_signature_offers_no_widening_hook(self) -> None:
        """
        加载函数只收 project_id 与 user_id，没有 *args / **kwargs、两参都无默认值。

        这是「Planner 传放宽参数也不行」的结构证据：想放宽根本没有参数可传，
        也不会因为漏传 user_id 而退化成读全项目所有人的记忆。
        """
        signature = inspect.signature(injector.load_injectable_memory)

        assert tuple(signature.parameters) == _LOADER_PARAM_NAMES
        kinds = {param.kind for param in signature.parameters.values()}
        assert inspect.Parameter.VAR_POSITIONAL not in kinds
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        for name in _LOADER_PARAM_NAMES:
            assert signature.parameters[name].default is inspect.Parameter.empty

    @pytest.mark.parametrize(
        "override",
        [
            {"stale_state": _STALE_STALE},
            {"status": _STATUS_DRAFT},
            {"review_state": _REVIEW_PENDING},
            {"maturity": _MATURITY_INTUITION},
        ],
        ids=["stale_state", "status", "review_state", "maturity"],
    )
    def test_caller_cannot_override_any_gate(self, override: dict[str, Any]) -> None:
        """调用方试图把任一道门的取值传进来时直接 TypeError，放宽请求连编译期都过不去。"""
        with pytest.raises(TypeError):
            _load_with_repo(_CapturingRepo(), override)

    @pytest.mark.parametrize("blank_user_id", _BLANK_USER_IDS, ids=["none", "empty", "whitespace"])
    def test_blank_user_id_rejected(self, blank_user_id: Any) -> None:
        """
        用户身份传空一律 ValueError：空身份进到仓储会让 user_id 那道门整段消失。

        仓储的过滤循环遇到 None 会跳过拼 WHERE，于是读出该项目下所有人的记忆。
        空身份从来不是合法输入，按系统边界的入参校验拒掉，不留半合法的口子。
        """
        with pytest.raises(ValueError):
            _load_with_repo(_ForbiddenRepo(), user_id=blank_user_id)

    @pytest.mark.parametrize("blank_user_id", _BLANK_USER_IDS, ids=["none", "empty", "whitespace"])
    def test_blank_user_id_rejected_before_touching_repository(self, blank_user_id: Any) -> None:
        """
        空身份必须在碰仓储之前就抛错，不能被 except Exception 收成 failed。

        failed 的语义是「记忆存储不可用」；用它承载接线错误，接线接错时会显示成
        环境问题被容忍过去。假 repo 一被调用即失败，锁住「校验在 try 之外」这件事。
        """
        repo = _ForbiddenRepo()
        with pytest.raises(ValueError):
            _load_with_repo(repo, user_id=blank_user_id)

        assert repo.touched is False


# ─── B 组：真库合规（库不可连时整个 class 跳过） ───

# 四轴全合格的记忆形状；用例只覆盖其中一个轴来制造「单点不合格」
_QUALIFIED_AXES = MappingProxyType(
    {
        "memory_type": _MEMORY_TYPE,
        "user_id": _OWNER_USER,
        "status": _STATUS_CONFIRMED,
        "review_state": _REVIEW_ACCEPTED,
        "stale_state": _STALE_FRESH,
        "maturity": _MATURITY_LEANING,
    }
)


async def _seed(project_id: str, title: str, **axis_overrides: Any) -> str:
    """
    在真库里造一条记忆，默认四轴全合格。

    直连仓储而不走 service：service 的 _derive_initial_status 会按来源改写 status，
    而本文件要精确控制每个轴的取值。

    参数:
        project_id: 记忆落在哪个项目
        title: 记忆标题，断言里靠它认条目
        axis_overrides: 要打破的轴取值，如 status=draft
    返回:
        新建记忆的 memory_id
    """
    detail = await repo_mod.get_research_memory_repository().create_item(
        project_id, {**_QUALIFIED_AXES, "title": title, **axis_overrides}
    )
    return detail["item"]["memory_id"]


async def _load(project_id: str, user_id: str) -> dict[str, Any]:
    """
    按当前用户读一次可注入记忆。

    参数:
        project_id: 归属项目
        user_id: 记忆归属人
    返回:
        记忆读取结果
    """
    with patch.object(injector, "get_settings", return_value=_FakeSettings()):
        return await injector.load_injectable_memory(project_id, user_id)


def _titles(result: dict[str, Any]) -> list[str]:
    """取读取结果里各条记忆的标题，用于核对到底哪些条目通过了门。"""
    return [item["title"] for item in result["items"]]


@asynccontextmanager
async def _two_temp_projects():
    """开两个临时项目验跨项目不互串，退出时各自级联清理。"""
    async with db_fixture.temp_project() as first_project:
        async with db_fixture.temp_project() as second_project:
            yield first_project, second_project


@requires_db
class TestGateComplianceOnRealDb:
    """在真实 Postgres 上验四道门真的筛对了，而不只是过滤参数递了过去。"""

    def test_only_fully_qualified_memory_enters_context(self) -> None:
        """
        五个轴各造一条单点不合格的记忆，只有全满足那条进上下文。

        这条等价于「记忆注入合规率 100%」：任一轴不合格即出局，不存在「降权混入」。
        """
        async def scenario() -> dict[str, Any]:
            async with _two_temp_projects() as (project_id, other_project_id):
                await _seed(project_id, _TITLE_QUALIFIED)
                await _seed(project_id, "待人审的草稿", review_state=_REVIEW_PENDING)
                await _seed(project_id, "尚未确认的条目", status=_STATUS_DRAFT)
                await _seed(project_id, "证据已过期", stale_state=_STALE_STALE)
                await _seed(project_id, "证据待重绑", stale_state=_STALE_NEEDS_REBIND)
                await _seed(project_id, _TITLE_OTHER_USER, user_id=_OTHER_USER)
                await _seed(other_project_id, "别的项目里的判断")
                return await _load(project_id, _OWNER_USER)

        result = _run(scenario())
        assert result["status"] == _READ_SUCCESS
        assert _titles(result) == [_TITLE_QUALIFIED]

    @pytest.mark.parametrize("stale_state", [_STALE_STALE, _STALE_NEEDS_REBIND])
    def test_stale_memory_never_leaks(self, stale_state: str) -> None:
        """
        失效记忆泄漏率 0：stale / needs_rebind 即便已接纳已确认也读不出来。

        另外三轴全合格是这条的关键——失效判定不依赖别的轴，单凭新鲜度就该出局。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                await _seed(project_id, "证据已失效的判断", stale_state=stale_state)
                return await _load(project_id, _OWNER_USER)

        result = _run(scenario())
        assert result["status"] == _READ_EMPTY
        assert result["items"] == []

    def test_pending_ai_draft_never_enters_context(self) -> None:
        """AI 草稿在人审前不许喂回 LLM：draft + pending_review 的条目读不出来。"""
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                await _seed(
                    project_id,
                    "AI 抽出的候选判断",
                    status=_STATUS_DRAFT,
                    review_state=_REVIEW_PENDING,
                )
                return await _load(project_id, _OWNER_USER)

        result = _run(scenario())
        assert result["status"] == _READ_EMPTY
        assert result["items"] == []

    def test_memory_does_not_cross_users(self) -> None:
        """
        user_id 隔离：同项目里另一个人的合格记忆不进本人上下文。

        双向都读一次：证明这道门是按归属人筛，而不是恰好把某条记忆整体漏掉了。
        """
        async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
            async with db_fixture.temp_project() as project_id:
                await _seed(project_id, _TITLE_QUALIFIED)
                await _seed(project_id, _TITLE_OTHER_USER, user_id=_OTHER_USER)
                return await _load(project_id, _OWNER_USER), await _load(project_id, _OTHER_USER)

        mine, theirs = _run(scenario())
        assert _titles(mine) == [_TITLE_QUALIFIED]
        assert _titles(theirs) == [_TITLE_OTHER_USER]

    @pytest.mark.parametrize("maturity", contracts.VALID_MATURITIES)
    def test_every_maturity_is_eligible(self, maturity: str) -> None:
        """
        三档成熟度都有资格进上下文（决策 A），身份标签按档派生。

        这条防的是日后有人把 maturity 加回资格门：直觉级判断也要能进，只是以「假设」
        身份出现，让模型看清它还没被证据坐实。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                await _seed(project_id, _TITLE_QUALIFIED, maturity=maturity)
                return await _load(project_id, _OWNER_USER)

        result = _run(scenario())
        assert result["status"] == _READ_SUCCESS
        assert _titles(result) == [_TITLE_QUALIFIED]
        assert result["items"][0]["maturity"] == maturity
        assert result["items"][0]["provenance_label"] == _EXPECTED_PROVENANCE_LABEL[maturity]

    def test_empty_project_reports_empty_not_failed(self) -> None:
        """
        项目里没有任何合格记忆时是 empty 而不是 failed，且不带 error_kind。

        两态归并会让调用方分不清「该重试」还是「本项目确实还没攒下判断」。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                return await _load(project_id, _OWNER_USER)

        result = _run(scenario())
        assert result["status"] == _READ_EMPTY
        assert "error_kind" not in result
        assert result["items"] == []
        assert result["excluded_counts"] == {}
