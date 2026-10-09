"""focus_resolver 单元测试。"""

import pytest

from app.services.internal_research.focus_resolver import (
    FOCUS_REFERENCE_UNRESOLVED,
    resolve_focus,
)

# 本轮清单：ID 顺序故意不是字典序，用来证明序数指代按清单顺序数而不是按 ID 排序数
_TURN_SCOPE = ["p-zz", "p-aa", "p-mm"]

_TURN_PAPERS = (
    {"paper_id": "p-zz", "title": "Attention Is All You Need"},
    {"paper_id": "p-aa", "title": "Deep Residual Learning"},
    {"paper_id": "p-mm", "title": "BERT 预训练模型"},
)


class TestResolveFocus:
    """三层焦点 + 降级路径测试。"""

    def test_explore_default_fallback(self):
        result = resolve_focus(
            "这个领域有什么趋势",
            intent="explore",
            project_paper_ids=["paper_a", "paper_b"],
        )
        assert result["project_scope"] == ["paper_a", "paper_b"]
        assert result["turn_focus"] == ["paper_a", "paper_b"]
        assert result["fallback_reason"] == "explore_default"

    def test_lookup_narrows_by_title(self):
        """点名标题时收窄到对应论文，不再靠 UUID 子串碰运气。"""
        result = resolve_focus(
            "Attention Is All You Need 的方法是什么",
            intent="lookup",
            project_paper_ids=[paper["paper_id"] for paper in _TURN_PAPERS],
            papers=_TURN_PAPERS,
        )
        assert result["turn_focus"] == ["p-zz"]
        assert result["fallback_reason"] == ""

    def test_lookup_ignores_paper_id_substring(self):
        """问题里出现 paper_id 字面量也不该被当成点名：A6 明确禁掉 UUID 子串匹配。"""
        result = resolve_focus(
            "p-zz 的方法是什么",
            intent="lookup",
            project_paper_ids=[paper["paper_id"] for paper in _TURN_PAPERS],
            papers=_TURN_PAPERS,
        )
        assert result["turn_focus"] == [paper["paper_id"] for paper in _TURN_PAPERS]
        assert result["fallback_reason"] == "lookup_no_focus"

    # 序数指代按本轮清单顺序数：清单 ID 故意不是字典序，命中说明数的是清单而不是 ID 排序。
    # 这几例都不传 papers，说明序数指代不依赖标题，读不到标题时仍然能收窄
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("第二篇的实验结果是什么", ["p-aa"]),
            ("第 1 篇讲了什么", ["p-zz"]),
            ("第一篇和第三篇的结论差在哪", ["p-zz", "p-mm"]),
            ("前两篇的方法", ["p-zz", "p-aa"]),
        ],
    )
    def test_ordinal_reference_narrows_by_turn_order(self, question, expected):
        result = resolve_focus(
            question,
            intent="lookup",
            project_paper_ids=list(_TURN_SCOPE),
        )
        assert result["turn_focus"] == expected
        assert result["fallback_reason"] == ""

    def test_this_two_papers_covers_whole_turn_scope(self):
        """「这两篇」在两篇清单下就是这两篇，既不退回也不算解析失败。"""
        result = resolve_focus(
            "这两篇的实验设置有什么不同",
            intent="compare",
            project_paper_ids=["p-zz", "p-aa"],
        )
        assert result["turn_focus"] == ["p-zz", "p-aa"]
        assert result["fallback_reason"] == ""

    # F11：同一句里混用「前X篇」「这X篇」「第X篇」时必须一起收下，按各指代在问题里
    # 出现的先后排列。只认第一个命中的指代会把后一个指代的论文整句丢掉
    @pytest.mark.parametrize(
        ("question", "intent", "expected"),
        [
            ("对比前两篇与第三篇的实验结果", "compare", ["p-zz", "p-aa", "p-mm"]),
            ("第三篇跟前两篇的实验条件有什么不同", "compare", ["p-mm", "p-zz", "p-aa"]),
            ("第一篇和第三篇的实验结果", "compare", ["p-zz", "p-mm"]),
            # 单篇结果走 lookup：compare 只解出一篇时会按既有策略补会话层再退回全量，
            # 这里要验的是去重本身，不是 compare 的补足策略
            ("第一篇和第一篇文章的结论一致吗", "lookup", ["p-zz"]),
            ("这两篇与前两篇的区别", "compare", ["p-zz", "p-aa"]),
        ],
    )
    def test_mixed_ordinal_references_are_merged_in_order(self, question, intent, expected):
        result = resolve_focus(
            question,
            intent=intent,
            project_paper_ids=list(_TURN_SCOPE),
        )
        assert result["turn_focus"] == expected
        assert result["fallback_reason"] == ""

    @pytest.mark.parametrize(
        "question",
        ["对比前两篇与第四篇的实验结果", "对比前五篇与第一篇的实验结果"],
    )
    def test_mixed_reference_with_one_out_of_range_fails_whole_sentence(self, question):
        """合并后仍不出半个答案：只要一条指代越界，整句退回「没解析出」。"""
        result = resolve_focus(
            question,
            intent="compare",
            project_paper_ids=list(_TURN_SCOPE),
        )
        assert result["turn_focus"] == list(_TURN_SCOPE)
        assert result["fallback_reason"] == FOCUS_REFERENCE_UNRESOLVED

    @pytest.mark.parametrize(
        "question",
        ["倒数第二篇的结论", "刚才那篇的实验结果", "第五篇怎么说"],
    )
    def test_unresolvable_reference_records_unresolved(self, question):
        """指代解析不出来时退回项目全量，原因记「没解析出」，与「没收窄」区分开。"""
        result = resolve_focus(
            question,
            intent="lookup",
            project_paper_ids=list(_TURN_SCOPE),
        )
        assert result["turn_focus"] == list(_TURN_SCOPE)
        assert result["fallback_reason"] == FOCUS_REFERENCE_UNRESOLVED

    def test_lookup_no_focus_uses_previous(self):
        previous = {
            "project_scope": ["paper_a"],
            "conversation_scope": [],
            "turn_focus": ["paper_a"],
            "fallback_reason": "",
        }
        result = resolve_focus(
            "那它的实验结果呢",
            intent="lookup",
            project_paper_ids=["paper_a", "paper_b"],
            previous_focus=previous,
        )
        assert result["turn_focus"] == ["paper_a"]
        assert result["fallback_reason"] == ""

    def test_lookup_no_focus_no_previous(self):
        result = resolve_focus(
            "那它的实验结果呢",
            intent="lookup",
            project_paper_ids=["paper_a", "paper_b"],
        )
        assert result["turn_focus"] == ["paper_a", "paper_b"]
        assert result["fallback_reason"] == "lookup_no_focus"

    def test_compare_with_two_titles(self):
        """compare 点名两篇标题时收窄到这两篇，不退回项目全量。"""
        result = resolve_focus(
            "Attention Is All You Need 和 BERT 的方法有什么不同",
            intent="compare",
            project_paper_ids=["paper_a", "paper_b", "paper_c"],
            papers=[
                {"paper_id": "paper_a", "title": "Attention Is All You Need"},
                {"paper_id": "paper_b", "title": "BERT"},
                {"paper_id": "paper_c", "title": "ResNet"},
            ],
        )
        assert result["turn_focus"] == ["paper_a", "paper_b"]
        assert result["fallback_reason"] == ""

    def test_compare_insufficient_supplements_from_conversation(self):
        result = resolve_focus(
            "和上次那篇比呢",
            intent="compare",
            project_paper_ids=["paper_a", "paper_b"],
            conversation_history=[
                {"paper_ids": ["paper_b"]},
                {"paper_ids": ["paper_a"]},
            ],
        )
        assert len(result["turn_focus"]) >= 2

    def test_compare_insufficient_no_conversation(self):
        result = resolve_focus(
            "和上次那篇比呢",
            intent="compare",
            project_paper_ids=["paper_a", "paper_b"],
        )
        assert result["fallback_reason"] == "compare_insufficient_focus"

    def test_conversation_scope_extraction(self):
        history = [
            {"paper_ids": ["p1"]},
            {"paper_ids": ["p2", "p3"]},
            {"paper_ids": ["p1"]},
        ]
        result = resolve_focus(
            "总结一下",
            intent="explore",
            project_paper_ids=["p1", "p2", "p3"],
            conversation_history=history,
        )
        assert "p1" in result["conversation_scope"]
        assert "p2" in result["conversation_scope"]
        assert "p3" in result["conversation_scope"]

    def test_empty_project_scope(self):
        result = resolve_focus(
            "随便聊聊",
            intent="explore",
            project_paper_ids=[],
        )
        assert result["project_scope"] == []
        assert result["turn_focus"] == []
        assert result["fallback_reason"] == "explore_default"


class TestAuthorizedNarrowing:
    """F01/C1：历史与上一轮焦点都只是候选输入，不得把焦点扩大到授权范围之外。

    锁的是本函数的不变量——输出的 `turn_focus` 永远是授权范围的子集；越界候选被丢掉
    之后照既有退回落回授权全量（design §「求交后 compare 焦点不足 2 篇时沿用既有退回」），
    不能默默收窄到空集。
    """

    def test_history_candidate_outside_authorization_is_dropped(self):
        """历史带上范围外论文时，焦点不再用它补足，退回授权全量。"""
        result = resolve_focus(
            "对比第一篇的方法",
            intent="compare",
            project_paper_ids=["own_a", "own_b"],
            conversation_history=[{"paper_ids": ["foreign_b"]}],
        )
        assert "foreign_b" not in result["turn_focus"]
        assert result["turn_focus"] == ["own_a", "own_b"]
        assert result["fallback_reason"] == "compare_insufficient_focus"

    def test_history_candidate_inside_authorization_still_supplements(self):
        """合法范围内的历史补焦点行为不变，按历史出现的先后顺序补。"""
        result = resolve_focus(
            "和上次那篇比呢",
            intent="compare",
            project_paper_ids=["own_a", "own_b"],
            conversation_history=[{"paper_ids": ["own_b"]}, {"paper_ids": ["own_a"]}],
        )
        assert result["turn_focus"] == ["own_b", "own_a"]
        assert result["fallback_reason"] == ""

    def test_mixed_history_never_promotes_out_of_scope_candidate(self):
        """历史同时带合法与越界候选：越界的那篇既不补进焦点，也不顶替合法候选的位置。"""
        result = resolve_focus(
            "和上次那篇比呢",
            intent="compare",
            project_paper_ids=["own_a", "own_b"],
            conversation_history=[{"paper_ids": ["foreign_c", "own_b"]}],
        )
        assert "foreign_c" not in result["turn_focus"]
        assert set(result["turn_focus"]) <= {"own_a", "own_b"}

    def test_previous_focus_outside_authorization_is_dropped(self):
        """承接的上一轮焦点同样只能收窄：上一轮合法、本轮授权已缩小后不得再带出来。"""
        previous = {
            "project_scope": ["own_a", "own_b"],
            "conversation_scope": [],
            "turn_focus": ["own_b"],
            "fallback_reason": "",
        }
        result = resolve_focus(
            "那它的实验结果呢",
            intent="lookup",
            project_paper_ids=["own_a"],
            previous_focus=previous,
        )
        assert "own_b" not in result["turn_focus"]
        assert result["turn_focus"] == ["own_a"]
        assert result["fallback_reason"] == "lookup_no_focus"

    def test_conversation_scope_is_kept_as_information(self):
        """越界候选只从焦点里丢掉；conversation_scope 是信息不是权限，保持原样。"""
        result = resolve_focus(
            "总结一下",
            intent="explore",
            project_paper_ids=["own_a"],
            conversation_history=[{"paper_ids": ["foreign_c"]}],
        )
        assert result["conversation_scope"] == ["foreign_c"]
        assert "foreign_c" not in result["turn_focus"]
