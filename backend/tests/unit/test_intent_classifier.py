"""intent_classifier 单元测试：主链唯一一处意图判断（正则四档）。"""

import pytest

from app.services.internal_research.intent_classifier import classify_intent


class TestClassifyIntent:
    """规则意图识别的四档判定与置信度口径。"""

    # inventory
    def test_inventory_keyword_chinese(self):
        r = classify_intent("列出项目中的所有论文")
        assert r["intent"] == "inventory"
        assert r["confidence"] >= 0.8

    def test_inventory_keyword_english(self):
        r = classify_intent("list all papers in the project")
        assert r["intent"] == "inventory"

    def test_inventory_count(self):
        r = classify_intent("有多少篇论文？")
        assert r["intent"] == "inventory"

    # compare
    def test_compare_keyword(self):
        r = classify_intent("对比这两篇论文的方法")
        assert r["intent"] == "compare"

    def test_compare_multi_paper(self):
        """论文数量不再参与判定，只由问题文本的关键词决定。"""
        r = classify_intent("这篇论文怎么样")
        assert r["intent"] == "explore"

    def test_compare_vs(self):
        r = classify_intent("method A vs method B")
        assert r["intent"] == "compare"

    # lookup
    def test_lookup_definition(self):
        r = classify_intent("什么是 Transformer 架构？")
        assert r["intent"] == "lookup"

    def test_lookup_explain(self):
        r = classify_intent("解释下这个方法的原理")
        assert r["intent"] == "lookup"

    def test_lookup_english(self):
        r = classify_intent("what is the attention mechanism?")
        assert r["intent"] == "lookup"

    def test_lookup_ending_wins_over_compare_keyword(self):
        """带结尾式提问的查询句不得被对比关键词抢走。"""
        r = classify_intent("这两个模型的注意力机制区别是什么？")
        assert r["intent"] == "lookup"
        assert "lookup_ending" in r["signals"]

    # explore (default)
    def test_explore_default(self):
        r = classify_intent("这个领域有什么值得关注的趋势")
        assert r["intent"] == "explore"
        assert r["confidence"] < 0.6

    # conflicting signals
    def test_conflicting_signals_low_confidence(self):
        r = classify_intent("列出并比较所有论文的方法")
        assert r["confidence"] < 0.6
        assert "conflicting_signals" in r["signals"]


def test_result_has_no_route_keys():
    """语义路由已整体删除，意图结果不再携带任何 route_* 键。"""
    result = classify_intent("对比这两篇论文的方法")

    assert not [key for key in result if key.startswith("route_")]


def test_result_is_pure_rule_source():
    """唯一产出者被删后，source 只剩 rule 一种取值。"""
    for question in ("列出所有论文", "什么是注意力机制", "对比两篇论文", "随便聊聊"):
        assert classify_intent(question)["source"] == "rule"


def test_only_classify_intent_remains():
    """透传包装与 fallback 判定随语义路由一起删除，只留 classify_intent 一处入口。"""
    from app.services.internal_research import intent_classifier

    assert not hasattr(intent_classifier, "classify_intent_with_llm_fallback")
    assert not hasattr(intent_classifier, "needs_llm_fallback")


@pytest.mark.parametrize(
    "module_name",
    [
        "app.services.internal_research.semantic_intent_classifier",
        "app.services.internal_research.intent_routes",
    ],
)
def test_semantic_router_modules_are_gone(module_name: str):
    """两个语义路由模块已在运行期不可导入。"""
    with pytest.raises(ModuleNotFoundError):
        __import__(module_name)
