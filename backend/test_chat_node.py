from app.nodes import chat_node


def test_free_chat_system_prompt_avoids_latex_by_default() -> None:
    assert "plain text expressions" in chat_node._FREE_CHAT_SYSTEM
    assert "LaTeX" in chat_node._FREE_CHAT_SYSTEM
