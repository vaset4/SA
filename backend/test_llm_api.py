"""LLM 接口连通性测试脚本。

运行示例：
    python test_llm_api.py
    python test_llm_api.py --prompt "你好，请返回 ok"
    python test_llm_api.py --http-only
    python test_llm_api.py --langchain-only

用途：
1. 打印当前项目实际读取到的 LLM 配置
2. 直接调用 OpenAI 兼容接口 `/chat/completions`
3. 调用 `langchain_openai.ChatOpenAI` 做一次最小请求
4. 清晰暴露错误响应，便于判断是 base_url、模型名还是提供商兼容性问题
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests
from langchain_openai import ChatOpenAI

from app.config import get_settings


DEFAULT_PROMPT = "你好，请只回复：ok"
DEFAULT_TIMEOUT = 60
CHAT_COMPLETIONS_PATH = "/chat/completions"


def _ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _mask_secret(secret: str) -> str:
    if not secret:
        return "<EMPTY>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").rstrip("/")


def _build_chat_completions_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        raise ValueError("LLM_BASE_URL 为空，无法拼接 chat/completions 地址")
    return f"{normalized}{CHAT_COMPLETIONS_PATH}"


def _print_config() -> None:
    settings = get_settings()
    print("=" * 70)
    print("LLM 配置检查")
    print("=" * 70)
    print(f"模型: {settings.llm_model_name}")
    print(f"Base URL: {_normalize_base_url(settings.llm_base_url)}")
    print(f"API Key: {_mask_secret(settings.llm_api_key)}")
    print(f"Temperature: {settings.llm_temperature}")
    print()


def _print_response_preview(data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    preview = text[:3000]
    print(preview)
    if len(text) > len(preview):
        print("...<truncated>")


def test_http_chat_completions(prompt: str, timeout: int) -> bool:
    settings = get_settings()
    url = _build_chat_completions_url(settings.llm_base_url)
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.llm_temperature,
    }

    print("- HTTP 测试开始")
    print(f"  URL: {url}")
    print(f"  Payload.model: {payload['model']}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        print(f"  HTTP 请求异常: {type(exc).__name__}: {exc}")
        return False

    print(f"  HTTP 状态码: {response.status_code}")
    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text[:3000]}

    _print_response_preview(data)

    if not response.ok:
        print("  结果: FAIL（服务端返回非 2xx）")
        return False

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        print("  结果: FAIL（响应中没有 choices 字段）")
        return False

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    print(f"  模型返回: {content!r}")
    print("  结果: PASS")
    return True


def test_langchain_chatopenai(prompt: str, timeout: int) -> bool:
    settings = get_settings()

    print("- LangChain 测试开始")
    print(f"  Base URL: {_normalize_base_url(settings.llm_base_url)}")
    print(f"  Model: {settings.llm_model_name}")

    try:
        llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            timeout=timeout,
        )
        response = llm.invoke(prompt)
        print(f"  模型返回: {response.content!r}")
        print("  结果: PASS")
        return True
    except Exception as exc:
        print(f"  LangChain 调用异常: {type(exc).__name__}: {exc}")
        print("  结果: FAIL")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试当前项目的 LLM 接口配置")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="测试提示词")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="请求超时秒数")
    parser.add_argument("--http-only", action="store_true", help="只测试原始 HTTP 接口")
    parser.add_argument("--langchain-only", action="store_true", help="只测试 LangChain ChatOpenAI")
    return parser.parse_args()


def main() -> int:
    _ensure_utf8_stdout()
    args = parse_args()
    _print_config()

    if args.http_only and args.langchain_only:
        print("不能同时指定 --http-only 和 --langchain-only")
        return 2

    run_http = not args.langchain_only
    run_langchain = not args.http_only

    http_ok = True
    langchain_ok = True

    if run_http:
        http_ok = test_http_chat_completions(args.prompt, args.timeout)
        print()

    if run_langchain:
        langchain_ok = test_langchain_chatopenai(args.prompt, args.timeout)
        print()

    if http_ok and langchain_ok:
        print("结论: 所有测试通过")
        return 0

    print("结论: 存在失败项，请根据上面的 URL / 状态码 / 异常信息排查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
