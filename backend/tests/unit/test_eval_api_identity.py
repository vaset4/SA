"""API 评测条件不绑定本地权重的回归，作者：xxx。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.services.embedding_client import EmbeddingClient
from evals.suites import identity


def test_api_embedding_identity_captures_actual_overrides_without_weights(monkeypatch):
    """参数为依赖替身；实际服务、模型覆盖值和维度进入身份，不读取模型目录或密钥。"""
    settings = Settings(_env_file=None, embedding_mode="api", embedding_api_key="test-key",
        embedding_api_base_url="https://example.com/v1/")
    client = EmbeddingClient(config=settings, model_name="configured-model", dimension=7, batch_size=3)
    monkeypatch.setattr("app.services.embedding_client.get_embedding_client", lambda: client)
    resolve = Mock(side_effect=AssertionError("API 身份不得查本地权重"))
    monkeypatch.setattr("app.services.hf_local.resolve_local_model_source", resolve)
    result = identity._embedding_identity()
    assert result["mode"] == "api" and result["model"] == "configured-model"
    assert result["dimension"] == 7 and result["batch_size"] == 3
    assert result["service"] == "https://example.com/v1"
    assert "test-key" not in repr(result)
    resolve.assert_not_called()


@pytest.mark.parametrize("mode", ["api", "local"])
def test_dependency_identity_requires_local_packages_only_in_local_mode(monkeypatch, mode):
    """参数为替身和模式；API 不要求可选模型包，local 缺包仍明确失败。"""
    settings = Settings(_env_file=None, embedding_mode=mode, rerank_mode="api")
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    actual = identity.importlib.metadata.version
    def version(name):
        """参数为包名；显式模拟未安装本地模型包，其余包使用实际版本。"""
        if name == "sentence-transformers":
            raise identity.importlib.metadata.PackageNotFoundError(name)
        return actual(name)
    monkeypatch.setattr(identity.importlib.metadata, "version", version)
    bindings = SimpleNamespace(kind="planner_fixture", fixture_snapshot="{}",
        prepared=SimpleNamespace(profile=SimpleNamespace(grading_timing=None, repetition_order=None)))
    if mode == "local":
        with pytest.raises(identity.importlib.metadata.PackageNotFoundError):
            identity._environment_identity(bindings)
        return
    result = identity._environment_identity(bindings)
    assert "sentence-transformers" not in result["dependencies"]
    assert "tokenizers" in result["dependencies"]
