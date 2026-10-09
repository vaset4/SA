"""模型模式、向量身份和公开副本配置边界测试，作者：xxx。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings, get_embedding_identity, normalize_model_api_url

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def identity_config(**changes):
    """接收测试字段覆盖，返回纯向量身份配置，不使用运行时环境文件。"""
    return SimpleNamespace(**({"embedding_model": "BAAI/bge-m3", "embedding_dimension": 1024,
        "embedding_mode": "local", "embedding_api_base_url": "https://models.example/v1",
        "embedding_api_key": "test-key"} | changes))


def test_local_identity_is_byte_compatible_and_api_identity_is_scoped():
    """local 摘要逐字节兼容旧算法，API 按服务与模型隔离且不受密钥轮换影响。"""
    config = identity_config()
    old_payload = {"model": config.embedding_model, "dimension": config.embedding_dimension, "normalized": True}
    legacy = hashlib.sha256(json.dumps(old_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert get_embedding_identity(config) == legacy
    remote = get_embedding_identity(identity_config(embedding_mode="api"))
    assert len(remote) == 64
    assert remote != legacy
    assert remote == get_embedding_identity(identity_config(embedding_mode="api", embedding_api_key="rotated"))
    assert remote == get_embedding_identity(identity_config(embedding_mode="api",
        embedding_api_base_url="https://MODELS.EXAMPLE:443/v1/"))
    for changes in [{"embedding_api_base_url": "https://other.example/v1"},
            {"embedding_model": "other-model"}, {"embedding_dimension": 768}]:
        assert remote != get_embedding_identity(identity_config(embedding_mode="api", **changes))


@pytest.mark.parametrize("url", ["", "file:///model", "https://user:private@host/v1",
    "https://host/v1?api_key=private", "https://host/v1#private", "https://host:bad/v1", "https://a b/v1"])
def test_invalid_service_url_is_rejected_without_echoing_credentials(url):
    """地址仅允许无凭证的 HTTP(S)，错误信息不回显输入中的秘密。"""
    with pytest.raises(ValueError, match="EMBEDDING_API_BASE_URL") as caught:
        normalize_model_api_url(url, setting_name="EMBEDDING_API_BASE_URL")
    assert "private" not in str(caught.value)


def test_api_defaults_and_mode_validation(monkeypatch):
    """无环境配置时默认 API；非法模式和非正批大小在配置边界失败。"""
    for key in list(os.environ):
        if key.lower() in Settings.model_fields:
            monkeypatch.delenv(key)
    settings = Settings(_env_file=None)
    assert settings.embedding_mode == settings.rerank_mode == "api"
    assert settings.embedding_api_key == settings.rerank_api_key == ""
    assert settings.embedding_api_base_url == "https://api.siliconflow.cn/v1"
    assert settings.rerank_endpoint == "https://api.siliconflow.cn/v1/rerank"
    assert "rerank_provider" not in Settings.model_fields
    for options in [{"embedding_mode": "automatic"}, {"rerank_mode": "disabled"}, {"embedding_batch_size": 0}]:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **options)


def isolated_environment():
    """复制进程环境并移除应用字段，返回不受本机应用配置影响的子进程环境。"""
    return {key: value for key, value in os.environ.items() if key.lower() not in Settings.model_fields}


def copied_config(tmp_path):
    """建立公开副本形状的配置文件，返回独立 config.py 路径。"""
    target = tmp_path / "public-release" / "backend" / "app" / "config.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(BACKEND_ROOT / "app" / "config.py", target)
    return target


def test_public_copy_ignores_outer_and_current_directory_env(tmp_path):
    """复制配置到独立目录后，即使父目录和启动目录有 .env 也不能读取它们。"""
    target = copied_config(tmp_path)
    (tmp_path / ".env").write_text("EMBEDDING_API_KEY=outer-private\n", encoding="utf-8")
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text("EMBEDDING_MODEL=foreign-model\n", encoding="utf-8")
    script = """
import runpy, sys
module = runpy.run_path(sys.argv[1])
for settings in (module['get_settings'](), module['reset_settings']()):
    assert settings.embedding_api_key == ''
    assert settings.embedding_model == 'BAAI/bge-m3'
print('isolated')
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)], cwd=launch,
        env=isolated_environment(), text=True, capture_output=True, timeout=15, check=True)
    assert result.stdout.strip() == "isolated"


def test_owned_file_precedence_environment_override_and_reload(tmp_path):
    """本副本根配置优先于 backend 配置，进程注入优先；重载读取最新本地文件。"""
    target = copied_config(tmp_path)
    (target.parents[1] / ".env").write_text("EMBEDDING_MODEL=backend-model\nJWT_SECRET_KEY=backend-secret\n",
        encoding="utf-8")
    (target.parents[2] / ".env").write_text("JWT_SECRET_KEY=root-secret\nRERANK_MODEL=root-model\n", encoding="utf-8")
    environment = isolated_environment() | {"RERANK_MODEL": "injected-model"}
    script = """
import runpy, sys
from pathlib import Path
module = runpy.run_path(sys.argv[1])
settings = module['get_settings']()
assert settings.embedding_model == 'backend-model'
assert settings.jwt_secret_key == 'root-secret'
assert settings.rerank_model == 'injected-model'
(Path(sys.argv[1]).parents[1] / '.env').write_text('EMBEDDING_MODEL=new-model\\n', encoding='utf-8')
settings = module['reset_settings']()
assert settings.embedding_model == 'new-model'
assert settings.jwt_secret_key == 'root-secret'
assert settings.rerank_model == 'injected-model'
print('precedence')
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)], cwd=tmp_path,
        env=environment, text=True, capture_output=True, timeout=15, check=True)
    assert result.stdout.strip() == "precedence"


def test_api_imports_do_not_load_local_packages_or_mutate_hf_environment():
    """独立进程禁止导入本地模型包，验证 API 客户端与 FAISS 入口无 HF 环境副作用。"""
    script = """
import importlib.abc, os, sys
class RejectLocal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers'}:
            raise AssertionError('API imported local dependency: ' + fullname)
sys.meta_path.insert(0, RejectLocal())
keys = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'SSLKEYLOGFILE')
before = {key: os.environ.get(key) for key in keys}
from app.services.embedding_client import EmbeddingClient
from app.services.reranker import Reranker
import app.services.vector_store
from app.config import get_settings
config = get_settings().model_copy(update={'embedding_mode': 'api', 'rerank_mode': 'api',
    'embedding_dimension': 2, 'embedding_api_key': 'unit-test-key',
    'rerank_api_key': 'unit-test-key', 'vector_rerank_enabled': True})
class Response:
    status_code = 200
    def json(self):
        return {'data': [{'index': 0, 'embedding': [3, 4]}],
            'results': [{'index': 0, 'relevance_score': 0.8}]}
    def close(self):
        pass
def post(*args, **kwargs):
    return Response()
assert EmbeddingClient(mode='api').embed_texts([]) == []
assert Reranker().rerank('query', [], 1) == []
assert EmbeddingClient(config=config, post=post).embed_texts(['text']) == [[0.6, 0.8]]
assert Reranker(config=config, post=post).rerank('q', [{'text': 'text'}], 1)[0]['rerank_score'] == 0.8
assert {key: os.environ.get(key) for key in keys} == before
print('api-import-safe')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=BACKEND_ROOT,
        env=isolated_environment() | {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "SSLKEYLOGFILE": ""},
        text=True, capture_output=True, timeout=20, check=True)
    assert result.stdout.strip() == "api-import-safe"


@pytest.mark.parametrize("injected", [False, True])
def test_profile_activation_changes_runtime_and_persists_to_own_root(tmp_path, monkeypatch, injected):
    """实际激活入口立即切换模型并写回权威文件，普通重载仍遵守部署变量优先级。"""
    from app import config
    from app.services import llm_profile_service

    for key in list(os.environ):
        if key.lower() in Settings.model_fields:
            monkeypatch.delenv(key)
    backend_env = tmp_path / "backend.env"
    root_env = tmp_path / "root.env"
    backend_env.write_text("LLM_MODEL_NAME=backend-original\n", encoding="utf-8")
    root_env.write_text("LLM_MODEL_NAME=root-original\nJWT_SECRET_KEY=root-test-secret\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", (backend_env, root_env))
    monkeypatch.setattr(config, "_settings_instance", None)
    monkeypatch.setattr(llm_profile_service, "_ENV_PATH", root_env)
    monkeypatch.setattr(llm_profile_service, "_PROFILE_STORE_PATH", tmp_path / "profiles.json")
    monkeypatch.setattr(llm_profile_service, "reset_llm", lambda: None)
    if injected:
        monkeypatch.setenv("LLM_MODEL_NAME", "process-original")
    llm_profile_service._save_store({"active_id": "", "profiles": [{
        "id": "chosen", "name": "test-profile", "model_name": "selected-model",
        "base_url": "https://model.example/v1", "api_key": "test-only-token",
    }]})
    view = llm_profile_service.activate_profile("chosen")
    assert view.is_active
    assert config.get_settings().llm_model_name == "selected-model"
    assert config.get_settings().llm_api_key == "test-only-token"
    assert config.get_settings().jwt_secret_key == "root-test-secret"
    assert "selected-model" in root_env.read_text(encoding="utf-8")
    assert backend_env.read_text(encoding="utf-8") == "LLM_MODEL_NAME=backend-original\n"
    expected = "process-original" if injected else "selected-model"
    assert config.reset_settings().llm_model_name == expected
    with pytest.raises(ValueError, match="LLM 预设"):
        config.reset_settings(llm_overrides={"jwt_secret_key": "disallowed"})


def test_flat_container_config_never_reads_parent_environment(tmp_path):
    """容器的 app/app/config.py 只读取 app/.env，不把文件系统父目录当作项目。"""
    target = tmp_path / "runtime" / "app" / "config.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(BACKEND_ROOT / "app" / "config.py", target)
    (tmp_path / ".env").write_text("EMBEDDING_API_KEY=outer-private\n", encoding="utf-8")
    own_env = target.parents[1] / ".env"
    own_env.write_text("LLM_MODEL_NAME=own-model\n", encoding="utf-8")
    script = """
import runpy, sys
from pathlib import Path
module = runpy.run_path(sys.argv[1])
assert module['get_settings']().embedding_api_key == ''
assert module['get_settings']().llm_model_name == 'own-model'
assert module['get_settings_env_path']() == Path(sys.argv[1]).parents[1] / '.env'
print('flat-bounded')
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)], cwd=tmp_path,
        env=isolated_environment(), text=True, capture_output=True, timeout=15, check=True)
    assert result.stdout.strip() == "flat-bounded"


def test_hf_and_trace_settings_read_own_dotenv_without_global_injection(tmp_path):
    """缓存目录和记录大小沿用本副本 .env；API 导入不把这些字段写入进程环境。"""
    target = copied_config(tmp_path)
    for relative in ("services/hf_local.py", "middleware/trace_context.py"):
        destination = target.parent / relative
        destination.parent.mkdir(parents=True)
        shutil.copyfile(BACKEND_ROOT / "app" / relative, destination)
    (target.parents[1] / ".env").write_text(
        "HF_HOME=unit-hf\nHUGGINGFACE_HUB_CACHE=unit-hub\nOBS_TRACE_MAX_BYTES=101\n"
        "OBS_CALL_INPUT_MAX=202\nOBS_CALL_OUTPUT_MAX=303\n", encoding="utf-8")
    script = """
import os
from pathlib import Path
from app.services import hf_local
from app.middleware import trace_context
assert hf_local.get_hub_cache_dir() == Path('unit-hub')
assert (trace_context.OBS_TRACE_MAX_BYTES, trace_context.OBS_CALL_INPUT_MAX,
        trace_context.OBS_CALL_OUTPUT_MAX) == (101, 202, 303)
assert 'HF_HOME' not in os.environ and 'OBS_TRACE_MAX_BYTES' not in os.environ
hf_local.configure_offline_env()
assert os.environ['HF_HOME'] == 'unit-hf'
assert os.environ['HUGGINGFACE_HUB_CACHE'] == 'unit-hub'
print('own-consumers')
"""
    environment = isolated_environment()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=target.parents[1],
        env=environment, text=True, capture_output=True, timeout=15, check=True)
    assert result.stdout.strip() == "own-consumers"
