"""默认 API 部署和显式本地依赖的配置契约。作者：xxx。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_default_api_deployment_does_not_install_or_mount_local_models() -> None:
    """无参数；验证默认 API、空密钥、无权重挂载及可选本地依赖，无返回值。"""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    requirements = (ROOT / "backend/requirements.txt").read_text(encoding="utf-8")
    local = (ROOT / "backend/requirements-local.txt").read_text(encoding="utf-8")
    dockerfile = (ROOT / "backend/Dockerfile").read_text(encoding="utf-8")
    for prefix in ("EMBEDDING", "RERANK"):
        assert f"{prefix}_MODE: ${{{prefix}_MODE:-api}}" in compose
        assert f"{prefix}_API_KEY: ${{{prefix}_API_KEY:-}}" in compose
    assert "/app/models/" not in compose
    assert "sentence-transformers" not in requirements
    assert "sentence-transformers==5.1.1" in local
    assert "ARG INSTALL_LOCAL_MODELS=false" in dockerfile
    assert "pip install -r requirements-local.txt" in dockerfile
    assert "tokenizers==0.22.2" in requirements
    assert "python scripts/prepare_chunk_tokenizer.py" in dockerfile


def test_local_override_installs_local_dependencies_and_mounts_existing_weights() -> None:
    """无参数；验证本地覆盖显式启用依赖、模型模式和只读已有目录，无返回值。"""
    override = (ROOT / "docker-compose.local-models.yml").read_text(encoding="utf-8")
    assert 'INSTALL_LOCAL_MODELS: "true"' in override
    for prefix in ("EMBEDDING", "RERANK"):
        assert f"{prefix}_MODE: local" in override
    assert override.count("read_only: true") == 2
    assert override.count("create_host_path: false") == 2
