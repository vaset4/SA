"""默认部署与运行数据边界的安全契约测试。

作者：xxx
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
import pytest

from app.config import Settings
from app.runtime import lifespan as lifespan_module


REPO_ROOT = Path(__file__).resolve().parents[3]


def _protected_settings(**overrides: object) -> Settings:
    """
    构造一份凭证完整的受保护环境配置。

    Args:
        overrides: 需要覆盖的配置字段。

    Returns:
        Settings: 可直接交给启动守卫校验的配置。
    """
    values: dict[str, object] = {
        "app_env": "prod",
        "auth_enabled": True,
        "host": "0.0.0.0",
        "jwt_secret_key": "generated-strong-jwt-secret",
        "db_backend": "postgresql",
        "database_url": (
            "postgresql://studyassistant:generated-strong-db-secret"
            "@postgres:5432/studyassistant"
        ),
        "neo4j_enabled": True,
        "neo4j_uri": "bolt://neo4j:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "generated-strong-neo4j-secret",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("app_env", ["staging", "prod"])
@pytest.mark.parametrize(
    "case",
    [
        ({"jwt_secret_key": "change-me-in-production"}, "JWT secret"),
        ({"database_url": ""}, "DATABASE_URL"),
        (
            {"database_url": "postgresql://studyassistant:changeme@postgres:5432/studyassistant"},
            "DATABASE_URL",
        ),
        ({"neo4j_password": ""}, "Neo4j"),
        ({"neo4j_password": "neo4j_password"}, "Neo4j"),
    ],
    ids=["jwt", "database-missing", "database-weak", "neo4j-missing", "neo4j-weak"],
)
def test_protected_environment_rejects_missing_or_weak_secrets(
    monkeypatch: pytest.MonkeyPatch,
    app_env: str,
    case: tuple[dict[str, object], str],
) -> None:
    """
    验证 staging/prod 在资源初始化前拒绝空凭证和已知弱凭证。

    Args:
        monkeypatch: pytest 提供的替换工具。
        app_env: 当前受保护部署环境。
        case: 本场景覆盖的弱配置和预期配置名称。

    Returns:
        None.
    """
    overrides, message = case
    settings = _protected_settings(app_env=app_env, **overrides)
    monkeypatch.setattr(lifespan_module, "get_settings", lambda: settings)

    with pytest.raises((HTTPException, ValueError), match=message):
        lifespan_module.validate_startup_auth_configuration()


def test_disabled_neo4j_does_not_require_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证关闭 Neo4j 后不要求无实际用途的图数据库凭证。

    Args:
        monkeypatch: pytest 提供的替换工具。

    Returns:
        None.
    """
    settings = _protected_settings(neo4j_enabled=False, neo4j_password="")
    monkeypatch.setattr(lifespan_module, "get_settings", lambda: settings)

    lifespan_module.validate_startup_auth_configuration()


def test_local_environment_allows_explicit_development_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证仅绑定本机的 local 环境可以使用显式开发凭证。

    Args:
        monkeypatch: pytest 提供的替换工具。

    Returns:
        None.
    """
    settings = _protected_settings(
        app_env="local",
        host="127.0.0.1",
        jwt_secret_key="change-me-in-production",
        database_url="postgresql://studyassistant:local-dev@localhost:5432/studyassistant",
        neo4j_password="local-dev",
    )
    monkeypatch.setattr(lifespan_module, "get_settings", lambda: settings)

    lifespan_module.validate_startup_auth_configuration()


def test_database_config_error_does_not_echo_connection_secret() -> None:
    """
    验证数据库连接串解析失败时不会在异常中回显凭证。

    Returns:
        None.
    """
    secret_marker = "private-database-secret-marker"
    settings = _protected_settings(
        database_url=(
            f"postgresql://studyassistant:{secret_marker}@[invalid/studyassistant"
        )
    )

    with pytest.raises(ValueError) as exc_info:
        lifespan_module.validate_startup_infrastructure_configuration(settings)

    assert secret_marker not in str(exc_info.value)


def test_lifespan_validates_before_resolving_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证安全配置失败时不会解析或初始化任何运行时依赖。

    Args:
        monkeypatch: pytest 提供的替换工具。

    Returns:
        None.
    """
    events: list[str] = []

    # 1. 构造必然失败的安全守卫，以及任何调用都应被视为错误的 runtime getter
    def reject_configuration() -> None:
        events.append("validate")
        raise ValueError("invalid startup security configuration")

    def resolve_runtime() -> object:
        events.append("resolve-runtime")
        return object()

    # 2. 将所有运行时解析入口替换为同一事件记录器，覆盖完整依赖集合
    monkeypatch.setattr(
        lifespan_module,
        "validate_startup_auth_configuration",
        reject_configuration,
    )
    for getter_name in (
        "get_runtime_container",
        "get_db_runtime",
        "get_conversation_runtime",
        "get_agent_runtime",
    ):
        monkeypatch.setattr(lifespan_module, getter_name, resolve_runtime)

    app = FastAPI()
    app.state = SimpleNamespace()
    context = lifespan_module.build_app_lifespan()(app)

    # 3. 守卫失败后不应出现任何 runtime 解析事件
    with pytest.raises(ValueError, match="startup security"):
        asyncio.run(context.__aenter__())

    assert events == ["validate"]


def test_compose_requires_secrets_and_binds_host_ports_to_loopback() -> None:
    """
    验证默认 Compose 不含公开弱凭证，数据和应用端口只绑定本机。

    Returns:
        None.
    """
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    required_variables = (
        "POSTGRES_PASSWORD",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "APP_ENV",
        "JWT_SECRET_KEY",
    )
    loopback_ports = (5432, 9000, 9001, 19530, 9091, 7474, 7687, 8000, 3000)

    # 1. 所有凭证都必须通过 Compose 的必填插值显式注入
    for variable in required_variables:
        assert f"${{{variable}:?" in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?" in compose
    assert "DATABASE_URL: postgresql://studyassistant:${POSTGRES_PASSWORD:?" in compose
    assert "MINIO_ROOT_USER: ${MINIO_ROOT_USER:?" in compose
    assert "MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:?" in compose
    assert "MINIO_ACCESS_KEY_ID: ${MINIO_ROOT_USER:?" in compose
    assert "MINIO_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD:?" in compose
    assert "NEO4J_USERNAME: ${NEO4J_USERNAME:?" in compose
    assert "NEO4J_PASSWORD: ${NEO4J_PASSWORD:?" in compose
    assert "path: ./backend/.env" in compose
    assert "required: false" in compose
    assert ":-changeme" not in compose
    assert "minioadmin" not in compose.lower()

    # 2. 宿主端口统一绑定 loopback，容器内部监听地址不在此契约内
    for port in loopback_ports:
        assert f'"127.0.0.1:{port}:{port}"' in compose


def test_environment_example_contains_only_blank_secret_slots() -> None:
    """
    验证环境变量示例列出必需凭证，但不提供可直接使用的固定密码。

    Returns:
        None.
    """
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    secret_variables = (
        "JWT_SECRET_KEY",
        "POSTGRES_PASSWORD",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "NEO4J_PASSWORD",
        "LLM_API_KEY",
        "EMBEDDING_API_KEY",
        "RERANK_API_KEY",
        "MINERU_API_TOKEN",
    )

    for variable in secret_variables:
        assert f"{variable}=\n" in example
    for weak_value in ("123456", "changeme", "change-me-in-production", "minioadmin"):
        assert weak_value not in example.lower()


def test_runtime_data_paths_are_ignored_without_custom_merge_driver() -> None:
    """
    验证新增会话正文与 checkpoint 被忽略，且不再配置 journal 合并驱动。

    Returns:
        None.
    """
    root_ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert ".trellis/workspace/**" in root_ignore
    assert "backend/checkpoints/**" in root_ignore
    assert "/public-release/" in root_ignore
    assert "journal-*.md merge=union" not in attributes
