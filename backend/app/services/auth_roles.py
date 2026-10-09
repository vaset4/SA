"""User role resolution for authenticated accounts."""

from __future__ import annotations

from typing import Any

from app.config import get_settings

USER_ROLE = "user"
DEVELOPER_ROLE = "developer"
ADMIN_ROLE = "admin"
KNOWN_ROLES = {USER_ROLE, DEVELOPER_ROLE, ADMIN_ROLE}
OBSERVABILITY_ROLES = {DEVELOPER_ROLE, ADMIN_ROLE}


def _settings():
    return get_settings()


def normalize_role(role: Any) -> str:
    value = str(role or USER_ROLE).strip().lower()
    return value if value in KNOWN_ROLES else USER_ROLE


def _configured_usernames(raw_value: Any) -> set[str]:
    return {
        item.strip().lower()
        for item in str(raw_value or "").split(",")
        if item.strip()
    }


def configured_role_for_username(username: str, settings=None) -> str | None:
    active_settings = settings or _settings()
    normalized_username = str(username or "").strip().lower()
    if not normalized_username:
        return None
    admin_usernames = _configured_usernames(getattr(active_settings, "admin_usernames", ""))
    if normalized_username in admin_usernames:
        return ADMIN_ROLE
    developer_usernames = _configured_usernames(
        getattr(active_settings, "developer_usernames", "")
    )
    if normalized_username in developer_usernames:
        return DEVELOPER_ROLE
    return None


def resolve_user_role(user: dict[str, Any], settings=None) -> str:
    configured_role = configured_role_for_username(
        user.get("username", ""),
        settings=settings,
    )
    if configured_role:
        return configured_role
    return normalize_role(user.get("role"))


def can_view_observability(user: dict[str, Any]) -> bool:
    return normalize_role(user.get("role")) in OBSERVABILITY_ROLES
