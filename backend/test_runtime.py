import types

from app.runtime import (
    configure_windows_event_loop_policy,
    create_compatible_event_loop,
    get_uvicorn_loop_config,
)


class FakeAsyncio:
    class WindowsSelectorEventLoopPolicy:
        pass

    class SelectorEventLoop:
        pass

    def __init__(self) -> None:
        self.policy = None
        self.new_loop = object()

    def set_event_loop_policy(self, policy) -> None:
        self.policy = policy

    def new_event_loop(self):
        return self.new_loop


def test_configure_windows_event_loop_policy_on_windows() -> None:
    fake_asyncio = FakeAsyncio()

    changed = configure_windows_event_loop_policy(
        asyncio_module=fake_asyncio,
        platform_name="win32",
    )

    assert changed is True
    assert isinstance(fake_asyncio.policy, fake_asyncio.WindowsSelectorEventLoopPolicy)


def test_configure_windows_event_loop_policy_skips_non_windows() -> None:
    fake_asyncio = FakeAsyncio()

    changed = configure_windows_event_loop_policy(
        asyncio_module=fake_asyncio,
        platform_name="linux",
    )

    assert changed is False
    assert fake_asyncio.policy is None


def test_create_compatible_event_loop_uses_selector_on_windows() -> None:
    fake_asyncio = FakeAsyncio()

    loop = create_compatible_event_loop(
        asyncio_module=fake_asyncio,
        platform_name="win32",
    )

    assert isinstance(loop, fake_asyncio.SelectorEventLoop)


def test_create_compatible_event_loop_uses_default_elsewhere() -> None:
    fake_asyncio = FakeAsyncio()

    loop = create_compatible_event_loop(
        asyncio_module=fake_asyncio,
        platform_name="linux",
    )

    assert loop is fake_asyncio.new_loop


def test_get_uvicorn_loop_config_uses_factory_on_windows() -> None:
    assert get_uvicorn_loop_config(platform_name="win32") is create_compatible_event_loop


def test_get_uvicorn_loop_config_keeps_auto_elsewhere() -> None:
    assert get_uvicorn_loop_config(platform_name="linux") == "auto"
