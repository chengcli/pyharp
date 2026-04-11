import types

from pyharp.spectra import shared_cli


def test_process_pool_context_uses_spawn_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(shared_cli.sys, "platform", "darwin")
    calls = []

    def fake_get_context(method):
        calls.append(method)
        return types.SimpleNamespace(method=method)

    monkeypatch.setattr(shared_cli.mp, "get_context", fake_get_context)

    ctx = shared_cli.process_pool_context()

    assert ctx.method == "spawn"
    assert calls == ["spawn"]


def test_process_pool_context_prefers_fork_when_available(monkeypatch) -> None:
    monkeypatch.setattr(shared_cli.sys, "platform", "linux")
    monkeypatch.setattr(shared_cli.mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    calls = []

    def fake_get_context(method):
        calls.append(method)
        return types.SimpleNamespace(method=method)

    monkeypatch.setattr(shared_cli.mp, "get_context", fake_get_context)

    ctx = shared_cli.process_pool_context()

    assert ctx.method == "fork"
    assert calls == ["fork"]
