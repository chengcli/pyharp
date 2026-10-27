import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any spectra test that opens a real network connection (HITRAN, CIA downloads).

    pytest.fail raises an exception that ``except Exception`` in library code does not catch.
    """

    def _fail(*args, **kwargs):
        pytest.fail(f"spectra tests run offline; a real network call was made with {args!r}")

    monkeypatch.setattr(socket.socket, "connect", _fail)
    monkeypatch.setattr(socket.socket, "connect_ex", _fail)
    monkeypatch.setattr(socket, "create_connection", _fail)
    monkeypatch.setattr(socket, "getaddrinfo", _fail)
