from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit and integration tests independent of external services."""

    class GuardedSocket(socket.socket):
        def connect(self, address: object) -> None:
            raise AssertionError(f"Network access is forbidden during tests: {address!r}")

        def connect_ex(self, address: object) -> int:
            raise AssertionError(f"Network access is forbidden during tests: {address!r}")

    monkeypatch.setattr(socket, "socket", GuardedSocket)
