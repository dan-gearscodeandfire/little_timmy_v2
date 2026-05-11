"""Unit tests for the LT-OS streamerpi-main toggle (services.check_streamerpi_server_status
and services.toggle_streamerpi_server). Pure-logic — no real TCP, no SSH, no Pi.
Run:
    .venv/bin/pytest tests/test_streamerpi_toggle.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "little_timmy_os"))

import services  # noqa: E402


@pytest.mark.asyncio
async def test_status_running_short_circuits_when_8080_open(monkeypatch):
    """When port 8080 accepts the connection the function returns immediately
    without probing SSH — running implies reachable."""
    calls = []

    async def fake_connect(host, port, timeout):
        calls.append(port)
        return port == 8080

    monkeypatch.setattr(services, "_tcp_connect", fake_connect)
    result = await services.check_streamerpi_server_status()
    assert result == {"reachable": True, "running": True}
    assert calls == [8080]


@pytest.mark.asyncio
async def test_status_stopped_but_host_reachable(monkeypatch):
    """8080 refused but SSH up = service stopped on a live host."""
    async def fake_connect(host, port, timeout):
        return port == 22

    monkeypatch.setattr(services, "_tcp_connect", fake_connect)
    result = await services.check_streamerpi_server_status()
    assert result == {"reachable": True, "running": False}


@pytest.mark.asyncio
async def test_status_host_unreachable(monkeypatch):
    """Both probes fail — host is offline."""
    async def fake_connect(host, port, timeout):
        return False

    monkeypatch.setattr(services, "_tcp_connect", fake_connect)
    result = await services.check_streamerpi_server_status()
    assert result == {"reachable": False, "running": False}


@pytest.mark.asyncio
async def test_toggle_enable_bails_before_ssh_when_unreachable(monkeypatch):
    """When the user clicks the toggle on but the Pi is offline, the SSH
    subprocess must NOT be invoked — the dashboard would otherwise sit on
    a 3 s ConnectTimeout per attempt. The function returns immediately
    with reachable=False."""
    async def fake_connect(host, port, timeout):
        return False

    async def fake_subprocess_exec(*args, **kwargs):
        raise AssertionError("subprocess should not have run on unreachable host")

    monkeypatch.setattr(services, "_tcp_connect", fake_connect)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    result = await services.toggle_streamerpi_server(True)
    assert result["running"] is False
    assert result["reachable"] is False
    assert "unreachable" in (result.get("error") or "").lower()
