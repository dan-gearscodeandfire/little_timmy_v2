"""Unit tests for the LT-OS streamerpi-main toggle (services.check_streamerpi_server_status
and services.toggle_streamerpi_server). Pure-logic — no real TCP, no SSH, no Pi.
Run:
    .venv/bin/pytest tests/test_streamerpi_toggle.py -v
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

# LT-OS has flat-module config.py and services.py that collide with LT-root
# config.py and the LT-root services/ package. Any prior test in the same
# pytest session that loads main.py (which transitively imports LT-root
# config) will leave 'config' cached in sys.modules pointing at the wrong
# module, and our subsequent `import services` would then see that stale
# config when its `import config as cfg` runs.
#
# Load LT-OS config/services from disk under unique aliases, temporarily
# shimming sys.modules['config'] so services.py's `import config as cfg`
# resolves to the LT-OS variant during its exec_module pass.
_LT_OS_DIR = Path(__file__).resolve().parents[1] / "little_timmy_os"


def _load_ltos_module(name, deps=None):
    spec = importlib.util.spec_from_file_location(
        f"_ltos_{name}", str(_LT_OS_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_ltos_{name}"] = mod
    saved = {}
    if deps:
        for k, v in deps.items():
            saved[k] = sys.modules.get(k)
            sys.modules[k] = v
    try:
        spec.loader.exec_module(mod)
    finally:
        if deps:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    return mod


_ltos_config = _load_ltos_module("config")
services = _load_ltos_module("services", deps={"config": _ltos_config})


@pytest.fixture(autouse=True)
def _shim_config_in_sysmodules():
    """services.toggle_streamerpi_server() does `import config as cfg` at
    call time (not module load), so we have to keep the LT-OS config aliased
    in sys.modules['config'] for the duration of each test. Restore the
    prior cached value on teardown so other test files keep their own
    config."""
    saved = sys.modules.get("config")
    sys.modules["config"] = _ltos_config
    yield
    if saved is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = saved


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
