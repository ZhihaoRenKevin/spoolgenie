"""Tests for PWA endpoints, LAN QR pairing, and cross-platform Desktop App launcher."""

from __future__ import annotations

from fastapi.testclient import TestClient

from spoolgenie.desktop import find_available_port, launch_desktop
from spoolgenie.web.app import app, set_server_runtime


def test_pwa_manifest_service_worker_and_icon() -> None:
    client = TestClient(app)

    r_man = client.get("/manifest.json")
    assert r_man.status_code == 200
    data = r_man.json()
    assert data["short_name"] == "SpoolGenie"
    assert data["display"] == "standalone"

    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert "self.addEventListener" in r_sw.text

    r_ico = client.get("/icon.svg")
    assert r_ico.status_code == 200
    assert "<svg" in r_ico.text


def test_lan_info_returns_qr_svg_and_flags() -> None:
    client = TestClient(app)
    set_server_runtime("0.0.0.0", 8765)

    r = client.get("/api/lan-info")
    assert r.status_code == 200
    info = r.json()
    assert info["lan_enabled"] is True
    assert info["lan_url"].startswith("http://")
    assert "<svg" in info["qr_svg"]
    assert info["cli_hint"].startswith("spoolgenie serve --lan")


def test_desktop_dry_run_starts_and_stops_cleanly() -> None:
    free_port = find_available_port(preferred=0)
    assert free_port > 0
    res = launch_desktop(port=free_port, lan=True, dry_run=True)
    assert res["ready"] is True
    assert res["bind_host"] == "0.0.0.0"
    assert str(res["local_url"]).startswith("http://127.0.0.1:")
