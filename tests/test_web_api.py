"""Tests for FastAPI Web Studio endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from spoolgenie.web import app as web_mod


def test_web_api_full_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPOOLGENIE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("SPOOLGENIE_LLM_PROVIDER", "fallback")
    monkeypatch.setenv("SPOOLGENIE_IMAGE_BACKEND", "studio")

    client = TestClient(web_mod.app)

    r_meta = client.get("/api/meta")
    assert r_meta.status_code == 200
    assert "bambu_p1s" in r_meta.json()["printers"]

    r_new = client.post("/api/projects")
    assert r_new.status_code == 200
    pid = r_new.json()["project_id"]

    r_chat = client.post(
        f"/api/projects/{pid}/chat",
        json={"message": "做一个手机支架，82x95x115mm，黑和橙双色，1台AMS"},
    )
    assert r_chat.status_code == 200
    proj = r_chat.json()["project"]
    assert proj["concepts"][-1]["image_url"].startswith(f"/api/projects/{pid}/files/")

    # Fetch 2D preview PNG
    r_img = client.get(proj["concepts"][-1]["image_url"])
    assert r_img.status_code == 200
    assert r_img.content[:4] == b"\x89PNG"

    # Confirm and build 3D + 3MF
    r_build = client.post(f"/api/projects/{pid}/confirm-build", json={"run_slicer": False})
    assert r_build.status_code == 200
    built = r_build.json()
    assert built["dfm"]["watertight"] is True
    assert built["exports"]["urls"]["threemf"] is not None

    r_3mf = client.get(built["exports"]["urls"]["threemf"])
    assert r_3mf.status_code == 200
    assert r_3mf.content[:2] == b"PK"  # valid ZIP/3MF
