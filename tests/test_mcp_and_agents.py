"""Tests for SpoolGenie's zero-dependency MCP stdio server and AI-Agent auto-deploy manifests."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from spoolgenie.mcp_server import build_agent_configs, handle_rpc_message, serve_stdio

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agent_discovery_files_and_installers_exist() -> None:
    required = [
        "AGENTS.md",
        "CLAUDE.md",
        "llms.txt",
        ".mcp.json",
        "skills/spoolgenie/SKILL.md",
        "scripts/install.ps1",
        "scripts/install.sh",
    ]
    for rel in required:
        p = REPO_ROOT / rel
        assert p.exists(), f"Missing AI-agent integration file: {rel}"
        assert p.stat().st_size > 40

    mcp_manifest = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    assert "spoolgenie" in mcp_manifest["mcpServers"]


def test_mcp_rpc_initialize_tools_list_and_doctor() -> None:
    init_resp = handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "pytest"}},
        }
    )
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "spoolgenie"

    # Notification (no id) must return None
    assert handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    list_resp = handle_rpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert {
        "spoolgenie_doctor",
        "spoolgenie_search_makerworld",
        "spoolgenie_create_design",
        "spoolgenie_build_3d",
        "spoolgenie_colorize_mesh",
        "spoolgenie_list_projects",
        "spoolgenie_lan_info",
    } <= tool_names

    doc_resp = handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "spoolgenie_doctor", "arguments": {}},
        }
    )
    assert doc_resp is not None
    assert doc_resp["result"]["isError"] is False
    payload = json.loads(doc_resp["result"]["content"][0]["text"])
    assert payload["silent_subject_priors"] >= 170
    assert payload["engineering_kb_entries"] >= 19


def test_mcp_stdio_loop_and_end_to_end_design_build(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPOOLGENIE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("SPOOLGENIE_LLM_PROVIDER", "fallback")
    monkeypatch.setenv("SPOOLGENIE_IMAGE_BACKEND", "studio")
    monkeypatch.setenv("SPOOLGENIE_MESH_BACKEND", "relief")

    requests = [
        {"jsonrpc": "2.0", "id": 10, "method": "ping"},
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "spoolgenie_create_design",
                "arguments": {
                    "prompt": "Desktop headphone stand with cable slot, 210mm tall, black and orange, no AMS",
                    "project_id": "mcp_demo",
                    "ams_units": 0,
                    "joint_type": "dovetail",
                    "render_2d": True,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "spoolgenie_build_3d",
                "arguments": {"project_id": "mcp_demo", "run_slicer": False},
            },
        },
    ]
    inp = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    out = io.StringIO()
    serve_stdio(stdin=inp, stdout=out)

    responses = [json.loads(line) for line in out.getvalue().strip().splitlines()]
    assert len(responses) == 3
    assert responses[0]["id"] == 10 and responses[0]["result"] == {}

    design_data = json.loads(responses[1]["result"]["content"][0]["text"])
    assert design_data["project_id"] == "mcp_demo"
    assert Path(design_data["concept_image_path"]).exists()

    build_data = json.loads(responses[2]["result"]["content"][0]["text"])
    assert build_data["dfm"]["watertight"] is True
    assert build_data["dfm"]["effective_multicolor_mode"] == "split_no_ams"
    assert Path(build_data["exports"]["threemf_path"]).exists()
    assert Path(build_data["exports"]["split_plates_3mf_path"]).exists()


def test_mcp_subprocess_smoke_and_agent_config() -> None:
    cfg = build_agent_configs()
    assert "spoolgenie" in cfg["mcp_json"]["mcpServers"]
    assert "claude mcp add spoolgenie" in cfg["claude_code_cli"]
    assert "codex mcp add spoolgenie" in cfg["codex_cli"]

    proc = subprocess.run(
        [sys.executable, "-m", "spoolgenie.mcp_server"],
        input=json.dumps({"jsonrpc": "2.0", "id": 99, "method": "tools/list"}) + "\n",
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    assert reply["id"] == 99
    assert len(reply["result"]["tools"]) == 7
