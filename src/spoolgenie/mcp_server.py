"""Zero-dependency Model Context Protocol (MCP) stdio server for SpoolGenie.

Allows any mainstream AI agent (Claude Code, Claude Desktop, OpenAI Codex, Cursor,
Antigravity, Windsurf, Cline, OpenClaw, Gemini CLI) to directly inspect hardware,
search MakerWorld, design 2D concepts, and build print-ready Bambu Lab .3mf/.stl/.step files.

Protocol: MCP JSON-RPC 2.0 over newline-delimited stdio (stdout = JSON-RPC only, stderr = logs).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

from spoolgenie import __version__
from spoolgenie.config import get_settings
from spoolgenie.doctor import run_doctor
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG
from spoolgenie.knowledge.kb import PrintKnowledgeBase
from spoolgenie.pipeline import SpoolGenieSession
from spoolgenie.print_prep.bambu_3mf import export_painted_bambu_3mf
from spoolgenie.print_prep.color_quantizer import quantize_mesh_to_ams_slots
from spoolgenie.print_prep.repair import repair_single_mesh
from spoolgenie.reference.makerworld import search_makerworld
from spoolgenie.schemas import FilamentSlot, JointType, MulticolorMode

PROTOCOL_VERSION = "2024-11-05"

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "spoolgenie_doctor",
        "description": (
            "Inspect local OS, GPU (NVIDIA RTX 5090 sm_120 / CUDA / Apple Silicon MPS / CPU), "
            "Ollama models, Bambu/Orca slicers, and SpoolGenie silent prior counts."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "spoolgenie_search_makerworld",
        "description": "Search live public MakerWorld 3D printing models by keyword and return URLs, titles, and thumbnails.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords (e.g. 'wall-e pen holder', 'gridfinity organizer')"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12, "default": 6},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "spoolgenie_create_design",
        "description": (
            "Create or update a SpoolGenie 3D-printing project from natural language. "
            "Automatically matches 171+ silent character/animal/plant priors and 19 MakerWorld CAD engineering standards, "
            "sets Bambu AMS color budget (0-4 AMS units) & glue-free mechanical joints, and optionally renders a 2D confirmation preview."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to design, desired function, dimensions (mm), and colors"},
                "project_id": {"type": "string", "description": "Optional existing project ID to continue iterating on"},
                "printer_id": {
                    "type": "string",
                    "description": "Bambu printer profile (bambu_p1s, bambu_x1c, bambu_a1, bambu_a1_mini, bambu_h2d)",
                    "default": "bambu_p1s",
                },
                "ams_units": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                    "description": "Number of Bambu AMS units (0 = No-AMS glue-free split, 1 = <=4 colors, ..., 4 = <=16 colors)",
                },
                "joint_type": {
                    "type": "string",
                    "enum": ["dovetail", "snap_pin", "press_tenon", "none"],
                    "description": "Glue-free mechanical joint type when splitting multi-color or oversized parts",
                },
                "render_2d": {
                    "type": "boolean",
                    "default": True,
                    "description": "Render a 2D concept confirmation image (PNG) immediately",
                },
                "makerworld_url": {
                    "type": "string",
                    "description": "Optional MakerWorld model URL to attach as structural/palette reference",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    {
        "name": "spoolgenie_build_3d",
        "description": (
            "Confirm a SpoolGenie project and build watertight 3D print files: Bambu Lab multicolor .3mf, "
            "No-AMS glue-free split-plate .3mf (when ams_units=0), universal .stl, parametric .step (CAD route), "
            "interactive .glb preview, 4-angle turntable PNG, and full DFM report."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project ID returned by spoolgenie_create_design"},
                "run_slicer": {"type": "boolean", "default": False, "description": "Invoke local Bambu Studio / OrcaSlicer CLI if installed"},
                "force_mesh_backend": {"type": "string", "description": "Optional override: hunyuan3d | relief | http"},
            },
            "required": ["project_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "spoolgenie_colorize_mesh",
        "description": "Quantize any existing .glb/.obj/.stl mesh into a Bambu Studio painted multicolor .3mf with CIELAB color budgeting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mesh_path": {"type": "string", "description": "Absolute or relative path to input .glb/.obj/.stl file"},
                "output_3mf": {"type": "string", "description": "Output .3mf path"},
                "hex_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of up to 16 AMS filament HEX colors (e.g. ['#2B2D30', '#F36C21', '#F2EFE9'])",
                },
                "reference_image": {"type": "string", "description": "Optional 2D image path to project onto untextured meshes"},
            },
            "required": ["mesh_path", "output_3mf"],
            "additionalProperties": False,
        },
    },
    {
        "name": "spoolgenie_list_projects",
        "description": "List recent SpoolGenie projects in the local workspace along with their stages and exported files.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15}},
            "additionalProperties": False,
        },
    },
    {
        "name": "spoolgenie_lan_info",
        "description": "Return local Wi-Fi LAN URL and CLI commands to launch the Web/Desktop/Mobile PWA Studio for human visual review.",
        "inputSchema": {
            "type": "object",
            "properties": {"port": {"type": "integer", "default": 8765}},
            "additionalProperties": False,
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a SpoolGenie MCP tool call and return a JSON-serializable dict."""
    args = arguments or {}
    settings = get_settings()

    if name == "spoolgenie_doctor":
        doc = run_doctor(settings)
        kb = PrintKnowledgeBase(settings.workspace_dir / "knowledge_store.jsonl")
        return {
            "version": __version__,
            "hardware": doc.to_dict(),
            "silent_subject_priors": len(CHARACTER_CATALOG),
            "engineering_kb_entries": len(kb.entries),
            "workspace_dir": str(settings.workspace_dir.resolve()),
        }

    if name == "spoolgenie_search_makerworld":
        query = str(args["query"])
        limit = int(args.get("limit", 6))
        refs = search_makerworld(query, max_results=min(max(limit, 1), 12))
        return {"query": query, "count": len(refs), "results": [r.model_dump() for r in refs]}

    if name == "spoolgenie_create_design":
        session = SpoolGenieSession(project_id=args.get("project_id"), settings=settings)
        if args.get("printer_id"):
            session.state.spec.printer_id = str(args["printer_id"])
        if args.get("ams_units") is not None:
            ams_n = int(args["ams_units"])
            session.state.spec.ams_units = ams_n
            if ams_n == 0 and session.state.spec.multicolor_mode == MulticolorMode.AUTO:
                session.state.spec.multicolor_mode = MulticolorMode.SPLIT_NO_AMS
        if args.get("joint_type"):
            session.state.spec.joint_type = JointType(str(args["joint_type"]))
        session.save()

        if args.get("makerworld_url"):
            session.add_reference_url(str(args["makerworld_url"]))

        res = session.chat(str(args["prompt"]), auto_render_2d=bool(args.get("render_2d", True)))
        concept = res.get("concept")
        return {
            "project_id": session.state.project_id,
            "stage": session.state.stage.value,
            "reply": res["reply"],
            "spec": session.state.spec.model_dump(mode="json"),
            "concept_image_path": str(Path(concept.image_path).resolve()) if concept else None,
            "concept_backend": concept.backend if concept else None,
            "next_step": f"Inspect concept image or call spoolgenie_build_3d(project_id='{session.state.project_id}') to export .3mf/.stl/.step.",
        }

    if name == "spoolgenie_build_3d":
        pid = str(args["project_id"])
        session = SpoolGenieSession(project_id=pid, settings=settings)
        state = session.confirm_and_build(
            force_mesh_backend=args.get("force_mesh_backend"),
            run_slicer=bool(args.get("run_slicer", False)),
        )
        return {
            "project_id": state.project_id,
            "stage": state.stage.value,
            "title": state.spec.title,
            "route": state.spec.route.value,
            "dfm": state.dfm.model_dump(mode="json") if state.dfm else None,
            "exports": state.exports.model_dump(mode="json") if state.exports else None,
        }

    if name == "spoolgenie_colorize_mesh":
        import trimesh

        mesh_path = Path(str(args["mesh_path"]))
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        raw = trimesh.load(mesh_path, force="mesh")
        rep = repair_single_mesh(raw)
        hex_list = args.get("hex_colors") or ["#2B2D30", "#F36C21", "#F2EFE9", "#1E4C8A"]
        filaments = [
            FilamentSlot(slot=i + 1, name=f"AMS Slot {i + 1}", hex_color=str(hx))
            for i, hx in enumerate(hex_list[:16])
        ]
        ref_img = Path(str(args["reference_image"])) if args.get("reference_image") else None
        face_slots, active = quantize_mesh_to_ams_slots(rep.mesh, filaments, reference_image_path=ref_img)
        out = export_painted_bambu_3mf(
            rep.mesh, face_slots, active, Path(str(args["output_3mf"])), model_title=mesh_path.stem
        )
        return {
            "output_3mf": str(out.resolve()),
            "active_slots": len(active),
            "faces": int(len(rep.mesh.faces)),
            "watertight": bool(rep.mesh.is_watertight),
        }

    if name == "spoolgenie_list_projects":
        ws = settings.workspace_dir
        limit = int(args.get("limit", 15))
        items: list[dict[str, Any]] = []
        if ws.exists():
            for p in sorted(ws.glob("*/project.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
                try:
                    sess = SpoolGenieSession(project_id=p.parent.name, settings=settings)
                    items.append(
                        {
                            "project_id": sess.state.project_id,
                            "title": sess.state.spec.title,
                            "stage": sess.state.stage.value,
                            "updated_at": sess.state.updated_at,
                            "threemf_path": sess.state.exports.threemf_path if sess.state.exports else None,
                        }
                    )
                except Exception:
                    continue
        return {"workspace_dir": str(ws.resolve()), "count": len(items), "projects": items}

    if name == "spoolgenie_lan_info":
        from spoolgenie.web.app import detect_lan_ips

        port = int(args.get("port", 8765))
        lan_ips = detect_lan_ips()
        return {
            "local_url": f"http://127.0.0.1:{port}",
            "lan_url": f"http://{lan_ips[0]}:{port}",
            "lan_ips": lan_ips,
            "start_web_lan_cmd": f"spoolgenie serve --lan --port {port}",
            "start_desktop_cmd": f"spoolgenie desktop --port {port}",
        }

    raise ValueError(f"Unknown SpoolGenie MCP tool: {name}")


def handle_rpc_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 MCP message and return a response dict (or None for notifications)."""
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    # JSON-RPC notifications have no id and expect no reply
    if req_id is None:
        return None

    if method == "initialize":
        client_ver = params.get("protocolVersion") or PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_ver,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "spoolgenie", "version": __version__},
                "instructions": (
                    "SpoolGenie turns natural-language prompts and MakerWorld references into print-ready "
                    "Bambu Lab AMS multicolor .3mf, No-AMS glue-free split .3mf, .stl, and .step files. "
                    "Typical workflow: spoolgenie_create_design -> show 2D concept -> spoolgenie_build_3d."
                ),
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}

    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}

    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": []}}

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        tool_args = params.get("arguments") or {}
        try:
            payload = execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            err_payload = {"error": str(exc), "tool": tool_name}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err_payload, ensure_ascii=False)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Run the MCP JSON-RPC 2.0 stdio loop."""
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    for raw_line in inp:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            out.write(json.dumps(err) + "\n")
            out.flush()
            continue
        try:
            resp = handle_rpc_message(msg)
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            resp = {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "error": {"code": -32603, "message": str(exc)},
            }
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()


def build_agent_configs(python_exe: str | None = None) -> dict[str, Any]:
    """Generate ready-to-use MCP configuration snippets for all major AI Agents."""
    py = python_exe or sys.executable
    mcp_entry = {"command": py, "args": ["-m", "spoolgenie.mcp_server"]}
    return {
        "mcp_json": {"mcpServers": {"spoolgenie": mcp_entry}},
        "claude_code_cli": f'claude mcp add spoolgenie -- "{py}" -m spoolgenie.mcp_server',
        "codex_cli": f'codex mcp add spoolgenie -- "{py}" -m spoolgenie.mcp_server',
        "cursor_windsurf_antigravity_claude_desktop": {"mcpServers": {"spoolgenie": mcp_entry}},
        "skill_path": str((Path(__file__).resolve().parents[2] / "skills" / "spoolgenie" / "SKILL.md")),
    }


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
