"""FastAPI backend powering SpoolGenie Web UI (shares SpoolGenieSession with CLI)."""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spoolgenie import __version__
from spoolgenie.config import (
    BAMBU_OFFICIAL_FILAMENTS,
    BAMBU_PLA_SWATCHES,
    PRINTER_PROFILES,
    get_settings,
)
from spoolgenie.doctor import run_doctor
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG, list_local_loras
from spoolgenie.knowledge.kb import CHARACTER_STYLE_PRESETS, PrintKnowledgeBase
from spoolgenie.pipeline import SpoolGenieSession
from spoolgenie.print_prep.color_quantizer import match_canon_to_owned_filaments
from spoolgenie.reference.makerworld import search_makerworld
from spoolgenie.schemas import FilamentSlot

STATIC_DIR = Path(__file__).parent / "static"
SERVER_RUNTIME: dict[str, Any] = {
    "host": os.environ.get("SPOOLGENIE_BIND_HOST", "127.0.0.1"),
    "port": int(os.environ.get("SPOOLGENIE_BIND_PORT", "8765")),
}


def set_server_runtime(host: str, port: int) -> None:
    """Record active bind host/port so /api/lan-info reports exact LAN accessibility."""
    SERVER_RUNTIME["host"] = host
    SERVER_RUNTIME["port"] = int(port)


def detect_lan_ips() -> list[str]:
    """Detect local private IPv4 addresses for same-Wi-Fi mobile pairing without sending traffic."""
    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in candidates and not ip.startswith("127.") and not ip.startswith("169.254."):
                try:
                    if ipaddress.ip_address(ip).is_private:
                        candidates.append(ip)
                except ValueError:
                    pass
    except OSError:
        pass
    return candidates or ["127.0.0.1"]


def build_qr_svg(url: str) -> str:
    """Render an inline SVG QR code using pure-Python segno."""
    try:
        import segno

        qr = segno.make(url, error="m")
        return qr.svg_inline(scale=5, border=2, dark="#0b0f17", light="#ffffff")
    except Exception:
        return ""

app = FastAPI(title="SpoolGenie API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    auto_render_2d: bool = True
    force_2d_backend: str | None = None
    lora_path: str | None = None


class MakerWorldURLRequest(BaseModel):
    url: str


class BuildRequest(BaseModel):
    custom_cad_code: str | None = None
    force_mesh_backend: str | None = None
    run_slicer: bool = True


class SpecPatchRequest(BaseModel):
    patch: dict[str, Any]
    re_render_2d: bool = True


def _get_session(project_id: str | None = None) -> SpoolGenieSession:
    if project_id and ("/" in project_id or "\\" in project_id or ".." in project_id):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    return SpoolGenieSession(project_id=project_id, settings=get_settings())


def _decorate_state_urls(session: SpoolGenieSession) -> dict[str, Any]:
    """Return ProjectState JSON plus browser-downloadable relative URLs for every artifact."""
    data = session.state.model_dump(mode="json")
    pid = session.state.project_id

    def to_url(p: str | None) -> str | None:
        if not p:
            return None
        path = Path(p)
        if not path.exists():
            return None
        return f"/api/projects/{pid}/files/{path.name}"

    exp = data.get("exports") or {}
    exp["urls"] = {
        "threemf": to_url(exp.get("threemf_path")),
        "split_plates_3mf": to_url(exp.get("split_plates_3mf_path")),
        "stl": to_url(exp.get("stl_path")),
        "step": to_url(exp.get("step_path")),
        "preview_glb": to_url(exp.get("preview_glb_path")),
        "split_preview_glb": to_url(exp.get("split_preview_glb_path")),
        "turntable_preview": to_url(exp.get("turntable_preview_path")),
        "cad_script": to_url(exp.get("cad_script_path")),
        "gcode_3mf": to_url(exp.get("gcode_3mf_path")),
        "per_color_stls": {k: to_url(v) for k, v in (exp.get("per_color_stls") or {}).items()},
    }
    data["exports"] = exp
    for c in data.get("concepts") or []:
        if c.get("image_path"):
            c["image_url"] = to_url(c["image_path"])
    return data


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/manifest.json")
def pwa_manifest() -> FileResponse:
    return FileResponse(STATIC_DIR / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def pwa_service_worker() -> FileResponse:
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/icon.svg")
@app.get("/favicon.ico")
def app_icon() -> FileResponse:
    return FileResponse(STATIC_DIR / "icon.svg", media_type="image/svg+xml")


@app.get("/api/lan-info")
def get_lan_info(request: Request) -> dict[str, Any]:
    """Return local Wi-Fi URL + inline SVG QR code for one-scan iOS/Android mobile pairing."""
    bind_host = str(SERVER_RUNTIME.get("host") or "127.0.0.1")
    port = request.url.port or int(SERVER_RUNTIME.get("port") or 8765)
    lan_ips = detect_lan_ips()
    primary_ip = lan_ips[0]
    lan_enabled = bind_host in ("0.0.0.0", "::", "") or not bind_host.startswith("127.")
    lan_url = f"http://{primary_ip}:{port}"
    return {
        "bind_host": bind_host,
        "port": port,
        "lan_enabled": lan_enabled,
        "lan_ips": lan_ips,
        "primary_ip": primary_ip,
        "lan_url": lan_url,
        "qr_svg": build_qr_svg(lan_url),
        "cli_hint": f"spoolgenie serve --lan --port {port}",
    }


@app.get("/api/meta")
def get_meta() -> dict[str, Any]:
    doc = run_doctor(get_settings())
    kb = PrintKnowledgeBase(get_settings().workspace_dir / "knowledge_store.jsonl")
    return {
        "version": __version__,
        "doctor": doc.to_dict(),
        "prior_counts": {
            "subjects": len(CHARACTER_CATALOG),
            "engineering_kb": len(kb.entries),
        },
        "printers": {
            pid: {
                "id": p.id,
                "name": p.name,
                "bed_x_mm": p.bed_x_mm,
                "bed_y_mm": p.bed_y_mm,
                "bed_z_mm": p.bed_z_mm,
                "ams_slots": p.ams_slots,
            }
            for pid, p in PRINTER_PROFILES.items()
        },
        "swatches": BAMBU_PLA_SWATCHES,
        "official_filaments": [
            {
                "name": f.name,
                "name_zh": f.name_zh,
                "hex_color": f.hex_color,
                "series": f.series,
                "store_url_cn": f.store_url_cn,
                "store_url_global": f.store_url_global,
            }
            for f in BAMBU_OFFICIAL_FILAMENTS
        ],
        "character_presets": {k: v["label"] for k, v in CHARACTER_STYLE_PRESETS.items()},
        "character_catalog": [
            {
                "id": p.id,
                "franchise": p.franchise,
                "name": p.name,
                "name_zh": p.name_zh,
                "route": p.route,
                "default_dims_mm": list(p.default_dims_mm),
                "colors": [c[1] for c in p.filaments],
                "print_tips": p.print_tips,
                "no_ams_split_tip": p.no_ams_split_tip,
            }
            for p in CHARACTER_CATALOG.values()
        ],
        "local_loras": list_local_loras("loras"),
        "kb_count": len(kb.entries),
    }


@app.post("/api/projects")
def create_project() -> dict[str, Any]:
    session = _get_session()
    return _decorate_state_urls(session)


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    ws = get_settings().workspace_dir
    out: list[dict[str, Any]] = []
    if not ws.exists():
        return out
    for p in sorted(ws.glob("*/project.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:25]:
        try:
            sess = _get_session(p.parent.name)
            out.append(
                {
                    "project_id": sess.state.project_id,
                    "title": sess.state.spec.title,
                    "stage": sess.state.stage.value,
                    "updated_at": sess.state.updated_at,
                }
            )
        except Exception:
            continue
    return out


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return _decorate_state_urls(_get_session(project_id))


class Render2DRequest(BaseModel):
    force_2d_backend: str | None = None
    lora_path: str | None = None


@app.post("/api/projects/{project_id}/chat")
def chat_project(project_id: str, req: ChatRequest) -> dict[str, Any]:
    session = _get_session(project_id)
    res = session.chat(
        req.message,
        auto_render_2d=req.auto_render_2d,
        force_2d_backend=req.force_2d_backend,
        lora_path=req.lora_path,
    )
    return {
        "reply": res["reply"],
        "project": _decorate_state_urls(session),
    }


@app.post("/api/projects/{project_id}/render-2d")
def render_project_2d(project_id: str, req: Render2DRequest | None = None) -> dict[str, Any]:
    req = req or Render2DRequest()
    session = _get_session(project_id)
    session.render_2d_concept(
        force_backend=req.force_2d_backend,
        lora_path=req.lora_path,
    )
    return _decorate_state_urls(session)


@app.patch("/api/projects/{project_id}/spec")
def patch_spec(project_id: str, req: SpecPatchRequest) -> dict[str, Any]:
    session = _get_session(project_id)
    session.update_spec(req.patch, re_render_2d=req.re_render_2d)
    return _decorate_state_urls(session)


@app.post("/api/projects/{project_id}/confirm-build")
def confirm_build(project_id: str, req: BuildRequest | None = None) -> dict[str, Any]:
    req = req or BuildRequest()
    session = _get_session(project_id)
    session.confirm_and_build(
        custom_cad_code=req.custom_cad_code,
        force_mesh_backend=req.force_mesh_backend,
        run_slicer=req.run_slicer,
    )
    return _decorate_state_urls(session)


class FilamentMatchRequest(BaseModel):
    canon_filaments: list[FilamentSlot]
    owned_filaments: list[FilamentSlot] = []
    max_delta_e: float = 22.0


@app.post("/api/match-filaments")
def api_match_filaments(req: FilamentMatchRequest) -> dict[str, Any]:
    """Match canonical model colors against user's owned spools and recommend Bambu Lab official store links for missing colors."""
    resolved, recs = match_canon_to_owned_filaments(
        req.canon_filaments,
        owned_filaments=req.owned_filaments,
        max_delta_e=req.max_delta_e,
    )
    return {
        "resolved_filaments": [f.model_dump() for f in resolved],
        "recommendations": [r.model_dump() for r in recs],
    }


@app.get("/api/makerworld/search")
def api_makerworld_search(q: str, limit: int = 6) -> list[dict[str, Any]]:
    return [r.model_dump() for r in search_makerworld(q, max_results=min(limit, 12))]


class WebRAGRequest(BaseModel):
    query: str | None = None
    max_images: int = 4
    max_3d_refs: int = 6


@app.post("/api/projects/{project_id}/web-rag")
def api_project_web_rag(project_id: str, req: WebRAGRequest | None = None) -> dict[str, Any]:
    """Retrieve live multi-view character images + MakerWorld/Printables/Thingiverse 3D model references."""
    req = req or WebRAGRequest()
    session = _get_session(project_id)
    pack = session.enrich_from_web_rag(
        query=req.query,
        max_images=min(max(req.max_images, 1), 6),
        max_3d_refs=min(max(req.max_3d_refs, 1), 8),
    )
    return {"web_rag": pack, "project": _decorate_state_urls(session)}


@app.post("/api/projects/{project_id}/references/url")
def add_ref_url(project_id: str, req: MakerWorldURLRequest) -> dict[str, Any]:
    session = _get_session(project_id)
    ref = session.add_reference_url(req.url)
    return {"reference": ref.model_dump(), "project": _decorate_state_urls(session)}


@app.post("/api/projects/{project_id}/references/upload")
async def upload_ref_file(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    session = _get_session(project_id)
    safe_name = Path(file.filename or "reference.png").name
    refs_dir = session.project_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    dst = refs_dir / safe_name
    dst.write_bytes(await file.read())
    ref = session.add_reference_file(dst)
    return {"reference": ref.model_dump(), "project": _decorate_state_urls(session)}


@app.get("/api/projects/{project_id}/files/{filename}")
def download_project_file(project_id: str, filename: str) -> FileResponse:
    session = _get_session(project_id)
    safe_name = Path(filename).name
    target = (session.project_dir / safe_name).resolve()
    if not target.exists() or target.parent != session.project_dir.resolve():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=str(target), filename=safe_name)
