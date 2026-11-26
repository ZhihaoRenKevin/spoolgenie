"""SpoolGenie cross-platform CLI (shares the exact same core engine as the Web UI)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spoolgenie.config import PRINTER_PROFILES, get_settings
from spoolgenie.doctor import run_doctor
from spoolgenie.knowledge.kb import PrintKnowledgeBase
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.pipeline import SpoolGenieSession
from spoolgenie.print_prep.bambu_3mf import export_painted_bambu_3mf
from spoolgenie.print_prep.color_quantizer import quantize_mesh_to_ams_slots
from spoolgenie.print_prep.repair import repair_single_mesh
from spoolgenie.reference.makerworld import search_makerworld
from spoolgenie.schemas import FilamentSlot, JointType, MulticolorMode
from spoolgenie.training.trainer import export_cad_sft_dataset, prepare_character_lora_dataset

app = typer.Typer(name="spoolgenie", help="Conversational 2D -> 3D -> Bambu AMS 3MF studio (CLI + Web UI).")
kb_app = typer.Typer(help="Manage MakerWorld functional priors & printable character knowledge bank.")
train_app = typer.Typer(help="Prepare local RTX GPU LoRA (characters) & QLoRA SFT (MakerWorld CAD) datasets/scripts.")
app.add_typer(kb_app, name="kb")
app.add_typer(train_app, name="train")

console = Console()


@app.command()
def doctor() -> None:
    """Inspect OS, GPU (RTX 5090 sm_120 / CUDA / Apple MPS / CPU), Ollama models, packages, and slicers."""
    rep = run_doctor()
    tbl = Table(title="SpoolGenie Hardware & Environment Diagnostics", show_header=True)
    tbl.add_column("Component", style="cyan")
    tbl.add_column("Status")
    tbl.add_row("OS / Arch", f"{rep.os_name} ({rep.arch})")
    tbl.add_row("Python", rep.python_version)
    tbl.add_row("Accelerator", f"{rep.accelerator} | GPU: {rep.gpu_name or 'None'} ({rep.gpu_vram_gb or '-'} GB)")
    tbl.add_row("PyTorch", f"{rep.torch_version or 'Not installed'} (CUDA ready={rep.torch_cuda_ready}, CC={rep.cuda_compute_cap or '-'})")
    if rep.blackwell_warning:
        tbl.add_row("RTX 50-Series Warning", f"[red]{rep.blackwell_warning}[/red]")
    tbl.add_row("Ollama", f"{'Connected' if rep.ollama_reachable else 'Offline'} | Models: {', '.join(rep.ollama_models) or 'None'}")
    tbl.add_row("Auto LLM / VLM", f"LLM={rep.recommended_llm or 'fallback'} | VLM={rep.recommended_vlm or 'none'}")
    tbl.add_row("Recommended Backends", f"2D={rep.recommended_2d_backend} | CAD={rep.recommended_3d_cad_backend} | Organic3D={rep.recommended_3d_organic_backend}")
    tbl.add_row("Bambu Studio", rep.slicers.get("bambu_studio") or "Not found")
    tbl.add_row("OrcaSlicer", rep.slicers.get("orca_slicer") or "Not found")
    pkg_str = "  ".join(f"{k}:{'[green]OK[/green]' if v else '[dim]no[/dim]'}" for k, v in rep.packages.items())
    tbl.add_row("Packages", pkg_str)
    console.print(tbl)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (use --lan for 0.0.0.0 phone QR pairing)"),
    port: int = typer.Option(8765, help="Port"),
    lan: bool = typer.Option(False, "--lan", help="Expose on 0.0.0.0 so iPhone/Android on the same Wi-Fi can scan QR & connect"),
) -> None:
    """Launch the SpoolGenie interactive Web/PWA Studio (Desktop Browser + Mobile LAN QR Pairing)."""
    import uvicorn

    from spoolgenie.web.app import detect_lan_ips, set_server_runtime

    bind_host = "0.0.0.0" if lan else host
    set_server_runtime(bind_host, port)
    lan_ip = detect_lan_ips()[0]
    lines = [f"Local URL:  [bold cyan]http://127.0.0.1:{port}[/bold cyan]"]
    if bind_host in ("0.0.0.0", "::"):
        lines.append(f"Mobile LAN: [bold green]http://{lan_ip}:{port}[/bold green]  (Click '📱 Mobile' in topbar for QR code)")
    else:
        lines.append("Tip: pass [bold]--lan[/bold] to pair iOS/Android phones over Wi-Fi.")
    console.print(Panel("\n".join(lines), title="SpoolGenie"))
    uvicorn.run("spoolgenie.web.app:app", host=bind_host, port=port, reload=False)


@app.command()
def desktop(
    port: int = typer.Option(8765, help="Embedded local server port (falls back to a free port if busy)"),
    lan: bool = typer.Option(True, "--lan/--local-only", help="Allow same-Wi-Fi mobile phones to scan QR and pair with this Desktop App"),
) -> None:
    """Launch the native Windows / macOS Desktop Application (Edge WebView2 / macOS WKWebView)."""
    from spoolgenie.desktop import launch_desktop

    launch_desktop(port=port, lan=lan)


@app.command()
def makerworld(
    query: str = typer.Argument(..., help="Search keywords for MakerWorld reference models"),
    limit: int = typer.Option(6, help="Max search results"),
) -> None:
    """Search public MakerWorld model references."""
    refs = search_makerworld(query, max_results=limit)
    tbl = Table(title=f"MakerWorld References: {query}")
    tbl.add_column("#", style="dim")
    tbl.add_column("Title", style="bold")
    tbl.add_column("URL", style="cyan")
    for idx, r in enumerate(refs, start=1):
        tbl.add_row(str(idx), r.title, r.url or "-")
    console.print(tbl)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Natural-language description of what you want to 3D print"),
    printer: str = typer.Option("bambu_p1s", help=f"Printer ID: {', '.join(PRINTER_PROFILES)}"),
    ams_units: int = typer.Option(1, min=0, max=4, help="Number of AMS units (0=No-AMS glue-free split, 1=4 colors, ..., 4=16 colors)"),
    mode: MulticolorMode = typer.Option(MulticolorMode.AUTO, help="auto | ams | split_no_ams | monochrome"),
    joint: JointType = typer.Option(JointType.DOVETAIL, help="Glue-free joint: dovetail | snap_pin | press_tenon | none"),
    ref_image: Optional[Path] = typer.Option(None, help="Optional MakerWorld screenshot/reference image"),
    lora: Optional[Path] = typer.Option(None, help="Optional local character 2D LoRA weights path"),
    project_id: Optional[str] = typer.Option(None, help="Optional project folder ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive 2D confirmation prompt"),
) -> None:
    """End-to-end conversational generation: Prompt -> 2D Concept Gate -> Confirm -> 3D + Bambu 3MF/STL/STEP."""
    session = SpoolGenieSession(project_id=project_id)
    session.state.spec.printer_id = printer
    session.state.spec.ams_units = ams_units
    session.state.spec.multicolor_mode = mode
    session.state.spec.joint_type = joint
    session.save()

    if ref_image and ref_image.exists():
        ref = session.add_reference_file(ref_image)
        console.print(f"[green]Imported reference:[/green] {ref.title} | Colors: {', '.join(ref.extracted_colors)}")

    res = session.chat(prompt, auto_render_2d=True, lora_path=str(lora) if lora else None)
    console.print(Panel(res["reply"], title=f"Stage 1: 2D Concept Gate ({session.state.project_id})"))
    if res["concept"]:
        console.print(f"2D Preview saved: [cyan]{res['concept'].image_path}[/cyan] ({res['concept'].backend})")

    if not yes:
        while True:
            ans = typer.prompt("Type 'confirm' to build 3D print files, or enter tweaks (e.g. '高度改190mm, 没有AMS用卡扣')", default="confirm")
            if ans.strip().lower() in ("confirm", "y", "yes", "确认"):
                break
            res = session.chat(ans, auto_render_2d=True, lora_path=str(lora) if lora else None)
            console.print(Panel(res["reply"], title="Updated 2D Concept"))
            if res["concept"]:
                console.print(f"2D Preview updated: [cyan]{res['concept'].image_path}[/cyan]")

    console.print("[yellow]Building 3D model, DFM repair, AMS color budgeting, and glue-free joints...[/yellow]")
    state = session.confirm_and_build()
    dfm = state.dfm
    exp = state.exports

    summary = (
        f"Title: [bold]{state.spec.title}[/bold]  |  Route: {state.spec.route.value}\n"
        f"Mode: {dfm.effective_multicolor_mode} (AMS units={dfm.ams_units_configured}, Colors={dfm.ams_slots_used}, Joint={dfm.joint_type_used})\n"
        f"BBox: {dfm.bbox_mm[0]} x {dfm.bbox_mm[1]} x {dfm.bbox_mm[2]} mm  |  Watertight: {dfm.watertight}  |  Mass: ~{dfm.estimated_mass_g} g\n"
        f"Est. Time: ~{exp.estimated_print_minutes} min  |  Filament: ~{exp.estimated_filament_g} g (AMS flush ~{exp.estimated_flush_g} g)\n"
        f"Bambu 3MF: [cyan]{exp.threemf_path}[/cyan]\n"
        + (f"No-AMS Split Plates 3MF: [green]{exp.split_plates_3mf_path}[/green]\n" if exp.split_plates_3mf_path else "")
        + f"Universal STL: {exp.stl_path}\n"
        + (f"CAD STEP: {exp.step_path}\n" if exp.step_path else "")
        + f"3D Preview GLB: {exp.preview_glb_path}"
    )
    console.print(Panel(summary, title="SpoolGenie Export Ready"))


@app.command()
def colorize(
    mesh_file: Path = typer.Argument(..., help="Input GLB/OBJ/STL file to quantize into Bambu AMS multicolor 3MF"),
    output: Path = typer.Option(Path("colored_bambu.3mf"), "--output", "-o", help="Output .3mf path"),
    colors: str = typer.Option("#2B2D30,#F36C21,#F2EFE9,#1E4C8A", help="Comma-separated AMS filament HEX colors"),
    image: Optional[Path] = typer.Option(None, help="Optional 2D image to project onto an untextured mesh"),
) -> None:
    """Convert any textured GLB/OBJ (or untextured mesh + 2D image) into a Bambu Studio painted multicolor .3mf."""
    import trimesh

    raw = trimesh.load(mesh_file, force="mesh")
    rep = repair_single_mesh(raw)
    hex_list = [c.strip() for c in colors.split(",") if c.strip()]
    filaments = [FilamentSlot(slot=i + 1, name=f"AMS Slot {i+1}", hex_color=hx) for i, hx in enumerate(hex_list[:16])]
    face_slots, active = quantize_mesh_to_ams_slots(rep.mesh, filaments, reference_image_path=image)
    out = export_painted_bambu_3mf(rep.mesh, face_slots, active, output, model_title=mesh_file.stem)
    console.print(f"[green]Exported Bambu AMS painted 3MF:[/green] {out.resolve()} ({len(active)} slots, {len(rep.mesh.faces):,} faces)")


@kb_app.command("search")
def kb_search(query: str = typer.Argument(...), domain: Optional[str] = typer.Option(None)) -> None:
    """Search SpoolGenie's built-in + ingested character & MakerWorld functional knowledge bank."""
    kb = PrintKnowledgeBase(get_settings().workspace_dir / "knowledge_store.jsonl")
    hits = kb.retrieve(query, domain=domain, top_k=5)
    for h in hits:
        console.print(f"[bold cyan]{h.title}[/bold cyan] [{h.domain}]\n  {h.summary}")


@kb_app.command("ingest-makerworld")
def kb_ingest_mw(topic: str = typer.Argument(..., help="Functional topic to index from public MakerWorld listings")) -> None:
    """Index MakerWorld functional design priors into the local knowledge bank."""
    kb = PrintKnowledgeBase(get_settings().workspace_dir / "knowledge_store.jsonl")
    added = kb.ingest_makerworld_topic(topic)
    console.print(f"[green]Ingested {len(added)} MakerWorld functional entries for '{topic}'.[/green]")


@kb_app.command("ingest-dir")
def kb_ingest_dir(folder: Path = typer.Argument(..., help="Local folder of downloaded .3mf/.stl files")) -> None:
    """Scan local MakerWorld .3mf/.stl files and ingest bounding boxes/metadata into the knowledge bank."""
    kb = PrintKnowledgeBase(get_settings().workspace_dir / "knowledge_store.jsonl")
    added = kb.ingest_local_models_dir(folder)
    console.print(f"[green]Indexed {len(added)} local 3D models from {folder}.[/green]")


@train_app.command("prepare-character")
def train_prepare_char(
    images_dir: Path = typer.Argument(..., help="Folder of reference images for your character/style"),
    output_dir: Path = typer.Option(Path("outputs/train_character_lora"), "--out", "-o"),
    trigger: str = typer.Option("pf_chibi_print", help="LoRA trigger token"),
) -> None:
    """Prepare square-padded character images + VLM captions + RTX 5090 LoRA training script."""
    res = prepare_character_lora_dataset(images_dir, output_dir, trigger_token=trigger, llm=UnifiedLLMClient())
    console.print(Panel(f"Images: {res['image_count']}\nMetadata: {res['metadata_jsonl']}\nTrainer: {res['train_script']}", title="Character LoRA Kit Ready"))


@train_app.command("export-cad-sft")
def train_export_cad(
    output_dir: Path = typer.Option(Path("outputs/train_cad_sft"), "--out", "-o"),
) -> None:
    """Export MakerWorld functional priors + verified build123d builds into QLoRA SFT JSONL + Ollama Modelfile."""
    settings = get_settings()
    kb = PrintKnowledgeBase(settings.workspace_dir / "knowledge_store.jsonl")
    res = export_cad_sft_dataset(output_dir, workspace_dir=settings.workspace_dir, kb=kb)
    console.print(Panel(f"Samples: {res['sample_count']}\nDataset: {res['dataset_jsonl']}\nQLoRA Script: {res['qlora_script']}\nOllama Modelfile: {res['ollama_modelfile']}", title="MakerWorld CAD SFT Kit Ready"))


if __name__ == "__main__":
    app()
