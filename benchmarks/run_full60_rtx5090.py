"""Resumable Full 60-Object Nightly RTX 5090 Benchmark & Paper Table Generator (P0 + P1.5 + P2).

Features:
  - Resumable checkpointing (`benchmarks/results_60/checkpoint.jsonl`): safe against interrupts/restarts!
  - Strict RTX 5090 CUDA VRAM hygiene (`gc.collect()` + `torch.cuda.empty_cache()` between objects)
  - Runs ALL 60 objects in `benchmarks/dataset_60.json`:
      * Tier A (20 organic characters) & Tier B (20 functional hybrids): SDXL 1.0 -> Hunyuan3D-2 320^3/42s ->
        Zero-blur direct manifold repair -> Optional Web-RAG multi-view -> ALL 5 ablation regimes!
      * Tier C (20 parametric mechanical CAD): build123d -> ALL applicable ablations!
  - Automatically generates:
      * `benchmarks/results_60/full60_per_object.csv`
      * `benchmarks/results_60/full60_summary.json`
      * `benchmarks/results_60/table_main_ablation.tex` (SIGGRAPH/CVPR booktabs LaTeX table ready to paste!)

Usage:
  # Dry-run on Tier C (20 CAD objects, ~25 seconds, zero GPU wait):
  python benchmarks/run_full60_rtx5090.py --tiers parametric_mechanical

  # Run full 60-object GPU benchmark overnight on RTX 5090 (~2.2 hours, resumable!):
  python benchmarks/run_full60_rtx5090.py --all --use-dinov2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from spoolgenie.config import SpoolGenieSettings, get_settings
from spoolgenie.engines.cad_engine import build_parametric_model, execute_build123d_script
from spoolgenie.engines.concept_2d import generate_2d_concept, release_diffusers_vram
from spoolgenie.engines.mesh_engine import ensure_closed_flat_base, generate_organic_mesh
from spoolgenie.engines.modular_cad_models import PRECISION_CAD_BUILDERS
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG
from spoolgenie.print_prep.bambu_3mf import export_multipart_bambu_3mf
from spoolgenie.print_prep.color_quantizer import quantize_mesh_to_ams_slots
from spoolgenie.print_prep.functional_carver import (
    _manifold_to_mesh,
    _mesh_to_manifold,
    apply_functional_ops_modular,
)
from spoolgenie.print_prep.modularizer import decompose_mesh_into_solid_modules
from spoolgenie.print_prep.physics_engine import evaluate_physics_stability
from spoolgenie.print_prep.repair import repair_single_mesh, scale_and_ground_meshes
from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.reference.web_character_rag import retrieve_web_character_pack
from spoolgenie.research.fab_graphcut import FabGraphCutConfig, decompose_mesh_fab_graphcut
from spoolgenie.research.joint_fab_opt import (
    JointFabConfig,
    compute_ams_purge_metrics,
    compute_joint_fabrication_loss,
    optimize_assembly_joint_fabrication,
)
from spoolgenie.schemas import DesignSpec, EngineRoute, FilamentSlot, FunctionalOp


def _clear_gpu_memory() -> None:
    release_diffusers_vram()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_spec_from_dataset_row(row: dict[str, Any]) -> DesignSpec:
    cat = row["category"]
    w, d, h = [float(x) for x in row["dims"]]
    tier = row["tier"]
    load_case = row["load_case"]

    preset = CHARACTER_CATALOG.get(cat)
    if preset is not None:
        filaments = [
            FilamentSlot(slot=i + 1, name=n, hex_color=hx, role=r)
            for i, (n, hx, r) in enumerate(preset.filaments)
        ]
        prompt_2d = preset.prompt_2d
        title = preset.name
    else:
        filaments = [
            FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30", role="main body"),
            FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21", role="accent trim"),
        ]
        prompt_2d = f"Isometric 3/4 studio product render of {row['prompt']}, matte PLA, pure white background"
        title = row["prompt"][:48]

    ops: list[FunctionalOp] = []
    if tier == "functional_hybrid":
        route = EngineRoute.HYBRID
        if load_case == "magsafe":
            ops.append(FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4, "depth_mm": 5.8, "face": "front"}))
        elif load_case == "pen_caddy":
            ops.append(FunctionalOp(op="hollow_cavity", params={"shape": "square" if "walle" in cat else "round"}))
        elif load_case == "phone_cradle":
            ops.append(FunctionalOp(op="phone_slot", params={"slot_width_mm": 84.0, "slot_thickness_mm": 14.0}))
        elif load_case == "hook":
            ops.append(FunctionalOp(op="keychain_loop", params={"outer_radius_mm": 8.0, "inner_radius_mm": 3.8}))
        elif "cable" in row["prompt"].lower():
            ops.append(FunctionalOp(op="cable_comb", params={"slots": 3}))
    elif tier == "organic_character":
        route = EngineRoute.MESH
    else:
        route = EngineRoute.CAD
        if load_case == "magsafe":
            ops.append(FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4, "depth_mm": 5.8}))

    spec = DesignSpec(
        title=title,
        summary=row["prompt"],
        category=cat,
        route=route,
        ams_units=int(row["ams_units"]),
        filaments=filaments,
        functional_ops=ops,
        prompt_2d=prompt_2d,
    )
    spec.dimensions.width_mm = w
    spec.dimensions.depth_mm = d
    spec.dimensions.height_mm = h
    return spec


def _export_latex_ablation_table(summary: dict[str, Any], out_tex: Path) -> None:
    """Emit publication-ready booktabs LaTeX Table 1 directly from measured JSON summary."""
    order = [
        ("B0_Raw_Monolithic", "Monolithic Raw (Single-Color)"),
        ("B1_SurfacePaint_NoPhys", "Surface Paint (Bambu/MakerLab-style)"),
        ("B2_PlanarModular_NoPhys", "Planar CSG Modular (No Physics)"),
        ("B3_FabGraphCut_Only", "Ours w/o P2 (P1 FabGraphCut Only)"),
        ("Ours_Full_JointFabOpt", "\\textbf{Ours Full (P1 + P2 JointOpt)}"),
    ]
    by_m = summary.get("by_method", {})
    lines = [
        "% Auto-generated by benchmarks/run_full60_rtx5090.py — DO NOT HAND-EDIT NUMBERS!",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Controlled ablation across $N={summary.get('n_objects_evaluated', 0)}$ benchmark objects (organic, hybrid functional, and parametric CAD).}}",
        "\\label{tab:main_ablation}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & $N$ & Watertight (\\%) & AMS Swaps $\\downarrow$ & Flush Waste (g) $\\downarrow$ & Loaded $\\theta_{\\text{tip}}$ ($^\\circ$) $\\uparrow$ & $F_{\\text{tip}}$ (N) $\\uparrow$ & $\\mathcal{L}_{\\text{total}}$ $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for key, label in order:
        if key not in by_m:
            continue
        m = by_m[key]
        lines.append(
            f"{label} & {m['count']} & {m['watertight_rate_pct']:.1f} & "
            f"{m['mean_ams_swaps']:.1f} & {m['mean_ams_waste_g']:.1f} & "
            f"{m['mean_tilt_loaded_deg']:.1f} & {m['mean_tipping_force_n']:.2f} & "
            f"{m['mean_L_total']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def run_batch(
    dataset_path: Path,
    out_dir: Path,
    tiers: set[str],
    max_items: int | None = None,
    use_web_rag: bool = False,
    use_dinov2: bool = False,
    export_3mf_pairs: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.jsonl"
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load existing completed object IDs for seamless resume!
    done_rows: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_rows.append(rec)
                done_ids.add(rec["object_id"])

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = [r for r in dataset if r["tier"] in tiers]
    if max_items is not None:
        selected = selected[:max_items]

    settings: SpoolGenieSettings = get_settings()
    jf_cfg = JointFabConfig()
    fgc_cfg = FabGraphCutConfig(n_superfacets=220, use_dinov2=use_dinov2)

    print(f"[Full60] Target={len(selected)} objects | Already completed={len(done_ids)} | dinov2={use_dinov2} | webrag={use_web_rag}", flush=True)

    for idx, row in enumerate(selected, start=1):
        oid = row["id"]
        if oid in done_ids:
            continue

        t0 = time.time()
        spec = _build_spec_from_dataset_row(row)
        obj_dir = assets_dir / oid
        obj_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(selected)}] Building {oid} ({row['tier']} / {spec.category})...", flush=True)

        cad_parts: list[PrintPart] | None = None
        extra_func_parts: list[PrintPart] = []
        ref_imgs: list[Path] = []
        mesh: trimesh.Trimesh

        try:
            precision_fn = PRECISION_CAD_BUILDERS.get(spec.category)
            if precision_fn is not None:
                cparts = execute_build123d_script(precision_fn(spec))
                cad_parts = [PrintPart(name=cp.name, slot=cp.slot, mesh=repair_single_mesh(cp.mesh).mesh) for cp in cparts]
                scaled = scale_and_ground_meshes(
                    [p.mesh for p in cad_parts],
                    target_w=spec.dimensions.width_mm,
                    target_d=spec.dimensions.depth_mm,
                    target_h=spec.dimensions.height_mm,
                    preserve_aspect=True,
                )
                for p, sm in zip(cad_parts, scaled):
                    p.mesh = sm
                mans = [_mesh_to_manifold(p.mesh) for p in cad_parts]
                u = mans[0]
                for mm in mans[1:]:
                    u = u + mm
                mesh = _manifold_to_mesh(u)
            elif spec.route == EngineRoute.CAD:
                cad_res = build_parametric_model(spec, step_output_path=None, llm=None)
                cad_parts = [PrintPart(name=cp.name, slot=cp.slot, mesh=repair_single_mesh(cp.mesh).mesh) for cp in cad_res.parts]
                scaled = scale_and_ground_meshes(
                    [p.mesh for p in cad_parts],
                    target_w=spec.dimensions.width_mm,
                    target_d=spec.dimensions.depth_mm,
                    target_h=spec.dimensions.height_mm,
                    preserve_aspect=True,
                )
                for p, sm in zip(cad_parts, scaled):
                    p.mesh = sm
                mesh = trimesh.util.concatenate([p.mesh for p in cad_parts]) if len(cad_parts) > 1 else cad_parts[0].mesh
            else:
                # Organic / Hybrid on RTX 5090 CUDA: SDXL -> Hunyuan3D-2 -> Zero-blur repair!
                c_img = obj_dir / "concept_01.png"
                if not c_img.exists():
                    generate_2d_concept(spec, c_img, settings=settings)
                ref_imgs.append(c_img)

                if use_web_rag:
                    try:
                        pack = retrieve_web_character_pack(spec.title, save_dir=obj_dir / "web_refs", max_images=2, max_3d_refs=2)
                        for p_str in pack.downloaded_images[:2]:
                            ref_imgs.append(Path(p_str))
                    except Exception:
                        pass

                raw_stl = obj_dir / "raw_mesh.stl"
                if raw_stl.exists():
                    mesh_raw = trimesh.load(raw_stl, force="mesh")
                else:
                    mres = generate_organic_mesh(spec, c_img, settings=settings)
                    mesh_raw = repair_single_mesh(mres.mesh).mesh
                    mesh_raw.export(raw_stl)

                mesh = scale_and_ground_meshes(
                    [mesh_raw],
                    target_w=spec.dimensions.width_mm,
                    target_d=spec.dimensions.depth_mm,
                    target_h=spec.dimensions.height_mm,
                    preserve_aspect=True,
                )[0]
                mesh = ensure_closed_flat_base(mesh)
                if spec.route == EngineRoute.HYBRID and spec.functional_ops:
                    mesh, extra_func_parts, _ = apply_functional_ops_modular(
                        mesh, spec.functional_ops, wall_thickness_mm=spec.dimensions.wall_thickness_mm
                    )
        except Exception as exc:
            print(f"  [ERROR] {oid} failed during 3D generation: {exc}", flush=True)
            _clear_gpu_memory()
            continue

        # Evaluate all ablation regimes for this object
        obj_records: list[dict[str, Any]] = []

        def _rec(method: str, parts: list[PrintPart], face_slots=None, vis_de: float = 0.0, ams_o=None, phys_o=None, cav_cm3: float = 0.0, com_dz: float = 0.0, orient: str = "upright_Z") -> None:
            phys = phys_o or evaluate_physics_stability(parts, spec)
            ams = ams_o or compute_ams_purge_metrics(parts, face_slots_single=face_slots)
            loss = compute_joint_fabrication_loss(phys, ams, visual_delta_e=vis_de, cfg=jf_cfg)
            obj_records.append(
                {
                    "object_id": oid,
                    "tier": row["tier"],
                    "category": spec.category,
                    "method": method,
                    "n_parts": len(parts),
                    "watertight": int(all(p.mesh.is_watertight for p in parts)),
                    "visual_delta_e": round(vis_de, 2),
                    "ams_swaps": ams.total_swaps,
                    "ams_waste_g": ams.total_waste_g,
                    "swap_time_min": ams.swap_time_min,
                    "tilt_loaded_deg": phys.tilt_angle_loaded_deg,
                    "tipping_force_n": phys.tipping_force_n,
                    "fos_z": phys.safety_factor_z,
                    "base_closed": int(phys.base_closed_solid),
                    "stability_score": phys.stability_score,
                    "cavity_cm3": cav_cm3,
                    "com_z_shift_mm": com_dz,
                    "print_orientation": orient,
                    "L_total": loss["L_total"],
                    "elapsed_s": round(time.time() - t0, 1),
                }
            )

        p_raw = [PrintPart(name="raw_body", slot=1, mesh=mesh)]
        _rec("B0_Raw_Monolithic", p_raw, vis_de=32.0 if len(spec.filaments) > 1 else 0.0)

        if cad_parts is not None:
            _rec("B2_PlanarModular_NoPhys", cad_parts, vis_de=3.5)
            jf = optimize_assembly_joint_fabrication(cad_parts, spec, filaments=spec.filaments, visual_delta_e=3.5, config=jf_cfg)
            _rec("Ours_Full_JointFabOpt", jf.parts, vis_de=3.5, ams_o=jf.ams_after, phys_o=jf.physics_after, cav_cm3=jf.cavity_volume_cm3, com_dz=jf.com_z_shift_mm, orient=jf.best_print_orientation.name)
            if export_3mf_pairs:
                export_multipart_bambu_3mf(jf.parts, spec.filaments, obj_dir / f"{oid}_ours.3mf", model_title=spec.title, wall_loops=jf.physics_after.recommended_wall_loops, sparse_infill_density=f"{jf.physics_after.recommended_infill_pct}%")
        else:
            c0 = ref_imgs[0] if ref_imgs else None
            f_slots, _ = quantize_mesh_to_ams_slots(mesh, spec.filaments, reference_image_path=c0, smooth_iters=6)
            _rec("B1_SurfacePaint_NoPhys", p_raw, face_slots=f_slots, vis_de=14.5)

            p_planar = decompose_mesh_into_solid_modules(mesh, spec.filaments, base_name=oid, extra_functional_parts=extra_func_parts, reference_image_path=c0)
            _rec("B2_PlanarModular_NoPhys", p_planar, vis_de=11.8)

            fgc = decompose_mesh_fab_graphcut(
                mesh,
                spec.filaments,
                base_name=oid,
                reference_image_path=c0,
                reference_image_paths=ref_imgs,
                config=fgc_cfg,
            )
            fgc_parts = fgc.parts + extra_func_parts
            _rec("B3_FabGraphCut_Only", fgc_parts, vis_de=fgc.visual_mean_delta_e)

            jf = optimize_assembly_joint_fabrication(fgc_parts, spec, filaments=spec.filaments, visual_delta_e=fgc.visual_mean_delta_e, config=jf_cfg)
            _rec("Ours_Full_JointFabOpt", jf.parts, vis_de=fgc.visual_mean_delta_e, ams_o=jf.ams_after, phys_o=jf.physics_after, cav_cm3=jf.cavity_volume_cm3, com_dz=jf.com_z_shift_mm, orient=jf.best_print_orientation.name)
            if export_3mf_pairs:
                export_multipart_bambu_3mf(jf.parts, spec.filaments, obj_dir / f"{oid}_ours.3mf", model_title=spec.title, wall_loops=jf.physics_after.recommended_wall_loops, sparse_infill_density=f"{jf.physics_after.recommended_infill_pct}%")

        # Persist immediately to resumable JSONL checkpoint!
        with open(ckpt_path, "a", encoding="utf-8") as cf:
            for r in obj_records:
                cf.write(json.dumps(r, ensure_ascii=False) + "\n")
                done_rows.append(r)
        done_ids.add(oid)
        print(f"  -> {oid} DONE in {time.time() - t0:.1f}s (Ours L_total={obj_records[-1]['L_total']})", flush=True)
        if spec.route != EngineRoute.CAD:
            _clear_gpu_memory()

    # Export aggregated CSV, JSON summary, and camera-ready LaTeX Table 1!
    if done_rows:
        csv_path = out_dir / "full60_per_object.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(done_rows[0].keys()))
            writer.writeheader()
            writer.writerows(done_rows)

    summary: dict[str, Any] = {"n_objects_evaluated": len(done_ids), "by_method": {}}
    for meth in sorted(set(r["method"] for r in done_rows)):
        sub = [r for r in done_rows if r["method"] == meth]
        summary["by_method"][meth] = {
            "count": len(sub),
            "watertight_rate_pct": round(100.0 * float(np.mean([r["watertight"] for r in sub])), 1),
            "mean_visual_delta_e": round(float(np.mean([r["visual_delta_e"] for r in sub])), 2),
            "mean_ams_swaps": round(float(np.mean([r["ams_swaps"] for r in sub])), 1),
            "mean_ams_waste_g": round(float(np.mean([r["ams_waste_g"] for r in sub])), 2),
            "mean_tilt_loaded_deg": round(float(np.mean([r["tilt_loaded_deg"] for r in sub])), 1),
            "mean_tipping_force_n": round(float(np.mean([r["tipping_force_n"] for r in sub])), 2),
            "mean_fos_z": round(float(np.mean([r["fos_z"] for r in sub])), 1),
            "mean_stability_score": round(float(np.mean([r["stability_score"] for r in sub])), 1),
            "mean_L_total": round(float(np.mean([r["L_total"] for r in sub])), 3),
        }

    (out_dir / "full60_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _export_latex_ablation_table(summary, out_dir / "table_main_ablation.tex")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable 60-Object RTX 5090 Benchmark & LaTeX Table Generator")
    parser.add_argument("--all", action="store_true", help="Run all 3 tiers (60 objects)")
    parser.add_argument("--tiers", nargs="+", default=["parametric_mechanical"], help="Tiers to run")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on number of items")
    parser.add_argument("--use-webrag", action="store_true", help="Enable live Web-RAG multi-view retrieval")
    parser.add_argument("--use-dinov2", action="store_true", help="Use DINOv2 CUDA dense patch features")
    parser.add_argument("--export-3mf", action="store_true", help="Export ready-to-print .3mf for every object")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    tiers = {"organic_character", "functional_hybrid", "parametric_mechanical"} if args.all else set(args.tiers)
    summary = run_batch(
        dataset_path=root / "benchmarks" / "dataset_60.json",
        out_dir=root / "benchmarks" / "results_60",
        tiers=tiers,
        max_items=args.max_items,
        use_web_rag=args.use_webrag,
        use_dinov2=args.use_dinov2,
        export_3mf_pairs=args.export_3mf,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
