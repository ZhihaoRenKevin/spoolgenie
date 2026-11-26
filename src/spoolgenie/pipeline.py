"""Shared high-level SpoolGenie pipeline orchestrator used identically by CLI, Python API, and Web UI."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from spoolgenie.agent.designer import ConversationalDesigner
from spoolgenie.config import SpoolGenieSettings, get_settings
from spoolgenie.engines.cad_engine import build_parametric_model
from spoolgenie.engines.concept_2d import generate_2d_concept, render_3d_turntable_sheet
from spoolgenie.engines.mesh_engine import generate_organic_mesh
from spoolgenie.knowledge.kb import PrintKnowledgeBase
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.print_prep.bambu_3mf import (
    export_colored_preview_glb,
    export_multipart_bambu_3mf,
    export_painted_bambu_3mf,
)
from spoolgenie.print_prep.color_quantizer import (
    match_canon_to_owned_filaments,
    quantize_mesh_to_ams_slots,
    resolve_ams_color_budget,
)
from spoolgenie.print_prep.functional_carver import apply_functional_ops_modular
from spoolgenie.print_prep.modularizer import decompose_mesh_into_solid_modules
from spoolgenie.print_prep.repair import (
    evaluate_dfm,
    repair_single_mesh,
    scale_and_ground_meshes,
)
from spoolgenie.print_prep.split_joiner import (
    PrintPart,
    add_glue_free_joints_between_color_parts,
    arrange_parts_on_build_plate,
    split_oversize_part_with_joint,
)
from spoolgenie.reference.makerworld import (
    analyze_reference_image,
    analyze_reference_mesh,
    inspect_makerworld_url,
    search_makerworld,
)
from spoolgenie.schemas import (
    ChatTurn,
    DesignSpec,
    EngineRoute,
    ExportBundle,
    JointType,
    MulticolorMode,
    ProjectState,
    ReferenceAsset,
    SplitPartInfo,
    WorkflowStage,
)
from spoolgenie.slicer.cli_slicer import run_slicer_or_estimate


class SpoolGenieSession:
    """Stateful project session persisted cleanly under `outputs/<project_id>/`."""

    def __init__(
        self,
        project_id: str | None = None,
        settings: SpoolGenieSettings | None = None,
        llm: UnifiedLLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or UnifiedLLMClient(self.settings)
        self.designer = ConversationalDesigner(self.llm)
        self.kb = PrintKnowledgeBase(self.settings.workspace_dir / "knowledge_store.jsonl")

        pid = project_id or f"pf_{uuid.uuid4().hex[:8]}"
        self.project_dir = (self.settings.workspace_dir / pid).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.project_dir / "project.json"

        if self.state_file.exists():
            self.state = ProjectState.model_validate_json(self.state_file.read_text(encoding="utf-8"))
        else:
            self.state = ProjectState(project_id=pid)
            self.state.spec.printer_id = self.settings.default_printer
            self.save()

    def save(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Stage 0: MakerWorld & Local Reference Inspiration
    # ------------------------------------------------------------------
    def search_makerworld_refs(self, query: str, max_results: int = 6) -> list[ReferenceAsset]:
        return search_makerworld(query, max_results=max_results)

    def add_reference_url(self, url: str) -> ReferenceAsset:
        ref = inspect_makerworld_url(url)
        self.state.references.append(ref)
        self.save()
        return ref

    def add_reference_file(self, src_path: str | Path) -> ReferenceAsset:
        src = Path(src_path)
        refs_dir = self.project_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        dst = refs_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

        if dst.suffix.lower() in (".stl", ".obj", ".glb", ".3mf"):
            ref = analyze_reference_mesh(dst)
        else:
            ref = analyze_reference_image(dst, llm=self.llm)

        self.state.references.append(ref)
        self.save()
        return ref

    # ------------------------------------------------------------------
    # Stage 1: Chat Clarification + 2D Concept Gate
    # ------------------------------------------------------------------
    def chat(
        self,
        user_message: str,
        auto_render_2d: bool = True,
        force_2d_backend: str | None = None,
        lora_path: str | None = None,
    ) -> dict[str, Any]:
        self.state.history.append(ChatTurn(role="user", content=user_message))
        new_spec, reply = self.designer.process_message(self.state, user_message)

        # Enrich with Character Preset (only when generic figurine without a curated catalog prompt) or MakerWorld RAG priors
        is_catalog_char = any(
            new_spec.category.startswith(p)
            for p in (
                "marvel_", "dc_", "disney_", "pixar_", "universal_", "dreamworks_",
                "scifi_", "starwars_", "hp_", "movie_", "anime_", "game_", "toy_",
                "animal_", "plant_", "decor_", "veh_", "food_", "hobby_", "myth_", "season_",
            )
        )
        char_match = self.kb.match_character_preset(user_message)
        if char_match and not is_catalog_char and new_spec.route in (EngineRoute.MESH, EngineRoute.HYBRID):
            preset_key, preset = char_match
            new_spec.prompt_2d = f"{preset['prompt_prefix']}. Subject: {new_spec.title} ({user_message})."
            self.state.extra["character_preset"] = preset_key
        else:
            kb_hits = self.kb.retrieve(f"{new_spec.category} {user_message}", domain="functional_makerworld", top_k=3)
            self.state.extra["kb_priors"] = [e.title for e in kb_hits]
            self.state.extra["kb_context"] = "\n".join(
                f"- {e.title}: {' '.join(e.design_rules[:2])} {e.cad_snippet}" for e in kb_hits
            )

        self.state.spec = new_spec
        self.state.stage = WorkflowStage.CONCEPT_REVIEW
        self.state.history.append(ChatTurn(role="assistant", content=reply))

        concept = None
        if auto_render_2d:
            concept = self.render_2d_concept(force_backend=force_2d_backend, lora_path=lora_path)

        self.save()
        return {"reply": reply, "spec": self.state.spec, "concept": concept, "state": self.state}

    def update_spec(self, patch: dict[str, Any], re_render_2d: bool = True) -> ProjectState:
        """Apply direct UI slider/color/AMS-count/joint tweaks before or after 3D generation."""
        data = self.state.spec.model_dump()
        for k, v in patch.items():
            if k == "dimensions" and isinstance(v, dict):
                data["dimensions"].update(v)
            elif k in data:
                data[k] = v
        self.state.spec = DesignSpec.model_validate(data)
        self.state.spec.confirmed_by_user = False
        if re_render_2d:
            self.render_2d_concept()
        self.save()
        return self.state

    def render_2d_concept(
        self,
        force_backend: str | None = None,
        lora_path: str | None = None,
    ):
        idx = len(self.state.concepts) + 1
        out_img = self.project_dir / f"concept_{idx:02d}.png"
        if self.state.spec.route in (EngineRoute.MESH, EngineRoute.HYBRID):
            self.llm.release_vram()
        concept = generate_2d_concept(
            self.state.spec,
            out_img,
            settings=self.settings,
            force_backend=force_backend,
            lora_path=lora_path,
        )
        concept.image_url = f"/api/projects/{self.state.project_id}/files/{out_img.name}"
        self.state.concepts.append(concept)
        self.save()
        return concept

    # ------------------------------------------------------------------
    # Stage 2 & 3: Confirmed 3D Generation + AMS Budget + Glue-Free Joints + 3MF Export
    # ------------------------------------------------------------------
    def confirm_and_build(
        self,
        custom_cad_code: str | None = None,
        force_mesh_backend: str | None = None,
        run_slicer: bool = True,
    ) -> ProjectState:
        """Execute 3D modeling, watertight repair, AMS color budgeting, glue-free joints, and 3MF/STL/STEP export."""
        spec = self.state.spec
        spec.confirmed_by_user = True
        latest_concept_path = self.state.concepts[-1].image_path if self.state.concepts else None

        # 0. Canon-First Palette -> Match Against User's Owned Filaments + Official Bambu Lab Store Recommendations!
        matched_filaments, filament_recs = match_canon_to_owned_filaments(
            spec.filaments,
            owned_filaments=spec.owned_filaments,
            max_delta_e=22.0,
        )
        effective_spec_for_budget = spec.model_copy(update={"filaments": matched_filaments})

        repaired_any = False
        face_slots_single: np.ndarray | None = None
        raw_parts: list[PrintPart] = []
        extra_func_parts: list[PrintPart] = []
        step_path: Path | None = None
        cad_script_path: Path | None = None

        # 1. Generate Raw 3D Geometry via True General-Purpose Dual-Engine Router
        if spec.route == EngineRoute.CAD:
            step_file = self.project_dir / f"{self.state.project_id}.step"
            cad_res = build_parametric_model(
                effective_spec_for_budget,
                step_output_path=step_file,
                llm=self.llm,
                custom_code=custom_cad_code,
                kb_context=str(self.state.extra.get("kb_context") or ""),
            )
            self.state.cad_code = cad_res.code_used
            self.state.extra["cad_used_fallback"] = cad_res.used_fallback
            cad_script_path = self.project_dir / f"{self.state.project_id}_cad.py"
            cad_script_path.write_text(cad_res.code_used, encoding="utf-8")
            step_path = step_file if step_file.exists() else None

            for cp in cad_res.parts:
                rep = repair_single_mesh(cp.mesh)
                repaired_any = repaired_any or rep.repaired
                raw_parts.append(PrintPart(name=cp.name, slot=cp.slot, mesh=rep.mesh))
        else:
            # MESH or HYBRID: Start with the TRUE 360-degree 3D sculpted object/character from Hunyuan3D-2 / 3D engine!
            self.llm.release_vram()
            mesh_res = generate_organic_mesh(
                effective_spec_for_budget, latest_concept_path, self.settings, force_backend=force_mesh_backend
            )
            rep = repair_single_mesh(mesh_res.mesh)
            repaired_any = rep.repaired
            body_mesh = scale_and_ground_meshes(
                [rep.mesh],
                target_w=spec.dimensions.width_mm,
                target_d=spec.dimensions.depth_mm,
                target_h=spec.dimensions.height_mm,
                preserve_aspect=True,
            )[0]

            # For HYBRID (e.g. WALL-E pen holder / Totoro phone dock / Iron Man MagSafe charger):
            # Carve exact subtractive cavities AND emit additive functional attachments as SEPARATE colored solid modules!
            if spec.route == EngineRoute.HYBRID and spec.functional_ops:
                body_mesh, extra_func_parts, ops_log = apply_functional_ops_modular(
                    body_mesh,
                    spec.functional_ops,
                    wall_thickness_mm=spec.dimensions.wall_thickness_mm,
                )
                self.state.extra["functional_ops_applied"] = ops_log

            raw_parts = [PrintPart(name=spec.title.replace(" ", "_"), slot=1, mesh=body_mesh)]
            self.state.extra["mesh_backend_used"] = mesh_res.backend
            self.state.extra["mesh_notes"] = mesh_res.notes

        # 2. Millimeter Scaling & Z=0 Grounding (keeps character + modular functional accessories aligned!)
        all_pre_scale = raw_parts + extra_func_parts
        scaled_meshes = scale_and_ground_meshes(
            [p.mesh for p in all_pre_scale],
            target_w=spec.dimensions.width_mm,
            target_d=spec.dimensions.depth_mm,
            target_h=spec.dimensions.height_mm,
            preserve_aspect=True,
        )
        for p, sm in zip(all_pre_scale, scaled_meshes):
            p.mesh = sm

        # 3. AMS-Count Color Budget Resolution (decides final colors & AMS vs No-AMS glue-free split!)
        slot_areas: dict[int, float] = {}
        for p in all_pre_scale:
            slot_areas[p.slot] = slot_areas.get(p.slot, 0.0) + float(p.mesh.area)

        budget = resolve_ams_color_budget(effective_spec_for_budget, slot_area_weights=slot_areas)
        active_filaments = budget.active_filaments
        effective_mode = budget.effective_mode

        for p in all_pre_scale:
            p.slot = budget.slot_remap.get(p.slot, 1)

        # Modular Whole-Part Coloring (Default!) vs Optional Legacy Surface Paint
        is_single_organic = spec.route in (EngineRoute.MESH, EngineRoute.HYBRID) and len(raw_parts) == 1
        coloring_method = "modular_solid"
        if is_single_organic:
            if spec.coloring_mode == "surface_paint" and len(active_filaments) > 1:
                coloring_method = "surface_paint"
                face_slots_single, active_filaments = quantize_mesh_to_ams_slots(
                    raw_parts[0].mesh,
                    active_filaments,
                    reference_image_path=latest_concept_path,
                    smooth_iters=8,
                )
                raw_parts.extend(extra_func_parts)
            else:
                # Decompose organic solid into clean watertight anatomical solid modules, each in 1 pure color!
                raw_parts = decompose_mesh_into_solid_modules(
                    raw_parts[0].mesh,
                    active_filaments,
                    base_name=spec.title.replace(" ", "_"),
                    extra_functional_parts=extra_func_parts,
                )
                face_slots_single = None
                coloring_method = "modular_solid"

        # 4. Oversize-Bed Auto-Split & No-AMS Glue-Free Mechanical Joints!
        assembled_parts = list(raw_parts)
        split_plate_parts: list[PrintPart] = []
        joint_used = JointType.NONE.value

        # Check oversize bed splitting on each part first
        bed_checked_parts: list[PrintPart] = []
        for p in assembled_parts:
            split_out = split_oversize_part_with_joint(
                p,
                printer_id=spec.printer_id,
                joint_type=spec.joint_type,
                clearance_mm=spec.joint_clearance_mm,
            )
            if len(split_out) > 1:
                joint_used = spec.joint_type.value
            bed_checked_parts.extend(split_out)
        assembled_parts = bed_checked_parts

        # No-AMS Split-By-Color with Glue-Free Mechanical Interlocking Joints!
        if effective_mode == MulticolorMode.SPLIT_NO_AMS and len(active_filaments) > 1:
            if len(assembled_parts) > 1:
                joined_parts = add_glue_free_joints_between_color_parts(
                    assembled_parts,
                    joint_type=spec.joint_type,
                    clearance_mm=spec.joint_clearance_mm,
                )
                joint_used = spec.joint_type.value
                assembled_parts = joined_parts
                split_plate_parts = arrange_parts_on_build_plate(joined_parts, spacing_mm=14.0)
            elif is_single_organic and face_slots_single is not None:
                # For single organic mesh in No-AMS mode: split horizontally at Z-mid with glue-free dovetail/pin
                # so base & top can be printed in contrasting colors without AMS or glue!
                m = assembled_parts[0].mesh
                bmin, bmax = m.bounds
                ext = bmax - bmin
                # Force a horizontal Z split using a temporary small bed height check
                fake_parts = split_oversize_part_with_joint(
                    assembled_parts[0],
                    printer_id="bambu_a1_mini",
                    joint_type=spec.joint_type,
                    clearance_mm=spec.joint_clearance_mm,
                )
                if len(fake_parts) == 2:
                    fake_parts[0].slot = active_filaments[0].slot
                    fake_parts[1].slot = active_filaments[1].slot
                    joint_used = spec.joint_type.value
                    assembled_parts = fake_parts
                    split_plate_parts = arrange_parts_on_build_plate(fake_parts, spacing_mm=14.0)
                    face_slots_single = None

        # 5. Export Bambu AMS .3mf, No-AMS Split-Plates .3mf, per-color STLs, unified STL, and Three.js GLB!
        threemf_path = self.project_dir / f"{self.state.project_id}_bambu.3mf"
        stl_path = self.project_dir / f"{self.state.project_id}.stl"
        preview_glb_path = self.project_dir / f"{self.state.project_id}_preview.glb"

        if face_slots_single is not None and len(assembled_parts) == 1:
            export_painted_bambu_3mf(
                assembled_parts[0].mesh,
                face_slots_single,
                active_filaments,
                threemf_path,
                model_title=spec.title,
            )
        else:
            export_multipart_bambu_3mf(
                assembled_parts,
                active_filaments,
                threemf_path,
                model_title=spec.title,
                as_separate_objects=False,
            )

        # Unified STL for universal slicers
        combined_stl_mesh = (
            trimesh.util.concatenate([p.mesh for p in assembled_parts])
            if len(assembled_parts) > 1
            else assembled_parts[0].mesh
        )
        combined_stl_mesh.export(stl_path)

        # Interactive Three.js colored GLB preview
        export_colored_preview_glb(
            assembled_parts,
            active_filaments,
            preview_glb_path,
            face_slots_override=face_slots_single,
        )

        # Render 4-angle shaded color turntable sheet of the ACTUAL 3D model for instant visual inspection
        turntable_path = self.project_dir / f"{self.state.project_id}_3d_turntable.png"
        slot_rgb_map = {
            f.slot: tuple(int(f.hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            for f in active_filaments
        }
        tt_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        if face_slots_single is not None and len(assembled_parts) == 1:
            m0 = assembled_parts[0].mesh
            for s_val in sorted(set(int(s) for s in face_slots_single)):
                f_idx = np.where(face_slots_single == s_val)[0]
                if len(f_idx) > 0:
                    tt_parts.append((np.asarray(m0.vertices), np.asarray(m0.faces[f_idx]), slot_rgb_map.get(s_val, (60, 65, 75))))
        else:
            for p in assembled_parts:
                tt_parts.append((np.asarray(p.mesh.vertices), np.asarray(p.mesh.faces), slot_rgb_map.get(p.slot, (60, 65, 75))))
        render_3d_turntable_sheet(
            tt_parts,
            turntable_path,
            title=spec.title,
            subtitle=f"Route: {spec.route.value.upper()} | Mode: {effective_mode.value} ({len(active_filaments)} colors) | Joint: {joint_used}",
        )

        # Export No-AMS split print-plate layout .3mf + individual per-color STLs!
        split_3mf_path: Path | None = None
        split_glb_path: Path | None = None
        per_color_stls: dict[str, str] = {}
        split_parts_info: list[SplitPartInfo] = []
        filament_by_slot = {f.slot: f for f in active_filaments}

        parts_for_inventory = split_plate_parts if split_plate_parts else assembled_parts
        if split_plate_parts:
            split_3mf_path = self.project_dir / f"{self.state.project_id}_no_ams_split_plates.3mf"
            split_glb_path = self.project_dir / f"{self.state.project_id}_split_preview.glb"
            export_multipart_bambu_3mf(
                split_plate_parts,
                active_filaments,
                split_3mf_path,
                model_title=f"{spec.title} (No-AMS Glue-Free Plates)",
                as_separate_objects=True,
            )
            export_colored_preview_glb(split_plate_parts, active_filaments, split_glb_path)

        for idx, part in enumerate(parts_for_inventory, start=1):
            fl = filament_by_slot.get(part.slot, active_filaments[0])
            p_stl = self.project_dir / f"part_{idx:02d}_slot{part.slot}_{part.name}.stl"
            part.mesh.export(p_stl)
            per_color_stls[f"slot{part.slot}_{part.name}"] = str(p_stl.resolve())
            ext = part.mesh.bounds[1] - part.mesh.bounds[0]
            split_parts_info.append(
                SplitPartInfo(
                    name=part.name,
                    slot=part.slot,
                    filament_name=fl.name,
                    hex_color=fl.hex_color,
                    joint_type=part.joint_type,
                    bbox_mm=(round(float(ext[0]), 1), round(float(ext[1]), 1), round(float(ext[2]), 1)),
                    stl_path=str(p_stl.resolve()),
                    plate_index=part.plate_index,
                )
            )

        # 6. Evaluate DFM Printability Report
        dfm = evaluate_dfm(
            [p.mesh for p in (split_plate_parts or assembled_parts)],
            spec=spec,
            repaired_any=repaired_any,
            ams_slots_used=len(active_filaments),
            effective_mode=effective_mode.value,
            joint_type_used=joint_used,
        )
        dfm.coloring_method = coloring_method
        dfm.color_decisions = budget.decisions
        dfm.filament_recommendations = filament_recs
        dfm.split_parts = split_parts_info

        # 7. Headless Slicer or Physics-Based Estimate
        target_for_slice = split_3mf_path if (effective_mode == MulticolorMode.SPLIT_NO_AMS and split_3mf_path) else threemf_path
        slice_res = (
            run_slicer_or_estimate(target_for_slice, self.project_dir, dfm, spec, self.settings)
            if run_slicer
            else None
        )

        self.state.dfm = dfm
        self.state.exports = ExportBundle(
            threemf_path=str(threemf_path.resolve()),
            split_plates_3mf_path=str(split_3mf_path.resolve()) if split_3mf_path else None,
            per_color_stls=per_color_stls,
            stl_path=str(stl_path.resolve()),
            step_path=str(step_path.resolve()) if step_path else None,
            preview_glb_path=str(preview_glb_path.resolve()),
            split_preview_glb_path=str(split_glb_path.resolve()) if split_glb_path else None,
            turntable_preview_path=str(turntable_path.resolve()) if turntable_path.exists() else None,
            cad_script_path=str(cad_script_path.resolve()) if cad_script_path else None,
            gcode_3mf_path=slice_res.gcode_3mf_path if slice_res else None,
            slicer_used=slice_res.slicer_used if slice_res else None,
            estimated_print_minutes=slice_res.estimated_print_minutes if slice_res else None,
            estimated_filament_g=slice_res.estimated_filament_g if slice_res else None,
            estimated_flush_g=slice_res.estimated_flush_g if slice_res else None,
            slicer_notes=slice_res.notes if slice_res else "",
        )
        self.state.stage = WorkflowStage.EXPORTED
        self.save()
        return self.state
