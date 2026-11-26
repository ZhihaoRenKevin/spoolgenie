"""Tests for anisotropic FDM physics stability engine, precision closed-tread WALL-E CAD, and Web-RAG."""

from __future__ import annotations

import trimesh

from spoolgenie.engines.cad_engine import execute_build123d_script
from spoolgenie.engines.modular_cad_models import build_precision_walle_cad_code
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG
from spoolgenie.print_prep.bambu_3mf import export_multipart_bambu_3mf
from spoolgenie.print_prep.modularizer import decompose_mesh_into_solid_modules
from spoolgenie.print_prep.physics_engine import (
    evaluate_physics_stability,
    optimize_assembly_with_physics,
)
from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.reference.web_character_rag import _extract_chromatic_canon_colors
from spoolgenie.schemas import DesignSpec, FilamentSlot, FunctionalOp


def test_precision_walle_closed_treads_and_physics_stability() -> None:
    preset = CHARACTER_CATALOG["pixar_walle"]
    spec = DesignSpec(
        title="WALL-E Desk Pen Holder",
        category="pixar_walle",
        filaments=[
            FilamentSlot(slot=i + 1, name=n, hex_color=hx, role=r)
            for i, (n, hx, r) in enumerate(preset.filaments)
        ],
        functional_ops=[FunctionalOp(op="hollow_cavity", params={"shape": "square"})],
    )
    code = build_precision_walle_cad_code(spec)
    cad_parts = execute_build123d_script(code)
    assert len(cad_parts) == 4
    parts = [PrintPart(name=cp.name, slot=cp.slot, mesh=cp.mesh) for cp in cad_parts]
    for p in parts:
        assert p.mesh.is_watertight is True

    report = evaluate_physics_stability(parts, spec)
    assert report.base_closed_solid is True
    assert report.flat_bottom_area_mm2 > 2000.0
    assert report.tilt_angle_loaded_deg >= 20.0
    assert report.safety_factor_z >= 5.0
    assert report.stability_score >= 85
    assert report.overall_verdict == "stable"


def test_physics_catches_top_heavy_unstable_needle_and_magsafe_pla_heat() -> None:
    # Unstable thin needle balanced on tiny tip
    needle = trimesh.creation.cone(radius=3.0, height=160.0)
    needle.apply_transform(trimesh.transformations.rotation_matrix(3.14159, [1, 0, 0]))
    needle.apply_translation([0, 0, -needle.bounds[0][2]])
    spec_magsafe = DesignSpec(
        title="MagSafe Wireless Charger Stand",
        filaments=[FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30", material="PLA")],
        functional_ops=[FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4})],
    )
    rep = evaluate_physics_stability([needle], spec_magsafe)
    assert rep.stability_score < 85
    assert any(c.name == "Thermal HDT Suitability" and c.status in ("warn", "fail") for c in rep.checks)
    assert any(c.name == "Base Closure & Bed Contact" and c.status in ("warn", "fail") for c in rep.checks)


def test_chromatic_canon_color_extractor(tmp_path) -> None:
    from PIL import Image

    img = Image.new("RGB", (128, 128), (255, 255, 255))
    for y in range(24, 104):
        for x in range(24, 64):
            img.putpixel((x, y), (240, 180, 24))  # yellow half
        for x in range(64, 104):
            img.putpixel((x, y), (40, 44, 52))   # charcoal half
    p = tmp_path / "sample.png"
    img.save(p)
    cols = _extract_chromatic_canon_colors([p], n_colors=2)
    assert len(cols) == 2
    assert all(c.startswith("#") and len(c) == 7 for c in cols)


def test_2d_voted_whole_module_color_accuracy(tmp_path) -> None:
    """Verify 2D-to-3D whole-module chromatic voting assigns brown body + orange top crown with zero black bleed."""
    import manifold3d
    from PIL import Image

    from spoolgenie.print_prep.functional_carver import _manifold_to_mesh

    # Synthetic plump brown capybara body (Z=0..82) with pinched orange tangerine perched on top (Z=82..102)
    body = manifold3d.Manifold.cylinder(82.0, 34.0, 26.0, circular_segments=36)
    neck_pinch = manifold3d.Manifold.cylinder(6.0, 10.0, 12.0, circular_segments=32).translate([0.0, 0.0, 80.0])
    tangerine = manifold3d.Manifold.sphere(13.0, circular_segments=32).translate([0.0, 0.0, 94.0])
    mesh = _manifold_to_mesh(body + neck_pinch + tangerine)

    # Matching synthetic front 2D studio image: white backdrop, orange top 20%, brown lower 80%
    img = Image.new("RGB", (200, 200), (252, 252, 252))
    for y in range(20, 56):
        for x in range(78, 122):
            img.putpixel((x, y), (243, 108, 33))  # bright orange tangerine
    for y in range(56, 184):
        for x in range(52, 148):
            img.putpixel((x, y), (156, 115, 75))  # matte latte brown fur
    img_path = tmp_path / "capy_concept.png"
    img.save(img_path)

    filaments = [
        FilamentSlot(slot=1, name="Matte Latte Brown", hex_color="#9C734B", role="fur body"),
        FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21", role="tangerine on head"),
        FilamentSlot(slot=3, name="Matte Grass Green", hex_color="#4D9A4E", role="leaf"),
        FilamentSlot(slot=4, name="Basic Black", hex_color="#231F1D", role="sleepy eyes"),
    ]

    modules = decompose_mesh_into_solid_modules(
        mesh,
        filaments,
        base_name="capybara",
        reference_image_path=img_path,
    )
    assert len(modules) >= 2
    by_name = {m.name: m for m in modules}
    assert "top_crown" in by_name
    assert by_name["top_crown"].slot == 2  # Mandarin Orange tangerine!
    assert by_name["core_body"].slot == 1  # Latte Brown fur body!
    assert all(m.slot != 4 for m in modules)  # No mistaken black pants on brown capybara!


def test_closed_loop_physics_auto_stabilizes_and_injects_3mf_settings(tmp_path) -> None:
    """Verify closed-loop optimizer auto-adds stability_plinth to top-heavy geometry and injects .3mf settings."""
    import json
    import zipfile

    # Unstable inverted cone balancing on tiny tip (radius=3 -> 24, height=130 mm)
    cone = trimesh.creation.cone(radius=22.0, height=130.0)
    cone.apply_transform(trimesh.transformations.rotation_matrix(3.1415926, [1, 0, 0]))
    cone.apply_translation([0.0, 0.0, -cone.bounds[0][2]])
    spec = DesignSpec(
        title="Iron Man MagSafe Charger Dock",
        filaments=[
            FilamentSlot(slot=1, name="Matte Scarlet Red", hex_color="#B81D24", material="PLA", role="armor"),
            FilamentSlot(slot=2, name="Matte Charcoal", hex_color="#2B2D30", material="PLA", role="joints & base"),
        ],
        functional_ops=[FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4})],
    )
    parts_in = [PrintPart(name="body", slot=1, mesh=cone)]
    before = evaluate_physics_stability(parts_in, spec)
    parts_out, after = optimize_assembly_with_physics(parts_in, spec, filaments=spec.filaments, auto_stabilize=True)

    assert any(p.name == "stability_plinth" for p in parts_out)
    assert after.base_closed_solid is True
    assert after.tilt_angle_loaded_deg > before.tilt_angle_loaded_deg + 10.0
    assert after.stability_score > before.stability_score
    assert after.recommended_wall_loops >= 4
    assert after.recommended_infill_pct >= 25
    assert after.recommended_material == "PETG"
    assert len(after.auto_remediations) >= 2

    out_3mf = tmp_path / "opt.3mf"
    export_multipart_bambu_3mf(
        parts_out,
        spec.filaments,
        out_3mf,
        wall_loops=after.recommended_wall_loops,
        sparse_infill_density=f"{after.recommended_infill_pct}%",
    )
    with zipfile.ZipFile(out_3mf, "r") as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8"))
        assert int(cfg["wall_loops"]) == after.recommended_wall_loops
        assert cfg["sparse_infill_density"] == f"{after.recommended_infill_pct}%"


def test_curated_character_prompt_and_palette_not_corrupted_by_object_nouns() -> None:
    """'头顶橘子的卡皮巴拉' must keep all 4 canonical filaments & curated brown-capybara-with-orange prompt!"""
    from spoolgenie.agent.designer import ConversationalDesigner
    from spoolgenie.config import SpoolGenieSettings
    from spoolgenie.llm.client import UnifiedLLMClient
    from spoolgenie.schemas import ProjectState

    designer = ConversationalDesigner(UnifiedLLMClient(SpoolGenieSettings(llm_provider="fallback")))
    st = ProjectState(project_id="test_capy")
    spec, _ = designer.process_message(st, "做一个头顶橘子的卡皮巴拉水豚桌面摆件，高115mm，1台AMS四色打印")
    assert spec.category == "animal_capybara"
    assert len(spec.filaments) == 4
    assert spec.filaments[0].name == "Matte Latte Brown"
    assert spec.filaments[1].name == "Matte Mandarin Orange"
    assert "tangerine" in spec.prompt_2d.lower()
    assert "brown capybara" in spec.prompt_2d.lower()
