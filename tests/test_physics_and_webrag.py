"""Tests for anisotropic FDM physics stability engine, precision closed-tread WALL-E CAD, and Web-RAG."""

from __future__ import annotations

import trimesh

from spoolgenie.engines.cad_engine import execute_build123d_script
from spoolgenie.engines.modular_cad_models import build_precision_walle_cad_code
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG
from spoolgenie.print_prep.physics_engine import evaluate_physics_stability
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
