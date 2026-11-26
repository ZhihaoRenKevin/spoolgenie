"""Tests for modular whole-part solid coloring and canon-first Bambu Lab official store recommendations."""

from __future__ import annotations

import manifold3d
from fastapi.testclient import TestClient

from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG
from spoolgenie.print_prep.color_quantizer import (
    find_closest_bambu_filament,
    match_canon_to_owned_filaments,
)
from spoolgenie.print_prep.functional_carver import _manifold_to_mesh
from spoolgenie.print_prep.modularizer import decompose_mesh_into_solid_modules
from spoolgenie.schemas import FilamentSlot
from spoolgenie.web.app import app


def _make_synthetic_robot_mesh():
    """Create a watertight robot-like solid (twin feet/treads + torso cube + arms + narrow neck + head)."""
    foot_l = manifold3d.Manifold.cube([24, 52, 26], center=True).translate([-26, 0, 13])
    foot_r = manifold3d.Manifold.cube([24, 52, 26], center=True).translate([26, 0, 13])
    torso = manifold3d.Manifold.cube([62, 54, 48], center=True).translate([0, 0, 48])
    arm_l = manifold3d.Manifold.cube([22, 16, 16], center=True).translate([-40, 0, 52])
    arm_r = manifold3d.Manifold.cube([22, 16, 16], center=True).translate([40, 0, 52])
    neck = manifold3d.Manifold.cylinder(16, 7, 7, circular_segments=24).translate([0, 0, 72])
    head = manifold3d.Manifold.cube([52, 26, 26], center=True).translate([0, 0, 96])
    man = foot_l + foot_r + torso + arm_l + arm_r + neck + head
    return _manifold_to_mesh(man)


def test_modular_solid_decomposition_creates_watertight_whole_color_modules() -> None:
    robot = _make_synthetic_robot_mesh()
    preset = CHARACTER_CATALOG["pixar_walle"]
    filaments = [
        FilamentSlot(slot=i + 1, name=n, hex_color=hx, role=r)
        for i, (n, hx, r) in enumerate(preset.filaments)
    ]

    parts = decompose_mesh_into_solid_modules(robot, filaments, base_name="WALL_E")
    assert len(parts) >= 3
    names = [p.name for p in parts]
    assert any("core_body" in n for n in names)
    assert any("lower_base" in n for n in names)
    assert any("upper_head" in n for n in names)

    for p in parts:
        assert p.mesh.is_watertight is True
        assert 1 <= p.slot <= len(filaments)

    # Canon color semantics: core_body gets yellow (slot 1), lower_base gets charcoal treads (slot 3)
    core = next(p for p in parts if p.name == "core_body")
    base = next(p for p in parts if p.name.startswith("lower_base"))
    assert core.slot == 1
    assert base.slot == 3


def test_canon_first_owned_filament_matching_and_bambu_store_links() -> None:
    preset = CHARACTER_CATALOG["pixar_walle"]
    canon = [
        FilamentSlot(slot=i + 1, name=n, hex_color=hx, role=r)
        for i, (n, hx, r) in enumerate(preset.filaments)
    ]
    owned = [
        FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30"),
        FilamentSlot(slot=2, name="Matte Ivory White", hex_color="#F2EFE9"),
    ]

    resolved, recs = match_canon_to_owned_filaments(canon, owned, max_delta_e=22.0)
    assert len(resolved) == len(canon)
    assert len(recs) == len(canon)

    by_role = {r.role: r for r in recs}
    # Charcoal & Ivory White match user's owned spools
    assert by_role["treads & lenses"].status == "owned_match"
    assert by_role["accents"].status == "owned_match"
    # Yellow body & Gray head have no close neighbor -> keep canon faithful & recommend official Bambu store!
    yellow_rec = by_role["cube body"]
    assert yellow_rec.status == "purchase_recommended"
    assert yellow_rec.delta_e > 22.0
    assert "store.bambulab.cn" in yellow_rec.store_url_cn
    assert "store.bambulab.com" in yellow_rec.store_url_global
    assert yellow_rec.bambu_series != "PLA Silk"  # Must recommend matte/basic yellow, NOT metallic Silk Gold!

    # Verify non-metallic yellow recommendation directly
    bambu_y, _ = find_closest_bambu_filament("#E8A825", prefer_name="Matte Lemon Yellow cube body")
    assert bambu_y.series in ("PLA Basic", "PLA Matte")


def test_api_match_filaments_endpoint() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/match-filaments",
        json={
            "canon_filaments": [
                {"slot": 1, "name": "Matte Lemon Yellow", "hex_color": "#E8A825", "role": "cube body"},
                {"slot": 2, "name": "Matte Charcoal", "hex_color": "#2B2D30", "role": "treads"},
            ],
            "owned_filaments": [
                {"slot": 1, "name": "Basic Black", "hex_color": "#18181A"},
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["status"] == "purchase_recommended"
    assert data["recommendations"][1]["status"] == "owned_match"
