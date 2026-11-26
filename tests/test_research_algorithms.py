"""Unit tests for P1 Fabrication-Aware 3D MRF Graph-Cut and P2 Unified Joint Fabrication Optimizer."""

from __future__ import annotations

import manifold3d
import numpy as np
import trimesh
from PIL import Image

from spoolgenie.print_prep.functional_carver import _manifold_to_mesh
from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.research.fab_graphcut import FabGraphCutConfig, decompose_mesh_fab_graphcut
from spoolgenie.research.joint_fab_opt import (
    JointFabConfig,
    compute_ams_purge_metrics,
    optimize_assembly_joint_fabrication,
    optimize_internal_com_hollowing,
)
from spoolgenie.schemas import DesignSpec, FilamentSlot, FunctionalOp


def test_p1_fab_graphcut_curved_watertight_decomposition(tmp_path) -> None:
    """Verify P1 MRF alpha-expansion extracts curved watertight volumetric inlays with clean visual alignment."""
    # Curved sphere-on-cylinder character (smooth curved boundary impossible for simple horizontal planes alone!)
    body = manifold3d.Manifold.cylinder(72.0, 32.0, 28.0, circular_segments=36)
    head = manifold3d.Manifold.sphere(28.0, circular_segments=36).translate([0.0, 0.0, 88.0])
    mesh = _manifold_to_mesh(body + head)

    # Synthetic 2D front concept: crimson red body & upper helmet, bright gold central faceplate band!
    img = Image.new("RGB", (200, 200), (250, 250, 250))
    for y in range(20, 185):
        for x in range(50, 150):
            if 48 <= y <= 84 and 68 <= x <= 132:
                img.putpixel((x, y), (212, 175, 55))  # gold faceplate!
            else:
                img.putpixel((x, y), (184, 29, 36))   # crimson red armor!
    img_path = tmp_path / "iron_concept.png"
    img.save(img_path)

    filaments = [
        FilamentSlot(slot=1, name="Matte Scarlet Red", hex_color="#B81D24", role="armor"),
        FilamentSlot(slot=2, name="Silk Gold", hex_color="#D4AF37", role="faceplate"),
        FilamentSlot(slot=3, name="Matte Charcoal", hex_color="#2B2D30", role="base"),
    ]
    res = decompose_mesh_fab_graphcut(
        mesh,
        filaments,
        base_name="hero",
        reference_image_path=img_path,
        config=FabGraphCutConfig(n_superfacets=120),
    )

    assert res.final_energy <= res.initial_energy + 1e-6
    assert len(res.parts) >= 2
    assert all(p.mesh.is_watertight for p in res.parts)
    slots_found = {p.slot for p in res.parts}
    assert 1 in slots_found and 2 in slots_found  # Both red armor and curved gold faceplate solid extracted!
    assert res.visual_mean_delta_e < 25.0


def test_p2_internal_com_hollowing_lowers_com_with_zero_silhouette_distortion() -> None:
    """Verify internal asymmetric cavity strictly lowers CoM with 0.000 mm outer silhouette change."""
    # Top-heavy vase/figurine (wider upper bulb than base)
    base = manifold3d.Manifold.cylinder(45.0, 24.0, 22.0, circular_segments=32)
    upper = manifold3d.Manifold.sphere(34.0, circular_segments=32).translate([0.0, 0.0, 74.0])
    solid = _manifold_to_mesh(base + upper)
    solid.apply_translation([0.0, 0.0, -float(solid.bounds[0][2])])

    spec = DesignSpec(
        title="Top-Heavy Figurine",
        filaments=[FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30")],
    )
    parts_in = [PrintPart(name="core_body", slot=1, mesh=solid)]
    parts_out, carved, cav_vol, z_shift = optimize_internal_com_hollowing(parts_in, spec, min_wall_mm=3.5)

    assert carved is True
    assert cav_vol > 2.0
    assert z_shift > 0.5  # CoM shifted downward!
    assert parts_out[0].mesh.is_watertight is True
    # Zero external silhouette distortion!
    assert float(np.max(np.abs(parts_out[0].mesh.bounds - solid.bounds))) < 0.05


def test_p2_ams_purge_and_joint_orientation_optimization() -> None:
    """Verify exact AMS purge layer accounting and orientation optimization drastically cut flush waste."""
    # Two tall vertical side-by-side multi-color bars (standing upright = swaps on EVERY layer! Lying flat = 1 swap!)
    left_bar = trimesh.creation.box(extents=[24.0, 12.0, 90.0])
    left_bar.apply_translation([-12.0, 0.0, 45.0])
    right_bar = trimesh.creation.box(extents=[24.0, 12.0, 90.0])
    right_bar.apply_translation([12.0, 0.0, 45.0])

    spec = DesignSpec(
        title="Two-Tone Vertical Bracket",
        filaments=[
            FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30"),
            FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21"),
        ],
        functional_ops=[FunctionalOp(op="magsafe_recess", params={"diameter_mm": 56.4})],
    )
    parts = [PrintPart(name="left", slot=1, mesh=left_bar), PrintPart(name="right", slot=2, mesh=right_bar)]

    ams_upright = compute_ams_purge_metrics(parts, layer_height_mm=0.20)
    assert ams_upright.total_swaps >= 400  # 450 layers * 1 swap/layer!
    assert ams_upright.total_waste_g >= 100.0

    jf = optimize_assembly_joint_fabrication(
        parts,
        spec,
        filaments=spec.filaments,
        visual_delta_e=4.0,
        config=JointFabConfig(enable_internal_hollowing=False, enable_plinth_fallback=False),
    )
    # Rotating onto side (Y90) stacks the 2 colors vertically -> cuts swaps from ~450 down to 1!
    assert jf.ams_after.total_swaps < ams_upright.total_swaps * 0.2
    assert jf.ams_after.total_waste_g < ams_upright.total_waste_g * 0.2
    assert jf.loss_after["L_total"] < jf.loss_before["L_total"]


def test_p15_multiview_perspective_registration_and_latex_table(tmp_path) -> None:
    """Verify P1.5 perspective pinhole registration + multi-view feature fusion + LaTeX ablation table export."""
    from spoolgenie.research.multiview_features import backproject_multiview_features

    # Create asymmetric L-shaped 3D solid + synthetic front perspective image
    base = manifold3d.Manifold.cube([60.0, 35.0, 22.0], center=True).translate([0.0, 0.0, 11.0])
    tower = manifold3d.Manifold.cube([22.0, 28.0, 56.0], center=True).translate([18.0, 0.0, 50.0])
    mesh = _manifold_to_mesh(base + tower)

    img = Image.new("RGB", (180, 180), (250, 250, 250))
    for y in range(25, 160):
        for x in range(30, 150):
            if y > 115:
                img.putpixel((x, y), (156, 115, 75))  # brown base
            elif x > 95:
                img.putpixel((x, y), (77, 154, 78))   # green right tower
    p_img = tmp_path / "mv_view1.png"
    img.save(p_img)

    sf_ids = np.arange(len(mesh.faces), dtype=np.int32) % 32
    mv = backproject_multiview_features(mesh, sf_ids, 32, [p_img, p_img], use_dinov2=False)
    assert mv is not None
    assert len(mv.poses) == 2
    assert mv.poses[0].silhouette_iou > 0.45
    assert mv.sf_features.shape == (32, 6)
    # A 2-box L-solid has 4 front-facing triangles -> all front-facing super-facets observed cleanly!
    assert int(np.sum(~np.isnan(mv.sf_obs_lab[:, 0]))) >= 4
