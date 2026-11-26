"""Tests for AMS-count color budgeting (CIELAB merging) and glue-free mechanical joints."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spoolgenie.print_prep.color_quantizer import (
    quantize_mesh_to_ams_slots,
    resolve_ams_color_budget,
)
from spoolgenie.print_prep.functional_carver import apply_functional_ops_to_mesh
from spoolgenie.print_prep.split_joiner import (
    PrintPart,
    add_glue_free_joints_between_color_parts,
    arrange_parts_on_build_plate,
    split_oversize_part_with_joint,
)
from spoolgenie.schemas import DesignSpec, FilamentSlot, FunctionalOp, JointType, MulticolorMode


def test_ams_color_budget_merges_excess_colors_for_single_ams() -> None:
    spec = DesignSpec(
        ams_units=1,  # 4-color cap
        multicolor_mode=MulticolorMode.AMS,
        filaments=[
            FilamentSlot(slot=1, name="Charcoal", hex_color="#2B2D30"),
            FilamentSlot(slot=2, name="Dark Gray", hex_color="#35383D"),  # very close to slot 1!
            FilamentSlot(slot=3, name="Orange", hex_color="#F36C21"),
            FilamentSlot(slot=4, name="White", hex_color="#F2EFE9"),
            FilamentSlot(slot=5, name="Marine Blue", hex_color="#1E4C8A"),
            FilamentSlot(slot=6, name="Green", hex_color="#4D9A4E"),
        ],
    )
    res = resolve_ams_color_budget(spec)
    assert res.effective_mode == MulticolorMode.AMS
    assert len(res.active_filaments) == 4
    merged_actions = [d for d in res.decisions if d.action == "merged"]
    assert len(merged_actions) == 2
    assert all(1 <= v <= 4 for v in res.slot_remap.values())


def test_no_ams_switches_auto_to_glue_free_split_mode() -> None:
    spec = DesignSpec(
        ams_units=0,
        multicolor_mode=MulticolorMode.AUTO,
        filaments=[
            FilamentSlot(slot=1, name="Black", hex_color="#2B2D30"),
            FilamentSlot(slot=2, name="Orange", hex_color="#F36C21"),
        ],
    )
    res = resolve_ams_color_budget(spec)
    assert res.effective_mode == MulticolorMode.SPLIT_NO_AMS
    assert len(res.active_filaments) == 2


def test_two_ams_units_allow_six_colors() -> None:
    spec = DesignSpec(
        ams_units=2,  # 8-color cap
        multicolor_mode=MulticolorMode.AMS,
        filaments=[
            FilamentSlot(slot=i + 1, name=f"C{i+1}", hex_color=hx)
            for i, hx in enumerate(["#2B2D30", "#F36C21", "#F2EFE9", "#1E4C8A", "#4D9A4E", "#F4D03F"])
        ],
    )
    res = resolve_ams_color_budget(spec)
    assert len(res.active_filaments) == 6
    assert all(d.action == "kept" for d in res.decisions)


@pytest.mark.parametrize("joint_type", [JointType.DOVETAIL, JointType.SNAP_PIN, JointType.PRESS_TENON])
def test_glue_free_joints_between_color_parts_are_watertight(joint_type: JointType) -> None:
    main = trimesh.creation.box(extents=[60, 40, 20])
    main.apply_translation([0, 0, 10])
    accent = trimesh.creation.box(extents=[24, 16, 8])
    accent.apply_translation([0, 0, 22])

    joined = add_glue_free_joints_between_color_parts(
        [PrintPart("chassis", 1, main), PrintPart("badge", 2, accent)],
        joint_type=joint_type,
        clearance_mm=0.22,
    )
    assert len(joined) == 2
    for p in joined:
        assert len(p.mesh.faces) > 20
        assert p.mesh.is_watertight
        assert p.joint_type == joint_type.value

    laid_out = arrange_parts_on_build_plate(joined, spacing_mm=12.0)
    # Parts should sit at Z=0 and not overlap in X
    assert laid_out[0].mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-3)
    assert laid_out[1].mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-3)
    assert laid_out[1].mesh.bounds[0][0] > laid_out[0].mesh.bounds[1][0]


def test_oversize_bed_auto_split_with_dovetail() -> None:
    tall = trimesh.creation.box(extents=[40, 40, 220])
    tall.apply_translation([0, 0, 110])
    split = split_oversize_part_with_joint(
        PrintPart("tall_pillar", 1, tall),
        printer_id="bambu_a1_mini",  # 180 mm max Z
        joint_type=JointType.DOVETAIL,
    )
    assert len(split) == 2
    for p in split:
        assert p.mesh.is_watertight
        assert (p.mesh.bounds[1][2] - p.mesh.bounds[0][2]) < 165.0


def test_quantize_mesh_to_ams_slots() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=25.0)
    # Paint top half orange, bottom half dark gray
    vc = np.zeros((len(sphere.vertices), 4), dtype=np.uint8)
    vc[:, 3] = 255
    top = sphere.vertices[:, 2] >= 0
    vc[top, :3] = [243, 108, 33]
    vc[~top, :3] = [43, 45, 48]
    sphere.visual.vertex_colors = vc

    filaments = [
        FilamentSlot(slot=1, name="Black", hex_color="#2B2D30"),
        FilamentSlot(slot=2, name="Orange", hex_color="#F36C21"),
    ]
    face_slots, active = quantize_mesh_to_ams_slots(sphere, filaments)
    assert len(active) == 2
    assert set(int(s) for s in face_slots) == {1, 2}


@pytest.mark.parametrize(
    "op_name,params",
    [
        ("hollow_cavity", {"shape": "square", "width_ratio": 0.56, "depth_ratio": 0.62}),
        ("phone_slot", {"slot_width_mm": 80.0, "slot_thickness_mm": 14.0}),
        ("magsafe_recess", {"diameter_mm": 56.4, "depth_mm": 5.8, "face": "front"}),
        ("magnet_sockets", {"diameter_mm": 6.2, "depth_mm": 2.4}),
        ("cable_comb", {"slots": 3}),
        ("plinth_base", {"height_mm": 5.0}),
        ("keychain_loop", {"outer_radius_mm": 7.0, "inner_radius_mm": 3.4}),
    ],
)
def test_functional_carver_all_ops_stay_watertight(op_name: str, params: dict) -> None:
    cube_robot = trimesh.creation.box(extents=[96.0, 90.0, 110.0])
    cube_robot.apply_translation([0.0, 0.0, 55.0])
    carved, log = apply_functional_ops_to_mesh(cube_robot, [FunctionalOp(op=op_name, params=params)])
    assert len(log) == 1
    assert carved.is_watertight
    assert carved.bounds[0][2] == pytest.approx(0.0, abs=1e-2)

