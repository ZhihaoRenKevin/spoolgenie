"""Mesh watertight repair, millimeter scaling, flat-base grounding, and FDM printability (DFM) checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from spoolgenie.config import PRINTER_PROFILES
from spoolgenie.schemas import DesignSpec, DFMCheckItem, DFMReport

PLA_DENSITY_G_PER_CM3 = 1.24
EFFECTIVE_INFILL_FACTOR = 0.34  # Approximate shell + 15% gyroid infill solid fraction


@dataclass
class RepairOutcome:
    mesh: trimesh.Trimesh
    repaired: bool


def repair_single_mesh(mesh: trimesh.Trimesh) -> RepairOutcome:
    """Clean degenerate faces, merge vertices, repair holes/normals, and verify manifoldness."""
    m = mesh.copy()
    was_watertight = bool(m.is_watertight)

    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()

    repaired = False
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_winding(m)
        trimesh.repair.fix_inversion(m)
        repaired = bool(m.is_watertight) and not was_watertight

    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return RepairOutcome(mesh=m, repaired=repaired)


def scale_and_ground_meshes(
    meshes: list[trimesh.Trimesh],
    target_w: float,
    target_d: float,
    target_h: float,
    preserve_aspect: bool = True,
    flat_base_cut_mm: float = 0.0,
) -> list[trimesh.Trimesh]:
    """Scale an assembly cleanly to millimeter target dimensions and seat bottom at Z = 0, centered on X/Y."""
    if not meshes:
        return []

    bounds_min = np.min([m.bounds[0] for m in meshes], axis=0)
    bounds_max = np.max([m.bounds[1] for m in meshes], axis=0)
    ext = np.maximum(bounds_max - bounds_min, 1e-4)

    if preserve_aspect:
        # Uniform scale to fit inside target box while hitting primary target height/width accurately
        scale_val = min(target_w / ext[0], target_d / ext[1], target_h / ext[2])
        # Only rescale if off by more than 2% (keeps exact CAD dimensions untouched!)
        if abs(scale_val - 1.0) > 0.02:
            s_vec = np.array([scale_val, scale_val, scale_val], dtype=np.float64)
        else:
            s_vec = np.ones(3, dtype=np.float64)
    else:
        s_vec = np.array([target_w / ext[0], target_d / ext[1], target_h / ext[2]], dtype=np.float64)

    out: list[trimesh.Trimesh] = []
    for m in meshes:
        mc = m.copy()
        if not np.allclose(s_vec, 1.0):
            mc.apply_scale(s_vec)
        out.append(mc)

    new_min = np.min([m.bounds[0] for m in out], axis=0)
    new_max = np.max([m.bounds[1] for m in out], axis=0)
    cx = (new_min[0] + new_max[0]) * 0.5
    cy = (new_min[1] + new_max[1]) * 0.5
    z0 = new_min[2] + flat_base_cut_mm

    grounded: list[trimesh.Trimesh] = []
    for m in out:
        m.apply_translation([-cx, -cy, -z0])
        if flat_base_cut_mm > 0.0:
            # Clamp vertices slightly below Z=0 to Z=0 for a crisp bed contact footprint
            v = m.vertices.copy()
            v[:, 2] = np.maximum(v[:, 2], 0.0)
            m.vertices = v
        grounded.append(m)
    return grounded


def evaluate_dfm(
    meshes: list[trimesh.Trimesh],
    spec: DesignSpec,
    repaired_any: bool = False,
    ams_slots_used: int = 1,
    effective_mode: str = "ams",
    joint_type_used: str = "none",
) -> DFMReport:
    """Compute geometric printability checks (watertight, build plate, overhangs >45 deg, mass)."""
    printer = PRINTER_PROFILES.get(spec.printer_id, PRINTER_PROFILES["bambu_p1s"])

    if not meshes:
        return DFMReport(
            watertight=False,
            manifold_repaired=False,
            triangle_count=0,
            vertex_count=0,
            bbox_mm=(0.0, 0.0, 0.0),
            volume_cm3=0.0,
            estimated_mass_g=0.0,
            overhang_face_ratio=0.0,
            fits_build_plate=False,
            printer_id=printer.id,
            ams_slots_used=ams_slots_used,
            ams_units_configured=spec.ams_units,
            effective_multicolor_mode=effective_mode,
            joint_type_used=joint_type_used,
            checks=[DFMCheckItem(name="Geometry", status="fail", detail="No triangles produced.")],
        )

    bmin = np.min([m.bounds[0] for m in meshes], axis=0)
    bmax = np.max([m.bounds[1] for m in meshes], axis=0)
    ext = bmax - bmin
    bbox = (round(float(ext[0]), 2), round(float(ext[1]), 2), round(float(ext[2]), 2))

    tri_count = sum(len(m.faces) for m in meshes)
    vert_count = sum(len(m.vertices) for m in meshes)
    all_watertight = all(bool(m.is_watertight) for m in meshes)
    vol_mm3 = sum(abs(float(m.volume)) for m in meshes)
    vol_cm3 = round(vol_mm3 / 1000.0, 2)
    est_mass_g = round(max(1.0, vol_cm3 * PLA_DENSITY_G_PER_CM3 * EFFECTIVE_INFILL_FACTOR), 1)

    # Compute area-weighted overhang ratio (downward faces steeper than 45 deg off vertical, excluding bed contact Z<=0.5mm)
    total_area = 0.0
    overhang_area = 0.0
    sin_45 = math.sin(math.radians(45.0))
    for m in meshes:
        if len(m.faces) == 0:
            continue
        areas = m.area_faces
        normals = m.face_normals
        centroids_z = m.triangles_center[:, 2]
        total_area += float(areas.sum())
        # Downward normal z < -sin(45 deg) and lifted above the bed
        is_overhang = (normals[:, 2] < -sin_45) & (centroids_z > 0.8)
        overhang_area += float(areas[is_overhang].sum())

    overhang_ratio = round(overhang_area / max(total_area, 1e-6), 4)
    needs_supports = overhang_ratio > 0.06

    fits_bed = (
        bbox[0] <= printer.bed_x_mm
        and bbox[1] <= printer.bed_y_mm
        and bbox[2] <= printer.bed_z_mm
    )

    checks: list[DFMCheckItem] = [
        DFMCheckItem(
            name="Watertight Manifold",
            status="pass" if all_watertight else "warn",
            detail=(
                "All parts are closed watertight manifolds."
                if all_watertight
                else "Non-manifold edges detected; slicer auto-repair recommended."
            ),
        ),
        DFMCheckItem(
            name="Build Plate Fit",
            status="pass" if fits_bed else "warn",
            detail=(
                f"{bbox[0]} × {bbox[1]} × {bbox[2]} mm fits {printer.name} ({printer.bed_x_mm:.0f}×{printer.bed_y_mm:.0f}×{printer.bed_z_mm:.0f} mm)."
                if fits_bed
                else f"{bbox[0]} × {bbox[1]} × {bbox[2]} mm exceeds {printer.name} bed; auto-split with glue-free joints recommended/applied."
            ),
        ),
        DFMCheckItem(
            name="Overhangs (>45°)",
            status="pass" if not needs_supports else "warn",
            detail=(
                f"Low overhang ratio ({overhang_ratio*100:.1f}%) — prints cleanly without supports."
                if not needs_supports
                else f"Overhang area is {overhang_ratio*100:.1f}% — enable Auto Tree Supports in Bambu Studio / OrcaSlicer."
            ),
        ),
        DFMCheckItem(
            name="AMS & Multicolor Strategy",
            status="pass",
            detail=(
                f"Mode={effective_mode}, AMS units={spec.ams_units} ({ams_slots_used} color(s)), mechanical joint={joint_type_used}."
            ),
        ),
    ]

    return DFMReport(
        watertight=all_watertight,
        manifold_repaired=repaired_any,
        triangle_count=tri_count,
        vertex_count=vert_count,
        bbox_mm=bbox,
        volume_cm3=vol_cm3,
        estimated_mass_g=est_mass_g,
        overhang_face_ratio=overhang_ratio,
        needs_supports=needs_supports,
        fits_build_plate=fits_bed,
        printer_id=printer.id,
        ams_slots_used=ams_slots_used,
        ams_units_configured=spec.ams_units,
        effective_multicolor_mode=effective_mode,
        joint_type_used=joint_type_used,
        checks=checks,
    )
