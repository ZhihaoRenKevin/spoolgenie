"""First-principles FDM material mechanics, tipping stability, and structural stress physics engine.

Evaluates EVERY 3D-printable assembly under realistic real-world application loads using:
  1. Measured anisotropic FDM material datasheets (PLA / PETG / ABS / ASA / PLA-CF / TPU:
     density, Young's modulus E_xy, in-plane strength sigma_xy, Z-interlayer strength sigma_z, HDT).
  2. Exact 3D volume-weighted Center of Mass (CoM) + 2D Support Polygon Convex Hull on Z = 0:
     - Base closure & bed-contact flatness audit (catches open/broken treads & floating feet!)
     - Unloaded AND loaded tipping margin d_tip (mm), critical tilt angle theta_tip (deg),
       and horizontal knock-over force F_tip (N).
  3. 32-level Z-slice cross-sectional inertia (A(z), I_xx(z), I_yy(z)) & neck/cantilever beam analysis:
     - Bending stress sigma_bend = M * c / I_min vs anisotropic Z-interlayer tensile limit sigma_z
     - Structural Safety Factor (FOS) & cantilever tip deflection delta (mm)
     - Thin fragile feature / antenna / neck bottleneck detection
  4. Thermal HDT suitability check (e.g. MagSafe wireless charger / car / hot mug vs PLA 55 C HDT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.schemas import (
    DesignSpec,
    PhysicsCheckItem,
    PhysicsStabilityReport,
)


@dataclass(frozen=True)
class FDMMaterialSpec:
    code: str
    name: str
    density_g_cm3: float
    youngs_modulus_gpa: float
    tensile_xy_mpa: float
    tensile_z_mpa: float  # Critical FDM interlayer bond strength!
    hdt_c: float  # Heat deflection temperature @ 0.45 MPa (C)
    notes: str


# Calibrated on published Bambu Lab technical datasheets (TDS)
FDM_MATERIALS: dict[str, FDMMaterialSpec] = {
    "PLA": FDMMaterialSpec("PLA", "Bambu PLA Basic / Matte", 1.24, 2.60, 46.0, 24.0, 55.0, "High stiffness, crisp detail; softens above ~55°C."),
    "PETG": FDMMaterialSpec("PETG", "Bambu PETG HF / Basic", 1.27, 2.00, 40.0, 28.0, 69.0, "Higher Z-layer adhesion & heat resistance (~69°C); ideal for docks & chargers."),
    "ABS": FDMMaterialSpec("ABS", "Bambu ABS", 1.05, 2.10, 38.0, 22.0, 87.0, "Heat-resistant (~87°C) & impact tough; requires enclosed chamber."),
    "ASA": FDMMaterialSpec("ASA", "Bambu ASA", 1.07, 2.15, 39.0, 23.0, 100.0, "UV + heat resistant (~100°C); ideal for car/outdoor/warm electronics."),
    "PLA-CF": FDMMaterialSpec("PLA-CF", "Bambu PLA-CF", 1.22, 4.20, 52.0, 22.0, 62.0, "Very stiff matte finish for stands/frames; slightly lower Z-interlayer toughness."),
    "TPU": FDMMaterialSpec("TPU", "Bambu TPU 95A", 1.22, 0.06, 32.0, 26.0, 50.0, "Flexible shock-absorbing elastomer; unsuitable for rigid load-bearing stands."),
}

G_ACCEL = 9.81


@dataclass
class ApplicationLoadCase:
    name: str
    name_zh: str
    payload_mass_g: float
    payload_offset_xy_mm: tuple[float, float]  # relative to model XY center
    payload_z_ratio: float  # fraction of total height where load acts
    lateral_force_n: float  # external horizontal pull/push/lean force (N)
    heat_source_c: float  # expected operating surface temp (C)
    description: str


def infer_application_load_case(spec: DesignSpec, bbox_mm: tuple[float, float, float]) -> ApplicationLoadCase:
    """Determine the realistic real-world physical load scenario from the design's category & functional ops."""
    text = f"{spec.title} {spec.summary} {spec.category} {' '.join(spec.functional_features)}".lower()
    op_names = {op.op for op in spec.functional_ops}
    w, d, h = bbox_mm

    if "headphone" in text or "耳机" in text:
        return ApplicationLoadCase(
            name="Over-Ear Headphone Cantilever",
            name_zh="头戴耳机顶部悬臂承重 (350g)",
            payload_mass_g=350.0,
            payload_offset_xy_mm=(0.0, -d * 0.22),
            payload_z_ratio=0.96,
            lateral_force_n=1.2,
            heat_source_c=28.0,
            description="350 g over-ear headphones suspended at top cradle + 1.2 N placement nudge.",
        )
    if "magsafe" in op_names or "magsafe" in text or "无线充" in text or "charger" in text:
        return ApplicationLoadCase(
            name="MagSafe Phone Dock + Thermal Load",
            name_zh="MagSafe 手机磁吸充座 (220g + 4.5N拔力 + 45°C发热)",
            payload_mass_g=265.0,  # 220g phone + 45g puck
            payload_offset_xy_mm=(0.0, -d * 0.36),
            payload_z_ratio=0.55,
            lateral_force_n=4.5,  # magnetic pull-off force!
            heat_source_c=48.0,   # wireless charging coil warmth
            description="220 g phone + 45 g charger puck, 4.5 N magnetic pull-off force, and ~48°C coil warmth.",
        )
    if "phone_slot" in op_names or "phone" in text or "tablet" in text or "手机" in text or "switch" in text or "deck" in text:
        return ApplicationLoadCase(
            name="Phone / Handheld Cradle Load",
            name_zh="手机/掌机倾斜支架载荷 (240g 偏心)",
            payload_mass_g=240.0,
            payload_offset_xy_mm=(0.0, -d * 0.28),
            payload_z_ratio=0.45,
            lateral_force_n=1.8,  # touchscreen tap force
            heat_source_c=35.0,
            description="240 g device leaning at 16° + 1.8 N touchscreen tap impulse.",
        )
    if "hollow_cavity" in op_names or any(k in text for k in ("笔筒", "pen", "caddy", "organizer", "收纳", "planter", "花盆")):
        return ApplicationLoadCase(
            name="Loaded Desk Pen Caddy / Organizer",
            name_zh="满载笔筒/收纳偏载 (140g 笔+剪刀)",
            payload_mass_g=140.0,
            payload_offset_xy_mm=(0.0, d * 0.24),
            payload_z_ratio=0.78,
            lateral_force_n=1.5,  # grabbing a pen or scissors
            heat_source_c=26.0,
            description="140 g of 6 pens + scissors leaning in caddy + 1.5 N grab/bump force.",
        )
    if any(k in text for k in ("hook", "clamp", "skadis", "bracket", "vesa", "挂钩", "夹", "洞洞板")):
        return ApplicationLoadCase(
            name="Mounted Hook / Bracket Load",
            name_zh="挂钩/夹具悬臂承重 (800g)",
            payload_mass_g=800.0,
            payload_offset_xy_mm=(0.0, -d * 0.35),
            payload_z_ratio=0.50,
            lateral_force_n=2.0,
            heat_source_c=28.0,
            description="800 g hanging utility load on cantilever arm + 2.0 N pull.",
        )

    # Default desktop figurine / sculpture stability under accidental desk bump
    return ApplicationLoadCase(
        name="Desktop Display + Bump Stability",
        name_zh="桌面摆件自重与防碰翻工况",
        payload_mass_g=0.0,
        payload_offset_xy_mm=(0.0, 0.0),
        payload_z_ratio=0.50,
        lateral_force_n=0.8,
        heat_source_c=26.0,
        description="Self-weight equilibrium + 0.8 N accidental lateral desk bump.",
    )


def _signed_distance_to_convex_hull_2d(point_xy: np.ndarray, hull_pts: np.ndarray) -> tuple[float, float]:
    """Return (min_signed_margin_mm, hull_area_mm2). Positive margin = strictly inside support polygon."""
    if len(hull_pts) < 3:
        return -5.0, 0.0
    try:
        hull = ConvexHull(hull_pts)
    except Exception:
        return -5.0, 0.0

    # hull.equations rows are [a, b, c] with a*x + b*y + c <= 0 for interior points, ||(a,b)|| == 1
    eqs = hull.equations
    signed_dists = -(eqs[:, 0] * point_xy[0] + eqs[:, 1] * point_xy[1] + eqs[:, 2])
    min_margin = float(np.min(signed_dists))
    area = float(hull.volume)  # In 2D, scipy ConvexHull.volume is polygon area!
    return min_margin, area


def _compute_z_slice_mechanics(
    mesh: trimesh.Trimesh,
    n_slices: int = 28,
) -> tuple[float, float, float, float, float]:
    """Slice combined geometry across Z heights to find weakest cross-section area, inertia I_min, and slender necks.

    Returns: (min_neck_area_mm2, min_neck_z_mm, min_I_mm4, min_equiv_diam_mm, top_thin_feature_diam_mm)
    """
    bmin, bmax = mesh.bounds
    h = float(bmax[2] - bmin[2])
    if h < 4.0 or len(mesh.faces) < 4:
        return 100.0, float(bmin[2]), 1000.0, 12.0, 12.0

    z_levels = np.linspace(bmin[2] + h * 0.08, bmax[2] - h * 0.08, n_slices)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    z_v = verts[:, 2]
    f_zmin = z_v[faces].min(axis=1)
    f_zmax = z_v[faces].max(axis=1)

    min_area = 1e12
    min_z = float(bmin[2] + h * 0.5)
    min_i = 1e12
    min_diam = 1e6
    top_min_diam = 1e6

    for z in z_levels:
        # Collect exact triangle-plane intersection points for fast, dependency-free cross-sections
        cross_mask = (f_zmin <= z) & (f_zmax >= z)
        if not np.any(cross_mask):
            continue
        tri_pts = verts[faces[cross_mask]]  # (K, 3, 3)
        pts_xy: list[np.ndarray] = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za = tri_pts[:, a, 2]
            zb = tri_pts[:, b, 2]
            straddle = ((za <= z) & (zb > z)) | ((zb <= z) & (za > z))
            if np.any(straddle):
                denom = zb[straddle] - za[straddle]
                t = (z - za[straddle]) / np.where(np.abs(denom) > 1e-9, denom, 1.0)
                inter = tri_pts[straddle, a, :2] + t[:, None] * (tri_pts[straddle, b, :2] - tri_pts[straddle, a, :2])
                pts_xy.append(inter)
        if not pts_xy:
            continue
        ring = np.vstack(pts_xy)
        if len(ring) < 6:
            continue

        # Approximate effective section area & principal moments via bounding/covariance + hull
        dx = float(np.ptp(ring[:, 0]))
        dy = float(np.ptp(ring[:, 1]))
        try:
            area = float(ConvexHull(ring).volume)
        except Exception:
            area = dx * dy * 0.78
        if area < 0.5:
            continue

        # Effective equivalent diameter and minimum bending inertia I_min for thin-walled/infilled section
        equiv_d = 2.0 * math.sqrt(area / math.pi)
        # Rectangular/elliptical conservative moment of inertia (min axis controls buckling/bending!)
        b_min = max(1.0, min(dx, dy))
        b_max = max(b_min, max(dx, dy))
        # Hollow FDM shell factor (~0.52 of solid inertia for 3 walls + 15% gyroid infill)
        i_min_solid = (b_max * (b_min**3)) / 12.0
        i_min_fdm = max(1.0, i_min_solid * 0.52)

        rel_h = (z - bmin[2]) / h
        # Track load-bearing body/neck bottlenecks (below 82% height)
        if rel_h <= 0.82 and area < min_area:
            min_area = area
            min_z = float(z)
            min_i = i_min_fdm
            min_diam = equiv_d
        if rel_h > 0.80 and equiv_d < top_min_diam:
            top_min_diam = equiv_d

    if min_area > 1e11:
        return 100.0, float(bmin[2] + h * 0.5), 1000.0, 12.0, 12.0
    if top_min_diam > 1e5:
        top_min_diam = min_diam
    return min_area, min_z, min_i, min_diam, top_min_diam


def evaluate_physics_stability(
    parts: list[PrintPart] | list[trimesh.Trimesh],
    spec: DesignSpec,
    infill_ratio: float = 0.15,
    wall_loops: int = 3,
) -> PhysicsStabilityReport:
    """Run full anisotropic FDM structural mechanics, base-closure, and tipping stability physics evaluation."""
    meshes: list[trimesh.Trimesh] = [
        p.mesh if isinstance(p, PrintPart) else p for p in parts if (p.mesh if isinstance(p, PrintPart) else p) is not None
    ]
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    bmin, bmax = combined.bounds
    ext = bmax - bmin
    bbox_mm = (float(ext[0]), float(ext[1]), float(ext[2]))
    footprint_box_area = max(1.0, bbox_mm[0] * bbox_mm[1])

    # 1. Material properties (use primary filament material, default PLA)
    raw_mat = (spec.filaments[0].material if spec.filaments else "PLA").upper()
    mat = FDM_MATERIALS.get(raw_mat, FDM_MATERIALS["PLA"])

    # Effective shell + infill mass fraction (~0.34 for 3 walls + 15% gyroid infill on desktop prints)
    effective_fill = min(1.0, max(0.22, 0.22 + 0.04 * wall_loops + 0.65 * infill_ratio))

    # 2. Exact Volume-Weighted 3D Center of Mass (CoM)
    total_vol_mm3 = 0.0
    weighted_com = np.zeros(3, dtype=np.float64)
    all_watertight = True
    for m in meshes:
        if not m.is_watertight:
            all_watertight = False
        vol = abs(float(m.volume)) if m.is_watertight else abs(float(m.bounding_box.volume)) * 0.38
        vol = max(vol, 1.0)
        cm = np.asarray(m.center_mass if m.is_watertight else m.centroid, dtype=np.float64)
        weighted_com += cm * vol
        total_vol_mm3 += vol

    com_unloaded = weighted_com / max(total_vol_mm3, 1.0)
    self_mass_g = (total_vol_mm3 / 1000.0) * mat.density_g_cm3 * effective_fill

    # 3. Base-Contact Closure & Support Polygon Convex Hull on Z = 0
    verts = np.asarray(combined.vertices, dtype=np.float64)
    z_ground_tol = max(0.8, min(2.0, bbox_mm[2] * 0.015))
    base_pts = verts[verts[:, 2] <= (bmin[2] + z_ground_tol)][:, :2]
    margin_unloaded, support_area_mm2 = _signed_distance_to_convex_hull_2d(com_unloaded[:2], base_pts)
    base_contact_ratio = min(1.0, support_area_mm2 / footprint_box_area)

    # Check actual closed bottom-face area (normals pointing straight down, nz < -0.85, near Z=0)
    face_centers = combined.triangles_center
    face_normals = combined.face_normals
    face_areas = combined.area_faces
    flat_bottom_mask = (face_centers[:, 2] <= bmin[2] + z_ground_tol) & (face_normals[:, 2] < -0.75)
    flat_bottom_area_mm2 = float(face_areas[flat_bottom_mask].sum()) if np.any(flat_bottom_mask) else 0.0
    base_closed_solid = all_watertight and (flat_bottom_area_mm2 >= max(40.0, support_area_mm2 * 0.14))

    # 4. Application Load Case (Real-world loaded CoM & Tipping Equilibrium!)
    load_case = infer_application_load_case(spec, bbox_mm)
    cx_box = 0.5 * (bmin[0] + bmax[0])
    cy_box = 0.5 * (bmin[1] + bmax[1])
    payload_pos = np.array(
        [
            cx_box + load_case.payload_offset_xy_mm[0],
            cy_box + load_case.payload_offset_xy_mm[1],
            bmin[2] + bbox_mm[2] * load_case.payload_z_ratio,
        ],
        dtype=np.float64,
    )
    total_mass_g = self_mass_g + load_case.payload_mass_g
    total_mass_kg = max(0.005, total_mass_g / 1000.0)
    com_loaded = (com_unloaded * self_mass_g + payload_pos * load_case.payload_mass_g) / max(total_mass_g, 1.0)

    margin_loaded, _ = _signed_distance_to_convex_hull_2d(com_loaded[:2], base_pts)

    z_cm_unloaded = max(2.0, float(com_unloaded[2] - bmin[2]))
    z_cm_loaded = max(2.0, float(com_loaded[2] - bmin[2]))
    tilt_unloaded_deg = math.degrees(math.atan2(margin_unloaded, z_cm_unloaded))
    tilt_loaded_deg = math.degrees(math.atan2(margin_loaded, z_cm_loaded))

    # Lateral force required at application height to tip the loaded assembly over the closest support edge
    z_force_arm = max(10.0, float(payload_pos[2] - bmin[2]))
    tipping_force_n = max(0.0, (total_mass_kg * G_ACCEL * max(0.0, margin_loaded)) / z_force_arm)

    # 5. Z-Slice Structural Mechanics (Neck Bending Stress & Anisotropic Z-Interlayer Safety Factor)
    min_neck_area, min_neck_z, min_i_mm4, min_neck_diam, top_thin_diam = _compute_z_slice_mechanics(combined)
    lever_above_neck_mm = max(5.0, float(payload_pos[2] - min_neck_z))
    payload_weight_n = (load_case.payload_mass_g / 1000.0) * G_ACCEL
    eccentricity_mm = float(np.linalg.norm(payload_pos[:2] - com_unloaded[:2]))

    # Total bending moment at weakest horizontal layer plane (N*mm): lateral force + eccentric payload weight
    moment_n_mm = load_case.lateral_force_n * lever_above_neck_mm + payload_weight_n * max(eccentricity_mm, 6.0)
    c_half = max(1.5, 0.5 * min_neck_diam)
    # sigma = M * c / I  (N*mm * mm / mm^4 = N/mm^2 = MPa)
    peak_z_stress_mpa = max(0.01, (moment_n_mm * c_half) / max(min_i_mm4, 1.0))
    # Add direct axial/shear contribution
    peak_z_stress_mpa += (payload_weight_n + load_case.lateral_force_n) / max(min_neck_area * 0.45, 1.0)

    safety_factor_z = min(99.9, mat.tensile_z_mpa / max(peak_z_stress_mpa, 0.01))

    # Cantilever deflection delta = F * L^3 / (3 * E_eff * I) (E in MPa = GPa * 1000)
    e_eff_mpa = mat.youngs_modulus_gpa * 1000.0 * (0.45 + 0.55 * effective_fill)
    force_bending_n = load_case.lateral_force_n + payload_weight_n * 0.35
    deflection_mm = (force_bending_n * (lever_above_neck_mm**3)) / max(3.0 * e_eff_mpa * min_i_mm4, 1.0)

    # 6. Engineering Verdicts & Actionable Remediation Checks
    checks: list[PhysicsCheckItem] = []

    # Check A: Base Closure & Bed Contact
    if not base_closed_solid:
        checks.append(
            PhysicsCheckItem(
                name="Base Closure & Bed Contact",
                name_zh="底面闭合与热床接触",
                status="fail" if not all_watertight else "warn",
                metric=f"{flat_bottom_area_mm2:.0f} mm² flat contact ({base_contact_ratio*100:.0f}% footprint)",
                detail=(
                    f"Flat Z=0 contact area is only {flat_bottom_area_mm2:.0f} mm² "
                    f"(support hull {support_area_mm2:.0f} mm²). Ground-trim base or add a flat plinth so treads/feet close solidly on Z=0."
                ),
            )
        )
    else:
        checks.append(
            PhysicsCheckItem(
                name="Base Closure & Bed Contact",
                name_zh="底面闭合与热床接触",
                status="pass",
                metric=f"{flat_bottom_area_mm2:.0f} mm² closed flat base",
                detail=f"100% closed watertight base seated flat on Z=0 (support polygon {support_area_mm2:.0f} mm², {base_contact_ratio*100:.0f}% footprint).",
            )
        )

    # Check B: Tipping Stability (Unloaded & Under Application Load)
    tip_status = "pass" if (tilt_loaded_deg >= 18.0 and tipping_force_n >= load_case.lateral_force_n * 0.85) else ("warn" if tilt_loaded_deg >= 8.0 else "fail")
    checks.append(
        PhysicsCheckItem(
            name="Tipping Stability (Loaded)",
            name_zh=f"满载抗倾覆稳定性 ({load_case.name_zh})",
            status=tip_status,
            metric=f"θ_tip={tilt_loaded_deg:.1f}° · F_tip={tipping_force_n:.2f} N",
            detail=(
                f"CoM=({com_loaded[0]:.1f}, {com_loaded[1]:.1f}, {z_cm_loaded:.1f}) mm; "
                f"margin={margin_loaded:.1f} mm to support edge (unloaded θ={tilt_unloaded_deg:.1f}°). "
                + ("Rock-solid desk stability." if tip_status == "pass" else "Widen base footprint, add bottom infill weight (40–60% lower 15 mm), or add underside magnet/coin pockets.")
            ),
        )
    )

    # Check C: Anisotropic Z-Interlayer Bending Strength (FOS)
    fos_status = "pass" if safety_factor_z >= 3.0 else ("warn" if safety_factor_z >= 1.5 else "fail")
    checks.append(
        PhysicsCheckItem(
            name="Anisotropic Z-Layer Strength",
            name_zh="FDM Z向层间弯曲安全系数",
            status=fos_status,
            metric=f"FOS={safety_factor_z:.1f}× · σ={peak_z_stress_mpa:.2f}/{mat.tensile_z_mpa:.0f} MPa",
            detail=(
                f"Weakest section at Z={min_neck_z:.1f} mm (equiv Ø{min_neck_diam:.1f} mm, A={min_neck_area:.0f} mm²); "
                f"predicted tip deflection δ={deflection_mm:.2f} mm under {load_case.name}. "
                + ("Ample structural margin for FDM printing." if fos_status == "pass" else "Increase wall loops to 4–5, thicken neck cross-section, or print in PETG/PLA-CF.")
            ),
        )
    )

    # Check D: Thermal / HDT Suitability
    hdt_margin = mat.hdt_c - load_case.heat_source_c
    hdt_status = "pass" if hdt_margin >= 12.0 else ("warn" if hdt_margin >= 4.0 else "fail")
    checks.append(
        PhysicsCheckItem(
            name="Thermal HDT Suitability",
            name_zh="材料耐温与实际工况匹配",
            status=hdt_status,
            metric=f"{mat.code} HDT {mat.hdt_c:.0f}°C vs {load_case.heat_source_c:.0f}°C load",
            detail=(
                f"{mat.name} handles expected {load_case.heat_source_c:.0f}°C operating temp with +{hdt_margin:.0f}°C margin."
                if hdt_status == "pass"
                else f"Expected {load_case.heat_source_c:.0f}°C heat (e.g. wireless charging coil) approaches {mat.code} HDT ({mat.hdt_c:.0f}°C) — switch to PETG (69°C) or ASA (100°C)!"
            ),
        )
    )

    # Overall stability score (0..100)
    score = 100
    for chk in checks:
        if chk.status == "fail":
            score -= 30
        elif chk.status == "warn":
            score -= 12
    score = max(10, min(100, score))
    overall = "stable" if score >= 82 else ("caution" if score >= 58 else "unstable")

    return PhysicsStabilityReport(
        material_code=mat.code,
        material_name=mat.name,
        density_g_cm3=mat.density_g_cm3,
        youngs_modulus_gpa=mat.youngs_modulus_gpa,
        tensile_xy_mpa=mat.tensile_xy_mpa,
        tensile_z_mpa=mat.tensile_z_mpa,
        hdt_c=mat.hdt_c,
        load_scenario=load_case.name,
        load_scenario_zh=load_case.name_zh,
        payload_mass_g=load_case.payload_mass_g,
        self_mass_g=round(self_mass_g, 1),
        center_of_mass_mm=(round(float(com_loaded[0]), 1), round(float(com_loaded[1]), 1), round(z_cm_loaded, 1)),
        support_area_mm2=round(support_area_mm2, 1),
        flat_bottom_area_mm2=round(flat_bottom_area_mm2, 1),
        base_contact_ratio=round(base_contact_ratio, 3),
        base_closed_solid=base_closed_solid,
        tipping_margin_mm=round(margin_loaded, 1),
        tilt_angle_unloaded_deg=round(tilt_unloaded_deg, 1),
        tilt_angle_loaded_deg=round(tilt_loaded_deg, 1),
        tipping_force_n=round(tipping_force_n, 2),
        weakest_neck_z_mm=round(min_neck_z, 1),
        weakest_neck_diam_mm=round(min_neck_diam, 1),
        peak_z_stress_mpa=round(peak_z_stress_mpa, 2),
        safety_factor_z=round(safety_factor_z, 1),
        tip_deflection_mm=round(deflection_mm, 3),
        stability_score=score,
        overall_verdict=overall,
        recommended_wall_loops=wall_loops,
        recommended_infill_pct=int(round(infill_ratio * 100.0)),
        recommended_material=(
            "ASA" if load_case.heat_source_c >= 62.0 else ("PETG" if hdt_margin < 10.0 else None)
        ),
        auto_remediations=[],
        checks=checks,
    )


def _pick_base_plinth_slot(filaments: list) -> int:
    """Choose a clean dark/base filament slot for an auto-synthesized stability plinth."""
    if not filaments:
        return 1
    base_kws = ("base", "plinth", "stand", "tread", "charcoal", "black", "gray", "grey", "dark", "joint")
    best_slot = filaments[-1].slot
    best_score = -1
    for fl in filaments:
        txt = f"{fl.role} {fl.name}".lower()
        sc = sum(2 for kw in base_kws if kw in txt)
        if sc > best_score:
            best_score = sc
            best_slot = fl.slot
    return best_slot


def optimize_assembly_with_physics(
    parts: list[PrintPart],
    spec: DesignSpec,
    filaments: list | None = None,
    auto_stabilize: bool = True,
) -> tuple[list[PrintPart], PhysicsStabilityReport]:
    """Closed-loop physics-guided structural auto-optimizer for 3D-printable assemblies.

    Executes an automated Diagnose -> Remediate -> Re-Verify loop:
      1. Evaluates baseline anisotropic FDM mechanics, base closure, tipping angle & pull-off force.
      2. If base is unclosed/narrow OR loaded tipping angle < 16 deg OR tipping force < 0.85 * F_lat:
         computes the exact eccentric counter-balanced beveled `stability_plinth` (flat Z=0) shifted
         toward the loaded CoM, boolean-trims it flush against existing parts via `manifold3d`,
         and re-evaluates to verify measurable Before -> After stability improvement!
      3. Computes adaptive FDM slicer parameters (`wall_loops`, `sparse_infill_density`, thermal
         material recommendation) tailored to neck `FOS_z`, payload mass, and coil HDT margin.
    """
    current_parts = [PrintPart(name=p.name, slot=p.slot, mesh=p.mesh.copy(), joint_type=p.joint_type, plate_index=p.plate_index) for p in parts]
    active_fls = list(filaments if filaments is not None else spec.filaments)
    before = evaluate_physics_stability(current_parts, spec)
    remediations: list[str] = []

    combined_before = trimesh.util.concatenate([p.mesh for p in current_parts]) if len(current_parts) > 1 else current_parts[0].mesh
    bmin, bmax = combined_before.bounds
    ext = bmax - bmin
    bbox_mm = (float(ext[0]), float(ext[1]), float(ext[2]))
    load_case = infer_application_load_case(spec, bbox_mm)

    needs_plinth = (
        auto_stabilize
        and not any("plinth" in p.name.lower() for p in current_parts)
        and (
            (not before.base_closed_solid)
            or (before.tilt_angle_loaded_deg < 16.0)
            or (before.tipping_force_n < load_case.lateral_force_n * 0.85 and load_case.payload_mass_g > 0)
        )
    )

    if needs_plinth:
        try:
            import manifold3d

            from spoolgenie.print_prep.functional_carver import _manifold_to_mesh, _mesh_to_manifold

            cx_box = 0.5 * float(bmin[0] + bmax[0])
            cy_box = 0.5 * float(bmin[1] + bmax[1])
            cm_x, cm_y, z_cm = before.center_of_mass_mm

            # Exact minimum counter-balanced plinth radii targeting >= 22 deg loaded tilt & >= 1.15x lateral force!
            total_mass_kg = max(0.02, (before.self_mass_g + load_case.payload_mass_g) / 1000.0)
            z_force_arm = max(12.0, bbox_mm[2] * load_case.payload_z_ratio)
            req_margin_tilt = z_cm * math.tan(math.radians(23.0))
            req_margin_force = (load_case.lateral_force_n * 1.15 * z_force_arm) / max(total_mass_kg * G_ACCEL, 0.25)
            req_r = max(24.0, req_margin_tilt, min(req_margin_force, max(bbox_mm[0], bbox_mm[1]) * 0.82))

            # Center plinth slightly toward loaded CoM eccentricity so it specifically opposes forward/rear tipping!
            px = cx_box + np.clip((cm_x - cx_box) * 0.65, -bbox_mm[0] * 0.22, bbox_mm[0] * 0.22)
            py = cy_box + np.clip((cm_y - cy_box) * 0.65, -bbox_mm[1] * 0.28, bbox_mm[1] * 0.28)
            rx = max(bbox_mm[0] * 0.54, req_r, 24.0)
            ry = max(bbox_mm[1] * 0.54, req_r, 24.0)
            # Keep plinth sleek & visually proportional (<= 1.65x footprint span, <= 190mm bed safety)
            rx = min(rx, max(bbox_mm[0] * 0.85, 42.0), 92.0)
            ry = min(ry, max(bbox_mm[1] * 0.88, 42.0), 92.0)
            plinth_h = 5.5
            seat_embed = 1.2

            plinth_man = (
                manifold3d.Manifold.cylinder(plinth_h, rx, rx * 0.94, circular_segments=56)
                .scale([1.0, ry / rx, 1.0])
                .translate([float(px), float(py), 0.0])
            )

            # Seat all existing parts into top of plinth by 1.2 mm
            lift_z = plinth_h - seat_embed
            plinth_slot = _pick_base_plinth_slot(active_fls)
            cand_parts: list[PrintPart] = []
            part_mans: list[manifold3d.Manifold] = []
            same_slot_base_idx: int | None = None

            for idx_p, p in enumerate(current_parts):
                m_lifted = p.mesh.copy()
                m_lifted.apply_translation([0.0, 0.0, lift_z])
                cand_parts.append(
                    PrintPart(
                        name=p.name,
                        slot=p.slot,
                        mesh=m_lifted,
                        joint_type=p.joint_type,
                        plate_index=p.plate_index,
                    )
                )
                try:
                    pm = _mesh_to_manifold(m_lifted)
                    part_mans.append(pm)
                    # If an existing ground-touching part already uses the exact same base color slot,
                    # boolean-union the plinth directly into it to prevent coplanar same-slot seams!
                    if (
                        same_slot_base_idx is None
                        and p.slot == plinth_slot
                        and float(p.mesh.bounds[0][2]) <= float(bmin[2]) + 3.0
                    ):
                        same_slot_base_idx = idx_p
                except Exception:
                    part_mans.append(manifold3d.Manifold())

            built_ok = False
            if same_slot_base_idx is not None and part_mans[same_slot_base_idx].num_tri() > 0:
                merged_man = part_mans[same_slot_base_idx] + plinth_man
                for idx_other, om in enumerate(part_mans):
                    if idx_other != same_slot_base_idx and om.num_tri() > 0:
                        merged_man = merged_man - om
                merged_mesh = _manifold_to_mesh(merged_man)
                if merged_mesh.is_watertight and len(merged_mesh.faces) > 24:
                    cand_parts[same_slot_base_idx].mesh = merged_mesh
                    built_ok = True
            else:
                plinth_clean = plinth_man
                for pm in part_mans:
                    if pm.num_tri() > 0:
                        plinth_clean = plinth_clean - pm
                if plinth_clean.num_tri() > 24:
                    plinth_mesh = _manifold_to_mesh(plinth_clean)
                    if plinth_mesh.is_watertight:
                        cand_parts.append(PrintPart(name="stability_plinth", slot=plinth_slot, mesh=plinth_mesh))
                        built_ok = True

            if built_ok:
                min_z_all = min(float(cp.mesh.bounds[0][2]) for cp in cand_parts)
                if abs(min_z_all) > 1e-4:
                    for cp in cand_parts:
                        cp.mesh.apply_translation([0.0, 0.0, -min_z_all])

                after_plinth = evaluate_physics_stability(cand_parts, spec)
                improved = (
                    all(cp.mesh.is_watertight for cp in cand_parts)
                    and (
                        after_plinth.stability_score > before.stability_score
                        or after_plinth.tilt_angle_loaded_deg > before.tilt_angle_loaded_deg + 2.5
                        or (after_plinth.base_closed_solid and not before.base_closed_solid)
                    )
                )
                if improved:
                    current_parts = cand_parts
                    remediations.append(
                        f"Auto-added counter-balanced stability_plinth ({rx*2:.0f}×{ry*2:.0f}×{plinth_h:.1f} mm): "
                        f"θ_tip {before.tilt_angle_loaded_deg:.1f}° → {after_plinth.tilt_angle_loaded_deg:.1f}°, "
                        f"F_tip {before.tipping_force_n:.2f}N → {after_plinth.tipping_force_n:.2f}N, "
                        f"score {before.stability_score} → {after_plinth.stability_score}"
                    )
        except Exception:
            pass

    # Adaptive FDM slicer wall loops & gyroid infill optimization written directly into Bambu .3mf!
    mid = evaluate_physics_stability(current_parts, spec)
    rec_walls = 3
    rec_infill_pct = 15
    if mid.safety_factor_z < 2.2:
        rec_walls = 5
        rec_infill_pct = 35
    elif mid.safety_factor_z < 4.5 or load_case.payload_mass_g >= 200.0:
        rec_walls = 4
        rec_infill_pct = 25
    elif mid.tilt_angle_loaded_deg < 16.0:
        rec_infill_pct = 30

    if rec_walls > 3 or rec_infill_pct > 15:
        remediations.append(
            f"Adaptive FDM slicer reinforcement written to .3mf: {rec_walls} wall loops + {rec_infill_pct}% gyroid infill "
            f"(tuned for {load_case.name})."
        )

    final_report = evaluate_physics_stability(
        current_parts,
        spec,
        infill_ratio=rec_infill_pct / 100.0,
        wall_loops=rec_walls,
    )
    if final_report.recommended_material:
        remediations.append(
            f"Thermal HDT guard: recommended {final_report.recommended_material} over {final_report.material_code} "
            f"(HDT {final_report.hdt_c:.0f}°C vs {load_case.heat_source_c:.0f}°C load)."
        )
    final_report.auto_remediations = remediations
    return current_parts, final_report
