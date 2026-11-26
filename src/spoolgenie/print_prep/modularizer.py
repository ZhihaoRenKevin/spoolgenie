"""Modular 3D solid decomposition and 2D-guided whole-module canonical coloring.

Instead of spraying blotchy 2D pixels onto individual mesh triangles, this module:
  1. Uses exact `manifold3d` CSG planar booleans along real 3D structural transitions
     (lower base/treads, core torso, lateral arms/fins, upper head, and top crown/hat/fruit)
     to decompose any organic/character mesh into 100% watertight solid modules.
  2. Projects each WHOLE solid module onto the background-cleaned 2D concept image and computes
     its shading-tolerant chroma-weighted median CIELAB color, fused with canonical role priors,
     so every solid module gets ONE crisp, visually accurate filament color (zero speckles!).
  3. Re-unions adjacent body modules that vote for the exact same filament slot, eliminating
     artificial horizontal seams while keeping distinct-colored features (e.g., tangerine on a
     capybara's head, gold helmet/faceplate, dark base/treads, functional accessories) crisp!
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import manifold3d
import numpy as np
import trimesh
from PIL import Image

from spoolgenie.print_prep.functional_carver import _manifold_to_mesh, _mesh_to_manifold
from spoolgenie.print_prep.split_joiner import PrintPart
from spoolgenie.schemas import FilamentSlot


def _hex_to_rgb_array(hex_color: str) -> np.ndarray:
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return np.array([43.0, 45.0, 48.0], dtype=np.float64)
    return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float64)


def _rgb_to_lab_batch(rgb_u8: np.ndarray) -> np.ndarray:
    """Vectorized sRGB (0..255) -> CIE L*a*b* (D65) for shape (..., 3)."""
    arr = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    lin = np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    # sRGB -> XYZ (D65)
    x = lin[..., 0] * 0.4124564 + lin[..., 1] * 0.3575761 + lin[..., 2] * 0.1804375
    y = lin[..., 0] * 0.2126729 + lin[..., 1] * 0.7151522 + lin[..., 2] * 0.0721750
    z = lin[..., 0] * 0.0193339 + lin[..., 1] * 0.1191920 + lin[..., 2] * 0.9503041

    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    eps = 0.008856
    kappa = 903.3
    fx = np.where(xr > eps, np.cbrt(np.maximum(xr, 0.0)), (kappa * xr + 16.0) / 116.0)
    fy = np.where(yr > eps, np.cbrt(np.maximum(yr, 0.0)), (kappa * yr + 16.0) / 116.0)
    fz = np.where(zr > eps, np.cbrt(np.maximum(zr, 0.0)), (kappa * zr + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


@dataclass
class _Reference2DContext:
    rgb: np.ndarray  # (H, W, 3) uint8
    fg_mask: np.ndarray  # (H, W) bool
    u_min: int
    u_max: int
    v_min: int
    v_max: int
    flip_x: bool = False


def _prepare_2d_context(
    reference_image_path: str | Path | None,
    mesh: trimesh.Trimesh,
) -> _Reference2DContext | None:
    """Load the 2D concept image, isolate the clean foreground silhouette, and align X polarity to 3D mesh."""
    if not reference_image_path:
        return None
    path = Path(reference_image_path)
    if not path.exists():
        return None

    try:
        pil = Image.open(path).convert("RGBA")
        # Resize to <= 384px for instant millisecond projection voting
        pil.thumbnail((384, 384), Image.Resampling.LANCZOS)
        arr = np.asarray(pil)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        h_img, w_img = rgb.shape[:2]

        if alpha.min() < 245:
            fg_mask = alpha > 40
        else:
            # Inspect the 4 image corners: studio renders & clean concepts have uniform solid/light backdrop
            border = np.concatenate(
                [
                    rgb[0:6, :, :].reshape(-1, 3),
                    rgb[-6:, :, :].reshape(-1, 3),
                    rgb[:, 0:6, :].reshape(-1, 3),
                    rgb[:, -6:, :].reshape(-1, 3),
                ],
                axis=0,
            ).astype(np.float64)
            bg_med = np.median(border, axis=0)
            bg_std = float(np.mean(np.std(border, axis=0)))
            if bg_std < 18.0:
                dist_bg = np.linalg.norm(rgb.astype(np.float64) - bg_med[None, None, :], axis=-1)
                fg_mask = dist_bg > 22.0
            elif importlib.util.find_spec("rembg"):
                from rembg import remove

                rgba_clean = np.asarray(remove(pil))
                fg_mask = rgba_clean[:, :, 3] > 40
            else:
                dist_bg = np.linalg.norm(rgb.astype(np.float64) - bg_med[None, None, :], axis=-1)
                fg_mask = dist_bg > 28.0

        ys, xs = np.where(fg_mask)
        if len(xs) < 60 or len(ys) < 60:
            return None

        u_min, u_max = int(xs.min()), int(xs.max())
        v_min, v_max = int(ys.min()), int(ys.max())
        if (u_max - u_min) < 10 or (v_max - v_min) < 10:
            return None

        # Automatically verify whether mesh +X matches image +U or mirrored -U via silhouette overlap
        bmin, bmax = mesh.bounds
        ext = np.maximum(bmax - bmin, 1e-3)
        verts = np.asarray(mesh.vertices)
        step = max(1, len(verts) // 1500)
        pts = verts[::step]
        nx = np.clip((pts[:, 0] - bmin[0]) / ext[0], 0.0, 1.0)
        nz = np.clip((pts[:, 2] - bmin[2]) / ext[2], 0.0, 1.0)

        vi = np.clip((v_max - nz * (v_max - v_min)).astype(np.int32), 0, h_img - 1)
        ui_normal = np.clip((u_min + nx * (u_max - u_min)).astype(np.int32), 0, w_img - 1)
        ui_flip = np.clip((u_max - nx * (u_max - u_min)).astype(np.int32), 0, w_img - 1)
        hit_normal = float(fg_mask[vi, ui_normal].mean())
        hit_flip = float(fg_mask[vi, ui_flip].mean())

        return _Reference2DContext(
            rgb=rgb,
            fg_mask=fg_mask,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
            flip_x=(hit_flip > hit_normal + 0.03),
        )
    except Exception:
        return None


def _sample_module_lab_from_2d(
    mod_mesh: trimesh.Trimesh,
    full_bounds: np.ndarray,
    ctx: _Reference2DContext,
) -> np.ndarray | None:
    """Compute a shading-tolerant chroma-weighted median CIELAB color for ONE whole 3D solid module."""
    if len(mod_mesh.faces) < 4:
        return None

    bmin, bmax = full_bounds[0], full_bounds[1]
    ext = np.maximum(bmax - bmin, 1e-3)
    h_img, w_img = ctx.rgb.shape[:2]

    fc = np.asarray(mod_mesh.triangles_center)
    fn = np.asarray(mod_mesh.face_normals)
    # Prefer front-facing triangles (visible in the canonical front concept view)
    front_mask = fn[:, 1] < 0.25
    pts = fc[front_mask] if int(front_mask.sum()) >= 16 else fc
    if len(pts) > 2500:
        step = max(1, len(pts) // 2500)
        pts = pts[::step]

    nx = np.clip((pts[:, 0] - bmin[0]) / ext[0], 0.0, 1.0)
    nz = np.clip((pts[:, 2] - bmin[2]) / ext[2], 0.0, 1.0)
    if ctx.flip_x:
        ui = np.clip((ctx.u_max - nx * (ctx.u_max - ctx.u_min)).astype(np.int32), 0, w_img - 1)
    else:
        ui = np.clip((ctx.u_min + nx * (ctx.u_max - ctx.u_min)).astype(np.int32), 0, w_img - 1)
    vi = np.clip((ctx.v_max - nz * (ctx.v_max - ctx.v_min)).astype(np.int32), 0, h_img - 1)

    valid = ctx.fg_mask[vi, ui]
    if int(valid.sum()) < 8:
        return None

    pix_rgb = ctx.rgb[vi[valid], ui[valid]]
    pix_lab = _rgb_to_lab_batch(pix_rgb)

    # Filter out extreme specular highlights & deep ambient-occlusion crease shadows when chromatic pixels exist
    chroma = np.hypot(pix_lab[:, 1], pix_lab[:, 2])
    lum = pix_lab[:, 0]
    chromatic = (chroma > 16.0) & (lum > 18.0) & (lum < 94.0)
    if float(chromatic.mean()) >= 0.20 and int(chromatic.sum()) >= 8:
        # Use upper-chroma half of chromatic pixels so studio shadow falloff never turns brown/red/orange into black!
        sub_lab = pix_lab[chromatic]
        sub_c = chroma[chromatic]
        c_thresh = float(np.percentile(sub_c, 35.0))
        vivid = sub_lab[sub_c >= c_thresh]
        return np.median(vivid if len(vivid) >= 6 else sub_lab, axis=0)

    # Truly achromatic module (black, charcoal, gray, white): use clean mid-range median
    return np.median(pix_lab, axis=0)


def _shading_tolerant_lab_dist(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Weighted CIELAB distance emphasizing hue/chroma (a*, b*) over studio lighting brightness (L*)."""
    dL = (float(lab_a[0]) - float(lab_b[0])) * 0.55
    da = (float(lab_a[1]) - float(lab_b[1])) * 1.25
    db = (float(lab_a[2]) - float(lab_b[2])) * 1.25
    return float(np.sqrt(dL * dL + da * da + db * db))


def _make_slab_box(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> manifold3d.Manifold:
    sx = max(0.5, x1 - x0)
    sy = max(0.5, y1 - y0)
    sz = max(0.5, z1 - z0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    return manifold3d.Manifold.cube([sx, sy, sz], center=True).translate([cx, cy, cz])


def _has_top_crown_role(filaments: list[FilamentSlot]) -> bool:
    crown_kws = (
        "tangerine", "orange", "fruit", "leaf", "crown", "hat", "cap",
        "flower", "blossom", "sprout", "antenna", "horn", "stem", "ribbon", "bow",
    )
    for fl in filaments:
        txt = f"{fl.role} {fl.name}".lower()
        if any(kw in txt for kw in crown_kws):
            return True
    return False


def _detect_anatomical_planes(
    mesh: trimesh.Trimesh,
    filaments: list[FilamentSlot] | None = None,
) -> tuple[float, float | None, float | None, float | None]:
    """Detect natural 3D structural split planes: (z_base, z_neck, shoulder_half_w, z_crown).

    Works generically on characters, robots (WALL-E), animals (e.g. Capybara with tangerine on head),
    potted plants, mushrooms, and rockets by analyzing Z cross-sectional areas & lateral protrusions.
    """
    bmin, bmax = mesh.bounds
    ext = bmax - bmin
    cx = 0.5 * (bmin[0] + bmax[0])
    w, h = float(ext[0]), float(ext[2])
    # Sample uniform surface points so both low-poly CAD cylinders AND dense 384^3 AI meshes have full Z profiles!
    try:
        surf_pts, _ = trimesh.sample.sample_surface(mesh, 6000)
        verts = np.vstack([np.asarray(mesh.vertices), np.asarray(surf_pts)])
    except Exception:
        verts = np.asarray(mesh.vertices)

    # 1. Sample 48 vertical bands along Z to inspect width & cross-section profile
    n_bins = 48
    zs = np.linspace(bmin[2], bmax[2], n_bins + 1)
    band_centers = 0.5 * (zs[:-1] + zs[1:])
    band_area = np.zeros(n_bins, dtype=np.float64)
    band_xspan = np.zeros(n_bins, dtype=np.float64)
    band_gap_mid = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (verts[:, 2] >= zs[i]) & (verts[:, 2] <= zs[i + 1])
        pts = verts[mask]
        if len(pts) < 8:
            continue
        dx = float(np.ptp(pts[:, 0]))
        dy = float(np.ptp(pts[:, 1]))
        band_xspan[i] = dx
        band_area[i] = dx * dy
        mid_mask = np.abs(pts[:, 0] - cx) < (w * 0.14)
        band_gap_mid[i] = 1.0 - (float(mid_mask.sum()) / float(len(pts)))

    kernel = np.array([0.25, 0.5, 0.25])
    area_s = np.convolve(band_area, kernel, mode="same")

    # A. Lower Base / Legs / Treads / Pot cut plane (18%..38% of height)
    lo_idx = int(n_bins * 0.18)
    hi_idx = int(n_bins * 0.38)
    z_base = float(bmin[2] + h * 0.28)
    gap_slice = band_gap_mid[lo_idx:hi_idx]
    if len(gap_slice) > 3 and float(np.max(band_gap_mid[: lo_idx + 2])) > 0.78:
        for k in range(lo_idx, hi_idx):
            if band_gap_mid[k] < 0.68:
                z_base = float(band_centers[k])
                break
    else:
        diffs = np.abs(np.diff(area_s[lo_idx : hi_idx + 1]))
        if len(diffs) > 0 and float(diffs.max()) > 0:
            z_base = float(band_centers[lo_idx + int(np.argmax(diffs))])

    # B. Upper Head / Neck cut plane (54%..80% of height)
    z_neck: float | None = None
    n_lo = int(n_bins * 0.54)
    n_hi = int(n_bins * 0.80)
    if n_hi > n_lo + 2:
        sub_area = area_s[n_lo:n_hi]
        min_rel = int(np.argmin(sub_area))
        min_idx = n_lo + min_rel
        min_val = float(area_s[min_idx])
        torso_peak = float(np.max(area_s[hi_idx:min_idx])) if min_idx > hi_idx else float(np.max(area_s))
        head_peak = float(np.max(area_s[min_idx : n_bins - 1])) if min_idx < n_bins - 2 else 0.0
        if torso_peak > 1e-3 and head_peak > min_val * 1.12 and min_val < torso_peak * 0.78:
            z_neck = float(band_centers[min_idx])
        elif h > 45.0:
            z_neck = float(bmin[2] + h * 0.68)

    # C. Distinct Top Crown / Hat / Tangerine / Blossom perched on top of the head (72%..94% of height)
    z_crown: float | None = None
    want_crown = _has_top_crown_role(filaments or [])
    c_lo = int(n_bins * 0.72)
    c_hi = int(n_bins * 0.94)
    if c_hi > c_lo + 2:
        peak_below = float(np.max(area_s[int(n_bins * 0.25) : c_lo + 2]))
        if peak_below > 1e-3:
            best_pinch_idx: int | None = None
            best_pinch_ratio = 1.0
            for k in range(c_lo, c_hi):
                local_above = float(np.max(area_s[k + 1 : min(n_bins, k + 7)])) if k + 1 < n_bins else 0.0
                if (
                    area_s[k] < peak_below * (0.62 if want_crown else 0.45)
                    and local_above > area_s[k] * (1.03 if want_crown else 1.10)
                    and local_above > peak_below * 0.04
                ):
                    ratio = float(area_s[k]) / peak_below
                    if ratio < best_pinch_ratio:
                        best_pinch_ratio = ratio
                        best_pinch_idx = k

            if best_pinch_idx is not None:
                z_crown = float(band_centers[best_pinch_idx])
            elif want_crown:
                top_drop = -np.diff(area_s[c_lo : c_hi + 1])
                if len(top_drop) > 0 and float(np.max(top_drop)) > peak_below * 0.04:
                    step_idx = c_lo + int(np.argmax(top_drop)) + 1
                    step_idx = min(step_idx, c_hi - 1)
                    z_crown = float(band_centers[step_idx])
                else:
                    z_crown = float(bmin[2] + h * 0.82)

    # Resolve overlap between z_neck and z_crown cleanly:
    # - If canonical palette explicitly features a top crown/fruit/hat (e.g. tangerine on capybara), prefer z_crown.
    # - Otherwise keep z_neck (upper_head, e.g. robot eyes/helmet) and drop colliding z_crown.
    if z_crown is not None and z_neck is not None and abs(z_crown - z_neck) < h * 0.14:
        if want_crown:
            z_neck = None
        else:
            z_crown = None

    # D. Lateral Arms / Wings / Fins cut distance from cx inside torso band (z_base .. z_neck)
    shoulder_half_w: float | None = None
    z_top_torso = z_neck if z_neck is not None else float(bmin[2] + h * 0.74)
    torso_pts = verts[(verts[:, 2] >= z_base + 2.0) & (verts[:, 2] <= z_top_torso - 2.0)]
    if len(torso_pts) > 40:
        abs_x = np.abs(torso_pts[:, 0] - cx)
        max_rx = float(np.percentile(abs_x, 99.0))
        r_bins = 24
        rs = np.linspace(0.0, max_rx, r_bins + 1)
        r_zspan = np.zeros(r_bins, dtype=np.float64)
        for i in range(r_bins):
            rmask = (abs_x >= rs[i]) & (abs_x <= rs[i + 1])
            rp = torso_pts[rmask]
            if len(rp) >= 6:
                r_zspan[i] = float(np.ptp(rp[:, 2]) * np.ptp(rp[:, 1]))
        core_ref = float(np.median(r_zspan[2:8])) if r_bins >= 8 else float(np.max(r_zspan))
        if core_ref > 1e-3:
            for i in range(int(r_bins * 0.38), int(r_bins * 0.80)):
                if r_zspan[i] < core_ref * 0.52 and float(np.max(r_zspan[i + 1 :])) > core_ref * 0.06:
                    shoulder_half_w = float(0.5 * (rs[i] + rs[i + 1]))
                    break

    return z_base, z_neck, shoulder_half_w, z_crown


def _pick_slot_for_module_role(
    module_kind: str,
    filaments: list[FilamentSlot],
    used_slots: set[int],
    observed_lab: np.ndarray | None = None,
) -> int:
    """Assign ONE clean canonical filament slot fusing 2D visual CIELAB voting + semantic role priors."""
    if not filaments:
        return 1
    if len(filaments) == 1:
        return filaments[0].slot

    role_keywords: dict[str, tuple[str, ...]] = {
        "core_body": (
            "body", "torso", "chest", "suit", "armor", "shell", "coat", "fur", "skin",
            "boxy", "cube", "hull", "fuselage", "cactus", "rosette", "main", "primary",
        ),
        "lower_base": (
            "tread", "track", "boot", "feet", "foot", "shoe", "leg", "wheel", "tire",
            "base", "pot", "plinth", "stand", "soil", "moss", "tray", "bottom", "pants",
            "shorts", "overalls", "cushion",
        ),
        "upper_head": (
            "head", "faceplate", "helmet", "binocular", "visor", "face", "mane", "hood", "mask",
        ),
        "top_crown": (
            "tangerine", "orange", "fruit", "leaf", "crown", "hat", "cap", "flower",
            "blossom", "sprout", "antenna", "horn", "stem", "top", "roof", "lid", "bow",
        ),
        "side_arms": (
            "arm", "hand", "wing", "fin", "joint", "trim", "accent", "stripe", "rim",
            "sleeve", "cape", "cloak", "collar", "shield",
        ),
        "functional_accessory": (
            "caddy", "dock", "base", "stand", "plinth", "accent", "trim", "detail",
            "backpack", "charcoal", "gray", "grey", "black", "pot",
        ),
    }

    kws = role_keywords.get(module_kind, ())
    fl_labs = [_rgb_to_lab_batch(_hex_to_rgb_array(fl.hex_color)) for fl in filaments]
    has_visual = observed_lab is not None

    best_slot: int | None = None
    best_score = -1e9

    for idx, fl in enumerate(filaments):
        text = f"{fl.role} {fl.name}".lower()
        # Use MAX keyword score (not sum!) so duplicate synonyms like "tangerine" + "orange" never drown out 2D visual evidence!
        role_score = 0.0
        for rank, kw in enumerate(kws):
            if kw in text:
                role_score = max(role_score, 2.6 - min(rank * 0.06, 0.8))
        if module_kind == "core_body" and idx == 0:
            role_score += 1.8

        # When 2D visual evidence exists, let shading-tolerant CIELAB voting lead cleanly!
        if has_visual and observed_lab is not None:
            d_vis = _shading_tolerant_lab_dist(observed_lab, fl_labs[idx])
            vis_score = 10.0 - (d_vis * 0.16)
            contrast_bonus = 0.30 if (module_kind != "core_body" and fl.slot not in used_slots) else 0.0
            role_weight = 0.55 if module_kind in ("top_crown", "functional_accessory") else 0.40
            score = vis_score + role_score * role_weight + contrast_bonus
        else:
            contrast_bonus = 0.85 if (module_kind != "core_body" and fl.slot not in used_slots) else 0.0
            score = role_score + contrast_bonus

        if score > best_score:
            best_score = score
            best_slot = fl.slot

    if best_slot is None:
        fallback_order = {
            "core_body": 0,
            "lower_base": 1,
            "side_arms": 2,
            "upper_head": 3 if len(filaments) >= 4 else 0,
            "top_crown": 1,
            "functional_accessory": 1,
        }
        best_slot = filaments[fallback_order.get(module_kind, 0) % len(filaments)].slot

    return best_slot


def _clean_components(man: manifold3d.Manifold, min_vol_ratio: float, total_vol: float) -> list[manifold3d.Manifold]:
    """Decompose a manifold into solid islands, dropping microscopic boolean dust."""
    if man.num_tri() == 0:
        return []
    parts = man.decompose()
    keep = [p for p in parts if p.num_tri() >= 12 and float(p.volume()) >= total_vol * min_vol_ratio]
    if not keep and float(man.volume()) > 1e-3:
        return [man]
    return keep


def _merge_manifolds(mans: list[manifold3d.Manifold]) -> manifold3d.Manifold:
    acc = mans[0]
    for m in mans[1:]:
        acc = acc + m
    return acc


def decompose_mesh_into_solid_modules(
    mesh: trimesh.Trimesh,
    filaments: list[FilamentSlot],
    base_name: str = "model",
    extra_functional_parts: list[PrintPart] | None = None,
    reference_image_path: str | Path | None = None,
) -> list[PrintPart]:
    """Decompose an organic/character mesh into clean watertight solid modules, each in 1 pure color.

    Guarantees:
      - Every returned PrintPart is a closed watertight solid (manifold3d exact planar CSG).
      - ZERO jagged triangle-level surface painting or shadow blotches.
      - When `reference_image_path` is provided, each whole solid module votes its true color from the
        2D concept image via shading-tolerant CIELAB matching fused with canonical semantic roles.
      - Adjacent character zones that vote for the SAME filament slot are seamlessly re-merged into
        one continuous watertight solid (eliminating fake horizontal color seams!).
    """
    extra_parts = list(extra_functional_parts or [])

    if len(filaments) <= 1 and not extra_parts:
        return [PrintPart(name=base_name, slot=filaments[0].slot if filaments else 1, mesh=mesh)]

    try:
        man = _mesh_to_manifold(mesh)
    except Exception:
        slot_main = _pick_slot_for_module_role("core_body", filaments, set())
        out = [PrintPart(name=f"{base_name}_body", slot=slot_main, mesh=mesh)]
        used = {slot_main}
        for ep in extra_parts:
            ep.slot = _pick_slot_for_module_role("functional_accessory", filaments, used)
            used.add(ep.slot)
            out.append(ep)
        return out

    total_vol = max(float(man.volume()), 1.0)
    bmin, bmax = mesh.bounds
    full_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    pad = 15.0
    x0, x1 = float(bmin[0] - pad), float(bmax[0] + pad)
    y0, y1 = float(bmin[1] - pad), float(bmax[1] + pad)
    z0, z1 = float(bmin[2] - pad), float(bmax[2] + pad)
    cx = 0.5 * (float(bmin[0]) + float(bmax[0]))

    ctx_2d = _prepare_2d_context(reference_image_path, mesh)
    z_base, z_neck, shoulder_half_w, z_crown = _detect_anatomical_planes(mesh, filaments=filaments)

    # 1. Exact CSG planar cuts into anatomical zones (zero overlap, zero gap!)
    lower_box = _make_slab_box(x0, x1, y0, y1, z0, z_base)
    lower_man = man ^ lower_box
    above_base = man - lower_box

    crown_man = manifold3d.Manifold()
    if z_crown is not None and z_crown > z_base + 10.0 and z_crown < float(bmax[2]) - 4.0:
        crown_box = _make_slab_box(x0, x1, y0, y1, z_crown, z1)
        cand_crown = above_base ^ crown_box
        c_vol = float(cand_crown.volume())
        if total_vol * 0.008 <= c_vol <= total_vol * 0.24:
            crown_man = cand_crown
            above_base = above_base - crown_box

    if z_neck is not None and z_neck > z_base + 8.0 and (z_crown is None or z_neck < z_crown - 6.0):
        head_box = _make_slab_box(x0, x1, y0, y1, z_neck, z1)
        head_man = above_base ^ head_box
        torso_region = above_base - head_box
    else:
        head_man = manifold3d.Manifold()
        torso_region = above_base

    arms_man = manifold3d.Manifold()
    if shoulder_half_w is not None and shoulder_half_w > 10.0:
        core_box = _make_slab_box(cx - shoulder_half_w, cx + shoulder_half_w, y0, y1, z0, z1)
        core_man = torso_region ^ core_box
        arms_man = torso_region - core_box
        if float(arms_man.volume()) < total_vol * 0.025 or float(core_man.volume()) < total_vol * 0.15:
            core_man = torso_region
            arms_man = manifold3d.Manifold()
    else:
        core_man = torso_region

    # Re-absorb microscopic slivers (< 2.5% total volume, except top_crown which can be 0.8%+)
    if float(lower_man.volume()) < total_vol * 0.025:
        core_man = core_man + lower_man
        lower_man = manifold3d.Manifold()
    if float(head_man.volume()) < total_vol * 0.025:
        core_man = core_man + head_man
        head_man = manifold3d.Manifold()

    # 2. Sample 2D shading-tolerant median CIELAB color per WHOLE solid module & pick optimal canonical slot!
    def _obs_lab(m_obj: manifold3d.Manifold) -> np.ndarray | None:
        if ctx_2d is None or m_obj.num_tri() == 0:
            return None
        try:
            return _sample_module_lab_from_2d(_manifold_to_mesh(m_obj), full_bounds, ctx_2d)
        except Exception:
            return None

    used_slots: set[int] = set()
    slot_core = _pick_slot_for_module_role("core_body", filaments, used_slots, observed_lab=_obs_lab(core_man))
    used_slots.add(slot_core)

    slot_crown = (
        _pick_slot_for_module_role("top_crown", filaments, used_slots, observed_lab=_obs_lab(crown_man))
        if crown_man.num_tri() > 0
        else slot_core
    )
    if crown_man.num_tri() > 0:
        used_slots.add(slot_crown)

    slot_base = (
        _pick_slot_for_module_role("lower_base", filaments, used_slots, observed_lab=_obs_lab(lower_man))
        if lower_man.num_tri() > 0
        else slot_core
    )
    if lower_man.num_tri() > 0:
        used_slots.add(slot_base)

    slot_arms = (
        _pick_slot_for_module_role("side_arms", filaments, used_slots, observed_lab=_obs_lab(arms_man))
        if arms_man.num_tri() > 0
        else slot_core
    )
    if arms_man.num_tri() > 0:
        used_slots.add(slot_arms)

    slot_head = (
        _pick_slot_for_module_role("upper_head", filaments, used_slots, observed_lab=_obs_lab(head_man))
        if head_man.num_tri() > 0
        else slot_core
    )
    if head_man.num_tri() > 0:
        used_slots.add(slot_head)

    # 3. Cohesive Same-Color Merger: heal unnecessary seam planes between adjacent zones with identical color!
    if ctx_2d is not None:
        if lower_man.num_tri() > 0 and slot_base == slot_core:
            core_man = core_man + lower_man
            lower_man = manifold3d.Manifold()
        if arms_man.num_tri() > 0 and slot_arms == slot_core:
            core_man = core_man + arms_man
            arms_man = manifold3d.Manifold()
        if head_man.num_tri() > 0 and slot_head == slot_core:
            core_man = core_man + head_man
            head_man = manifold3d.Manifold()
        if crown_man.num_tri() > 0 and slot_crown == (slot_head if head_man.num_tri() > 0 else slot_core):
            if head_man.num_tri() > 0:
                head_man = head_man + crown_man
            else:
                core_man = core_man + crown_man
            crown_man = manifold3d.Manifold()

    modules: list[PrintPart] = []

    def _emit_group(
        m_obj: manifold3d.Manifold,
        name_prefix: str,
        slot: int,
        split_islands: bool = False,
        min_vol_ratio: float = 0.008,
    ) -> None:
        if m_obj.num_tri() == 0 or float(m_obj.volume()) < total_vol * min_vol_ratio:
            return
        if split_islands:
            islands = _clean_components(m_obj, min_vol_ratio=min_vol_ratio, total_vol=total_vol)
            for idx, isl in enumerate(islands, start=1):
                mesh_i = _manifold_to_mesh(isl)
                if len(mesh_i.faces) > 0:
                    suffix = f"_{idx}" if len(islands) > 1 else ""
                    modules.append(PrintPart(name=f"{name_prefix}{suffix}", slot=slot, mesh=mesh_i))
        else:
            islands = _clean_components(m_obj, min_vol_ratio=min_vol_ratio * 0.5, total_vol=total_vol)
            if not islands:
                return
            merged = _merge_manifolds(islands)
            mesh_m = _manifold_to_mesh(merged)
            if len(mesh_m.faces) > 0:
                modules.append(PrintPart(name=name_prefix, slot=slot, mesh=mesh_m))

    _emit_group(core_man, "core_body", slot_core, split_islands=False)
    _emit_group(lower_man, "lower_base", slot_base, split_islands=True)
    _emit_group(arms_man, "side_arms", slot_arms, split_islands=True)
    _emit_group(head_man, "upper_head", slot_head, split_islands=False)
    _emit_group(crown_man, "top_crown", slot_crown, split_islands=False, min_vol_ratio=0.005)

    # Append clean separate functional modules (e.g. rear_pen_caddy, plinth_base, phone_cradle)
    for ep in extra_parts:
        ep.slot = _pick_slot_for_module_role("functional_accessory", filaments, used_slots)
        used_slots.add(ep.slot)
        modules.append(ep)

    if not modules:
        return [PrintPart(name=base_name, slot=slot_core, mesh=mesh)]

    return modules
